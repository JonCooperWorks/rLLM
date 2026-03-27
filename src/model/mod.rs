// ===========================================================================
// Transformer model: weights, KV cache, scratch buffers, and forward pass
// dispatch.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Defines the `Model` struct that holds all inference state (weights,
//   scratch buffers, KV cache) and dispatches the forward pass to the
//   appropriate architecture-specific module.
//
// Architecture dispatch (registry pattern, similar to vLLM):
//   Each model family lives in its own file under model/registry/.
//   Standard dense transformers (Llama, Phi, Qwen) share a common forward
//   pass in registry/llama.rs, parameterised by ArchFeatures.  Complex
//   architectures (Gemma, Qwen3 MoE, Qwen3.5 hybrid) have custom files.
//
//   The dispatch chain:
//     Model::forward_*() → registry::llama/qwen/... → primitives → GpuBackend
//
//   Adding a new architecture means:
//     1. Create a new file in model/registry/ (e.g. deepseek.rs)
//     2. Add a variant to ModelArch in config.rs
//     3. Add dispatch arms in this file
//
// Key design: backend abstraction.
//   This file NEVER imports Metal (or CUDA) types.  Every GPU operation goes
//   through the `GpuBackend` trait: `self.backend.rms_norm(...)`, etc.
//   This means the same forward-pass code works on any backend — Metal today,
//   CUDA tomorrow — without changing a single line here.
//
// KV cache:
//   Autoregressive generation requires attending to ALL previous tokens, not
//   just the current one.  The KV cache stores the K and V projections for
//   every token processed so far, avoiding redundant recomputation.
//
//   Layout: paged allocation from a shared pool (see kv_cache.rs).  Memory
//   is allocated in fixed-size blocks (16 tokens each) on demand.  Each
//   sequence gets a block table mapping logical blocks to physical ones.
//   This enables continuous batching — sequences can be added/removed
//   without fragmenting GPU memory.
//
// Buffer reuse:
//   Scratch buffers are allocated once at model creation and reused every
//   forward pass.  This avoids per-token GPU memory allocation, which would
//   be expensive.  The same `norm_buf` tensor holds the RMSNorm output for
//   both the attention sub-layer and the FFN sub-layer (sequentially, never
//   simultaneously).
//
// Residual connections:
//   The `hidden` buffer serves as the residual stream.  After each sub-layer
//   (attention and FFN), the sub-layer output is ADDED to `hidden`:
//     hidden = hidden + sublayer_output
//   This is critical — residual connections allow gradients to flow through
//   deep networks and prevent the vanishing gradient problem.
// ===========================================================================

pub(crate) mod chat;
pub(crate) mod config;
pub(crate) mod expert_stream;
pub(crate) mod kv_cache;
pub(crate) mod loader;
pub(crate) mod primitives;
pub(crate) mod profile;
pub(crate) mod registry;
pub(crate) mod sampler;
pub(crate) mod thinking;
pub(crate) mod tokenizer;
pub(crate) mod tools;
pub(crate) mod turboquant;
pub(crate) mod vision;

use self::config::{ModelArch, ModelConfig};
use self::kv_cache::{KvPool, SeqKvState};
use self::loader::ModelWeights;
use crate::gpu::{GpuBackend, GpuCore, GpuElementwise, TensorDtype};

/// Maximum sequence length supported by the flat KV cache (legacy mode).
/// The paged cache mode supports up to 131072 tokens (8192 blocks x 16
/// positions) and is the primary code path.  This constant is only used
/// by the flat cache allocation path.
const MAX_SEQ_LEN: usize = 4096;

/// The transformer model: weights, KV cache, scratch buffers, and a backend reference.
///
/// Generic over `B: GpuBackend` — the model doesn't know (or care) whether
/// it's running on Metal or CUDA.  The lifetime `'a` ties the model to
/// the backend that owns the GPU device.
///
/// KV cache mode: either flat (original) or paged (new).
///
/// Flat mode pre-allocates [MAX_SEQ_LEN, kv_dim] per layer — simple but
/// wastes memory for short sequences and can't share across concurrent
/// requests.  Paged mode allocates blocks on demand from a shared pool.
#[allow(dead_code)]
pub(crate) enum KvMode<B: GpuCore> {
    /// Original flat cache (backward compatible).
    Flat {
        k_cache: Vec<B::Tensor>,
        v_cache: Vec<B::Tensor>,
        pos: usize,
    },
    /// Paged cache with block table.
    Paged {
        pool: KvPool<B>,
        seq_state: SeqKvState<B>,
    },
}

pub(crate) struct Model<'a, B: GpuCore> {
    pub(crate) config: ModelConfig,
    pub(crate) weights: ModelWeights<B>,
    pub(crate) backend: &'a B,

    /// Which architecture's forward pass to use.
    arch: ModelArch,

    /// Tensor parallelism world size (1 = single GPU).
    pub(crate) world_size: usize,

    // -----------------------------------------------------------------------
    // KV cache: either flat or paged.
    // -----------------------------------------------------------------------
    #[allow(dead_code)]
    pub(crate) kv_mode: KvMode<B>,

    // -----------------------------------------------------------------------
    // Pre-allocated scratch buffers (reused every forward pass).
    //
    // These avoid per-token GPU allocation.  The naming convention:
    //   hidden     — the residual stream [hidden_size=2048]
    //   norm_buf   — output of RMSNorm [hidden_size=2048]
    //   q_buf      — query projection output [num_heads * head_dim]
    //   k_buf      — key projection output [num_kv_heads * head_dim]
    //   v_buf      — value projection output [num_kv_heads * head_dim]
    //   attn_out   — attention output [num_heads * head_dim]
    //   gate_buf   — gate projection for FFN [intermediate_size = 8192]
    //   up_buf     — up projection for FFN [intermediate_size = 8192]
    //   logits_buf — final vocabulary logits [vocab_size = 128256]
    // -----------------------------------------------------------------------
    pub(crate) hidden: B::Tensor,
    pub(crate) norm_buf: B::Tensor,
    pub(crate) q_buf: B::Tensor,
    pub(crate) k_buf: B::Tensor,
    pub(crate) v_buf: B::Tensor,
    pub(crate) attn_out: B::Tensor,
    pub(crate) gate_buf: B::Tensor,
    pub(crate) up_buf: B::Tensor,
    pub(crate) logits_buf: B::Tensor,

    // -----------------------------------------------------------------------
    // MoE-specific scratch buffers (None for dense models).
    //
    // Learning note: MoE layers need separate buffers because expert FFNs
    // have a different intermediate size (e.g. 768) than the dense FFN
    // (e.g. 8192).  The router logits buffer holds per-expert scores for
    // top-k selection.  The moe_output buffer accumulates the weighted
    // sum of expert outputs before adding to the residual stream.
    // -----------------------------------------------------------------------
    #[allow(dead_code)]
    pub(crate) router_logits: Option<B::Tensor>, // [num_experts] f32
    pub(crate) moe_gate_buf: Option<B::Tensor>, // [moe_inter] bf16
    pub(crate) moe_up_buf: Option<B::Tensor>,   // [moe_inter] bf16
    pub(crate) moe_output: Option<B::Tensor>,   // [hidden_size] bf16
    /// GPU-side routing output: [2 * num_experts_per_tok] f32 pairs of
    /// (expert_index, routing_weight).  Eliminates per-layer GPU→CPU sync
    /// for top-k selection.
    pub(crate) routing_output: Option<B::Tensor>, // [2*k] f32

    // -----------------------------------------------------------------------
    // DeltaNet state (Qwen 3.5 hybrid models only).
    //
    // DeltaNet layers maintain a fixed-size [head_dim, head_dim] recurrent
    // state matrix per QK-head (in f32 for precision), plus a Conv1D history
    // buffer for causal convolution.  These persist across tokens.
    // -----------------------------------------------------------------------
    pub(crate) deltanet_states: Option<Vec<B::Tensor>>, // per-DeltaNet-layer f32 state
    pub(crate) deltanet_conv_history: Option<Vec<B::Tensor>>, // per-DeltaNet-layer bf16 history

    // DeltaNet scratch buffers (reused every forward pass).
    pub(crate) dn_qkv_buf: Option<B::Tensor>, // [qk_dim*2 + v_dim] bf16 — fused QKV output
    pub(crate) dn_alpha_buf: Option<B::Tensor>, // [num_v_heads] f32 — decay gates
    pub(crate) dn_beta_buf: Option<B::Tensor>, // [num_v_heads] f32 — update gates
    pub(crate) dn_z_buf: Option<B::Tensor>,   // [v_dim] bf16 — output gate
    pub(crate) dn_conv_out: Option<B::Tensor>, // [conv_dim] bf16 — conv1d output
    pub(crate) dn_attn_out: Option<B::Tensor>, // [v_dim] bf16 — DeltaNet attention output
    pub(crate) dn_norm_out: Option<B::Tensor>, // [v_dim] bf16 — RMSNorm-no-weight output

    // -----------------------------------------------------------------------
    // Mamba-2 state (Nemotron-H Mamba layers only).
    //
    // Learning note: Mamba-2 layers maintain a [num_heads, head_dim, state_size]
    // recurrent state matrix per layer in f32.  This is the "memory" of the SSM —
    // it grows and decays based on the input at each step, via the B/C/dt/A_log
    // parameters.  Unlike KV cache which grows with sequence length, the Mamba
    // state is fixed-size.  The trade-off: fixed-size state can't represent
    // every detail of long contexts, but it's O(1) in memory and compute.
    //
    // Total per sequence: 23 layers × 64 heads × 64 head_dim × 128 state_size
    // × 4 bytes = ~46 MB.  Conv history adds ~550 KB.
    // -----------------------------------------------------------------------
    pub(crate) mamba2_states: Option<Vec<B::Tensor>>,       // per-Mamba-layer f32 state
    pub(crate) mamba2_conv_history: Option<Vec<B::Tensor>>,  // per-Mamba-layer bf16 history

    // Mamba-2 scratch buffers (reused every forward pass).
    pub(crate) m2_in_proj_buf: Option<B::Tensor>,  // [in_proj_dim] bf16
    pub(crate) m2_conv_out: Option<B::Tensor>,      // [conv_dim] bf16 (holds [x, B, C] after conv1d)
    pub(crate) m2_ssm_out: Option<B::Tensor>,       // [d_inner] bf16

    // KV layer mapping: layer_idx → kv_pool_idx (None for DeltaNet/Mamba/MoE layers).
    pub(crate) kv_layer_map: Vec<Option<usize>>,

    // SSD expert streaming (None when all experts are GPU-resident).
    //
    // When set, expert weights are NOT in LayerWeights::experts.  Instead,
    // the streamer loads selected experts from disk on-demand during the
    // MoE dispatch loop.  See expert_stream.rs.
    pub(crate) expert_streamer: Option<expert_stream::ExpertStreamer<B>>,

    // -----------------------------------------------------------------------
    // Vision encoder (VLM models only).
    //
    // When present, the model can process images.  Vision weights are loaded
    // from the safetensors file's `model.visual.*` (Qwen) or
    // `vision_tower.*` (Gemma) tensors.  Vision buffers are scratch memory
    // reused for each image encoding pass.
    // -----------------------------------------------------------------------
    pub(crate) vision_weights: Option<vision::VisionWeights<B>>,
    pub(crate) vision_bufs: Option<vision::VisionBuffers<B>>,

    // -----------------------------------------------------------------------
    // TurboQuant KV cache quantization context.
    //
    // When present, K/V vectors are quantized before writing to the paged
    // cache and dequantized inline during attention.  Contains the rotation
    // matrix Pi, its transpose Pi^T, codebook centroids, and a scratch
    // buffer for the rotated query vector.
    //
    // See model/turboquant.rs and docs/turboquant.md.
    // -----------------------------------------------------------------------
    pub(crate) turbo_ctx: Option<turboquant::TurboContext<B>>,
}

// ---------------------------------------------------------------------------
// Construction: needs GpuCore (tensor allocation) + GpuElementwise (fill_zero
// for DeltaNet state init).
// ---------------------------------------------------------------------------

impl<'a, B: GpuCore + GpuElementwise> Model<'a, B> {
    /// Create a new model with flat KV cache (original mode).
    pub fn new(
        config: ModelConfig,
        weights: ModelWeights<B>,
        backend: &'a B,
    ) -> anyhow::Result<Self> {
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let num_kv_layers = config.num_kv_layers();

        // Allocate per-layer KV caches (only for GQA layers in hybrid models).
        let mut k_cache = Vec::with_capacity(num_kv_layers);
        let mut v_cache = Vec::with_capacity(num_kv_layers);
        for _ in 0..num_kv_layers {
            k_cache.push(backend.alloc_tensor(&[MAX_SEQ_LEN, kv_dim], TensorDtype::BF16));
            v_cache.push(backend.alloc_tensor(&[MAX_SEQ_LEN, kv_dim], TensorDtype::BF16));
        }

        let kv_mode = KvMode::Flat {
            k_cache,
            v_cache,
            pos: 0,
        };
        Self::new_with_kv_mode(config, weights, backend, kv_mode)
    }

    /// Create a new model with paged KV cache.
    ///
    /// `num_blocks` controls the total KV cache capacity across all sequences.
    /// Each block holds BLOCK_SIZE (16) token positions.
    #[allow(dead_code)]
    pub fn new_paged(
        config: ModelConfig,
        weights: ModelWeights<B>,
        backend: &'a B,
        num_blocks: usize,
    ) -> anyhow::Result<Self> {
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let num_kv_layers = config.num_kv_layers();
        let pool = KvPool::new(backend, num_blocks, kv_dim, num_kv_layers, turboquant::KvQuantMode::None, config.head_dim);
        let seq_state = pool.new_sequence(backend);

        let kv_mode = KvMode::Paged { pool, seq_state };
        Self::new_with_kv_mode(config, weights, backend, kv_mode)
    }

    /// Create a model with TP-aware scratch buffer sizing.
    ///
    /// For `world_size > 1`, Q/KV/attention/FFN scratch buffers are sized
    /// for the per-rank dimensions (heads / world_size, inter / world_size),
    /// while hidden and logits buffers remain full-sized.
    pub fn new_tp(
        config: ModelConfig,
        weights: ModelWeights<B>,
        backend: &'a B,
        world_size: usize,
    ) -> anyhow::Result<Self> {
        if world_size == 1 {
            return Self::new(config, weights, backend);
        }

        // Flat KV cache not used for TP — the multi-GPU orchestrator manages
        // paged KV pools externally.  Use a minimal flat cache as placeholder.
        let kv_dim = (config.num_key_value_heads / world_size) * config.head_dim;
        let num_kv_layers = config.num_kv_layers();
        let mut k_cache = Vec::with_capacity(num_kv_layers);
        let mut v_cache = Vec::with_capacity(num_kv_layers);
        for _ in 0..num_kv_layers {
            k_cache.push(backend.alloc_tensor(&[1, kv_dim], TensorDtype::BF16));
            v_cache.push(backend.alloc_tensor(&[1, kv_dim], TensorDtype::BF16));
        }
        let kv_mode = KvMode::Flat {
            k_cache,
            v_cache,
            pos: 0,
        };

        Self::new_with_kv_mode_tp(config, weights, backend, kv_mode, world_size)
    }

    /// Internal: create model with a specific KV mode.
    fn new_with_kv_mode(
        config: ModelConfig,
        weights: ModelWeights<B>,
        backend: &'a B,
        kv_mode: KvMode<B>,
    ) -> anyhow::Result<Self> {
        let arch = config.arch()?;
        let hidden = config.hidden_size;
        // Q dimension: num_heads × head_dim.  Usually equals hidden_size, but
        // Qwen3 MoE has hidden=2048 with 32×128=4096 attention dimension.
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let inter = config.effective_intermediate_size();
        let vocab = config.vocab_size;
        let is_hybrid = config.is_hybrid_deltanet();

        // For hybrid models, scratch buffers must be sized for the max dimension
        // across both layer types (DeltaNet and GQA).
        let (eff_q_dim, eff_kv_dim, eff_attn_dim) = if is_hybrid {
            let dn_qk_dim = config.linear_num_key_heads * config.linear_key_head_dim;
            let dn_v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
            (
                q_dim.max(dn_qk_dim),  // GQA Q vs DeltaNet Q
                kv_dim.max(dn_qk_dim), // GQA KV vs DeltaNet K
                q_dim.max(dn_v_dim),   // GQA attn_out vs DeltaNet V output
            )
        } else {
            (q_dim, kv_dim, q_dim)
        };

        // Allocate scratch buffers — one of each, reused across all layers.
        let hidden_buf = backend.alloc_tensor(&[hidden], TensorDtype::BF16);
        let norm_buf = backend.alloc_tensor(&[hidden], TensorDtype::BF16);
        let q_buf = backend.alloc_tensor(&[eff_q_dim], TensorDtype::BF16);
        let k_buf = backend.alloc_tensor(&[eff_kv_dim], TensorDtype::BF16);
        let v_buf = backend.alloc_tensor(&[eff_attn_dim], TensorDtype::BF16);
        let attn_out = backend.alloc_tensor(&[eff_attn_dim], TensorDtype::BF16);
        let gate_buf = backend.alloc_tensor(&[inter], TensorDtype::BF16);
        let up_buf = backend.alloc_tensor(&[inter], TensorDtype::BF16);
        let logits_buf = backend.alloc_tensor(&[vocab], TensorDtype::BF16);

        // MoE buffers: only allocated for MoE models.
        let (router_logits, moe_gate_buf, moe_up_buf, moe_output, routing_output) =
            if config.is_moe() {
                let moe_inter = config.moe_intermediate_size;
                let k = config.num_experts_per_tok;
                (
                    // Router logits are f32 for precision during top-k selection.
                    Some(backend.alloc_tensor(&[config.num_experts], TensorDtype::F32)),
                    Some(backend.alloc_tensor(&[moe_inter], TensorDtype::BF16)),
                    Some(backend.alloc_tensor(&[moe_inter], TensorDtype::BF16)),
                    Some(backend.alloc_tensor(&[hidden], TensorDtype::BF16)),
                    // GPU routing output: [2*k] f32 (index, weight) pairs.
                    Some(backend.alloc_tensor(&[2 * k], TensorDtype::F32)),
                )
            } else {
                (None, None, None, None, None)
            };

        // DeltaNet state and scratch buffers: only for hybrid models.
        let (
            deltanet_states,
            deltanet_conv_history,
            dn_qkv_buf,
            dn_alpha_buf,
            dn_beta_buf,
            dn_z_buf,
            dn_conv_out,
            dn_attn_out,
            dn_norm_out,
        ) = if is_hybrid {
            let num_qk_heads = config.linear_num_key_heads;
            let num_v_heads = config.linear_num_value_heads;
            let hd = config.linear_key_head_dim;
            let v_per_qk = num_v_heads / num_qk_heads;
            let v_dim = num_v_heads * config.linear_value_head_dim;
            let qk_dim = num_qk_heads * hd;
            let conv_dim = qk_dim * 2 + v_dim; // Q + K + V concatenated
            let kernel_size = config.linear_conv_kernel_dim;

            // State matrices: one [num_qk_heads * v_per_qk * hd * hd] f32 per DeltaNet layer.
            let num_dn_layers = config
                .layer_types
                .iter()
                .filter(|t| t.as_str() == "linear_attention")
                .count();
            let state_size = num_qk_heads * v_per_qk * hd * hd;
            let mut states = Vec::with_capacity(num_dn_layers);
            let mut conv_histories = Vec::with_capacity(num_dn_layers);
            for _ in 0..num_dn_layers {
                states.push(backend.alloc_tensor(&[state_size], TensorDtype::F32));
                conv_histories
                    .push(backend.alloc_tensor(&[(kernel_size - 1) * conv_dim], TensorDtype::BF16));
            }

            // Zero-initialise states and conv histories.
            for s in &states {
                backend.fill_zero(s, state_size as u32);
            }
            for h in &conv_histories {
                backend.fill_zero(h, ((kernel_size - 1) * conv_dim) as u32);
            }

            (
                Some(states),
                Some(conv_histories),
                Some(backend.alloc_tensor(&[conv_dim], TensorDtype::BF16)),
                Some(backend.alloc_tensor(&[num_v_heads], TensorDtype::F32)),
                Some(backend.alloc_tensor(&[num_v_heads], TensorDtype::F32)),
                Some(backend.alloc_tensor(&[v_dim], TensorDtype::BF16)),
                Some(backend.alloc_tensor(&[conv_dim], TensorDtype::BF16)),
                Some(backend.alloc_tensor(&[v_dim], TensorDtype::BF16)),
                Some(backend.alloc_tensor(&[v_dim], TensorDtype::BF16)),
            )
        } else {
            (None, None, None, None, None, None, None, None, None)
        };

        // Mamba-2 state and scratch buffers: only for Nemotron-H.
        let is_mamba2 = config.is_hybrid_mamba2();
        let (mamba2_states, mamba2_conv_history, m2_in_proj_buf, m2_conv_out, m2_ssm_out) =
            if is_mamba2 {
                let d_inner = config.mamba2_d_inner();
                let conv_dim = config.mamba2_conv_dim(); // d_inner + 2*n_groups*state_size
                let in_proj_dim = config.mamba2_in_proj_dim();
                let ks = config.mamba_conv_kernel;
                let n_heads = config.mamba_num_heads;
                let hd = config.mamba_head_dim;
                let ss = config.ssm_state_size;

                let num_m2_layers = config
                    .layer_types
                    .iter()
                    .filter(|t| t.as_str() == "mamba2")
                    .count();
                let state_size = n_heads * hd * ss;
                let mut states = Vec::with_capacity(num_m2_layers);
                let mut conv_histories = Vec::with_capacity(num_m2_layers);
                for _ in 0..num_m2_layers {
                    states.push(backend.alloc_tensor(&[state_size], TensorDtype::F32));
                    // Conv history is [(ks-1) * conv_dim] because conv1d operates
                    // on the full [x, B, C] concatenation, not just x.
                    conv_histories.push(
                        backend.alloc_tensor(&[(ks - 1) * conv_dim], TensorDtype::BF16),
                    );
                }
                // Zero-initialise states and conv histories.
                for s in &states {
                    backend.fill_zero(s, state_size as u32);
                }
                for h in &conv_histories {
                    backend.fill_zero(h, ((ks - 1) * conv_dim) as u32);
                }

                (
                    Some(states),
                    Some(conv_histories),
                    Some(backend.alloc_tensor(&[in_proj_dim], TensorDtype::BF16)),
                    // conv_out needs conv_dim (not d_inner) — it holds [x, B, C] post-conv.
                    Some(backend.alloc_tensor(&[conv_dim], TensorDtype::BF16)),
                    Some(backend.alloc_tensor(&[d_inner], TensorDtype::BF16)),
                )
            } else {
                (None, None, None, None, None)
            };

        let kv_layer_map = config.kv_layer_map();

        Ok(Self {
            config,
            weights,
            backend,
            arch,
            world_size: 1,
            kv_mode,
            hidden: hidden_buf,
            norm_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out,
            gate_buf,
            up_buf,
            logits_buf,
            router_logits,
            moe_gate_buf,
            moe_up_buf,
            moe_output,
            routing_output,
            deltanet_states,
            deltanet_conv_history,
            dn_qkv_buf,
            dn_alpha_buf,
            dn_beta_buf,
            dn_z_buf,
            dn_conv_out,
            dn_attn_out,
            dn_norm_out,
            mamba2_states,
            mamba2_conv_history,
            m2_in_proj_buf,
            m2_conv_out,
            m2_ssm_out,
            kv_layer_map,
            expert_streamer: None,
            vision_weights: None,
            vision_bufs: None,
            turbo_ctx: None,
        })
    }

    /// Internal: create model with TP-aware scratch buffer sizing.
    ///
    /// Identical to `new_with_kv_mode` but Q/KV/attn/FFN buffers are sized
    /// for `world_size`-way tensor parallelism.
    fn new_with_kv_mode_tp(
        config: ModelConfig,
        weights: ModelWeights<B>,
        backend: &'a B,
        kv_mode: KvMode<B>,
        world_size: usize,
    ) -> anyhow::Result<Self> {
        if world_size <= 1 {
            return Self::new_with_kv_mode(config, weights, backend, kv_mode);
        }

        let arch = config.arch()?;
        let hidden = config.hidden_size;
        let q_dim = (config.num_attention_heads / world_size) * config.head_dim;
        let kv_dim = (config.num_key_value_heads / world_size) * config.head_dim;
        let inter = config.effective_intermediate_size() / world_size;
        let vocab = config.vocab_size;
        let is_hybrid = config.is_hybrid_deltanet();

        // For hybrid models, scratch buffers must be sized for the max dimension
        // across both layer types (DeltaNet and GQA), with TP-aware splitting.
        let (eff_q_dim, eff_kv_dim, eff_attn_dim) = if is_hybrid {
            let dn_qk_dim = (config.linear_num_key_heads / world_size) * config.linear_key_head_dim;
            let dn_v_dim =
                (config.linear_num_value_heads / world_size) * config.linear_value_head_dim;
            (
                q_dim.max(dn_qk_dim),
                kv_dim.max(dn_qk_dim),
                q_dim.max(dn_v_dim),
            )
        } else {
            (q_dim, kv_dim, q_dim)
        };

        // Allocate scratch buffers — TP-aware sizes.
        let hidden_buf = backend.alloc_tensor(&[hidden], TensorDtype::BF16);
        let norm_buf = backend.alloc_tensor(&[hidden], TensorDtype::BF16);
        let q_buf = backend.alloc_tensor(&[eff_q_dim], TensorDtype::BF16);
        let k_buf = backend.alloc_tensor(&[eff_kv_dim], TensorDtype::BF16);
        let v_buf = backend.alloc_tensor(&[eff_kv_dim], TensorDtype::BF16);
        let attn_out = backend.alloc_tensor(&[eff_attn_dim], TensorDtype::BF16);
        let gate_buf = backend.alloc_tensor(&[inter], TensorDtype::BF16);
        let up_buf = backend.alloc_tensor(&[inter], TensorDtype::BF16);
        let logits_buf = backend.alloc_tensor(&[vocab], TensorDtype::BF16);

        // MoE buffers: TP splits each expert's weights, so moe_inter is divided.
        let (router_logits, moe_gate_buf, moe_up_buf, moe_output, routing_output) =
            if config.is_moe() {
                let moe_inter = config.moe_intermediate_size / world_size;
                let k = config.num_experts_per_tok;
                (
                    Some(backend.alloc_tensor(&[config.num_experts], TensorDtype::F32)),
                    Some(backend.alloc_tensor(&[moe_inter], TensorDtype::BF16)),
                    Some(backend.alloc_tensor(&[moe_inter], TensorDtype::BF16)),
                    Some(backend.alloc_tensor(&[hidden], TensorDtype::BF16)),
                    Some(backend.alloc_tensor(&[2 * k], TensorDtype::F32)),
                )
            } else {
                (None, None, None, None, None)
            };

        // DeltaNet state and scratch buffers: TP-aware sizes.
        let (
            deltanet_states,
            deltanet_conv_history,
            dn_qkv_buf,
            dn_alpha_buf,
            dn_beta_buf,
            dn_z_buf,
            dn_conv_out,
            dn_attn_out,
            dn_norm_out,
        ) = if is_hybrid {
            let num_qk_heads = config.linear_num_key_heads / world_size;
            let num_v_heads = config.linear_num_value_heads / world_size;
            let hd = config.linear_key_head_dim;
            let v_per_qk = num_v_heads / num_qk_heads;
            let v_dim = num_v_heads * config.linear_value_head_dim;
            let qk_dim = num_qk_heads * hd;
            let conv_dim = qk_dim * 2 + v_dim;
            let kernel_size = config.linear_conv_kernel_dim;

            let num_dn_layers = config
                .layer_types
                .iter()
                .filter(|t| t.as_str() == "linear_attention")
                .count();
            let state_size = num_qk_heads * v_per_qk * hd * hd;
            let mut states = Vec::with_capacity(num_dn_layers);
            let mut conv_histories = Vec::with_capacity(num_dn_layers);
            for _ in 0..num_dn_layers {
                states.push(backend.alloc_tensor(&[state_size], TensorDtype::F32));
                conv_histories
                    .push(backend.alloc_tensor(&[(kernel_size - 1) * conv_dim], TensorDtype::BF16));
            }
            for s in &states {
                backend.fill_zero(s, state_size as u32);
            }
            for h in &conv_histories {
                backend.fill_zero(h, ((kernel_size - 1) * conv_dim) as u32);
            }

            (
                Some(states),
                Some(conv_histories),
                Some(backend.alloc_tensor(&[conv_dim], TensorDtype::BF16)),
                Some(backend.alloc_tensor(&[num_v_heads], TensorDtype::F32)),
                Some(backend.alloc_tensor(&[num_v_heads], TensorDtype::F32)),
                Some(backend.alloc_tensor(&[v_dim], TensorDtype::BF16)),
                Some(backend.alloc_tensor(&[conv_dim], TensorDtype::BF16)),
                Some(backend.alloc_tensor(&[v_dim], TensorDtype::BF16)),
                Some(backend.alloc_tensor(&[v_dim], TensorDtype::BF16)),
            )
        } else {
            (None, None, None, None, None, None, None, None, None)
        };

        let kv_layer_map = config.kv_layer_map();

        Ok(Self {
            config,
            weights,
            backend,
            arch,
            world_size,
            kv_mode,
            hidden: hidden_buf,
            norm_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out,
            gate_buf,
            up_buf,
            logits_buf,
            router_logits,
            moe_gate_buf,
            moe_up_buf,
            moe_output,
            routing_output,
            deltanet_states,
            deltanet_conv_history,
            dn_qkv_buf,
            dn_alpha_buf,
            dn_beta_buf,
            dn_z_buf,
            dn_conv_out,
            dn_attn_out,
            dn_norm_out,
            // Mamba-2 state: not yet TP-aware, use None for TP path.
            // TODO: add TP-aware Mamba-2 state allocation.
            mamba2_states: None,
            mamba2_conv_history: None,
            m2_in_proj_buf: None,
            m2_conv_out: None,
            m2_ssm_out: None,
            kv_layer_map,
            expert_streamer: None,
            vision_weights: None,
            vision_bufs: None,
            turbo_ctx: None,
        })
    }

    /// Accessor for config (needed by engine for buffer allocation).
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get TP-aware dimensions for the forward pass.
    pub fn dims(&self) -> primitives::Dims {
        if self.world_size > 1 {
            primitives::Dims::from_config_tp(&self.config, self.world_size)
        } else {
            primitives::Dims::from_config(&self.config)
        }
    }
}

// ---------------------------------------------------------------------------
// Read-only accessors: only need GpuCore (tensor types).
// ---------------------------------------------------------------------------

impl<'a, B: GpuCore> Model<'a, B> {
    /// Returns a reference to the logits buffer (vocab-sized output).
    /// Call this after `forward()` to read the model's predictions.
    pub fn logits(&self) -> &B::Tensor {
        &self.logits_buf
    }
}

// ---------------------------------------------------------------------------
// Forward pass dispatch: needs GpuBackend (full supertrait) because the
// dispatch hub calls into ALL architectures, including Qwen 3.5 which
// requires GpuDeltaNet.
// ---------------------------------------------------------------------------

impl<'a, B: GpuBackend> Model<'a, B> {
    // =======================================================================
    // Forward pass dispatch — model registry.
    //
    // Each model family lives in its own file under registry/ and composes
    // shared primitives from primitives.rs.  The dispatch chain is:
    //   Model::forward_*() → registry::llama/qwen/... → primitives → Gpu*
    //
    // Two entry points, matching the two inference modes:
    //   - forward_single_paged: decode one token (mat-vec, paged KV).
    //   - forward_prefill_paged: process prompt chunk (GEMM, paged KV).
    //
    // The legacy `forward()` (flat/paged KV, `&mut self`) was removed:
    // no call sites remain — the engine exclusively uses the paged path.
    // =======================================================================

    /// Decode a single token using an external paged KV pool and sequence state.
    ///
    /// Used by the engine for continuous batching: the model's own kv_mode
    /// is ignored — the caller provides the pool and sequence state for the
    /// specific sequence being processed.
    pub fn forward_single_paged(
        &self,
        token_id: u32,
        pool: &KvPool<B>,
        seq_state: &SeqKvState<B>,
    ) -> anyhow::Result<()> {
        match self.arch {
            ModelArch::Gemma3 => {
                registry::gemma::forward_single_paged(self, token_id, pool, seq_state)
            }
            ModelArch::GptOss => {
                registry::gpt_oss::forward_single_paged(self, token_id, pool, seq_state)
            }
            ModelArch::Llama => {
                registry::llama::forward_single_paged(self, token_id, pool, seq_state)
            }
            ModelArch::Mistral => {
                registry::mistral::forward_single_paged(self, token_id, pool, seq_state)
            }
            ModelArch::Mixtral => {
                registry::mixtral::forward_single_paged(self, token_id, pool, seq_state)
            }
            ModelArch::Phi => registry::phi::forward_single_paged(self, token_id, pool, seq_state),
            ModelArch::Qwen2 => {
                registry::qwen::forward_single_paged(self, token_id, pool, seq_state)
            }
            ModelArch::Qwen3Moe => {
                registry::qwen3_moe::forward_single_paged(self, token_id, pool, seq_state)
            }
            ModelArch::Qwen3_5 => {
                registry::qwen3_5::forward_single_paged(self, token_id, pool, seq_state)
            }
            ModelArch::NemotronH => {
                registry::nemotron_h::forward_single_paged(self, token_id, pool, seq_state)
            }
        }
    }

    /// Batched prefill: process `tokens` through the entire model in one pass.
    ///
    /// Uses GEMM for projections, causal prefill attention, and writes K/V
    /// to the paged cache.  Only the last token's logits are produced (into
    /// the model's single-token logits_buf).
    ///
    /// When `images` is non-empty, vision encoding runs after embedding lookup
    /// and the resulting vision tokens are scattered into the embedding buffer
    /// at positions where `token_id == image_token_id`.
    ///
    /// LEARNING NOTE: The prefill pipeline for VLMs is:
    ///
    ///   1. Embedding lookup — convert token IDs to embeddings (shared code)
    ///   2. Vision encode — run the SigLIP ViT on each image (model/vision.rs)
    ///   3. Scatter — overwrite `<image>` placeholder embeddings with vision
    ///      encoder output (gpu/ops/vision.rs :: scatter_vision_tokens)
    ///   4. Transformer forward — arch-specific attention + FFN layers
    ///
    /// Steps 1-3 are shared across ALL architectures and happen here in mod.rs.
    /// Embedding lookup used to live inside each arch's prefill function, but
    /// was hoisted out so that vision scatter can happen between embedding and
    /// the transformer — the scatter must overwrite rows in the embedding
    /// buffer *before* the first transformer layer sees them.  This avoids
    /// duplicating vision logic in every architecture file.
    ///
    /// The caller must:
    ///   - Pre-allocate KV blocks via `seq_state.ensure_slots(tokens.len())`
    ///   - Sync the block table via `seq_state.sync_block_table()`
    ///   - Advance seq_state after this returns via `seq_state.advance_by(tokens.len())`
    pub fn forward_prefill_paged(
        &self,
        tokens: &[u32],
        pool: &KvPool<B>,
        seq_state: &SeqKvState<B>,
        bufs: &PrefillBuffers<B>,
        images: &[vision::ProcessedImage],
    ) -> anyhow::Result<()> {
        let bs = tokens.len() as u32;
        let start_pos = seq_state.seq_len as u32;

        // Shared embedding: upload token IDs and look up embeddings.
        // This used to live inside each arch's prefill function.
        primitives::upload_prefill_inputs(self.backend, bufs, tokens, start_pos, bs);
        primitives::embed_batch(self.backend, &self.weights, bufs, bs, self.config.hidden_size as u32);

        // Gemma 3: scale embeddings by √hidden_size (other archs skip this).
        if self.arch == ModelArch::Gemma3 {
            let embed_scale = (self.config.hidden_size as f32).sqrt();
            self.backend.scalar_mul(&bufs.hidden, &bufs.hidden, embed_scale, bs * self.config.hidden_size as u32);
        }

        // Vision: encode images and scatter into embedding buffer.
        // LEARNING NOTE: This block runs the full vision pipeline — encode each
        // image through the SigLIP ViT, then scatter the resulting embeddings
        // into the text embedding buffer at placeholder positions.  The token_ids
        // buffer (still on GPU from embedding lookup) is reused by the scatter
        // kernel to locate `<image>` placeholder positions.
        if !images.is_empty() {
            if let (Some(vw), Some(vb), Some(vc)) =
                (&self.vision_weights, &self.vision_bufs, &self.config.vision)
            {
                // Encode each image and collect vision embeddings.
                // For now, support single image per prefill chunk.
                for img in images {
                    vision::vision_encode(self.backend, img, vw, vc, vb)?;

                    // Scatter vision embeddings into the text embedding buffer.
                    if let Some(image_token_id) = self.config.image_token_id {
                        self.backend.scatter_vision_tokens(
                            &bufs.hidden,
                            &vb.proj_out,
                            &bufs.token_ids,
                            image_token_id,
                            bs,
                            self.config.hidden_size as u32,
                        );
                    }
                }
            }
        }

        match self.arch {
            ModelArch::Gemma3 => {
                registry::gemma::forward_prefill_paged(self, tokens, pool, seq_state, bufs)
            }
            ModelArch::GptOss => {
                registry::gpt_oss::forward_prefill_paged(self, tokens, pool, seq_state, bufs)
            }
            ModelArch::Llama => {
                registry::llama::forward_prefill_paged(self, tokens, pool, seq_state, bufs)
            }
            ModelArch::Mistral => {
                registry::mistral::forward_prefill_paged(self, tokens, pool, seq_state, bufs)
            }
            ModelArch::Mixtral => {
                registry::mixtral::forward_prefill_paged(self, tokens, pool, seq_state, bufs)
            }
            ModelArch::Phi => {
                registry::phi::forward_prefill_paged(self, tokens, pool, seq_state, bufs)
            }
            ModelArch::Qwen2 => {
                registry::qwen::forward_prefill_paged(self, tokens, pool, seq_state, bufs)
            }
            ModelArch::Qwen3Moe => {
                registry::qwen3_moe::forward_prefill_paged(self, tokens, pool, seq_state, bufs)
            }
            ModelArch::Qwen3_5 => {
                registry::qwen3_5::forward_prefill_paged(self, tokens, pool, seq_state, bufs)
            }
            ModelArch::NemotronH => {
                registry::nemotron_h::forward_prefill_paged(self, tokens, pool, seq_state, bufs)
            }
        }
    }

    /// Whether this architecture supports batched decode.
    ///
    /// Only standard dense transformers (Llama-like) support batched decode
    /// today.  MoE models (Mixtral, Qwen3-MoE, GPT-OSS), Gemma (sandwich
    /// norms), and Qwen 3.5 (DeltaNet layers) fall back to the serial
    /// per-sequence decode path.
    pub fn arch_supports_batched_decode(&self) -> bool {
        matches!(
            self.arch,
            ModelArch::Llama | ModelArch::Mistral | ModelArch::Phi | ModelArch::Qwen2
        )
    }

    /// Batched decode: process N tokens from N different sequences in one pass.
    ///
    /// Turns N mat-vec operations into 1 GEMM for projections and FFN, with
    /// per-sequence paged attention in between.  Produces [N, vocab_size]
    /// logits in `logits_batch` (NOT in the model's single-token logits_buf).
    ///
    /// The caller must:
    ///   - Pre-allocate one KV slot per sequence via `prepare_decode`
    ///   - Sync block tables for all sequences
    ///   - Call `finish_decode` for each sequence after this returns
    pub fn forward_decode_batch_paged(
        &self,
        tokens: &[u32],
        positions: &[u32],
        pool: &KvPool<B>,
        seq_states: &[&SeqKvState<B>],
        bufs: &PrefillBuffers<B>,
        logits_batch: &B::Tensor,
    ) -> anyhow::Result<()> {
        match self.arch {
            ModelArch::Llama => registry::llama::forward_decode_batch_paged(
                self, tokens, positions, pool, seq_states, bufs, logits_batch,
            ),
            ModelArch::Mistral => registry::mistral::forward_decode_batch_paged(
                self, tokens, positions, pool, seq_states, bufs, logits_batch,
            ),
            ModelArch::Phi => registry::phi::forward_decode_batch_paged(
                self, tokens, positions, pool, seq_states, bufs, logits_batch,
            ),
            ModelArch::Qwen2 => registry::qwen::forward_decode_batch_paged(
                self, tokens, positions, pool, seq_states, bufs, logits_batch,
            ),
            _ => anyhow::bail!(
                "batched decode not supported for {:?} (should not be called — \
                 check arch_supports_batched_decode() first)",
                self.arch
            ),
        }
    }
}

// ===========================================================================
// Prefill scratch buffers.
//
// LEARNING OVERVIEW
//
// Why separate buffers?
//   The model has single-token scratch buffers (hidden, norm_buf, q_buf, etc.)
//   sized [dim].  Batched prefill needs [batch_size, dim]-sized buffers.
//   We can't reuse the single-token buffers because:
//     1. They're too small (1D vs 2D).
//     2. The model's single-token buffers are used during decode (generation),
//        which runs AFTER prefill.  They must stay independent.
//
// Memory cost:
//   For max_chunk=1024, Llama 3.2 1B (hidden=2048, kv_dim=512, inter=8192):
//     hidden + norm + q + attn_out: 4 x 1024 x 2048 x 2 bytes = 16 MB
//     k + v:                         2 x 1024 x 512 x 2 bytes  =  2 MB
//     gate + up:                     2 x 1024 x 8192 x 2 bytes = 32 MB
//     Total: ~50 MB (one-time allocation, reused for every prefill).
//
// The buffers act as a "scratchpad" — the GEMM kernels write to them, the
// next layer reads from them, and the cycle repeats.  No data persists
// between prefills.
// ===========================================================================

pub(crate) struct PrefillBuffers<B: GpuCore> {
    pub hidden: B::Tensor,    // [max_chunk, hidden_size]
    pub norm_buf: B::Tensor,  // [max_chunk, hidden_size]
    pub q_buf: B::Tensor,     // [max_chunk, num_heads * head_dim]
    pub k_buf: B::Tensor,     // [max_chunk, kv_dim]
    pub v_buf: B::Tensor,     // [max_chunk, kv_dim]
    pub attn_out: B::Tensor,  // [max_chunk, num_heads * head_dim]
    pub gate_buf: B::Tensor,  // [max_chunk, intermediate_size]
    pub up_buf: B::Tensor,    // [max_chunk, intermediate_size]
    pub positions: B::Tensor, // [max_chunk] u32 (stored as F32 for byte compat)
    pub token_ids: B::Tensor, // [max_chunk] u32 (stored as F32 for byte compat)
    #[allow(dead_code)]
    pub max_chunk: usize,
}

impl<B: GpuCore> PrefillBuffers<B> {
    pub fn new(backend: &B, config: &ModelConfig, max_chunk: usize) -> Self {
        Self::new_tp(backend, config, max_chunk, 1)
    }

    /// TP-aware prefill buffer allocation.
    pub fn new_tp(backend: &B, config: &ModelConfig, max_chunk: usize, world_size: usize) -> Self {
        let hidden = config.hidden_size;
        let q_dim = (config.num_attention_heads / world_size) * config.head_dim;
        let kv_dim = (config.num_key_value_heads / world_size) * config.head_dim;
        let inter = config.effective_intermediate_size() / world_size;

        Self {
            hidden: backend.alloc_tensor(&[max_chunk, hidden], TensorDtype::BF16),
            norm_buf: backend.alloc_tensor(&[max_chunk, hidden], TensorDtype::BF16),
            q_buf: backend.alloc_tensor(&[max_chunk, q_dim], TensorDtype::BF16),
            k_buf: backend.alloc_tensor(&[max_chunk, kv_dim], TensorDtype::BF16),
            v_buf: backend.alloc_tensor(&[max_chunk, kv_dim], TensorDtype::BF16),
            attn_out: backend.alloc_tensor(&[max_chunk, q_dim], TensorDtype::BF16),
            gate_buf: backend.alloc_tensor(&[max_chunk, inter], TensorDtype::BF16),
            up_buf: backend.alloc_tensor(&[max_chunk, inter], TensorDtype::BF16),
            // u32 tensors stored as F32 (same byte size: 4 bytes per element).
            positions: backend.alloc_tensor(&[max_chunk], TensorDtype::F32),
            token_ids: backend.alloc_tensor(&[max_chunk], TensorDtype::F32),
            max_chunk,
        }
    }
}
