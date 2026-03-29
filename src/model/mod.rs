// ===========================================================================
// Transformer model: weights, KV cache, and scratch buffers.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Defines the `Model` struct that holds all inference state (weights,
//   scratch buffers, KV cache).  The Model is shared context — it has no
//   forward pass methods or architecture knowledge.
//
// Forward pass dispatch:
//   Each model family implements the `ModelForward` trait (model/forward.rs)
//   in its own file under model/registry/.  The engine holds a
//   `Box<dyn ModelForward<B>>` constructed at load time (engine/loader.rs).
//   No match dispatch on ModelArch — just trait dispatch.
//
//   Adding a new architecture means:
//     1. Create a new file in model/registry/ (e.g. deepseek.rs)
//     2. Add a variant to ModelArch in config.rs
//     3. Implement ModelForward trait in the new file
//     4. Add a construction match arm in engine/loader.rs
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
pub(crate) mod forward;
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

use self::config::ModelConfig;
use self::kv_cache::{KvPool, SeqKvState};
use self::loader::ModelWeights;
use crate::gpu::{GpuCore, TensorDtype};

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

    // KV layer mapping: layer_idx → kv_pool_idx (None for DeltaNet/Mamba/MoE layers).
    pub(crate) kv_layer_map: Vec<Option<usize>>,

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
// Construction: needs GpuCore for tensor allocation.
// ---------------------------------------------------------------------------

impl<'a, B: GpuCore> Model<'a, B> {
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
        // Validate arch is known (errors early on unsupported models).
        let _arch = config.arch()?;
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


        let kv_layer_map = config.kv_layer_map();

        Ok(Self {
            config,
            weights,
            backend,
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
            kv_layer_map,
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

        let _arch = config.arch()?;
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


        let kv_layer_map = config.kv_layer_map();

        Ok(Self {
            config,
            weights,
            backend,
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
            kv_layer_map,
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

// Forward pass dispatch removed — now lives in model/forward.rs (ModelForward
// trait) and the per-architecture implementations in model/registry/*.rs.
// The engine holds a Box<dyn ModelForward<B>> constructed at load time.


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
