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
//   Each model family lives in its own file under model/registry/:
//     - model/registry/llama.rs  — Llama 3.x family (no QKV bias)
//     - model/registry/qwen.rs   — Qwen 2.5 family (QKV bias)
//
//   Model families compose shared primitives from model/primitives.rs, which
//   build on top of the GpuBackend trait.  The dispatch chain is:
//     Model::forward() → registry::llama/qwen → primitives → GpuBackend
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
//   Layout: flat arrays of shape [max_seq_len, num_kv_heads * head_dim].
//   Each layer has its own K cache and V cache.  Position `pos` is written
//   once when that token is processed, then read by all future tokens.
//
//   Memory: 2 (K+V) x 16 layers x 4096 positions x 8 heads x 64 dim x 2 bytes
//         = ~64 MB for the full cache.
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
pub(crate) mod kv_cache;
pub(crate) mod loader;
pub(crate) mod primitives;
pub(crate) mod registry;
pub(crate) mod sampler;
pub(crate) mod tokenizer;

use self::config::{ModelArch, ModelConfig};
use self::kv_cache::{KvPool, SeqKvState};
use self::loader::ModelWeights;
use crate::gpu::{GpuBackend, TensorDtype};

/// Maximum sequence length supported by the flat KV cache.
/// Llama 3.2 supports up to 131072 with RoPE scaling, but we cap at 4096
/// for the flat cache mode.  The paged cache mode supports the same limit
/// (256 blocks x 16 positions = 4096) but allocates memory on demand.
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
pub(crate) enum KvMode<B: GpuBackend> {
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

pub(crate) struct Model<'a, B: GpuBackend> {
    pub(crate) config: ModelConfig,
    pub(crate) weights: ModelWeights<B>,
    pub(crate) backend: &'a B,

    /// Which architecture's forward pass to use.
    arch: ModelArch,

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
    //   q_buf      — query projection output [num_heads * head_dim = 2048]
    //   k_buf      — key projection output [num_kv_heads * head_dim = 512]
    //   v_buf      — value projection output [num_kv_heads * head_dim = 512]
    //   attn_out   — attention output [num_heads * head_dim = 2048]
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
}

impl<'a, B: GpuBackend> Model<'a, B> {
    /// Create a new model with flat KV cache (original mode).
    pub fn new(
        config: ModelConfig,
        weights: ModelWeights<B>,
        backend: &'a B,
    ) -> anyhow::Result<Self> {
        let kv_dim = config.num_key_value_heads * config.head_dim;

        // Allocate per-layer KV caches.
        // Each cache is [MAX_SEQ_LEN, kv_dim] — flattened to a 1D buffer.
        // For Llama 3.2 1B: kv_dim = 8 heads x 64 dim = 512 elements per position.
        let mut k_cache = Vec::with_capacity(config.num_hidden_layers);
        let mut v_cache = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
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
        let pool = KvPool::new(backend, num_blocks, kv_dim, config.num_hidden_layers);
        let seq_state = pool.new_sequence(backend);

        let kv_mode = KvMode::Paged { pool, seq_state };
        Self::new_with_kv_mode(config, weights, backend, kv_mode)
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
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        // Allocate scratch buffers — one of each, reused across all layers.
        let hidden_buf = backend.alloc_tensor(&[hidden], TensorDtype::BF16);
        let norm_buf = backend.alloc_tensor(&[hidden], TensorDtype::BF16);
        let q_buf = backend.alloc_tensor(&[hidden], TensorDtype::BF16); // num_heads * head_dim = hidden_size
        let k_buf = backend.alloc_tensor(&[kv_dim], TensorDtype::BF16);
        let v_buf = backend.alloc_tensor(&[kv_dim], TensorDtype::BF16);
        let attn_out = backend.alloc_tensor(&[hidden], TensorDtype::BF16);
        let gate_buf = backend.alloc_tensor(&[inter], TensorDtype::BF16);
        let up_buf = backend.alloc_tensor(&[inter], TensorDtype::BF16);
        let logits_buf = backend.alloc_tensor(&[vocab], TensorDtype::BF16);

        Ok(Self {
            config,
            weights,
            backend,
            arch,
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
        })
    }

    /// Returns a reference to the logits buffer (vocab-sized output).
    /// Call this after `forward()` to read the model's predictions.
    pub fn logits(&self) -> &B::Tensor {
        &self.logits_buf
    }

    // =======================================================================
    // Forward pass dispatch — model registry.
    //
    // Each model family lives in its own file under registry/ and composes
    // shared primitives from primitives.rs.  The dispatch pattern is similar
    // to vLLM's model registry: the Model struct routes to the right family
    // based on ModelArch.
    // =======================================================================

    /// Forward pass with flat/paged KV cache (original mode).
    #[allow(dead_code)]
    pub fn forward(&mut self, token_id: u32) -> anyhow::Result<()> {
        match self.arch {
            ModelArch::Llama => registry::llama::forward(self, token_id),
            ModelArch::Qwen2 => registry::qwen::forward(self, token_id),
        }
    }

    /// Forward pass using an EXTERNAL paged KV pool and sequence state.
    ///
    /// This is used by the engine for continuous batching: the model's own
    /// kv_mode is ignored, and the caller provides the pool and sequence state
    /// for the specific sequence being processed.
    pub fn forward_single_paged(
        &mut self,
        token_id: u32,
        pool: &KvPool<B>,
        seq_state: &SeqKvState<B>,
    ) -> anyhow::Result<()> {
        match self.arch {
            ModelArch::Llama => registry::llama::forward_single_paged(self, token_id, pool, seq_state),
            ModelArch::Qwen2 => registry::qwen::forward_single_paged(self, token_id, pool, seq_state),
        }
    }

    /// Batched prefill: process `tokens` through the entire model in one pass.
    ///
    /// Uses GEMM for projections, causal prefill attention, and writes K/V
    /// to the paged cache.  Only the last token's logits are produced (into
    /// the model's single-token logits_buf).
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
    ) -> anyhow::Result<()> {
        match self.arch {
            ModelArch::Llama => registry::llama::forward_prefill_paged(self, tokens, pool, seq_state, bufs),
            ModelArch::Qwen2 => registry::qwen::forward_prefill_paged(self, tokens, pool, seq_state, bufs),
        }
    }

    /// Accessor for config (needed by engine for buffer allocation).
    pub fn config(&self) -> &ModelConfig {
        &self.config
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

pub(crate) struct PrefillBuffers<B: GpuBackend> {
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

impl<B: GpuBackend> PrefillBuffers<B> {
    pub fn new(backend: &B, config: &ModelConfig, max_chunk: usize) -> Self {
        let hidden = config.hidden_size;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let inter = config.intermediate_size;

        Self {
            hidden: backend.alloc_tensor(&[max_chunk, hidden], TensorDtype::BF16),
            norm_buf: backend.alloc_tensor(&[max_chunk, hidden], TensorDtype::BF16),
            q_buf: backend.alloc_tensor(&[max_chunk, hidden], TensorDtype::BF16),
            k_buf: backend.alloc_tensor(&[max_chunk, kv_dim], TensorDtype::BF16),
            v_buf: backend.alloc_tensor(&[max_chunk, kv_dim], TensorDtype::BF16),
            attn_out: backend.alloc_tensor(&[max_chunk, hidden], TensorDtype::BF16),
            gate_buf: backend.alloc_tensor(&[max_chunk, inter], TensorDtype::BF16),
            up_buf: backend.alloc_tensor(&[max_chunk, inter], TensorDtype::BF16),
            // u32 tensors stored as F32 (same byte size: 4 bytes per element).
            positions: backend.alloc_tensor(&[max_chunk], TensorDtype::F32),
            token_ids: backend.alloc_tensor(&[max_chunk], TensorDtype::F32),
            max_chunk,
        }
    }
}
