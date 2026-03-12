// ===========================================================================
// Transformer forward pass and KV cache management.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the transformer forward pass for Llama 3 and Qwen 2.5 models:
//   given a single token ID, produce logits (unnormalised probabilities) over
//   the vocabulary.  This is the "hot path" of inference — every generated
//   token requires one full forward pass through all transformer layers.
//
// Multi-architecture support:
//   Llama 3 and Qwen 2.5 share the same forward pass structure: RMSNorm,
//   GQA attention, SwiGLU FFN, RoPE.  The ONLY structural difference is that
//   Qwen adds bias to Q/K/V projections (output = W @ x + b instead of W @ x).
//   This is handled with a simple `if let Some(bias)` check — no branching
//   on an architecture enum needed in the hot path.
//
// Forward pass pipeline (one token):
//
//     token_id → embed_lookup → hidden
//     for each layer:
//       hidden → RMSNorm → Q/K/V projections [+ bias if Qwen] → RoPE
//             → KV cache store → attention → O projection → residual add
//       hidden → RMSNorm → gate/up projections → SwiGLU → down projection
//             → residual add
//     hidden → final RMSNorm → lm_head projection → logits
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
//   Memory: 2 (K+V) × 16 layers × 4096 positions × 8 heads × 64 dim × 2 bytes
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

use crate::config::ModelConfig;
use crate::gpu::{GpuBackend, TensorDtype};
use crate::kv_cache::{KvPool, SeqKvState};
use crate::loader::ModelWeights;

/// Maximum sequence length supported by the flat KV cache.
/// Llama 3.2 supports up to 131072 with RoPE scaling, but we cap at 4096
/// for the flat cache mode.  The paged cache mode supports the same limit
/// (256 blocks × 16 positions = 4096) but allocates memory on demand.
const MAX_SEQ_LEN: usize = 4096;

/// The transformer model: weights, KV cache, scratch buffers, and a backend reference.
///
/// Supports Llama 3 and Qwen 2.5 (same architecture, differs only in QKV bias).
/// Generic over `B: GpuBackend` — the model doesn't know (or care) whether
/// it's running on Metal or CUDA.  The lifetime `'a` ties the model to
/// the backend that owns the GPU device.
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
    config: ModelConfig,
    weights: ModelWeights<B>,
    backend: &'a B,

    // -----------------------------------------------------------------------
    // KV cache: either flat or paged.
    // -----------------------------------------------------------------------
    #[allow(dead_code)]
    kv_mode: KvMode<B>,

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
    hidden: B::Tensor,
    norm_buf: B::Tensor,
    q_buf: B::Tensor,
    k_buf: B::Tensor,
    v_buf: B::Tensor,
    attn_out: B::Tensor,
    gate_buf: B::Tensor,
    up_buf: B::Tensor,
    logits_buf: B::Tensor,
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
        // For Llama 3.2 1B: kv_dim = 8 heads × 64 dim = 512 elements per position.
        let mut k_cache = Vec::with_capacity(config.num_hidden_layers);
        let mut v_cache = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            k_cache.push(backend.alloc_tensor(&[MAX_SEQ_LEN, kv_dim], TensorDtype::BF16));
            v_cache.push(backend.alloc_tensor(&[MAX_SEQ_LEN, kv_dim], TensorDtype::BF16));
        }

        let kv_mode = KvMode::Flat { k_cache, v_cache, pos: 0 };
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
    // Forward pass: one token in, logits out.
    //
    // This is the core of the inference engine.  Each call:
    //   1. Looks up the token's embedding vector
    //   2. Passes it through 16 transformer layers
    //   3. Projects to vocabulary logits
    //   4. Advances the KV cache position
    //
    // Learning note: during prefill (processing the prompt), this is called
    // once per prompt token, sequentially.  During generation, it's called
    // once per generated token.  The KV cache accumulates across all calls,
    // so each new token attends to the full history.
    // =======================================================================
    #[allow(dead_code)]
    pub fn forward(&mut self, token_id: u32) -> anyhow::Result<()> {
        let hidden_size = self.config.hidden_size as u32;
        let num_heads = self.config.num_attention_heads as u32;
        let num_kv_heads = self.config.num_key_value_heads as u32;
        let head_dim = self.config.head_dim as u32;
        let inter_size = self.config.intermediate_size as u32;
        let kv_dim = (self.config.num_key_value_heads * self.config.head_dim) as u32;
        let eps = self.config.rms_norm_eps as f32;
        let rope_theta = self.config.rope_theta as f32;

        // Get position and prepare paged state if needed.
        let pos = match &mut self.kv_mode {
            KvMode::Flat { pos, .. } => *pos as u32,
            KvMode::Paged { pool, seq_state } => {
                seq_state.ensure_slot(pool)?;
                seq_state.sync_block_table(self.backend);
                seq_state.seq_len as u32
            }
        };

        // -------------------------------------------------------------------
        // Step 1: Embedding lookup.
        // -------------------------------------------------------------------
        self.backend.embed_lookup(
            &self.weights.embed_tokens,
            token_id,
            &self.hidden,
            hidden_size,
        );

        // -------------------------------------------------------------------
        // Step 2: Transformer layers.
        // -------------------------------------------------------------------
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.weights.layers[layer_idx];

            // ---------------------------------------------------------------
            // Attention sub-block.
            // ---------------------------------------------------------------

            self.backend.rms_norm(
                &self.hidden,
                &layer.input_layernorm,
                eps,
                &self.norm_buf,
            );

            // Q, K, V linear projections.
            self.backend.matmul(&layer.q_proj, &self.norm_buf, &self.q_buf, hidden_size, hidden_size);
            self.backend.matmul(&layer.k_proj, &self.norm_buf, &self.k_buf, kv_dim, hidden_size);
            self.backend.matmul(&layer.v_proj, &self.norm_buf, &self.v_buf, kv_dim, hidden_size);

            // Bias-add for Qwen 2.5.
            if let Some(ref q_bias) = layer.q_bias {
                self.backend.add(&self.q_buf, q_bias, &self.q_buf, hidden_size);
            }
            if let Some(ref k_bias) = layer.k_bias {
                self.backend.add(&self.k_buf, k_bias, &self.k_buf, kv_dim);
            }
            if let Some(ref v_bias) = layer.v_bias {
                self.backend.add(&self.v_buf, v_bias, &self.v_buf, kv_dim);
            }

            // RoPE.
            self.backend.rope(
                &self.q_buf,
                &self.k_buf,
                pos,
                rope_theta,
                num_heads,
                num_kv_heads,
                head_dim,
            );

            // Store K/V and compute attention — dispatch based on KV mode.
            match &self.kv_mode {
                KvMode::Flat { k_cache, v_cache, .. } => {
                    self.backend.copy_to_kv_cache(
                        &self.k_buf, &k_cache[layer_idx], pos, num_kv_heads, head_dim,
                    );
                    self.backend.copy_to_kv_cache(
                        &self.v_buf, &v_cache[layer_idx], pos, num_kv_heads, head_dim,
                    );
                    self.backend.attention(
                        &self.q_buf, &k_cache[layer_idx], &v_cache[layer_idx],
                        &self.attn_out, pos + 1, num_heads, num_kv_heads, head_dim,
                    );
                }
                KvMode::Paged { pool, seq_state } => {
                    self.backend.copy_to_paged_kv_cache(
                        &self.k_buf, &pool.k_pool[layer_idx],
                        &seq_state.block_table_gpu, pos, num_kv_heads, head_dim,
                    );
                    self.backend.copy_to_paged_kv_cache(
                        &self.v_buf, &pool.v_pool[layer_idx],
                        &seq_state.block_table_gpu, pos, num_kv_heads, head_dim,
                    );
                    self.backend.paged_attention(
                        &self.q_buf, &pool.k_pool[layer_idx], &pool.v_pool[layer_idx],
                        &seq_state.block_table_gpu, &self.attn_out,
                        pos + 1, num_heads, num_kv_heads, head_dim,
                    );
                }
            }

            // O projection + residual.
            self.backend.matmul(&layer.o_proj, &self.attn_out, &self.norm_buf, hidden_size, hidden_size);
            self.backend.add(&self.hidden, &self.norm_buf, &self.hidden, hidden_size);

            // ---------------------------------------------------------------
            // FFN sub-block.
            // ---------------------------------------------------------------

            self.backend.rms_norm(
                &self.hidden,
                &layer.post_attention_layernorm,
                eps,
                &self.norm_buf,
            );

            self.backend.matmul(&layer.gate_proj, &self.norm_buf, &self.gate_buf, inter_size, hidden_size);
            self.backend.matmul(&layer.up_proj, &self.norm_buf, &self.up_buf, inter_size, hidden_size);
            self.backend.silu_mul(&self.gate_buf, &self.up_buf, &self.gate_buf, inter_size);
            self.backend.matmul(&layer.down_proj, &self.gate_buf, &self.norm_buf, hidden_size, inter_size);
            self.backend.add(&self.hidden, &self.norm_buf, &self.hidden, hidden_size);
        }

        // -------------------------------------------------------------------
        // Step 3: Final normalisation + LM head projection.
        // -------------------------------------------------------------------
        self.backend.rms_norm(
            &self.hidden,
            &self.weights.norm_weight,
            eps,
            &self.norm_buf,
        );

        let lm_head_weight = self.weights.lm_head.as_ref()
            .unwrap_or(&self.weights.embed_tokens);
        self.backend.matmul(
            lm_head_weight,
            &self.norm_buf,
            &self.logits_buf,
            self.config.vocab_size as u32,
            hidden_size,
        );

        // Advance KV cache position.
        match &mut self.kv_mode {
            KvMode::Flat { pos, .. } => *pos += 1,
            KvMode::Paged { seq_state, .. } => seq_state.advance(),
        }
        Ok(())
    }

    /// Forward pass using an EXTERNAL paged KV pool and sequence state.
    ///
    /// This is used by the engine for continuous batching: the model's own
    /// kv_mode is ignored, and the caller provides the pool and sequence state
    /// for the specific sequence being processed.  The sequence state is NOT
    /// advanced here — the caller (engine) handles that.
    pub fn forward_single_paged(
        &mut self,
        token_id: u32,
        pool: &KvPool<B>,
        seq_state: &SeqKvState<B>,
    ) -> anyhow::Result<()> {
        let hidden_size = self.config.hidden_size as u32;
        let num_heads = self.config.num_attention_heads as u32;
        let num_kv_heads = self.config.num_key_value_heads as u32;
        let head_dim = self.config.head_dim as u32;
        let inter_size = self.config.intermediate_size as u32;
        let kv_dim = (self.config.num_key_value_heads * self.config.head_dim) as u32;
        let eps = self.config.rms_norm_eps as f32;
        let rope_theta = self.config.rope_theta as f32;
        let pos = seq_state.seq_len as u32;

        self.backend.embed_lookup(
            &self.weights.embed_tokens, token_id, &self.hidden, hidden_size,
        );

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.weights.layers[layer_idx];

            self.backend.rms_norm(&self.hidden, &layer.input_layernorm, eps, &self.norm_buf);
            self.backend.matmul(&layer.q_proj, &self.norm_buf, &self.q_buf, hidden_size, hidden_size);
            self.backend.matmul(&layer.k_proj, &self.norm_buf, &self.k_buf, kv_dim, hidden_size);
            self.backend.matmul(&layer.v_proj, &self.norm_buf, &self.v_buf, kv_dim, hidden_size);

            if let Some(ref q_bias) = layer.q_bias {
                self.backend.add(&self.q_buf, q_bias, &self.q_buf, hidden_size);
            }
            if let Some(ref k_bias) = layer.k_bias {
                self.backend.add(&self.k_buf, k_bias, &self.k_buf, kv_dim);
            }
            if let Some(ref v_bias) = layer.v_bias {
                self.backend.add(&self.v_buf, v_bias, &self.v_buf, kv_dim);
            }

            self.backend.rope(&self.q_buf, &self.k_buf, pos, rope_theta, num_heads, num_kv_heads, head_dim);

            self.backend.copy_to_paged_kv_cache(
                &self.k_buf, &pool.k_pool[layer_idx],
                &seq_state.block_table_gpu, pos, num_kv_heads, head_dim,
            );
            self.backend.copy_to_paged_kv_cache(
                &self.v_buf, &pool.v_pool[layer_idx],
                &seq_state.block_table_gpu, pos, num_kv_heads, head_dim,
            );
            self.backend.paged_attention(
                &self.q_buf, &pool.k_pool[layer_idx], &pool.v_pool[layer_idx],
                &seq_state.block_table_gpu, &self.attn_out,
                pos + 1, num_heads, num_kv_heads, head_dim,
            );

            self.backend.matmul(&layer.o_proj, &self.attn_out, &self.norm_buf, hidden_size, hidden_size);
            self.backend.add(&self.hidden, &self.norm_buf, &self.hidden, hidden_size);

            self.backend.rms_norm(&self.hidden, &layer.post_attention_layernorm, eps, &self.norm_buf);
            self.backend.matmul(&layer.gate_proj, &self.norm_buf, &self.gate_buf, inter_size, hidden_size);
            self.backend.matmul(&layer.up_proj, &self.norm_buf, &self.up_buf, inter_size, hidden_size);
            self.backend.silu_mul(&self.gate_buf, &self.up_buf, &self.gate_buf, inter_size);
            self.backend.matmul(&layer.down_proj, &self.gate_buf, &self.norm_buf, hidden_size, inter_size);
            self.backend.add(&self.hidden, &self.norm_buf, &self.hidden, hidden_size);
        }

        self.backend.rms_norm(&self.hidden, &self.weights.norm_weight, eps, &self.norm_buf);
        let lm_head_weight = self.weights.lm_head.as_ref()
            .unwrap_or(&self.weights.embed_tokens);
        self.backend.matmul(lm_head_weight, &self.norm_buf, &self.logits_buf, self.config.vocab_size as u32, hidden_size);

        Ok(())
    }

    // =======================================================================
    // Batched prefill forward pass.
    //
    // LEARNING OVERVIEW
    //
    // What this does:
    //   Processes an entire prompt in ONE forward pass using GEMM (mat-mat)
    //   instead of mat-vec for all projections.  This shifts from bandwidth-
    //   bound to compute-bound, giving 3-10x prefill speedup.
    //
    // The key insight — GEMM vs. mat-vec:
    //   Token-by-token: for 100 tokens, loads the weight matrix 100 times.
    //     Each load does only 1 dot product per row → wasted bandwidth.
    //   Batched: loads the weight matrix ONCE, does 100 dot products per row.
    //     Same memory traffic, 100× more compute → GPU actually busy.
    //
    // How the pipeline works:
    //   1. Upload token IDs and positions to GPU buffers.
    //   2. Batched embedding lookup: [chunk_size] → [chunk_size, hidden_size]
    //   3. For each transformer layer (16 layers for 1B):
    //      a. RMSNorm (batched): normalise each row independently
    //      b. Q/K/V GEMM projections: the big win (mat-mat, not mat-vec)
    //      c. Batched RoPE: position-dependent rotations per token
    //      d. Write K/V to paged cache (for future decode steps)
    //      e. Causal prefill attention: Q @ K^T with triangular mask
    //      f. O projection GEMM + residual add
    //      g. FFN: RMSNorm + gate/up GEMM + SwiGLU + down GEMM + residual
    //   4. Final RMSNorm + LM head projection (LAST token only → single matmul)
    //
    // Element-wise ops (add, silu_mul) need NO new batch kernels:
    //   They operate on flat memory — passing size = batch * dim handles
    //   batched tensors without any kernel changes.  [100, 2048] is just
    //   204800 contiguous bf16 values to the add kernel.
    //
    // LM head optimisation:
    //   Only the LAST token's logits are needed for sampling.  Instead of
    //   computing [chunk_size, vocab_size] (e.g. [100, 128256] — 25M elements!),
    //   we extract just the last hidden state and do a single mat-vec.
    //   This is done via copy_to_host → slice → copy_to_tensor, which
    //   triggers one GPU flush.  Negligible cost at end of prefill.
    // =======================================================================

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
        let chunk_size = tokens.len();
        let bs = chunk_size as u32;
        let hidden_size = self.config.hidden_size as u32;
        let num_heads = self.config.num_attention_heads as u32;
        let num_kv_heads = self.config.num_key_value_heads as u32;
        let head_dim = self.config.head_dim as u32;
        let inter_size = self.config.intermediate_size as u32;
        let kv_dim = (self.config.num_key_value_heads * self.config.head_dim) as u32;
        let eps = self.config.rms_norm_eps as f32;
        let rope_theta = self.config.rope_theta as f32;
        let start_pos = seq_state.seq_len as u32;

        // Upload token IDs and positions to GPU.
        let token_bytes: &[u8] = bytemuck::cast_slice(tokens);
        self.backend.copy_to_tensor(&bufs.token_ids, token_bytes);

        let positions: Vec<u32> = (start_pos..start_pos + bs).collect();
        let pos_bytes: &[u8] = bytemuck::cast_slice(&positions);
        self.backend.copy_to_tensor(&bufs.positions, pos_bytes);

        // Step 1: Batched embedding lookup.
        self.backend.embed_lookup_batch(
            &self.weights.embed_tokens, &bufs.token_ids, &bufs.hidden,
            bs, hidden_size,
        );

        // Step 2: Transformer layers.
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.weights.layers[layer_idx];

            // Attention sub-block.
            self.backend.rms_norm_batch(&bufs.hidden, &layer.input_layernorm, eps, &bufs.norm_buf, bs);

            // Q/K/V GEMM projections — the core speedup.
            self.backend.matmul_batch(&layer.q_proj, &bufs.norm_buf, &bufs.q_buf, bs, hidden_size, hidden_size);
            self.backend.matmul_batch(&layer.k_proj, &bufs.norm_buf, &bufs.k_buf, bs, kv_dim, hidden_size);
            self.backend.matmul_batch(&layer.v_proj, &bufs.norm_buf, &bufs.v_buf, bs, kv_dim, hidden_size);

            // TODO: batched bias-add for Qwen would need a broadcast-add kernel.
            // For now, Llama models have no QKV bias so this is fine.

            // Batched RoPE with per-token positions.
            self.backend.rope_batch(
                &bufs.q_buf, &bufs.k_buf, &bufs.positions,
                rope_theta, bs, num_heads, num_kv_heads, head_dim,
            );

            // Write K/V into paged cache for future decode steps.
            self.backend.copy_to_paged_kv_cache_batch(
                &bufs.k_buf, &pool.k_pool[layer_idx],
                &seq_state.block_table_gpu, &bufs.positions,
                bs, num_kv_heads, head_dim,
            );
            self.backend.copy_to_paged_kv_cache_batch(
                &bufs.v_buf, &pool.v_pool[layer_idx],
                &seq_state.block_table_gpu, &bufs.positions,
                bs, num_kv_heads, head_dim,
            );

            // Causal prefill attention on dense Q/K/V.
            self.backend.prefill_attention(
                &bufs.q_buf, &bufs.k_buf, &bufs.v_buf, &bufs.attn_out,
                bs, start_pos, num_heads, num_kv_heads, head_dim,
            );

            // O projection (GEMM) + residual.
            self.backend.matmul_batch(&layer.o_proj, &bufs.attn_out, &bufs.norm_buf, bs, hidden_size, hidden_size);
            self.backend.add(&bufs.hidden, &bufs.norm_buf, &bufs.hidden, bs * hidden_size);

            // FFN sub-block.
            self.backend.rms_norm_batch(&bufs.hidden, &layer.post_attention_layernorm, eps, &bufs.norm_buf, bs);
            self.backend.matmul_batch(&layer.gate_proj, &bufs.norm_buf, &bufs.gate_buf, bs, inter_size, hidden_size);
            self.backend.matmul_batch(&layer.up_proj, &bufs.norm_buf, &bufs.up_buf, bs, inter_size, hidden_size);
            self.backend.silu_mul(&bufs.gate_buf, &bufs.up_buf, &bufs.gate_buf, bs * inter_size);
            self.backend.matmul_batch(&layer.down_proj, &bufs.gate_buf, &bufs.norm_buf, bs, hidden_size, inter_size);
            self.backend.add(&bufs.hidden, &bufs.norm_buf, &bufs.hidden, bs * hidden_size);
        }

        // Step 3: Final norm + LM head on LAST token only.
        //
        // Only the last token's logits matter for sampling.  We run batched
        // rms_norm on the full batch, then extract the last row to the
        // single-token norm_buf for the vocab projection (single matmul).
        // This avoids a wasteful [chunk_size, 128K] GEMM.
        //
        // The extraction uses copy_to_host + copy_to_tensor, which triggers
        // a GPU flush.  This is a one-time cost at the end of prefill —
        // negligible compared to the GEMM savings in the layer loop.
        self.backend.rms_norm_batch(&bufs.hidden, &self.weights.norm_weight, eps, &bufs.norm_buf, bs);

        let hidden_byte_size = self.config.hidden_size * 2; // bf16
        let full_tensor_bytes = self.backend.tensor_byte_count(&bufs.norm_buf);
        let mut host_buf = vec![0u8; full_tensor_bytes];
        self.backend.copy_to_host(&bufs.norm_buf, &mut host_buf);
        let last_row_start = (chunk_size - 1) * hidden_byte_size;
        self.backend.copy_to_tensor(
            &self.norm_buf,
            &host_buf[last_row_start..last_row_start + hidden_byte_size],
        );

        let lm_head_weight = self.weights.lm_head.as_ref()
            .unwrap_or(&self.weights.embed_tokens);
        self.backend.matmul(
            lm_head_weight, &self.norm_buf, &self.logits_buf,
            self.config.vocab_size as u32, hidden_size,
        );

        Ok(())
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
//     hidden + norm + q + attn_out: 4 × 1024 × 2048 × 2 bytes = 16 MB
//     k + v:                         2 × 1024 × 512 × 2 bytes  =  2 MB
//     gate + up:                     2 × 1024 × 8192 × 2 bytes = 32 MB
//     Total: ~50 MB (one-time allocation, reused for every prefill).
//
// The buffers act as a "scratchpad" — the GEMM kernels write to them, the
// next layer reads from them, and the cycle repeats.  No data persists
// between prefills.
// ===========================================================================

pub(crate) struct PrefillBuffers<B: GpuBackend> {
    pub hidden: B::Tensor,      // [max_chunk, hidden_size]
    pub norm_buf: B::Tensor,    // [max_chunk, hidden_size]
    pub q_buf: B::Tensor,       // [max_chunk, num_heads * head_dim]
    pub k_buf: B::Tensor,       // [max_chunk, kv_dim]
    pub v_buf: B::Tensor,       // [max_chunk, kv_dim]
    pub attn_out: B::Tensor,    // [max_chunk, num_heads * head_dim]
    pub gate_buf: B::Tensor,    // [max_chunk, intermediate_size]
    pub up_buf: B::Tensor,      // [max_chunk, intermediate_size]
    pub positions: B::Tensor,   // [max_chunk] u32 (stored as F32 for byte compat)
    pub token_ids: B::Tensor,   // [max_chunk] u32 (stored as F32 for byte compat)
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
