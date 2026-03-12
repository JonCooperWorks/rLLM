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
use crate::loader::ModelWeights;

/// Maximum sequence length supported by the KV cache.
/// Llama 3.2 supports up to 131072 with RoPE scaling, but we cap at 4096
/// for Phase 1 (flat cache, no paging).  Increasing this just costs more
/// GPU memory for the cache buffers.
const MAX_SEQ_LEN: usize = 4096;

/// The transformer model: weights, KV cache, scratch buffers, and a backend reference.
///
/// Supports Llama 3 and Qwen 2.5 (same architecture, differs only in QKV bias).
/// Generic over `B: GpuBackend` — the model doesn't know (or care) whether
/// it's running on Metal or CUDA.  The lifetime `'a` ties the model to
/// the backend that owns the GPU device.
pub(crate) struct Model<'a, B: GpuBackend> {
    config: ModelConfig,
    weights: ModelWeights<B>,
    backend: &'a B,

    // -----------------------------------------------------------------------
    // KV cache: per-layer key and value tensors.
    //
    // Each tensor is [MAX_SEQ_LEN, num_kv_heads * head_dim] in bf16.
    // `pos` tracks how many tokens have been written so far — the next token
    // will be written at index `pos`, and attention will scan indices 0..pos.
    // -----------------------------------------------------------------------
    k_cache: Vec<B::Tensor>,
    v_cache: Vec<B::Tensor>,
    pos: usize,

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
    /// Create a new model, allocating KV caches and scratch buffers.
    pub fn new(
        config: ModelConfig,
        weights: ModelWeights<B>,
        backend: &'a B,
    ) -> anyhow::Result<Self> {
        let hidden = config.hidden_size;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        // Allocate per-layer KV caches.
        // Each cache is [MAX_SEQ_LEN, kv_dim] — flattened to a 1D buffer.
        // For Llama 3.2 1B: kv_dim = 8 heads × 64 dim = 512 elements per position.
        let mut k_cache = Vec::with_capacity(config.num_hidden_layers);
        let mut v_cache = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            k_cache.push(backend.alloc_tensor(&[MAX_SEQ_LEN, kv_dim], TensorDtype::BF16));
            v_cache.push(backend.alloc_tensor(&[MAX_SEQ_LEN, kv_dim], TensorDtype::BF16));
        }

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
            k_cache,
            v_cache,
            pos: 0,
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
    pub fn forward(&mut self, token_id: u32) -> anyhow::Result<()> {
        let hidden_size = self.config.hidden_size as u32;
        let num_heads = self.config.num_attention_heads as u32;
        let num_kv_heads = self.config.num_key_value_heads as u32;
        let head_dim = self.config.head_dim as u32;
        let inter_size = self.config.intermediate_size as u32;
        let kv_dim = (self.config.num_key_value_heads * self.config.head_dim) as u32;
        let eps = self.config.rms_norm_eps as f32;
        let rope_theta = self.config.rope_theta as f32;
        let pos = self.pos as u32;

        // -------------------------------------------------------------------
        // Step 1: Embedding lookup.
        //
        // Convert the discrete token ID (e.g. 9822 = "Paris") into a
        // continuous vector in R^2048 by looking up row `token_id` in the
        // embedding table.  This vector enters the residual stream.
        // -------------------------------------------------------------------
        self.backend.embed_lookup(
            &self.weights.embed_tokens,
            token_id,
            &self.hidden,
            hidden_size,
        );

        // -------------------------------------------------------------------
        // Step 2: Transformer layers (×16).
        //
        // Each layer has two sub-blocks:
        //   (a) Self-attention with residual connection
        //   (b) Feed-forward network (FFN) with residual connection
        //
        // The residual stream (`hidden`) flows through unchanged except for
        // the additions after each sub-block.
        // -------------------------------------------------------------------
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.weights.layers[layer_idx];

            // ---------------------------------------------------------------
            // Attention sub-block.
            // ---------------------------------------------------------------

            // RMSNorm: normalise hidden → norm_buf.
            // `hidden` stays untouched — it's the residual for the add later.
            self.backend.rms_norm(
                &self.hidden,
                &layer.input_layernorm,
                eps,
                &self.norm_buf,
            );

            // Q, K, V linear projections from the normalised hidden state.
            //   Q: [2048] → [2048] (32 heads × 64 dim)
            //   K: [2048] → [512]  (8 KV heads × 64 dim)
            //   V: [2048] → [512]  (8 KV heads × 64 dim)
            self.backend.matmul(&layer.q_proj, &self.norm_buf, &self.q_buf, hidden_size, hidden_size);
            self.backend.matmul(&layer.k_proj, &self.norm_buf, &self.k_buf, kv_dim, hidden_size);
            self.backend.matmul(&layer.v_proj, &self.norm_buf, &self.v_buf, kv_dim, hidden_size);

            // Bias-add for models with QKV bias (Qwen 2.5).
            //
            // In a linear layer with bias, the full operation is:
            //   output = W @ input + bias
            // The matmul above computed W @ input.  Now add the bias vector.
            // This reuses the existing `add` kernel — the bias is the same
            // size as the matmul output, so element-wise add works directly.
            //
            // In-place: add(q_buf, bias, q_buf) is safe because each element
            // is independent — reads from both inputs finish before the write.
            //
            // For Llama, these Options are None, so the bias-add is skipped.
            if let Some(ref q_bias) = layer.q_bias {
                self.backend.add(&self.q_buf, q_bias, &self.q_buf, hidden_size);
            }
            if let Some(ref k_bias) = layer.k_bias {
                self.backend.add(&self.k_buf, k_bias, &self.k_buf, kv_dim);
            }
            if let Some(ref v_bias) = layer.v_bias {
                self.backend.add(&self.v_buf, v_bias, &self.v_buf, kv_dim);
            }

            // RoPE: apply rotary positional embeddings to Q and K.
            // This encodes the token's absolute position so the model can
            // distinguish "cat sat on mat" from "mat sat on cat".
            self.backend.rope(
                &self.q_buf,
                &self.k_buf,
                pos,
                rope_theta,
                num_heads,
                num_kv_heads,
                head_dim,
            );

            // Store K and V into the cache at the current position.
            // Future tokens will read these when computing attention.
            self.backend.copy_to_kv_cache(
                &self.k_buf,
                &self.k_cache[layer_idx],
                pos,
                num_kv_heads,
                head_dim,
            );
            self.backend.copy_to_kv_cache(
                &self.v_buf,
                &self.v_cache[layer_idx],
                pos,
                num_kv_heads,
                head_dim,
            );

            // Attention: Q attends to all cached K/V vectors [0..pos+1].
            // Uses grouped-query attention (GQA): 4 Q heads share 1 KV head.
            // Output: weighted sum of V vectors, shape [2048].
            self.backend.attention(
                &self.q_buf,
                &self.k_cache[layer_idx],
                &self.v_cache[layer_idx],
                &self.attn_out,
                pos + 1, // seq_len includes the current token.
                num_heads,
                num_kv_heads,
                head_dim,
            );

            // O projection: [2048] → [2048].  Mixes information across heads.
            self.backend.matmul(&layer.o_proj, &self.attn_out, &self.norm_buf, hidden_size, hidden_size);

            // Residual connection: hidden = hidden + attention_output.
            // Learning note: `hidden` still holds the pre-norm value (the input
            // to this sub-block), and `norm_buf` now holds the O projection output.
            self.backend.add(&self.hidden, &self.norm_buf, &self.hidden, hidden_size);

            // ---------------------------------------------------------------
            // FFN (feed-forward network) sub-block.
            // ---------------------------------------------------------------

            // RMSNorm before FFN (same pattern as before attention).
            self.backend.rms_norm(
                &self.hidden,
                &layer.post_attention_layernorm,
                eps,
                &self.norm_buf,
            );

            // Gate and up projections: [2048] → [8192] each.
            // The FFN expands the hidden dimension by 4× before contracting.
            self.backend.matmul(&layer.gate_proj, &self.norm_buf, &self.gate_buf, inter_size, hidden_size);
            self.backend.matmul(&layer.up_proj, &self.norm_buf, &self.up_buf, inter_size, hidden_size);

            // SwiGLU activation: gate_buf = silu(gate_buf) * up_buf.
            // This is the gated activation function that Llama uses instead of ReLU.
            self.backend.silu_mul(&self.gate_buf, &self.up_buf, &self.gate_buf, inter_size);

            // Down projection: [8192] → [2048].  Contract back to hidden size.
            self.backend.matmul(&layer.down_proj, &self.gate_buf, &self.norm_buf, hidden_size, inter_size);

            // Residual connection: hidden = hidden + ffn_output.
            self.backend.add(&self.hidden, &self.norm_buf, &self.hidden, hidden_size);
        }

        // -------------------------------------------------------------------
        // Step 3: Final normalisation + LM head projection.
        //
        // After all layers, normalise once more, then project from hidden
        // space (2048) to vocabulary space (128256).  The result is logits
        // — unnormalised scores for each token in the vocabulary.
        //
        // Learning note: tie_word_embeddings=true means the LM head reuses
        // the embedding table as its weight matrix.  This is a common trick
        // that saves ~500MB of parameters and empirically works well because
        // "the best output representation for a token should be close to its
        // input representation".
        // -------------------------------------------------------------------
        self.backend.rms_norm(
            &self.hidden,
            &self.weights.norm_weight,
            eps,
            &self.norm_buf,
        );

        // LM head projection.  If tie_word_embeddings=true (Llama 3.2 1B/3B),
        // the embedding table IS the lm_head weight.  Otherwise (Llama 3.1 8B+,
        // all Qwen 2.5), there's a separate lm_head weight tensor.
        let lm_head_weight = self.weights.lm_head.as_ref()
            .unwrap_or(&self.weights.embed_tokens);
        self.backend.matmul(
            lm_head_weight,
            &self.norm_buf,
            &self.logits_buf,
            self.config.vocab_size as u32,
            hidden_size,
        );

        self.pos += 1;
        Ok(())
    }
}
