// ===========================================================================
// ModelForward trait — architecture-specific forward pass dispatch.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Defines the trait that each model architecture implements for its forward
//   pass.  The engine holds a `Box<dyn ModelForward<B>>` and calls into it —
//   no match dispatch on ModelArch, no god-method on Model.
//
//   Each architecture (Llama, Gemma, Mixtral, etc.) implements this trait in
//   its registry file.  Architectures that share a forward pass (Qwen, Phi,
//   Mistral → Llama) use a factory function returning the shared impl with
//   architecture-specific features.
//
// Design:
//   - `forward_decode`: single-token decode (mat-vec path)
//   - `forward_prefill`: batched prefill (GEMM path)
//   - `forward_decode_batch`: optional batched decode (N tokens, N sequences)
//   - `prefill_preamble`: embedding lookup + vision scatter (shared default,
//     overridden by Gemma for √hidden scaling)
//
// Related files:
//   model/registry/*.rs  — trait implementations per architecture
//   engine/mod.rs        — SingleGpuDispatch holds Box<dyn ModelForward<B>>
//   engine/loader.rs     — constructs the Box at load time
//   gpu/multi_gpu.rs     — RankState holds Box<dyn ModelForward<CudaBackend>>
// ===========================================================================

use crate::gpu::{GpuBackend, GpuCore};
use crate::model::expert_stream::ExpertStreamer;
use crate::model::kv_cache::{KvPool, SeqKvState};
use crate::model::vision::{self, ProcessedImage};
use crate::model::{primitives, Model, PrefillBuffers};

// ===========================================================================
// Arch-specific buffer structs.
//
// These hold GPU tensors that only certain architectures need.  Moving them
// out of Model into the ModelForward implementor structs eliminates ~28
// Option<> fields from Model and removes .unwrap() calls in hot paths.
// ===========================================================================

/// MoE scratch buffers and expert streaming state.
///
/// Shared by all MoE architectures: Mixtral, Qwen3-MoE, Qwen3.5, GPT-OSS,
/// Nemotron-H.  Allocated once at model load time, reused every forward pass.
pub(crate) struct MoeBuffers<B: GpuCore> {
    pub router_logits: B::Tensor,    // [num_experts] f32
    pub moe_gate_buf: B::Tensor,     // [moe_inter] bf16
    pub moe_up_buf: B::Tensor,       // [moe_inter] bf16
    pub moe_output: B::Tensor,       // [hidden_size] bf16
    pub routing_output: B::Tensor,   // [2*k] f32 — (expert_index, weight) pairs
    pub expert_streamer: Option<ExpertStreamer<B>>,
}

/// DeltaNet recurrent state and scratch buffers (Qwen 3.5 only).
///
/// DeltaNet layers maintain a fixed-size [head_dim, head_dim] recurrent state
/// matrix per QK-head (in f32 for precision), plus a Conv1D history buffer
/// for causal convolution.  These persist across tokens.
pub(crate) struct DeltaNetBuffers<B: GpuCore> {
    pub states: Vec<B::Tensor>,        // per-DeltaNet-layer f32 state
    pub conv_history: Vec<B::Tensor>,  // per-DeltaNet-layer bf16 history
    pub qkv_buf: B::Tensor,           // [qk_dim*2 + v_dim] bf16
    pub alpha_buf: B::Tensor,         // [num_v_heads] f32
    pub beta_buf: B::Tensor,          // [num_v_heads] f32
    pub z_buf: B::Tensor,             // [v_dim] bf16
    pub conv_out: B::Tensor,          // [conv_dim] bf16
    pub attn_out: B::Tensor,          // [v_dim] bf16
    pub norm_out: B::Tensor,          // [v_dim] bf16
}

/// Mamba-2 SSM recurrent state and scratch buffers (Nemotron-H only).
///
/// Mamba-2 layers maintain a [num_heads, head_dim, state_size] recurrent state
/// matrix per layer in f32.  Unlike KV cache which grows with sequence length,
/// the Mamba state is fixed-size: O(1) in memory and compute.
pub(crate) struct Mamba2Buffers<B: GpuCore> {
    pub states: Vec<B::Tensor>,        // per-Mamba-layer f32 state
    pub conv_history: Vec<B::Tensor>,  // per-Mamba-layer bf16 history
    pub in_proj_buf: B::Tensor,        // [in_proj_dim] bf16
    pub conv_out: B::Tensor,           // [conv_dim] bf16
    pub ssm_out: B::Tensor,            // [d_inner] bf16
}

/// Architecture-specific forward pass.
///
/// Each model family implements this trait.  The engine holds a
/// `Box<dyn ModelForward<B>>` and delegates forward passes to it,
/// eliminating match dispatch on `ModelArch`.
pub(crate) trait ModelForward<B: GpuBackend>: Send + Sync {
    /// Single-token decode (mat-vec path).
    /// Produces logits in `m.logits_buf`.
    fn forward_decode(
        &self,
        m: &Model<'_, B>,
        token_id: u32,
        pool: &KvPool<B>,
        seq_state: &SeqKvState<B>,
    ) -> anyhow::Result<()>;

    /// Batched prefill (GEMM path).
    /// Embeddings already in `bufs.hidden` (preamble ran first).
    /// Produces logits in `m.logits_buf`.
    fn forward_prefill(
        &self,
        m: &Model<'_, B>,
        tokens: &[u32],
        pool: &KvPool<B>,
        seq_state: &SeqKvState<B>,
        bufs: &PrefillBuffers<B>,
    ) -> anyhow::Result<()>;

    /// Whether this architecture supports batched decode.
    fn supports_batched_decode(&self) -> bool {
        false
    }

    /// Batched decode: N tokens from N sequences in one GEMM pass.
    /// Produces [N, vocab_size] logits in `logits_batch`.
    fn forward_decode_batch(
        &self,
        _m: &Model<'_, B>,
        _tokens: &[u32],
        _positions: &[u32],
        _pool: &KvPool<B>,
        _seq_states: &[&SeqKvState<B>],
        _bufs: &PrefillBuffers<B>,
        _logits_batch: &B::Tensor,
    ) -> anyhow::Result<()> {
        anyhow::bail!("batched decode not supported")
    }

    /// Prefill preamble: embed lookup + optional transforms + vision scatter.
    ///
    /// Default: upload token IDs, look up embeddings, encode images, scatter
    /// vision tokens.  Override for arch-specific transforms (e.g. Gemma
    /// scales embeddings by √hidden_size).
    fn prefill_preamble(
        &self,
        m: &Model<'_, B>,
        tokens: &[u32],
        seq_state: &SeqKvState<B>,
        bufs: &PrefillBuffers<B>,
        images: &[ProcessedImage],
    ) -> anyhow::Result<()> {
        default_prefill_preamble(m, tokens, seq_state, bufs, images)
    }
}

/// Shared prefill preamble: embed lookup + vision scatter.
///
/// Extracted as a free function so architectures that override
/// `prefill_preamble` can call this first, then add their own transforms
/// (e.g. Gemma's √hidden scaling).
pub(crate) fn default_prefill_preamble<B: GpuBackend>(
    m: &Model<'_, B>,
    tokens: &[u32],
    seq_state: &SeqKvState<B>,
    bufs: &PrefillBuffers<B>,
    images: &[ProcessedImage],
) -> anyhow::Result<()> {
    let bs = tokens.len() as u32;
    let start_pos = seq_state.seq_len as u32;

    // Upload token IDs and look up embeddings.
    primitives::upload_prefill_inputs(m.backend, bufs, tokens, start_pos, bs);
    primitives::embed_batch(
        m.backend,
        &m.weights,
        bufs,
        bs,
        m.config.hidden_size as u32,
    );

    // Vision: encode images and scatter into embedding buffer.
    if !images.is_empty() {
        if let (Some(vw), Some(vb), Some(vc)) =
            (&m.vision_weights, &m.vision_bufs, &m.config.vision)
        {
            for img in images {
                vision::vision_encode(m.backend, img, vw, vc, vb)?;

                if let Some(image_token_id) = m.config.image_token_id {
                    m.backend.scatter_vision_tokens(
                        &bufs.hidden,
                        &vb.proj_out,
                        &bufs.token_ids,
                        image_token_id,
                        bs,
                        m.config.hidden_size as u32,
                    );
                }
            }
        }
    }

    Ok(())
}
