// ===========================================================================
// GPU backend trait abstraction for LLM inference.
//
// LEARNING OVERVIEW
//
// This module defines the interface between the model layer (forward pass,
// KV cache, sampling) and the GPU compute layer (Metal kernels, CUDA
// kernels).  The model code is generic over `B: GpuBackend` and never
// touches platform-specific types — it calls `backend.matmul(...)`,
// `backend.attention(...)`, etc. and the trait implementation dispatches
// the right GPU kernel.
//
// PLATFORM SELECTION
//
// We use OS-conditional compilation (`#[cfg(target_os = "...")]`) rather
// than Cargo feature flags.  On macOS we compile the Metal backend; on
// Linux we compile the CUDA backend.  Only one backend exists in a given
// binary.  The `Backend` type alias resolves to whichever backend is
// active, so call sites just write `gpu::Backend` and `gpu::create_backend()`.
//
// This is the same pattern used in jotcrack (see jotcrack/src/gpu/mod.rs).
//
// ASSOCIATED TYPES
//
// The trait has one associated type: `Tensor`.  Each backend defines its
// own tensor type wrapping the platform's buffer handle plus shape/dtype
// metadata.  For Metal, `MetalTensor` wraps a `metal::Buffer`; for CUDA
// it would wrap a device pointer.  The model layer never sees these
// concrete types — it only works with `B::Tensor`.
// ===========================================================================

// ---------------------------------------------------------------------------
// Conditional module compilation gates.
//
// Only one of these modules is compiled into the binary.  The other module's
// source code still exists on disk but the compiler ignores it entirely.
// This is a zero-cost abstraction — there's no runtime dispatch or vtable.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub(crate) mod metal;

#[cfg(target_os = "linux")]
pub(crate) mod cuda;

// ---------------------------------------------------------------------------
// Platform type aliases.
//
// These aliases let the rest of the crate write `gpu::Backend` without
// knowing which backend is active.  The `main.rs` and `loader.rs` use
// these aliases; the model layer uses the trait directly via generics.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub(crate) type Backend = self::metal::MetalBackend;

#[cfg(target_os = "linux")]
pub(crate) type Backend = self::cuda::CudaBackend;

// ---------------------------------------------------------------------------
// Tensor data types and the memory bandwidth bottleneck.
//
// WHY DTYPE MATTERS FOR PERFORMANCE:
//   LLM inference is *memory-bandwidth bound*, not compute bound.  A single
//   token's forward pass does ~145 matrix-vector multiplies.  Each one reads
//   the entire weight matrix from memory but produces just a single output
//   vector.  The arithmetic intensity (FLOPs per byte loaded) is extremely
//   low — roughly 1 FLOP per byte.
//
//   On M4 Max: ~546 GB/s memory bandwidth, ~56 TFLOPS compute.
//   For a bf16 matmul reading a [4096, 4096] weight matrix:
//     Data loaded: 4096 × 4096 × 2 = 32 MB → takes 32/546 = 0.059 ms
//     Compute:     4096 × 4096 × 2 = 33.5 MFLOP → takes 33.5M/56T = 0.0006 ms
//   The GPU spends 99% of its time waiting for memory, not computing.
//
//   This means: SMALLER DTYPE = FASTER INFERENCE.  If we halve the weight
//   bytes, we halve the memory load time, which dominates wall clock time.
//   That's why Q4 (0.625 bytes/weight) is ~2x faster than bf16 (2 bytes/weight).
//
// DTYPE HIERARCHY:
//   F32:  4 bytes/weight — full precision, used for some norm weights
//   BF16: 2 bytes/weight — standard inference precision, good quality
//   Q4:   0.625 bytes/weight (20 bytes / 32 weights) — 3.2× less memory,
//         slightly lower quality, significantly faster matmul
//
// All GPU kernels accumulate in float32 internally for numerical stability,
// then narrow the result back to BF16 for output.  The precision loss from
// BF16 weights is negligible for inference; Q4 loses more but larger models
// (8B+) are robust enough to handle it.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorDtype {
    BF16,
    F32,
    /// Block-wise 4-bit quantization: 32 weights per block, each block is
    /// 20 bytes (4-byte f32 scale + 16 bytes of packed nibbles).
    ///
    /// Memory savings:  bf16 [4096, 4096] = 32 MB → Q4 = 10 MB (3.2× smaller)
    /// Speed impact:    ~1.5-2× faster matmul (fewer bytes to load from memory)
    /// Quality impact:  only 16 distinct values per block (vs 65536 for bf16),
    ///                  acceptable for 3B+ models, noticeable degradation on 1B
    Q4,
}

impl TensorDtype {
    /// Bytes per element for fixed-size types.  Panics for Q4 (use q4_byte_count).
    pub fn byte_size(self) -> usize {
        match self {
            TensorDtype::BF16 => 2,
            TensorDtype::F32 => 4,
            TensorDtype::Q4 => panic!("Q4 has variable byte size; use q4_byte_count()"),
        }
    }
}

/// Compute total byte count for a Q4 weight tensor [m, k].
/// Block layout: 20 bytes per block of 32 weights (4-byte scale + 16 bytes data).
pub(crate) fn q4_byte_count(m: usize, k: usize) -> usize {
    let blocks_per_row = k / 32;
    m * blocks_per_row * 20
}

// ---------------------------------------------------------------------------
// GpuBackend trait — the core abstraction.
//
// Every operation needed by the Llama transformer forward pass is a method
// on this trait.  The methods are deliberately low-level (one method per
// kernel) rather than high-level (e.g. "run one transformer layer") so
// that the forward pass logic lives in Rust, not in platform-specific code.
//
// Design note: `&self` receivers on all methods.  Metal's command buffer
// submission only needs a shared reference to the device and queue.
// CUDA's cudarc API similarly allows kernel launch from shared refs.
// Interior mutability (if needed) is handled inside the backend impl.
//
// `Send + Sync` bounds are required because the backend may be shared
// across async tasks (e.g. the scheduler in Phase 3).
// ---------------------------------------------------------------------------

pub(crate) trait GpuBackend: Send + Sync {
    /// Opaque tensor handle.  Each backend defines its own type wrapping
    /// the platform's buffer handle, tensor shape, and dtype metadata.
    type Tensor;

    /// Human-readable GPU device name (e.g. "Apple M4 Max").
    fn device_name(&self) -> &str;

    /// Maximum recommended GPU working set size in bytes.
    ///
    /// On Apple Silicon (unified memory), this is the GPU's share of system
    /// RAM before performance degrades.  Used to size the KV cache dynamically
    /// so large models don't cause memory pressure and swap.
    fn recommended_max_memory(&self) -> u64;

    /// Wait for all pending GPU work to complete.
    ///
    /// Used for profiling (to time individual components) and by copy_to_host.
    /// Most callers should not need this — it's called automatically when
    /// reading GPU results.
    fn flush(&self);

    /// Submit pending GPU work without waiting for completion.
    ///
    /// Commits the current command buffer so the GPU can start executing it,
    /// but returns immediately without blocking.  Used during prefill to
    /// overlap GPU execution of layer N with CPU encoding of layer N+1.
    fn submit(&self);

    // --- Memory management ---

    /// Allocate an uninitialised tensor on the GPU.
    /// The caller must fill it (via a kernel or upload) before reading.
    fn alloc_tensor(&self, shape: &[usize], dtype: TensorDtype) -> Self::Tensor;

    /// Allocate a tensor and copy `data` (raw bytes) from the host into it.
    /// Used by the weight loader to transfer safetensors data to the GPU.
    fn upload_tensor(&self, data: &[u8], shape: &[usize], dtype: TensorDtype) -> Self::Tensor;

    /// Copy tensor contents from GPU to a host byte buffer.
    /// Used by the sampler to read logits back to the CPU for argmax.
    fn copy_to_host(&self, tensor: &Self::Tensor, dst: &mut [u8]);

    /// Copy raw bytes from the host into an existing GPU tensor.
    /// Used to update the block table for paged KV cache.
    fn copy_to_tensor(&self, tensor: &Self::Tensor, src: &[u8]);

    /// Return the total byte count of a tensor's data.
    fn tensor_byte_count(&self, tensor: &Self::Tensor) -> usize;

    // --- Compute kernels ---
    //
    // Each method corresponds to one GPU kernel dispatch.  The model's
    // forward pass calls these in sequence for each transformer layer.

    /// RMSNorm: out = weight * (input / sqrt(mean(input²) + eps))
    /// Used before every attention and FFN block in Llama.
    fn rms_norm(&self, input: &Self::Tensor, weight: &Self::Tensor, eps: f32, out: &Self::Tensor);

    /// Matrix-vector multiply: out[i] = dot(weight[i, :], input)
    /// `m` = number of output rows, `k` = input dimension.
    /// Called 9 times per layer (Q/K/V/O projections + gate/up/down FFN)
    /// plus once for the final vocabulary projection = 145 calls per token.
    fn matmul(
        &self,
        weight: &Self::Tensor,
        input: &Self::Tensor,
        out: &Self::Tensor,
        m: u32,
        k: u32,
    );

    /// Rotary Positional Embeddings (RoPE): rotate Q and K vectors in-place.
    /// Applies position-dependent sin/cos rotations to pairs of elements,
    /// encoding absolute position information into the attention computation.
    fn rope(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        pos: u32,
        rope_theta: f32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    );

    /// Grouped-Query Attention: compute softmax(Q·K^T / scale) · V
    /// Uses the flat KV cache (all positions 0..seq_len).
    /// With GQA, multiple query heads share one KV head (4:1 in Llama 3.2).
    ///
    /// `window_size`: sliding window (0 = attend to all positions).
    /// `attn_scale`: custom scale factor (0.0 = default 1/√head_dim).
    #[allow(dead_code)]
    fn attention(
        &self,
        q: &Self::Tensor,
        k_cache: &Self::Tensor,
        v_cache: &Self::Tensor,
        out: &Self::Tensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
    );

    /// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
    /// where silu(x) = x * sigmoid(x).  Used in the FFN block (Llama, Qwen, Phi).
    fn silu_mul(&self, gate: &Self::Tensor, up: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// GeGLU activation: out[i] = gelu(gate[i]) * up[i]
    ///
    /// Learning note: Gemma 3 uses GELU instead of SiLU as the gate activation
    /// in its FFN.  The PyTorch tanh-approximated GELU is:
    ///   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    /// Both SwiGLU and GeGLU are "gated linear units" — they just differ in
    /// the gate activation function.
    fn gelu_mul(&self, gate: &Self::Tensor, up: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Scalar multiply: out[i] = scalar * input[i]
    ///
    /// Used by Gemma 3 for embedding scaling: after looking up a token's
    /// embedding vector, it's multiplied by √(hidden_size) to match the
    /// expected magnitude of the residual stream.
    fn scalar_mul(&self, input: &Self::Tensor, out: &Self::Tensor, scalar: f32, size: u32);

    /// Element-wise addition: out[i] = a[i] + b[i]
    /// Used for residual connections after attention and FFN.
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Scaled accumulate: dst[i] += scale * src[i]
    /// Used in MoE to accumulate weighted expert outputs into a running sum.
    fn scale_add(&self, dst: &Self::Tensor, src: &Self::Tensor, scale: f32, size: u32);

    /// Fill tensor with zeros.  Used to clear the MoE accumulator buffer
    /// before summing expert outputs.
    fn fill_zero(&self, dst: &Self::Tensor, size: u32);

    /// Broadcast bias-add: out[i] = input[i] + bias[i % dim]
    /// Adds a [dim] bias vector to each row of a [batch_size, dim] tensor.
    /// Used in batched prefill for Qwen 2.5's QKV bias.
    fn bias_add_batch(
        &self,
        input: &Self::Tensor,
        bias: &Self::Tensor,
        out: &Self::Tensor,
        batch_size: u32,
        dim: u32,
    );

    /// Embedding lookup: copy row `token_id` from the embedding table to `out`.
    /// Converts a discrete token ID into a continuous vector representation.
    fn embed_lookup(
        &self,
        table: &Self::Tensor,
        token_id: u32,
        out: &Self::Tensor,
        hidden_dim: u32,
    );

    /// Write a new K or V vector into the flat KV cache at position `pos`.
    /// The cache layout is [max_seq_len, num_kv_heads, head_dim].
    #[allow(dead_code)]
    fn copy_to_kv_cache(
        &self,
        src: &Self::Tensor,
        cache: &Self::Tensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
    );

    // --- Paged KV cache operations ---
    //
    // These methods work with a block-paged KV cache instead of the flat
    // cache above.  The pool is a large buffer [num_blocks * BLOCK_SIZE, kv_dim]
    // and each sequence has a block table mapping logical blocks to physical ones.

    /// Write a new K or V vector into a PAGED KV cache pool.
    /// The kernel reads the block table to find the physical block for `pos`,
    /// then writes src into pool[physical_block * BLOCK_SIZE + offset].
    fn copy_to_paged_kv_cache(
        &self,
        src: &Self::Tensor,
        pool: &Self::Tensor,
        block_table: &Self::Tensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
    );

    // --- Batched / prefill operations ---
    //
    // These methods process multiple tokens at once.  The projections become
    // GEMM (mat-mat) instead of mat-vec.  Used for batched prefill in Phase 3.

    /// Batched matrix multiply (GEMM): out = input @ weight^T
    /// input: [batch_size, k], weight: [m, k], out: [batch_size, m].
    fn matmul_batch(
        &self,
        weight: &Self::Tensor,
        input: &Self::Tensor,
        out: &Self::Tensor,
        batch_size: u32,
        m: u32,
        k: u32,
    );

    /// Batched RMSNorm: normalise each row of [batch_size, hidden_dim] independently.
    fn rms_norm_batch(
        &self,
        input: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
        out: &Self::Tensor,
        batch_size: u32,
    );

    /// Batched embedding lookup: look up N token IDs, write to [batch_size, hidden_dim].
    fn embed_lookup_batch(
        &self,
        table: &Self::Tensor,
        token_ids: &Self::Tensor,
        out: &Self::Tensor,
        batch_size: u32,
        hidden_dim: u32,
    );

    /// Batched RoPE: apply rotary embeddings to [batch_size, num_heads, head_dim]
    /// Q and K tensors, with per-token positions.
    fn rope_batch(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        positions: &Self::Tensor,
        rope_theta: f32,
        batch_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    );

    /// Write N K/V vectors into a paged KV cache pool at different positions.
    fn copy_to_paged_kv_cache_batch(
        &self,
        src: &Self::Tensor,
        pool: &Self::Tensor,
        block_table: &Self::Tensor,
        positions: &Self::Tensor,
        batch_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
    );

    /// Causal prefill attention: compute causal self-attention for a chunk
    /// of tokens.  Token at position i attends only to positions 0..=i.
    ///
    /// q: [chunk_size, num_heads * head_dim]
    /// k: [chunk_size, num_kv_heads * head_dim]
    /// v: [chunk_size, num_kv_heads * head_dim]
    /// out: [chunk_size, num_heads * head_dim]
    ///
    /// `window_size`: sliding window (0 = full causal attention).
    /// `attn_scale`: custom scale factor (0.0 = default 1/√head_dim).
    fn prefill_attention(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        out: &Self::Tensor,
        chunk_size: u32,
        start_pos: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
    );

    // --- DeltaNet operations (Qwen 3.5 hybrid attention) ---
    //
    // These operations implement Gated DeltaNet linear attention, which replaces
    // softmax attention with a recurrent state matrix in 75% of Qwen 3.5's layers.

    /// Causal depthwise Conv1D for single-token decode.
    /// Convolves the current input with the history buffer using depthwise weights,
    /// then applies SiLU activation.  Provides local positional context for DeltaNet
    /// layers (which don't use RoPE).
    fn conv1d_depthwise_single(
        &self,
        input: &Self::Tensor,
        history: &Self::Tensor,
        weight: &Self::Tensor,
        out: &Self::Tensor,
        dim: u32,
        kernel_size: u32,
    );

    /// Shift Conv1D history buffer: discard oldest entry, append current input.
    fn conv1d_shift_history(
        &self,
        history: &Self::Tensor,
        input: &Self::Tensor,
        dim: u32,
        kernel_size: u32,
    );

    /// L2-normalize each head's vector in place (no learned weights).
    /// After normalization, each head's vector has unit L2 norm.
    /// `elem_offset` allows normalizing a sub-region of the buffer (e.g.,
    /// Q or K portion of a fused QKV buffer).
    fn l2_normalize_heads(
        &self,
        data: &Self::Tensor,
        num_heads: u32,
        head_dim: u32,
        elem_offset: u32,
    );

    /// Element-wise sigmoid: out[i] = 1/(1+exp(-input[i])).
    /// Input is bf16 (from matmul), output is f32 (for DeltaNet gates).
    fn sigmoid(&self, input: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Element-wise sigmoid (bf16→bf16): out[i] = sigmoid(input[i]).
    /// Used for GQA output gating where both input and output are bf16.
    fn sigmoid_bf16(&self, input: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Mamba-style decay gate: g = exp(softplus(x + dt_bias) * (-exp(A_log))).
    /// x is bf16, dt_bias and A_log are f32, output is f32.
    fn deltanet_decay_gate(
        &self,
        x: &Self::Tensor,
        dt_bias: &Self::Tensor,
        a_log: &Self::Tensor,
        out: &Self::Tensor,
        size: u32,
    );

    /// Element-wise SiLU on bf16 tensors: out[i] = silu(input[i]).
    fn silu(&self, input: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Element-wise multiply (bf16): out[i] = a[i] * b[i].
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// DeltaNet recurrent state update + output computation (single token).
    /// Updates the state matrix and computes the attention output for each head.
    /// Q/K/V can be the same buffer at different element offsets (e.g., after
    /// splitting a fused QKV conv output).
    fn deltanet_step(
        &self,
        state: &Self::Tensor,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        alpha: &Self::Tensor,
        beta: &Self::Tensor,
        out: &Self::Tensor,
        num_qk_heads: u32,
        num_v_heads: u32,
        head_dim: u32,
        q_offset: u32,
        k_offset: u32,
        v_offset: u32,
    );

    /// RMSNorm without learned weights: out = input / sqrt(mean(input^2) + eps).
    /// Used in DeltaNet output path before gating.
    fn rms_norm_no_weight(
        &self,
        input: &Self::Tensor,
        out: &Self::Tensor,
        size: u32,
        eps: f32,
    );

    /// Partial RoPE: apply rotary embeddings to only the first `rotary_dim`
    /// dimensions of each head, leaving the rest unchanged.
    /// Used by Qwen 3.5 GQA layers where partial_rotary_factor=0.25.
    fn rope_partial(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        pos: u32,
        rope_theta: f32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
    );

    /// Paged attention: compute softmax(Q·K^T / scale) · V using a paged KV pool.
    /// Same algorithm as attention() but reads K/V through block table indirection.
    ///
    /// `window_size`: sliding window (0 = attend to all positions).
    /// `attn_scale`: custom scale factor (0.0 = default 1/√head_dim).
    fn paged_attention(
        &self,
        q: &Self::Tensor,
        k_pool: &Self::Tensor,
        v_pool: &Self::Tensor,
        block_table: &Self::Tensor,
        out: &Self::Tensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
        attn_scale: f32,
    );

    /// GPU-side top-k selection with softmax for MoE expert routing.
    ///
    /// Input:  logits buffer with at least `num_experts` bf16 values.
    /// Output: [2*k] f32 values — alternating (expert_index, routing_weight).
    ///
    /// Replaces the CPU routing path to eliminate per-layer GPU→CPU syncs
    /// in MoE models.
    fn top_k_softmax(
        &self,
        logits: &Self::Tensor,
        output: &Self::Tensor,
        num_experts: u32,
        k: u32,
    );
}

// ---------------------------------------------------------------------------
// Factory function.
//
// Creates the platform-appropriate backend.  On macOS this initialises
// the Metal device, command queue, and compiles all shader sources.
// On Linux it would initialise the CUDA context.
// ---------------------------------------------------------------------------

pub(crate) fn create_backend() -> anyhow::Result<Backend> {
    Backend::new()
}
