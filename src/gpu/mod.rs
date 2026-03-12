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
    fn matmul(&self, weight: &Self::Tensor, input: &Self::Tensor, out: &Self::Tensor, m: u32, k: u32);

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

    /// Grouped-Query Attention: compute softmax(Q·K^T / √d) · V
    /// Uses the flat KV cache (all positions 0..seq_len).
    /// With GQA, multiple query heads share one KV head (4:1 in Llama 3.2).
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
    );

    /// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
    /// where silu(x) = x * sigmoid(x).  Used in the FFN block.
    fn silu_mul(&self, gate: &Self::Tensor, up: &Self::Tensor, out: &Self::Tensor, size: u32);

    /// Element-wise addition: out[i] = a[i] + b[i]
    /// Used for residual connections after attention and FFN.
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor, out: &Self::Tensor, size: u32);

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
    );

    /// Paged attention: compute softmax(Q·K^T / √d) · V using a paged KV pool.
    /// Same algorithm as attention() but reads K/V through block table indirection.
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
