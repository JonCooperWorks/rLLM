// ---------------------------------------------------------------------------
// GpuTurboQuant — TurboQuant KV cache quantization kernels.
//
// TurboQuant (Zandieh et al., arXiv:2504.19874) quantizes KV cache vectors
// by applying a random orthogonal rotation followed by optimal scalar
// quantization (Max-Lloyd) per coordinate.  This trait defines the GPU
// kernel interface for:
//   1. Quantizing K/V vectors into the paged cache (rotation + quantize)
//   2. Pre-rotating Q for efficient inner product computation
//   3. Paged attention with inline dequantization
//
// TurboQuant+ extends the base algorithm with QJL residual quantization:
//   - Stage 1: PolarQuant at (bits-1) bits → centroid codes
//   - Stage 2: QJL sign quantization of the residual → 1-bit signs + gamma norm
//   The `is_plus` flag on quantize/attention methods activates this two-stage
//   pipeline.  When is_plus=false, `bits` is used for centroid codes only.
//   When is_plus=true, `bits` is stage-1 bits and the extra sign bits + gamma
//   are packed after the codes.
//
// Key efficiency insight:
//   K dot products: rotate Q once, then <Pi*Q, dequant(Pi*K)> ≈ <Q,K>.
//   V accumulation: accumulate in rotated space, apply Pi^T once at the end.
//   Per-position cost is just a centroid table lookup (4-16 entries).
//
// Metal shaders: shaders/turboquant.metal
// Metal impl:    gpu/metal/kernels/turboquant.rs
// Algorithm:     model/turboquant.rs
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuTurboQuant: GpuCore {
    /// Rotate and quantize a KV vector, write to quantized paged pool.
    ///
    /// Algorithm per KV head:
    ///   1. Compute norm = ||src_head||₂
    ///   2. Rotate: y = Pi × (src_head / norm)
    ///   3. Scalar quantize each y_j to nearest centroid
    ///   4. Pack b-bit codes + bf16 norm into pool at paged position
    ///
    /// When `is_plus` is true, additionally computes QJL residual:
    ///   5. r_j = y_j - centroid[code_j]
    ///   6. gamma = ||r||₂
    ///   7. sign_j = sign(r_j)
    ///   8. Pack: [bf16 norm] [bf16 gamma] [stage1 codes] [sign bits]
    ///
    /// `src`: [num_kv_heads × head_dim] bf16 — raw K or V after RoPE.
    /// `pool`: quantized paged pool buffer for this layer (K or V).
    /// `block_table`: [MAX_BLOCKS_PER_SEQ] u32 — sequence's block table.
    /// `pi`: [head_dim, head_dim] f32 — rotation matrix.
    /// `centroids`: [num_centroids] f32 — stage-1 codebook centroids.
    /// `bits`: stage-1 bits per coordinate (for centroid code extraction).
    /// `is_plus`: whether to compute and store QJL residual.
    fn turbo_quantize_to_paged(
        &self,
        src: &Self::Tensor,
        pool: &Self::Tensor,
        block_table: &Self::Tensor,
        pi: &Self::Tensor,
        centroids: &Self::Tensor,
        pos: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        bytes_per_head_pos: u32,
        is_plus: bool,
    );

    /// Batched quantize for prefill: N vectors at different positions.
    ///
    /// `src`: [batch_size, num_kv_heads × head_dim] bf16.
    /// `positions`: [batch_size] u32 — position for each vector.
    fn turbo_quantize_to_paged_batch(
        &self,
        src: &Self::Tensor,
        pool: &Self::Tensor,
        block_table: &Self::Tensor,
        positions: &Self::Tensor,
        pi: &Self::Tensor,
        centroids: &Self::Tensor,
        batch_size: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        bytes_per_head_pos: u32,
        is_plus: bool,
    );

    /// Pre-rotate Q for quantized-KV attention: q_rot = Pi × Q.
    ///
    /// Each query head is independently rotated by the shared rotation matrix.
    /// Output is f32 for dot-product precision during attention.
    ///
    /// `q`: [num_heads × head_dim] bf16 — query after RoPE.
    /// `q_rot`: [num_heads × head_dim] f32 — output rotated query.
    /// `pi`: [head_dim, head_dim] f32 — rotation matrix.
    fn turbo_rotate_q(
        &self,
        q: &Self::Tensor,
        q_rot: &Self::Tensor,
        pi: &Self::Tensor,
        num_heads: u32,
        head_dim: u32,
    );

    /// Quantized paged attention: softmax(q_rot · dequant(K)) × dequant(V).
    ///
    /// The attention kernel reads packed codes from the quantized K/V pools,
    /// dequantizes inline via centroid lookup, and computes the standard
    /// softmax attention.  V accumulation happens in rotated space; Pi^T
    /// is applied once per query head at the end.
    ///
    /// When `is_plus` is true, dequantization includes the QJL residual:
    ///   dequant[j] = centroid[code_j] * norm + gamma * sqrt(π/2) / sqrt(hd) * sign_j * norm
    ///
    /// `q_rot`: [num_heads, head_dim] f32 — pre-rotated query.
    /// `k_pool`, `v_pool`: quantized paged pools.
    /// `pi_t`: [head_dim, head_dim] f32 — rotation matrix transpose.
    /// `centroids`: [num_centroids] f32 — stage-1 codebook.
    /// `out`: [num_heads, head_dim] bf16.
    /// `bits`: stage-1 bits per coordinate.
    /// `is_plus`: whether pools contain QJL residual data.
    fn turbo_paged_attention(
        &self,
        q_rot: &Self::Tensor,
        k_pool: &Self::Tensor,
        v_pool: &Self::Tensor,
        block_table: &Self::Tensor,
        pi_t: &Self::Tensor,
        centroids: &Self::Tensor,
        out: &Self::Tensor,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        bytes_per_head_pos: u32,
        window_size: u32,
        attn_scale: f32,
        sinks: Option<&Self::Tensor>,
        is_plus: bool,
    );

    /// V-only quantized paged attention: BF16 K scoring + turbo V accumulation.
    ///
    /// For asymmetric mode (K=BF16, V=TurboQuant): the K pool stores BF16
    /// vectors, so Q·K scoring uses standard dot products (no rotation needed).
    /// The V pool stores turbo-quantized vectors, so V accumulation uses
    /// centroid dequantization in rotated space with Pi^T inverse rotation.
    ///
    /// `q`: [num_heads, head_dim] bf16 — raw query (NOT rotated).
    /// `k_pool`: BF16 paged pool.
    /// `v_pool`: quantized paged pool.
    /// `pi_t`: [head_dim, head_dim] f32 — rotation matrix transpose (for V).
    /// `centroids`: [num_centroids] f32 — stage-1 codebook (for V).
    /// `kv_dim`: num_kv_heads * head_dim (for BF16 K pool addressing).
    /// `v_bytes_per_head_pos`: bytes per V head per position (quantized).
    /// `is_plus`: whether V pool contains QJL residual data.
    fn turbo_paged_attention_v_only(
        &self,
        _q: &Self::Tensor,
        _k_pool: &Self::Tensor,
        _v_pool: &Self::Tensor,
        _block_table: &Self::Tensor,
        _pi_t: &Self::Tensor,
        _centroids: &Self::Tensor,
        _out: &Self::Tensor,
        _seq_len: u32,
        _num_heads: u32,
        _num_kv_heads: u32,
        _head_dim: u32,
        _bits: u32,
        _kv_dim: u32,
        _v_bytes_per_head_pos: u32,
        _window_size: u32,
        _attn_scale: f32,
        _sinks: Option<&Self::Tensor>,
        _is_plus: bool,
    ) {
        unimplemented!("V-only turbo paged attention not implemented on this backend");
    }

    /// Fused: quantize K/V + pre-rotate Q + quantized attention.
    ///
    /// Default implementation calls the four separate methods.  Backends can
    /// override with a fused kernel that avoids redundant block table lookups.
    fn turbo_paged_attention_fused(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        k_pool: &Self::Tensor,
        v_pool: &Self::Tensor,
        block_table: &Self::Tensor,
        pi: &Self::Tensor,
        pi_t: &Self::Tensor,
        centroids: &Self::Tensor,
        q_rot: &Self::Tensor,
        out: &Self::Tensor,
        pos: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bits: u32,
        bytes_per_head_pos: u32,
        window_size: u32,
        attn_scale: f32,
        sinks: Option<&Self::Tensor>,
        is_plus: bool,
    ) {
        // Write quantized K and V to paged cache.
        self.turbo_quantize_to_paged(
            k, k_pool, block_table, pi, centroids,
            pos, num_kv_heads, head_dim, bits, bytes_per_head_pos, is_plus,
        );
        self.turbo_quantize_to_paged(
            v, v_pool, block_table, pi, centroids,
            pos, num_kv_heads, head_dim, bits, bytes_per_head_pos, is_plus,
        );
        // Pre-rotate Q.
        self.turbo_rotate_q(q, q_rot, pi, num_heads, head_dim);
        // Run quantized attention over positions 0..=pos.
        self.turbo_paged_attention(
            q_rot, k_pool, v_pool, block_table, pi_t, centroids, out,
            pos + 1, num_heads, num_kv_heads, head_dim,
            bits, bytes_per_head_pos, window_size, attn_scale, sinks, is_plus,
        );
    }
}
