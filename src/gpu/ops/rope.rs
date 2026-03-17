// ---------------------------------------------------------------------------
// GpuRope — Rotary Positional Embedding kernels.
//
// RoPE encodes absolute position information into Q and K vectors by
// applying position-dependent sin/cos rotations to pairs of elements.
// Three variants: single-token, batched prefill, and partial (Qwen 3.5
// GQA layers where only the first `rotary_dim` dims are rotated).
//
// Metal shaders: shaders/rope.metal
// Metal impl:    gpu/metal/kernels/rope.rs
// ---------------------------------------------------------------------------

use super::core::GpuCore;

pub(crate) trait GpuRope: GpuCore {
    /// RoPE: rotate Q and K vectors in-place with position-dependent sin/cos.
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

    /// Batched RoPE: per-token positions for prefill.
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

    /// Partial RoPE: rotate only the first `rotary_dim` dims of each head.
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

    /// YaRN RoPE: RoPE with YaRN (Yet another RoPE extensioN) scaling for
    /// extended context lengths.  Uses frequency-dependent interpolation with
    /// NTK-aware scaling — low frequencies are interpolated, high frequencies
    /// are kept at their original values, and mid-range uses a smooth ramp.
    ///
    /// Parameters:
    ///   - `factor`: context extension factor (e.g. 32 for 4k→128k)
    ///   - `beta_fast`: upper frequency boundary (high freq, no interpolation)
    ///   - `beta_slow`: lower frequency boundary (low freq, full interpolation)
    ///   - `original_max_pos`: original training context length (e.g. 4096)
    fn rope_yarn(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        pos: u32,
        rope_theta: f32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_max_pos: u32,
    );

    /// Batched YaRN RoPE for prefill (per-token positions).
    fn rope_yarn_batch(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        positions: &Self::Tensor,
        rope_theta: f32,
        batch_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_max_pos: u32,
    );
}
