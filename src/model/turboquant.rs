// ===========================================================================
// TurboQuant — online KV cache vector quantization.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements TurboQuant (Zandieh et al., arXiv:2504.19874), a data-oblivious
//   vector quantization algorithm for compressing the KV cache during inference.
//   At 4 bits per channel it matches full-precision quality while reducing KV
//   cache memory by ~4x.
//
// Why TurboQuant?
//   The KV cache is the primary memory bottleneck for long-context inference.
//   Each token adds num_kv_heads × head_dim × 2 bytes × 2 (K+V) × num_layers
//   to the cache.  For a 32-layer model with 8 KV heads × 128 head_dim, that
//   is 128 KB per token in BF16.  At 128K context, the KV cache alone is 16 GB.
//
//   TurboQuant compresses each KV vector by:
//     1. Applying a random orthogonal rotation (making coordinates approximately
//        independent and Gaussian — a consequence of the high-dimensional
//        concentration of measure phenomenon).
//     2. Applying an optimal Max-Lloyd scalar quantizer per coordinate, using
//        precomputed codebook centroids matched to the Gaussian distribution.
//
//   The codebook is tiny (4-16 entries for 2-4 bits) and the rotation matrix is
//   generated once at model load time from a fixed seed (for reproducibility
//   and prefix cache compatibility).
//
// Attention-time efficiency:
//   The rotation can be folded into the attention computation:
//     - For K (inner products): rotate Q once, then <Pi*Q, dequant(Pi*K)> ≈ <Q,K>.
//       No inverse rotation needed per position — just centroid table lookup.
//     - For V (weighted sums): accumulate dequantized centroids in rotated space,
//       then apply Pi^T once per query head at the end.
//   This means the per-position cost is just a centroid lookup (trivial),
//   while memory bandwidth drops ~4x (the decode bottleneck on Apple Silicon).
//
// Paper citation:
//   "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//   Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google
//   DeepMind), Vahab Mirrokni (Google Research).  arXiv:2504.19874v1, Apr 2025.
//   https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
//
// Storage format:
//   Per KV head per position:
//     [2 bytes bf16 norm] [ceil(head_dim × bits / 8) bytes packed codes]
//   For 4-bit, head_dim=128: 2 + 64 = 66 bytes (vs 256 bytes BF16 = 3.9x compression)
//   For 3-bit, head_dim=128: 2 + 48 = 50 bytes (5.1x compression)
//   For 2-bit, head_dim=128: 2 + 32 = 34 bytes (7.5x compression)
//
// Simplifications vs paper:
//   1. Single shared rotation matrix across all layers (paper is agnostic,
//      but per-layer matrices would give ~1% better quality by decorrelating
//      layer-specific weight distributions — not worth the complexity for v1).
//   2. Hardcoded Max-Lloyd centroids rather than running the Lloyd-Max iterative
//      algorithm at startup.  The centroids for N(0,1) are universal constants,
//      so precomputation is both correct and faster.
//   3. Gram-Schmidt orthogonalisation instead of Householder QR for generating
//      the rotation matrix.  Both produce valid random orthogonal matrices;
//      Gram-Schmidt is simpler and fast enough for head_dim ≤ 256.
//   4. Prefill uses full BF16 attention (K/V are quantized into the paged pool
//      for future decode, but the prefill attention itself reads BF16 Q/K/V
//      directly).  This is strictly better quality than quantized prefill.
//   5. No product quantization — the paper discusses splitting dimensions into
//      subvectors for higher-dimensional inputs, but we use scalar quantization
//      per coordinate only.  At head_dim ≤ 256 this is sufficient.
//
// Related files:
//   gpu/ops/turboquant.rs        — GpuTurboQuant trait (GPU kernel interface)
//   gpu/metal/shaders/turboquant.metal — Metal shader kernels
//   gpu/metal/kernels/turboquant.rs    — Metal dispatch code
//   model/kv_cache.rs            — KvPool (allocates quantized-size buffers)
//   model/primitives.rs          — paged_kv_and_attention_maybe_quantized()
//   docs/turboquant.md           — full documentation
// ===========================================================================

use crate::gpu::{GpuCore, TensorDtype};

// ---------------------------------------------------------------------------
// KV cache quantization mode — selects bit width (or disables quantization).
// ---------------------------------------------------------------------------

/// KV cache quantization mode.
///
/// TurboQuant 4-bit is the default — it matches full-precision quality while
/// reducing KV cache memory by ~4x.  Lower bit widths trade quality for
/// further compression.  `None` disables quantization for debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvQuantMode {
    /// No quantization — store KV in BF16 (debugging/benchmarking only).
    None,
    /// 2-bit TurboQuant (~7.5x compression, marginal quality degradation).
    Turbo2,
    /// 3-bit TurboQuant (~5.1x compression, near-lossless).
    Turbo3,
    /// 4-bit TurboQuant (~3.9x compression, quality-neutral).  Default.
    Turbo4,
}

impl KvQuantMode {
    /// Parse from CLI string: "none", "turbo2", "turbo3", "turbo4".
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "turbo2" => Some(Self::Turbo2),
            "turbo3" => Some(Self::Turbo3),
            "turbo4" => Some(Self::Turbo4),
            _ => Option::None,
        }
    }

    /// Bits per coordinate for this mode.  Returns 0 for None (BF16).
    pub fn bits(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Turbo2 => 2,
            Self::Turbo3 => 3,
            Self::Turbo4 => 4,
        }
    }

    /// Number of codebook centroids: 2^bits.
    pub fn num_centroids(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Turbo2 => 4,
            Self::Turbo3 => 8,
            Self::Turbo4 => 16,
        }
    }

    /// Whether quantization is active.
    pub fn is_quantized(self) -> bool {
        self != Self::None
    }
}

// ---------------------------------------------------------------------------
// Max-Lloyd codebook centroids for the standard Gaussian distribution.
//
// These are the optimal scalar quantizer centroids for N(0, 1), computed via
// the Max-Lloyd (Lloyd-Max) algorithm — an iterative algorithm that solves
// the continuous 1D k-means problem.  Each set of centroids minimizes the
// MSE distortion for quantizing a Gaussian-distributed scalar.
//
// For TurboQuant, the random rotation makes each coordinate of a unit-norm
// vector approximately distributed as N(0, 1/sqrt(d)).  We scale these
// centroids by 1/sqrt(head_dim) at runtime.
//
// References:
//   Lloyd, "Least squares quantization in PCM", IEEE Trans. IT, 1982.
//   Max, "Quantizing for minimum distortion", IRE Trans. IT, 1960.
// ---------------------------------------------------------------------------

/// 2-bit (4 centroids) Max-Lloyd codebook for N(0,1).
/// Optimal partition boundaries: {-0.9816, 0, 0.9816}.
const CENTROIDS_2BIT: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];

/// 3-bit (8 centroids) Max-Lloyd codebook for N(0,1).
/// These minimise MSE for 8-level scalar quantization of Gaussian data.
const CENTROIDS_3BIT: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];

/// 4-bit (16 centroids) Max-Lloyd codebook for N(0,1).
/// These minimise MSE for 16-level scalar quantization of Gaussian data.
const CENTROIDS_4BIT: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
    0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
];

// ---------------------------------------------------------------------------
// TurboQuant configuration — computed once at model load time.
// ---------------------------------------------------------------------------

/// Precomputed TurboQuant parameters for a given bit width and head dimension.
///
/// The centroids are scaled for unit-norm vectors in head_dim dimensions:
/// each coordinate of Pi*x (where ||x||=1) is approximately N(0, 1/sqrt(d)),
/// so we scale the standard Gaussian centroids by 1/sqrt(head_dim).
pub(crate) struct TurboQuantConfig {
    #[allow(dead_code)] // stored for diagnostics and future per-layer mode selection
    pub mode: KvQuantMode,
    pub bits: u32,
    pub num_centroids: u32,
    /// Scaled Max-Lloyd centroids for the target distribution.
    /// Length = num_centroids.  Sorted ascending.
    pub centroids: Vec<f32>,
    /// Bytes per KV head per position: 2 (bf16 norm) + ceil(head_dim × bits / 8).
    pub bytes_per_head_pos: usize,
    /// Head dimension (for validation and kernel dispatch).
    #[allow(dead_code)] // stored for validation; kernel uses bytes_per_head_pos for sizing
    pub head_dim: usize,
}

impl TurboQuantConfig {
    /// Construct TurboQuant configuration for the given mode and head dimension.
    ///
    /// Centroids are scaled from the standard N(0,1) codebook to N(0, 1/√d)
    /// by dividing each centroid by √head_dim.
    pub fn new(mode: KvQuantMode, head_dim: usize) -> Self {
        assert!(mode.is_quantized(), "TurboQuantConfig::new called with None mode");
        let bits = mode.bits();
        let num_centroids = mode.num_centroids();

        // Scale centroids from N(0,1) to N(0, 1/sqrt(d)).
        let scale = 1.0 / (head_dim as f32).sqrt();
        let base_centroids: &[f32] = match mode {
            KvQuantMode::Turbo2 => &CENTROIDS_2BIT,
            KvQuantMode::Turbo3 => &CENTROIDS_3BIT,
            KvQuantMode::Turbo4 => &CENTROIDS_4BIT,
            KvQuantMode::None => unreachable!(),
        };
        let centroids: Vec<f32> = base_centroids.iter().map(|&c| c * scale).collect();

        // Packed code bytes: ceil(head_dim * bits / 8).
        let code_bytes = (head_dim * bits as usize + 7) / 8;
        let bytes_per_head_pos = 2 + code_bytes; // 2 bytes bf16 norm + packed codes

        Self {
            mode,
            bits,
            num_centroids,
            centroids,
            bytes_per_head_pos,
            head_dim,
        }
    }
}

/// Bytes per KV position across all KV heads for one pool (K or V).
///
/// For BF16: num_kv_heads × head_dim × 2.
/// For TurboQuant: num_kv_heads × bytes_per_head_pos.
pub(crate) fn bytes_per_kv_position(
    head_dim: usize,
    num_kv_heads: usize,
    mode: KvQuantMode,
) -> usize {
    if mode.is_quantized() {
        let code_bytes = (head_dim * mode.bits() as usize + 7) / 8;
        num_kv_heads * (2 + code_bytes)
    } else {
        num_kv_heads * head_dim * 2 // BF16
    }
}

// ---------------------------------------------------------------------------
// Rotation matrix generation.
//
// TurboQuant requires a random orthogonal matrix Pi ∈ R^{d×d} to rotate
// input vectors before scalar quantization.  The rotation makes coordinates
// approximately independent and Gaussian (concentration of measure on S^{d-1}).
//
// We generate Pi via QR decomposition of a random Gaussian matrix:
//   1. Fill a d×d matrix G with i.i.d. N(0, 1) entries.
//   2. Compute G = Q × R via Householder QR.
//   3. Pi = Q (the orthogonal factor).
//
// The seed is fixed per model for two reasons:
//   - Reproducibility: same model always uses the same rotation.
//   - Prefix cache compatibility: shared prefix blocks produce identical
//     quantized values across different sessions.
//
// Note: we use a simple Gram-Schmidt implementation rather than pulling in
// a full LAPACK dependency.  For head_dim ≤ 256 this is fast enough (< 1ms).
// ---------------------------------------------------------------------------

/// Generate a random orthogonal matrix of size dim × dim.
///
/// Uses modified Gram-Schmidt orthogonalisation on columns of a random
/// Gaussian matrix.  The seed ensures reproducibility across sessions.
///
/// Returns the matrix in row-major order: element [i, j] = result[i * dim + j].
pub(crate) fn generate_rotation_matrix(dim: usize, seed: u64) -> Vec<f32> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let n = dim;

    // Generate random Gaussian matrix (column-major for Gram-Schmidt).
    let mut cols: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..n).map(|_| rng.random_normal()).collect())
        .collect();

    // Modified Gram-Schmidt orthogonalisation.
    for i in 0..n {
        // Normalise column i.
        let norm = cols[i].iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in cols[i].iter_mut() {
                *x /= norm;
            }
        }

        // Subtract projection of column i from all subsequent columns.
        for j in (i + 1)..n {
            let dot: f32 = cols[i]
                .iter()
                .zip(cols[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            // cols[j] -= dot * cols[i]
            // Can't borrow both mutably, so clone col i.
            let col_i: Vec<f32> = cols[i].clone();
            for (xj, &xi) in cols[j].iter_mut().zip(col_i.iter()) {
                *xj -= dot * xi;
            }
        }
    }

    // Convert to row-major: result[row * dim + col] = cols[col][row].
    let mut result = vec![0.0f32; n * n];
    for col in 0..n {
        for row in 0..n {
            result[row * n + col] = cols[col][row];
        }
    }

    result
}

/// Transpose a row-major dim×dim matrix.
pub(crate) fn transpose_matrix(m: &[f32], dim: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            t[j * dim + i] = m[i * dim + j];
        }
    }
    t
}

// ---------------------------------------------------------------------------
// TurboContext — GPU-resident state for TurboQuant, bundled for forward pass.
//
// Created once at model load and passed by reference to every attention call.
// Holds the rotation matrix, its transpose, the codebook centroids, and a
// scratch buffer for the rotated query vector.
// ---------------------------------------------------------------------------

/// GPU-resident TurboQuant state for a single model instance.
///
/// All tensors are uploaded once at model construction and reused across
/// all layers and tokens.  The rotation matrix is shared across layers
/// (one matrix for all layers) — per-layer matrices would give ~1% better
/// quality but aren't worth the complexity for v1.
pub(crate) struct TurboContext<B: GpuCore> {
    /// Rotation matrix Pi [head_dim, head_dim] in F32.
    pub pi: B::Tensor,
    /// Transpose Pi^T [head_dim, head_dim] in F32.
    pub pi_t: B::Tensor,
    /// Codebook centroids [num_centroids] in F32.
    pub centroids: B::Tensor,
    /// Scratch buffer for rotated query [num_heads × head_dim] in F32.
    /// Reused every layer — only valid between turbo_rotate_q and
    /// turbo_paged_attention within the same layer.
    pub q_rot_buf: B::Tensor,
    /// Configuration (bits, bytes_per_head_pos, etc.).
    pub config: TurboQuantConfig,
}

impl<B: GpuCore> TurboContext<B> {
    /// Create and upload TurboQuant context to GPU.
    ///
    /// The rotation matrix is generated from a fixed seed (42) for
    /// reproducibility and prefix cache compatibility.
    pub fn new(backend: &B, mode: KvQuantMode, head_dim: usize, num_heads: usize) -> Self {
        let config = TurboQuantConfig::new(mode, head_dim);

        // Generate and upload rotation matrix.
        let pi_data = generate_rotation_matrix(head_dim, 42);
        let pi_t_data = transpose_matrix(&pi_data, head_dim);

        let pi_bytes: &[u8] = bytemuck::cast_slice(&pi_data);
        let pi = backend.upload_tensor(pi_bytes, &[head_dim, head_dim], TensorDtype::F32);

        let pi_t_bytes: &[u8] = bytemuck::cast_slice(&pi_t_data);
        let pi_t = backend.upload_tensor(pi_t_bytes, &[head_dim, head_dim], TensorDtype::F32);

        // Upload centroids.
        let centroid_bytes: &[u8] = bytemuck::cast_slice(&config.centroids);
        let centroids = backend.upload_tensor(
            centroid_bytes,
            &[config.num_centroids as usize],
            TensorDtype::F32,
        );

        // Allocate scratch buffer for rotated Q (f32 for precision).
        let q_rot_buf = backend.alloc_tensor(
            &[num_heads * head_dim],
            TensorDtype::F32,
        );

        Self {
            pi,
            pi_t,
            centroids,
            q_rot_buf,
            config,
        }
    }
}

/// Trait extension for random normal generation on rand 0.9 Rng.
///
/// rand 0.9 doesn't ship a built-in Normal distribution in the core crate
/// (it's in rand_distr), so we use the Box-Muller transform.
trait RngNormalExt {
    fn random_normal(&mut self) -> f32;
}

impl<R: rand::Rng> RngNormalExt for R {
    fn random_normal(&mut self) -> f32 {
        // Box-Muller transform: two uniform samples → two independent normals.
        let u1: f32 = self.random::<f32>().max(1e-10); // avoid log(0)
        let u2: f32 = self.random::<f32>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_quant_mode_parsing() {
        assert_eq!(KvQuantMode::from_str("turbo4"), Some(KvQuantMode::Turbo4));
        assert_eq!(KvQuantMode::from_str("turbo3"), Some(KvQuantMode::Turbo3));
        assert_eq!(KvQuantMode::from_str("turbo2"), Some(KvQuantMode::Turbo2));
        assert_eq!(KvQuantMode::from_str("none"), Some(KvQuantMode::None));
        assert_eq!(KvQuantMode::from_str("invalid"), Option::None);
    }

    #[test]
    fn test_kv_quant_mode_bits() {
        assert_eq!(KvQuantMode::None.bits(), 0);
        assert_eq!(KvQuantMode::Turbo2.bits(), 2);
        assert_eq!(KvQuantMode::Turbo3.bits(), 3);
        assert_eq!(KvQuantMode::Turbo4.bits(), 4);
    }

    #[test]
    fn test_bytes_per_kv_position_bf16() {
        // BF16: 8 heads × 128 dim × 2 bytes = 2048
        assert_eq!(bytes_per_kv_position(128, 8, KvQuantMode::None), 2048);
    }

    #[test]
    fn test_bytes_per_kv_position_turbo4() {
        // 4-bit: 8 heads × (2 + 128*4/8) = 8 × (2 + 64) = 8 × 66 = 528
        assert_eq!(bytes_per_kv_position(128, 8, KvQuantMode::Turbo4), 528);
    }

    #[test]
    fn test_bytes_per_kv_position_turbo3() {
        // 3-bit: 8 heads × (2 + ceil(128*3/8)) = 8 × (2 + 48) = 8 × 50 = 400
        assert_eq!(bytes_per_kv_position(128, 8, KvQuantMode::Turbo3), 400);
    }

    #[test]
    fn test_bytes_per_kv_position_turbo2() {
        // 2-bit: 8 heads × (2 + 128*2/8) = 8 × (2 + 32) = 8 × 34 = 272
        assert_eq!(bytes_per_kv_position(128, 8, KvQuantMode::Turbo2), 272);
    }

    #[test]
    fn test_rotation_matrix_is_orthogonal() {
        let dim = 64;
        let pi = generate_rotation_matrix(dim, 42);

        // Check Pi × Pi^T ≈ I.
        for i in 0..dim {
            for j in 0..dim {
                let dot: f32 = (0..dim)
                    .map(|k| pi[i * dim + k] * pi[j * dim + k])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "Pi*Pi^T[{},{}] = {} (expected {})",
                    i,
                    j,
                    dot,
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_rotation_matrix_deterministic() {
        let a = generate_rotation_matrix(32, 42);
        let b = generate_rotation_matrix(32, 42);
        assert_eq!(a, b, "Same seed must produce identical rotation matrices");
    }

    #[test]
    fn test_centroids_are_sorted() {
        for mode in [KvQuantMode::Turbo2, KvQuantMode::Turbo3, KvQuantMode::Turbo4] {
            let cfg = TurboQuantConfig::new(mode, 128);
            for w in cfg.centroids.windows(2) {
                assert!(w[0] < w[1], "Centroids must be sorted ascending");
            }
        }
    }

    #[test]
    fn test_centroids_symmetric() {
        // Max-Lloyd centroids for a symmetric distribution should be symmetric.
        for mode in [KvQuantMode::Turbo2, KvQuantMode::Turbo3, KvQuantMode::Turbo4] {
            let cfg = TurboQuantConfig::new(mode, 128);
            let n = cfg.centroids.len();
            for i in 0..n / 2 {
                let sum = cfg.centroids[i] + cfg.centroids[n - 1 - i];
                assert!(
                    sum.abs() < 1e-6,
                    "Centroids should be symmetric: c[{}]={}, c[{}]={}",
                    i,
                    cfg.centroids[i],
                    n - 1 - i,
                    cfg.centroids[n - 1 - i],
                );
            }
        }
    }

    /// CPU reference: quantize a vector using rotation + nearest centroid.
    /// Returns (norm, codes).
    fn cpu_quantize(x: &[f32], pi: &[f32], centroids: &[f32], hd: usize) -> (f32, Vec<usize>) {
        // L2 norm
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv_norm = 1.0 / norm.max(1e-6);

        // Rotate normalized vector: y = Pi * (x / norm)
        let mut codes = Vec::with_capacity(hd);
        for j in 0..hd {
            let mut y = 0.0f32;
            for i in 0..hd {
                y += pi[j * hd + i] * (x[i] * inv_norm);
            }
            // Nearest centroid
            let mut best = 0;
            let mut best_dist = f32::INFINITY;
            for (c, &cent) in centroids.iter().enumerate() {
                let d = (y - cent).abs();
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }
            codes.push(best);
        }
        (norm, codes)
    }

    /// CPU reference: compute Q·K score via rotated quantized path.
    fn cpu_turbo_score(
        q: &[f32], k: &[f32], pi: &[f32], centroids: &[f32], hd: usize,
    ) -> f32 {
        // Rotate Q
        let mut q_rot = vec![0.0f32; hd];
        for j in 0..hd {
            for i in 0..hd {
                q_rot[j] += pi[j * hd + i] * q[i];
            }
        }
        // Quantize K
        let (k_norm, k_codes) = cpu_quantize(k, pi, centroids, hd);
        // Score = q_rot · dequant(k) = sum q_rot[j] * centroid[code_j] * k_norm
        let mut score = 0.0f32;
        for j in 0..hd {
            score += q_rot[j] * centroids[k_codes[j]] * k_norm;
        }
        score
    }

    #[test]
    fn test_turbo_roundtrip_score_matches_dot_product() {
        let hd = 64;
        let pi = generate_rotation_matrix(hd, 42);
        let config = TurboQuantConfig::new(KvQuantMode::Turbo4, hd);

        // Create random-ish Q and K vectors
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(123);
        let q: Vec<f32> = (0..hd).map(|_| rng.random_normal()).collect();
        let k: Vec<f32> = (0..hd).map(|_| rng.random_normal()).collect();

        // Direct dot product (ground truth)
        let direct: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();

        // TurboQuant score (rotate, quantize K, dot product with rotated Q)
        let turbo = cpu_turbo_score(&q, &k, &pi, &config.centroids, hd);

        // Should be close (4-bit quantization introduces small error)
        let rel_error = ((turbo - direct) / direct.abs().max(1e-6)).abs();
        assert!(
            rel_error < 0.15,
            "TurboQuant score should be close to direct dot product: \
             direct={direct}, turbo={turbo}, rel_error={rel_error}"
        );
    }

    #[test]
    fn test_turbo_config_head_dim_scaling() {
        // Centroids should be smaller for larger head_dim (1/sqrt(d) scaling).
        let cfg64 = TurboQuantConfig::new(KvQuantMode::Turbo4, 64);
        let cfg128 = TurboQuantConfig::new(KvQuantMode::Turbo4, 128);

        let ratio = cfg64.centroids[0] / cfg128.centroids[0];
        let expected_ratio = (128.0f32 / 64.0).sqrt();
        assert!(
            (ratio - expected_ratio).abs() < 0.01,
            "Centroid ratio {} should be √2 ≈ {}",
            ratio,
            expected_ratio,
        );
    }
}
