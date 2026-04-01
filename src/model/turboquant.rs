// ===========================================================================
// TurboQuant — online KV cache vector quantization with QJL residual.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements a two-stage vector quantization pipeline for compressing the
//   KV cache during inference, combining PolarQuant (Zandieh et al.,
//   arXiv:2504.19874) with QJL residual quantization (Turney, 2025):
//
//   Stage 1 — PolarQuant ((b-1) bits):
//     Random orthogonal rotation makes coordinates approximately independent
//     and Gaussian, then Max-Lloyd scalar quantization per coordinate.
//
//   Stage 2 — QJL residual (1 bit):
//     The quantization residual's sign is stored as a single bit per
//     coordinate, along with the residual's L2 norm (gamma).  At decode
//     time: dequant[j] = centroid[code_j] + gamma * sqrt(π/2)/sqrt(d) * sign_j.
//     This halves the MSE vs stage-1 alone at the cost of 2 extra bytes
//     (bf16 gamma) + ceil(d/8) sign bytes per head per position.
//
//   Boundary layer protection:
//     First/last 2 layers use higher-precision quantization (turbo4 when
//     base is turbo2/3).  Recovers 37-91% of quality gap at aggressive
//     compression with zero speed penalty.
//
//   Sparse V dequantization:
//     Attention kernel skips V dequant when weight < 1e-6.  At long contexts,
//     most positions get negligible attention — saves significant compute.
//
// Why TurboQuant?
//   The KV cache is the primary memory bottleneck for long-context inference.
//   Each token adds num_kv_heads × head_dim × 2 bytes × 2 (K+V) × num_layers
//   to the cache.  For a 32-layer model with 8 KV heads × 128 head_dim, that
//   is 128 KB per token in BF16.  At 128K context, the KV cache alone is 16 GB.
//
// Attention-time efficiency:
//   The rotation can be folded into the attention computation:
//     - For K (inner products): rotate Q once, then <Pi*Q, dequant(Pi*K)> ≈ <Q,K>.
//       No inverse rotation needed per position — just centroid + sign lookup.
//     - For V (weighted sums): accumulate dequantized values in rotated space,
//       then apply Pi^T once per query head at the end.
//   Per-position cost is a centroid lookup + sign bit read (trivial),
//   while memory bandwidth drops ~3.8x (the decode bottleneck on Apple Silicon).
//
// Storage format:
//   Per KV head per position (two-stage: PolarQuant + QJL):
//     [2 bytes bf16 norm] [2 bytes bf16 gamma] [stage1 codes] [sign bits]
//   For 4-bit (3-bit stage1), head_dim=128: 2+2+48+16 = 68 bytes (3.8x compression)
//   For 3-bit (2-bit stage1), head_dim=128: 2+2+32+16 = 52 bytes (4.9x compression)
//   For 2-bit (1-bit stage1), head_dim=128: 2+2+16+16 = 36 bytes (7.1x compression)
//
// References:
//   PolarQuant: "TurboQuant: Online Vector Quantization with Near-optimal
//   Distortion Rate", Zandieh et al.  arXiv:2504.19874v1, Apr 2025.
//   QJL residual + boundary protection + sparse V: Turney, 2025.
//   https://github.com/TheTom/turboquant_plus
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
    /// 2-bit TurboQuant (1-bit PolarQuant + 1-bit QJL sign, ~7.1x compression).
    Turbo2,
    /// 3-bit TurboQuant (2-bit PolarQuant + 1-bit QJL sign, ~4.9x compression).
    Turbo3,
    /// 4-bit TurboQuant (3-bit PolarQuant + 1-bit QJL sign, ~3.8x compression).  Default.
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

    /// Total bits per coordinate for this mode.  Returns 0 for None (BF16).
    pub fn bits(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Turbo2 => 2,
            Self::Turbo3 => 3,
            Self::Turbo4 => 4,
        }
    }

    /// Bits used for stage-1 PolarQuant codebook.
    ///
    /// Bits used for stage-1 PolarQuant codebook.
    ///
    /// Turbo3/Turbo4 use the two-stage QJL pipeline: (bits-1) bits for
    /// PolarQuant + 1 bit for QJL sign.  Turbo2 uses all 2 bits for
    /// PolarQuant (4 centroids) without QJL — at only 2 total bits,
    /// the 4-centroid codebook outperforms the 2-centroid + sign split.
    pub fn stage1_bits(self) -> u32 {
        match self {
            Self::Turbo2 => 2, // all bits to PolarQuant (no QJL at 2-bit)
            Self::Turbo3 | Self::Turbo4 => self.bits() - 1,
            Self::None => 0,
        }
    }

    /// Whether this mode uses the QJL residual (two-stage pipeline).
    /// Turbo2 does not — at 2 bits, all bits go to PolarQuant.
    pub fn has_qjl(self) -> bool {
        matches!(self, Self::Turbo3 | Self::Turbo4)
    }

    /// Number of stage-1 codebook centroids: 2^stage1_bits.
    pub fn num_centroids(self) -> u32 {
        if self.is_quantized() { 1 << self.stage1_bits() } else { 0 }
    }

    /// Whether quantization is active.
    pub fn is_quantized(self) -> bool {
        self != Self::None
    }
}

// ---------------------------------------------------------------------------
// KV cache quantization pair — allows asymmetric K/V quantization modes.
//
// Models with QKV bias (Qwen2, GPT-OSS) exhibit correlated K quantization
// error that softmax amplifies.  V is tolerant because errors average out
// in weighted sums.  KvQuantPair allows K=BF16, V=TurboX ("asymmetric mode")
// while preserving the existing K=V=TurboX ("symmetric mode") path.
// ---------------------------------------------------------------------------

/// Paired K/V quantization modes, supporting asymmetric configurations.
///
/// Symmetric: both K and V use the same mode (the common case).
/// Asymmetric: K=BF16, V=TurboX (for models with QKV bias on Metal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvQuantPair {
    pub k: KvQuantMode,
    pub v: KvQuantMode,
}

impl KvQuantPair {
    /// Both K and V use the same quantization mode.
    pub fn symmetric(mode: KvQuantMode) -> Self {
        Self { k: mode, v: mode }
    }

    /// Different quantization modes for K and V.
    pub fn asymmetric(k: KvQuantMode, v: KvQuantMode) -> Self {
        Self { k, v }
    }

    /// Whether either K or V is quantized.
    pub fn is_any_quantized(self) -> bool {
        self.k.is_quantized() || self.v.is_quantized()
    }

    /// Whether K and V use different modes.
    pub fn is_asymmetric(self) -> bool {
        self.k != self.v
    }

    /// Whether K and V use the same mode.
    #[allow(dead_code)]
    pub fn is_symmetric(self) -> bool {
        self.k == self.v
    }

    /// Parse from CLI string.
    ///
    /// Accepted formats:
    ///   "turbo4"       → symmetric (K=Turbo4, V=Turbo4)
    ///   "none"         → symmetric (K=None, V=None)
    ///   "none:turbo4"  → asymmetric (K=None, V=Turbo4) — K:V format
    pub fn from_str(s: &str) -> Option<Self> {
        if let Some((k_str, v_str)) = s.split_once(':') {
            let k = KvQuantMode::from_str(k_str)?;
            let v = KvQuantMode::from_str(v_str)?;
            Some(Self { k, v })
        } else {
            let mode = KvQuantMode::from_str(s)?;
            Some(Self::symmetric(mode))
        }
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

/// 1-bit (2 centroids) Max-Lloyd codebook for N(0,1).
/// Optimal: ±E[|X|] where X ~ N(0,1), i.e. ±√(2/π) ≈ ±0.7979.
/// Used by Turbo2Plus (1-bit PolarQuant stage-1 + 1-bit QJL sign).
const CENTROIDS_1BIT: [f32; 2] = [-0.7979, 0.7979];

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
    /// Total bits per coordinate (2, 3, or 4).
    pub bits: u32,
    /// Stage-1 (PolarQuant) bits per coordinate.
    /// Same as `bits` for base modes; `bits - 1` for Plus modes.
    pub stage1_bits: u32,
    pub num_centroids: u32,
    /// Scaled Max-Lloyd centroids for the stage-1 codebook.
    /// Length = num_centroids.  Sorted ascending.
    pub centroids: Vec<f32>,
    /// Bytes per KV head per position (including norm, gamma, codes, signs).
    pub bytes_per_head_pos: usize,
    /// Head dimension (for validation and kernel dispatch).
    pub head_dim: usize,
    /// Whether this is a TurboQuant+ mode (two-stage with QJL residual).
    pub is_plus: bool,
}

impl TurboQuantConfig {
    /// Construct TurboQuant configuration for the given mode and head dimension.
    ///
    /// Centroids are scaled from the standard N(0,1) codebook to N(0, 1/√d)
    /// by dividing each centroid by √head_dim.
    ///
    /// The centroids correspond to the stage-1 PolarQuant codebook
    /// ((bits-1) bits), and bytes_per_head_pos includes space for the
    /// QJL gamma norm + sign bits.
    pub fn new(mode: KvQuantMode, head_dim: usize) -> Self {
        assert!(mode.is_quantized(), "TurboQuantConfig::new called with None mode");
        let bits = mode.bits();
        let stage1_bits = mode.stage1_bits();
        let num_centroids = mode.num_centroids();

        // Scale centroids from N(0,1) to N(0, 1/sqrt(d)).
        // Use the stage-1 codebook (fewer centroids than total bit budget).
        let scale = 1.0 / (head_dim as f32).sqrt();
        let base_centroids: &[f32] = match stage1_bits {
            1 => &CENTROIDS_1BIT,
            2 => &CENTROIDS_2BIT,
            3 => &CENTROIDS_3BIT,
            4 => &CENTROIDS_4BIT,
            _ => unreachable!("unsupported stage1_bits: {stage1_bits}"),
        };
        let centroids: Vec<f32> = base_centroids.iter().map(|&c| c * scale).collect();

        // Storage layout per head per position:
        //   With QJL:    [2 bf16 norm] [2 bf16 gamma] [stage1 codes] [sign bits]
        //   Without QJL: [2 bf16 norm] [codes]  (turbo2: all bits to PolarQuant)
        let is_plus = mode.has_qjl();
        let bytes_per_head_pos = if is_plus {
            let stage1_code_bytes = (head_dim * stage1_bits as usize + 7) / 8;
            let sign_bytes = (head_dim + 7) / 8;
            2 + 2 + stage1_code_bytes + sign_bytes
        } else {
            let code_bytes = (head_dim * bits as usize + 7) / 8;
            2 + code_bytes
        };

        Self {
            mode,
            bits,
            stage1_bits,
            num_centroids,
            centroids,
            bytes_per_head_pos,
            head_dim,
            is_plus,
        }
    }
}

/// Bytes per KV position across all KV heads for one pool (K or V).
///
/// For BF16: num_kv_heads × head_dim × 2.
/// For TurboQuant with QJL: num_kv_heads × (2 norm + 2 gamma + stage1_codes + sign_bits).
/// For TurboQuant without QJL (turbo2): num_kv_heads × (2 norm + codes).
pub(crate) fn bytes_per_kv_position(
    head_dim: usize,
    num_kv_heads: usize,
    mode: KvQuantMode,
) -> usize {
    if mode.is_quantized() {
        if mode.has_qjl() {
            let stage1_code_bytes = (head_dim * mode.stage1_bits() as usize + 7) / 8;
            let sign_bytes = (head_dim + 7) / 8;
            num_kv_heads * (2 + 2 + stage1_code_bytes + sign_bytes)
        } else {
            let code_bytes = (head_dim * mode.bits() as usize + 7) / 8;
            num_kv_heads * (2 + code_bytes)
        }
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
// Boundary layer protection — per-layer quantization mode override.
//
// TurboQuant+ discovery (Turney, 2025): K compression drives quality loss
// while V compression is nearly lossless.  Boundary layers (first/last N)
// are most sensitive, so protecting them at higher precision recovers
// 37-91% of the quality gap at turbo2/turbo3 with zero speed penalty.
//
// Implementation: the engine allocates boundary layers with larger KV pool
// buffers (more bytes_per_pos) and the attention kernel receives different
// bits/bytes_per_head_pos/centroids for those layers.
// ---------------------------------------------------------------------------

/// Boundary layer protection configuration.
///
/// When active, the first `first_n` and last `last_n` transformer layers
/// use `boundary_mode` instead of the base quantization mode.  This is
/// most effective for aggressive compression (turbo2, turbo3) where
/// boundary layers disproportionately affect quality.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoundaryConfig {
    /// Number of initial layers to protect (typically 2).
    pub first_n: usize,
    /// Number of final layers to protect (typically 2).
    pub last_n: usize,
    /// Quantization mode for boundary layers (e.g., Turbo4 when base is Turbo2).
    pub boundary_mode: KvQuantMode,
}

impl BoundaryConfig {
    /// Default boundary protection for a given base mode.
    ///
    /// Returns Some for aggressive modes (Turbo2, Turbo3) where boundary
    /// protection significantly improves quality.  Returns None for Turbo4
    /// (already quality-neutral) and None (BF16).
    pub fn default_for(mode: KvQuantMode) -> Option<Self> {
        match mode {
            KvQuantMode::Turbo2 | KvQuantMode::Turbo3 => Some(Self {
                first_n: 2,
                last_n: 2,
                boundary_mode: KvQuantMode::Turbo4,
            }),
            _ => None,
        }
    }

    /// Whether a layer index is a boundary layer.
    pub fn is_boundary_layer(&self, layer_idx: usize, num_layers: usize) -> bool {
        layer_idx < self.first_n || layer_idx >= num_layers.saturating_sub(self.last_n)
    }
}

/// Compute the effective KV quantization pair for a specific layer.
///
/// Interior layers use the base pair.  Boundary layers (first/last N) use
/// the boundary mode for whichever of K/V is quantized in the base pair.
pub(crate) fn effective_kv_pair_for_layer(
    base: KvQuantPair,
    boundary: Option<&BoundaryConfig>,
    layer_idx: usize,
    num_layers: usize,
) -> KvQuantPair {
    match boundary {
        Some(bc) if bc.is_boundary_layer(layer_idx, num_layers) => {
            // Replace quantized modes with the boundary mode.
            KvQuantPair {
                k: if base.k.is_quantized() { bc.boundary_mode } else { base.k },
                v: if base.v.is_quantized() { bc.boundary_mode } else { base.v },
            }
        }
        _ => base,
    }
}

// ---------------------------------------------------------------------------
// TurboContext — GPU-resident state for TurboQuant, bundled for forward pass.
//
// Created once at model load and passed by reference to every attention call.
// Holds the rotation matrix, its transpose, the codebook centroids, and a
// scratch buffer for the rotated query vector.
//
// When boundary layer protection is active, stores a second set of centroids
// and config for boundary layers (which may use a different bit width).
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
    /// K/V quantization pair — needed by primitives to choose the right
    /// attention path (symmetric turbo, asymmetric V-only, or BF16).
    pub kv_pair: KvQuantPair,

    // Boundary layer protection (TurboQuant+).
    /// Boundary config, if boundary layer protection is active.
    pub boundary: Option<BoundaryConfig>,
    /// Centroids for boundary layers (different bit width than base).
    /// None when boundary is None or boundary mode == base mode.
    pub boundary_centroids: Option<B::Tensor>,
    /// Config for boundary layers.
    pub boundary_config: Option<TurboQuantConfig>,
}

impl<B: GpuCore> TurboContext<B> {
    /// Create and upload TurboQuant context to GPU.
    ///
    /// The rotation matrix is generated from a fixed seed (42) for
    /// reproducibility and prefix cache compatibility.
    ///
    /// `kv_pair` stores the full K/V quantization pair so primitives can
    /// choose the right attention path.  The V mode determines the config
    /// (bits, codebook, bytes_per_head_pos) since only V is guaranteed to
    /// be quantized when this context exists.
    ///
    /// `boundary` enables boundary layer protection: first/last N layers
    /// use a higher-precision mode.  When the boundary mode differs from
    /// the base mode, a separate set of centroids is uploaded for those layers.
    pub fn new(
        backend: &B,
        kv_pair: KvQuantPair,
        head_dim: usize,
        num_heads: usize,
        boundary: Option<BoundaryConfig>,
    ) -> Self {
        // V mode drives the TurboQuant config — it's always quantized when
        // a TurboContext is created.
        let mode = kv_pair.v;
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

        // Upload boundary centroids if boundary mode differs from base.
        let (boundary_centroids, boundary_config) = match boundary {
            Some(bc) if bc.boundary_mode != mode => {
                let bc_config = TurboQuantConfig::new(bc.boundary_mode, head_dim);
                let bc_bytes: &[u8] = bytemuck::cast_slice(&bc_config.centroids);
                let bc_centroids = backend.upload_tensor(
                    bc_bytes,
                    &[bc_config.num_centroids as usize],
                    TensorDtype::F32,
                );
                (Some(bc_centroids), Some(bc_config))
            }
            _ => (None, None),
        };

        Self {
            pi,
            pi_t,
            centroids,
            q_rot_buf,
            config,
            kv_pair,
            boundary,
            boundary_centroids,
            boundary_config,
        }
    }

    /// Get the config and centroids for a specific layer.
    ///
    /// Returns the boundary config/centroids for boundary layers,
    /// or the base config/centroids for interior layers.
    pub fn config_for_layer(
        &self, layer_idx: usize, num_layers: usize,
    ) -> (&TurboQuantConfig, &B::Tensor) {
        if let (Some(bc), Some(bc_config), Some(bc_centroids)) =
            (&self.boundary, &self.boundary_config, &self.boundary_centroids)
        {
            if bc.is_boundary_layer(layer_idx, num_layers) {
                return (bc_config, bc_centroids);
            }
        }
        (&self.config, &self.centroids)
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
    fn test_kv_quant_pair_parsing() {
        // Symmetric: single mode string.
        let pair = KvQuantPair::from_str("turbo4").unwrap();
        assert_eq!(pair.k, KvQuantMode::Turbo4);
        assert_eq!(pair.v, KvQuantMode::Turbo4);
        assert!(!pair.is_asymmetric());
        assert!(pair.is_any_quantized());

        // Asymmetric: "K:V" format.
        let pair = KvQuantPair::from_str("none:turbo4").unwrap();
        assert_eq!(pair.k, KvQuantMode::None);
        assert_eq!(pair.v, KvQuantMode::Turbo4);
        assert!(pair.is_asymmetric());
        assert!(pair.is_any_quantized());

        // Both none.
        let pair = KvQuantPair::from_str("none").unwrap();
        assert!(!pair.is_any_quantized());

        // Invalid.
        assert!(KvQuantPair::from_str("invalid").is_none());
        assert!(KvQuantPair::from_str("none:invalid").is_none());
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
        // 4-bit (3-bit stage1 + 1-bit QJL):
        // 8 heads × (2 norm + 2 gamma + ceil(128*3/8) codes + ceil(128/8) signs)
        // = 8 × (2 + 2 + 48 + 16) = 8 × 68 = 544
        assert_eq!(bytes_per_kv_position(128, 8, KvQuantMode::Turbo4), 544);
    }

    #[test]
    fn test_bytes_per_kv_position_turbo3() {
        // 3-bit (2-bit stage1 + 1-bit QJL):
        // 8 heads × (2 + 2 + ceil(128*2/8) + ceil(128/8))
        // = 8 × (2 + 2 + 32 + 16) = 8 × 52 = 416
        assert_eq!(bytes_per_kv_position(128, 8, KvQuantMode::Turbo3), 416);
    }

    #[test]
    fn test_bytes_per_kv_position_turbo2() {
        // 2-bit PolarQuant (no QJL): 8 heads × (2 + 128*2/8) = 8 × 34 = 272
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

    /// CPU reference: compute Q·K score via the two-stage TurboQuant pipeline.
    ///
    /// Stage 1: PolarQuant — rotate + nearest centroid at (bits-1) bits.
    /// Stage 2 (if has_qjl): QJL — 1-bit sign of the residual + gamma norm.
    /// Dequant: centroid[code] + gamma * sqrt(π/2) / sqrt(hd) * sign.
    /// Without QJL (turbo2): dequant = centroid[code].
    fn cpu_turbo_score(
        q: &[f32], k: &[f32], pi: &[f32], centroids: &[f32], hd: usize,
        has_qjl: bool,
    ) -> f32 {
        // Rotate Q.
        let mut q_rot = vec![0.0f32; hd];
        for j in 0..hd {
            for i in 0..hd {
                q_rot[j] += pi[j * hd + i] * q[i];
            }
        }

        // Quantize K: rotate + stage-1 centroid.
        let k_norm: f32 = k.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv_norm = 1.0 / k_norm.max(1e-6);

        let mut codes = Vec::with_capacity(hd);
        let mut rotated_k = vec![0.0f32; hd];
        for j in 0..hd {
            let mut y = 0.0f32;
            for i in 0..hd {
                y += pi[j * hd + i] * (k[i] * inv_norm);
            }
            rotated_k[j] = y;
            let mut best = 0;
            let mut best_dist = f32::INFINITY;
            for (c, &cent) in centroids.iter().enumerate() {
                let d = (y - cent).abs();
                if d < best_dist { best_dist = d; best = c; }
            }
            codes.push(best);
        }

        if has_qjl {
            // QJL residual: sign + gamma.
            let mut residuals = vec![0.0f32; hd];
            for j in 0..hd {
                residuals[j] = rotated_k[j] - centroids[codes[j]];
            }
            let gamma: f32 = residuals.iter().map(|r| r * r).sum::<f32>().sqrt();
            let qjl_scale = (std::f32::consts::PI / 2.0).sqrt() / (hd as f32).sqrt();

            let mut score = 0.0f32;
            for j in 0..hd {
                let sign = if residuals[j] >= 0.0 { 1.0f32 } else { -1.0f32 };
                let dequant = centroids[codes[j]] + gamma * qjl_scale * sign;
                score += q_rot[j] * dequant * k_norm;
            }
            score
        } else {
            // No QJL: simple centroid dequant.
            let mut score = 0.0f32;
            for j in 0..hd {
                score += q_rot[j] * centroids[codes[j]] * k_norm;
            }
            score
        }
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

        // TurboQuant score (3-bit PolarQuant + 1-bit QJL sign)
        let turbo = cpu_turbo_score(&q, &k, &pi, &config.centroids, hd, config.is_plus);

        // Should be close (quantization introduces small error)
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

    // -----------------------------------------------------------------------
    // Boundary layer protection tests.
    // -----------------------------------------------------------------------

    #[test]
    fn test_boundary_config_default_for() {
        // Turbo2 and Turbo3 get boundary protection; Turbo4 and None do not.
        assert!(BoundaryConfig::default_for(KvQuantMode::Turbo2).is_some());
        assert!(BoundaryConfig::default_for(KvQuantMode::Turbo3).is_some());
        assert!(BoundaryConfig::default_for(KvQuantMode::Turbo4).is_none());
        assert!(BoundaryConfig::default_for(KvQuantMode::None).is_none());

        let bc = BoundaryConfig::default_for(KvQuantMode::Turbo2).unwrap();
        assert_eq!(bc.first_n, 2);
        assert_eq!(bc.last_n, 2);
        assert_eq!(bc.boundary_mode, KvQuantMode::Turbo4);
    }

    #[test]
    fn test_boundary_is_boundary_layer() {
        let bc = BoundaryConfig { first_n: 2, last_n: 2, boundary_mode: KvQuantMode::Turbo4 };
        let num_layers = 32;

        // First 2 layers are boundary.
        assert!(bc.is_boundary_layer(0, num_layers));
        assert!(bc.is_boundary_layer(1, num_layers));
        assert!(!bc.is_boundary_layer(2, num_layers));

        // Last 2 layers are boundary.
        assert!(!bc.is_boundary_layer(29, num_layers));
        assert!(bc.is_boundary_layer(30, num_layers));
        assert!(bc.is_boundary_layer(31, num_layers));

        // Interior layers are not.
        assert!(!bc.is_boundary_layer(15, num_layers));
    }

    #[test]
    fn test_boundary_small_model() {
        // Model with only 4 layers: all layers are boundary.
        let bc = BoundaryConfig { first_n: 2, last_n: 2, boundary_mode: KvQuantMode::Turbo4 };
        for i in 0..4 {
            assert!(
                bc.is_boundary_layer(i, 4),
                "layer {i} should be boundary in a 4-layer model",
            );
        }
    }

    #[test]
    fn test_effective_kv_pair_for_layer() {
        let base = KvQuantPair::symmetric(KvQuantMode::Turbo2);
        let bc = BoundaryConfig { first_n: 2, last_n: 2, boundary_mode: KvQuantMode::Turbo4 };
        let num_layers = 32;

        // Boundary layer: upgraded to Turbo4.
        let pair = effective_kv_pair_for_layer(base, Some(&bc), 0, num_layers);
        assert_eq!(pair.k, KvQuantMode::Turbo4);
        assert_eq!(pair.v, KvQuantMode::Turbo4);

        // Interior layer: stays at Turbo2.
        let pair = effective_kv_pair_for_layer(base, Some(&bc), 15, num_layers);
        assert_eq!(pair.k, KvQuantMode::Turbo2);
        assert_eq!(pair.v, KvQuantMode::Turbo2);

        // No boundary config: always base.
        let pair = effective_kv_pair_for_layer(base, None, 0, num_layers);
        assert_eq!(pair.k, KvQuantMode::Turbo2);
    }

    #[test]
    fn test_effective_kv_pair_asymmetric_with_boundary() {
        // Asymmetric K=None, V=Turbo2.  Boundary should only upgrade V.
        let base = KvQuantPair::asymmetric(KvQuantMode::None, KvQuantMode::Turbo2);
        let bc = BoundaryConfig { first_n: 2, last_n: 2, boundary_mode: KvQuantMode::Turbo4 };

        let pair = effective_kv_pair_for_layer(base, Some(&bc), 0, 32);
        assert_eq!(pair.k, KvQuantMode::None); // K stays BF16
        assert_eq!(pair.v, KvQuantMode::Turbo4); // V upgraded
    }

    // -----------------------------------------------------------------------
    // QJL two-stage pipeline tests.
    // -----------------------------------------------------------------------

    #[test]
    fn test_kv_quant_mode_stage1_bits() {
        // Turbo3/4: (bits-1) for stage-1 PolarQuant + 1-bit QJL.
        // Turbo2: all bits to PolarQuant (no QJL split).
        assert_eq!(KvQuantMode::Turbo4.stage1_bits(), 3);
        assert_eq!(KvQuantMode::Turbo3.stage1_bits(), 2);
        assert_eq!(KvQuantMode::Turbo2.stage1_bits(), 2); // no QJL
        assert_eq!(KvQuantMode::None.stage1_bits(), 0);
    }

    #[test]
    fn test_kv_quant_mode_num_centroids() {
        // Centroids = 2^stage1_bits.
        assert_eq!(KvQuantMode::Turbo4.num_centroids(), 8);  // 2^3
        assert_eq!(KvQuantMode::Turbo3.num_centroids(), 4);  // 2^2
        assert_eq!(KvQuantMode::Turbo2.num_centroids(), 4);  // 2^2, no QJL
        assert_eq!(KvQuantMode::None.num_centroids(), 0);
    }

    #[test]
    fn test_kv_quant_mode_has_qjl() {
        assert!(KvQuantMode::Turbo4.has_qjl());
        assert!(KvQuantMode::Turbo3.has_qjl());
        assert!(!KvQuantMode::Turbo2.has_qjl());
        assert!(!KvQuantMode::None.has_qjl());
    }

    #[test]
    fn test_turbo_config_qjl() {
        // Turbo3 uses 2-bit stage-1 (4 centroids) + 1-bit QJL sign.
        let cfg = TurboQuantConfig::new(KvQuantMode::Turbo3, 128);
        assert_eq!(cfg.centroids.len(), 4);
        assert_eq!(cfg.num_centroids, 4);
        assert_eq!(cfg.stage1_bits, 2);
        assert_eq!(cfg.bits, 3);
        assert!(cfg.is_plus);

        // Turbo4 uses 3-bit stage-1 (8 centroids) + 1-bit QJL sign.
        let cfg4 = TurboQuantConfig::new(KvQuantMode::Turbo4, 128);
        assert_eq!(cfg4.centroids.len(), 8);
        assert_eq!(cfg4.stage1_bits, 3);
        assert!(cfg4.is_plus);

        // Turbo2 uses 2-bit PolarQuant (4 centroids), no QJL.
        let cfg2 = TurboQuantConfig::new(KvQuantMode::Turbo2, 128);
        assert_eq!(cfg2.centroids.len(), 4);
        assert_eq!(cfg2.stage1_bits, 2);
        assert!(!cfg2.is_plus);
    }

    #[test]
    fn test_turbo_score_all_modes() {
        // Verify QJL roundtrip for all quantized modes.
        let hd = 64;
        let pi = generate_rotation_matrix(hd, 42);

        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(123);
        let q: Vec<f32> = (0..hd).map(|_| rng.random_normal()).collect();
        let k: Vec<f32> = (0..hd).map(|_| rng.random_normal()).collect();
        let direct: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();

        for mode in [KvQuantMode::Turbo2, KvQuantMode::Turbo3, KvQuantMode::Turbo4] {
            let config = TurboQuantConfig::new(mode, hd);
            let turbo = cpu_turbo_score(&q, &k, &pi, &config.centroids, hd, config.is_plus);
            let rel_error = ((turbo - direct) / direct.abs().max(1e-6)).abs();
            // Turbo2 has higher error at small head_dim (only 4 centroids, no QJL).
            let tolerance = if mode == KvQuantMode::Turbo2 { 0.40 } else { 0.30 };
            assert!(
                rel_error < tolerance,
                "{:?} score error too large: {rel_error} (direct={direct}, turbo={turbo})",
                mode,
            );
        }
    }
}
