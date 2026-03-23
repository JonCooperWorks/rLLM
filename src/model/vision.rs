// ===========================================================================
// Vision encoder — SigLIP ViT forward pass and image preprocessing.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the vision encoder for multimodal (VLM) models.  Converts raw
//   images into embedding vectors that slot into the language model's token
//   sequence — letting the LLM "see" images alongside text.
//
// How vision models work (the big picture):
//
//   A vision-language model has TWO encoders:
//     1. Vision encoder (this file): image → vision tokens  [~0.6B params]
//     2. Language model (model/registry/): tokens → text     [~27B params]
//
//   The trick: the vision encoder converts an image into a SEQUENCE of
//   embedding vectors that look just like text token embeddings.  These are
//   injected into the text sequence at placeholder positions, so from the
//   LLM's perspective, the image IS a sequence of tokens.
//
//   Example token sequence for "What's in this image?":
//     [<|im_start|>, user, \n, <|vision_start|>,
//      <|image_pad|>, <|image_pad|>, ..., <|image_pad|>,  ← 98 placeholders
//      <|vision_end|>, \n, What, 's, in, this, image, ?,
//      <|im_end|>, \n, <|im_start|>, assistant, \n]
//
//   During prefill, the embed_lookup fills all tokens with text embeddings.
//   Then scatter_vision_tokens OVERWRITES the <|image_pad|> positions with
//   the vision encoder's output.  The LLM sees a mix of text and image
//   embeddings and generates text that describes the image.
//
// Vision encoder architecture (SigLIP ViT):
//
//   Both Qwen 3.5 and Gemma 3 VLMs use a SigLIP-based Vision Transformer:
//     - 27 transformer blocks, 1152 hidden dim, 16 attention heads
//     - LayerNorm (NOT RMSNorm — vision encoders use full LayerNorm with bias)
//     - Plain GELU activation (NOT GeGLU or SwiGLU)
//     - Bidirectional attention (no causal mask — every patch sees every other)
//
//   Pipeline:
//     Image [3, H, W]
//       ↓ CPU: decode, resize, normalise, reshape into patches
//     Patches [num_patches, 768]     (768 = 3 channels × 16² pixels per patch)
//       ↓ GPU: linear projection (matmul with patch_embed_weight)
//     Patch embeddings [num_patches, 1152]
//       ↓ GPU: add learned positional embeddings
//     Positioned embeddings [num_patches, 1152]
//       ↓ GPU: 27× ViT blocks (LayerNorm → MHA → residual → LayerNorm → FFN → residual)
//     Vision features [num_patches, 1152]
//       ↓ GPU: merger/projector (spatial merge + MLP to map to LLM hidden size)
//     Vision tokens [num_output_tokens, 5120]
//       ↓ GPU: scatter into text embedding buffer at <|image_pad|> positions
//
// Key differences between Qwen 3.5 and Gemma 3:
//
//   | Aspect              | Qwen 3.5                    | Gemma 3              |
//   |---------------------|-----------------------------|----------------------|
//   | Patch size          | 16×16                       | 14×14                |
//   | QKV weights         | Fused [3*hd, hd] in ckpt    | Separate Q/K/V       |
//   | Spatial merge       | 2×2 (4× token reduction)    | None (fixed 256 tok) |
//   | Projector           | 2-layer MLP with LayerNorm  | Single linear layer  |
//   | Temporal patch      | Size 2 (for video support)  | None                 |
//   | Weight prefix       | model.visual.*              | vision_tower.*       |
//
// Why ViT uses LayerNorm instead of RMSNorm:
//   RMSNorm drops the mean-centering step for efficiency.  LLMs adopted it
//   because the performance difference is negligible at scale.  But the
//   SigLIP vision encoder was trained with standard LayerNorm, and switching
//   would change the learned representations.  Since the vision encoder is
//   frozen (pre-trained), we must match its original normalisation exactly.
//
// Why vision attention is bidirectional:
//   In text generation, each token should only attend to past tokens (causal
//   mask) to preserve the autoregressive property.  But an image has no
//   natural ordering — patch (3,5) is just as relevant to patch (1,2) as
//   the reverse.  So vision attention uses no mask: every patch attends to
//   every other patch (full self-attention).
//
// Related files:
//   config.rs             — VisionConfig struct (parsed from config.json)
//   loader.rs             — load_vision_weights() (loads from safetensors)
//   mod.rs                — forward_prefill_paged() (calls vision_encode + scatter)
//   gpu/ops/norm.rs       — layer_norm_batch (LayerNorm kernel for ViT)
//   gpu/ops/vision.rs     — spatial_merge, scatter_vision_tokens
//   gpu/ops/attention.rs  — prefill_attention with causal=false for bidirectional
//   gpu/ops/elementwise.rs — gelu() for vision FFN activation
//   chat.rs               — vision placeholder tokens in chat templates
// ===========================================================================

use half::bf16;

use super::config::VisionConfig;
use crate::gpu::{GpuBackend, GpuCore, GpuElementwise, GpuMatmul, GpuNorm, GpuVision, TensorDtype};

// ---------------------------------------------------------------------------
// Vision weight structures.
//
// Design decision: Q/K/V weights are always stored as SEPARATE tensors,
// even for Qwen 3.5 which ships fused QKV [3*hidden, hidden] in the
// checkpoint.  The fused tensor is split during loading (see loader.rs).
// This keeps the forward pass simple — no sub-tensor slicing needed.
// Same pattern as how Phi's fused qkv_proj is split on load elsewhere.
// ---------------------------------------------------------------------------

/// Weights for a single ViT transformer block.
///
/// Each block is a standard pre-norm transformer:
///   LayerNorm → Multi-Head Attention → residual → LayerNorm → FFN → residual
///
/// All projections include bias terms (unlike LLM layers which often omit
/// bias).  This is a SigLIP design choice — the vision encoder was trained
/// with biases and we must match.
pub(crate) struct VisionBlockWeights<B: GpuCore> {
    // --- Pre-attention LayerNorm ---
    pub norm1_weight: B::Tensor, // [hidden_size] — scale parameter γ
    pub norm1_bias: B::Tensor,   // [hidden_size] — shift parameter β

    // --- Pre-FFN LayerNorm ---
    pub norm2_weight: B::Tensor, // [hidden_size]
    pub norm2_bias: B::Tensor,   // [hidden_size]

    // --- Multi-head self-attention ---
    // Q/K/V are separate tensors (split on load for Qwen's fused checkpoint).
    // All are [hidden_size, hidden_size] with bias [hidden_size].
    pub q_weight: B::Tensor,
    pub k_weight: B::Tensor,
    pub v_weight: B::Tensor,
    pub q_bias: Option<B::Tensor>,
    pub k_bias: Option<B::Tensor>,
    pub v_bias: Option<B::Tensor>,
    pub proj_weight: B::Tensor, // output projection [hidden_size, hidden_size]
    pub proj_bias: B::Tensor,

    // --- Feed-forward network ---
    //
    // Both Qwen 3.5 and Gemma 3 vision encoders use plain GELU activation
    // (NOT GeGLU/SwiGLU).  The FFN is: fc1 → GELU → fc2.
    //
    // The up_weight field exists for future GeGLU support but is currently
    // always None for vision encoders.
    pub fc1_weight: B::Tensor,           // [intermediate_size, hidden_size]
    pub fc1_bias: B::Tensor,             // [intermediate_size]
    pub up_weight: Option<B::Tensor>,    // None for plain GELU
    pub up_bias: Option<B::Tensor>,      // None for plain GELU
    pub fc2_weight: B::Tensor,           // [hidden_size, intermediate_size]
    pub fc2_bias: B::Tensor,             // [hidden_size]
}

/// All weights for the vision encoder + merger/projector.
///
/// The vision encoder is ~0.6B parameters (27 blocks × 1152 hidden dim).
/// The merger/projector maps vision hidden states to the LLM's hidden dimension.
pub(crate) struct VisionWeights<B: GpuCore> {
    /// Patch embedding projection: [hidden_size, patch_dim].
    ///
    /// This is the FIRST operation in the vision encoder.  It's a linear
    /// projection that maps each flattened image patch to the vision hidden
    /// dimension.  Equivalent to a Conv2D with kernel_size == stride == patch_size.
    ///
    /// patch_dim = in_channels × patch_size² = 3 × 16² = 768 (Qwen 3.5)
    ///
    /// For Qwen 3.5, the checkpoint stores this as a Conv3D weight
    /// [out, in, temporal, kH, kW] with temporal_patch_size=2 (for video).
    /// For single-image inference, the temporal dimension is averaged during
    /// loading so this becomes [hidden_size, in_channels × kH × kW].
    pub patch_embed_weight: B::Tensor,
    pub patch_embed_bias: Option<B::Tensor>,

    /// Learned positional embeddings: [num_positions, hidden_size].
    ///
    /// Unlike RoPE (used by the LLM), ViT uses absolute learned positional
    /// embeddings.  Each patch position in the grid has a unique learned
    /// embedding vector that encodes "where in the image this patch is."
    pub pos_embed: B::Tensor,

    /// Per-block (layer) weights — one per ViT transformer block.
    pub blocks: Vec<VisionBlockWeights<B>>,

    /// Post-LayerNorm (Gemma 3 only — applied after all ViT blocks).
    pub post_norm_weight: Option<B::Tensor>,
    pub post_norm_bias: Option<B::Tensor>,

    // --- Merger / projector ---
    //
    // The merger bridges the vision encoder and language model by projecting
    // vision hidden states (1152-dim) to the LLM's hidden dimension (5120).
    //
    // Qwen 3.5: spatial merge (2×2 → 4× fewer tokens) → LayerNorm → 2-layer MLP
    // Gemma 3:  single linear projection (no spatial merge)
    pub merger_norm_weight: Option<B::Tensor>, // LayerNorm before MLP (Qwen only)
    pub merger_norm_bias: Option<B::Tensor>,
    pub merger_fc1_weight: B::Tensor,          // first projection layer
    pub merger_fc1_bias: B::Tensor,
    pub merger_fc2_weight: Option<B::Tensor>,  // second projection layer (Qwen only)
    pub merger_fc2_bias: Option<B::Tensor>,
}

// ---------------------------------------------------------------------------
// Preprocessed image — CPU-processed pixel data ready for GPU upload.
// ---------------------------------------------------------------------------

/// A preprocessed image ready for GPU upload and vision encoding.
///
/// Image preprocessing runs on the CPU (handler thread for serve, main thread
/// for CLI).  The result is a flat bf16 tensor that gets uploaded to GPU
/// memory as the first step of vision_encode().
pub(crate) struct ProcessedImage {
    /// Pixel data as raw bytes in bf16 format.
    ///
    /// Layout: [num_patches, patch_dim] row-major, where
    ///   num_patches = grid_h × grid_w
    ///   patch_dim = in_channels × patch_size²
    ///
    /// Within each patch, pixels are channel-first (CHW order):
    ///   [R₀₀, R₀₁, ..., R₁₅₁₅, G₀₀, G₀₁, ..., B₁₅₁₅]
    pub pixels: Vec<u8>,
    /// Number of patches along the height axis.
    pub grid_h: usize,
    /// Number of patches along the width axis.
    pub grid_w: usize,
    /// Total number of output vision tokens after spatial merge.
    ///
    /// For Qwen 3.5 (merge_size=2): (grid_h/2) × (grid_w/2)
    /// For Gemma 3 (no merge): grid_h × grid_w
    ///
    /// This must match the number of <|image_pad|> tokens in the chat template.
    pub num_vision_tokens: usize,
}

// ---------------------------------------------------------------------------
// Image preprocessing (CPU-side).
//
// Preprocessing converts a raw image file (JPEG/PNG/WebP) into the normalised
// patch tensor that the vision encoder expects.  This runs on the CPU because
// it involves image decoding and pixel-level manipulation — not worth a GPU
// kernel for a one-time operation.
//
// The preprocessing pipeline matches HuggingFace's Qwen2VLImageProcessor:
//   1. Decode the image from compressed bytes
//   2. Resize to fit within max_dim while maintaining aspect ratio
//   3. Snap dimensions to multiples of (patch_size × merge_size)
//   4. Normalise each channel with CLIP statistics (mean/std)
//   5. Rearrange into [num_patches, channels × patch_size²] layout
//
// CLIP normalisation (same constants used by SigLIP, CLIP, and most ViTs):
//   normalised = (pixel / 255.0 - mean) / std
//   mean = [0.48145466, 0.4578275, 0.40821073]  (per-channel)
//   std  = [0.26862954, 0.26130258, 0.27577711]
//
// These constants come from the ImageNet training distribution and are baked
// into the vision encoder's weights.  Using wrong normalisation would be like
// feeding the LLM tokens from a different vocabulary.
// ---------------------------------------------------------------------------

/// CLIP image normalisation constants (ImageNet statistics).
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Preprocess an image for the vision encoder.
///
/// Converts raw image bytes (JPEG/PNG/WebP) into a normalised patch tensor.
/// Returns a `ProcessedImage` containing the bf16 pixel data, grid dimensions,
/// and the number of vision tokens the encoder will produce.
pub(crate) fn preprocess_image(
    image_bytes: &[u8],
    config: &VisionConfig,
) -> anyhow::Result<ProcessedImage> {
    // Decode the compressed image into an in-memory bitmap.
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode image: {e}"))?;

    let ps = config.patch_size;
    let ms = config.spatial_merge_size.max(1);

    // Determine target resolution.
    //
    // The target dimensions must be multiples of (patch_size × merge_size) so
    // that the image divides evenly into patches and the spatial merge can
    // group patches into complete merge_size × merge_size blocks.
    //
    // Qwen 3.5: max 448×448 (with merge_size=2, factor=32)
    // Gemma 3:  max 896×896 (no merge, factor=14)
    let factor = ps * ms;
    let max_dim: u32 = if config.spatial_merge_size > 0 { 448 } else { 896 };
    let (orig_w, orig_h) = (img.width(), img.height());
    let scale = (max_dim as f32 / orig_w.max(orig_h) as f32).min(1.0);
    let target_w = ((orig_w as f32 * scale) as usize / factor * factor).max(factor);
    let target_h = ((orig_h as f32 * scale) as usize / factor * factor).max(factor);

    // Resize with Lanczos3 (high-quality downsampling).
    let resized = img.resize_exact(
        target_w as u32,
        target_h as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let rgb = resized.to_rgb8();

    let grid_h = target_h / ps;
    let grid_w = target_w / ps;
    let num_patches = grid_h * grid_w;
    let patch_dim = config.in_channels * ps * ps;

    // Convert to normalised bf16 patches.
    //
    // Layout: [num_patches, patch_dim] where patches are in row-major grid order,
    // and within each patch, pixels are channel-first (CHW):
    //   patch = [R_00, R_01, ..., R_15_15, G_00, ..., G_15_15, B_00, ..., B_15_15]
    //
    // This matches ViT's expected input format — the patch embedding weight
    // is [hidden_size, patch_dim] so each patch is projected via matmul.
    let mut pixels = vec![bf16::ZERO; num_patches * patch_dim];
    for py in 0..grid_h {
        for px in 0..grid_w {
            let patch_idx = py * grid_w + px;
            for c in 0..config.in_channels {
                for dy in 0..ps {
                    for dx in 0..ps {
                        let pixel = rgb.get_pixel((px * ps + dx) as u32, (py * ps + dy) as u32);
                        let raw = pixel[c] as f32 / 255.0;
                        let normalised = (raw - CLIP_MEAN[c]) / CLIP_STD[c];
                        pixels[patch_idx * patch_dim + c * ps * ps + dy * ps + dx] =
                            bf16::from_f32(normalised);
                    }
                }
            }
        }
    }

    // Compute output token count.
    //
    // Qwen spatial merge (2×2): groups of 4 adjacent patches → 1 token.
    //   A 28×14 patch grid (392 patches) becomes 14×7 = 98 vision tokens.
    // Gemma (no merge): each patch is one token (typically 256).
    let num_vision_tokens = if config.spatial_merge_size > 0 {
        (grid_h / ms) * (grid_w / ms)
    } else {
        num_patches
    };

    Ok(ProcessedImage {
        pixels: bytemuck::cast_slice(&pixels).to_vec(),
        grid_h,
        grid_w,
        num_vision_tokens,
    })
}

// ---------------------------------------------------------------------------
// Vision encoder forward pass.
//
// This is the GPU-side computation that transforms preprocessed image patches
// into embedding vectors compatible with the language model.
//
// Buffer reuse:
//   Like the LLM forward pass, vision buffers are pre-allocated once and
//   reused for each image.  This avoids per-image GPU memory allocation.
//   The buffers are sized for the maximum expected patch count (28×28 = 784
//   patches for a 448×448 image with patch_size=16).
//
// Performance note:
//   Vision encoding runs ONCE per image during prefill.  The 27-block ViT
//   with 1152 hidden dim is ~0.6B parameters — tiny compared to the 27B
//   language model.  The dominant cost is the LLM prefill, not vision.
// ---------------------------------------------------------------------------

/// Scratch buffers for the vision encoder forward pass.
///
/// Allocated once during model loading (in engine/loader.rs) and reused
/// for each image.  Follows the same pattern as the LLM's scratch buffers
/// in Model (hidden, norm_buf, q_buf, etc.).
pub(crate) struct VisionBuffers<B: GpuCore> {
    pub hidden: B::Tensor,    // [max_patches, hidden_size] — residual stream
    pub norm_out: B::Tensor,  // [max_patches, hidden_size] — LayerNorm output
    pub q_buf: B::Tensor,     // [max_patches, hidden_size] — query projection
    pub k_buf: B::Tensor,     // [max_patches, hidden_size] — key projection
    pub v_buf: B::Tensor,     // [max_patches, hidden_size] — value projection
    pub attn_out: B::Tensor,  // [max_patches, hidden_size] — attention output
    pub ffn_gate: B::Tensor,  // [max_patches, intermediate_size] — FFN fc1 output
    pub ffn_up: B::Tensor,    // [max_patches, intermediate_size] — FFN up (GeGLU only)
    pub ffn_out: B::Tensor,   // [max_patches, hidden_size] — FFN fc2 output
    pub merge_buf: Option<B::Tensor>,  // [max_merged, hidden × merge²] — spatial merge
    pub proj_out: B::Tensor,  // [max_tokens, out_hidden_size] — final output
}

/// Allocate scratch buffers for the vision encoder.
pub(crate) fn alloc_vision_buffers<B: GpuCore>(
    backend: &B,
    config: &VisionConfig,
    max_patches: usize,
) -> VisionBuffers<B> {
    let hd = config.hidden_size;
    let inter = config.intermediate_size;
    let ms = config.spatial_merge_size.max(1);
    let max_merged = max_patches / (ms * ms);

    VisionBuffers {
        hidden: backend.alloc_tensor(&[max_patches, hd], TensorDtype::BF16),
        norm_out: backend.alloc_tensor(&[max_patches, hd], TensorDtype::BF16),
        q_buf: backend.alloc_tensor(&[max_patches, hd], TensorDtype::BF16),
        k_buf: backend.alloc_tensor(&[max_patches, hd], TensorDtype::BF16),
        v_buf: backend.alloc_tensor(&[max_patches, hd], TensorDtype::BF16),
        attn_out: backend.alloc_tensor(&[max_patches, hd], TensorDtype::BF16),
        ffn_gate: backend.alloc_tensor(&[max_patches, inter], TensorDtype::BF16),
        ffn_up: backend.alloc_tensor(&[max_patches, inter], TensorDtype::BF16),
        ffn_out: backend.alloc_tensor(&[max_patches, hd], TensorDtype::BF16),
        merge_buf: if config.spatial_merge_size > 0 {
            Some(backend.alloc_tensor(&[max_merged, hd * ms * ms], TensorDtype::BF16))
        } else {
            None
        },
        proj_out: backend.alloc_tensor(&[max_merged, config.out_hidden_size], TensorDtype::BF16),
    }
}

/// Run the vision encoder on a preprocessed image.
///
/// Transforms patches into vision tokens and writes them to `bufs.proj_out`.
/// The caller (forward_prefill_paged in mod.rs) then scatters these tokens
/// into the text embedding buffer at <|image_pad|> positions.
///
/// The forward pass mirrors a standard ViT:
///   1. Patch embedding (linear projection)
///   2. Positional embedding addition
///   3. N transformer blocks (LayerNorm → Attention → FFN, each with residual)
///   4. Merger/projector (maps vision dim → LLM hidden dim)
pub(crate) fn vision_encode<B: GpuBackend>(
    backend: &B,
    processed: &ProcessedImage,
    weights: &VisionWeights<B>,
    config: &VisionConfig,
    bufs: &VisionBuffers<B>,
) -> anyhow::Result<()> {
    let n = processed.grid_h * processed.grid_w; // total patches
    let hd = config.hidden_size;
    let patch_dim = config.in_channels * config.patch_size * config.patch_size;
    let eps = 1e-6f32; // LayerNorm epsilon (standard ViT value)

    // Upload pixel data from CPU to GPU.
    let pixel_tensor = backend.upload_tensor(&processed.pixels, &[n, patch_dim], TensorDtype::BF16);

    // -----------------------------------------------------------------------
    // Stage 1: Patch embedding.
    //
    // Each patch is a flattened [768] vector (3 channels × 16 × 16 pixels).
    // The patch_embed_weight [1152, 768] projects each patch to the vision
    // hidden dimension.  This is equivalent to a Conv2D with kernel_size ==
    // stride == patch_size, but implemented as a matmul for simplicity.
    //
    //   patches [N, 768] × weight^T [768, 1152] → embeddings [N, 1152]
    // -----------------------------------------------------------------------
    backend.matmul_batch(
        &weights.patch_embed_weight, &pixel_tensor, &bufs.hidden,
        hd as u32, patch_dim as u32, n as u32,
    );
    if let Some(ref bias) = weights.patch_embed_bias {
        backend.bias_add_batch(&bufs.hidden, bias, &bufs.hidden, n as u32, hd as u32);
    }

    // -----------------------------------------------------------------------
    // Stage 2: Add positional embeddings.
    //
    // ViT uses learned absolute positional embeddings — each grid position
    // (row, col) has a unique learned [1152] vector.  Adding these tells the
    // transformer WHERE each patch comes from in the image.
    //
    // Unlike RoPE (which encodes position via rotations in the Q/K vectors),
    // absolute embeddings are added directly to the patch embeddings.  This
    // is simpler but less flexible for varying resolutions.
    // -----------------------------------------------------------------------
    backend.add(&bufs.hidden, &weights.pos_embed, &bufs.hidden, (n * hd) as u32);

    // -----------------------------------------------------------------------
    // Stage 3: ViT transformer blocks.
    //
    // 27 blocks of standard pre-norm transformer:
    //   hidden = hidden + Attention(LayerNorm(hidden))
    //   hidden = hidden + FFN(LayerNorm(hidden))
    //
    // Key differences from the LLM transformer:
    //   - LayerNorm instead of RMSNorm (includes mean-centering + bias)
    //   - Bidirectional attention (causal=false — all patches see all patches)
    //   - Plain GELU activation instead of SwiGLU/GeGLU
    //   - All projections have bias terms
    //   - No KV cache (vision encoding is one-shot, not autoregressive)
    // -----------------------------------------------------------------------
    let num_heads = config.num_heads as u32;
    let head_dim = (hd / config.num_heads) as u32;
    let inter = config.intermediate_size;

    for block in &weights.blocks {
        let is_geglu = block.up_weight.is_some();

        // Pre-attention LayerNorm: normalise hidden state before attention.
        backend.layer_norm_batch(
            &bufs.hidden, &block.norm1_weight, &block.norm1_bias,
            eps, &bufs.norm_out, n as u32,
        );

        // Q, K, V projections.
        // Each projects [N, 1152] → [N, 1152] (= [N, 16 heads × 72 dim/head]).
        backend.matmul_batch(&block.q_weight, &bufs.norm_out, &bufs.q_buf, hd as u32, hd as u32, n as u32);
        backend.matmul_batch(&block.k_weight, &bufs.norm_out, &bufs.k_buf, hd as u32, hd as u32, n as u32);
        backend.matmul_batch(&block.v_weight, &bufs.norm_out, &bufs.v_buf, hd as u32, hd as u32, n as u32);

        // Add bias to Q/K/V if present (SigLIP uses bias on all projections).
        if let Some(ref qb) = block.q_bias {
            backend.bias_add_batch(&bufs.q_buf, qb, &bufs.q_buf, n as u32, hd as u32);
        }
        if let Some(ref kb) = block.k_bias {
            backend.bias_add_batch(&bufs.k_buf, kb, &bufs.k_buf, n as u32, hd as u32);
        }
        if let Some(ref vb) = block.v_bias {
            backend.bias_add_batch(&bufs.v_buf, vb, &bufs.v_buf, n as u32, hd as u32);
        }

        // Bidirectional multi-head self-attention.
        // causal=false: every patch attends to every other patch (no mask).
        // This is the key difference from LLM attention — images have no
        // natural left-to-right ordering, so masking would lose information.
        backend.prefill_attention(
            &bufs.q_buf, &bufs.k_buf, &bufs.v_buf, &bufs.attn_out,
            n as u32, 0, num_heads, num_heads, head_dim, 0, 0.0, None,
            false, // bidirectional — NOT causal
        );

        // Output projection: [N, 1152] → [N, 1152].
        backend.matmul_batch(&block.proj_weight, &bufs.attn_out, &bufs.norm_out, hd as u32, hd as u32, n as u32);
        backend.bias_add_batch(&bufs.norm_out, &block.proj_bias, &bufs.norm_out, n as u32, hd as u32);

        // Residual connection: hidden = hidden + attention_output.
        backend.add(&bufs.hidden, &bufs.norm_out, &bufs.hidden, (n * hd) as u32);

        // Pre-FFN LayerNorm.
        backend.layer_norm_batch(
            &bufs.hidden, &block.norm2_weight, &block.norm2_bias,
            eps, &bufs.norm_out, n as u32,
        );

        // Feed-forward network.
        if is_geglu {
            // GeGLU: out = gelu(gate(x)) × up(x)  (not currently used by vision)
            backend.matmul_batch(&block.fc1_weight, &bufs.norm_out, &bufs.ffn_gate, inter as u32, hd as u32, n as u32);
            backend.bias_add_batch(&bufs.ffn_gate, &block.fc1_bias, &bufs.ffn_gate, n as u32, inter as u32);
            if let Some(ref up_w) = block.up_weight {
                backend.matmul_batch(up_w, &bufs.norm_out, &bufs.ffn_up, inter as u32, hd as u32, n as u32);
                if let Some(ref up_b) = block.up_bias {
                    backend.bias_add_batch(&bufs.ffn_up, up_b, &bufs.ffn_up, n as u32, inter as u32);
                }
            }
            backend.gelu_mul(&bufs.ffn_gate, &bufs.ffn_up, &bufs.ffn_gate, (n * inter) as u32);
        } else {
            // Plain GELU: out = gelu(fc1(x))
            //
            // The standard ViT FFN: linear → GELU → linear.
            // fc1 expands 1152 → 4304, fc2 contracts 4304 → 1152.
            backend.matmul_batch(&block.fc1_weight, &bufs.norm_out, &bufs.ffn_gate, inter as u32, hd as u32, n as u32);
            backend.bias_add_batch(&bufs.ffn_gate, &block.fc1_bias, &bufs.ffn_gate, n as u32, inter as u32);
            backend.gelu(&bufs.ffn_gate, &bufs.ffn_gate, (n * inter) as u32);
        }

        // Down projection: [N, 4304] → [N, 1152].
        backend.matmul_batch(&block.fc2_weight, &bufs.ffn_gate, &bufs.ffn_out, hd as u32, inter as u32, n as u32);
        backend.bias_add_batch(&bufs.ffn_out, &block.fc2_bias, &bufs.ffn_out, n as u32, hd as u32);

        // Residual connection: hidden = hidden + ffn_output.
        backend.add(&bufs.hidden, &bufs.ffn_out, &bufs.hidden, (n * hd) as u32);
    }

    // -----------------------------------------------------------------------
    // Stage 4: Post-LayerNorm (Gemma 3 only).
    //
    // SigLIP applies a final LayerNorm after all transformer blocks.
    // Qwen 3.5's vision encoder does NOT have this — the merger handles
    // normalisation via its own LayerNorm.
    // -----------------------------------------------------------------------
    let src_buf = if let (Some(w), Some(b)) = (&weights.post_norm_weight, &weights.post_norm_bias) {
        backend.layer_norm_batch(&bufs.hidden, w, b, eps, &bufs.norm_out, n as u32);
        &bufs.norm_out
    } else {
        &bufs.hidden
    };

    // -----------------------------------------------------------------------
    // Stage 5: Merger / projector.
    //
    // The vision encoder produces [N, 1152] features, but the LLM expects
    // [M, 5120] embeddings (where M ≤ N).  The merger bridges this gap.
    //
    // Qwen 3.5 merger:
    //   1. Spatial merge: every 2×2 block of tokens → 1 concatenated token.
    //      [N, 1152] → [N/4, 4608]  (4 patches × 1152 dim = 4608)
    //   2. LayerNorm on the merged tokens
    //   3. 2-layer MLP: fc1 [5120, 4608] → GELU → fc2 [5120, 5120]
    //
    // Gemma 3 projector:
    //   Single linear layer: [1152, hidden_size] — no spatial merge.
    // -----------------------------------------------------------------------
    if config.spatial_merge_size > 0 {
        // Qwen spatial merge + MLP merger.
        let ms = config.spatial_merge_size as u32;
        let merge_buf = bufs.merge_buf.as_ref().unwrap();

        // Rearrange 2D grid: every 2×2 block → one concatenated token.
        // E.g., 14×28 grid (392 patches) → 7×14 grid (98 merged tokens).
        backend.spatial_merge(
            src_buf, merge_buf,
            processed.grid_h as u32, processed.grid_w as u32,
            hd as u32, ms,
        );

        let merged_n = processed.num_vision_tokens;
        let merged_hd = hd * (ms as usize) * (ms as usize);

        // LayerNorm on merged tokens (before the MLP).
        if let (Some(nw), Some(nb)) = (&weights.merger_norm_weight, &weights.merger_norm_bias) {
            backend.layer_norm_batch(merge_buf, nw, nb, eps, merge_buf, merged_n as u32);
        }

        // Merger MLP: fc1 → GELU → fc2 (or just fc1 if single-layer).
        let fc1_out_dim = config.out_hidden_size;
        backend.matmul_batch(
            &weights.merger_fc1_weight, merge_buf, &bufs.proj_out,
            fc1_out_dim as u32, merged_hd as u32, merged_n as u32,
        );
        backend.bias_add_batch(&bufs.proj_out, &weights.merger_fc1_bias, &bufs.proj_out, merged_n as u32, fc1_out_dim as u32);

        if let Some(ref fc2_w) = weights.merger_fc2_weight {
            backend.gelu(&bufs.proj_out, &bufs.proj_out, (merged_n * fc1_out_dim) as u32);
            backend.matmul_batch(
                fc2_w, &bufs.proj_out, &bufs.proj_out,
                config.out_hidden_size as u32, fc1_out_dim as u32, merged_n as u32,
            );
            if let Some(ref fc2_b) = weights.merger_fc2_bias {
                backend.bias_add_batch(&bufs.proj_out, fc2_b, &bufs.proj_out, merged_n as u32, config.out_hidden_size as u32);
            }
        }
    } else {
        // Gemma: simple linear projection [N, 1152] → [N, hidden_size].
        backend.matmul_batch(
            &weights.merger_fc1_weight, src_buf, &bufs.proj_out,
            config.out_hidden_size as u32, hd as u32, n as u32,
        );
        backend.bias_add_batch(&bufs.proj_out, &weights.merger_fc1_bias, &bufs.proj_out, n as u32, config.out_hidden_size as u32);
    }

    // Output is now in bufs.proj_out: [num_vision_tokens, out_hidden_size].
    // The caller (forward_prefill_paged) will scatter these into the text
    // embedding buffer at <|image_pad|> positions.
    Ok(())
}
