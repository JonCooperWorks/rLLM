// ===========================================================================
// Vision encoder weight loading (SigLIP ViT for VLMs).
//
// Loads the vision encoder weights for Qwen 3.5 and Gemma 3 vision-language
// models.  Handles f32→bf16 conversion, fused QKV splitting, temporal
// averaging of positional embeddings, and arch-specific weight layouts.
//
// Related files:
//   model/vision.rs           — VisionWeights, VisionBuffers, vision_encode()
//   loader/mod.rs             — load_model() calls load_vision_weights()
//   gpu/ops/vision.rs         — GpuVision trait (spatial merge, scatter)
// ===========================================================================


use crate::gpu::{GpuCore, TensorDtype};
use crate::model::config::ModelConfig;
use crate::model::vision::{VisionBlockWeights, VisionWeights};
use super::TensorStore;



/// Convert a safetensors tensor view to bf16 bytes, regardless of source dtype.
/// Used for splitting fused weights (QKV, gate+up) that may be stored as f32.
///
/// LEARNING NOTE: We need this separate from `upload_vision` because fused weight
/// splitting requires byte-level slicing of the converted data *before* uploading.
/// `upload_vision` converts and uploads in one step, but for QKV [3*hd, hd] we
/// need to: (1) convert to bf16, (2) slice into Q/K/V byte ranges, (3) upload
/// each slice separately.  This helper handles step (1).
fn to_bf16_vec(view: &safetensors::tensor::TensorView<'_>) -> Vec<u8> {
    match view.dtype() {
        safetensors::Dtype::BF16 => view.data().to_vec(),
        safetensors::Dtype::F32 => {
            let f32_data: &[f32] = bytemuck::cast_slice(view.data());
            let bf16_data: Vec<half::bf16> =
                f32_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
            bytemuck::cast_slice(&bf16_data).to_vec()
        }
        safetensors::Dtype::F16 => {
            let f16_data: &[half::f16] = bytemuck::cast_slice(view.data());
            let bf16_data: Vec<half::bf16> = f16_data
                .iter()
                .map(|v| half::bf16::from_f32(v.to_f32()))
                .collect();
            bytemuck::cast_slice(&bf16_data).to_vec()
        }
        other => panic!("unsupported dtype {:?} for vision tensor", other),
    }
}

/// Upload a vision tensor, converting f32→bf16 if needed.
///
/// Vision weights may be stored as f32 (SigLIP encoder) while the rest of
/// the model is bf16.  This helper detects the dtype and converts as needed.
///
/// LEARNING NOTE: SigLIP vision encoders are often trained and saved in f32
/// (or sometimes f16), even when the text model weights are bf16.  This is
/// because vision encoders are smaller (~400M params) so the storage savings
/// of bf16 matter less, and some training pipelines default to f32.  We
/// normalise everything to bf16 on CPU before uploading to GPU, so the
/// forward pass only needs bf16 kernels.
fn upload_vision<B: GpuCore>(
    backend: &B,
    view: &safetensors::tensor::TensorView<'_>,
    shape: &[usize],
) -> B::Tensor {
    match view.dtype() {
        safetensors::Dtype::BF16 => backend.upload_tensor(view.data(), shape, TensorDtype::BF16),
        safetensors::Dtype::F32 => {
            // Convert f32 → bf16 on CPU before uploading.
            let f32_data: &[f32] = bytemuck::cast_slice(view.data());
            let bf16_data: Vec<half::bf16> =
                f32_data.iter().map(|&v| half::bf16::from_f32(v)).collect();
            backend.upload_tensor(bytemuck::cast_slice(&bf16_data), shape, TensorDtype::BF16)
        }
        safetensors::Dtype::F16 => {
            // Convert f16 → bf16 on CPU.
            let f16_data: &[half::f16] = bytemuck::cast_slice(view.data());
            let bf16_data: Vec<half::bf16> = f16_data
                .iter()
                .map(|v| half::bf16::from_f32(v.to_f32()))
                .collect();
            backend.upload_tensor(bytemuck::cast_slice(&bf16_data), shape, TensorDtype::BF16)
        }
        other => panic!("unsupported dtype {:?} for vision tensor", other),
    }
}

/// Load vision encoder weights from safetensors.
///
/// Returns None if the model has no vision config or if vision weight
/// tensors are not found in the safetensors files (e.g. shard not downloaded).
pub(crate) fn load_vision_weights<B: GpuCore>(
    backend: &B,
    store: &TensorStore,
    config: &ModelConfig,
) -> Option<VisionWeights<B>> {
    let vc = config.vision.as_ref()?;
    let vp = &vc.weight_prefix;
    let hd = vc.hidden_size;
    let inter = vc.intermediate_size;
    let patch_dim = vc.in_channels * vc.patch_size * vc.patch_size;

    // Check if vision weights exist by probing the first block.
    let probe = if vc.fused_qkv {
        format!("{}blocks.0.attn.qkv.weight", vp)
    } else {
        format!("{}encoder.layers.0.self_attn.q_proj.weight", vp)
    };
    match store.tensor(&probe) {
        Err(_) => {
            eprintln!("vision weights not found ({}), skipping vision encoder", probe);
            return None;
        }
        Ok(view) => {
            // Skip vision if weights are in an unsupported dtype (e.g. U8/int8
            // quantized vision weights in some Q4 model distributions).
            let dt = view.dtype();
            if !matches!(dt, safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16) {
                eprintln!("vision weights have unsupported dtype {:?}, skipping vision encoder", dt);
                return None;
            }
        }
    }

    eprintln!("loading vision encoder weights ({} blocks, hidden_size={})", vc.depth, hd);

    // Patch embedding — stored as conv2d [out_channels, in_channels, kH, kW].
    // Reshape to [hidden_size, patch_dim] for matmul-based patch embedding.
    // Patch embedding — stored as conv2d or conv3d weight.
    // Qwen 3.5 uses temporal_patch_size=2 for video, so the weight shape is
    // [out_ch, in_ch, temporal, kH, kW].  For images (single frame), we average
    // the temporal dimension to get [out_ch, in_ch * kH * kW].
    //
    // LEARNING NOTE: Conv3D → Conv2D conversion for image-only inference.
    // Qwen 3.5 VL's patch embedding is a 3D convolution that operates over
    // (temporal, height, width) to support video input.  For images, we only
    // have one frame, so the temporal dimension is meaningless.  Rather than
    // implementing 3D convolution, we average the temporal kernel weights to
    // collapse [out_ch, in_ch, T, kH, kW] → [out_ch, in_ch*kH*kW].  This
    // is mathematically equivalent to running the 3D conv on T identical
    // frames and taking the mean output — a standard trick for adapting
    // video models to image inputs.
    // Qwen uses "patch_embed.proj.weight", Gemma uses "embeddings.patch_embedding.weight".
    let patch_embed_name = format!("{}patch_embed.proj.weight", vp);
    let patch_embed_alt = format!("{}embeddings.patch_embedding.weight", vp);
    let patch_view = store
        .tensor(&patch_embed_name)
        .or_else(|_| store.tensor(&patch_embed_alt))
        .unwrap_or_else(|_| panic!(
            "missing patch_embed weight: tried '{}' and '{}'",
            patch_embed_name, patch_embed_alt
        ));
    let patch_shape = patch_view.shape();
    let patch_embed_weight = if patch_shape.len() == 5 {
        // [out_ch, in_ch, temporal, kH, kW] → average over temporal → [out_ch, in_ch*kH*kW]
        let temporal = patch_shape[2];
        let total_elements: usize = patch_shape.iter().product();
        let bf16_bytes = to_bf16_vec(&patch_view);
        let all: &[half::bf16] = bytemuck::cast_slice(&bf16_bytes);
        assert_eq!(all.len(), total_elements);
        // Average temporal frames: for each output row, average `temporal` sub-rows.
        let row_size = patch_dim; // in_ch * kH * kW
        let mut averaged = vec![0.0f32; hd * row_size];
        for out_ch in 0..hd {
            for t in 0..temporal {
                for j in 0..row_size {
                    averaged[out_ch * row_size + j] +=
                        all[out_ch * temporal * row_size + t * row_size + j].to_f32();
                }
            }
            for j in 0..row_size {
                averaged[out_ch * row_size + j] /= temporal as f32;
            }
        }
        let bf16_avg: Vec<half::bf16> = averaged.iter().map(|&v| half::bf16::from_f32(v)).collect();
        backend.upload_tensor(bytemuck::cast_slice(&bf16_avg), &[hd, patch_dim], TensorDtype::BF16)
    } else {
        // [out_ch, in_ch, kH, kW] → reshape to [out_ch, in_ch*kH*kW]
        upload_vision(backend, &patch_view, &[hd, patch_dim])
    };

    let patch_bias_name = format!("{}patch_embed.proj.bias", vp);
    let patch_bias_alt = format!("{}embeddings.patch_embedding.bias", vp);
    let patch_embed_bias = store.tensor(&patch_bias_name)
        .or_else(|_| store.tensor(&patch_bias_alt))
        .ok()
        .map(|v| upload_vision(backend, &v, &[hd]));

    // Positional embeddings.
    let pos_name = if vc.fused_qkv {
        format!("{}pos_embed.weight", vp)
    } else {
        format!("{}embeddings.position_embedding.weight", vp)
    };
    let pos_view = store.tensor(&pos_name).expect("missing pos_embed weight");
    let pos_embed = upload_vision(backend, &pos_view, pos_view.shape());

    // Per-block weights.
    let mut blocks = Vec::with_capacity(vc.depth);
    for i in 0..vc.depth {
        let block = if vc.fused_qkv {
            load_qwen_vision_block(backend, store, vp, i, hd, inter)?
        } else {
            load_gemma_vision_block(backend, store, vp, i, hd, inter)?
        };
        blocks.push(block);
    }

    // Post-LayerNorm (Gemma only).
    let post_norm_name_w = format!("{}post_layernorm.weight", vp);
    let post_norm_weight = store.tensor(&post_norm_name_w).ok()
        .map(|v| upload_vision(backend, &v, &[hd]));
    let post_norm_name_b = format!("{}post_layernorm.bias", vp);
    let post_norm_bias = store.tensor(&post_norm_name_b).ok()
        .map(|v| upload_vision(backend, &v, &[hd]));

    // Merger / projector.
    let pp = &vc.projector_prefix;
    // Merger LayerNorm: Qwen uses "norm.weight", Gemma 3 uses "mm_soft_emb_norm.weight".
    let merger_norm_w = store.tensor(&format!("{}norm.weight", pp)).ok()
        .or_else(|| store.tensor(&format!("{}mm_soft_emb_norm.weight", pp)).ok())
        .map(|v| upload_vision(backend, &v, v.shape()));
    let merger_norm_b = store.tensor(&format!("{}norm.bias", pp)).ok()
        .or_else(|| store.tensor(&format!("{}mm_soft_emb_norm.bias", pp)).ok())
        .map(|v| upload_vision(backend, &v, v.shape()));

    // Merger fc1: for Qwen this is the first layer of the 2-layer MLP.
    // For Gemma 3 this is a single linear projection with a different naming
    // convention (mm_input_projection_weight, no bias).
    //
    // Try known names in order: Qwen fused → Qwen split → Gemma 3.
    let fc1_candidates = [
        format!("{}linear_fc1.weight", pp),
        format!("{}linear_1.weight", pp),
        format!("{}mm_input_projection_weight", pp),
    ];
    let fc1_view = fc1_candidates.iter()
        .find_map(|name| store.tensor(name).ok());
    let fc1_view = match fc1_view {
        Some(v) => v,
        None => {
            eprintln!(
                "WARNING: vision projector weight not found under any known name \
                 (tried: {}), skipping vision encoder",
                fc1_candidates.join(", ")
            );
            return None;
        }
    };

    // Gemma 3's mm_input_projection_weight is stored as [in_dim, out_dim] = [1152, 2560],
    // but our matmul expects [out_dim, in_dim] = [2560, 1152].  Transpose if needed.
    let fc1_shape = fc1_view.shape();
    // Gemma 3's mm_input_projection_weight is [in, out] — detect by checking
    // if the first dim (1152) < second dim (2560).  Qwen's fc1 is square or [out, in].
    let merger_fc1_weight = if fc1_shape.len() == 2 && fc1_shape[0] < fc1_shape[1] {
        // Weight is [in, out] — transpose to [out, in] for matmul.
        let (rows, cols) = (fc1_shape[0], fc1_shape[1]);
        let src_data = fc1_view.data();
        let mut transposed = vec![0u8; src_data.len()];
        let elem_size = 2; // bf16
        for r in 0..rows {
            for c in 0..cols {
                let src_off = (r * cols + c) * elem_size;
                let dst_off = (c * rows + r) * elem_size;
                transposed[dst_off..dst_off + elem_size]
                    .copy_from_slice(&src_data[src_off..src_off + elem_size]);
            }
        }
        backend.upload_tensor(&transposed, &[cols, rows], crate::gpu::TensorDtype::BF16)
    } else {
        upload_vision(backend, &fc1_view, fc1_shape)
    };

    // Projector bias: Qwen has one, Gemma 3 does not (uses a separate norm
    // layer instead).  Try the bias name that matches the weight we found.
    let merger_fc1_bias = fc1_candidates.iter()
        .find_map(|name| {
            let bias_name = name.replace(".weight", ".bias");
            store.tensor(&bias_name).ok()
        })
        .map(|v| upload_vision(backend, &v, v.shape()));

    // Merger fc2 (Qwen only — 2-layer MLP merger).
    let fc2_name_w = if store.tensor(&format!("{}linear_fc2.weight", pp)).is_ok() {
        Some(format!("{}linear_fc2.weight", pp))
    } else {
        None
    };
    let merger_fc2_weight = fc2_name_w.as_ref().map(|name| {
        let v = store.tensor(name).unwrap();
        upload_vision(backend, &v, v.shape())
    });
    let merger_fc2_bias = fc2_name_w.as_ref().map(|name| {
        let bias_name = name.replace(".weight", ".bias");
        let v = store.tensor(&bias_name).unwrap();
        upload_vision(backend, &v, v.shape())
    });

    eprintln!("vision encoder weights loaded ({} blocks)", vc.depth);

    Some(VisionWeights {
        patch_embed_weight,
        patch_embed_bias,
        pos_embed,
        blocks,
        post_norm_weight,
        post_norm_bias,
        merger_norm_weight: merger_norm_w,
        merger_norm_bias: merger_norm_b,
        merger_fc1_weight,
        merger_fc1_bias,
        merger_fc2_weight,
        merger_fc2_bias,
    })
}

/// Load a Qwen 3.5 vision block (fused QKV, GeGLU FFN).
fn load_qwen_vision_block<B: GpuCore>(
    backend: &B,
    store: &TensorStore,
    vp: &str,
    idx: usize,
    hd: usize,
    _inter: usize,
) -> Option<VisionBlockWeights<B>> {
    let bp = format!("{}blocks.{}", vp, idx);
    let tv = |name: &str| -> B::Tensor {
        let v = store.tensor(name).unwrap_or_else(|e| panic!("missing {name}: {e}"));
        upload_vision(backend, &v, v.shape())
    };

    // Norms.
    let norm1_weight = tv(&format!("{bp}.norm1.weight"));
    let norm1_bias = tv(&format!("{bp}.norm1.bias"));
    let norm2_weight = tv(&format!("{bp}.norm2.weight"));
    let norm2_bias = tv(&format!("{bp}.norm2.bias"));

    // Fused QKV [3*hd, hd] — keep as-is (no splitting).
    //
    // LEARNING NOTE: Qwen stores Q, K, V as a single fused weight matrix
    // [3*hd, hd].  One matmul produces the entire QKV output [N, 3*hd],
    // and the fused attention kernel (prefill_attention_fused_qkv) reads
    // Q/K/V at stride offsets within each row.  This eliminates 2 of the
    // 3 matmul dispatches — a 3× reduction in kernel launch overhead.
    let qkv_view = store.tensor(&format!("{bp}.attn.qkv.weight")).unwrap();
    let qkv_weight = upload_vision(backend, &qkv_view, &[3 * hd, hd]);

    let qkv_bias = store.tensor(&format!("{bp}.attn.qkv.bias")).ok()
        .map(|v| upload_vision(backend, &v, &[3 * hd]));

    let proj_weight = tv(&format!("{bp}.attn.proj.weight"));
    let proj_bias = tv(&format!("{bp}.attn.proj.bias"));

    // FFN: fc1 is [intermediate, hidden] — plain GELU, NOT GeGLU.
    // Both Qwen 3.5 and Gemma 3 vision encoders use plain GELU activation
    // despite the config saying "gelu_pytorch_tanh" (no gate+up split).
    let fc1_weight = tv(&format!("{bp}.mlp.linear_fc1.weight"));
    let fc1_bias = tv(&format!("{bp}.mlp.linear_fc1.bias"));
    let up_weight: Option<B::Tensor> = None;
    let up_bias: Option<B::Tensor> = None;

    let fc2_weight = tv(&format!("{bp}.mlp.linear_fc2.weight"));
    let fc2_bias = tv(&format!("{bp}.mlp.linear_fc2.bias"));

    Some(VisionBlockWeights {
        norm1_weight, norm1_bias, norm2_weight, norm2_bias,
        qkv_weight, qkv_bias,
        proj_weight, proj_bias,
        fc1_weight, fc1_bias,
        up_weight, up_bias,
        fc2_weight, fc2_bias,
    })
}

/// Load a Gemma 3 vision block (separate Q/K/V, GELU FFN).
///
/// LEARNING NOTE: Gemma's vision encoder uses the HuggingFace standard naming
/// convention (encoder.layers.N.self_attn.q_proj, etc.) with separate Q/K/V
/// projections — no fused weight splitting needed.  This is simpler than the
/// Qwen path above but produces the same VisionBlockWeights struct, so the
/// forward pass doesn't care which loader created the weights.
fn load_gemma_vision_block<B: GpuCore>(
    backend: &B,
    store: &TensorStore,
    vp: &str,
    idx: usize,
    hd: usize,
    _inter: usize,
) -> Option<VisionBlockWeights<B>> {
    let bp = format!("{}encoder.layers.{}", vp, idx);
    let t = |name: &str| -> B::Tensor {
        let v = store.tensor(name).unwrap_or_else(|e| panic!("missing {name}: {e}"));
        upload_vision(backend, &v, v.shape())
    };
    let _tb = |name: &str| -> Option<B::Tensor> {
        store.tensor(name).ok().map(|v| upload_vision(backend, &v, v.shape()))
    };

    let norm1_weight = t(&format!("{bp}.layer_norm1.weight"));
    let norm1_bias = t(&format!("{bp}.layer_norm1.bias"));
    let norm2_weight = t(&format!("{bp}.layer_norm2.weight"));
    let norm2_bias = t(&format!("{bp}.layer_norm2.bias"));

    // Concatenate separate Q/K/V weights into a single fused [3*hd, hd] tensor
    // for the fused QKV attention kernel.
    let q_view = store.tensor(&format!("{bp}.self_attn.q_proj.weight")).unwrap();
    let k_view = store.tensor(&format!("{bp}.self_attn.k_proj.weight")).unwrap();
    let v_view = store.tensor(&format!("{bp}.self_attn.v_proj.weight")).unwrap();
    let mut qkv_data = to_bf16_vec(&q_view);
    qkv_data.extend_from_slice(&to_bf16_vec(&k_view));
    qkv_data.extend_from_slice(&to_bf16_vec(&v_view));
    let qkv_weight = backend.upload_tensor(&qkv_data, &[3 * hd, hd], TensorDtype::BF16);

    // Concatenate Q/K/V biases similarly.
    let qkv_bias = if let Ok(qb) = store.tensor(&format!("{bp}.self_attn.q_proj.bias")) {
        let kb = store.tensor(&format!("{bp}.self_attn.k_proj.bias")).unwrap();
        let vb = store.tensor(&format!("{bp}.self_attn.v_proj.bias")).unwrap();
        let mut bias_data = to_bf16_vec(&qb);
        bias_data.extend_from_slice(&to_bf16_vec(&kb));
        bias_data.extend_from_slice(&to_bf16_vec(&vb));
        Some(backend.upload_tensor(&bias_data, &[3 * hd], TensorDtype::BF16))
    } else {
        None
    };

    let proj_weight = t(&format!("{bp}.self_attn.out_proj.weight"));
    let proj_bias = t(&format!("{bp}.self_attn.out_proj.bias"));

    let fc1_weight = t(&format!("{bp}.mlp.fc1.weight"));
    let fc1_bias = t(&format!("{bp}.mlp.fc1.bias"));
    let fc2_weight = t(&format!("{bp}.mlp.fc2.weight"));
    let fc2_bias = t(&format!("{bp}.mlp.fc2.bias"));

    Some(VisionBlockWeights {
        norm1_weight, norm1_bias, norm2_weight, norm2_bias,
        qkv_weight, qkv_bias,
        proj_weight, proj_bias,
        fc1_weight, fc1_bias,
        up_weight: None,
        up_bias: None,
        fc2_weight, fc2_bias,
    })
}
