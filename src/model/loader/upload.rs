// ===========================================================================
// Tensor upload utilities — loading individual tensors from safetensors to GPU.
//
// These helpers handle the conversion from on-disk safetensors format to
// GPU-resident tensors, including auto-detection of Q4/Q8 pre-quantized
// weights, shape validation, norm residual offsets, and tensor-parallel
// sharding.
//
// Related files:
//   loader/store.rs — TensorStore (provides tensor data)
//   gpu/mod.rs      — GpuCore::upload_tensor (actual GPU upload)
// ===========================================================================

use half::bf16;

use crate::gpu::{GpuCore, TensorDtype};
use super::store::TensorStore;

// ---------------------------------------------------------------------------
// Tensor upload helpers.
// ---------------------------------------------------------------------------

/// Compute total byte count for a quantized [m, k] tensor.
pub(crate) fn quant_byte_count(dtype: TensorDtype, m: usize, k: usize) -> usize {
    match dtype {
        TensorDtype::Q4 => crate::gpu::q4_byte_count(m, k),
        TensorDtype::Q8 => crate::gpu::q8_byte_count(m, k),
        TensorDtype::FP8 => crate::gpu::fp8_byte_count(m, k),
        _ => panic!("quant_byte_count called for non-quantized dtype {dtype:?}"),
    }
}

/// Compute row byte stride for a weight dimension, given an optional quant dtype.
/// Returns bf16 row bytes (k * 2) when dtype is None.
pub(crate) fn quant_row_bytes(dtype: Option<TensorDtype>, k: usize) -> usize {
    match dtype {
        Some(TensorDtype::Q4) => (k / 32) * 18,
        Some(TensorDtype::Q8) => (k / 32) * 34,
        Some(TensorDtype::FP8) => k, // 1 byte per weight, no block overhead
        None => k * 2, // bf16
        _ => panic!("quant_row_bytes called for unsupported dtype {dtype:?}"),
    }
}

/// Short name for a quant dtype (for error messages).
pub(crate) fn quant_dtype_name(dtype: TensorDtype) -> &'static str {
    match dtype {
        TensorDtype::Q4 => "Q4",
        TensorDtype::Q8 => "Q8",
        TensorDtype::FP8 => "FP8",
        _ => "bf16",
    }
}

/// Upload a single tensor from the store to GPU memory (bf16, f32, Q4, or Q8).
///
/// If the tensor is in the store's quant map (pre-quantized by `rllm quantize`),
/// uploads the raw quantized bytes directly with the original logical shape.
pub(crate) fn upload_tensor<B: GpuCore>(
    store: &TensorStore,
    backend: &B,
    name: &str,
    expected_shape: &[usize],
) -> anyhow::Result<B::Tensor> {
    // Pre-quantized Q4/Q8/FP8: raw U8 bytes, upload directly.
    if let Some((m, k, dtype)) = store.quant_shape(name) {
        let view = store.tensor(name)?;
        let expected_bytes = quant_byte_count(dtype, m, k);
        anyhow::ensure!(
            view.data().len() == expected_bytes,
            "pre-quantized tensor '{name}' byte count mismatch: expected {expected_bytes}, got {}",
            view.data().len()
        );
        return Ok(backend.upload_tensor(view.data(), &[m, k], dtype));
    }

    let view = store.tensor(name)?;

    let shape = view.shape();
    anyhow::ensure!(
        shape == expected_shape,
        "tensor '{name}' shape mismatch: expected {expected_shape:?}, got {shape:?}"
    );

    let dtype = match view.dtype() {
        safetensors::Dtype::BF16 => TensorDtype::BF16,
        safetensors::Dtype::F32 => TensorDtype::F32,
        other => anyhow::bail!("unsupported dtype {:?} for tensor '{name}'", other),
    };

    Ok(backend.upload_tensor(view.data(), shape, dtype))
}

/// Upload a norm weight with residual form: effective_weight = 1.0 + stored_weight.
///
/// Qwen 3.5 stores RMSNorm weights as offsets from 1.0 (initialized to zeros, so
/// effective weight starts at 1.0).  This differs from Llama/Qwen2 which store
/// direct scale factors (initialized to ones).  We add 1.0 during loading so the
/// existing rms_norm kernel works unchanged.
pub(crate) fn upload_norm_residual<B: GpuCore>(
    store: &TensorStore,
    backend: &B,
    name: &str,
    expected_shape: &[usize],
) -> anyhow::Result<B::Tensor> {
    let view = store.tensor(name)?;
    let shape = view.shape();
    anyhow::ensure!(
        shape == expected_shape,
        "tensor '{name}' shape mismatch: expected {expected_shape:?}, got {shape:?}"
    );

    let bf16_out: Vec<bf16> = match view.dtype() {
        safetensors::Dtype::BF16 => bytemuck::cast_slice::<u8, bf16>(view.data())
            .iter()
            .map(|v| bf16::from_f32(v.to_f32() + 1.0))
            .collect(),
        safetensors::Dtype::F32 => bytemuck::cast_slice::<u8, f32>(view.data())
            .iter()
            .map(|v| bf16::from_f32(v + 1.0))
            .collect(),
        other => anyhow::bail!("unsupported dtype {:?} for tensor '{name}'", other),
    };

    Ok(backend.upload_tensor(bytemuck::cast_slice(&bf16_out), shape, TensorDtype::BF16))
}

/// Upload raw bf16 bytes as a tensor.
///
/// Used for pre-sliced byte buffers (e.g. when splitting a fused QKV or
/// MoE expert weight from a larger tensor).
pub(crate) fn upload_raw_bf16<B: GpuCore>(
    backend: &B,
    bf16_data: &[u8],
    shape: &[usize],
) -> B::Tensor {
    backend.upload_tensor(bf16_data, shape, TensorDtype::BF16)
}

/// Upload a weight tensor with optional sharding.
///
/// If a sharding plan is provided and has a non-Replicated entry for this
/// weight, the raw bytes are sliced to this rank's portion before uploading.
/// Otherwise falls through to the regular upload path.
pub(crate) fn upload_sharded<B: GpuCore>(
    store: &TensorStore,
    backend: &B,
    name: &str,
    expected_shape: &[usize],
    sharding: Option<&crate::gpu::parallel::ShardingPlan>,
) -> anyhow::Result<B::Tensor> {
    use crate::gpu::parallel::{SplitDimension, slice_tensor_data};

    if let Some(plan) = sharding {
        if let Some(ws) = plan.get(name) {
            if !matches!(ws.split, SplitDimension::Replicated) {
                // Pre-quantized weights (Q4, Q8, FP8): slice in the quantized
                // domain so each rank gets its shard without dequantizing.
                //
                // Q4/Q8 use a block format (32 weights per block), so we slice
                // in "block" coordinates: logical shape [m, k/32] with
                // bytes_per_block = 18 (Q4) or 34 (Q8).
                // FP8 has no block structure: 1 byte per element.
                if let Some((m, k, qdt)) = store.quant_shape(name) {
                    let view = store.tensor(name)?;
                    let data = view.data();
                    let expected_bytes = quant_byte_count(qdt, m, k);
                    anyhow::ensure!(
                        data.len() == expected_bytes,
                        "pre-quantized {qdt:?} tensor '{name}' byte count mismatch: \
                         expected {expected_bytes}, got {}",
                        data.len()
                    );

                    let (block_shape, bytes_per_block) = match qdt {
                        TensorDtype::Q4 => ([m, k / 32], 18usize),
                        TensorDtype::Q8 => ([m, k / 32], 34usize),
                        TensorDtype::FP8 => ([m, k], 1usize),
                        _ => unreachable!(),
                    };

                    let (sliced, shard_block_shape) = slice_tensor_data(
                        data,
                        &block_shape,
                        &ws.split,
                        plan.device.rank,
                        plan.device.world_size,
                        bytes_per_block,
                    );

                    // Convert shard block shape back to logical weight shape.
                    let shard_shape = match qdt {
                        TensorDtype::Q4 | TensorDtype::Q8 => {
                            [shard_block_shape[0], shard_block_shape[1] * 32]
                        }
                        _ => shard_block_shape,
                    };

                    return Ok(backend.upload_tensor(&sliced, &shard_shape, qdt));
                }

                // Read raw bytes from safetensors (bf16 / f32).
                let view = store.tensor(name)?;
                let data = view.data();

                // Determine bytes per element from the file's dtype.
                let bpe = match view.dtype() {
                    safetensors::Dtype::BF16 => 2,
                    safetensors::Dtype::F32 => 4,
                    other => {
                        anyhow::bail!("unsupported dtype {:?} for sharded tensor '{name}'", other)
                    }
                };

                // Slice to this rank's shard.
                let (sliced, shard_shape) = slice_tensor_data(
                    data,
                    &ws.original_shape,
                    &ws.split,
                    plan.device.rank,
                    plan.device.world_size,
                    bpe,
                );

                return Ok(upload_raw_bf16(backend, &sliced, &shard_shape));
            }
        }
    }
    // Fallback: no sharding or Replicated — use regular upload.
    upload_tensor(store, backend, name, expected_shape)
}
