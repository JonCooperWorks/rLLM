// ===========================================================================
// `rllm quantize` — Offline weight quantization.
//
// Pre-quantizes bf16 safetensors weights to Q4, Q8, or FP8 format on disk,
// so that subsequent loads (`rllm run`, `rllm serve`) skip the on-load
// quantization step entirely.  This is a pure-CPU operation — no GPU needed.
//
// Platform-aware dispatch:
//   `--format q8` produces FP8 E4M3 on NVIDIA SM 89+ (Ada/Hopper),
//   and Q8 block format on Metal or older NVIDIA GPUs.  The user always
//   types "q8" — FP8 selection is transparent.
//
// Output format:
//   Standard safetensors files with quantized blocks stored as Dtype::U8 tensors.
//   File-level metadata marks the file as quantized:
//     "quantization" = "rllm-q4" | "rllm-q8" | "rllm-fp8"
//     "rllm_q4:<tensor_name>" = "m,k"   (Q4 original logical shape)
//     "rllm_q8:<tensor_name>" = "m,k"   (Q8 original logical shape)
//     "rllm_fp8:<tensor_name>" = "m,k"  (FP8 original logical shape)
//
//   Non-quantizable tensors (norms, embeddings, biases) pass through as bf16.
//   The output directory is self-contained: includes config.json, tokenizer
//   files, and the quantized safetensors.
//
// Related files:
//   gpu/mod.rs          — quantize_bf16_to_q4(), quantize_bf16_to_q8(), quantize_bf16_to_fp8()
//   gpu/ops/quant.rs    — WeightQuantiser trait, QuantFormat enum
//   model/loader/     — load_safetensors_files(), pre-quantized detection
//   model/config.rs     — ModelConfig parsing
// ===========================================================================

use std::collections::HashMap;
use std::path::PathBuf;

use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use tracing::{info, debug};

use crate::gpu::ops::quant::{QuantFormat, quantiser, WeightQuantiser};
use crate::model::loader::mxfp4::dequantize_mxfp4;

#[derive(clap::Args)]
pub(crate) struct QuantizeArgs {
    /// Path to input model directory (contains config.json, tokenizer.json, *.safetensors).
    #[arg(long)]
    model: PathBuf,

    /// Path to output directory for quantized model.
    #[arg(long)]
    output: PathBuf,

    /// Quantization format: "q4" (4-bit, default), "q8" (8-bit), "fp8" (FP8 E4M3),
    /// or "tq3" (TurboQuant 3-bit, 4.0 bpw with Walsh-Hadamard rotation).
    /// When "q8" is used on NVIDIA SM 89+ (Ada/Hopper), FP8 E4M3 is selected
    /// automatically for better hardware utilisation.
    #[arg(long, default_value = "q4")]
    format: String,
}

/// Maximum bytes per output shard (5 GB).  Keeps files manageable and
/// matches the HuggingFace convention for large models.
const SHARD_LIMIT: usize = 5 * 1024 * 1024 * 1024;

/// Apply platform-aware format dispatch: Q8 → FP8 on NVIDIA SM 89+ (Ada/Hopper).
///
/// FP8 has native hardware support on these GPUs, giving better throughput
/// than Q8 block format.  On Metal or older NVIDIA GPUs, Q8 is unchanged.
fn platform_adjusted_format(format: QuantFormat) -> QuantFormat {
    #[cfg(feature = "cuda")]
    {
        let cc = cudarc::driver::CudaContext::new(0)
            .ok()
            .and_then(|ctx| ctx.compute_capability().ok())
            .unwrap_or((0, 0));
        if format == QuantFormat::Q4 && cc.0 >= 10 {
            info!(sm_major = cc.0, sm_minor = cc.1, "NVIDIA SM 100+ detected — using NVFP4 (E2M1) instead of Q4 blocks");
            return QuantFormat::NVFP4;
        }
        if format == QuantFormat::Q8 && (cc.0 > 8 || (cc.0 == 8 && cc.1 >= 9)) {
            info!(sm_major = cc.0, sm_minor = cc.1, "NVIDIA SM detected — using FP8 (E4M3) instead of Q8 blocks");
            return QuantFormat::FP8;
        }
    }
    format
}

pub(crate) fn exec(args: QuantizeArgs) -> anyhow::Result<()> {
    let input_dir = &args.model;
    let output_dir = &args.output;

    let format = QuantFormat::from_name(&args.format).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown quantization format '{}' (supported: q4, q8, fp8, tq3, nvfp4)",
            args.format
        )
    })?;

    let format = platform_adjusted_format(format);

    let quant = quantiser(format);

    // Validate input directory.
    anyhow::ensure!(
        input_dir.join("config.json").exists(),
        "no config.json found in {}",
        input_dir.display()
    );

    // Create output directory.
    std::fs::create_dir_all(output_dir)?;

    // Load and parse safetensors from input.
    let (mmaps, weight_map) = crate::model::loader::load_safetensors_files(input_dir)?;
    let shards: Vec<SafeTensors> = mmaps
        .iter()
        .map(|m| SafeTensors::deserialize(m))
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("failed to parse safetensors: {e}"))?;

    // Collect pre-existing quantization metadata from input shards so we can
    // propagate it for pass-through tensors (e.g. Q4 tensors that stay Q4 when
    // we're adding Q8 quantization on top).
    let mut input_quant_meta: HashMap<String, String> = HashMap::new();
    for mmap in &mmaps {
        if let Ok((_, metadata)) = SafeTensors::read_metadata(mmap.as_ref()) {
            if let Some(meta) = metadata.metadata() {
                for (key, val) in meta {
                    if key.starts_with("rllm_q4:") || key.starts_with("rllm_q8:") {
                        input_quant_meta.insert(key.clone(), val.clone());
                    }
                }
            }
        }
    }

    // Collect all tensors across shards in a stable order.
    let all_tensors = collect_all_tensors(&shards, &weight_map);

    let num_tensors = all_tensors.len();
    info!(tensors = num_tensors, shards = mmaps.len(), "found tensors");

    // Detect MXFP4 expert tensor groups (GPT-OSS format).
    //
    // MXFP4 experts are stored as _blocks + _scales + optional _bias triplets.
    // We collect these group prefixes so the main loop can dequant→requant them
    // instead of passing through the packed U8 bytes unchanged.
    let mxfp4_prefixes = detect_mxfp4_groups(&all_tensors);
    if !mxfp4_prefixes.is_empty() {
        info!(
            groups = mxfp4_prefixes.len(),
            "detected MXFP4 expert groups — will dequant to bf16 then re-quantize"
        );
    }

    // Process and write tensors incrementally to avoid holding all Q4 data
    // in memory.  For large MoE models (397B = 751 GB bf16 → 235 GB Q4),
    // accumulating all output would exceed RAM.  Instead, we fill one output
    // shard at a time and flush it to disk before starting the next.
    //
    // MXFP4 experts are dequantized per-expert within each layer to bound
    // peak RAM to ~one expert's bf16 weights at a time.
    let mut quantized_count = 0usize;
    let mut original_bytes = 0u64;
    let mut quantized_bytes = 0u64;
    let mut output_shard_count = 0usize;

    // Current shard accumulator — flushed when it exceeds SHARD_LIMIT.
    let mut current_shard: Vec<OutputTensor> = Vec::new();
    let mut current_shard_size = 0usize;

    // Global weight_map for the index file (tensor name → shard filename).
    let mut index_weight_map: Vec<(String, usize)> = Vec::new();

    // Build a lookup for quick tensor access by name (for MXFP4 group processing).
    let tensor_by_name: HashMap<&str, &safetensors::tensor::TensorView> = all_tensors
        .iter()
        .map(|(n, v)| (n.as_str(), v))
        .collect();

    // Pre-populate the set of MXFP4 component tensors to skip in the main loop.
    // _scales and _bias are consumed by the group processing triggered by _blocks.
    // _blocks itself is NOT pre-consumed — it's the trigger that initiates dequant.
    let mut mxfp4_consumed: std::collections::HashSet<String> = std::collections::HashSet::new();
    for base in &mxfp4_prefixes {
        mxfp4_consumed.insert(format!("{base}_scales"));
        mxfp4_consumed.insert(format!("{base}_bias"));
    }

    for (tensor_idx, (name, view)) in all_tensors.iter().enumerate() {
        // Skip MXFP4 component tensors already consumed by group processing.
        if mxfp4_consumed.contains(name.as_str()) {
            continue;
        }

        let shape = view.shape();
        let data = view.data();

        // Check if this is the _blocks tensor that triggers an MXFP4 group.
        if name.ends_with("_blocks") {
            let base = name.strip_suffix("_blocks").unwrap();
            if mxfp4_prefixes.contains(base) {
                // Process MXFP4 group: dequant per-expert, re-quantize, emit.
                let outputs = dequant_requant_mxfp4_group(
                    base, &tensor_by_name, quant, format,
                    &mut original_bytes, &mut quantized_bytes, &mut quantized_count,
                    &mut mxfp4_consumed,
                )?;

                for output in outputs {
                    let tensor_size = output.data.as_slice().len();
                    if current_shard_size > 0 && current_shard_size + tensor_size > SHARD_LIMIT {
                        write_single_shard(&current_shard, output_dir, output_shard_count, &mut index_weight_map, format)?;
                        output_shard_count += 1;
                        current_shard.clear();
                        current_shard_size = 0;
                    }
                    current_shard_size += tensor_size;
                    current_shard.push(output);
                }
                continue;
            }
        }

        let output = if should_quantize(name, shape, view.dtype()) {
            // For 3D fused expert tensors [num_experts, rows, k], flatten to
            // [num_experts * rows, k] — Q4 quantization is per-row so this
            // produces identical results to quantizing each expert separately.
            let (m, k) = if shape.len() == 3 {
                (shape[0] * shape[1], shape[2])
            } else {
                (shape[0], shape[1])
            };
            let quant_data = quant.quantise(data, m, k);

            original_bytes += data.len() as u64;
            quantized_bytes += quant_data.len() as u64;
            quantized_count += 1;

            if tensor_idx % 10 == 0 || quant_data.len() > 100_000_000 {
                debug!(
                    "[{}/{}] quantized {} ({:.0} MB → {:.0} MB)",
                    tensor_idx + 1, num_tensors, name,
                    data.len() as f64 / 1e6, quant_data.len() as f64 / 1e6,
                );
            }

            OutputTensor {
                name: name.clone(),
                data: TensorData::Owned(quant_data),
                dtype: safetensors::Dtype::U8,
                shape: vec![quant.byte_count(m, k)],
                quant_original_shape: Some((m, k)),
                passthrough_quant: None,
            }
        } else {
            original_bytes += data.len() as u64;
            quantized_bytes += data.len() as u64;

            // Propagate pre-existing quantization metadata for pass-through
            // tensors (e.g. Q4 tensors kept as-is in a Q4→Q8 hybrid model).
            let passthrough = ["rllm_q4:", "rllm_q8:"].iter().find_map(|prefix| {
                let key = format!("{prefix}{name}");
                input_quant_meta.get(&key).map(|v| (key, v.clone()))
            });

            OutputTensor {
                name: name.clone(),
                data: TensorData::Borrowed(data),
                dtype: view.dtype(),
                shape: shape.to_vec(),
                quant_original_shape: None,
                passthrough_quant: passthrough,
            }
        };

        let tensor_size = output.data.as_slice().len();

        // Flush current shard if adding this tensor would exceed the limit.
        if current_shard_size > 0 && current_shard_size + tensor_size > SHARD_LIMIT {
            write_single_shard(&current_shard, output_dir, output_shard_count, &mut index_weight_map, format)?;
            output_shard_count += 1;
            current_shard.clear();
            current_shard_size = 0;
        }

        current_shard_size += tensor_size;
        current_shard.push(output);
    }

    // Flush the final shard.
    if !current_shard.is_empty() {
        write_single_shard(&current_shard, output_dir, output_shard_count, &mut index_weight_map, format)?;
        output_shard_count += 1;
    }

    // Rename shards now that we know the total count, and write index.
    finalize_shards(output_dir, output_shard_count, &index_weight_map)?;

    // Copy config and tokenizer files.
    copy_support_files(input_dir, output_dir)?;

    // Print summary.
    let ratio = if quantized_bytes > 0 {
        original_bytes as f64 / quantized_bytes as f64
    } else {
        1.0
    };
    info!(
        quantized = quantized_count,
        total = num_tensors,
        shards = output_shard_count,
        "quantized tensors"
    );
    info!(
        "size: {:.1} GB → {:.1} GB ({:.1}x compression)",
        original_bytes as f64 / 1e9,
        quantized_bytes as f64 / 1e9,
        ratio
    );
    info!(output = %output_dir.display(), "output directory");

    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor collection — gathers all tensors across shards in deterministic order.
// ---------------------------------------------------------------------------

fn collect_all_tensors<'a>(
    shards: &'a [SafeTensors<'a>],
    weight_map: &HashMap<String, usize>,
) -> Vec<(String, TensorView<'a>)> {
    if weight_map.is_empty() {
        // Single-file model: iterate the one shard.
        let mut tensors: Vec<_> = shards[0].iter().map(|(n, v)| (n.to_string(), v)).collect();
        tensors.sort_by(|a, b| a.0.cmp(&b.0));
        tensors
    } else {
        // Multi-shard: use weight_map to find each tensor's shard.
        let mut tensors: Vec<(String, TensorView<'a>)> = Vec::new();
        for (name, &shard_idx) in weight_map {
            if let Ok(view) = shards[shard_idx].tensor(name) {
                tensors.push((name.clone(), view));
            }
        }
        tensors.sort_by(|a, b| a.0.cmp(&b.0));
        tensors
    }
}

// ---------------------------------------------------------------------------
// Quantization eligibility — name-based heuristic matching existing loader.
// ---------------------------------------------------------------------------

/// Determine if a tensor should be quantized based on its name, shape, and dtype.
///
/// Criteria (matching the on-load quantization in loader.rs):
///   - 2D weight tensor (not bias), OR 3D fused expert tensor
///   - Name contains a projection keyword (q_proj, k_proj, etc.) or is lm_head
///   - Inner dimension (k) divisible by 32 (Q4 block size)
///   - Not a norm, embedding, conv1d, or router tensor
///   - Source dtype is bf16 (the only format we quantize from)
fn should_quantize(name: &str, shape: &[usize], dtype: safetensors::Dtype) -> bool {
    if dtype != safetensors::Dtype::BF16 {
        return false;
    }

    // Exclude norms, embeddings, conv1d, routers, and Mamba-2 SSM projections.
    // Mamba-2's mixer.in_proj packs SSM-critical signals (dt, B, C, x, z) and
    // mixer.out_proj is the SSM output projection — both are too sensitive to
    // quantization because the recurrent state amplifies per-step errors.
    // Standard attention in_proj/out_proj (e.g. self_attn.in_proj) don't contain
    // "mixer." so they're unaffected.
    let exclude = ["layernorm", "norm", "embed_tokens", "conv1d", "router",
                   "mixer.in_proj", "mixer.out_proj"];
    if exclude.iter().any(|ex| name.contains(ex)) {
        return false;
    }

    match shape.len() {
        2 => {
            // Standard 2D weight: [m, k].
            if !name.ends_with(".weight") {
                return false;
            }
            let k = shape[1];
            if k % 32 != 0 {
                return false;
            }
            let proj_names = [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj", "out_proj", "lm_head",
                // Mixtral MoE expert weights (w1=gate, w2=down, w3=up).
                "w1", "w2", "w3",
            ];
            proj_names.iter().any(|p| name.contains(p))
        }
        3 => {
            // Fused expert tensor: [num_experts, rows, k].
            // Qwen3.5 stores gate_up_proj [num_experts, 2*moe_inter, hidden] and
            // down_proj [num_experts, hidden, moe_inter] as 3D tensors without
            // a .weight suffix.  Q4 quantization is per-row, so flattening the
            // first two dims to [num_experts * rows, k] works correctly.
            let k = shape[2];
            if k % 32 != 0 {
                return false;
            }
            name.contains("experts") && (name.ends_with("gate_up_proj") || name.ends_with("down_proj"))
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// MXFP4 detection and dequant→requant (GPT-OSS pre-quantized experts).
//
// MXFP4 experts are stored as triplets: _blocks (packed FP4 nibbles),
// _scales (E8M0), and optional _bias (bf16).  We detect these groups,
// dequant to bf16 per-expert (bounding RAM), then re-quantize to the
// target format.  This produces standard quantized weight tensors with
// the same naming convention as other models.
// ---------------------------------------------------------------------------

/// Detect MXFP4 expert tensor groups by finding _blocks tensors.
///
/// Returns the set of base prefixes (e.g.
/// "model.layers.0.mlp.experts.gate_up_proj") that have matching
/// _blocks + _scales pairs.
fn detect_mxfp4_groups(all_tensors: &[(String, TensorView)]) -> std::collections::HashSet<String> {
    let mut blocks_bases = std::collections::HashSet::new();
    let mut scales_bases = std::collections::HashSet::new();

    for (name, _) in all_tensors {
        if let Some(base) = name.strip_suffix("_blocks") {
            blocks_bases.insert(base.to_string());
        }
        if let Some(base) = name.strip_suffix("_scales") {
            scales_bases.insert(base.to_string());
        }
    }

    // Only keep bases that have both _blocks and _scales.
    blocks_bases.intersection(&scales_bases).cloned().collect()
}

/// Dequant one MXFP4 group (gate_up_proj or down_proj) and re-quantize.
///
/// For a base like "model.layers.0.mlp.experts.gate_up_proj":
///   - Reads _blocks and _scales tensors
///   - Dequants per-expert to bf16 (one expert at a time to bound RAM)
///   - For gate_up_proj: de-interleaves into separate gate and up tensors
///   - Re-quantizes each bf16 weight to the target format
///   - Emits OutputTensor entries with standard weight names
///
/// Biases are passed through as bf16 (they're already small).
fn dequant_requant_mxfp4_group<'a>(
    base: &str,
    tensor_by_name: &HashMap<&str, &safetensors::tensor::TensorView<'a>>,
    quant: &dyn WeightQuantiser,
    format: QuantFormat,
    original_bytes: &mut u64,
    quantized_bytes: &mut u64,
    quantized_count: &mut usize,
    consumed: &mut std::collections::HashSet<String>,
) -> anyhow::Result<Vec<OutputTensor<'a>>> {
    let blocks_name = format!("{base}_blocks");
    let scales_name = format!("{base}_scales");
    let bias_name = format!("{base}_bias");

    let blocks_view = tensor_by_name.get(blocks_name.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing MXFP4 _blocks tensor: {blocks_name}"))?;
    let scales_view = tensor_by_name.get(scales_name.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing MXFP4 _scales tensor: {scales_name}"))?;

    consumed.insert(blocks_name.clone());
    consumed.insert(scales_name.clone());

    let blocks_shape = blocks_view.shape();
    let scales_shape = scales_view.shape();
    let blocks_data = blocks_view.data();
    let scales_data = scales_view.data();

    // Infer dimensions from block shapes.
    // blocks: [num_experts, rows, num_scale_blocks, 16] (packed nibbles)
    // scales: [num_experts, rows, num_scale_blocks]     (E8M0 bytes)
    anyhow::ensure!(
        blocks_shape.len() == 4 && scales_shape.len() == 3,
        "unexpected MXFP4 shapes: blocks={blocks_shape:?}, scales={scales_shape:?}"
    );

    let num_experts = blocks_shape[0];
    let rows = blocks_shape[1];
    let num_scale_blocks = blocks_shape[2];
    let block_size = 32usize; // MXFP4 standard
    let cols = num_scale_blocks * block_size;

    // Per-expert byte sizes for slicing the fused tensors.
    let blocks_per_expert = rows * num_scale_blocks * 16; // packed nibbles
    let scales_per_expert = rows * num_scale_blocks;

    // Determine if this is a gate_up (interleaved) or down projection.
    let is_gate_up = base.ends_with("gate_up_proj");

    // Build the layer prefix for output tensor naming.
    // e.g. "model.layers.0.mlp.experts.gate_up_proj" → "model.layers.0.mlp.experts"
    let expert_prefix = base.rsplit_once('.').map(|(p, _)| p).unwrap_or(base);

    let mut outputs = Vec::new();

    // Optional bias — pass through as bf16, split per-expert.
    let bias_data = tensor_by_name.get(bias_name.as_str()).map(|v| {
        consumed.insert(bias_name.clone());
        v.data().to_vec()
    });
    let bias_per_expert = rows * 2; // bf16

    // Track original size as the bf16 equivalent (not packed MXFP4 bytes),
    // so the compression ratio reflects bf16 → target format.
    let bf16_expert_bytes = (rows * cols * 2) as u64; // per expert, bf16
    *original_bytes += bf16_expert_bytes * num_experts as u64;
    if let Some(ref b) = bias_data {
        *original_bytes += b.len() as u64;
    }

    info!(
        "dequant MXFP4: {} ({} experts, {}×{} per expert)",
        base, num_experts, rows, cols
    );

    for j in 0..num_experts {
        let b_off = j * blocks_per_expert;
        let s_off = j * scales_per_expert;

        // Dequant this expert's weights to bf16.
        let bf16_data = dequantize_mxfp4(
            &blocks_data[b_off..b_off + blocks_per_expert],
            &scales_data[s_off..s_off + scales_per_expert],
            rows,
            cols,
            block_size,
        );

        if is_gate_up {
            // De-interleave gate and up from fused [2*moe_inter, hidden].
            // Even rows = gate, odd rows = up.
            let moe_inter = rows / 2;
            let row_bytes = cols * 2; // bf16

            let mut gate_raw = vec![0u8; moe_inter * row_bytes];
            let mut up_raw = vec![0u8; moe_inter * row_bytes];
            for r in 0..moe_inter {
                let even_start = (2 * r) * row_bytes;
                let odd_start = (2 * r + 1) * row_bytes;
                gate_raw[r * row_bytes..(r + 1) * row_bytes]
                    .copy_from_slice(&bf16_data[even_start..even_start + row_bytes]);
                up_raw[r * row_bytes..(r + 1) * row_bytes]
                    .copy_from_slice(&bf16_data[odd_start..odd_start + row_bytes]);
            }

            // Re-quantize gate and up separately.
            let gate_quant = quant.quantise(&gate_raw, moe_inter, cols);
            let up_quant = quant.quantise(&up_raw, moe_inter, cols);

            *quantized_bytes += gate_quant.len() as u64 + up_quant.len() as u64;
            *quantized_count += 2;

            let gate_name = format!("{expert_prefix}.{j}.gate_proj.weight");
            let up_name = format!("{expert_prefix}.{j}.up_proj.weight");

            outputs.push(OutputTensor {
                name: gate_name,
                data: TensorData::Owned(gate_quant),
                dtype: safetensors::Dtype::U8,
                shape: vec![quant.byte_count(moe_inter, cols)],
                quant_original_shape: Some((moe_inter, cols)),
                passthrough_quant: None,
            });
            outputs.push(OutputTensor {
                name: up_name,
                data: TensorData::Owned(up_quant),
                dtype: safetensors::Dtype::U8,
                shape: vec![quant.byte_count(moe_inter, cols)],
                quant_original_shape: Some((moe_inter, cols)),
                passthrough_quant: None,
            });

            // Biases — de-interleave and pass through as bf16.
            if let Some(ref bias) = bias_data {
                let off = j * bias_per_expert;
                let bias_slice = &bias[off..off + bias_per_expert];
                let bias_bf16: &[u16] = bytemuck::cast_slice(bias_slice);
                let gate_vals: Vec<u16> = (0..moe_inter).map(|i| bias_bf16[2 * i]).collect();
                let up_vals: Vec<u16> = (0..moe_inter).map(|i| bias_bf16[2 * i + 1]).collect();
                let gate_bytes: Vec<u8> = bytemuck::cast_slice(&gate_vals).to_vec();
                let up_bytes: Vec<u8> = bytemuck::cast_slice(&up_vals).to_vec();

                *quantized_bytes += gate_bytes.len() as u64 + up_bytes.len() as u64;

                outputs.push(OutputTensor {
                    name: format!("{expert_prefix}.{j}.gate_proj.bias"),
                    data: TensorData::Owned(gate_bytes.clone()),
                    dtype: safetensors::Dtype::BF16,
                    shape: vec![moe_inter],
                    quant_original_shape: None,
                    passthrough_quant: None,
                });
                outputs.push(OutputTensor {
                    name: format!("{expert_prefix}.{j}.up_proj.bias"),
                    data: TensorData::Owned(up_bytes.clone()),
                    dtype: safetensors::Dtype::BF16,
                    shape: vec![moe_inter],
                    quant_original_shape: None,
                    passthrough_quant: None,
                });
            }
        } else {
            // down_proj: [hidden, moe_inter] — no de-interleaving needed.
            let quant_data = quant.quantise(&bf16_data, rows, cols);

            *quantized_bytes += quant_data.len() as u64;
            *quantized_count += 1;

            outputs.push(OutputTensor {
                name: format!("{expert_prefix}.{j}.down_proj.weight"),
                data: TensorData::Owned(quant_data),
                dtype: safetensors::Dtype::U8,
                shape: vec![quant.byte_count(rows, cols)],
                quant_original_shape: Some((rows, cols)),
                passthrough_quant: None,
            });

            // Down bias — pass through as bf16.
            if let Some(ref bias) = bias_data {
                let off = j * bias_per_expert;
                let bias_slice = bias[off..off + bias_per_expert].to_vec();
                *quantized_bytes += bias_slice.len() as u64;

                outputs.push(OutputTensor {
                    name: format!("{expert_prefix}.{j}.down_proj.bias"),
                    data: TensorData::Owned(bias_slice),
                    dtype: safetensors::Dtype::BF16,
                    shape: vec![rows],
                    quant_original_shape: None,
                    passthrough_quant: None,
                });
            }
        }
    }

    Ok(outputs)
}

// ---------------------------------------------------------------------------
// Output writing — serialize quantized + passthrough tensors to safetensors.
// ---------------------------------------------------------------------------

enum TensorData<'a> {
    Owned(Vec<u8>),
    Borrowed(&'a [u8]),
}

impl<'a> TensorData<'a> {
    fn as_slice(&self) -> &[u8] {
        match self {
            TensorData::Owned(v) => v,
            TensorData::Borrowed(s) => s,
        }
    }
}

struct OutputTensor<'a> {
    name: String,
    data: TensorData<'a>,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    /// If Some, this tensor was quantized from a [m, k] bf16 weight.
    quant_original_shape: Option<(usize, usize)>,
    /// Pre-existing quantization metadata to propagate (e.g. "rllm_q4:name" = "m,k"
    /// from an input model that was already quantized in a different format).
    passthrough_quant: Option<(String, String)>,
}

/// Write one shard of tensors to disk.  Called incrementally as the shard fills.
///
/// Uses a temporary filename (`.tmp.{shard_idx}`) because we don't know the total
/// shard count yet.  `finalize_shards()` renames to the final pattern afterwards.
fn write_single_shard(
    tensors: &[OutputTensor],
    output_dir: &std::path::Path,
    shard_idx: usize,
    index_weight_map: &mut Vec<(String, usize)>,
    format: QuantFormat,
) -> anyhow::Result<()> {
    let mut metadata: HashMap<String, String> = HashMap::new();
    metadata.insert("quantization".to_string(), format.metadata_tag().to_string());

    let mut views: Vec<(String, TensorView)> = Vec::with_capacity(tensors.len());

    for t in tensors {
        if let Some((m, k)) = t.quant_original_shape {
            metadata.insert(format!("{}{}", format.metadata_prefix(), t.name), format!("{m},{k}"));
        }
        if let Some((ref key, ref val)) = t.passthrough_quant {
            metadata.insert(key.clone(), val.clone());
        }

        let view = TensorView::new(t.dtype, t.shape.clone(), t.data.as_slice())
            .map_err(|e| anyhow::anyhow!("failed to create TensorView for '{}': {e}", t.name))?;
        views.push((t.name.clone(), view));

        index_weight_map.push((t.name.clone(), shard_idx));
    }

    let tmp_name = format!(".tmp.{shard_idx}.safetensors");
    let output_path = output_dir.join(&tmp_name);
    info!(shard = shard_idx + 1, tensors = tensors.len(), "writing shard");

    safetensors::tensor::serialize_to_file(views, &Some(metadata), &output_path)
        .map_err(|e| anyhow::anyhow!("failed to write shard {shard_idx}: {e}"))?;

    Ok(())
}

/// Rename temporary shard files to final names and write the index file.
fn finalize_shards(
    output_dir: &std::path::Path,
    num_shards: usize,
    index_weight_map: &[(String, usize)],
) -> anyhow::Result<()> {
    for shard_idx in 0..num_shards {
        let tmp_name = format!(".tmp.{shard_idx}.safetensors");
        let final_name = if num_shards == 1 {
            "model.safetensors".to_string()
        } else {
            format!("model-{:05}-of-{:05}.safetensors", shard_idx + 1, num_shards)
        };
        std::fs::rename(
            output_dir.join(&tmp_name),
            output_dir.join(&final_name),
        )?;
    }

    if num_shards > 1 {
        let mut weight_map = serde_json::Map::new();
        for (name, &shard_idx) in index_weight_map.iter().map(|(n, s)| (n, s)) {
            let filename = format!("model-{:05}-of-{:05}.safetensors", shard_idx + 1, num_shards);
            weight_map.insert(name.clone(), serde_json::Value::String(filename));
        }
        let index = serde_json::json!({ "weight_map": weight_map });
        let index_path = output_dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string_pretty(&index)?)?;
        info!("wrote index file");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Copy support files (config, tokenizer) to the output directory.
// ---------------------------------------------------------------------------

fn copy_support_files(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
) -> anyhow::Result<()> {
    let files = ["config.json", "tokenizer.json", "tokenizer_config.json"];
    for name in &files {
        let src = input_dir.join(name);
        if src.exists() {
            std::fs::copy(&src, output_dir.join(name))?;
            debug!(file = name, "copied");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_quantize_q_proj() {
        assert!(should_quantize(
            "model.layers.0.self_attn.q_proj.weight",
            &[2048, 2048],
            safetensors::Dtype::BF16,
        ));
    }

    #[test]
    fn test_should_not_quantize_norm() {
        assert!(!should_quantize(
            "model.layers.0.input_layernorm.weight",
            &[2048],
            safetensors::Dtype::BF16,
        ));
    }

    #[test]
    fn test_should_not_quantize_embed() {
        assert!(!should_quantize(
            "model.embed_tokens.weight",
            &[128256, 2048],
            safetensors::Dtype::BF16,
        ));
    }

    #[test]
    fn test_should_not_quantize_bias() {
        assert!(!should_quantize(
            "model.layers.0.self_attn.q_proj.bias",
            &[2048],
            safetensors::Dtype::BF16,
        ));
    }

    #[test]
    fn test_should_not_quantize_f32() {
        assert!(!should_quantize(
            "model.layers.0.self_attn.q_proj.weight",
            &[2048, 2048],
            safetensors::Dtype::F32,
        ));
    }

    #[test]
    fn test_should_quantize_lm_head() {
        assert!(should_quantize(
            "lm_head.weight",
            &[128256, 4096],
            safetensors::Dtype::BF16,
        ));
    }

    #[test]
    fn test_should_not_quantize_k_not_div_32() {
        assert!(!should_quantize(
            "model.layers.0.self_attn.q_proj.weight",
            &[2048, 100],
            safetensors::Dtype::BF16,
        ));
    }

    #[test]
    fn test_should_not_quantize_router() {
        assert!(!should_quantize(
            "model.layers.0.mlp.router.weight",
            &[128, 2048],
            safetensors::Dtype::BF16,
        ));
    }

    #[test]
    fn test_should_quantize_gate_proj() {
        assert!(should_quantize(
            "model.layers.0.mlp.gate_proj.weight",
            &[8192, 2048],
            safetensors::Dtype::BF16,
        ));
    }

    // Mamba-2 SSM projection exclusions (Nemotron-H).
    // These tensors pack SSM-critical signals (dt, B, C, x, z) that are
    // too sensitive to quantization — the recurrent state amplifies errors.

    #[test]
    fn test_should_not_quantize_mamba_mixer_in_proj() {
        assert!(!should_quantize(
            "backbone.layers.0.mixer.in_proj.weight",
            &[10304, 2688],
            safetensors::Dtype::BF16,
        ));
    }

    #[test]
    fn test_should_not_quantize_mamba_mixer_out_proj() {
        assert!(!should_quantize(
            "backbone.layers.0.mixer.out_proj.weight",
            &[2688, 4096],
            safetensors::Dtype::BF16,
        ));
    }

    // Ensure attention in_proj/out_proj (non-mixer) are still quantized.
    #[test]
    fn test_should_quantize_attention_in_proj() {
        assert!(should_quantize(
            "model.layers.0.self_attn.in_proj.weight",
            &[4096, 2048],
            safetensors::Dtype::BF16,
        ));
    }

    #[test]
    fn test_should_quantize_attention_out_proj() {
        assert!(should_quantize(
            "model.layers.0.self_attn.out_proj.weight",
            &[2048, 4096],
            safetensors::Dtype::BF16,
        ));
    }

    // --- End-to-end tests ---

    /// Build a minimal bf16 safetensors file with the given tensors.
    fn make_bf16_safetensors(tensors: &[(&str, Vec<usize>)]) -> Vec<u8> {
        use half::bf16;

        let mut views_data: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
        for (name, shape) in tensors {
            let n_elements: usize = shape.iter().product();
            // Fill with a simple pattern so we can verify round-trip correctness.
            let values: Vec<bf16> = (0..n_elements)
                .map(|i| bf16::from_f32((i % 17) as f32 * 0.1 - 0.8))
                .collect();
            let bytes: Vec<u8> = bytemuck::cast_slice(&values).to_vec();
            views_data.push((name.to_string(), bytes, shape.clone()));
        }

        let tv_refs: Vec<(String, TensorView)> = views_data
            .iter()
            .map(|(name, data, shape)| {
                let tv = TensorView::new(safetensors::Dtype::BF16, shape.clone(), data).unwrap();
                (name.clone(), tv)
            })
            .collect();

        safetensors::serialize(tv_refs, &None).unwrap()
    }

    /// Create a minimal model directory with config.json and a safetensors file.
    fn setup_test_model(dir: &std::path::Path, tensors: &[(&str, Vec<usize>)]) {
        let st_bytes = make_bf16_safetensors(tensors);
        std::fs::write(dir.join("model.safetensors"), st_bytes).unwrap();

        // Minimal config.json (just enough fields to exist — quantize doesn't parse it).
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","hidden_size":64,"num_hidden_layers":1,"num_attention_heads":2,"vocab_size":128}"#,
        )
        .unwrap();

        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    }

    #[test]
    fn test_end_to_end_quantize_produces_valid_output() {
        let input_dir = tempfile::tempdir().unwrap();
        let output_dir = tempfile::tempdir().unwrap();

        // Platform-aware: `--format q4` may produce NVFP4 on Blackwell.
        let actual_format = super::platform_adjusted_format(QuantFormat::Q4);

        // Create a model with one quantizable tensor and one passthrough tensor.
        let m = 4;
        let k = 64; // divisible by 32
        setup_test_model(
            input_dir.path(),
            &[
                ("model.layers.0.self_attn.q_proj.weight", vec![m, k]),
                ("model.layers.0.input_layernorm.weight", vec![64]),
            ],
        );

        // Run quantize.
        exec(QuantizeArgs {
            model: input_dir.path().to_path_buf(),
            output: output_dir.path().to_path_buf(),
            format: "q4".to_string(),
        })
        .unwrap();

        // Verify output file exists.
        let output_st = output_dir.path().join("model.safetensors");
        assert!(output_st.exists(), "output safetensors file should exist");

        // Verify support files were copied.
        assert!(output_dir.path().join("config.json").exists());
        assert!(output_dir.path().join("tokenizer.json").exists());

        // Parse the output safetensors and check metadata.
        let data = std::fs::read(&output_st).unwrap();
        let (_, metadata) = SafeTensors::read_metadata(&data).unwrap();
        let meta = metadata.metadata().as_ref().unwrap();

        assert_eq!(meta.get("quantization").unwrap(), actual_format.metadata_tag());
        let meta_key = format!("{}model.layers.0.self_attn.q_proj.weight", actual_format.metadata_prefix());
        assert_eq!(meta.get(&meta_key).unwrap(), &format!("{m},{k}"));
        // Norm tensor should NOT have a quantization metadata entry.
        let norm_key = format!("{}model.layers.0.input_layernorm.weight", actual_format.metadata_prefix());
        assert!(meta.get(&norm_key).is_none());

        // Verify the quantized tensor has the correct byte count.
        let st = SafeTensors::deserialize(&data).unwrap();
        let q_proj = st
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q_proj.dtype(), safetensors::Dtype::U8);
        let quant = crate::gpu::ops::quant::quantiser(actual_format);
        assert_eq!(q_proj.data().len(), quant.byte_count(m, k));

        // Verify the norm tensor passed through as bf16 unchanged.
        let norm = st
            .tensor("model.layers.0.input_layernorm.weight")
            .unwrap();
        assert_eq!(norm.dtype(), safetensors::Dtype::BF16);
        assert_eq!(norm.shape(), &[64]);
    }

    #[test]
    fn test_end_to_end_quantize_matches_on_load_quantize() {
        // Verify that pre-quantized bytes are identical to on-load quantization.
        use half::bf16;

        let actual_format = super::platform_adjusted_format(QuantFormat::Q4);
        let quant = crate::gpu::ops::quant::quantiser(actual_format);

        let m = 2;
        let k = 32;
        let n_elements = m * k;
        let values: Vec<bf16> = (0..n_elements)
            .map(|i| bf16::from_f32((i % 17) as f32 * 0.1 - 0.8))
            .collect();
        let bf16_bytes: &[u8] = bytemuck::cast_slice(&values);

        // On-load path: quantize directly.
        let on_load = quant.quantise(bf16_bytes, m, k);

        // Offline path: build safetensors, run quantize, read back.
        let input_dir = tempfile::tempdir().unwrap();
        let output_dir = tempfile::tempdir().unwrap();

        setup_test_model(
            input_dir.path(),
            &[("model.layers.0.self_attn.q_proj.weight", vec![m, k])],
        );

        exec(QuantizeArgs {
            model: input_dir.path().to_path_buf(),
            output: output_dir.path().to_path_buf(),
            format: "q4".to_string(),
        })
        .unwrap();

        let data = std::fs::read(output_dir.path().join("model.safetensors")).unwrap();
        let st = SafeTensors::deserialize(&data).unwrap();
        let offline = st
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        assert_eq!(
            offline.data(),
            on_load.as_slice(),
            "offline and on-load quantization must produce identical bytes (format: {})",
            actual_format.name(),
        );
    }

    #[test]
    fn test_end_to_end_quantize_multiple_tensors() {
        let input_dir = tempfile::tempdir().unwrap();
        let output_dir = tempfile::tempdir().unwrap();

        let actual_format = super::platform_adjusted_format(QuantFormat::Q4);

        // Several quantizable + non-quantizable tensors.
        setup_test_model(
            input_dir.path(),
            &[
                ("model.layers.0.self_attn.q_proj.weight", vec![8, 64]),
                ("model.layers.0.self_attn.k_proj.weight", vec![4, 64]),
                ("model.layers.0.self_attn.v_proj.weight", vec![4, 64]),
                ("model.layers.0.self_attn.o_proj.weight", vec![8, 64]),
                ("model.layers.0.mlp.gate_proj.weight", vec![16, 64]),
                ("model.layers.0.mlp.up_proj.weight", vec![16, 64]),
                ("model.layers.0.mlp.down_proj.weight", vec![8, 64]),
                ("model.layers.0.input_layernorm.weight", vec![64]),
                ("model.layers.0.post_attention_layernorm.weight", vec![64]),
                ("model.embed_tokens.weight", vec![128, 64]),
            ],
        );

        exec(QuantizeArgs {
            model: input_dir.path().to_path_buf(),
            output: output_dir.path().to_path_buf(),
            format: "q4".to_string(),
        })
        .unwrap();

        let data = std::fs::read(output_dir.path().join("model.safetensors")).unwrap();
        let (_, metadata) = SafeTensors::read_metadata(&data).unwrap();
        let meta = metadata.metadata().as_ref().unwrap();
        let st = SafeTensors::deserialize(&data).unwrap();

        // Count quantized tensors from metadata.
        let prefix = actual_format.metadata_prefix();
        let quant_count = meta.keys().filter(|k| k.starts_with(prefix)).count();
        assert_eq!(quant_count, 7, "7 projection weights should be quantized");

        // Verify all tensors are present and accessible.
        assert_eq!(st.len(), 10);

        // Spot-check: embed_tokens should remain bf16.
        let embed = st.tensor("model.embed_tokens.weight").unwrap();
        assert_eq!(embed.dtype(), safetensors::Dtype::BF16);

        // Spot-check: gate_proj should be U8 (quantized).
        let gate = st
            .tensor("model.layers.0.mlp.gate_proj.weight")
            .unwrap();
        assert_eq!(gate.dtype(), safetensors::Dtype::U8);
    }

    #[test]
    fn test_end_to_end_quantize_q8_produces_valid_output() {
        use crate::gpu::ops::quant::QuantFormat;

        let input_dir = tempfile::tempdir().unwrap();
        let output_dir = tempfile::tempdir().unwrap();

        let m = 4;
        let k = 64;
        setup_test_model(
            input_dir.path(),
            &[
                ("model.layers.0.self_attn.q_proj.weight", vec![m, k]),
                ("model.layers.0.input_layernorm.weight", vec![64]),
            ],
        );

        exec(QuantizeArgs {
            model: input_dir.path().to_path_buf(),
            output: output_dir.path().to_path_buf(),
            format: "q8".to_string(),
        })
        .unwrap();

        // Platform-aware: `--format q8` may produce FP8 on NVIDIA SM 89+.
        let actual_format = super::platform_adjusted_format(QuantFormat::Q8);

        let output_st = output_dir.path().join("model.safetensors");
        let data = std::fs::read(&output_st).unwrap();
        let (_, metadata) = SafeTensors::read_metadata(&data).unwrap();
        let meta = metadata.metadata().as_ref().unwrap();

        assert_eq!(meta.get("quantization").unwrap(), actual_format.metadata_tag());
        let meta_key = format!("{}model.layers.0.self_attn.q_proj.weight", actual_format.metadata_prefix());
        assert!(meta.get(&meta_key).is_some(), "expected per-tensor metadata at {meta_key}");
        // No Q4 metadata should be present.
        assert!(
            meta.get("rllm_q4:model.layers.0.self_attn.q_proj.weight")
                .is_none()
        );

        let st = SafeTensors::deserialize(&data).unwrap();
        let q_proj = st
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q_proj.dtype(), safetensors::Dtype::U8);
        let expected_bytes = match actual_format {
            QuantFormat::FP8 => crate::gpu::fp8_byte_count(m, k),
            _ => crate::gpu::q8_byte_count(m, k),
        };
        assert_eq!(q_proj.data().len(), expected_bytes);
    }

    #[test]
    fn test_end_to_end_quantize_q8_matches_on_load() {
        use crate::gpu::ops::quant::QuantFormat;
        use half::bf16;

        let m = 2;
        let k = 32;
        let values: Vec<bf16> = (0..m * k)
            .map(|i| bf16::from_f32((i % 17) as f32 * 0.1 - 0.8))
            .collect();
        let bf16_bytes: &[u8] = bytemuck::cast_slice(&values);

        // Platform-aware: `--format q8` may produce FP8 on NVIDIA SM 89+.
        let actual_format = super::platform_adjusted_format(QuantFormat::Q8);
        let on_load_quantized = match actual_format {
            QuantFormat::FP8 => crate::gpu::quantize_bf16_to_fp8(bf16_bytes, m, k),
            _ => crate::gpu::quantize_bf16_to_q8(bf16_bytes, m, k),
        };

        let input_dir = tempfile::tempdir().unwrap();
        let output_dir = tempfile::tempdir().unwrap();

        setup_test_model(
            input_dir.path(),
            &[("model.layers.0.self_attn.q_proj.weight", vec![m, k])],
        );

        exec(QuantizeArgs {
            model: input_dir.path().to_path_buf(),
            output: output_dir.path().to_path_buf(),
            format: "q8".to_string(),
        })
        .unwrap();

        let data = std::fs::read(output_dir.path().join("model.safetensors")).unwrap();
        let st = SafeTensors::deserialize(&data).unwrap();
        let offline = st
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        assert_eq!(
            offline.data(),
            on_load_quantized.as_slice(),
            "offline and on-load quantization must produce identical bytes (format: {})",
            actual_format.name(),
        );
    }

    // --- MXFP4 dequant→requant tests ---

    /// Build a fake MXFP4 expert block: pack bf16 values into FP4 E2M1 + E8M0 scales.
    ///
    /// This mimics the on-disk format of GPT-OSS expert weights. We round-trip
    /// through the FP4 encoding so the test can verify dequant→requant correctness.
    fn make_mxfp4_blocks_and_scales(
        values_bf16: &[half::bf16],
        rows: usize,
        cols: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        use crate::model::loader::mxfp4::FP4_E2M1_LUT;

        let block_size = 32usize;
        let num_scale_blocks = (cols + block_size - 1) / block_size;

        let mut blocks = vec![0u8; rows * (cols / 2)];
        let mut scales = vec![0u8; rows * num_scale_blocks];

        for r in 0..rows {
            for sb in 0..num_scale_blocks {
                // Find max absolute value in this block to compute E8M0 scale.
                let col_start = sb * block_size;
                let col_end = (col_start + block_size).min(cols);
                let mut max_abs: f32 = 0.0;
                for c in col_start..col_end {
                    max_abs = max_abs.max(values_bf16[r * cols + c].to_f32().abs());
                }

                // Encode scale as E8M0: find exponent such that 2^(e-127) >= max_abs/6.0.
                let scale_f32 = if max_abs == 0.0 { 1.0 } else { max_abs / 6.0 };
                let scale_exp = if scale_f32 <= 0.0 {
                    0u8
                } else {
                    let bits = scale_f32.to_bits();
                    ((bits >> 23) & 0xFF) as u8
                };
                scales[r * num_scale_blocks + sb] = scale_exp;
                let scale = crate::model::loader::mxfp4::e8m0_to_f32(scale_exp);

                // Quantize each value to nearest FP4 E2M1 nibble.
                for c in col_start..col_end {
                    let val = values_bf16[r * cols + c].to_f32();
                    let scaled = if scale > 0.0 { val / scale } else { 0.0 };
                    // Find nearest LUT entry.
                    let mut best_nibble = 0u8;
                    let mut best_dist = f32::MAX;
                    for (i, &lut_val) in FP4_E2M1_LUT.iter().enumerate() {
                        let dist = (scaled - lut_val).abs();
                        if dist < best_dist {
                            best_dist = dist;
                            best_nibble = i as u8;
                        }
                    }
                    // Pack nibble.
                    let byte_idx = r * (cols / 2) + c / 2;
                    if c % 2 == 0 {
                        blocks[byte_idx] |= best_nibble;
                    } else {
                        blocks[byte_idx] |= best_nibble << 4;
                    }
                }
            }
        }

        (blocks, scales)
    }

    /// Create a model directory with MXFP4 expert tensors (GPT-OSS format).
    fn setup_mxfp4_test_model(dir: &std::path::Path) {
        use half::bf16;

        let num_experts = 2usize;
        let moe_inter = 32usize; // Small for testing, must be divisible by 32.
        let hidden = 64usize;    // Must be divisible by 32.

        // Generate deterministic bf16 values for gate_up (interleaved) and down.
        let gu_rows = 2 * moe_inter;
        let gu_elements = num_experts * gu_rows * hidden;
        let gu_values: Vec<bf16> = (0..gu_elements)
            .map(|i| bf16::from_f32((i % 23) as f32 * 0.15 - 1.5))
            .collect();

        let down_rows = hidden;
        let down_elements = num_experts * down_rows * moe_inter;
        let down_values: Vec<bf16> = (0..down_elements)
            .map(|i| bf16::from_f32((i % 19) as f32 * 0.12 - 1.0))
            .collect();

        // Build MXFP4 packed blocks + scales for each expert.
        let mut all_gu_blocks = Vec::new();
        let mut all_gu_scales = Vec::new();
        let mut all_down_blocks = Vec::new();
        let mut all_down_scales = Vec::new();
        let mut all_gu_bias = Vec::new();
        let mut all_down_bias = Vec::new();

        let block_size = 32;
        let gu_num_sb = (hidden + block_size - 1) / block_size;
        let down_num_sb = (moe_inter + block_size - 1) / block_size;

        for j in 0..num_experts {
            let gu_start = j * gu_rows * hidden;
            let (gu_b, gu_s) = make_mxfp4_blocks_and_scales(
                &gu_values[gu_start..gu_start + gu_rows * hidden],
                gu_rows, hidden,
            );
            all_gu_blocks.extend_from_slice(&gu_b);
            all_gu_scales.extend_from_slice(&gu_s);

            let down_start = j * down_rows * moe_inter;
            let (down_b, down_s) = make_mxfp4_blocks_and_scales(
                &down_values[down_start..down_start + down_rows * moe_inter],
                down_rows, moe_inter,
            );
            all_down_blocks.extend_from_slice(&down_b);
            all_down_scales.extend_from_slice(&down_s);

            // Interleaved bias: [gate[0], up[0], gate[1], up[1], ...]
            let bias_vals: Vec<bf16> = (0..gu_rows)
                .map(|i| bf16::from_f32(i as f32 * 0.01))
                .collect();
            all_gu_bias.extend_from_slice(bytemuck::cast_slice::<bf16, u8>(&bias_vals));

            let down_bias_vals: Vec<bf16> = (0..down_rows)
                .map(|i| bf16::from_f32(i as f32 * -0.005))
                .collect();
            all_down_bias.extend_from_slice(bytemuck::cast_slice::<bf16, u8>(&down_bias_vals));
        }

        // Build safetensors with MXFP4 tensors + attention weights.
        let tensors: Vec<(String, Vec<u8>, safetensors::Dtype, Vec<usize>)> = vec![
            (
                "model.layers.0.mlp.experts.gate_up_proj_blocks".into(),
                all_gu_blocks,
                safetensors::Dtype::U8,
                vec![num_experts, gu_rows, gu_num_sb, 16],
            ),
            (
                "model.layers.0.mlp.experts.gate_up_proj_scales".into(),
                all_gu_scales,
                safetensors::Dtype::U8,
                vec![num_experts, gu_rows, gu_num_sb],
            ),
            (
                "model.layers.0.mlp.experts.gate_up_proj_bias".into(),
                all_gu_bias,
                safetensors::Dtype::BF16,
                vec![num_experts, gu_rows],
            ),
            (
                "model.layers.0.mlp.experts.down_proj_blocks".into(),
                all_down_blocks,
                safetensors::Dtype::U8,
                vec![num_experts, down_rows, down_num_sb, 16],
            ),
            (
                "model.layers.0.mlp.experts.down_proj_scales".into(),
                all_down_scales,
                safetensors::Dtype::U8,
                vec![num_experts, down_rows, down_num_sb],
            ),
            (
                "model.layers.0.mlp.experts.down_proj_bias".into(),
                all_down_bias,
                safetensors::Dtype::BF16,
                vec![num_experts, down_rows],
            ),
            // A standard bf16 attention weight (should be quantized normally).
            {
                let m = 4; let k = 64;
                let vals: Vec<bf16> = (0..m*k).map(|i| bf16::from_f32((i % 17) as f32 * 0.1 - 0.8)).collect();
                (
                    "model.layers.0.self_attn.q_proj.weight".into(),
                    bytemuck::cast_slice(&vals).to_vec(),
                    safetensors::Dtype::BF16,
                    vec![m, k],
                )
            },
            // Norm (passthrough).
            {
                let vals: Vec<bf16> = (0..64).map(|i| bf16::from_f32(i as f32 * 0.01)).collect();
                (
                    "model.layers.0.input_layernorm.weight".into(),
                    bytemuck::cast_slice(&vals).to_vec(),
                    safetensors::Dtype::BF16,
                    vec![64],
                )
            },
        ];

        let tv_refs: Vec<(String, TensorView)> = tensors
            .iter()
            .map(|(name, data, dtype, shape)| {
                let tv = TensorView::new(*dtype, shape.clone(), data).unwrap();
                (name.clone(), tv)
            })
            .collect();

        let st_bytes = safetensors::serialize(tv_refs, &None).unwrap();
        std::fs::write(dir.join("model.safetensors"), st_bytes).unwrap();

        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"gpt-oss","hidden_size":64,"num_hidden_layers":1,"num_attention_heads":2,"vocab_size":128}"#,
        ).unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    }

    #[test]
    fn test_mxfp4_dequant_requant_produces_per_expert_tensors() {
        let input_dir = tempfile::tempdir().unwrap();
        let output_dir = tempfile::tempdir().unwrap();

        setup_mxfp4_test_model(input_dir.path());

        exec(QuantizeArgs {
            model: input_dir.path().to_path_buf(),
            output: output_dir.path().to_path_buf(),
            format: "q4".to_string(),
        })
        .unwrap();

        let output_st = output_dir.path().join("model.safetensors");
        assert!(output_st.exists());

        let data = std::fs::read(&output_st).unwrap();
        let st = SafeTensors::deserialize(&data).unwrap();
        let (_, metadata) = SafeTensors::read_metadata(&data).unwrap();
        let meta = metadata.metadata().as_ref().unwrap();

        let actual_format = super::platform_adjusted_format(QuantFormat::Q4);

        // Verify per-expert tensors were created.
        for j in 0..2 {
            let gate_name = format!("model.layers.0.mlp.experts.{j}.gate_proj.weight");
            let up_name = format!("model.layers.0.mlp.experts.{j}.up_proj.weight");
            let down_name = format!("model.layers.0.mlp.experts.{j}.down_proj.weight");

            // Weights should exist as quantized U8.
            let gate = st.tensor(&gate_name).unwrap();
            assert_eq!(gate.dtype(), safetensors::Dtype::U8, "gate_proj should be quantized");
            let up = st.tensor(&up_name).unwrap();
            assert_eq!(up.dtype(), safetensors::Dtype::U8, "up_proj should be quantized");
            let down = st.tensor(&down_name).unwrap();
            assert_eq!(down.dtype(), safetensors::Dtype::U8, "down_proj should be quantized");

            // Metadata should have shape entries.
            let prefix = actual_format.metadata_prefix();
            assert!(meta.get(&format!("{prefix}{gate_name}")).is_some(),
                "missing metadata for {gate_name}");
            assert!(meta.get(&format!("{prefix}{up_name}")).is_some(),
                "missing metadata for {up_name}");
            assert!(meta.get(&format!("{prefix}{down_name}")).is_some(),
                "missing metadata for {down_name}");

            // Biases should exist as bf16.
            let gate_bias = st.tensor(&format!("model.layers.0.mlp.experts.{j}.gate_proj.bias")).unwrap();
            assert_eq!(gate_bias.dtype(), safetensors::Dtype::BF16);
            let up_bias = st.tensor(&format!("model.layers.0.mlp.experts.{j}.up_proj.bias")).unwrap();
            assert_eq!(up_bias.dtype(), safetensors::Dtype::BF16);
            let down_bias = st.tensor(&format!("model.layers.0.mlp.experts.{j}.down_proj.bias")).unwrap();
            assert_eq!(down_bias.dtype(), safetensors::Dtype::BF16);
        }

        // Original MXFP4 tensors should NOT be present.
        assert!(st.tensor("model.layers.0.mlp.experts.gate_up_proj_blocks").is_err());
        assert!(st.tensor("model.layers.0.mlp.experts.gate_up_proj_scales").is_err());
        assert!(st.tensor("model.layers.0.mlp.experts.gate_up_proj_bias").is_err());

        // Standard bf16 weight should also be quantized.
        let q_proj = st.tensor("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(q_proj.dtype(), safetensors::Dtype::U8);

        // Norm should pass through as bf16.
        let norm = st.tensor("model.layers.0.input_layernorm.weight").unwrap();
        assert_eq!(norm.dtype(), safetensors::Dtype::BF16);
    }

    #[test]
    fn test_mxfp4_dequant_requant_matches_direct_dequant() {
        // Verify that the quantize command's MXFP4 dequant produces the same bf16
        // bytes as calling dequantize_mxfp4() directly, then quantizing.
        use half::bf16;

        let num_experts = 1;
        let rows = 64;  // down_proj: [hidden, moe_inter]
        let cols = 32;   // Must be divisible by 32.

        let values: Vec<bf16> = (0..rows * cols)
            .map(|i| bf16::from_f32((i % 11) as f32 * 0.2 - 1.0))
            .collect();

        let (blocks, scales) = make_mxfp4_blocks_and_scales(&values, rows, cols);

        // Direct dequant path.
        let direct_bf16 = crate::model::loader::mxfp4::dequantize_mxfp4(
            &blocks, &scales, rows, cols, 32,
        );

        // Quantize the dequanted bf16.
        let actual_format = super::platform_adjusted_format(QuantFormat::Q4);
        let quant = crate::gpu::ops::quant::quantiser(actual_format);
        let direct_quant = quant.quantise(&direct_bf16, rows, cols);

        // Now do it through the quantize command.
        let input_dir = tempfile::tempdir().unwrap();
        let output_dir = tempfile::tempdir().unwrap();

        let block_size = 32;
        let num_sb = (cols + block_size - 1) / block_size;

        let tensors: Vec<(String, Vec<u8>, safetensors::Dtype, Vec<usize>)> = vec![
            (
                "model.layers.0.mlp.experts.down_proj_blocks".into(),
                blocks,
                safetensors::Dtype::U8,
                vec![num_experts, rows, num_sb, 16],
            ),
            (
                "model.layers.0.mlp.experts.down_proj_scales".into(),
                scales,
                safetensors::Dtype::U8,
                vec![num_experts, rows, num_sb],
            ),
        ];

        let tv_refs: Vec<(String, TensorView)> = tensors
            .iter()
            .map(|(name, data, dtype, shape)| {
                let tv = TensorView::new(*dtype, shape.clone(), data).unwrap();
                (name.clone(), tv)
            })
            .collect();

        let st_bytes = safetensors::serialize(tv_refs, &None).unwrap();
        std::fs::write(input_dir.path().join("model.safetensors"), st_bytes).unwrap();
        std::fs::write(input_dir.path().join("config.json"), r#"{"model_type":"gpt-oss"}"#).unwrap();
        std::fs::write(input_dir.path().join("tokenizer.json"), "{}").unwrap();

        exec(QuantizeArgs {
            model: input_dir.path().to_path_buf(),
            output: output_dir.path().to_path_buf(),
            format: "q4".to_string(),
        })
        .unwrap();

        let data = std::fs::read(output_dir.path().join("model.safetensors")).unwrap();
        let st = SafeTensors::deserialize(&data).unwrap();
        let offline = st.tensor("model.layers.0.mlp.experts.0.down_proj.weight").unwrap();

        assert_eq!(
            offline.data(),
            direct_quant.as_slice(),
            "MXFP4 dequant→requant through quantize command must match direct path"
        );
    }
}
