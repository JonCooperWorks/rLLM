// ===========================================================================
// `rllm quantize` — Offline weight quantization.
//
// Pre-quantizes bf16 safetensors weights to Q4 or Q8 format on disk, so that
// subsequent loads (`rllm run`, `rllm serve`) skip the on-load quantization
// step entirely.  This is a pure-CPU operation — no GPU backend needed.
//
// Output format:
//   Standard safetensors files with quantized blocks stored as Dtype::U8 tensors.
//   File-level metadata marks the file as quantized:
//     "quantization" = "rllm-q4" | "rllm-q8"
//     "rllm_q4:<tensor_name>" = "m,k"   (Q4 original logical shape)
//     "rllm_q8:<tensor_name>" = "m,k"   (Q8 original logical shape)
//
//   Non-quantizable tensors (norms, embeddings, biases) pass through as bf16.
//   The output directory is self-contained: includes config.json, tokenizer
//   files, and the quantized safetensors.
//
// Related files:
//   gpu/mod.rs          — quantize_bf16_to_q4(), quantize_bf16_to_q8()
//   gpu/ops/quant.rs    — WeightQuantiser trait, QuantFormat enum
//   model/loader/     — load_safetensors_files(), pre-quantized detection
//   model/config.rs     — ModelConfig parsing
// ===========================================================================

use std::collections::HashMap;
use std::path::PathBuf;

use safetensors::tensor::TensorView;
use safetensors::SafeTensors;

use crate::gpu::ops::quant::{QuantFormat, quantiser};

#[derive(clap::Args)]
pub(crate) struct QuantizeArgs {
    /// Path to input model directory (contains config.json, tokenizer.json, *.safetensors).
    #[arg(long)]
    model: PathBuf,

    /// Path to output directory for quantized model.
    #[arg(long)]
    output: PathBuf,

    /// Quantization format: "q4" (4-bit, default) or "q8" (8-bit).
    #[arg(long, default_value = "q4")]
    format: String,
}

/// Maximum bytes per output shard (5 GB).  Keeps files manageable and
/// matches the HuggingFace convention for large models.
const SHARD_LIMIT: usize = 5 * 1024 * 1024 * 1024;

pub(crate) fn exec(args: QuantizeArgs) -> anyhow::Result<()> {
    let input_dir = &args.model;
    let output_dir = &args.output;

    let format = QuantFormat::from_name(&args.format).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown quantization format '{}' (supported: q4, q8)",
            args.format
        )
    })?;
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
    eprintln!("found {num_tensors} tensors across {} shard(s)", mmaps.len());

    // Process and write tensors incrementally to avoid holding all Q4 data
    // in memory.  For large MoE models (397B = 751 GB bf16 → 235 GB Q4),
    // accumulating all output would exceed RAM.  Instead, we fill one output
    // shard at a time and flush it to disk before starting the next.
    let mut quantized_count = 0usize;
    let mut original_bytes = 0u64;
    let mut quantized_bytes = 0u64;
    let mut output_shard_count = 0usize;

    // Current shard accumulator — flushed when it exceeds SHARD_LIMIT.
    let mut current_shard: Vec<OutputTensor> = Vec::new();
    let mut current_shard_size = 0usize;

    // Global weight_map for the index file (tensor name → shard filename).
    let mut index_weight_map: Vec<(String, usize)> = Vec::new();

    for (tensor_idx, (name, view)) in all_tensors.iter().enumerate() {
        let shape = view.shape();
        let data = view.data();

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
                eprintln!(
                    "  [{}/{}] quantized {} ({:.0} MB → {:.0} MB)",
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
    eprintln!(
        "quantized {quantized_count}/{num_tensors} tensors across {output_shard_count} shard(s)"
    );
    eprintln!(
        "size: {:.1} GB → {:.1} GB ({:.1}x compression)",
        original_bytes as f64 / 1e9,
        quantized_bytes as f64 / 1e9,
        ratio
    );
    eprintln!("output: {}", output_dir.display());

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

    // Exclude norms, embeddings, conv1d, routers.
    let exclude = ["layernorm", "norm", "embed_tokens", "conv1d", "router"];
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
    eprintln!("writing shard {} ({} tensors)...", shard_idx + 1, tensors.len());

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
        eprintln!("wrote index file");
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
            eprintln!("copied {name}");
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

        assert_eq!(meta.get("quantization").unwrap(), "rllm-q4");
        assert_eq!(
            meta.get("rllm_q4:model.layers.0.self_attn.q_proj.weight")
                .unwrap(),
            &format!("{m},{k}")
        );
        // Norm tensor should NOT have a q4 metadata entry.
        assert!(
            meta.get("rllm_q4:model.layers.0.input_layernorm.weight")
                .is_none()
        );

        // Verify the Q4 tensor has the correct byte count.
        let st = SafeTensors::deserialize(&data).unwrap();
        let q_proj = st
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q_proj.dtype(), safetensors::Dtype::U8);
        let expected_bytes = crate::gpu::q4_byte_count(m, k);
        assert_eq!(q_proj.data().len(), expected_bytes);

        // Verify the norm tensor passed through as bf16 unchanged.
        let norm = st
            .tensor("model.layers.0.input_layernorm.weight")
            .unwrap();
        assert_eq!(norm.dtype(), safetensors::Dtype::BF16);
        assert_eq!(norm.shape(), &[64]);
    }

    #[test]
    fn test_end_to_end_quantize_matches_on_load_quantize() {
        // Verify that pre-quantized Q4 bytes are identical to on-load quantization.
        use half::bf16;

        let m = 2;
        let k = 32;
        let n_elements = m * k;
        let values: Vec<bf16> = (0..n_elements)
            .map(|i| bf16::from_f32((i % 17) as f32 * 0.1 - 0.8))
            .collect();
        let bf16_bytes: &[u8] = bytemuck::cast_slice(&values);

        // On-load path: quantize_bf16_to_q4 directly.
        let on_load_q4 = crate::gpu::quantize_bf16_to_q4(bf16_bytes, m, k);

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
        let offline_q4 = st
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        assert_eq!(
            offline_q4.data(),
            on_load_q4.as_slice(),
            "offline and on-load quantization must produce identical Q4 bytes"
        );
    }

    #[test]
    fn test_end_to_end_quantize_multiple_tensors() {
        let input_dir = tempfile::tempdir().unwrap();
        let output_dir = tempfile::tempdir().unwrap();

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

        // Count Q4 tensors from metadata.
        let q4_count = meta.keys().filter(|k| k.starts_with("rllm_q4:")).count();
        assert_eq!(q4_count, 7, "7 projection weights should be quantized");

        // Verify all tensors are present and accessible.
        assert_eq!(st.len(), 10);

        // Spot-check: embed_tokens should remain bf16.
        let embed = st.tensor("model.embed_tokens.weight").unwrap();
        assert_eq!(embed.dtype(), safetensors::Dtype::BF16);

        // Spot-check: gate_proj should be U8 (Q4).
        let gate = st
            .tensor("model.layers.0.mlp.gate_proj.weight")
            .unwrap();
        assert_eq!(gate.dtype(), safetensors::Dtype::U8);
    }

    #[test]
    fn test_end_to_end_quantize_q8_produces_valid_output() {
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

        let output_st = output_dir.path().join("model.safetensors");
        let data = std::fs::read(&output_st).unwrap();
        let (_, metadata) = SafeTensors::read_metadata(&data).unwrap();
        let meta = metadata.metadata().as_ref().unwrap();

        assert_eq!(meta.get("quantization").unwrap(), "rllm-q8");
        assert_eq!(
            meta.get("rllm_q8:model.layers.0.self_attn.q_proj.weight")
                .unwrap(),
            &format!("{m},{k}")
        );
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
        let expected_bytes = crate::gpu::q8_byte_count(m, k);
        assert_eq!(q_proj.data().len(), expected_bytes);
    }

    #[test]
    fn test_end_to_end_quantize_q8_matches_on_load() {
        use half::bf16;

        let m = 2;
        let k = 32;
        let values: Vec<bf16> = (0..m * k)
            .map(|i| bf16::from_f32((i % 17) as f32 * 0.1 - 0.8))
            .collect();
        let bf16_bytes: &[u8] = bytemuck::cast_slice(&values);

        let on_load_q8 = crate::gpu::quantize_bf16_to_q8(bf16_bytes, m, k);

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
        let offline_q8 = st
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        assert_eq!(
            offline_q8.data(),
            on_load_q8.as_slice(),
            "offline and on-load quantization must produce identical Q8 bytes"
        );
    }
}
