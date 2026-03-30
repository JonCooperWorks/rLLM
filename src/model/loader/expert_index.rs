// ===========================================================================
// Expert index building for SSD streaming.
//
// Reads safetensors file headers to locate expert tensors without loading
// their data.  The resulting ExpertIndex maps (layer, expert_id) → file
// offset for on-demand pread() during inference.
//
// Related files:
//   model/expert_stream.rs — ExpertIndex, ExpertStreamer, pread dispatch
//   loader/mod.rs          — load_weights_maybe_streamed() calls this
// ===========================================================================

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use safetensors::SafeTensors;
use tracing::{info, debug};

use crate::model::config::ModelConfig;
use crate::model::expert_stream::{self, FusedLayerInfo, PerExpertInfo, safetensors_data_start};
use super::store::{TensorStore, load_safetensors_files};

pub(crate) fn build_expert_index_from_safetensors(
    model_dir: &Path,
    config: &ModelConfig,
) -> anyhow::Result<expert_stream::ExpertIndex> {
    let hidden = config.hidden_size;
    let moe_inter = config.moe_intermediate_size;
    let num_experts = config.num_experts;
    let num_layers = config.num_hidden_layers;

    // Load safetensors headers to compute tensor file offsets.
    let (mmaps, weight_map) = load_safetensors_files(model_dir)?;

    // Collect per-tensor Q4 metadata from shard headers.
    // We need this to determine if expert tensors specifically are Q4
    // (not just whether the model has ANY Q4 tensors — attention weights
    // may be Q4 while expert weights remain BF16, as with Mixtral).
    let mut q4_expert_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut q8_expert_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for mmap in &mmaps {
        if let Ok((_, metadata)) = SafeTensors::read_metadata(mmap.as_ref()) {
            if let Some(meta) = metadata.metadata() {
                for key in meta.keys() {
                    if let Some(tensor_name) = key.strip_prefix("rllm_q4:") {
                        if tensor_name.contains("expert") {
                            q4_expert_names.insert(tensor_name.to_string());
                        }
                    } else if let Some(tensor_name) = key.strip_prefix("rllm_q8:") {
                        if tensor_name.contains("expert") {
                            q8_expert_names.insert(tensor_name.to_string());
                        }
                    }
                }
            }
        }
    }

    // Compute data_start for each shard (8 + header_len).
    let data_starts: Vec<u64> = mmaps.iter().map(|m| safetensors_data_start(m)).collect();

    // Parse safetensors to get tensor views (for data pointer offsets).
    let shards: Vec<SafeTensors> = mmaps
        .iter()
        .map(|m| SafeTensors::deserialize(m).expect("failed to parse safetensors"))
        .collect();

    let store = TensorStore {
        shards,
        weight_map: weight_map.clone(),
        q4_map: HashMap::new(),
        q8_map: HashMap::new(),
        fp8_map: HashMap::new(),
    };

    // Determine the layer prefix pattern.
    let prefix_base = format!("{}layers.", config.weight_prefix);

    // Open file handles for pread (kept alive in ExpertIndex).
    let shard_paths = get_shard_paths(model_dir)?;
    let shard_files: Vec<std::fs::File> = shard_paths
        .iter()
        .map(|p| std::fs::File::open(p).expect("failed to open shard for streaming"))
        .collect();

    // Find the first MoE layer for naming detection.  Nemotron-H interleaves
    // mamba2/moe/attention layers — layer 0 may be mamba2 with no expert tensors.
    // For models without layer_types (Mixtral, Qwen3-MoE), all layers are MoE.
    let first_moe_layer = if config.layer_types.is_empty() {
        0
    } else {
        config.layer_types.iter().position(|t| t == "moe").unwrap_or(0)
    };

    // Detect fused vs per-expert format (same logic as load_ffn_weights).
    let test_prefix = format!("{prefix_base}{first_moe_layer}");
    let fused_name = format!("{test_prefix}.mlp.experts.gate_up_proj");
    let is_fused = store.tensor(&fused_name).is_ok();

    if is_fused {
        // Fused format (Qwen3.5): gate_up_proj [num_experts, 2*moe_inter, hidden]
        let mut layer_info = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let prefix = format!("{prefix_base}{layer_idx}");
            let gu_name = format!("{prefix}.mlp.experts.gate_up_proj");
            let down_name = format!("{prefix}.mlp.experts.down_proj");

            let gu_view = store.tensor(&gu_name)?;
            let down_view = store.tensor(&down_name)?;

            // Compute file offset: data pointer - mmap base + data_start
            let gu_shard = shard_index(&weight_map, &gu_name);
            let down_shard = shard_index(&weight_map, &down_name);

            let gu_offset = tensor_file_offset(
                gu_view.data(), mmaps[gu_shard].as_ref(), data_starts[gu_shard],
            );
            let down_offset = tensor_file_offset(
                down_view.data(), mmaps[down_shard].as_ref(), data_starts[down_shard],
            );

            layer_info.push(FusedLayerInfo {
                shard_gate_up: gu_shard,
                shard_down: down_shard,
                gate_up_file_offset: gu_offset,
                down_file_offset: down_offset,
            });
        }

        // Check if fused expert tensors are pre-quantized (Q4 or Q8).
        let fused_gate_up = format!("{prefix_base}0.mlp.experts.gate_up_proj");
        let expert_quant = if q4_expert_names.contains(&fused_gate_up) {
            debug!("detected pre-quantized expert data (rllm-q4)");
            Some(crate::gpu::ops::quant::QuantFormat::Q4)
        } else if q8_expert_names.contains(&fused_gate_up) {
            debug!("detected pre-quantized expert data (rllm-q8)");
            Some(crate::gpu::ops::quant::QuantFormat::Q8)
        } else {
            None
        };

        info!(layers = num_layers, experts = num_experts, format = "fused", "built expert index");

        Ok(expert_stream::build_fused_expert_index(
            layer_info, shard_files, hidden, moe_inter, num_experts, expert_quant,
        ))
    } else {
        // Per-expert format (Qwen3-MoE, Mixtral, Nemotron-H): experts.{j}.* etc.
        let mut layer_info = Vec::with_capacity(num_layers);

        // Detect per-expert naming pattern.
        //   Qwen:     {prefix}.mlp.experts.0.gate_proj.weight  (3 tensors: gate, up, down)
        //   Mixtral:  {prefix}.block_sparse_moe.experts.0.w1.weight (3 tensors: w1, w3, w2)
        //   Nemotron: {prefix}.mixer.experts.0.up_proj.weight   (2 tensors: up, down — no gate)
        let test_qwen = format!("{test_prefix}.mlp.experts.0.gate_proj.weight");
        let test_nemotron = format!("{test_prefix}.mixer.experts.0.up_proj.weight");
        let is_qwen_naming = store.tensor(&test_qwen).is_ok();
        let is_nemotron_naming = !is_qwen_naming && store.tensor(&test_nemotron).is_ok();
        let has_gate = !is_nemotron_naming; // Nemotron uses relu² (no gate_proj)

        for layer_idx in 0..num_layers {
            // Nemotron-H: only "moe" layers have expert tensors.  Non-MoE layers
            // (mamba2, attention) get an empty vec so layer indices stay aligned.
            if !config.layer_types.is_empty() && config.layer_types[layer_idx] != "moe" {
                layer_info.push(Vec::new());
                continue;
            }

            let prefix = format!("{prefix_base}{layer_idx}");
            let mut experts = Vec::with_capacity(num_experts);

            for j in 0..num_experts {
                if is_nemotron_naming {
                    // Nemotron: mixer.experts.{j}.up_proj + down_proj (no gate).
                    let up_name = format!("{prefix}.mixer.experts.{j}.up_proj.weight");
                    let down_name = format!("{prefix}.mixer.experts.{j}.down_proj.weight");

                    let up_view = store.tensor(&up_name)?;
                    let down_view = store.tensor(&down_name)?;

                    let up_shard = shard_index(&weight_map, &up_name);
                    let down_shard = shard_index(&weight_map, &down_name);

                    let up_offset = tensor_file_offset(
                        up_view.data(), mmaps[up_shard].as_ref(), data_starts[up_shard],
                    );

                    // No gate tensor — point gate at up (will be ignored by streamer
                    // when has_gate_proj=false).
                    experts.push(PerExpertInfo {
                        shard_gate: up_shard,
                        shard_up: up_shard,
                        shard_down: down_shard,
                        gate_file_offset: up_offset,
                        up_file_offset: up_offset,
                        down_file_offset: tensor_file_offset(
                            down_view.data(), mmaps[down_shard].as_ref(), data_starts[down_shard],
                        ),
                    });
                } else {
                    let (gate_name, up_name, down_name) = if is_qwen_naming {
                        (
                            format!("{prefix}.mlp.experts.{j}.gate_proj.weight"),
                            format!("{prefix}.mlp.experts.{j}.up_proj.weight"),
                            format!("{prefix}.mlp.experts.{j}.down_proj.weight"),
                        )
                    } else {
                        // Mixtral naming: w1=gate, w3=up, w2=down
                        (
                            format!("{prefix}.block_sparse_moe.experts.{j}.w1.weight"),
                            format!("{prefix}.block_sparse_moe.experts.{j}.w3.weight"),
                            format!("{prefix}.block_sparse_moe.experts.{j}.w2.weight"),
                        )
                    };

                    let gate_view = store.tensor(&gate_name)?;
                    let up_view = store.tensor(&up_name)?;
                    let down_view = store.tensor(&down_name)?;

                    let gate_shard = shard_index(&weight_map, &gate_name);
                    let up_shard = shard_index(&weight_map, &up_name);
                    let down_shard = shard_index(&weight_map, &down_name);

                    experts.push(PerExpertInfo {
                        shard_gate: gate_shard,
                        shard_up: up_shard,
                        shard_down: down_shard,
                        gate_file_offset: tensor_file_offset(
                            gate_view.data(), mmaps[gate_shard].as_ref(), data_starts[gate_shard],
                        ),
                        up_file_offset: tensor_file_offset(
                            up_view.data(), mmaps[up_shard].as_ref(), data_starts[up_shard],
                        ),
                        down_file_offset: tensor_file_offset(
                            down_view.data(), mmaps[down_shard].as_ref(), data_starts[down_shard],
                        ),
                    });
                }
            }

            layer_info.push(experts);
        }

        // Check if per-expert tensors are pre-quantized (Q4 or Q8).
        // Mixtral's expert weights (w1/w2/w3) may remain BF16 even in a Q4 model
        // because the quantizer only quantizes weight names it recognises.
        let first_expert_up = if is_nemotron_naming {
            format!("{test_prefix}.mixer.experts.0.up_proj.weight")
        } else if is_qwen_naming {
            format!("{test_prefix}.mlp.experts.0.gate_proj.weight")
        } else {
            format!("{test_prefix}.block_sparse_moe.experts.0.w1.weight")
        };
        let expert_quant = if q4_expert_names.contains(&first_expert_up) {
            debug!("detected pre-quantized expert data (rllm-q4)");
            Some(crate::gpu::ops::quant::QuantFormat::Q4)
        } else if q8_expert_names.contains(&first_expert_up) {
            debug!("detected pre-quantized expert data (rllm-q8)");
            Some(crate::gpu::ops::quant::QuantFormat::Q8)
        } else {
            None
        };

        let moe_layers = layer_info.iter().filter(|l| !l.is_empty()).count();
        info!(layers = moe_layers, experts = num_experts, format = "per-expert",
              has_gate = has_gate, "built expert index");

        Ok(expert_stream::build_per_expert_index(
            layer_info, shard_files, hidden, moe_inter, expert_quant, has_gate,
        ))
    }
}

/// Compute the absolute file offset of a tensor's data within its shard file.
///
/// pointer arithmetic: the tensor view's data slice is within the mmap,
/// so its offset is (data_ptr - mmap_ptr) which already accounts for the
/// safetensors header.
fn tensor_file_offset(tensor_data: &[u8], mmap: &[u8], _data_start: u64) -> u64 {
    let tensor_ptr = tensor_data.as_ptr() as usize;
    let mmap_ptr = mmap.as_ptr() as usize;
    (tensor_ptr - mmap_ptr) as u64
}

/// Get shard index for a tensor name (0 for single-file models).
fn shard_index(weight_map: &HashMap<String, usize>, name: &str) -> usize {
    weight_map.get(name).copied().unwrap_or(0)
}

/// Get ordered shard file paths for a model directory.
fn get_shard_paths(model_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    let index_str = std::fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_str)?;
    let wm = index["weight_map"].as_object().unwrap();

    let mut shard_files: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for filename in wm.values() {
        let f = filename.as_str().unwrap().to_string();
        if seen.insert(f.clone()) {
            shard_files.push(f);
        }
    }

    Ok(shard_files.iter().map(|f| model_dir.join(f)).collect())
}
