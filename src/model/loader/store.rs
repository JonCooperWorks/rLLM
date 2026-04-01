// ===========================================================================
// TensorStore — abstracts single-file vs sharded safetensors access.
//
// For single-file models, there's one shard and no weight map.
// For sharded models, the index JSON maps tensor names to shard indices.
// The caller just calls `store.tensor("name")` and gets the data regardless
// of which shard file it lives in.
//
// Also contains the safetensors file I/O (load_safetensors_files) and
// the quick pre-quantization check (is_prequantized_model).
//
// Related files:
//   loader/mod.rs — uses TensorStore during weight loading
//   commands/quantize.rs — calls load_safetensors_files for the quant tool
//   engine/loader.rs — calls is_prequantized_model
// ===========================================================================

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use safetensors::SafeTensors;
use tracing::info;

pub(crate) struct TensorStore<'a> {
    pub(crate) shards: Vec<SafeTensors<'a>>,
    /// Maps tensor names to shard indices.  Empty for single-file models
    /// (all tensors are in shards[0]).
    pub(crate) weight_map: HashMap<String, usize>,
    /// Pre-quantized Q4 tensors: name → original (m, k) shape.
    /// Populated from safetensors metadata when loading a model quantized
    /// by `rllm quantize`.  Empty for normal bf16 models.
    pub(crate) q4_map: HashMap<String, (usize, usize)>,
    /// Pre-quantized Q8 tensors: name → original (m, k) shape.
    pub(crate) q8_map: HashMap<String, (usize, usize)>,
    /// Pre-quantized FP8 tensors: name → original (m, k) shape.
    /// Populated when loading NVIDIA FP8 E4M3 models produced by `rllm quantize`.
    pub(crate) fp8_map: HashMap<String, (usize, usize)>,
    /// Pre-quantized TQ3 tensors: name → original (m, k) shape.
    /// TQ3 = TurboQuant 3-bit with Walsh-Hadamard rotation (4.0 bpw).
    pub(crate) tq3_map: HashMap<String, (usize, usize)>,
    /// Pre-quantized NVFP4 tensors: name → original (m, k) shape.
    /// NVFP4 = NVIDIA FP4 E2M1, same block layout as Q4 (4.5 bpw).
    pub(crate) nvfp4_map: HashMap<String, (usize, usize)>,
}

impl<'a> TensorStore<'a> {
    pub(crate) fn tensor(&self, name: &str) -> anyhow::Result<safetensors::tensor::TensorView<'a>> {
        if let Some(&idx) = self.weight_map.get(name) {
            self.shards[idx]
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("tensor '{name}' not in shard {idx}: {e}"))
        } else {
            // Single-file: try shard 0.
            self.shards[0]
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("tensor '{name}' not found: {e}"))
        }
    }

    /// Check if a tensor is pre-quantized Q4, returning its original shape.
    pub(crate) fn q4_shape(&self, name: &str) -> Option<(usize, usize)> {
        self.q4_map.get(name).copied()
    }

    /// Check if a tensor is pre-quantized Q8, returning its original shape.
    pub(crate) fn q8_shape(&self, name: &str) -> Option<(usize, usize)> {
        self.q8_map.get(name).copied()
    }

    /// Check if a tensor is pre-quantized FP8, returning its original shape.
    pub(crate) fn fp8_shape(&self, name: &str) -> Option<(usize, usize)> {
        self.fp8_map.get(name).copied()
    }

    /// Check if a tensor is pre-quantized TQ3, returning its original shape.
    pub(crate) fn tq3_shape(&self, name: &str) -> Option<(usize, usize)> {
        self.tq3_map.get(name).copied()
    }

    /// Check if a tensor is pre-quantized NVFP4, returning its original shape.
    pub(crate) fn nvfp4_shape(&self, name: &str) -> Option<(usize, usize)> {
        self.nvfp4_map.get(name).copied()
    }

    /// Check if a tensor is pre-quantized (Q4, Q8, FP8, TQ3, or NVFP4), returning (m, k, dtype).
    pub(crate) fn quant_shape(&self, name: &str) -> Option<(usize, usize, crate::gpu::TensorDtype)> {
        if let Some((m, k)) = self.q4_shape(name) {
            Some((m, k, crate::gpu::TensorDtype::Q4))
        } else if let Some((m, k)) = self.q8_shape(name) {
            Some((m, k, crate::gpu::TensorDtype::Q8))
        } else if let Some((m, k)) = self.fp8_shape(name) {
            Some((m, k, crate::gpu::TensorDtype::FP8))
        } else if let Some((m, k)) = self.tq3_shape(name) {
            Some((m, k, crate::gpu::TensorDtype::TQ3))
        } else if let Some((m, k)) = self.nvfp4_shape(name) {
            Some((m, k, crate::gpu::TensorDtype::NVFP4))
        } else {
            None
        }
    }
}

/// Quick check if a model directory contains pre-quantized (rllm-q4 or rllm-q8) safetensors.
/// Only reads the first shard's metadata header — no full mmap needed.
pub(crate) fn is_prequantized_model(model_dir: &Path) -> bool {
    let first_shard = if model_dir.join("model.safetensors").exists() {
        model_dir.join("model.safetensors")
    } else {
        model_dir.join("model-00001-of-00001.safetensors") // try first shard
    };
    // Find the actual first shard from index if simple names don't exist.
    let path = if first_shard.exists() {
        first_shard
    } else if let Ok(idx_str) = std::fs::read_to_string(model_dir.join("model.safetensors.index.json")) {
        if let Ok(idx) = serde_json::from_str::<serde_json::Value>(&idx_str) {
            if let Some(wm) = idx["weight_map"].as_object() {
                if let Some(first_file) = wm.values().next().and_then(|v| v.as_str()) {
                    model_dir.join(first_file)
                } else { return false; }
            } else { return false; }
        } else { return false; }
    } else { return false; };

    let Ok(file) = std::fs::File::open(&path) else { return false };
    let Ok(mmap) = (unsafe { Mmap::map(&file) }) else { return false };
    if let Ok((_, metadata)) = SafeTensors::read_metadata(mmap.as_ref()) {
        if let Some(meta) = metadata.metadata() {
            if let Some(tag) = meta.get("quantization").map(|v| v.as_str()) {
                return crate::gpu::QuantFormat::from_metadata_tag(tag).is_some();
            }
        }
    }
    false
}

/// Load safetensors files from a model directory.
///
/// Returns the mmaps (kept alive for the SafeTensors references) and a weight
/// map for sharded models.
pub(crate) fn load_safetensors_files(model_dir: &Path) -> anyhow::Result<(Vec<Mmap>, HashMap<String, usize>)> {
    // Case 1: single model.safetensors file.
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        info!(path = %single.display(), "loading safetensors");
        let file = std::fs::File::open(&single)?;
        let mmap = unsafe { Mmap::map(&file)? };
        return Ok((vec![mmap], HashMap::new()));
    }

    // Case 2: sharded model with index file.
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        anyhow::bail!(
            "no safetensors file found in {} (expected model.safetensors or model.safetensors.index.json)",
            model_dir.display()
        );
    }

    let index_str = std::fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_str)?;
    let wm = index["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("invalid index: missing weight_map"))?;

    // Collect unique shard filenames (preserving order).
    let mut shard_files: Vec<String> = Vec::new();
    let mut file_to_idx: HashMap<String, usize> = HashMap::new();
    for filename in wm.values() {
        let f = filename.as_str().unwrap().to_string();
        if !file_to_idx.contains_key(&f) {
            file_to_idx.insert(f.clone(), shard_files.len());
            shard_files.push(f);
        }
    }

    // Build weight_map: tensor_name → shard_index.
    let mut weight_map = HashMap::new();
    for (tensor_name, filename) in wm {
        let f = filename.as_str().unwrap();
        weight_map.insert(tensor_name.clone(), file_to_idx[f]);
    }

    // Validate all shard files exist before attempting to mmap.
    let missing: Vec<&str> = shard_files
        .iter()
        .filter(|f| !model_dir.join(f).exists())
        .map(|f| f.as_str())
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "incomplete model in {}: missing shard file(s): {}. \
             Re-download with: huggingface-cli download <model> --local-dir {}",
            model_dir.display(),
            missing.join(", "),
            model_dir.display()
        );
    }

    // Memory-map each shard file.
    info!(
        shards = shard_files.len(),
        dir = %model_dir.display(),
        "loading safetensors shards"
    );
    let mmaps: Vec<Mmap> = shard_files
        .iter()
        .map(|f| {
            let path = model_dir.join(f);
            let file = std::fs::File::open(&path)
                .map_err(|e| anyhow::anyhow!("failed to open {f}: {e}"))?;
            unsafe { Mmap::map(&file) }.map_err(|e| anyhow::anyhow!("failed to mmap {f}: {e}"))
        })
        .collect::<anyhow::Result<_>>()?;

    Ok((mmaps, weight_map))
}
