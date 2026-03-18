// ===========================================================================
// Tensor parallelism planning — how to split model weights across GPUs.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Given a ModelConfig and a DeviceConfig (how many GPUs, which rank am I),
//   compute a ShardingPlan that tells the weight loader how to slice each
//   tensor.  The plan is derived automatically — no per-model configuration
//   needed.
//
// Key insight: the sharding rules are the SAME for every transformer:
//   - Column-split projections that fan out (QKV, gate, up)
//   - Row-split projections that fan in (O, down)
//   - Replicate small tensors (norms, biases, embeddings)
//   This is why the plan can be derived from config alone.
//
// What happens at each stage:
//   1. DeviceConfig describes the GPU topology (world_size, rank, strategy)
//   2. ShardingPlan::derive() validates divisibility and assigns splits
//   3. The weight loader (loader.rs) uses the plan to slice tensors on load
//   4. Primitives (primitives.rs) insert AllReduce after row-split matmuls
//   5. Models are completely unaware — they see smaller weight matrices
//      and the AllReduce calls are no-ops for single GPU
//
// Why derive from config instead of annotating models?
//   rLLM supports 9 architectures.  They all use the same transformer
//   pattern (QKV → attention → O → FFN).  Annotating each model's forward
//   pass with sharding info would be 8x the maintenance and easy to get
//   wrong.  Deriving from config means adding a new model automatically
//   gets TP support.
//
// Related files:
//   Weight loader:    model/loader.rs (consumes the plan)
//   Primitives:       model/primitives.rs (inserts AllReduce)
//   AllReduce trait:  gpu/ops/allreduce.rs
//   Model configs:    model/config.rs (input to derive)
// ===========================================================================

use crate::model::config::ModelConfig;

// ---------------------------------------------------------------------------
// Parallelism strategy and device topology.
// ---------------------------------------------------------------------------

/// Parallelism strategy for multi-GPU inference.
///
/// Learning note: these strategies differ in WHAT gets split:
///   - TensorParallel: split weight matrices (heads, intermediate dim)
///   - ExpertParallel: split experts across GPUs (MoE only)
///   - Hybrid: TP for attention, EP for MoE FFN (best of both)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParallelStrategy {
    /// Standard Megatron-LM tensor parallelism: split QKV by heads,
    /// FFN by intermediate dim.  Two AllReduce per layer.
    TensorParallel,
    /// Expert parallelism (MoE only): assign whole experts to GPUs.
    /// Router is replicated, tokens are redistributed.
    ExpertParallel,
    /// Hybrid: TP for attention + EP for MoE FFN.
    /// Reduces both the number of AllReduce ops and expert splitting.
    Hybrid,
}

/// Device configuration for distributed inference.
///
/// Learning note: this mirrors the standard distributed training concepts:
///   - world_size: total GPU count (e.g. 4 for 4-way TP)
///   - rank: this process's index (0..world_size-1)
///   - strategy: how to distribute work
///
/// In production this would come from environment variables set by the
/// process launcher (like torchrun sets RANK and WORLD_SIZE).
#[derive(Debug, Clone)]
pub(crate) struct DeviceConfig {
    pub world_size: usize,
    pub rank: usize,
    pub strategy: ParallelStrategy,
}

impl DeviceConfig {
    /// Single-GPU configuration — no parallelism, no communication.
    pub fn single() -> Self {
        DeviceConfig {
            world_size: 1,
            rank: 0,
            strategy: ParallelStrategy::TensorParallel,
        }
    }
}

// ---------------------------------------------------------------------------
// Split dimensions — how each weight tensor is partitioned.
// ---------------------------------------------------------------------------

/// How a single weight tensor is split across GPUs.
///
/// Learning note: "Column" and "Row" refer to which dimension of the
/// weight matrix [out_dim, in_dim] is split:
///   Column = split out_dim (rows of the matrix, columns of output)
///   Row = split in_dim (cols of the matrix, rows of input)
///
/// Column-split is communication-free (independent output slices).
/// Row-split requires AllReduce (partial sums must be combined).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SplitDimension {
    /// Split the output dimension: each GPU gets out_dim/world_size rows.
    /// No communication needed after matmul.
    Column,
    /// Split the input dimension: each GPU gets in_dim/world_size columns.
    /// Requires AllReduce after matmul to sum partial results.
    Row,
    /// Full tensor on every GPU.  Used for norms, biases, embeddings.
    Replicated,
    /// Vocabulary-parallel: split vocab_size across GPUs.
    /// Each GPU embeds/projects only its vocab slice.
    /// Requires AllGather to reconstruct full logits.
    VocabParallel,
    /// Expert-parallel: each GPU owns a subset of MoE experts.
    /// expert_indices lists which experts this rank is responsible for.
    ExpertParallel { expert_indices: Vec<usize> },
}

// ---------------------------------------------------------------------------
// Sharding plan — the complete mapping from weight names to splits.
// ---------------------------------------------------------------------------

/// Sharding annotation for a single named weight tensor.
///
/// The loader looks up each weight's name in the plan to decide
/// how to slice the raw bytes before uploading to GPU.
#[derive(Debug, Clone)]
pub(crate) struct WeightSharding {
    pub name: String,
    pub split: SplitDimension,
    pub original_shape: [usize; 2],
    pub shard_shape: [usize; 2],
}

/// Complete sharding plan for a model.
///
/// Computed once from ModelConfig + DeviceConfig, then passed to the
/// loader.  The plan is deterministic — given the same config and
/// device setup, every rank computes the same plan and loads its own
/// slice.
#[derive(Debug)]
pub(crate) struct ShardingPlan {
    pub device: DeviceConfig,
    pub weights: Vec<WeightSharding>,
    pub embed_split: SplitDimension,
    pub lm_head_split: SplitDimension,
}

// ---------------------------------------------------------------------------
// Validation — ensure config is compatible with the requested parallelism.
// ---------------------------------------------------------------------------

/// Validate that the model config is compatible with the requested parallelism.
///
/// Learning note: tensor parallelism requires EXACT divisibility because
/// each GPU must get the same number of heads / intermediate neurons.
/// Unlike data parallelism (which just splits batches), TP splits the
/// model's internal dimensions — uneven splits would mean different GPUs
/// have different-shaped weight matrices, breaking the AllReduce.
fn validate(config: &ModelConfig, world_size: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.num_attention_heads % world_size == 0,
        "num_attention_heads ({}) must be divisible by world_size ({}) \
         for tensor parallelism — each GPU needs an equal number of \
         query heads",
        config.num_attention_heads,
        world_size
    );

    anyhow::ensure!(
        config.num_key_value_heads % world_size == 0,
        "num_key_value_heads ({}) must be divisible by world_size ({}) \
         — GQA head groups must split evenly across GPUs",
        config.num_key_value_heads,
        world_size
    );

    if config.intermediate_size > 0 {
        anyhow::ensure!(
            config.intermediate_size % world_size == 0,
            "intermediate_size ({}) must be divisible by world_size ({}) \
             — FFN columns must split evenly",
            config.intermediate_size,
            world_size
        );
    }

    if config.is_moe() {
        anyhow::ensure!(
            config.num_experts % world_size == 0,
            "num_experts ({}) must be divisible by world_size ({}) \
             for expert parallelism",
            config.num_experts,
            world_size
        );
    }

    // Qwen 3.5 DeltaNet heads.
    if config.is_hybrid_deltanet() {
        anyhow::ensure!(
            config.linear_num_key_heads % world_size == 0,
            "linear_num_key_heads ({}) must be divisible by world_size ({}) \
             for DeltaNet tensor parallelism",
            config.linear_num_key_heads,
            world_size
        );
        anyhow::ensure!(
            config.linear_num_value_heads % world_size == 0,
            "linear_num_value_heads ({}) must be divisible by world_size ({}) \
             for DeltaNet tensor parallelism",
            config.linear_num_value_heads,
            world_size
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Sharding plan derivation.
// ---------------------------------------------------------------------------

impl ShardingPlan {
    /// Derive a sharding plan from model config and device topology.
    ///
    /// When `vocab_parallel` is true, the embedding and LM head are split
    /// across GPUs by vocabulary slice (requires AllGather for logits).
    /// When false, they are replicated on every GPU.
    pub fn derive(
        config: &ModelConfig,
        device: DeviceConfig,
        vocab_parallel: bool,
    ) -> anyhow::Result<Self> {
        let ws = device.world_size;

        // Single GPU — everything is replicated, no splitting needed.
        if ws == 1 {
            return Ok(ShardingPlan {
                device,
                weights: Vec::new(),
                embed_split: SplitDimension::Replicated,
                lm_head_split: SplitDimension::Replicated,
            });
        }

        validate(config, ws)?;

        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        let q_dim = config.num_attention_heads * head_dim;
        let kv_dim = config.num_key_value_heads * head_dim;
        let inter = config.intermediate_size;
        let prefix = &config.weight_prefix;

        let mut weights = Vec::new();

        // Helper: push a weight sharding entry.
        let mut add = |name: String, split: SplitDimension, orig: [usize; 2]| {
            let shard = shard_shape(&split, orig, ws);
            weights.push(WeightSharding {
                name,
                split,
                original_shape: orig,
                shard_shape: shard,
            });
        };

        for i in 0..config.num_hidden_layers {
            let layer = format!("{prefix}layers.{i}");

            // -- Attention projections --
            // q_proj: [q_dim, hidden] → Column (split heads)
            add(
                format!("{layer}.self_attn.q_proj.weight"),
                SplitDimension::Column,
                [q_dim, hidden],
            );
            // k_proj: [kv_dim, hidden] → Column (split KV heads)
            add(
                format!("{layer}.self_attn.k_proj.weight"),
                SplitDimension::Column,
                [kv_dim, hidden],
            );
            // v_proj: [kv_dim, hidden] → Column
            add(
                format!("{layer}.self_attn.v_proj.weight"),
                SplitDimension::Column,
                [kv_dim, hidden],
            );
            // o_proj: [hidden, q_dim] → Row (requires AllReduce)
            add(
                format!("{layer}.self_attn.o_proj.weight"),
                SplitDimension::Row,
                [hidden, q_dim],
            );

            // -- QKV bias (Qwen only, 1D → Column to match projection split) --
            // Bias vectors are 1D [dim], treated as [dim, 1] for splitting.
            if config.num_key_value_heads > 0 {
                // Only add if model might have bias — the loader skips missing weights.
                add(
                    format!("{layer}.self_attn.q_proj.bias"),
                    SplitDimension::Column,
                    [q_dim, 1],
                );
                add(
                    format!("{layer}.self_attn.k_proj.bias"),
                    SplitDimension::Column,
                    [kv_dim, 1],
                );
                add(
                    format!("{layer}.self_attn.v_proj.bias"),
                    SplitDimension::Column,
                    [kv_dim, 1],
                );
            }

            // -- QK-norm (Qwen3 MoE, Qwen3.5, Gemma3): per-head, replicated --
            add(
                format!("{layer}.self_attn.q_norm.weight"),
                SplitDimension::Replicated,
                [head_dim, 1],
            );
            add(
                format!("{layer}.self_attn.k_norm.weight"),
                SplitDimension::Replicated,
                [head_dim, 1],
            );

            // -- Norms: always replicated --
            add(
                format!("{layer}.input_layernorm.weight"),
                SplitDimension::Replicated,
                [hidden, 1],
            );
            add(
                format!("{layer}.post_attention_layernorm.weight"),
                SplitDimension::Replicated,
                [hidden, 1],
            );

            // -- Dense FFN (when present) --
            if inter > 0 {
                // gate_proj: [inter, hidden] → Column
                add(
                    format!("{layer}.mlp.gate_proj.weight"),
                    SplitDimension::Column,
                    [inter, hidden],
                );
                // up_proj: [inter, hidden] → Column (must match gate)
                add(
                    format!("{layer}.mlp.up_proj.weight"),
                    SplitDimension::Column,
                    [inter, hidden],
                );
                // down_proj: [hidden, inter] → Row (requires AllReduce)
                add(
                    format!("{layer}.mlp.down_proj.weight"),
                    SplitDimension::Row,
                    [hidden, inter],
                );
            }

            // -- MoE experts --
            //
            // Weight naming varies by architecture:
            //   Mixtral:  block_sparse_moe.gate / block_sparse_moe.experts.{j}.w1/w3/w2
            //   Others:   mlp.gate / mlp.experts.{j}.gate_proj/up_proj/down_proj
            //
            // The plan must use the exact safetensors key so upload_sharded() can
            // match the tensor name when slicing.  See loader.rs load_moe_ffn_weights()
            // for the corresponding loading logic.
            if config.is_moe() {
                let moe_inter = config.moe_intermediate_size;
                let is_mixtral = config.model_type == "mixtral";
                let use_ep = matches!(
                    device.strategy,
                    ParallelStrategy::ExpertParallel | ParallelStrategy::Hybrid
                );

                // Router gate: always replicated (all GPUs need full routing).
                let router_name = if is_mixtral {
                    format!("{layer}.block_sparse_moe.gate.weight")
                } else {
                    format!("{layer}.mlp.gate.weight")
                };
                add(
                    router_name,
                    SplitDimension::Replicated,
                    [config.num_experts, hidden],
                );

                // Helper: build (gate, up, down) tensor names for expert `ei`.
                let expert_names = |ei: usize| -> (String, String, String) {
                    if is_mixtral {
                        let ep = format!("{layer}.block_sparse_moe.experts.{ei}");
                        (
                            format!("{ep}.w1.weight"),
                            format!("{ep}.w3.weight"),
                            format!("{ep}.w2.weight"),
                        )
                    } else {
                        let ep = format!("{layer}.mlp.experts.{ei}");
                        (
                            format!("{ep}.gate_proj.weight"),
                            format!("{ep}.up_proj.weight"),
                            format!("{ep}.down_proj.weight"),
                        )
                    }
                };

                if use_ep {
                    // Expert parallelism: assign whole experts to ranks.
                    let experts_per_rank = config.num_experts / ws;
                    let start = device.rank * experts_per_rank;
                    let indices: Vec<usize> = (start..start + experts_per_rank).collect();

                    for &ei in &indices {
                        let (gate, up, down) = expert_names(ei);
                        add(
                            gate,
                            SplitDimension::ExpertParallel {
                                expert_indices: indices.clone(),
                            },
                            [moe_inter, hidden],
                        );
                        add(
                            up,
                            SplitDimension::ExpertParallel {
                                expert_indices: indices.clone(),
                            },
                            [moe_inter, hidden],
                        );
                        add(
                            down,
                            SplitDimension::ExpertParallel {
                                expert_indices: indices.clone(),
                            },
                            [hidden, moe_inter],
                        );
                    }
                } else {
                    // Tensor parallelism for experts: split each expert's weights.
                    for ei in 0..config.num_experts {
                        let (gate, up, down) = expert_names(ei);
                        add(gate, SplitDimension::Column, [moe_inter, hidden]);
                        add(up, SplitDimension::Column, [moe_inter, hidden]);
                        add(down, SplitDimension::Row, [hidden, moe_inter]);
                    }
                }

                // Shared expert (Qwen3.5): always replicated (active on every GPU).
                if config.has_shared_expert() {
                    let shared_inter = config.shared_expert_intermediate_size;
                    add(
                        format!("{layer}.mlp.shared_expert.gate_proj.weight"),
                        SplitDimension::Replicated,
                        [shared_inter, hidden],
                    );
                    add(
                        format!("{layer}.mlp.shared_expert.up_proj.weight"),
                        SplitDimension::Replicated,
                        [shared_inter, hidden],
                    );
                    add(
                        format!("{layer}.mlp.shared_expert.down_proj.weight"),
                        SplitDimension::Replicated,
                        [hidden, shared_inter],
                    );
                    // Shared expert gate (scalar weight).
                    add(
                        format!("{layer}.mlp.shared_expert_gate.weight"),
                        SplitDimension::Replicated,
                        [1, 1],
                    );
                }
            }

            // -- DeltaNet (Qwen 3.5 linear attention layers) --
            if config.is_hybrid_deltanet() {
                let lk_dim = config.linear_num_key_heads * config.linear_key_head_dim;
                let lv_dim = config.linear_num_value_heads * config.linear_value_head_dim;
                let lqkv_dim = lk_dim + lk_dim + lv_dim; // Q + K + V fused

                // in_proj_qkv: [qkv_dim, hidden] → Column (split by heads)
                add(
                    format!("{layer}.linear_attn.in_proj_qkv.weight"),
                    SplitDimension::Column,
                    [lqkv_dim, hidden],
                );
                // in_proj_a, in_proj_b: per-V-head gates → Column
                add(
                    format!("{layer}.linear_attn.in_proj_a.weight"),
                    SplitDimension::Column,
                    [config.linear_num_value_heads, hidden],
                );
                add(
                    format!("{layer}.linear_attn.in_proj_b.weight"),
                    SplitDimension::Column,
                    [config.linear_num_value_heads, hidden],
                );
                // in_proj_z: output gate, split by v_dim
                add(
                    format!("{layer}.linear_attn.in_proj_z.weight"),
                    SplitDimension::Column,
                    [lv_dim, hidden],
                );
                // out_proj: [hidden, v_dim] → Row (requires AllReduce)
                add(
                    format!("{layer}.linear_attn.out_proj.weight"),
                    SplitDimension::Row,
                    [hidden, lv_dim],
                );
                // conv1d_weight: depthwise, channels = conv_dim → Column
                // Safetensors shape is [conv_dim, 1, kernel_size] but we treat as
                // [conv_dim, kernel_size] for 2D slicing (middle dim is 1).
                add(
                    format!("{layer}.linear_attn.conv1d.weight"),
                    SplitDimension::Column,
                    [lqkv_dim, config.linear_conv_kernel_dim],
                );
                // A_log, dt_bias: per-V-head parameters → Column
                add(
                    format!("{layer}.linear_attn.A_log"),
                    SplitDimension::Column,
                    [config.linear_num_value_heads, 1],
                );
                add(
                    format!("{layer}.linear_attn.dt_bias"),
                    SplitDimension::Column,
                    [config.linear_num_value_heads, 1],
                );
                // norm: per-head-dim, shared → Replicated
                add(
                    format!("{layer}.linear_attn.norm.weight"),
                    SplitDimension::Replicated,
                    [config.linear_value_head_dim, 1],
                );
            }

            // -- Gemma3 sandwich norms (replicated) --
            add(
                format!("{layer}.pre_feedforward_layernorm.weight"),
                SplitDimension::Replicated,
                [hidden, 1],
            );
            add(
                format!("{layer}.post_feedforward_layernorm.weight"),
                SplitDimension::Replicated,
                [hidden, 1],
            );

            // -- Qwen3.5 output gate z_proj --
            if config.is_hybrid_deltanet() {
                add(
                    format!("{layer}.self_attn.z_proj.weight"),
                    SplitDimension::Column,
                    [q_dim, hidden],
                );
            }
        }

        // -- Embedding and LM head --
        let embed_split = if vocab_parallel {
            SplitDimension::VocabParallel
        } else {
            SplitDimension::Replicated
        };
        let lm_head_split = embed_split.clone();

        Ok(ShardingPlan {
            device,
            weights,
            embed_split,
            lm_head_split,
        })
    }

    /// Look up the sharding for a weight by name.
    ///
    /// Returns None for weights not in the plan (which means Replicated
    /// for world_size > 1, or the plan was derived for world_size = 1).
    pub fn get(&self, name: &str) -> Option<&WeightSharding> {
        self.weights.iter().find(|w| w.name == name)
    }
}

// ---------------------------------------------------------------------------
// Byte-slicing helpers — extract a rank's shard from raw tensor bytes.
// ---------------------------------------------------------------------------

/// Compute the shard shape for a given split, original shape, and world_size.
fn shard_shape(split: &SplitDimension, orig: [usize; 2], ws: usize) -> [usize; 2] {
    match split {
        SplitDimension::Column => [orig[0] / ws, orig[1]],
        SplitDimension::Row => [orig[0], orig[1] / ws],
        SplitDimension::Replicated => orig,
        SplitDimension::VocabParallel => [orig[0] / ws, orig[1]],
        SplitDimension::ExpertParallel { .. } => orig, // whole expert, not split
    }
}

/// Slice raw tensor bytes according to the sharding plan.
///
/// Learning note: weight tensors in safetensors are stored as contiguous
/// row-major [out_dim, in_dim] arrays of bf16 values (2 bytes each).
///
/// Column-split (split out_dim):
///   Easy — take a contiguous slice of rows.
///   rows_per_rank = out_dim / world_size
///   offset = rank * rows_per_rank * in_dim * bytes_per_elem
///   length = rows_per_rank * in_dim * bytes_per_elem
///
/// Row-split (split in_dim):
///   Harder — must extract a COLUMN slice from each row.
///   For each of the out_dim rows, copy in_dim/world_size elements
///   starting at rank * (in_dim/world_size) * bytes_per_elem.
///   Result is [out_dim, in_dim/world_size].
///
/// `bytes_per_elem` is typically 2 (bf16).
pub(crate) fn slice_tensor_data(
    data: &[u8],
    shape: &[usize; 2],
    split: &SplitDimension,
    rank: usize,
    world_size: usize,
    bytes_per_elem: usize,
) -> (Vec<u8>, [usize; 2]) {
    let [out_dim, in_dim] = *shape;

    match split {
        SplitDimension::Column | SplitDimension::VocabParallel => {
            // Contiguous row slice.
            let rows_per_rank = out_dim / world_size;
            let row_bytes = in_dim * bytes_per_elem;
            let offset = rank * rows_per_rank * row_bytes;
            let length = rows_per_rank * row_bytes;
            let sliced = data[offset..offset + length].to_vec();
            (sliced, [rows_per_rank, in_dim])
        }
        SplitDimension::Row => {
            // Stride across rows, extracting column slices.
            let cols_per_rank = in_dim / world_size;
            let col_offset = rank * cols_per_rank * bytes_per_elem;
            let col_bytes = cols_per_rank * bytes_per_elem;
            let row_bytes = in_dim * bytes_per_elem;
            let mut out = Vec::with_capacity(out_dim * col_bytes);
            for row in 0..out_dim {
                let row_start = row * row_bytes + col_offset;
                out.extend_from_slice(&data[row_start..row_start + col_bytes]);
            }
            (out, [out_dim, cols_per_rank])
        }
        SplitDimension::Replicated | SplitDimension::ExpertParallel { .. } => {
            // No slicing needed.
            (data.to_vec(), *shape)
        }
    }
}

// ===========================================================================
// Tests for sharding plan derivation and tensor slicing.
//
// These test the PLAN LOGIC (which weights get which split, are shapes
// correct) — not the GPU kernels.  They run on any platform via
// `cargo test` with no GPU required.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a minimal ModelConfig for testing.
    fn test_config(
        num_heads: usize,
        num_kv_heads: usize,
        hidden_size: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> ModelConfig {
        let mut config: ModelConfig = serde_json::from_value(serde_json::json!({
            "model_type": "llama",
            "hidden_size": hidden_size,
            "num_hidden_layers": 2,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
            "vocab_size": 32000,
        }))
        .unwrap();
        // weight_prefix is #[serde(skip)] — set manually for tests.
        config.weight_prefix = "model.".to_string();
        config
    }

    /// Helper: create a MoE config for testing.
    fn test_moe_config(num_experts: usize) -> ModelConfig {
        let mut config: ModelConfig = serde_json::from_value(serde_json::json!({
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 0,
            "vocab_size": 152064,
            "num_experts": num_experts,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
        }))
        .unwrap();
        config.weight_prefix = "model.".to_string();
        config
    }

    #[test]
    fn test_single_gpu_plan() {
        let config = test_config(32, 8, 2048, 64, 8192);
        let device = DeviceConfig::single();
        let plan = ShardingPlan::derive(&config, device, false).unwrap();

        // world_size=1 → no weight entries, everything is implicitly replicated.
        assert!(plan.weights.is_empty());
        assert_eq!(plan.embed_split, SplitDimension::Replicated);
        assert_eq!(plan.lm_head_split, SplitDimension::Replicated);
    }

    #[test]
    fn test_tp2_llama_plan() {
        // Llama 3.2 1B: 32 heads, 8 kv_heads, head_dim=64, hidden=2048, inter=8192.
        let config = test_config(32, 8, 2048, 64, 8192);
        let device = DeviceConfig {
            world_size: 2,
            rank: 0,
            strategy: ParallelStrategy::TensorParallel,
        };
        let plan = ShardingPlan::derive(&config, device, false).unwrap();

        // q_proj: [2048, 2048] → Column → shard [1024, 2048]
        let q = plan.get("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(q.split, SplitDimension::Column);
        assert_eq!(q.original_shape, [2048, 2048]);
        assert_eq!(q.shard_shape, [1024, 2048]);

        // k_proj: [512, 2048] → Column → shard [256, 2048]
        let k = plan.get("model.layers.0.self_attn.k_proj.weight").unwrap();
        assert_eq!(k.split, SplitDimension::Column);
        assert_eq!(k.shard_shape, [256, 2048]);

        // o_proj: [2048, 2048] → Row → shard [2048, 1024]
        let o = plan.get("model.layers.0.self_attn.o_proj.weight").unwrap();
        assert_eq!(o.split, SplitDimension::Row);
        assert_eq!(o.shard_shape, [2048, 1024]);

        // gate_proj: [8192, 2048] → Column → shard [4096, 2048]
        let gate = plan.get("model.layers.0.mlp.gate_proj.weight").unwrap();
        assert_eq!(gate.split, SplitDimension::Column);
        assert_eq!(gate.shard_shape, [4096, 2048]);

        // down_proj: [2048, 8192] → Row → shard [2048, 4096]
        let down = plan.get("model.layers.0.mlp.down_proj.weight").unwrap();
        assert_eq!(down.split, SplitDimension::Row);
        assert_eq!(down.shard_shape, [2048, 4096]);

        // Norms: replicated
        let norm = plan.get("model.layers.0.input_layernorm.weight").unwrap();
        assert_eq!(norm.split, SplitDimension::Replicated);
    }

    #[test]
    fn test_tp4_moe_expert_parallel() {
        // 128 experts, world_size=4 → each rank gets 32 experts.
        let config = test_moe_config(128);
        let device = DeviceConfig {
            world_size: 4,
            rank: 1,
            strategy: ParallelStrategy::ExpertParallel,
        };
        let plan = ShardingPlan::derive(&config, device, false).unwrap();

        // Rank 1 should own experts 32..63.
        let expert_32 = plan
            .get("model.layers.0.mlp.experts.32.gate_proj.weight")
            .unwrap();
        if let SplitDimension::ExpertParallel { expert_indices } = &expert_32.split {
            assert_eq!(expert_indices.len(), 32);
            assert_eq!(expert_indices[0], 32);
            assert_eq!(expert_indices[31], 63);
        } else {
            panic!("expected ExpertParallel split for expert 32");
        }

        // Expert 0 should NOT be in rank 1's plan.
        assert!(
            plan.get("model.layers.0.mlp.experts.0.gate_proj.weight")
                .is_none()
        );

        // Router gate is always replicated.
        let router = plan.get("model.layers.0.mlp.gate.weight").unwrap();
        assert_eq!(router.split, SplitDimension::Replicated);
    }

    #[test]
    fn test_validation_indivisible_heads() {
        // 7 heads can't be split across 2 GPUs.
        let config = test_config(7, 7, 448, 64, 2048);
        let device = DeviceConfig {
            world_size: 2,
            rank: 0,
            strategy: ParallelStrategy::TensorParallel,
        };
        let result = ShardingPlan::derive(&config, device, false);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("num_attention_heads"));
        assert!(err_msg.contains("divisible"));
    }

    #[test]
    fn test_validation_indivisible_kv_heads() {
        // 32 query heads OK, but 3 KV heads can't split across 2.
        let config = test_config(32, 3, 2048, 64, 8192);
        let device = DeviceConfig {
            world_size: 2,
            rank: 0,
            strategy: ParallelStrategy::TensorParallel,
        };
        let result = ShardingPlan::derive(&config, device, false);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("num_key_value_heads")
        );
    }

    #[test]
    fn test_validation_indivisible_intermediate() {
        // intermediate_size=11008 not divisible by 3.
        let config = test_config(12, 12, 768, 64, 11008);
        let device = DeviceConfig {
            world_size: 3,
            rank: 0,
            strategy: ParallelStrategy::TensorParallel,
        };
        let result = ShardingPlan::derive(&config, device, false);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("intermediate_size")
        );
    }

    #[test]
    fn test_slice_column() {
        // 4x4 bf16 matrix (2 bytes per element), world_size=2.
        // Row 0: [0, 1, 2, 3], Row 1: [4, 5, 6, 7],
        // Row 2: [8, 9, 10, 11], Row 3: [12, 13, 14, 15]
        let mut data = Vec::new();
        for i in 0u16..16 {
            data.extend_from_slice(&i.to_le_bytes());
        }

        // Rank 0 gets rows 0-1: [0,1,2,3, 4,5,6,7]
        let (sliced, shape) = slice_tensor_data(&data, &[4, 4], &SplitDimension::Column, 0, 2, 2);
        assert_eq!(shape, [2, 4]);
        assert_eq!(sliced.len(), 2 * 4 * 2); // 2 rows × 4 cols × 2 bytes

        // First element of rank 0 should be 0.
        assert_eq!(u16::from_le_bytes([sliced[0], sliced[1]]), 0);
        // Last element of rank 0 should be 7.
        assert_eq!(u16::from_le_bytes([sliced[14], sliced[15]]), 7);

        // Rank 1 gets rows 2-3: [8,9,10,11, 12,13,14,15]
        let (sliced, shape) = slice_tensor_data(&data, &[4, 4], &SplitDimension::Column, 1, 2, 2);
        assert_eq!(shape, [2, 4]);
        assert_eq!(u16::from_le_bytes([sliced[0], sliced[1]]), 8);
        assert_eq!(u16::from_le_bytes([sliced[14], sliced[15]]), 15);
    }

    #[test]
    fn test_slice_row() {
        // 4x4 bf16 matrix, world_size=2.
        // Rank 0 gets cols 0-1 of each row, rank 1 gets cols 2-3.
        let mut data = Vec::new();
        for i in 0u16..16 {
            data.extend_from_slice(&i.to_le_bytes());
        }

        // Rank 0 should get: [0,1, 4,5, 8,9, 12,13]
        let (sliced, shape) = slice_tensor_data(&data, &[4, 4], &SplitDimension::Row, 0, 2, 2);
        assert_eq!(shape, [4, 2]);
        assert_eq!(sliced.len(), 4 * 2 * 2); // 4 rows × 2 cols × 2 bytes

        let vals: Vec<u16> = sliced
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(vals, vec![0, 1, 4, 5, 8, 9, 12, 13]);

        // Rank 1 should get: [2,3, 6,7, 10,11, 14,15]
        let (sliced, _) = slice_tensor_data(&data, &[4, 4], &SplitDimension::Row, 1, 2, 2);
        let vals: Vec<u16> = sliced
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(vals, vec![2, 3, 6, 7, 10, 11, 14, 15]);
    }

    #[test]
    fn test_vocab_parallel() {
        let config = test_config(32, 8, 2048, 64, 8192);
        let device = DeviceConfig {
            world_size: 2,
            rank: 0,
            strategy: ParallelStrategy::TensorParallel,
        };
        let plan = ShardingPlan::derive(&config, device, true).unwrap();
        assert_eq!(plan.embed_split, SplitDimension::VocabParallel);
        assert_eq!(plan.lm_head_split, SplitDimension::VocabParallel);
    }

    /// Helper: create a Qwen 3.5-style hybrid DeltaNet config for testing.
    fn test_qwen35_config() -> ModelConfig {
        let mut config: ModelConfig = serde_json::from_value(serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 8192,
            "vocab_size": 152064,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        }))
        .unwrap();
        config.weight_prefix = "model.".to_string();
        config.layer_types = vec!["linear_attention".into(), "full_attention".into()];
        config
    }

    #[test]
    fn test_tp2_qwen35_deltanet_plan() {
        let config = test_qwen35_config();
        assert!(config.is_hybrid_deltanet());

        let device = DeviceConfig {
            world_size: 2,
            rank: 0,
            strategy: ParallelStrategy::TensorParallel,
        };
        let plan = ShardingPlan::derive(&config, device, false).unwrap();

        // DeltaNet weight names must use `linear_attn` prefix (matching safetensors).
        let qkv = plan
            .get("model.layers.0.linear_attn.in_proj_qkv.weight")
            .unwrap();
        assert_eq!(qkv.split, SplitDimension::Column);
        // Q(4*128) + K(4*128) + V(4*128) = 1536 → shard = 768
        assert_eq!(qkv.original_shape, [1536, 2048]);
        assert_eq!(qkv.shard_shape, [768, 2048]);

        // in_proj_a: [num_value_heads=4, hidden=2048] → Column → shard [2, 2048]
        let a = plan
            .get("model.layers.0.linear_attn.in_proj_a.weight")
            .unwrap();
        assert_eq!(a.split, SplitDimension::Column);
        assert_eq!(a.original_shape, [4, 2048]);
        assert_eq!(a.shard_shape, [2, 2048]);

        // in_proj_b: same as in_proj_a
        let b = plan
            .get("model.layers.0.linear_attn.in_proj_b.weight")
            .unwrap();
        assert_eq!(b.split, SplitDimension::Column);
        assert_eq!(b.shard_shape, [2, 2048]);

        // in_proj_z: [v_dim=512, hidden=2048] → Column → shard [256, 2048]
        let z = plan
            .get("model.layers.0.linear_attn.in_proj_z.weight")
            .unwrap();
        assert_eq!(z.split, SplitDimension::Column);
        assert_eq!(z.original_shape, [512, 2048]);
        assert_eq!(z.shard_shape, [256, 2048]);

        // out_proj: [hidden=2048, v_dim=512] → Row → shard [2048, 256]
        let out = plan
            .get("model.layers.0.linear_attn.out_proj.weight")
            .unwrap();
        assert_eq!(out.split, SplitDimension::Row);
        assert_eq!(out.original_shape, [2048, 512]);
        assert_eq!(out.shard_shape, [2048, 256]);

        // Verify `deltanet` prefix does NOT appear in plan weight names.
        for ws in &plan.weights {
            assert!(
                !ws.name.contains(".deltanet."),
                "plan key '{}' uses wrong prefix — should be 'linear_attn'",
                ws.name
            );
        }

        // Both layers should have standard attention entries.
        assert!(plan.get("model.layers.1.self_attn.q_proj.weight").is_some());
    }

    /// Helper: create a Mixtral-style MoE config for testing.
    fn test_mixtral_moe_config() -> ModelConfig {
        let mut config: ModelConfig = serde_json::from_value(serde_json::json!({
            "model_type": "mixtral",
            "hidden_size": 4096,
            "num_hidden_layers": 2,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 0,
            "vocab_size": 32000,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 14336,
        }))
        .unwrap();
        config.weight_prefix = "model.".to_string();
        config
    }

    #[test]
    fn test_tp2_mixtral_moe_sharding() {
        let config = test_mixtral_moe_config();
        assert!(config.is_moe());

        let device = DeviceConfig {
            world_size: 2,
            rank: 0,
            strategy: ParallelStrategy::TensorParallel,
        };
        let plan = ShardingPlan::derive(&config, device, false).unwrap();

        // Expert weights must use Mixtral naming (block_sparse_moe, w1/w3/w2).
        let w1 = plan
            .get("model.layers.0.block_sparse_moe.experts.0.w1.weight")
            .unwrap();
        assert_eq!(w1.split, SplitDimension::Column);
        assert_eq!(w1.original_shape, [14336, 4096]);
        assert_eq!(w1.shard_shape, [7168, 4096]);

        let w3 = plan
            .get("model.layers.0.block_sparse_moe.experts.0.w3.weight")
            .unwrap();
        assert_eq!(w3.split, SplitDimension::Column);

        let w2 = plan
            .get("model.layers.0.block_sparse_moe.experts.0.w2.weight")
            .unwrap();
        assert_eq!(w2.split, SplitDimension::Row);
        assert_eq!(w2.original_shape, [4096, 14336]);
        assert_eq!(w2.shard_shape, [4096, 7168]);

        // All 8 experts should be in the plan.
        for ei in 0..8 {
            assert!(
                plan.get(&format!(
                    "model.layers.0.block_sparse_moe.experts.{ei}.w1.weight"
                ))
                .is_some(),
                "missing expert {ei} w1 in plan"
            );
        }

        // Generic mlp.experts naming must NOT appear.
        assert!(
            plan.get("model.layers.0.mlp.experts.0.gate_proj.weight")
                .is_none(),
            "Mixtral plan should not use generic mlp.experts naming"
        );

        // Router gate uses Mixtral naming.
        let router = plan
            .get("model.layers.0.block_sparse_moe.gate.weight")
            .unwrap();
        assert_eq!(router.split, SplitDimension::Replicated);

        // Generic mlp.gate must NOT appear.
        assert!(plan.get("model.layers.0.mlp.gate.weight").is_none());
    }
}
