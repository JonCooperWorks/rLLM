# Model Layer

The model layer handles configuration parsing, weight loading, architecture
dispatch, and the forward pass for 9 model families.  It sits between the
inference engine (which manages scheduling) and the GPU backend (which
executes kernels).

**Key files:**
- `src/model/mod.rs` — `Model<B>` struct, forward dispatch, buffer allocation
- `src/model/config.rs` — `ModelArch` enum, `ModelConfig` (from HF `config.json`)
- `src/model/loader/` — safetensors weight loading, Q4 quantization on-load
- `src/model/primitives.rs` — shared transformer building blocks
- `src/model/registry/*.rs` — 9 model family forward passes
- `src/model/chat.rs` — chat template formatting
- `src/model/tools.rs` — tool/function calling support
- `src/model/sampler.rs` — temperature + top-p sampling

---

## Model Config

`ModelConfig` is deserialized from a Hugging Face `config.json`.  The
`ModelArch` enum identifies the architecture:

| Variant | Model Family | Forward Pass |
|---------|-------------|-------------|
| `Llama` | Llama 3.x | `registry/llama.rs` |
| `Qwen2` | Qwen 2.5 | `registry/qwen.rs` (delegates to llama via ArchFeatures) |
| `Qwen3Moe` | Qwen 3 MoE | `registry/qwen3_moe.rs` |
| `Qwen3_5` | Qwen 3.5 hybrid | `registry/qwen3_5.rs` |
| `Phi` | Phi | `registry/phi.rs` (delegates to llama via ArchFeatures) |
| `Gemma3` | Gemma 3 | `registry/gemma.rs` |
| `Mistral` | Mistral | `registry/mistral.rs` (delegates to llama via ArchFeatures) |
| `Mixtral` | Mixtral MoE | `registry/mixtral.rs` |
| `GptOss` | GPT Open Source | `registry/gpt_oss.rs` |

The config provides all dimension info: `num_hidden_layers`, `hidden_size`,
`num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`,
`vocab_size`, `rope_theta`, `rms_norm_eps`, and architecture-specific flags.

---

## Weight Loading

`src/model/loader/` loads weights from safetensors format:

1. **TensorStore** abstracts single-file vs multi-shard models
2. **LoaderHints** encodes per-architecture boolean flags:
   - `has_qkv_bias` — Qwen 2.5 uses bias in QKV projections
   - `has_tied_embeddings` — Llama 3.2 1B shares embed/lm_head weights
   - `has_qk_norm` — Qwen 3 MoE normalizes Q and K
3. Helper functions: `load_attention_weights()`, `load_ffn_weights()`, `load_layer_norms()`
4. **Q4 weights**: pre-quantized Q4 models (produced by `rllm quantize`) are
   loaded directly — the loader detects the Q4 format and uses the appropriate
   upload path

Novel weight formats get dedicated helpers (e.g., `load_mxfp4_experts()` for
GPT-OSS's MXFP4 expert weights).

---

## Model Struct

`Model<'a, B>` holds everything needed for a forward pass:

| Field | Contents |
|-------|---------|
| `weights` | All model weights (embeddings, attention, FFN, norms, lm_head) |
| `config` | `ModelConfig` |
| `backend` | `&'a B` (borrowed reference to GPU backend) |
| `dims` | `Dims` — pre-computed u32 dimension constants |
| Pre-allocated buffers | `hidden`, `norm_buf`, `q_buf`, `k_buf`, `v_buf`, `attn_out`, `gate_buf`, `up_buf`, `logits_buf` |
| MoE buffers | `router_logits`, `moe_gate_buf`, `moe_up_buf`, `moe_output`, `routing_output` |
| DeltaNet state | `deltanet_states` (recurrent matrices), `deltanet_conv_history` |

Buffers are allocated once at model load time and reused every step.  This
eliminates per-step allocation overhead.

---

## Forward Pass Dispatch

The engine holds a `Box<dyn ModelForward<B>>` constructed once at load time in
`engine/loader.rs::create_forward()`.  Each architecture implements the
`ModelForward` trait (defined in `model/forward.rs`) with `forward_decode`,
`forward_prefill`, and optionally `forward_decode_batch`.

```rust
// engine/loader.rs — one match at construction time
let forward: Box<dyn ModelForward<B>> = match arch {
    ModelArch::Llama => Box::new(LlamaForward::new(false)),
    ModelArch::Qwen2 => Box::new(LlamaForward::new(true)),  // QKV bias
    ModelArch::Mixtral => Box::new(MixtralForward { moe }),
    // ... etc
};
```

No match dispatch at runtime — the trait vtable handles it.

---

## Primitives

`src/model/primitives.rs` contains shared building blocks that model families
compose.  Each primitive declares minimal trait bounds:

| Primitive | What it does | Trait bound |
|-----------|-------------|-------------|
| `embed_token` | Token → embedding vector | `GpuEmbed` |
| `qkv_projection` | Hidden → Q, K, V matrices | `GpuMatmul` |
| `apply_rope` | Apply rotary positional encoding | `GpuRope` |
| `paged_kv_and_attention` | Write KV cache + compute attention | `GpuAttention` |
| `fused_ffn` | Gate-up projection → activation → down projection | `GpuMatmul + GpuElementwise` |
| `final_norm_and_lm_head` | RMS norm → logits projection | `GpuNorm + GpuMatmul` |
| `apply_all_reduce` | Tensor parallel sync | `GpuAllReduce` |

The `Dims` struct pre-computes all dimension constants as `u32` to avoid
repeated casts in hot loops.

---

## ArchFeatures Pattern

Many model families are structurally identical to Llama with minor variations.
Instead of duplicating the forward pass, they delegate to `llama.rs` with
configuration flags:

```rust
struct ArchFeatures {
    has_qkv_bias: bool,        // Qwen 2.5
    has_tied_embeddings: bool,  // Llama 3.2 1B
    has_qk_norm: bool,         // Qwen 3 MoE
    rope_variant: RopeVariant, // standard, partial, YaRN
}
```

Qwen 2.5, Phi, and Mistral all use this pattern — their registry files are
thin wrappers that construct the right `ArchFeatures` and call into the shared
Llama forward pass.

---

## Specialized Architectures

### Mixture-of-Experts (MoE)

Qwen 3 MoE, Mixtral, and GPT-OSS use sparse expert routing:

1. Router projects hidden state → expert logits
2. Top-K experts selected per token (K=2 for Qwen3/Mixtral, K=4 for GPT-OSS)
3. Selected experts' FFN weights applied, outputs weighted-summed
4. `GpuElementwise::top_k_softmax()` handles the routing kernel

### Qwen 3.5 Hybrid (DeltaNet + GQA)

75% of layers use DeltaNet linear attention (O(1) per-step via recurrent
state), 25% use standard GQA softmax attention.  This requires:

- `GpuDeltaNet` trait for conv1d, L2 norm, decay gates, state update kernels
- Persistent `deltanet_states` (recurrent matrix per layer) across steps
- `deltanet_conv_history` for the causal convolution

### GPT-OSS (MXFP4 Experts)

Uses MXFP4 (microscaling FP4) format for expert weights — a different
quantization scheme from Q4, with per-block scaling factors.  The loader
has a dedicated `load_mxfp4_experts()` path.

---

## Chat Templates

`src/model/chat.rs` formats conversation messages for each architecture:

| Architecture | Template Style | Markers |
|-------------|---------------|---------|
| Llama 3 | Llama format | `<\|start_header_id\|>`, `<\|eot_id\|>` |
| Qwen, Qwen3, Qwen3.5, GPT-OSS | ChatML | `<\|im_start\|>`, `<\|im_end\|>` |
| Mistral, Mixtral | Mistral format | `[INST]`, `[/INST]` |
| Phi | Phi format | `<\|im_start\|>`, `<\|im_sep\|>`, `<\|im_end\|>` |
| Gemma 3 | Gemma format | `<start_of_turn>`, `<end_of_turn>` |

---

## Tool Calling

`src/model/tools.rs` supports function/tool calling for API endpoints:

- `format_tool_system_prompt()` — injects tool definitions into the system
  prompt using the architecture's expected format
- `parse_tool_calls()` — extracts structured tool calls from model output
- Supports Llama 3.1, Qwen, Mistral, and Anthropic tool-call formats

---

## Adding a New Model

1. Add `ModelArch` variant in `config.rs` + `from_model_type()` match
2. Set loader flags in `LoaderHints::new()` in `loader/mod.rs`
3. Create `registry/new_model.rs` implementing `ModelForward` trait (or use `LlamaForward`)
4. Add match arm in `engine/loader.rs::create_forward()` to construct the Forward struct
5. Add chat template in `chat.rs`
6. Add tool-call format in `tools.rs` (if applicable)

---

See also: [Architecture Overview](architecture-overview.md) ·
[GPU Backend](gpu-backend.md) · [KV Cache](kv-cache.md)
