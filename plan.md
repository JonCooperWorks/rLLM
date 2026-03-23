# Vision (VLM) Support — Implementation Plan

## Target: Gemma 3 VLM (SigLIP vision encoder)

Gemma 3 is the best starting point because:
- Text model already fully supported with sliding window, sandwich norms, GeGLU
- VLM config detection and weight prefix (`language_model.model.`) already implemented
- SigLIP is a clean, well-documented vision encoder (simpler than some alternatives)
- Single image support is sufficient for a first pass

## Architecture Overview

```
Image (448×448 RGB) → SigLIP Vision Encoder → Projection → Image Tokens
                                                              ↓
Text Tokens → Embed Lookup ──────────────────→ Merge ──→ LLM Forward Pass
```

Gemma 3 VLM uses a SigLIP-based vision encoder:
- Patch embedding: 14×14 patches from 448×448 image → 1024 patch tokens
- Vision transformer: LayerNorm (not RMSNorm), standard multi-head attention, GELU MLP
- Multi-modal projector: linear projection from vision hidden dim → LLM hidden dim

---

## Steps

### 1. Vision config parsing (`src/model/config.rs`)
- Add `VisionConfig` struct (image_size, patch_size, num_layers, hidden_size, intermediate_size, num_heads, projection_dim)
- Parse `vision_config` from Gemma 3 VLM config.json (currently discarded)
- Store on `ModelConfig` as `pub vision_config: Option<VisionConfig>`

### 2. Vision weight loading (`src/model/loader.rs`)
- Add `VisionWeights<B>` struct: patch_embed conv weights, LayerNorm params, attention QKV/O weights, MLP weights, projection layer
- Load from safetensors under `vision_tower.` or `vision_model.` prefix (Gemma 3 uses SigLIP weights)
- Load multi-modal projector weights (`multi_modal_projector.linear.weight/bias`)
- Store as `Option<VisionWeights<B>>` on `ModelWeights`

### 3. GPU vision ops trait (`src/gpu/ops/vision.rs`)
- New `GpuVision: GpuCore` trait with:
  - `patch_embed()` — 2D conv (kernel=14, stride=14) to extract patches + flatten
  - `layernorm()` — standard LayerNorm (vision uses LN, not RMSNorm)
  - `vision_attention()` — multi-head self-attention (no KV cache, no RoPE, no sliding window)
  - `gelu()` — GELU activation for vision MLP (if not already in GpuElementwise)
- Add to `GpuBackend` supertrait bound in `src/gpu/mod.rs`

### 4. Metal shaders (`src/gpu/metal/shaders/vision.metal`)
- Patch embedding kernel (2D conv, or restructured as matmul over flattened patches)
- LayerNorm kernel
- Vision attention kernel (simpler than LLM attention — no paging, no RoPE)
- Reuse existing GELU/matmul kernels where possible

### 5. Metal backend impl (`src/gpu/metal/kernels/vision.rs`)
- Implement `GpuVision for MetalBackend`
- Add pipeline compilation in `backend.rs`

### 6. CUDA stub (`src/gpu/cuda/mod.rs`)
- Add stub `GpuVision` impl (panic/unimplemented)

### 7. CPU backend (`src/gpu/cpu/mod.rs`)
- Add reference `GpuVision` impl for testing

### 8. Vision forward pass (`src/model/registry/gemma.rs`)
- Add `vision_forward()` function: patch_embed → N transformer blocks → projection
- Add `merge_vision_text()`: replace `<image>` placeholder tokens with vision embeddings in the hidden state buffer

### 9. Integrate into model forward pass (`src/model/mod.rs`)
- In `forward_prefill_paged()`, before the LLM layers:
  - If vision weights present and image data provided, run vision encoder
  - Merge vision tokens into the hidden state at image placeholder positions
- Pass image data through a new field on the prefill call path

### 10. API: multimodal message content (`src/model/chat.rs`, `src/api/openai.rs`)
- Extend `Message` to accept OpenAI-style multimodal content:
  ```json
  {"role": "user", "content": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}
  ```
- Add `ContentPart` enum (Text / ImageUrl)
- Custom deserializer: accept both `String` and `Vec<ContentPart>` for backwards compat
- Image preprocessing: decode base64 → resize to model's image_size → normalize ([0,1] with mean/std)

### 11. Image preprocessing
- Add `image` crate dependency to Cargo.toml (for decode + resize)
- Normalize pixel values: `(pixel / 255.0 - mean) / std`
- Convert to tensor: [3, H, W] float buffer → GPU tensor

---

## Key Design Decisions

- **Gemma 3 first**, Qwen 3.5 second (different vision encoder but same integration pattern)
- **Prefill-only vision**: images are processed during prefill, not decode (standard approach)
- **Single image first**, multi-image later
- **Base64 input only** initially (no URL fetching — avoids network complexity)
- **LayerNorm as a new kernel** rather than hacking RMSNorm (they're mathematically different)
- **2D conv as matmul**: flatten image patches and use existing GEMM kernels for patch embedding (avoids writing a full conv2d kernel)
