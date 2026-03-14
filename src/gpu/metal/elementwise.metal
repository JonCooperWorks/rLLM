// ===========================================================================
// Element-wise GPU kernels: SwiGLU activation and tensor addition.
//
// LEARNING OVERVIEW
//
// These are the simplest kind of GPU kernel: each thread processes exactly
// one element independently.  No shared memory, no barriers, no reductions.
// Metal's dispatch_threads creates N threads and each one reads its inputs,
// computes, and writes its output.
//
// SwiGLU (Swish-Gated Linear Unit):
//   The FFN (feed-forward network) in Llama uses the SwiGLU activation:
//     FFN(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down
//
//   where silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
//
//   This is a "gated" activation: the gate projection controls how much
//   of the up projection passes through.  SwiGLU was shown to outperform
//   ReLU and GELU in transformer FFNs (Shazeer, 2020).
//
// Residual connections:
//   The add kernel implements the residual (skip) connections in the
//   transformer:  output = input + sublayer_output.  Residual connections
//   prevent the vanishing gradient problem in deep networks by providing
//   a direct path for gradients to flow backwards.
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// Host → GPU parameter block.  Must match Rust `ElemParams`.
struct ElemParams {
    uint size; // Number of elements to process.
};

// ---------------------------------------------------------------------------
// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// All arithmetic in float32 for precision, narrowed to bfloat16 on output.
// ---------------------------------------------------------------------------

kernel void silu_mul(
    constant ElemParams& params [[buffer(0)]],
    device const bfloat* gate   [[buffer(1)]],
    device const bfloat* up     [[buffer(2)]],
    device bfloat* output       [[buffer(3)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    float g = float(gate[gid]);
    float u = float(up[gid]);
    // silu(g) = g / (1 + exp(-g)) = g * sigmoid(g)
    float silu = g / (1.0f + exp(-g));
    output[gid] = bfloat(silu * u);
}

// ---------------------------------------------------------------------------
// Element-wise addition: out[i] = a[i] + b[i]
//
// Used for residual connections in the transformer:
//   hidden = hidden + attention_output     (after attention)
//   hidden = hidden + ffn_output           (after FFN)
//
// Learning note: aliasing is safe here (a == out or b == out) because each
// thread reads and writes only index `gid`.  No thread reads another
// thread's output, so there are no data races.
// ---------------------------------------------------------------------------

kernel void add_tensors(
    constant ElemParams& params [[buffer(0)]],
    device const bfloat* a      [[buffer(1)]],
    device const bfloat* b      [[buffer(2)]],
    device bfloat* output       [[buffer(3)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    output[gid] = bfloat(float(a[gid]) + float(b[gid]));
}

// ---------------------------------------------------------------------------
// Broadcast bias-add: out[i] = input[i] + bias[i % dim]
//
// Used in batched prefill for Qwen 2.5's QKV bias.  The input tensor is
// [batch_size, dim] (flattened to batch_size * dim elements) and the bias
// is [dim].  Each row of the batch gets the same bias vector added.
//
// This is a "broadcast" because the bias is smaller than the input — it's
// repeated across the batch dimension.  The modulo (i % dim) maps each
// flat index back to the correct bias element for its column.
//
// Why not reuse add_tensors?
//   add_tensors requires both operands to be the SAME size.  The bias is
//   [dim] but the input is [batch_size * dim] — we'd need to tile the bias
//   into a temporary [batch_size, dim] buffer first, which wastes memory
//   and an extra copy.  This kernel handles the broadcast in a single pass.
// ---------------------------------------------------------------------------

struct BiasAddParams {
    uint total;  // batch_size * dim
    uint dim;    // bias vector length (number of columns)
};

kernel void bias_add(
    constant BiasAddParams& params [[buffer(0)]],
    device const bfloat* input     [[buffer(1)]],
    device const bfloat* bias      [[buffer(2)]],
    device bfloat* output          [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= params.total) return;
    output[gid] = bfloat(float(input[gid]) + float(bias[gid % params.dim]));
}

// ---------------------------------------------------------------------------
// Scaled accumulate: dst[i] += scale * src[i]
//
// Used in MoE (Mixture of Experts) to accumulate weighted expert FFN outputs.
// Each expert's output is multiplied by its routing weight and added to the
// running sum.  After all top-k experts are processed, dst contains the
// weighted combination of their outputs.
//
// Learning note: this is the "axpy" operation from BLAS (a*x + y).  In MoE
// with top-k=8 routing, this is called 8 times per layer (once per activated
// expert), so it's not performance-critical — the expert matmuls dominate.
//
// Aliasing note: dst is both read and written (accumulate), which is safe
// because each thread handles exactly one index.
// ---------------------------------------------------------------------------

struct ScaleAddParams {
    uint size;    // number of elements
    float scale;  // multiplier for src
};

kernel void scale_add(
    constant ScaleAddParams& params [[buffer(0)]],
    device bfloat* dst              [[buffer(1)]],
    device const bfloat* src        [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    dst[gid] = bfloat(float(dst[gid]) + params.scale * float(src[gid]));
}

// ---------------------------------------------------------------------------
// Fill tensor with zeros: dst[i] = 0
//
// Used to clear the MoE accumulator buffer before summing expert outputs.
// ---------------------------------------------------------------------------

kernel void fill_zero(
    constant ElemParams& params [[buffer(0)]],
    device bfloat* dst          [[buffer(1)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    dst[gid] = bfloat(0.0f);
}

// ---------------------------------------------------------------------------
// GPU-side top-k selection with softmax for MoE expert routing.
//
// Replaces the CPU-side routing path that required one GPU→CPU sync per layer
// (48 syncs per token for Qwen3-Coder-30B).  The kernel runs on GPU and writes
// results to a buffer that can be read later with a single copy_to_host.
//
// Input:  logits [num_experts] in bf16 (router matmul output)
// Output: [2*k] f32 values — alternating (expert_index_as_f32, routing_weight)
//
// Uses a single thread since num_experts is small (128).  The entire kernel
// runs faster than the GPU→CPU sync overhead it eliminates.
// ---------------------------------------------------------------------------

struct TopKParams {
    uint num_experts;
    uint k;
};

kernel void top_k_softmax(
    constant TopKParams& params  [[buffer(0)]],
    device const bfloat* logits  [[buffer(1)]],
    device float* output         [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    uint n = params.num_experts;
    uint k = params.k;

    // Convert bf16 logits to f32 and find top-k indices.
    // Simple selection: repeatedly find the max and mark it used.
    // For k=8, n=128 this is 8 × 128 = 1024 comparisons — trivial.

    // Use threadgroup memory for the f32 copy (max 256 experts).
    float vals[256];
    for (uint i = 0; i < n && i < 256; i++) {
        vals[i] = float(logits[i]);
    }

    // Find top-k by iteratively selecting the maximum.
    uint top_indices[32];  // max k=32
    float top_logits[32];
    for (uint j = 0; j < k; j++) {
        float best_val = -INFINITY;
        uint best_idx = 0;
        for (uint i = 0; i < n; i++) {
            if (vals[i] > best_val) {
                best_val = vals[i];
                best_idx = i;
            }
        }
        top_indices[j] = best_idx;
        top_logits[j] = best_val;
        vals[best_idx] = -INFINITY;  // Mark as used.
    }

    // Softmax over the top-k logits (normalized routing).
    float max_logit = top_logits[0];
    for (uint j = 1; j < k; j++) {
        max_logit = max(max_logit, top_logits[j]);
    }
    float exp_sum = 0.0f;
    for (uint j = 0; j < k; j++) {
        top_logits[j] = exp(top_logits[j] - max_logit);
        exp_sum += top_logits[j];
    }

    // Write (index, weight) pairs to output.
    for (uint j = 0; j < k; j++) {
        output[2 * j]     = float(top_indices[j]);
        output[2 * j + 1] = top_logits[j] / exp_sum;
    }
}
