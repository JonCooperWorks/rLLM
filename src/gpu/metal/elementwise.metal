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
