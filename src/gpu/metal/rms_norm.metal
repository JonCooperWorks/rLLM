// ===========================================================================
// RMSNorm (Root Mean Square Layer Normalisation) kernel.
//
// LEARNING OVERVIEW
//
// What this kernel does:
//   Normalises a hidden-state vector so its root-mean-square magnitude is ~1,
//   then scales each element by a learned weight:
//
//     out[i] = weight[i] * input[i] / sqrt(mean(input²) + eps)
//
//   RMSNorm is a simplified variant of LayerNorm that drops the mean-centering
//   step.  Llama (and most modern LLMs) use RMSNorm because it is cheaper to
//   compute — only one reduction (sum of squares) instead of two (mean + variance)
//   — and empirically works just as well (Zhang & Sennrich, 2019).
//
// Where it appears in the transformer:
//   Each transformer layer applies RMSNorm twice:
//     1. Before the attention block   (input_layernorm)
//     2. Before the FFN block         (post_attention_layernorm)
//   Plus one final RMSNorm after all layers, before the LM head projection.
//   Total: 2 × 16 layers + 1 = 33 RMSNorm calls per token.
//
// Why a GPU kernel for normalisation?
//   RMSNorm requires a REDUCTION (sum of squares across all 2048 elements)
//   followed by an element-wise scale.  This is the simplest cooperative kernel
//   in the codebase — threads must communicate through shared memory to compute
//   the global sum before any thread can produce its output.
//
// Dispatch model:
//   A single threadgroup of 256 threads processes the entire hidden vector.
//   Each thread handles hidden_size/256 = 8 elements in a strided loop.
//   The reduction uses SIMD-level intrinsics (simd_sum) within each 32-thread
//   SIMD group, then shared memory to combine across the 8 SIMD groups.
//
// Precision:
//   All arithmetic is in float32 — the sum-of-squares accumulator, the rsqrt,
//   and the final multiply are all f32.  Only the final result is narrowed to
//   bfloat16 on store.  This prevents catastrophic precision loss (bf16 has
//   only ~3 significant digits, which would corrupt the normalisation).
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// Host → GPU parameter block.  Must match Rust `RmsNormParams`.
struct RmsNormParams {
    uint hidden_size; // Number of elements to normalise (2048 for Llama 3.2 1B).
    float eps;        // Epsilon for numerical stability (1e-5).
};

kernel void rms_norm(
    constant RmsNormParams& params [[buffer(0)]],
    // buffer(1): input vector [hidden_size] in bfloat16.
    device const bfloat* input     [[buffer(1)]],
    // buffer(2): learned scale weights [hidden_size] in bfloat16.
    device const bfloat* weight    [[buffer(2)]],
    // buffer(3): output vector [hidden_size] in bfloat16.
    device bfloat* output          [[buffer(3)]],
    // Thread index within the single threadgroup (0..255).
    uint tid                       [[thread_position_in_threadgroup]],
    // Threadgroup size (256).
    uint tg_size                   [[threads_per_threadgroup]]
) {
    const uint hidden = params.hidden_size;

    // -----------------------------------------------------------------------
    // Phase 1: Each thread accumulates sum-of-squares for its strided elements.
    //
    // With 256 threads and hidden=2048, each thread processes 8 elements:
    //   thread 0: indices 0, 256, 512, ..., 1792
    //   thread 1: indices 1, 257, 513, ..., 1793
    //   etc.
    //
    // This strided access pattern is standard for cooperative reductions.
    // -----------------------------------------------------------------------
    float sum_sq = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float val = float(input[i]);
        sum_sq += val * val;
    }

    // -----------------------------------------------------------------------
    // Phase 2: SIMD-level reduction.
    //
    // `simd_sum` is a hardware intrinsic that sums a value across all 32 lanes
    // of a SIMD group in a single cycle.  After this call, every lane in the
    // SIMD group holds the same partial sum for that group's 32 threads.
    // -----------------------------------------------------------------------
    sum_sq = simd_sum(sum_sq);

    // -----------------------------------------------------------------------
    // Phase 3: Cross-SIMD-group reduction via threadgroup shared memory.
    //
    // 256 threads / 32 lanes = 8 SIMD groups.  Each group writes its partial
    // sum to shared[], then the first SIMD group reduces across all 8 values.
    //
    // Learning note: this two-level reduction (simd_sum then shared memory) is
    // the standard pattern for threadgroup reductions on Apple GPUs.  It avoids
    // the slower approach of having every thread do atomic adds or a tree
    // reduction in shared memory.
    // -----------------------------------------------------------------------
    threadgroup float shared[32]; // Space for up to 32 SIMD groups (only 8 used).
    uint simd_group_id = tid / 32;
    uint simd_lane_id = tid % 32;

    // Lane 0 of each SIMD group writes the group's partial sum.
    if (simd_lane_id == 0) {
        shared[simd_group_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First SIMD group loads all partial sums and reduces them.
    if (simd_group_id == 0) {
        uint num_simd_groups = (tg_size + 31) / 32;
        float val = (simd_lane_id < num_simd_groups) ? shared[simd_lane_id] : 0.0f;
        val = simd_sum(val);
        // Lane 0 writes the final total back to shared[0].
        if (simd_lane_id == 0) {
            shared[0] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads read the global sum of squares and compute the scale factor.
    //   scale = 1 / sqrt(mean(x²) + eps)
    // `rsqrt` computes 1/sqrt in a single GPU instruction.
    float mean_sq = shared[0] / float(hidden);
    float scale = rsqrt(mean_sq + params.eps);

    // -----------------------------------------------------------------------
    // Phase 4: Normalise and multiply by learned weight.
    //
    // Each thread writes its strided elements:
    //   out[i] = input[i] * scale * weight[i]
    //
    // The weight vector is a per-element learned parameter that allows the
    // model to re-scale dimensions it considers important.  Without it,
    // normalisation would destroy all magnitude information.
    // -----------------------------------------------------------------------
    for (uint i = tid; i < hidden; i += tg_size) {
        float val = float(input[i]);
        output[i] = bfloat(val * scale * float(weight[i]));
    }
}
