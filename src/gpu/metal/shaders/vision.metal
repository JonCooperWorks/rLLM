// ===========================================================================
// Vision encoder utility kernels.
//
// spatial_merge:          rearrange 2D grid tokens by merging merge_size×merge_size
//                         spatial neighbours into single concatenated tokens.
// scatter_vision_tokens:  overwrite text embedding rows at image placeholder
//                         positions with vision encoder output.
//
// LEARNING NOTE: These two kernels use very different dispatch strategies
// because of their different parallelism characteristics:
//
//   spatial_merge is a pure data rearrangement (no reductions, no barriers).
//   Every output element maps to exactly one input element via an index
//   calculation, so we dispatch one thread per output element across the
//   entire output tensor.  This is the simplest possible dispatch model —
//   a 1D grid of threads, each doing one read + one write.
//
//   scatter_vision_tokens has a serial dependency: we must scan token_ids
//   in order to count which placeholder we're at (the N-th `<image>` token
//   gets vision row N).  We solve this with a single threadgroup: the outer
//   loop over seq_len is serial (all threads see the same token_id), but
//   the inner copy of each hidden_dim-sized row is parallelised across the
//   threads in the threadgroup via strided access (i += tg_size).  A
//   threadgroup_barrier between rows ensures the copy completes before we
//   advance vision_idx.
//
// Trait contract: gpu/ops/vision.rs
// Metal impl:    gpu/metal/kernels/vision.rs
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Spatial merge: rearrange [grid_h*grid_w, hidden] → [merged_tokens, hidden*ms*ms]
//
// For merge_size=2, every 2×2 block of tokens from a 2D grid is concatenated
// into one output token of 4× the hidden dimension.  The output token count
// is (grid_h/ms) * (grid_w/ms).
//
// Dispatch: one thread per bf16 element in the OUTPUT tensor.
//   Grid size = (grid_h/ms) * (grid_w/ms) * hidden_dim * ms * ms
//   Each thread computes its output position, reverse-maps to the
//   corresponding source position in the input grid, and copies one bf16.
//
// LEARNING NOTE: The index arithmetic decomposes gid into three levels:
//   1. out_token = which merged output token (row-major in the output grid)
//   2. sub_token = which of the ms×ms source tokens within that merge block
//   3. sub_elem  = which element within that source token's hidden_dim
//   From sub_token we recover (dy, dx) in the merge block, then offset from
//   the output token's grid position to find the source row in the input.
// ---------------------------------------------------------------------------

struct SpatialMergeParams {
    uint grid_h;
    uint grid_w;
    uint hidden_dim;
    uint merge_size;
};

kernel void spatial_merge(
    constant SpatialMergeParams& params [[buffer(0)]],
    device const bfloat* input          [[buffer(1)]],
    device bfloat* output               [[buffer(2)]],
    uint gid                            [[thread_position_in_grid]]
) {
    const uint ms = params.merge_size;
    const uint hd = params.hidden_dim;
    const uint out_w = params.grid_w / ms;
    const uint merged_hd = hd * ms * ms;
    const uint total_elements = (params.grid_h / ms) * out_w * merged_hd;

    if (gid >= total_elements) return;

    // Decompose global thread ID into (out_row, out_col, element_within_merged_token).
    uint out_token = gid / merged_hd;
    uint elem      = gid % merged_hd;

    uint out_row = out_token / out_w;
    uint out_col = out_token % out_w;

    // Which sub-token within the merge block does this element belong to?
    uint sub_token = elem / hd;
    uint sub_elem  = elem % hd;

    // Map sub-token to 2D offset within the merge block (row-major).
    uint dy = sub_token / ms;
    uint dx = sub_token % ms;

    // Source position in the input grid.
    uint src_row = out_row * ms + dy;
    uint src_col = out_col * ms + dx;
    uint src_idx = (src_row * params.grid_w + src_col) * hd + sub_elem;

    output[gid] = input[src_idx];
}

// ---------------------------------------------------------------------------
// Scatter vision tokens into text embeddings.
//
// Scans token_ids[0..seq_len], and for each position where
// token_ids[i] == image_token_id, copies the next vision embedding row
// into text_embeds[i].
//
// This is a serial scan (vision tokens must be placed in order), so we use
// a single threadgroup.  The copy of each row is parallelised across threads.
// For typical vision token counts (~256-1024), this is fast enough.
//
// Dispatch: ONE threadgroup of N threads (typically 256 or 1024).
//   - The outer loop (over seq_len) runs on ALL threads in lockstep.
//   - When a placeholder is found, each thread copies a strided slice of the
//     hidden_dim row: thread `tid` handles elements tid, tid+tg_size, etc.
//   - threadgroup_barrier ensures the entire row is written before the next
//     placeholder is processed (so vision_idx is consistent across threads).
//
// LEARNING NOTE: An alternative design would be a two-pass approach: first
// a prefix-sum to find placeholder positions, then a fully parallel scatter.
// But for typical VLM sequences (~4K tokens, ~256 vision tokens), the single-
// threadgroup serial scan is simpler and fast enough — the bottleneck is the
// vision encoder itself, not this lightweight copy kernel.
// ---------------------------------------------------------------------------

struct ScatterVisionParams {
    uint image_token_id;
    uint seq_len;
    uint hidden_dim;
};

kernel void scatter_vision_tokens(
    constant ScatterVisionParams& params [[buffer(0)]],
    device bfloat* text_embeds           [[buffer(1)]],
    device const bfloat* vision_embeds   [[buffer(2)]],
    device const uint* token_ids         [[buffer(3)]],
    uint tid                             [[thread_position_in_threadgroup]],
    uint tg_size                         [[threads_per_threadgroup]]
) {
    const uint hd = params.hidden_dim;
    uint vision_idx = 0;

    for (uint pos = 0; pos < params.seq_len; pos++) {
        if (token_ids[pos] == params.image_token_id) {
            // Copy vision row `vision_idx` into text_embeds row `pos`.
            device bfloat* dst = text_embeds + pos * hd;
            device const bfloat* src = vision_embeds + vision_idx * hd;
            for (uint i = tid; i < hd; i += tg_size) {
                dst[i] = src[i];
            }
            threadgroup_barrier(mem_flags::mem_device);
            vision_idx++;
        }
    }
}
