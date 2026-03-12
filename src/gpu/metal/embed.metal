// ===========================================================================
// Embedding lookup kernel.
//
// LEARNING OVERVIEW
//
// What this kernel does:
//   Converts a discrete token ID (an integer) into a continuous vector
//   representation by looking up a row in the embedding table.  This is
//   the very first step of the transformer forward pass — before any
//   attention or FFN computation can happen, the token must be converted
//   from an index into a vector that the model can process.
//
// The embedding table:
//   Shape: [vocab_size, hidden_dim] = [128256, 2048] for Llama 3.2 1B.
//   Row i contains the learned vector representation for token i.
//   This table is ~500 MB in bfloat16 and is shared with the output
//   projection (lm_head) when tie_word_embeddings=true.
//
// Dispatch model:
//   One thread per element of the output vector (hidden_dim = 2048).
//   Each thread copies one element from the table row — trivially parallel.
//
// Learning note: why a GPU kernel for a simple table lookup?
//   The embedding table lives in GPU memory (uploaded during weight loading).
//   A CPU lookup would require copying the table back to host memory or
//   doing a GPU→CPU→GPU round-trip.  The kernel keeps everything on-device.
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// Host → GPU parameter block.  Must match Rust `EmbedParams`.
struct EmbedParams {
    uint token_id;   // Which row to look up (0..vocab_size-1).
    uint hidden_dim; // Number of elements per row (2048).
};

kernel void embed_lookup(
    constant EmbedParams& params [[buffer(0)]],
    // buffer(1): embedding table [vocab_size, hidden_dim] in bfloat16.
    device const bfloat* table   [[buffer(1)]],
    // buffer(2): output vector [hidden_dim] in bfloat16.
    device bfloat* output        [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= params.hidden_dim) return;
    // Simple table lookup: output[gid] = table[token_id * hidden_dim + gid]
    output[gid] = table[params.token_id * params.hidden_dim + gid];
}
