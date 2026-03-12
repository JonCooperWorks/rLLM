// ===========================================================================
// Rotary Positional Embeddings (RoPE) kernel.
//
// LEARNING OVERVIEW
//
// What this kernel does:
//   Applies position-dependent rotations to the Query and Key vectors,
//   encoding absolute position information into the attention computation.
//   Without positional encodings, a transformer cannot distinguish "the
//   cat sat on the mat" from "mat the on sat cat the" — the self-attention
//   mechanism is permutation-invariant.
//
// How RoPE works:
//   Each head's vector is treated as a sequence of 2D pairs.  For a head
//   of dimension D=64, there are D/2=32 pairs: (v[0], v[1]), (v[2], v[3]),
//   ..., (v[62], v[63]).  Each pair is rotated by an angle that depends
//   on both the token's position in the sequence AND the pair's index:
//
//     angle = position * (1 / theta^(2*pair_index / D))
//
//   where theta is a base frequency (500000.0 for Llama 3.2).  Higher-
//   indexed pairs rotate faster, encoding fine-grained position info,
//   while lower-indexed pairs rotate slowly, encoding coarse position.
//
//   The rotation formula (standard 2D rotation matrix):
//     v[2i]'   = v[2i]   * cos(angle) - v[2i+1] * sin(angle)
//     v[2i+1]' = v[2i]   * sin(angle) + v[2i+1] * cos(angle)
//
// Why RoPE instead of learned embeddings?
//   RoPE has a key property: the dot product of two rotated vectors
//   depends only on the RELATIVE distance between their positions, not
//   their absolute positions.  This means the model can generalise to
//   longer sequences than it was trained on.
//
// Dispatch model:
//   One thread per (head, pair).  Threads 0..(num_heads * head_dim/2 - 1)
//   handle Q heads, the remaining threads handle K heads.  Total threads:
//   (num_heads + num_kv_heads) * (head_dim / 2) = (32 + 8) * 32 = 1280.
//
// Precision:
//   sin/cos computation and rotation are done in float32 to avoid
//   catastrophic precision loss (bfloat16 has only ~3 significant digits,
//   which would corrupt the rotation at high positions).
// ===========================================================================

#include <metal_stdlib>
using namespace metal;

// Host → GPU parameter block.  Must match Rust `RopeParams`.
struct RopeParams {
    uint pos;           // Token position in the sequence (0-indexed).
    float rope_theta;   // Base frequency (500000.0 for Llama 3.2).
    uint num_heads;     // Number of query heads (32).
    uint num_kv_heads;  // Number of KV heads (8).
    uint head_dim;      // Dimension per head (64).
};

kernel void rotary_embedding(
    constant RopeParams& params [[buffer(0)]],
    // buffer(1): Q vector [num_heads * head_dim] in bfloat16. Modified in-place.
    device bfloat* q            [[buffer(1)]],
    // buffer(2): K vector [num_kv_heads * head_dim] in bfloat16. Modified in-place.
    device bfloat* k            [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]
) {
    const uint half_dim = params.head_dim / 2;
    const uint q_pairs = params.num_heads * half_dim;
    const uint total_pairs = q_pairs + params.num_kv_heads * half_dim;

    if (gid >= total_pairs) return;

    // Determine which tensor (Q or K) and which (head, pair) this thread handles.
    // Threads 0..q_pairs handle Q; threads q_pairs..total_pairs handle K.
    device bfloat* data;
    uint pair_within;
    if (gid < q_pairs) {
        data = q;
        pair_within = gid;
    } else {
        data = k;
        pair_within = gid - q_pairs;
    }

    // Which pair within the head (0..31 for head_dim=64).
    uint pair_in_head = pair_within % half_dim;

    // Compute the rotation angle for this (position, pair) combination.
    //   inv_freq = 1 / (theta ^ (2 * pair_in_head / head_dim))
    //   angle = position * inv_freq
    //
    // Learning note: the exponent 2*i/D creates a geometric progression of
    // frequencies.  Pair 0 has the lowest frequency (slowest rotation), pair
    // 31 has the highest (fastest rotation).  This is analogous to the
    // different frequencies in sinusoidal positional encodings from the
    // original Transformer paper, but applied as rotations rather than additions.
    float freq_exp = 2.0f * float(pair_in_head) / float(params.head_dim);
    float inv_freq = 1.0f / pow(params.rope_theta, freq_exp);
    float angle = float(params.pos) * inv_freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Apply the 2D rotation to the pair.
    //   [a']   [cos  -sin] [a]
    //   [b'] = [sin   cos] [b]
    uint idx = pair_within * 2;
    float a = float(data[idx]);
    float b = float(data[idx + 1]);
    data[idx]     = bfloat(a * cos_val - b * sin_val);
    data[idx + 1] = bfloat(a * sin_val + b * cos_val);
}
