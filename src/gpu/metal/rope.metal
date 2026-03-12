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
//   of dimension D, there are D/2 pairs.  Each pair is rotated by an angle
//   that depends on both the token's position AND the pair's index:
//
//     angle = position * (1 / theta^(2*pair_index / D))
//
//   where theta is a base frequency (500000 for Llama 3, 1000000 for Qwen).
//   Lower-indexed pairs rotate slowly (coarse position encoding), while
//   higher-indexed pairs rotate faster (fine-grained position info).
//
//   The rotation formula (standard 2D rotation matrix):
//     v[i]'       = v[i]       * cos(angle) - v[i+D/2] * sin(angle)
//     v[i+D/2]'   = v[i]       * sin(angle) + v[i+D/2] * cos(angle)
//
// Halved vs. interleaved pairing:
//   There are two common conventions for which elements form rotation pairs:
//     - Interleaved: (v[0], v[1]), (v[2], v[3]), ..., (v[D-2], v[D-1])
//     - Halved:      (v[0], v[D/2]), (v[1], v[D/2+1]), ..., (v[D/2-1], v[D-1])
//
//   HuggingFace transformers uses the HALVED convention via `rotate_half()`.
//   Since we load HF-format checkpoints (both Llama and Qwen), we use
//   halved pairing to match how the model was trained.  Using the wrong
//   convention would pair different Q/K elements at different frequencies,
//   corrupting the attention scores.
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
//   (num_heads + num_kv_heads) * (head_dim / 2).
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
    float rope_theta;   // Base frequency (500000 for Llama 3, 1000000 for Qwen).
    uint num_heads;     // Number of query heads.
    uint num_kv_heads;  // Number of KV heads.
    uint head_dim;      // Dimension per head.
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

    // Decompose flat pair index into (head_index, pair_within_head).
    uint head_idx = pair_within / half_dim;
    uint pair_in_head = pair_within % half_dim;

    // Compute the rotation angle for this (position, pair) combination.
    //   inv_freq = 1 / (theta ^ (2 * pair_in_head / head_dim))
    //   angle = position * inv_freq
    //
    // Learning note: the exponent 2*i/D creates a geometric progression of
    // frequencies.  Pair 0 has the lowest frequency (slowest rotation), pair
    // D/2-1 has the highest (fastest rotation).  This is analogous to the
    // different frequencies in sinusoidal positional encodings from the
    // original Transformer paper, but applied as rotations rather than additions.
    float freq_exp = 2.0f * float(pair_in_head) / float(params.head_dim);
    float inv_freq = 1.0f / pow(params.rope_theta, freq_exp);
    float angle = float(params.pos) * inv_freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Apply the 2D rotation using HALVED pairing.
    //
    // Halved convention: element i pairs with element i + D/2 within each head.
    //   [a']   [cos  -sin] [a]       where a = data[head_start + i]
    //   [b'] = [sin   cos] [b]             b = data[head_start + i + D/2]
    //
    // This matches HuggingFace's `rotate_half()`:
    //   out[i]       = x[i] * cos - x[i + D/2] * sin
    //   out[i + D/2] = x[i] * sin + x[i + D/2] * cos
    uint head_offset = head_idx * params.head_dim;
    uint idx_a = head_offset + pair_in_head;           // element i within head
    uint idx_b = head_offset + pair_in_head + half_dim; // element i + D/2 within head
    float a = float(data[idx_a]);
    float b = float(data[idx_b]);
    data[idx_a] = bfloat(a * cos_val - b * sin_val);
    data[idx_b] = bfloat(a * sin_val + b * cos_val);
}

// ===========================================================================
// Batched RoPE kernel.
//
// LEARNING OVERVIEW
//
// What this kernel does:
//   Applies rotary embeddings to [batch_size, num_heads, head_dim] Q and K
//   tensors, with per-token positions from a positions buffer.  Each token
//   in the batch gets a DIFFERENT rotation angle corresponding to its
//   sequence position.
//
// Key difference from single-token RoPE:
//   Single-token RoPE takes `pos` as a scalar in the params struct — all
//   threads rotate by the same position.  The batched version takes a
//   `positions[batch_size]` buffer where positions[i] = start_pos + i.
//   This is essential for prefill: token 0 of the prompt is at position 0,
//   token 1 at position 1, etc.  Each must get its own rotation.
//
// Dispatch model:
//   One thread per (batch, head, pair) combination.
//   Grid: batch_size * (num_heads + num_kv_heads) * (head_dim / 2).
//   Same rotation logic as the single-token version, but each batch
//   element reads its own position from positions[batch_idx].
// ===========================================================================

struct RopeBatchParams {
    uint batch_size;
    float rope_theta;
    uint num_heads;
    uint num_kv_heads;
    uint head_dim;
};

kernel void rotary_embedding_batch(
    constant RopeBatchParams& params [[buffer(0)]],
    device bfloat* q                 [[buffer(1)]],  // [batch_size, num_heads * head_dim]
    device bfloat* k                 [[buffer(2)]],  // [batch_size, num_kv_heads * head_dim]
    device const uint* positions     [[buffer(3)]],  // [batch_size]
    uint gid                         [[thread_position_in_grid]]
) {
    const uint half_dim = params.head_dim / 2;
    const uint q_pairs = params.num_heads * half_dim;
    const uint k_pairs = params.num_kv_heads * half_dim;
    const uint pairs_per_token = q_pairs + k_pairs;
    const uint total = params.batch_size * pairs_per_token;

    if (gid >= total) return;

    // Decompose: which batch element, and which (Q or K, head, pair).
    uint batch = gid / pairs_per_token;
    uint within = gid % pairs_per_token;

    uint pos = positions[batch];

    // Select Q or K tensor and compute head/pair indices.
    device bfloat* data;
    uint pair_within;
    uint q_dim = params.num_heads * params.head_dim;
    uint k_dim = params.num_kv_heads * params.head_dim;

    if (within < q_pairs) {
        data = q + batch * q_dim;
        pair_within = within;
    } else {
        data = k + batch * k_dim;
        pair_within = within - q_pairs;
    }

    uint head_idx = pair_within / half_dim;
    uint pair_in_head = pair_within % half_dim;

    float freq_exp = 2.0f * float(pair_in_head) / float(params.head_dim);
    float inv_freq = 1.0f / pow(params.rope_theta, freq_exp);
    float angle = float(pos) * inv_freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    uint head_offset = head_idx * params.head_dim;
    uint idx_a = head_offset + pair_in_head;
    uint idx_b = head_offset + pair_in_head + half_dim;
    float a = float(data[idx_a]);
    float b = float(data[idx_b]);
    data[idx_a] = bfloat(a * cos_val - b * sin_val);
    data[idx_b] = bfloat(a * sin_val + b * cos_val);
}
