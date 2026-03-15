// ===========================================================================
// Sampling — selecting the next token from model logits.
//
// LEARNING OVERVIEW
//
// What this file does:
//   After the model produces logits (a score for every token in the vocabulary),
//   the sampler decides which token to actually generate.  Two strategies:
//
//   1. Greedy (argmax):  Always pick the highest-scoring token.  Deterministic —
//      the same prompt always produces the same output.  Good for factual tasks
//      but tends to be repetitive.
//
//   2. Temperature + Top-p (nucleus):  Scale the logits, convert to a probability
//      distribution, trim the tail, and randomly sample.  Produces varied,
//      creative output.  This is what ChatGPT/Claude use by default.
//
// The sampling pipeline (for temperature + top-p):
//
//   logits  ──→  ÷ temperature  ──→  softmax  ──→  top-p filter  ──→  random sample
//   [128256]     (sharpen/flatten)   (→ probs)     (trim tail)       (→ token ID)
//
// Temperature intuition:
//   Temperature controls how "confident" the model acts.  Mathematically, it
//   scales the logits before softmax:  p_i = exp(logit_i / T) / Σ exp(logit_j / T)
//
//   T = 1.0 → use the model's natural distribution (default)
//   T < 1.0 → sharpen: high-probability tokens get even higher, low ones vanish
//   T → 0   → greedy: collapses to argmax (pick the single best token)
//   T > 1.0 → flatten: spread probability more evenly (more random/creative)
//
//   Example with logits [2.0, 1.0, 0.5]:
//     T=1.0 → softmax → [0.51, 0.19, 0.11, ...]  (natural)
//     T=0.5 → logits/T=[4.0, 2.0, 1.0] → softmax → [0.84, 0.05, 0.02, ...]  (sharp)
//     T=2.0 → logits/T=[1.0, 0.5, 0.25] → softmax → [0.38, 0.23, 0.18, ...]  (flat)
//
// Top-p (nucleus) intuition:
//   Even with temperature, the model might sample very unlikely tokens (the
//   "tail" of the distribution).  Top-p trims this tail:
//
//   1. Sort tokens by probability (descending)
//   2. Walk down the sorted list, accumulating probability
//   3. Once cumulative probability ≥ p, discard everything below
//   4. Renormalize the remaining tokens and sample from them
//
//   p = 0.9 means: "only consider tokens in the top 90% of probability mass."
//   This dynamically adjusts how many tokens are candidates — when the model
//   is confident, only a few tokens pass; when uncertain, many do.
//
//   Top-p vs. top-k:
//     Top-k always keeps exactly k tokens regardless of their probability.
//     Top-p adapts: it might keep 5 tokens or 500, depending on the distribution.
//     Top-p is generally preferred (used by most production systems).
//
// Why sample on the CPU?
//   The logits tensor has 128256 bfloat16 values (~250 KB).  Even with sorting,
//   this is trivial on a modern CPU — microseconds.  Not worth a GPU kernel
//   for single-sequence inference.
// ===========================================================================

use half::bf16;
use rand::Rng;

use crate::gpu::GpuBackend;

/// Sample the next token using temperature scaling and top-p (nucleus) filtering.
///
/// Pipeline: copy logits → convert bf16→f32 → scale by temperature → softmax
/// → top-p filter → weighted random sample.
///
/// Special case: temperature == 0.0 falls back to greedy (argmax).
pub(crate) fn sample<B: GpuBackend>(
    backend: &B,
    logits: &B::Tensor,
    temperature: f32,
    top_p: f32,
    rng: &mut impl Rng,
) -> anyhow::Result<u32> {
    // Temperature = 0 is the convention for "be greedy / deterministic".
    // Mathematically, lim(T→0) of softmax(logits/T) is a one-hot on the argmax.
    if temperature == 0.0 {
        return sample_greedy(backend, logits);
    }

    // --- Step 1: Copy logits from GPU to host ---
    // Same as greedy — on unified memory (Apple Silicon) this is just a pointer read.
    let byte_count = backend.tensor_byte_count(logits);
    let mut buf = vec![0u8; byte_count];
    backend.copy_to_host(logits, &mut buf);

    // --- Step 2: Convert bf16 → f32 ---
    // We need f32 for the math that follows (exp, division, accumulation).
    // bf16 has only ~3 decimal digits of precision — fine for storing logits,
    // but not enough for stable softmax or cumulative sums.
    let bf16_values: &[bf16] = bytemuck::cast_slice(&buf);
    let mut logits_f32: Vec<f32> = bf16_values.iter().map(|v| v.to_f32()).collect();

    // --- Step 3: Temperature scaling ---
    // Divide every logit by T.  This is equivalent to raising the softmax
    // distribution to the power 1/T — high T flattens, low T sharpens.
    let inv_temp = 1.0 / temperature;
    for logit in logits_f32.iter_mut() {
        *logit *= inv_temp;
    }

    // --- Step 4: Softmax → probabilities ---
    // softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)
    //
    // Why subtract max?  Numerical stability.  exp(1000) overflows f32, but
    // exp(1000 - 1000) = exp(0) = 1.  Subtracting the max shifts all values
    // into a safe range without changing the resulting probabilities (the
    // constant cancels in the numerator/denominator).
    let max_logit = logits_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for logit in logits_f32.iter_mut() {
        *logit = (*logit - max_logit).exp();
        sum += *logit;
    }
    // Now logits_f32[i] = exp(logit_i / T - max).  Divide by sum to get probs.
    let inv_sum = 1.0 / sum;
    for prob in logits_f32.iter_mut() {
        *prob *= inv_sum;
    }
    // logits_f32 is now a valid probability distribution (sums to 1.0).

    // --- Step 5: Top-p (nucleus) filtering ---
    // Sort tokens by probability, keep only the top-p fraction of probability
    // mass, zero out the rest.  This prevents sampling from the long tail of
    // very unlikely tokens that can produce garbage.
    //
    // Implementation: we don't actually need to sort the full 128K array.  We
    // build an index array, sort THAT by probability (descending), then walk
    // it to find the cutoff.  This avoids shuffling the probability array itself
    // (we need it indexed by token ID for the final sample).
    if top_p < 1.0 {
        // Build sorted indices.  For 128K elements, this sort takes ~1ms — fine.
        let mut indices: Vec<u32> = (0..logits_f32.len() as u32).collect();
        indices.sort_unstable_by(|&a, &b| {
            logits_f32[b as usize]
                .partial_cmp(&logits_f32[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Walk down the sorted list, accumulating probability.  Once we've
        // captured ≥ top_p of the total mass, zero out everything below.
        let mut cumulative = 0.0f32;
        let mut cutoff_idx = indices.len();
        for (i, &token_idx) in indices.iter().enumerate() {
            cumulative += logits_f32[token_idx as usize];
            if cumulative >= top_p {
                // Keep this token (it pushed us over the threshold), but
                // everything after it gets zeroed.
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out tokens below the cutoff.
        for &token_idx in &indices[cutoff_idx..] {
            logits_f32[token_idx as usize] = 0.0;
        }

        // Renormalize so the remaining probabilities sum to 1.0.
        // (We could reuse `cumulative` here, but it may have floating-point
        // drift, so recompute for accuracy.)
        let new_sum: f32 = logits_f32.iter().sum();
        let inv_new_sum = 1.0 / new_sum;
        for prob in logits_f32.iter_mut() {
            *prob *= inv_new_sum;
        }
    }

    // --- Step 6: Weighted random sampling ---
    // Generate a random number in [0, 1) and walk through the probability
    // distribution until the cumulative sum exceeds it.  This is O(n) but
    // for a single sample from 128K elements it's ~microseconds.
    //
    // This is the simplest correct algorithm.  Fancier approaches (alias method,
    // binary search on CDF) aren't worth the complexity for single-sequence
    // inference.
    let r: f32 = rng.random();
    let mut cumulative = 0.0f32;
    for (i, &prob) in logits_f32.iter().enumerate() {
        cumulative += prob;
        if cumulative > r {
            return Ok(i as u32);
        }
    }

    // Fallback: floating-point rounding could cause us to not exceed r.
    // Return the last token with nonzero probability.
    Ok((logits_f32.len() - 1) as u32)
}

/// Greedy sampling: copy logits to host, return the argmax token ID.
///
/// This is the only place where data moves from GPU → CPU during generation.
/// On Apple Silicon with unified memory, "copy" is just a pointer read —
/// the data is already in the same physical memory.
pub(crate) fn sample_greedy<B: GpuBackend>(backend: &B, logits: &B::Tensor) -> anyhow::Result<u32> {
    // Determine how many bytes the logits tensor occupies.
    let byte_count = backend.tensor_byte_count(logits);
    let mut buf = vec![0u8; byte_count];
    // Copy raw bytes from the GPU tensor into our host buffer.
    backend.copy_to_host(logits, &mut buf);
    Ok(argmax_bf16(&buf))
}

/// Find the index of the maximum value in a bf16 byte slice.
///
/// Uses `bytemuck::cast_slice` to reinterpret the raw bytes as bf16 values,
/// then a simple linear scan.  For 128256 elements this takes ~microseconds
/// on the CPU — not a bottleneck.
fn argmax_bf16(data: &[u8]) -> u32 {
    let values: &[bf16] = bytemuck::cast_slice(data);
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let bf16_values: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
        bytemuck::cast_slice(&bf16_values).to_vec()
    }

    #[test]
    fn test_argmax_bf16_basic() {
        let data = bf16_bytes(&[1.0, 3.0, 2.0, 0.5]);
        assert_eq!(argmax_bf16(&data), 1);
    }

    #[test]
    fn test_argmax_bf16_single() {
        let data = bf16_bytes(&[42.0]);
        assert_eq!(argmax_bf16(&data), 0);
    }

    #[test]
    fn test_argmax_bf16_all_negative() {
        let data = bf16_bytes(&[-5.0, -1.0, -3.0, -2.0]);
        assert_eq!(argmax_bf16(&data), 1); // -1.0 is the largest
    }

    #[test]
    fn test_argmax_bf16_last_is_max() {
        let data = bf16_bytes(&[0.0, 0.0, 0.0, 10.0]);
        assert_eq!(argmax_bf16(&data), 3);
    }

    #[test]
    fn test_argmax_bf16_first_is_max() {
        let data = bf16_bytes(&[10.0, 0.0, 0.0, 0.0]);
        assert_eq!(argmax_bf16(&data), 0);
    }

    #[test]
    fn test_argmax_bf16_mixed() {
        let data = bf16_bytes(&[-100.0, 0.0, 100.0, 50.0, -50.0]);
        assert_eq!(argmax_bf16(&data), 2);
    }
}
