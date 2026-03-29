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

use crate::gpu::{GpuBackend, TensorDtype};

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
    vocab_size: usize,
) -> anyhow::Result<u32> {
    // Temperature = 0 is the convention for "be greedy / deterministic".
    // Mathematically, lim(T→0) of softmax(logits/T) is a one-hot on the argmax.
    if temperature == 0.0 {
        return sample_greedy(backend, logits, vocab_size);
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
    // Truncate to tokenizer vocab size — the model embedding may be padded
    // beyond the tokenizer's vocabulary (e.g. Qwen 3.5: 248320 embedding vs
    // 248070 tokenizer tokens).  Sampling padding positions produces token IDs
    // that decode to empty strings.
    let effective = vocab_size.min(bf16_values.len());
    let mut logits_f32: Vec<f32> = bf16_values[..effective].iter().map(|v| v.to_f32()).collect();

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

/// Sample N tokens from a batched logits tensor [batch_size, vocab_size].
///
/// This is the batched-decode counterpart to `sample()`.  Instead of N
/// separate GPU→CPU copies (one per sequence), we do ONE copy of the full
/// [N, vocab_size] tensor, then iterate rows on the CPU.  Each sequence
/// gets its own temperature and top_p — different concurrent requests can
/// have different sampling parameters.
///
/// Why batch the copy but not the sampling math?
///   The GPU→CPU transfer is the expensive part (~250 KB per sequence at
///   vocab_size=128256).  The CPU sampling (softmax + sort + random walk)
///   takes microseconds per sequence — parallelizing it on the GPU would
///   add kernel complexity without meaningful speedup.
pub(crate) fn sample_batch<B: GpuBackend>(
    backend: &B,
    logits_batch: &B::Tensor,
    batch_size: usize,
    vocab_size: usize,
    temperatures: &[f32],
    top_ps: &[f32],
    rng: &mut impl Rng,
    tokenizer_vocab_size: usize,
) -> anyhow::Result<Vec<u32>> {
    assert_eq!(temperatures.len(), batch_size);
    assert_eq!(top_ps.len(), batch_size);

    // One GPU→CPU copy for all N sequences' logits.
    let total_bytes = batch_size * vocab_size * 2; // bf16 = 2 bytes
    let mut buf = vec![0u8; total_bytes];
    backend.copy_to_host(logits_batch, &mut buf);

    let row_bytes = vocab_size * 2;
    // Truncate each row to tokenizer vocab size (see sample() for rationale).
    let effective_row_bytes = tokenizer_vocab_size.min(vocab_size) * 2;
    let mut results = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let row_data = &buf[i * row_bytes..i * row_bytes + effective_row_bytes];
        let token = if temperatures[i] == 0.0 {
            // Greedy: argmax on this row.
            argmax_bf16(row_data)
        } else {
            // Temperature + top-p sampling on this row.
            sample_row(row_data, temperatures[i], top_ps[i], rng)
        };
        results.push(token);
    }

    Ok(results)
}

/// Sample one token from a single row of bf16 logits (CPU-side).
///
/// Extracted from `sample()` so it can be reused by `sample_batch()` without
/// needing a GPU tensor — it operates directly on a byte slice already on the CPU.
fn sample_row(logits_bytes: &[u8], temperature: f32, top_p: f32, rng: &mut impl Rng) -> u32 {
    let bf16_values: &[bf16] = bytemuck::cast_slice(logits_bytes);
    let mut logits_f32: Vec<f32> = bf16_values.iter().map(|v| v.to_f32()).collect();

    // Temperature scaling.
    let inv_temp = 1.0 / temperature;
    for logit in logits_f32.iter_mut() {
        *logit *= inv_temp;
    }

    // Softmax.
    let max_logit = logits_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for logit in logits_f32.iter_mut() {
        *logit = (*logit - max_logit).exp();
        sum += *logit;
    }
    let inv_sum = 1.0 / sum;
    for prob in logits_f32.iter_mut() {
        *prob *= inv_sum;
    }

    // Top-p filtering.
    if top_p < 1.0 {
        let mut indices: Vec<u32> = (0..logits_f32.len() as u32).collect();
        indices.sort_unstable_by(|&a, &b| {
            logits_f32[b as usize]
                .partial_cmp(&logits_f32[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cumulative = 0.0f32;
        let mut cutoff_idx = indices.len();
        for (i, &token_idx) in indices.iter().enumerate() {
            cumulative += logits_f32[token_idx as usize];
            if cumulative >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        for &token_idx in &indices[cutoff_idx..] {
            logits_f32[token_idx as usize] = 0.0;
        }
        let new_sum: f32 = logits_f32.iter().sum();
        let inv_new_sum = 1.0 / new_sum;
        for prob in logits_f32.iter_mut() {
            *prob *= inv_new_sum;
        }
    }

    // Weighted random sample.
    let r: f32 = rng.random();
    let mut cumulative = 0.0f32;
    for (i, &prob) in logits_f32.iter().enumerate() {
        cumulative += prob;
        if cumulative > r {
            return i as u32;
        }
    }
    (logits_f32.len() - 1) as u32
}

/// Greedy sampling: copy logits to host, return the argmax token ID.
///
/// This is the only place where data moves from GPU → CPU during generation.
/// On Apple Silicon with unified memory, "copy" is just a pointer read —
/// the data is already in the same physical memory.
pub(crate) fn sample_greedy<B: GpuBackend>(backend: &B, logits: &B::Tensor, vocab_size: usize) -> anyhow::Result<u32> {
    // Determine how many bytes the logits tensor occupies.
    let byte_count = backend.tensor_byte_count(logits);
    let mut buf = vec![0u8; byte_count];
    // Copy raw bytes from the GPU tensor into our host buffer.
    backend.copy_to_host(logits, &mut buf);
    // Truncate to tokenizer vocab size (see sample() for rationale).
    let effective_bytes = (vocab_size * 2).min(buf.len());
    Ok(argmax_bf16(&buf[..effective_bytes]))
}

/// GPU-resident greedy sampling: argmax computed entirely on device.
///
/// Only 4 bytes transferred (one u32 token ID), not the full logit vector.
/// For vocab_size=152K this reduces DtoH from ~300 KB to 4 bytes per sequence.
///
/// Technique from rvLLM (Andy Norris / m0at): the single most impactful
/// optimization for greedy decode.  See: https://github.com/m0at/rvllm
#[allow(dead_code)] // engine uses batched version; kept as building block
pub(crate) fn sample_greedy_gpu<B: GpuBackend>(
    backend: &B,
    logits: &B::Tensor,
    vocab_size: usize,
) -> anyhow::Result<u32> {
    // Allocate a tiny output buffer for one u32 token ID.
    let output = backend.alloc_tensor(&[1], TensorDtype::F32); // 4 bytes = 1 u32
    backend.argmax_gpu(logits, &output, vocab_size as u32, 1);

    // Copy just the single u32 result back to host.
    let mut buf = [0u8; 4];
    backend.copy_to_host(&output, &mut buf);
    Ok(u32::from_ne_bytes(buf))
}

/// GPU-resident batched greedy sampling: argmax for N sequences at once.
///
/// Only N × 4 bytes transferred instead of N × vocab_size × 2 bytes.
/// At batch=128, vocab=152K: 512 bytes vs ~37 MB — a ~72,000x reduction.
pub(crate) fn sample_batch_greedy_gpu<B: GpuBackend>(
    backend: &B,
    logits_batch: &B::Tensor,
    batch_size: usize,
    vocab_size: usize,
) -> anyhow::Result<Vec<u32>> {
    let output = backend.alloc_tensor(&[batch_size], TensorDtype::F32);
    backend.argmax_gpu(logits_batch, &output, vocab_size as u32, batch_size as u32);

    // Copy N u32 token IDs back to host.
    let mut buf = vec![0u8; batch_size * 4];
    backend.copy_to_host(&output, &mut buf);
    let ids: &[u32] = bytemuck::cast_slice(&buf);
    Ok(ids.to_vec())
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
    use rand::SeedableRng;

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

    // -- sample_row tests --

    #[test]
    fn test_sample_row_very_low_temperature_acts_like_argmax() {
        // Very low temperature should pick the highest-logit token.
        let data = bf16_bytes(&[1.0, 5.0, 2.0, 0.5]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        // Run multiple times — should always pick index 1.
        for _ in 0..20 {
            let token = sample_row(&data, 0.01, 1.0, &mut rng);
            assert_eq!(token, 1, "Very low temperature should pick argmax");
        }
    }

    #[test]
    fn test_sample_row_deterministic_with_same_seed() {
        let data = bf16_bytes(&[1.0, 2.0, 3.0, 2.0, 1.0]);
        // Same seed should produce the same sequence of tokens.
        let mut rng1 = rand::rngs::SmallRng::seed_from_u64(123);
        let mut rng2 = rand::rngs::SmallRng::seed_from_u64(123);
        for _ in 0..50 {
            let t1 = sample_row(&data, 1.0, 1.0, &mut rng1);
            let t2 = sample_row(&data, 1.0, 1.0, &mut rng2);
            assert_eq!(t1, t2, "Same seed should produce same token");
        }
    }

    #[test]
    fn test_sample_row_top_p_filters_tail() {
        // Create a distribution where one token dominates.
        // logits: [10.0, 0.0, 0.0, 0.0] — after softmax, token 0 has ~99.99%
        let data = bf16_bytes(&[10.0, 0.0, 0.0, 0.0]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(99);
        // With top_p=0.5, only the dominant token should be sampled.
        for _ in 0..50 {
            let token = sample_row(&data, 1.0, 0.5, &mut rng);
            assert_eq!(token, 0, "Top-p should filter to dominant token");
        }
    }

    #[test]
    fn test_sample_row_single_token_vocab() {
        let data = bf16_bytes(&[42.0]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let token = sample_row(&data, 1.0, 1.0, &mut rng);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_row_equal_logits_samples_all_tokens() {
        // With equal logits and temperature=1.0, all tokens should be sampled
        // over enough trials.
        let data = bf16_bytes(&[0.0, 0.0, 0.0, 0.0]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let mut counts = [0u32; 4];
        for _ in 0..1000 {
            let token = sample_row(&data, 1.0, 1.0, &mut rng);
            counts[token as usize] += 1;
        }
        // Each token should be sampled at least once (expected ~250 each).
        for (i, &count) in counts.iter().enumerate() {
            assert!(count > 50, "Token {i} sampled only {count} times out of 1000");
        }
    }

    #[test]
    fn test_sample_row_high_temperature_more_uniform() {
        // High temperature should spread probability more evenly.
        // Use logits with a clear winner and see if high temp reduces its dominance.
        let data = bf16_bytes(&[5.0, 0.0, 0.0, 0.0]);

        // Low temperature: token 0 should dominate.
        let mut rng_low = rand::rngs::SmallRng::seed_from_u64(42);
        let mut count_low = 0u32;
        for _ in 0..500 {
            if sample_row(&data, 0.5, 1.0, &mut rng_low) == 0 {
                count_low += 1;
            }
        }

        // High temperature: token 0 should still be most common but less dominant.
        let mut rng_high = rand::rngs::SmallRng::seed_from_u64(42);
        let mut count_high = 0u32;
        for _ in 0..500 {
            if sample_row(&data, 3.0, 1.0, &mut rng_high) == 0 {
                count_high += 1;
            }
        }

        assert!(
            count_low > count_high,
            "Low temp ({count_low}) should pick token 0 more than high temp ({count_high})"
        );
    }

    #[test]
    fn test_sample_row_top_p_one_no_filtering() {
        // top_p=1.0 should not filter any tokens (all are candidates).
        let data = bf16_bytes(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(55);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(sample_row(&data, 1.0, 1.0, &mut rng));
        }
        assert_eq!(seen.len(), 5, "top_p=1.0 should allow all 5 tokens");
    }

    #[test]
    fn test_argmax_bf16_two_elements_equal() {
        // Two equal values — max_by returns the last equal element.
        let data = bf16_bytes(&[5.0, 5.0]);
        let result = argmax_bf16(&data);
        // Either index is valid since they're equal.
        assert!(result == 0 || result == 1);
    }
}
