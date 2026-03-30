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
// The full sampling pipeline (when all features are active):
//
//   logits  ──→  penalties  ──→  logit bias  ──→  ÷ temperature  ──→  top-k
//   [128256]    (freq+pres)     (per-token)      (sharpen/flatten)   (keep k best)
//
//       ──→  softmax  ──→  [snapshot for logprobs]  ──→  min-p  ──→  top-p
//           (→ probs)                                   (floor)    (trim tail)
//
//       ──→  random sample  ──→  logprob extraction  ──→  SampleResult
//           (→ token ID)        (ln(prob) of chosen)     {id, logprob, top_logprobs}
//
// The order matters.  Penalties and bias operate on raw logits (before softmax).
// Top-k masks before softmax (avoids wasting exp() on garbage tokens).
// Min-p and top-p operate on probabilities (after softmax).
// Logprobs are captured from the post-softmax, pre-filter distribution so
// top_logprobs reflects the model's actual beliefs, not the filtered set.
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
// Top-k intuition:
//   Always keeps exactly the k highest-scoring tokens, regardless of their
//   probability.  Applied before softmax using select_nth_unstable (O(n)
//   partial sort).  Less adaptive than top-p but cheap to compute.
//
// Min-p intuition:
//   Filters tokens with probability < min_p × max_probability.  Adaptive
//   like top-p but simpler: when the model is confident (high max prob),
//   few tokens pass; when uncertain (low max prob), more pass.
//
// Frequency/presence penalties:
//   Reduce repetition by penalising tokens that have already appeared.
//   frequency_penalty: logit -= freq_pen × count (scales with repetition)
//   presence_penalty:  logit -= pres_pen × 1{count > 0} (flat penalty)
//   Applied before temperature scaling (on raw logits).
//
// Logit bias:
//   Per-token additive adjustment to raw logits.  Used by clients to steer
//   generation (e.g., ban specific tokens with -100, boost others).
//
// Why sample on the CPU?
//   The logits tensor has 128256 bfloat16 values (~250 KB).  Even with sorting,
//   this is trivial on a modern CPU — microseconds.  Not worth a GPU kernel
//   for single-sequence inference.
// ===========================================================================

use std::collections::HashMap;

use half::bf16;
use rand::Rng;

use crate::gpu::{GpuBackend, TensorDtype};

// ---------------------------------------------------------------------------
// Public types — returned from sampling, threaded through engine → API.
// ---------------------------------------------------------------------------

/// Result of sampling a single token, including optional log-probability info.
///
/// When logprobs are not requested, `logprob` is 0.0 and `top_logprobs` is empty.
/// The engine propagates this through StepOutput → InferenceEvent → API response.
pub(crate) struct SampleResult {
    /// The sampled token ID.
    pub token_id: u32,
    /// Log-probability of the selected token: ln(prob) after the full pipeline.
    /// Only meaningful when `SampleParams.logprobs` is true; 0.0 otherwise.
    pub logprob: f32,
    /// Top-N alternative tokens with their log-probabilities, sorted descending.
    /// Captured from the post-softmax, pre-filter distribution (reflects the
    /// model's actual beliefs, not the filtered candidate set).
    /// Empty when logprobs are not requested.
    pub top_logprobs: Vec<TokenLogprob>,
}

/// A single token's log-probability, used in top_logprobs arrays.
#[derive(Debug)]
pub(crate) struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
}

/// Extended sampling parameters — replaces the old (temperature, top_p) pair.
///
/// All fields have neutral defaults (no penalties, no filtering beyond top-p=1.0).
/// The API layer constructs this from the union of OpenAI/Anthropic request fields.
#[derive(Clone)]
pub(crate) struct SampleParams {
    /// Temperature scaling.  0.0 = greedy (argmax).  Default: 1.0.
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.  Default: 1.0 (disabled).
    pub top_p: f32,
    /// Top-k: keep only the k highest-scoring tokens.  0 = disabled.  Default: 0.
    pub top_k: u32,
    /// Min-p: discard tokens with prob < min_p × max_prob.  0.0 = disabled.  Default: 0.0.
    pub min_p: f32,
    /// Frequency penalty: logit -= freq_pen × count.  Default: 0.0.
    pub frequency_penalty: f32,
    /// Presence penalty: logit -= pres_pen × 1{count > 0}.  Default: 0.0.
    pub presence_penalty: f32,
    /// Whether to compute and return log-probabilities.  Default: false.
    pub logprobs: bool,
    /// Number of top alternative tokens to include in logprob output.
    /// Only used when `logprobs` is true.  Max 20 (OpenAI spec).  Default: 0.
    pub top_logprobs: u8,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logprobs: false,
            top_logprobs: 0,
        }
    }
}

impl SampleParams {
    /// Whether this configuration can use the fast GPU-resident greedy path.
    ///
    /// GPU argmax is only valid when: temperature=0, no logprobs needed,
    /// no penalties or bias would alter the logits, and no grammar mask.
    /// Any of these features require the full CPU sampling pipeline.
    pub fn can_use_gpu_greedy(&self) -> bool {
        self.temperature == 0.0
            && !self.logprobs
            && self.frequency_penalty == 0.0
            && self.presence_penalty == 0.0
    }
}

/// Convenience constant: empty maps, reusable to avoid allocation.
static EMPTY_COUNTS: std::sync::LazyLock<HashMap<u32, u32>> =
    std::sync::LazyLock::new(HashMap::new);
static EMPTY_BIAS: std::sync::LazyLock<HashMap<u32, f32>> =
    std::sync::LazyLock::new(HashMap::new);

// ---------------------------------------------------------------------------
// Public sampling functions.
// ---------------------------------------------------------------------------

/// Sample the next token using the full pipeline.
///
/// Pipeline: copy logits → penalties → logit bias → temperature → top-k
/// → softmax → [logprob snapshot] → min-p → top-p → weighted random sample
/// → logprob extraction → SampleResult.
///
/// Special case: temperature == 0.0 with no logprobs/penalties/bias falls back
/// to greedy (argmax) via the GPU-resident path for maximum efficiency.
pub(crate) fn sample<B: GpuBackend>(
    backend: &B,
    logits: &B::Tensor,
    params: &SampleParams,
    rng: &mut impl Rng,
    vocab_size: usize,
    allowed_tokens: Option<&[u32]>,
    token_counts: &HashMap<u32, u32>,
    logit_bias: &HashMap<u32, f32>,
) -> anyhow::Result<SampleResult> {
    // Fast path: GPU-resident argmax when no CPU-side work is needed.
    if params.can_use_gpu_greedy() && allowed_tokens.is_none() && logit_bias.is_empty() {
        let token_id = sample_greedy(backend, logits, vocab_size)?;
        return Ok(SampleResult {
            token_id,
            logprob: 0.0,
            top_logprobs: Vec::new(),
        });
    }

    // --- Step 1: Copy logits from GPU to host ---
    let byte_count = backend.tensor_byte_count(logits);
    let mut buf = vec![0u8; byte_count];
    backend.copy_to_host(logits, &mut buf);

    // --- Step 2: Convert bf16 → f32 ---
    let bf16_values: &[bf16] = bytemuck::cast_slice(&buf);
    let effective = vocab_size.min(bf16_values.len());
    let mut logits_f32: Vec<f32> = bf16_values[..effective].iter().map(|v| v.to_f32()).collect();

    // --- Step 2b: Grammar constraint masking ---
    if let Some(allowed) = allowed_tokens {
        apply_token_mask(&mut logits_f32, allowed);
    }

    let result = sample_from_logits(
        &mut logits_f32,
        params,
        rng,
        token_counts,
        logit_bias,
    );
    Ok(result)
}

/// Sample N tokens from a batched logits tensor [batch_size, vocab_size].
///
/// One GPU→CPU copy for all N sequences' logits, then iterate rows on CPU.
/// Each sequence gets its own SampleParams, token counts, and logit bias.
pub(crate) fn sample_batch<B: GpuBackend>(
    backend: &B,
    logits_batch: &B::Tensor,
    batch_size: usize,
    vocab_size: usize,
    params_per_seq: &[&SampleParams],
    rng: &mut impl Rng,
    tokenizer_vocab_size: usize,
    allowed_tokens_per_seq: &[Option<Vec<u32>>],
    token_counts_per_seq: &[&HashMap<u32, u32>],
    logit_bias_per_seq: &[&HashMap<u32, f32>],
) -> anyhow::Result<Vec<SampleResult>> {
    assert_eq!(params_per_seq.len(), batch_size);

    // One GPU→CPU copy for all N sequences' logits.
    let total_bytes = batch_size * vocab_size * 2; // bf16 = 2 bytes
    let mut buf = vec![0u8; total_bytes];
    backend.copy_to_host(logits_batch, &mut buf);

    let row_bytes = vocab_size * 2;
    let effective_vocab = tokenizer_vocab_size.min(vocab_size);
    let effective_row_bytes = effective_vocab * 2;
    let mut results = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let row_data = &buf[i * row_bytes..i * row_bytes + effective_row_bytes];
        let allowed = if i < allowed_tokens_per_seq.len() {
            allowed_tokens_per_seq[i].as_deref()
        } else {
            None
        };
        let params = params_per_seq[i];
        let counts = if i < token_counts_per_seq.len() {
            token_counts_per_seq[i]
        } else {
            &EMPTY_COUNTS
        };
        let bias = if i < logit_bias_per_seq.len() {
            logit_bias_per_seq[i]
        } else {
            &EMPTY_BIAS
        };

        // Fast greedy path for this row (no penalties, no logprobs, no grammar).
        if params.can_use_gpu_greedy() && allowed.is_none() && bias.is_empty() {
            let token_id = argmax_bf16(row_data);
            results.push(SampleResult {
                token_id,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            });
            continue;
        }

        // Full pipeline for this row.
        let bf16_values: &[bf16] = bytemuck::cast_slice(row_data);
        let mut logits_f32: Vec<f32> = bf16_values.iter().map(|v| v.to_f32()).collect();

        if let Some(allowed) = allowed {
            apply_token_mask(&mut logits_f32, allowed);
        }

        let result = sample_from_logits(&mut logits_f32, params, rng, counts, bias);
        results.push(result);
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Core sampling pipeline — operates on f32 logits already on CPU.
// ---------------------------------------------------------------------------

/// The full sampling pipeline on a single row of f32 logits.
///
/// This is the workhorse function shared by `sample()` and `sample_batch()`.
/// Grammar masking must be applied BEFORE calling this function.
///
/// Pipeline order:
///   1. Frequency/presence penalties
///   2. Logit bias
///   3. Temperature scaling
///   4. Top-k masking
///   5. Softmax → probabilities
///   6. Snapshot for logprobs (pre-filter)
///   7. Min-p filtering
///   8. Top-p filtering + renormalization
///   9. Weighted random sample
///  10. Logprob extraction
fn sample_from_logits(
    logits: &mut [f32],
    params: &SampleParams,
    rng: &mut impl Rng,
    token_counts: &HashMap<u32, u32>,
    logit_bias: &HashMap<u32, f32>,
) -> SampleResult {
    let vocab_size = logits.len();

    // --- Step 1: Frequency and presence penalties ---
    // These penalise tokens that have already appeared in the sequence.
    // frequency_penalty scales with count (more repetition → stronger penalty).
    // presence_penalty is a flat penalty for any token that appeared at all.
    if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
        for (&token_id, &count) in token_counts {
            if (token_id as usize) < vocab_size && count > 0 {
                logits[token_id as usize] -=
                    params.frequency_penalty * count as f32
                    + params.presence_penalty;
            }
        }
    }

    // --- Step 2: Logit bias ---
    // Per-token additive adjustment.  Clients use this to ban tokens (-100)
    // or boost specific tokens.  Applied to raw logits before temperature.
    for (&token_id, &bias) in logit_bias {
        if (token_id as usize) < vocab_size {
            logits[token_id as usize] += bias;
        }
    }

    // Greedy with no logprobs: just argmax after penalties/bias.
    if params.temperature == 0.0 && !params.logprobs {
        let token_id = argmax_f32(logits);
        return SampleResult {
            token_id,
            logprob: 0.0,
            top_logprobs: Vec::new(),
        };
    }

    // --- Step 3: Temperature scaling ---
    // Divide every logit by T.  T < 1 sharpens, T > 1 flattens.
    // Special case: temperature=0 with logprobs — we need probabilities for the
    // logprob output, but want deterministic (argmax) selection.  We skip
    // temperature scaling entirely and just compute softmax on raw logits,
    // then force-select the argmax.
    let greedy_with_logprobs = params.temperature == 0.0 && params.logprobs;
    if params.temperature != 0.0 {
        let inv_temp = 1.0 / params.temperature;
        for logit in logits.iter_mut() {
            *logit *= inv_temp;
        }
    }

    // --- Step 4: Top-k masking ---
    // Keep only the k highest logits, set the rest to -inf.
    // Uses select_nth_unstable for O(n) partial sort (no full sort needed).
    if params.top_k > 0 && (params.top_k as usize) < vocab_size {
        let k = params.top_k as usize;
        // Find the kth-largest value via partial sort on a copy of indices.
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        // select_nth_unstable_by partitions so element at k-1 is the kth-largest.
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = indexed[k - 1].1;
        // Mask everything below the threshold.  In case of ties at the boundary,
        // we keep all tokens at exactly the threshold value.
        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    // --- Step 5: Softmax → probabilities ---
    // softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)
    // Subtract max for numerical stability (exp(1000) overflows, exp(0) = 1).
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for logit in logits.iter_mut() {
        *logit = (*logit - max_logit).exp();
        sum += *logit;
    }
    let inv_sum = 1.0 / sum;
    for prob in logits.iter_mut() {
        *prob *= inv_sum;
    }
    // `logits` is now a valid probability distribution (sums to 1.0).
    // Rename conceptually: logits[i] is now probs[i].

    // --- Step 6: Snapshot for logprobs ---
    // Capture the post-softmax, pre-filter distribution for top_logprobs.
    // This reflects the model's actual beliefs before min-p/top-p filtering.
    let logprob_snapshot = if params.logprobs {
        Some(logits.to_vec())
    } else {
        None
    };

    // --- Step 7: Min-p filtering ---
    // Zero out tokens with probability < min_p × max_probability.
    // Adaptive like top-p but without sorting.
    if params.min_p > 0.0 {
        let max_prob = logits.iter().copied().fold(0.0f32, f32::max);
        let threshold = params.min_p * max_prob;
        let mut needs_renorm = false;
        for prob in logits.iter_mut() {
            if *prob < threshold {
                *prob = 0.0;
                needs_renorm = true;
            }
        }
        if needs_renorm {
            let new_sum: f32 = logits.iter().sum();
            if new_sum > 0.0 {
                let inv = 1.0 / new_sum;
                for prob in logits.iter_mut() {
                    *prob *= inv;
                }
            }
        }
    }

    // --- Step 8: Top-p (nucleus) filtering ---
    if params.top_p < 1.0 {
        let mut indices: Vec<u32> = (0..logits.len() as u32).collect();
        indices.sort_unstable_by(|&a, &b| {
            logits[b as usize]
                .partial_cmp(&logits[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cumulative = 0.0f32;
        let mut cutoff_idx = indices.len();
        for (i, &token_idx) in indices.iter().enumerate() {
            cumulative += logits[token_idx as usize];
            if cumulative >= params.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        for &token_idx in &indices[cutoff_idx..] {
            logits[token_idx as usize] = 0.0;
        }
        let new_sum: f32 = logits.iter().sum();
        if new_sum > 0.0 {
            let inv_new_sum = 1.0 / new_sum;
            for prob in logits.iter_mut() {
                *prob *= inv_new_sum;
            }
        }
    }

    // --- Step 9: Weighted random sampling ---
    // For greedy with logprobs, skip random sampling and force argmax.
    let selected_id = if greedy_with_logprobs {
        argmax_f32(logits)
    } else {
        let r: f32 = rng.random();
        let mut cumulative = 0.0f32;
        let mut id = (logits.len() - 1) as u32;
        for (i, &prob) in logits.iter().enumerate() {
            cumulative += prob;
            if cumulative > r {
                id = i as u32;
                break;
            }
        }
        id
    };

    // --- Step 10: Logprob extraction ---
    let (logprob, top_logprobs) = if let Some(ref snapshot) = logprob_snapshot {
        let selected_prob = snapshot[selected_id as usize];
        let logprob = if selected_prob > 0.0 { selected_prob.ln() } else { f32::NEG_INFINITY };

        let n = params.top_logprobs as usize;
        let top_logprobs = if n > 0 {
            // Partial sort to find the top-N tokens by probability.
            let mut indexed: Vec<(u32, f32)> = snapshot
                .iter()
                .enumerate()
                .filter(|(_, p)| **p > 0.0)
                .map(|(i, &p)| (i as u32, p))
                .collect();
            // Sort descending by probability; take top N.
            if indexed.len() > n {
                indexed.select_nth_unstable_by(n - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                indexed.truncate(n);
            }
            indexed.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed
                .into_iter()
                .map(|(id, p)| TokenLogprob {
                    token_id: id,
                    logprob: p.ln(),
                })
                .collect()
        } else {
            Vec::new()
        };

        (logprob, top_logprobs)
    } else {
        (0.0, Vec::new())
    };

    SampleResult {
        token_id: selected_id,
        logprob,
        top_logprobs,
    }
}

// ---------------------------------------------------------------------------
// Greedy sampling functions (unchanged — GPU-resident fast paths).
// ---------------------------------------------------------------------------

/// Greedy sampling: copy logits to host, return the argmax token ID.
///
/// This is the only place where data moves from GPU → CPU during generation.
/// On Apple Silicon with unified memory, "copy" is just a pointer read —
/// the data is already in the same physical memory.
pub(crate) fn sample_greedy<B: GpuBackend>(backend: &B, logits: &B::Tensor, vocab_size: usize) -> anyhow::Result<u32> {
    let byte_count = backend.tensor_byte_count(logits);
    let mut buf = vec![0u8; byte_count];
    backend.copy_to_host(logits, &mut buf);
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
    let output = backend.alloc_tensor(&[1], TensorDtype::F32);
    backend.argmax_gpu(logits, &output, vocab_size as u32, 1);
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
    let mut buf = vec![0u8; batch_size * 4];
    backend.copy_to_host(&output, &mut buf);
    let ids: &[u32] = bytemuck::cast_slice(&buf);
    Ok(ids.to_vec())
}

// ---------------------------------------------------------------------------
// Internal helpers.
// ---------------------------------------------------------------------------

/// Find the index of the maximum value in a bf16 byte slice.
fn argmax_bf16(data: &[u8]) -> u32 {
    let values: &[bf16] = bytemuck::cast_slice(data);
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Find the index of the maximum value in an f32 slice.
fn argmax_f32(data: &[f32]) -> u32 {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Mask disallowed tokens to negative infinity.
///
/// Sets all logits NOT in the `allowed_tokens` set to `f32::NEG_INFINITY`.
/// After softmax, these become probability 0 — they can never be sampled.
fn apply_token_mask(logits: &mut [f32], allowed_tokens: &[u32]) {
    let mut allowed = vec![false; logits.len()];
    for &id in allowed_tokens {
        if (id as usize) < allowed.len() {
            allowed[id as usize] = true;
        }
    }
    for (i, logit) in logits.iter_mut().enumerate() {
        if !allowed[i] {
            *logit = f32::NEG_INFINITY;
        }
    }
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

    /// Helper: run the full pipeline on f32 logits with given params.
    fn sample_with_params(
        logits: &[f32],
        params: &SampleParams,
        rng: &mut impl Rng,
        counts: &HashMap<u32, u32>,
        bias: &HashMap<u32, f32>,
    ) -> SampleResult {
        let mut logits_f32 = logits.to_vec();
        sample_from_logits(&mut logits_f32, params, rng, counts, bias)
    }

    fn default_params() -> SampleParams {
        SampleParams::default()
    }

    fn no_counts() -> HashMap<u32, u32> {
        HashMap::new()
    }

    fn no_bias() -> HashMap<u32, f32> {
        HashMap::new()
    }

    // -- argmax tests (unchanged) --

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
        assert_eq!(argmax_bf16(&data), 1);
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

    #[test]
    fn test_argmax_bf16_two_elements_equal() {
        let data = bf16_bytes(&[5.0, 5.0]);
        let result = argmax_bf16(&data);
        assert!(result == 0 || result == 1);
    }

    // -- basic sampling tests --

    #[test]
    fn test_very_low_temperature_acts_like_argmax() {
        let logits = [1.0, 5.0, 2.0, 0.5];
        let mut params = default_params();
        params.temperature = 0.01;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for _ in 0..20 {
            let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &no_bias());
            assert_eq!(result.token_id, 1, "Very low temperature should pick argmax");
        }
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let logits = [1.0, 2.0, 3.0, 2.0, 1.0];
        let params = default_params();
        let mut rng1 = rand::rngs::SmallRng::seed_from_u64(123);
        let mut rng2 = rand::rngs::SmallRng::seed_from_u64(123);
        for _ in 0..50 {
            let r1 = sample_with_params(&logits, &params, &mut rng1, &no_counts(), &no_bias());
            let r2 = sample_with_params(&logits, &params, &mut rng2, &no_counts(), &no_bias());
            assert_eq!(r1.token_id, r2.token_id, "Same seed should produce same token");
        }
    }

    #[test]
    fn test_top_p_filters_tail() {
        let logits = [10.0, 0.0, 0.0, 0.0];
        let mut params = default_params();
        params.top_p = 0.5;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(99);
        for _ in 0..50 {
            let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &no_bias());
            assert_eq!(result.token_id, 0, "Top-p should filter to dominant token");
        }
    }

    #[test]
    fn test_single_token_vocab() {
        let logits = [42.0];
        let params = default_params();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &no_bias());
        assert_eq!(result.token_id, 0);
    }

    #[test]
    fn test_equal_logits_samples_all_tokens() {
        let logits = [0.0, 0.0, 0.0, 0.0];
        let params = default_params();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let mut counts = [0u32; 4];
        for _ in 0..1000 {
            let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &no_bias());
            counts[result.token_id as usize] += 1;
        }
        for (i, &count) in counts.iter().enumerate() {
            assert!(count > 50, "Token {i} sampled only {count} times out of 1000");
        }
    }

    #[test]
    fn test_high_temperature_more_uniform() {
        let logits = [5.0, 0.0, 0.0, 0.0];

        let mut params_low = default_params();
        params_low.temperature = 0.5;
        let mut rng_low = rand::rngs::SmallRng::seed_from_u64(42);
        let count_low = (0..500)
            .filter(|_| {
                sample_with_params(&logits, &params_low, &mut rng_low, &no_counts(), &no_bias())
                    .token_id == 0
            })
            .count();

        let mut params_high = default_params();
        params_high.temperature = 3.0;
        let mut rng_high = rand::rngs::SmallRng::seed_from_u64(42);
        let count_high = (0..500)
            .filter(|_| {
                sample_with_params(&logits, &params_high, &mut rng_high, &no_counts(), &no_bias())
                    .token_id == 0
            })
            .count();

        assert!(
            count_low > count_high,
            "Low temp ({count_low}) should pick token 0 more than high temp ({count_high})"
        );
    }

    #[test]
    fn test_top_p_one_no_filtering() {
        let logits = [1.0, 1.0, 1.0, 1.0, 1.0];
        let params = default_params();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(55);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &no_bias());
            seen.insert(result.token_id);
        }
        assert_eq!(seen.len(), 5, "top_p=1.0 should allow all 5 tokens");
    }

    // -- New feature tests --

    #[test]
    fn test_frequency_penalty_suppresses_repeated_token() {
        // Token 0 has the highest logit, but heavy frequency penalty should
        // force the sampler to pick something else.
        let logits = [5.0, 4.0, 3.0, 2.0];
        let mut params = default_params();
        params.temperature = 0.01; // near-greedy
        let mut counts = HashMap::new();
        counts.insert(0, 10); // token 0 appeared 10 times
        params.frequency_penalty = 2.0; // penalty: 2.0 * 10 = 20 → logit becomes -15

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for _ in 0..20 {
            let result = sample_with_params(&logits, &params, &mut rng, &counts, &no_bias());
            assert_ne!(result.token_id, 0, "Frequency penalty should suppress token 0");
        }
    }

    #[test]
    fn test_presence_penalty_suppresses_seen_token() {
        let logits = [5.0, 4.8, 3.0, 2.0];
        let mut params = default_params();
        params.temperature = 0.01;
        let mut counts = HashMap::new();
        counts.insert(0, 1); // token 0 appeared once
        params.presence_penalty = 5.0; // flat penalty of 5.0

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for _ in 0..20 {
            let result = sample_with_params(&logits, &params, &mut rng, &counts, &no_bias());
            assert_ne!(result.token_id, 0, "Presence penalty should suppress token 0");
        }
    }

    #[test]
    fn test_top_k_limits_candidates() {
        let logits = [1.0, 2.0, 3.0, 4.0]; // token 3 highest, then 2, 1, 0
        let mut params = default_params();
        params.top_k = 2; // only tokens 3 and 2 should be candidates

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &no_bias());
            seen.insert(result.token_id);
        }
        assert!(
            seen.len() <= 2,
            "top_k=2 should limit to at most 2 tokens, got {seen:?}"
        );
        assert!(seen.contains(&3), "Should include the top token");
        assert!(seen.contains(&2), "Should include the second token");
    }

    #[test]
    fn test_min_p_filters_low_probability_tokens() {
        // Token 0 dominates; others are very low probability.
        let logits = [10.0, 0.0, 0.0, 0.0];
        let mut params = default_params();
        params.min_p = 0.1; // anything < 10% of max_prob gets filtered

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &no_bias());
            assert_eq!(result.token_id, 0, "min_p should filter weak tokens");
        }
    }

    #[test]
    fn test_logit_bias_boosts_token() {
        // Token 3 has the lowest logit, but a large positive bias should
        // make it the clear winner.
        let logits = [5.0, 4.0, 3.0, 0.0];
        let mut params = default_params();
        params.temperature = 0.01;
        let mut bias = HashMap::new();
        bias.insert(3, 100.0); // massive boost

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for _ in 0..20 {
            let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &bias);
            assert_eq!(result.token_id, 3, "Logit bias should boost token 3 to winner");
        }
    }

    #[test]
    fn test_logprob_near_certain() {
        // One token dominates → its logprob should be near 0.0 (ln(1.0) = 0).
        let logits = [100.0, 0.0, 0.0, 0.0];
        let mut params = default_params();
        params.logprobs = true;
        params.top_logprobs = 3;
        params.temperature = 1.0;

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &no_bias());
        assert_eq!(result.token_id, 0);
        assert!(
            result.logprob > -0.01,
            "Near-certain token should have logprob ≈ 0.0, got {}",
            result.logprob
        );
    }

    #[test]
    fn test_logprob_values() {
        // Uniform distribution over 10 tokens: each has prob ≈ 0.1, logprob ≈ -2.3.
        let logits = [0.0; 10];
        let mut params = default_params();
        params.logprobs = true;
        params.top_logprobs = 5;

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &no_bias());

        let expected_logprob = (0.1f32).ln(); // ≈ -2.302
        assert!(
            (result.logprob - expected_logprob).abs() < 0.05,
            "Uniform 10-token logprob should be ≈ {expected_logprob}, got {}",
            result.logprob
        );
        assert_eq!(result.top_logprobs.len(), 5);
        for tlp in &result.top_logprobs {
            assert!(
                (tlp.logprob - expected_logprob).abs() < 0.05,
                "Each top_logprob should be ≈ {expected_logprob}, got {}",
                tlp.logprob
            );
        }
    }

    #[test]
    fn test_top_logprobs_sorted_descending() {
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut params = default_params();
        params.logprobs = true;
        params.top_logprobs = 3;

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &no_bias());

        assert_eq!(result.top_logprobs.len(), 3);
        // Should be sorted descending by logprob.
        for w in result.top_logprobs.windows(2) {
            assert!(
                w[0].logprob >= w[1].logprob,
                "top_logprobs should be sorted descending: {} >= {}",
                w[0].logprob,
                w[1].logprob
            );
        }
        // The top token should be token 4 (highest logit).
        assert_eq!(result.top_logprobs[0].token_id, 4);
    }

    #[test]
    fn test_greedy_with_logprobs() {
        // Temperature=0 with logprobs should still compute probabilities.
        let logits = [1.0, 5.0, 2.0, 0.5];
        let mut params = default_params();
        params.temperature = 0.0;
        params.logprobs = true;
        params.top_logprobs = 2;

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let result = sample_with_params(&logits, &params, &mut rng, &no_counts(), &no_bias());

        // Should pick the argmax (token 1).
        assert_eq!(result.token_id, 1);
        // Logprob should be close to 0 (token 1 dominates with logit=5.0).
        // With no temperature scaling, softmax(5.0) ≈ 0.93, logprob ≈ -0.08.
        assert!(result.logprob > -0.1, "Greedy logprob should be near 0, got {}", result.logprob);
        assert_eq!(result.top_logprobs.len(), 2);
    }

    #[test]
    fn test_pipeline_order_penalties_before_temperature() {
        // Verify penalties are applied to raw logits (before temperature).
        // Token 0 logit=10, token 1 logit=9. With freq_penalty=1.5 and count[0]=2,
        // token 0 becomes 10 - 1.5*2 = 7 < 9 (token 1 wins).
        // If penalties were applied after temperature, the result would differ.
        let logits = [10.0, 9.0];
        let mut params = default_params();
        params.temperature = 0.01;
        params.frequency_penalty = 1.5;
        let mut counts = HashMap::new();
        counts.insert(0, 2);

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let result = sample_with_params(&logits, &params, &mut rng, &counts, &no_bias());
        assert_eq!(result.token_id, 1, "Penalty should flip the winner from token 0 to token 1");
    }
}
