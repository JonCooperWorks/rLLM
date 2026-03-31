// ===========================================================================
// Inference engine: drives the continuous batching loop.
//
// LEARNING OVERVIEW
//
// What this file does:
//   The engine owns the dispatch, sequence state, and tokenizer, and drives
//   the generation loop for MULTIPLE concurrent sequences.  Each step:
//     1. Admit new requests from the waiting queue.
//     2. Prefill: for sequences still processing their prompt, run the
//        ENTIRE prompt through the model in one batched forward pass (GEMM).
//     3. Decode: for sequences in the generation phase, run one token per
//        sequence through the single-token forward pass.
//     4. Sample next tokens for each sequence.
//     5. Advance state: mark finished sequences, collect results.
//
// Batched prefill:
//   The engine drains ALL pending prefill tokens for a sequence in one call
//   to ModelForward::forward_prefill().  This uses GEMM (mat-mat) instead of mat-vec
//   for all projections, shifting from bandwidth-bound to compute-bound.
//
//   The flow for each prefilling sequence:
//     1. Drain all pending tokens from the queue
//     2. Pre-allocate KV blocks for the whole prompt (prepare_prefill)
//     3. Run the GEMM forward pass (forward_prefill)
//     4. Record the positions as filled (finish_prefill)
//     5. Sample the first generated token from the logits
//
// Continuous batching:
//   Multiple sequences can be in flight simultaneously: some prefilling,
//   some decoding, some finishing.  Finished sequences' KV blocks are
//   returned to the free list for reuse.
//
// Dispatch trait:
//   The step loop is generic over the Dispatch trait (dispatch.rs), which
//   abstracts over single-GPU and multi-GPU backends.  run_step() contains
//   the loop logic once; SingleGpuDispatch and MultiGpuDispatch provide
//   the GPU-specific operations.
//
// Sequence lifecycle:
//   add_request() → waiting queue (pre-assigned SeqId)
//   schedule()    → admitted to active set (FCFS, checked against KV capacity)
//   run_step()    → prefill → decode → sample → finish check
//   collect_finished() → removed from active, KV blocks freed
//
// Related files:
//   - engine/dispatch.rs  — Dispatch trait (GPU-specific operations)
//   - engine/multi_gpu.rs — MultiGpuDispatch + MultiGpuEngine (CUDA N-GPU)
//   - model/kv_cache.rs   — KvPool, SeqKvState, block allocation
//   - api/mod.rs          — worker loop calls InferenceEngine trait
// ===========================================================================

pub(crate) mod dispatch;
pub(crate) mod loader;
pub(crate) mod multi_gpu;

use std::collections::{HashMap, VecDeque};

use crate::model::sampler::{SampleParams, SampleResult, TokenLogprob};

use tracing::{info, debug};

use self::dispatch::Dispatch;
use crate::gpu::GpuBackend;
use crate::model::forward::ModelForward;
use crate::model::kv_cache::{self, BlockHandle, KvPool, RadixPrefixCache, SeqKvState, BLOCK_SIZE};
use crate::model::tokenizer::Tokenizer;
use crate::model::{self, Model, PrefillBuffers};

// ---------------------------------------------------------------------------
// InferenceEngine trait — unified interface for the API worker loop.
//
// The API server's worker loop operates on `&mut dyn InferenceEngine`, so
// it never needs to know how many GPUs are involved or which Dispatch
// implementation is in use.
// ---------------------------------------------------------------------------

/// Unified interface for inference engines.
///
/// Abstracts over single-GPU (`Engine<B>`) and multi-GPU (`MultiGpuEngine`)
/// so the API server's worker loop doesn't need to know the GPU topology.
pub(crate) trait InferenceEngine {
    /// Submit a new completion request.  Returns a sequence ID for tracking.
    ///
    /// `seed` enables deterministic sampling: when provided, the per-sequence
    /// RNG is seeded with this value so the same prompt + seed produces the
    /// same output (single-sequence only — batched decoding may interleave
    /// RNG draws across sequences).
    fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_gen_tokens: usize,
        params: SampleParams,
        images: Vec<crate::model::vision::ProcessedImage>,
        seed: Option<u64>,
        grammar: Option<std::sync::Arc<crate::model::grammar::CompiledGrammar>>,
        logit_bias: HashMap<u32, f32>,
    ) -> SeqId;

    /// Run one engine step: admit → prefill → decode → sample → collect.
    fn step(&mut self) -> anyhow::Result<StepOutput>;

    /// Abort a sequence, freeing its KV cache blocks.
    fn abort_sequence(&mut self, id: SeqId);

    /// Whether the engine has pending or active work.
    fn has_work(&self) -> bool;

    /// Access the tokenizer (for incremental text decoding in the worker loop).
    fn tokenizer(&self) -> &Tokenizer;

    /// Number of sequences currently being processed.
    fn active_count(&self) -> usize;

    /// Number of sequences waiting to be admitted.
    fn waiting_count(&self) -> usize;
}

/// Why a sequence stopped generating.
#[derive(Clone, Copy)]
pub(crate) enum FinishReason {
    /// Hit an end-of-sequence token (EOS/EOT).
    Eos,
    /// Reached the max_tokens limit.
    MaxTokens,
}

/// A finished sequence with its generated text.
pub(crate) struct FinishedSequence {
    pub id: SeqId,
    pub tokens: Vec<u32>,
    pub text: String,
    pub reason: FinishReason,
    /// Number of prompt tokens served from the prefix cache (0 if no hit).
    pub cached_tokens: usize,
}

/// Log-probability information for a single generated token.
///
/// Propagated from the sampler through StepOutput → InferenceEvent → API response.
/// Only populated when the client requests `logprobs: true`.
#[derive(Debug)]
pub(crate) struct TokenLogprobInfo {
    /// Log-probability of the selected token: ln(prob).
    pub logprob: f32,
    /// Top-N alternative tokens with their log-probabilities, sorted descending.
    pub top_logprobs: Vec<TokenLogprob>,
}

/// Output from a single engine step.
pub(crate) struct StepOutput {
    /// Token generated this step for each active sequence (id, token_id, logprob_info).
    pub tokens: Vec<(SeqId, u32, Option<TokenLogprobInfo>)>,
    /// Sequences that finished this step.
    pub finished: Vec<FinishedSequence>,
}

// ---------------------------------------------------------------------------
// Sequence management — waiting queue, active set, FCFS admission.
//
// These types manage the lifecycle of concurrent sequences.  The Scheduler
// is a thin data container: it stores the waiting queue and active set,
// delegates GPU operations (block allocation, sequence creation) to the
// Dispatch trait, and provides admission (schedule) and cleanup
// (collect_finished, abort_sequence) helpers.
// ---------------------------------------------------------------------------

/// Unique identifier for a sequence request.
pub(crate) type SeqId = u64;

/// A new sequence request submitted to the engine.
pub(crate) struct SequenceRequest {
    pub prompt_tokens: Vec<u32>,
    pub max_gen_tokens: usize,
    /// Sampling parameters (temperature, top_p, top_k, penalties, logprobs, etc.).
    pub params: SampleParams,
    /// Preprocessed images for vision models (consumed during first prefill).
    pub images: Vec<crate::model::vision::ProcessedImage>,
    /// Optional seed for deterministic sampling.
    pub seed: Option<u64>,
    /// Precompiled grammar for structured output (None for unconstrained generation).
    pub grammar: Option<std::sync::Arc<crate::model::grammar::CompiledGrammar>>,
    /// Per-token logit bias (token_id → additive adjustment).
    pub logit_bias: HashMap<u32, f32>,
}

/// State of a single active sequence.
///
/// Generic over `S` — the per-sequence KV cache state provided by the
/// Dispatch implementation.  Single-GPU uses `SeqKvState<B>`, multi-GPU
/// uses `Vec<SeqKvState<CudaBackend>>`.
pub(crate) struct Sequence<S> {
    /// Tokens remaining to prefill (drained as prefill progresses).
    pub pending_prefill: VecDeque<u32>,
    /// KV cache state for this sequence (opaque to the scheduler).
    pub kv_state: S,
    /// Generated tokens so far (includes the first sampled token from prefill).
    pub generated_tokens: Vec<u32>,
    /// Maximum tokens to generate after prefill.
    pub max_gen_tokens: usize,
    /// Sampling parameters (temperature, top_p, top_k, penalties, logprobs, etc.).
    pub params: SampleParams,
    /// Whether this sequence has finished (EOS or max_tokens reached).
    pub finished: bool,
    /// Number of prompt tokens served from the prefix cache.
    pub cached_tokens: usize,
    /// Preprocessed images (consumed during first prefill chunk, then empty).
    pub images: Vec<crate::model::vision::ProcessedImage>,
    /// Per-sequence RNG for deterministic sampling.
    /// When the client provides a `seed`, this RNG is seeded with that value
    /// so the same prompt + seed produces the same output.  When None, the
    /// global RNG in run_step() is used instead.
    pub seeded_rng: Option<rand::rngs::SmallRng>,
    /// Original prompt tokens, preserved for recompute-based preemption.
    /// When a sequence is evicted to free KV blocks, this + generated_tokens
    /// are concatenated to form the new prompt for re-queuing.
    pub original_prompt_tokens: Vec<u32>,
    /// Whether this sequence had images (vision model).  Once images are
    /// consumed during prefill (std::mem::take), we can't recover them —
    /// so vision sequences are excluded from preemption.
    pub had_images: bool,
    /// Per-sequence grammar constraint state (None for unconstrained generation).
    /// Tracks the current DFA state for structured output.  Advanced after
    /// each sampled token; checked for completion alongside EOS/max_tokens.
    pub grammar_state: Option<crate::model::grammar::GrammarState>,
    /// Per-token frequency counts for this sequence's generated tokens.
    /// Used by frequency_penalty and presence_penalty to reduce repetition.
    pub token_counts: HashMap<u32, u32>,
    /// Per-token logit bias for this sequence.
    /// Applied as additive adjustment to raw logits before temperature scaling.
    pub logit_bias: HashMap<u32, f32>,
}

/// Manages the waiting queue and active set for continuous batching.
///
/// Generic over `S` — the per-sequence KV cache state.  Delegates GPU-specific
/// operations to the Dispatch trait.
pub(crate) struct Scheduler<S> {
    /// Waiting queue of new requests (ID pre-assigned at submission time).
    waiting: VecDeque<(SeqId, SequenceRequest)>,
    /// Active sequences currently being processed.
    pub active: HashMap<SeqId, Sequence<S>>,
    /// Next sequence ID.
    next_id: SeqId,
    /// Maximum concurrent sequences.
    max_active: usize,
}

impl<S> Scheduler<S> {
    pub fn new(max_active: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            active: HashMap::new(),
            next_id: 0,
            max_active,
        }
    }

    /// Submit a new sequence request.  Returns the pre-assigned sequence ID
    /// so the caller can track this request before it's admitted to the active set.
    pub fn add_request(&mut self, req: SequenceRequest) -> SeqId {
        let id = self.next_id;
        self.next_id += 1;
        self.waiting.push_back((id, req));
        id
    }

    /// Try to admit waiting requests into the active set (FCFS).
    ///
    /// Uses the Dispatch to check free block count (admission heuristic) and
    /// to create new per-sequence KV state.  Blocks are not actually allocated
    /// here — that happens later in prepare_prefill/prepare_decode.
    pub fn schedule(&mut self, dispatch: &impl Dispatch<SeqState = S>) -> Vec<SeqId> {
        let mut admitted = Vec::new();

        while !self.waiting.is_empty() && self.active.len() < self.max_active {
            let (_, req) = self.waiting.front().unwrap();
            let prompt_blocks = kv_cache::blocks_needed_for(req.prompt_tokens.len());
            if dispatch.free_block_count() < prompt_blocks {
                break;
            }

            let (id, req) = self.waiting.pop_front().unwrap();
            let seeded_rng = req.seed.map(|s| {
                use rand::SeedableRng;
                rand::rngs::SmallRng::seed_from_u64(s)
            });
            let had_images = !req.images.is_empty();
            let original_prompt_tokens = req.prompt_tokens.clone();
            let grammar_state = req.grammar.map(crate::model::grammar::GrammarState::new);
            let seq = Sequence {
                pending_prefill: req.prompt_tokens.into(),
                kv_state: dispatch.new_seq_state(),
                generated_tokens: Vec::new(),
                max_gen_tokens: req.max_gen_tokens,
                params: req.params,
                finished: false,
                cached_tokens: 0,
                images: req.images,
                seeded_rng,
                original_prompt_tokens,
                had_images,
                grammar_state,
                token_counts: HashMap::new(),
                logit_bias: req.logit_bias,
            };
            self.active.insert(id, seq);
            admitted.push(id);
        }

        admitted
    }

    /// Abort a sequence, removing it from the waiting queue or active set.
    /// Frees KV blocks if the sequence was active.
    pub fn abort_sequence(&mut self, id: SeqId, dispatch: &mut impl Dispatch<SeqState = S>) {
        self.waiting.retain(|(wid, _)| *wid != id);
        if let Some(seq) = self.active.remove(&id) {
            dispatch.free_seq_state(&seq.kv_state);
        }
    }

    /// Collect and remove finished sequences, freeing their KV blocks.
    pub fn collect_finished(
        &mut self,
        dispatch: &mut impl Dispatch<SeqState = S>,
    ) -> Vec<(SeqId, Sequence<S>)> {
        let finished_ids: Vec<SeqId> = self
            .active
            .iter()
            .filter(|(_, seq)| seq.finished)
            .map(|(&id, _)| id)
            .collect();

        let mut results = Vec::new();
        for id in finished_ids {
            if let Some(seq) = self.active.remove(&id) {
                dispatch.free_seq_state(&seq.kv_state);
                results.push((id, seq));
            }
        }
        results
    }

    /// Whether there's any work to do.
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.active.is_empty()
    }

    /// Whether there are requests waiting to be admitted.
    pub fn has_waiting(&self) -> bool {
        !self.waiting.is_empty()
    }

    /// Try to preempt (evict) an active sequence to free KV blocks for waiting
    /// requests.  Uses recompute-based preemption: the evicted sequence's
    /// original prompt + generated tokens are concatenated and re-queued as a
    /// new request.  On re-admission it re-prefills (prefix cache may hit on
    /// the system prompt portion, avoiding redundant work).
    ///
    /// Victim selection: pick the decoding sequence with the most generated
    /// tokens — it holds the most KV blocks, so evicting it frees the most
    /// memory.  Sequences still prefilling or with vision data are excluded.
    ///
    /// Returns the evicted sequence's original ID (for logging), or None if
    /// no suitable victim was found.
    pub fn try_preempt(&mut self, dispatch: &mut impl Dispatch<SeqState = S>) -> Option<SeqId> {
        // Don't evict if nothing is waiting — there's no pressure.
        if self.waiting.is_empty() {
            return None;
        }

        // Don't evict the only active sequence — it would just be re-admitted
        // into the same deadlock (needs blocks, but only its own blocks exist).
        if self.active.len() <= 1 {
            return None;
        }

        // Find the best victim: decoding (not prefilling), no vision images,
        // and with the most generated tokens (holds the most blocks).
        let victim_id = self
            .active
            .iter()
            .filter(|(_, seq)| {
                seq.pending_prefill.is_empty() // Must be decoding, not prefilling
                    && !seq.finished            // Not already finishing
                    && !seq.had_images          // Vision sequences can't be preempted
                    && seq.grammar_state.is_none() // Grammar state can't be cheaply replayed
            })
            .max_by_key(|(_, seq)| seq.generated_tokens.len())
            .map(|(&id, _)| id)?;

        let seq = self.active.remove(&victim_id).unwrap();

        info!(
            seq_id = victim_id,
            generated_tokens = seq.generated_tokens.len(),
            "preempt: evicting seq to free KV blocks",
        );

        // Free all KV blocks (releases prefix cache refs too).
        dispatch.free_seq_state(&seq.kv_state);

        // Re-queue: original prompt + generated tokens become the new prompt.
        // The model will re-prefill everything (prefix cache likely hits on the
        // system prompt portion).  max_gen_tokens is reduced by however many
        // tokens were already generated.
        let mut new_prompt = seq.original_prompt_tokens;
        new_prompt.extend_from_slice(&seq.generated_tokens);
        let remaining_gen = seq.max_gen_tokens.saturating_sub(seq.generated_tokens.len());

        let new_req = SequenceRequest {
            prompt_tokens: new_prompt,
            max_gen_tokens: remaining_gen,
            params: seq.params,
            images: Vec::new(), // Images were consumed during original prefill
            seed: None,         // Determinism lost after partial generation
            grammar: None,      // Grammar state can't be cheaply replayed
            logit_bias: seq.logit_bias,
        };

        // Push to front — this sequence already waited and did work.
        self.waiting.push_front((self.next_id, new_req));
        self.next_id += 1;

        Some(victim_id)
    }

    /// Number of waiting requests.
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Number of active (in-flight) sequences.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}

// ---------------------------------------------------------------------------
// run_step — the shared engine step loop, generic over Dispatch.
//
// This is the single implementation of admit → prefill → decode → sample →
// collect that both single-GPU and multi-GPU engines use.
// ---------------------------------------------------------------------------

/// Run one engine step: admit → prefill → decode → sample → collect.
///
/// This is the shared implementation used by both single-GPU and multi-GPU
/// engines.  The `dispatch` handles all GPU-specific operations; the
/// `scheduler` manages sequence lifecycle; the `tokenizer` detects EOS.
pub(crate) fn run_step<D: Dispatch>(
    dispatch: &mut D,
    scheduler: &mut Scheduler<D::SeqState>,
    tokenizer: &Tokenizer,
) -> anyhow::Result<StepOutput> {
    let mut rng = rand::rng();
    let mut step_tokens: Vec<(SeqId, u32, Option<TokenLogprobInfo>)> = Vec::new();

    // 1. Admit waiting requests.
    let admitted = scheduler.schedule(dispatch);

    // 1b. Preemption: if nothing was admitted but requests are waiting,
    //     evict the active sequence holding the most KV blocks and retry.
    //     This prevents deadlock when the KV cache is full and new requests
    //     can't be admitted.  The evicted sequence is re-queued with its
    //     original prompt + generated tokens (recompute-based preemption).
    if admitted.is_empty() && scheduler.has_waiting() {
        if scheduler.try_preempt(dispatch).is_some() {
            scheduler.schedule(dispatch);
        }
    }

    // 2. Chunked prefill: process ONE chunk per prefilling sequence per step.
    //
    //    Long prompts are split into max_prefill_chunk-sized pieces.  Only one
    //    chunk is processed per step, then control returns to the decode phase
    //    so decoding sequences aren't starved.  This is the key to avoiding
    //    head-of-line blocking on long prompts and thinking models.
    //
    //    Prefix caching runs on the FIRST chunk only (when the sequence hasn't
    //    started prefilling yet, i.e. seq_len == 0 and cached_tokens == 0).
    //    After all chunks are done (pending_prefill empty), we register the
    //    prefix and sample the first token.
    let prefilling_ids: Vec<SeqId> = scheduler
        .active
        .iter()
        .filter(|(_, seq)| !seq.pending_prefill.is_empty() && !seq.finished)
        .map(|(&id, _)| id)
        .collect();

    let max_chunk = dispatch.max_prefill_chunk();

    for id in prefilling_ids {
        let seq = scheduler.active.get_mut(&id).unwrap();

        // --- Prefix cache lookup (first chunk only) ---
        // On the first prefill step for this sequence (seq_len == 0 and no
        // cached prefix), check if a prefix is already in the cache.  If so,
        // link the cached blocks and skip those tokens.
        let first_chunk = D::seq_len(&seq.kv_state) == 0 && seq.cached_tokens == 0;
        if first_chunk {
            // Peek at full pending tokens for cache lookup (without draining).
            let all_tokens: Vec<u32> = seq.pending_prefill.iter().copied().collect();
            if let Some((prefix_blocks, prefix_token_count)) =
                dispatch.prefix_cache_lookup(&all_tokens)
            {
                let prefix_len = prefix_token_count;
                let prefix_block_count = prefix_blocks.len();

                // Block-aligned prefix tokens for cache release on free.
                let prefix_tokens = all_tokens[..prefix_len].to_vec();

                dispatch.link_prefix(
                    &mut seq.kv_state,
                    &prefix_blocks,
                    prefix_token_count,
                    prefix_tokens,
                );
                seq.cached_tokens = prefix_len;

                // Skip the cached prefix tokens in pending_prefill.
                seq.pending_prefill.drain(..prefix_len);

                debug!(
                    seq_id = id,
                    cached_tokens = prefix_len,
                    cached_blocks = prefix_block_count,
                    suffix_tokens = seq.pending_prefill.len(),
                    "prefix cache hit",
                );
            }
        }

        // Drain ONE chunk from pending_prefill.
        let chunk_size = max_chunk.min(seq.pending_prefill.len());
        if chunk_size > 0 {
            let chunk: Vec<u32> = seq.pending_prefill.drain(..chunk_size).collect();
            dispatch.prepare_prefill(&mut seq.kv_state, chunk_size)?;
            // Images are passed only on the FIRST chunk (vision encoding happens
            // once; subsequent chunks are pure text continuation).
            let images = if first_chunk {
                std::mem::take(&mut seq.images)
            } else {
                Vec::new()
            };
            dispatch.forward_prefill(&chunk, &seq.kv_state, &images)?;
            D::finish_prefill(&mut seq.kv_state, chunk_size);
        }

        // If all tokens are now prefilled, register the prefix cache and
        // sample the first token.  Otherwise, this sequence continues
        // prefilling in the next step.
        if seq.pending_prefill.is_empty() {
            // --- Prefix cache registration ---
            // Register the block-aligned prefix for future reuse.
            // Only register if we didn't already use a cached prefix.
            if seq.cached_tokens == 0 {
                let tokens = &seq.original_prompt_tokens;
                dispatch.prefix_cache_register(tokens, &mut seq.kv_state);
            }

            sample_and_finish(dispatch, seq, id, tokenizer, &mut step_tokens, &mut rng)?;
        }
    }

    // 3. Decode: run one token per decoding sequence.
    //
    //    Two paths: batched (GEMM) when multiple sequences are decoding and
    //    the architecture supports it, or serial (mat-vec) as fallback.
    //
    //    Batched decode turns N separate mat-vec passes into 1 GEMM pass,
    //    giving ~3-5x throughput for N=4-16 concurrent sequences.  On multi-GPU,
    //    this also reduces NCCL AllReduce calls from N×2×num_layers to 2×num_layers.
    let decoding_ids: Vec<SeqId> = scheduler
        .active
        .iter()
        .filter(|(_, seq)| seq.pending_prefill.is_empty() && !seq.finished)
        .map(|(&id, _)| id)
        .collect();

    if decoding_ids.len() > 1 && dispatch.supports_batched_decode() {
        // --- Batched decode path ---

        // Phase 1: prepare all sequences (mutable borrows for KV slot allocation).
        let mut tokens = Vec::with_capacity(decoding_ids.len());
        for &id in &decoding_ids {
            let seq = scheduler.active.get_mut(&id).unwrap();
            tokens.push(*seq.generated_tokens.last().unwrap());
            dispatch.prepare_decode(&mut seq.kv_state)?;
        }

        // Phase 2: collect immutable state refs and positions.
        // This is a separate phase because prepare_decode borrows mutably,
        // and we need immutable refs for the forward pass.
        let positions: Vec<u32> = decoding_ids
            .iter()
            .map(|id| D::seq_len(&scheduler.active[id].kv_state) as u32)
            .collect();
        let state_refs: Vec<&D::SeqState> = decoding_ids
            .iter()
            .map(|id| &scheduler.active[id].kv_state)
            .collect();

        // Phase 3: one batched forward pass for all N sequences.
        dispatch.forward_decode_batch(&tokens, &positions, &state_refs)?;

        // Phase 4: advance all KV states and sample.
        for &id in &decoding_ids {
            D::finish_decode(&mut scheduler.active.get_mut(&id).unwrap().kv_state);
        }
        model::profile::tick();

        // Collect per-sequence sampling params, grammar constraints, token counts, and bias.
        let params_per_seq: Vec<&SampleParams> = decoding_ids
            .iter()
            .map(|id| &scheduler.active[id].params)
            .collect();
        let allowed_tokens_per_seq: Vec<Option<Vec<u32>>> = decoding_ids
            .iter()
            .map(|id| {
                scheduler.active[id]
                    .grammar_state
                    .as_ref()
                    .and_then(|gs| gs.allowed_tokens())
            })
            .collect();
        let token_counts_per_seq: Vec<&HashMap<u32, u32>> = decoding_ids
            .iter()
            .map(|id| &scheduler.active[id].token_counts)
            .collect();
        let logit_bias_per_seq: Vec<&HashMap<u32, f32>> = decoding_ids
            .iter()
            .map(|id| &scheduler.active[id].logit_bias)
            .collect();
        let any_grammar = allowed_tokens_per_seq.iter().any(|a| a.is_some());

        // Greedy gate: if ALL sequences can use GPU-resident argmax (temperature==0,
        // no logprobs, no penalties, no bias) and none have grammar constraints,
        // try GPU-resident batched argmax.  Otherwise fall back to full CPU pipeline.
        let all_gpu_greedy = params_per_seq.iter().all(|p| p.can_use_gpu_greedy())
            && logit_bias_per_seq.iter().all(|b| b.is_empty());
        let sampled: Vec<SampleResult> = if all_gpu_greedy && !any_grammar {
            match dispatch.sample_batch_greedy_gpu(params_per_seq.len()) {
                Ok(ids) => ids
                    .into_iter()
                    .map(|token_id| SampleResult {
                        token_id,
                        logprob: 0.0,
                        top_logprobs: Vec::new(),
                    })
                    .collect(),
                Err(_) => dispatch.sample_batch(
                    &params_per_seq,
                    &mut rng,
                    &allowed_tokens_per_seq,
                    &token_counts_per_seq,
                    &logit_bias_per_seq,
                )?,
            }
        } else {
            dispatch.sample_batch(
                &params_per_seq,
                &mut rng,
                &allowed_tokens_per_seq,
                &token_counts_per_seq,
                &logit_bias_per_seq,
            )?
        };
        for (i, &id) in decoding_ids.iter().enumerate() {
            let seq = scheduler.active.get_mut(&id).unwrap();
            let result = &sampled[i];
            seq.generated_tokens.push(result.token_id);
            *seq.token_counts.entry(result.token_id).or_insert(0) += 1;

            let logprob_info = if seq.params.logprobs {
                Some(TokenLogprobInfo {
                    logprob: result.logprob,
                    top_logprobs: result.top_logprobs.iter().map(|t| TokenLogprob {
                        token_id: t.token_id,
                        logprob: t.logprob,
                    }).collect(),
                })
            } else {
                None
            };
            step_tokens.push((id, result.token_id, logprob_info));

            // Advance grammar state after sampling.
            if let Some(ref mut gs) = seq.grammar_state {
                let _ = gs.advance(result.token_id);
            }

            if tokenizer.is_eos(result.token_id)
                || seq.generated_tokens.len() >= seq.max_gen_tokens
                || seq.grammar_state.as_ref().is_some_and(|gs| gs.is_complete())
            {
                seq.finished = true;
            }
        }
    } else {
        // --- Serial decode path (single sequence or unsupported arch) ---
        for id in decoding_ids {
            let seq = scheduler.active.get_mut(&id).unwrap();
            let token = *seq.generated_tokens.last().unwrap();

            dispatch.prepare_decode(&mut seq.kv_state)?;
            dispatch.forward_decode(token, &seq.kv_state)?;
            D::finish_decode(&mut seq.kv_state);
            model::profile::tick();

            sample_and_finish(dispatch, seq, id, tokenizer, &mut step_tokens, &mut rng)?;
        }
    }

    // 4. Collect finished sequences.
    let finished_pairs = scheduler.collect_finished(dispatch);
    let mut finished = Vec::new();
    for (id, seq) in finished_pairs {
        let text = tokenizer.decode(&seq.generated_tokens).unwrap_or_default();
        let reason = if seq
            .generated_tokens
            .last()
            .map_or(false, |&t| tokenizer.is_eos(t))
        {
            FinishReason::Eos
        } else {
            FinishReason::MaxTokens
        };
        finished.push(FinishedSequence {
            id,
            tokens: seq.generated_tokens,
            text,
            reason,
            cached_tokens: seq.cached_tokens,
        });
    }

    Ok(StepOutput {
        tokens: step_tokens,
        finished,
    })
}

/// Sample a token and check if the sequence should finish (EOS or max_tokens).
///
/// Shared by both the prefill and decode phases of run_step() — after each
/// forward pass, we sample a token, record it, and check stopping conditions.
///
/// When the sequence has a seeded RNG (from a client-provided `seed`), that
/// RNG is used instead of the global one for deterministic output.
fn sample_and_finish<D: Dispatch>(
    dispatch: &D,
    seq: &mut Sequence<D::SeqState>,
    id: SeqId,
    tokenizer: &Tokenizer,
    step_tokens: &mut Vec<(SeqId, u32, Option<TokenLogprobInfo>)>,
    global_rng: &mut impl rand::Rng,
) -> anyhow::Result<()> {
    // Get allowed tokens from grammar state (if grammar-constrained).
    let allowed = seq.grammar_state.as_ref().and_then(|gs| gs.allowed_tokens());
    let allowed_slice = allowed.as_deref();

    let result = if let Some(ref mut seq_rng) = seq.seeded_rng {
        dispatch.sample(&seq.params, seq_rng, allowed_slice, &seq.token_counts, &seq.logit_bias)?
    } else {
        dispatch.sample(&seq.params, global_rng, allowed_slice, &seq.token_counts, &seq.logit_bias)?
    };

    let next_token = result.token_id;

    // Update token frequency counts (for frequency/presence penalty).
    *seq.token_counts.entry(next_token).or_insert(0) += 1;

    // Advance grammar state after sampling.
    if let Some(ref mut gs) = seq.grammar_state {
        gs.advance(next_token)?;
    }

    seq.generated_tokens.push(next_token);

    let logprob_info = if seq.params.logprobs {
        Some(TokenLogprobInfo {
            logprob: result.logprob,
            top_logprobs: result.top_logprobs,
        })
    } else {
        None
    };
    step_tokens.push((id, next_token, logprob_info));

    // Check finish: EOS, max_tokens, or grammar reached accepting state.
    if tokenizer.is_eos(next_token)
        || seq.generated_tokens.len() >= seq.max_gen_tokens
        || seq.grammar_state.as_ref().is_some_and(|gs| gs.is_complete())
    {
        seq.finished = true;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SingleGpuDispatch — Dispatch implementation for single-GPU inference.
//
// Wraps Model + KvPool + PrefillBuffers + backend reference.  Each Dispatch
// method delegates to the appropriate model/kv_cache operation.
// ---------------------------------------------------------------------------

/// Single-GPU dispatch: wraps one Model, one KvPool, one set of PrefillBuffers.
///
/// The `logits_batch` field is allocated for batched decode: when N decoding
/// sequences are processed in one GEMM pass, the LM head produces [N, vocab_size]
/// logits here instead of in the model's single-token logits_buf.
pub(crate) struct SingleGpuDispatch<'a, B: GpuBackend> {
    pub model: Model<'a, B>,
    /// Architecture-specific forward pass (eliminates match dispatch on ModelArch).
    forward: Box<dyn ModelForward<B> + 'a>,
    pub kv_pool: KvPool<B>,
    prefill_bufs: PrefillBuffers<B>,
    /// Tokenizer vocab size — may be smaller than the model's embedding vocab
    /// (e.g. Qwen 3.5: 248070 tokenizer vs 248320 embedding).  Used to clamp
    /// sampling so we never produce token IDs the tokenizer can't decode.
    pub tokenizer_vocab_size: usize,
    /// Batched logits buffer: [max_active, vocab_size] bf16.
    /// Used by `forward_decode_batch` to produce logits for N sequences at once.
    /// Allocated even when batched decode is unsupported — the memory cost is
    /// small (32 seqs × 128K vocab × 2 bytes = ~8 MB).
    logits_batch: B::Tensor,
    backend: &'a B,
    /// Radix tree prefix cache for sharing KV blocks across sequences with
    /// overlapping prefixes.  Block-level sharing: common prefix blocks are
    /// stored once and shared via the trie structure.
    pub prefix_cache: RadixPrefixCache,
}

impl<'a, B: GpuBackend> SingleGpuDispatch<'a, B> {
    pub fn new(
        model: Model<'a, B>,
        forward: Box<dyn ModelForward<B> + 'a>,
        kv_pool: KvPool<B>,
        backend: &'a B,
        max_active: usize,
        tokenizer_vocab_size: usize,
    ) -> Self {
        let prefill_bufs = PrefillBuffers::new(backend, model.config(), 1024);
        let logits_batch = backend.alloc_tensor(
            &[max_active, model.config().vocab_size],
            crate::gpu::TensorDtype::BF16,
        );
        // Radix tree prefix cache: 256 blocks capacity.  With BLOCK_SIZE=16,
        // this caches ~4096 tokens worth of shared prefixes.  Block-level
        // sharing means overlapping system prompts reuse internal nodes.
        let prefix_cache = RadixPrefixCache::new(256);
        Self {
            model,
            forward,
            kv_pool,
            prefill_bufs,
            logits_batch,
            backend,
            prefix_cache,
            tokenizer_vocab_size,
        }
    }
}

impl<'a, B: GpuBackend> Dispatch for SingleGpuDispatch<'a, B> {
    type SeqState = SeqKvState<B>;

    fn new_seq_state(&self) -> SeqKvState<B> {
        self.kv_pool.new_sequence(self.backend)
    }

    fn free_seq_state(&mut self, state: &SeqKvState<B>) {
        // Release prefix cache ref count if this sequence used a cached prefix.
        if let Some(prefix_tokens) = state.shared_prefix_tokens() {
            self.prefix_cache.release(prefix_tokens);
        }
        self.kv_pool.free_sequence(state);
    }

    fn max_prefill_chunk(&self) -> usize {
        self.prefill_bufs.max_chunk
    }

    fn free_block_count(&self) -> usize {
        self.kv_pool.free_block_count()
    }

    fn prepare_prefill(
        &mut self,
        state: &mut SeqKvState<B>,
        token_count: usize,
    ) -> anyhow::Result<()> {
        state.ensure_slots(&mut self.kv_pool, token_count)?;
        state.sync_block_table_validated(self.backend, &self.kv_pool);
        Ok(())
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        state: &SeqKvState<B>,
        images: &[crate::model::vision::ProcessedImage],
    ) -> anyhow::Result<()> {
        self.forward.prefill_preamble(&self.model, tokens, state, &self.prefill_bufs, images)?;
        self.forward.forward_prefill(&self.model, tokens, &self.kv_pool, state, &self.prefill_bufs)
    }

    fn finish_prefill(state: &mut SeqKvState<B>, token_count: usize) {
        state.advance_by(token_count);
    }

    fn seq_len(state: &SeqKvState<B>) -> usize {
        state.seq_len
    }

    fn prepare_decode(&mut self, state: &mut SeqKvState<B>) -> anyhow::Result<()> {
        state.ensure_slot(&mut self.kv_pool)?;
        state.sync_block_table_validated(self.backend, &self.kv_pool);
        Ok(())
    }

    fn forward_decode(&self, token: u32, state: &SeqKvState<B>) -> anyhow::Result<()> {
        self.forward.forward_decode(&self.model, token, &self.kv_pool, state)
    }

    fn finish_decode(state: &mut SeqKvState<B>) {
        state.advance();
    }

    fn sample(
        &self,
        params: &SampleParams,
        rng: &mut impl rand::Rng,
        allowed_tokens: Option<&[u32]>,
        token_counts: &HashMap<u32, u32>,
        logit_bias: &HashMap<u32, f32>,
    ) -> anyhow::Result<SampleResult> {
        crate::model::sampler::sample(
            self.backend, self.model.logits(), params, rng,
            self.tokenizer_vocab_size, allowed_tokens, token_counts, logit_bias,
        )
    }

    fn sample_greedy_gpu(&self) -> anyhow::Result<u32> {
        crate::model::sampler::sample_greedy_gpu(self.backend, self.model.logits(), self.tokenizer_vocab_size)
    }

    fn sample_batch_greedy_gpu(&self, batch_size: usize) -> anyhow::Result<Vec<u32>> {
        crate::model::sampler::sample_batch_greedy_gpu(
            self.backend,
            &self.logits_batch,
            batch_size,
            self.model.config().vocab_size,
        )
    }

    fn prefix_cache_lookup(&mut self, prompt_tokens: &[u32]) -> Option<(Vec<BlockHandle>, usize)> {
        self.prefix_cache.lookup(prompt_tokens)
    }

    fn link_prefix(
        &self,
        state: &mut SeqKvState<B>,
        prefix_handles: &[BlockHandle],
        prefix_token_count: usize,
        prefix_tokens: Vec<u32>,
    ) {
        state.link_prefix(prefix_handles, prefix_token_count, prefix_tokens);
    }

    fn prefix_cache_register(&mut self, tokens: &[u32], state: &mut SeqKvState<B>) {
        // Only cache block-aligned prefixes (partial blocks can't be shared
        // because the next sequence might write different tokens into the
        // remaining slots).
        let prefix_blocks = tokens.len() / BLOCK_SIZE;
        if prefix_blocks == 0 {
            return;
        }
        let prefix_len = prefix_blocks * BLOCK_SIZE;
        let prefix_tokens = tokens[..prefix_len].to_vec();
        let block_indices = state.block_table_cpu_slice()[..prefix_blocks].to_vec();

        let evicted = self.prefix_cache.insert(&prefix_tokens, &block_indices);
        // Return evicted blocks to the free list.
        for handle in &evicted {
            self.kv_pool.free_block(*handle);
        }

        // Mark the prefix blocks as shared on this sequence so free_sequence()
        // won't return them to the pool — they now belong to the cache.
        state.mark_prefix_shared(prefix_blocks, prefix_tokens);
    }

    fn supports_batched_decode(&self) -> bool {
        self.forward.supports_batched_decode()
    }

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        positions: &[u32],
        states: &[&SeqKvState<B>],
    ) -> anyhow::Result<()> {
        self.forward.forward_decode_batch(
            &self.model,
            tokens,
            positions,
            &self.kv_pool,
            states,
            &self.prefill_bufs,
            &self.logits_batch,
        )
    }

    fn sample_batch(
        &self,
        params_per_seq: &[&SampleParams],
        rng: &mut impl rand::Rng,
        allowed_tokens_per_seq: &[Option<Vec<u32>>],
        token_counts_per_seq: &[&HashMap<u32, u32>],
        logit_bias_per_seq: &[&HashMap<u32, f32>],
    ) -> anyhow::Result<Vec<SampleResult>> {
        crate::model::sampler::sample_batch(
            self.backend,
            &self.logits_batch,
            params_per_seq.len(),
            self.model.config().vocab_size,
            params_per_seq,
            rng,
            self.tokenizer_vocab_size,
            allowed_tokens_per_seq,
            token_counts_per_seq,
            logit_bias_per_seq,
        )
    }
}

// ---------------------------------------------------------------------------
// Engine — the single-GPU inference engine.
//
// Owns a SingleGpuDispatch, a Scheduler, and a Tokenizer.  Implements
// InferenceEngine by delegating to run_step().
// ---------------------------------------------------------------------------

/// The single-GPU inference engine.
pub(crate) struct Engine<'a, B: GpuBackend> {
    dispatch: SingleGpuDispatch<'a, B>,
    scheduler: Scheduler<SeqKvState<B>>,
    tokenizer: Tokenizer,
}

impl<'a, B: GpuBackend> Engine<'a, B> {
    pub fn new(
        model: Model<'a, B>,
        forward: Box<dyn ModelForward<B> + 'a>,
        kv_pool: KvPool<B>,
        tokenizer: Tokenizer,
        backend: &'a B,
        max_active: usize,
    ) -> Self {
        let tokenizer_vocab_size = tokenizer.vocab_size();
        let dispatch = SingleGpuDispatch::new(model, forward, kv_pool, backend, max_active, tokenizer_vocab_size);
        let scheduler = Scheduler::new(max_active);
        Self {
            dispatch,
            scheduler,
            tokenizer,
        }
    }
}

impl<'a, B: GpuBackend> InferenceEngine for Engine<'a, B> {
    fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_gen_tokens: usize,
        params: SampleParams,
        images: Vec<crate::model::vision::ProcessedImage>,
        seed: Option<u64>,
        grammar: Option<std::sync::Arc<crate::model::grammar::CompiledGrammar>>,
        logit_bias: HashMap<u32, f32>,
    ) -> SeqId {
        self.scheduler.add_request(SequenceRequest {
            prompt_tokens,
            max_gen_tokens,
            params,
            images,
            seed,
            grammar,
            logit_bias,
        })
    }

    fn step(&mut self) -> anyhow::Result<StepOutput> {
        run_step(&mut self.dispatch, &mut self.scheduler, &self.tokenizer)
    }

    fn abort_sequence(&mut self, id: SeqId) {
        self.scheduler.abort_sequence(id, &mut self.dispatch);
    }

    fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn active_count(&self) -> usize {
        self.scheduler.active_count()
    }

    fn waiting_count(&self) -> usize {
        self.scheduler.waiting_count()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    use crate::gpu::cpu::CpuBackend;
    use crate::model::kv_cache::{KvPool, PrefixCache, SeqKvState};

    // -----------------------------------------------------------------------
    // TestDispatch — real CpuBackend KvPool for scheduler admission tests.
    // -----------------------------------------------------------------------

    /// Dispatch backed by a real CpuBackend KvPool.
    ///
    /// Forward passes are no-ops, but KV block allocation is real — so tests
    /// can verify admission heuristics and block freeing.
    struct TestDispatch {
        pool: KvPool<CpuBackend>,
        backend: CpuBackend,
    }

    impl TestDispatch {
        fn new(num_blocks: usize) -> Self {
            let backend = CpuBackend;
            let pool = KvPool::new(&backend, num_blocks, 4, 1, crate::model::turboquant::KvQuantMode::None, 4);
            Self { pool, backend }
        }
    }

    impl Dispatch for TestDispatch {
        type SeqState = SeqKvState<CpuBackend>;

        fn new_seq_state(&self) -> Self::SeqState {
            self.pool.new_sequence(&self.backend)
        }

        fn free_seq_state(&mut self, state: &Self::SeqState) {
            self.pool.free_sequence(state);
        }

        fn free_block_count(&self) -> usize {
            self.pool.free_block_count()
        }

        fn seq_len(state: &Self::SeqState) -> usize {
            state.seq_len
        }

        fn prepare_prefill(
            &mut self,
            state: &mut Self::SeqState,
            token_count: usize,
        ) -> anyhow::Result<()> {
            state.ensure_slots(&mut self.pool, token_count)?;
            state.sync_block_table(&self.backend);
            Ok(())
        }

        fn forward_prefill(&self, _tokens: &[u32], _state: &Self::SeqState, _images: &[crate::model::vision::ProcessedImage]) -> anyhow::Result<()> {
            Ok(())
        }

        fn finish_prefill(state: &mut Self::SeqState, token_count: usize) {
            state.advance_by(token_count);
        }

        fn prepare_decode(&mut self, state: &mut Self::SeqState) -> anyhow::Result<()> {
            state.ensure_slot(&mut self.pool)?;
            state.sync_block_table(&self.backend);
            Ok(())
        }

        fn forward_decode(&self, _token: u32, _state: &Self::SeqState) -> anyhow::Result<()> {
            Ok(())
        }

        fn finish_decode(state: &mut Self::SeqState) {
            state.advance();
        }

        fn sample(
            &self,
            _params: &SampleParams,
            _rng: &mut impl rand::Rng,
            _allowed_tokens: Option<&[u32]>,
            _token_counts: &HashMap<u32, u32>,
            _logit_bias: &HashMap<u32, f32>,
        ) -> anyhow::Result<SampleResult> {
            Ok(SampleResult { token_id: 0, logprob: 0.0, top_logprobs: Vec::new() })
        }
    }

    // -----------------------------------------------------------------------
    // MockDispatch — deterministic tokens for engine step loop tests.
    // -----------------------------------------------------------------------

    /// Minimal sequence state for testing — just a position counter.
    struct MockSeqState {
        seq_len: usize,
    }

    /// Mock dispatch that returns deterministic tokens.
    ///
    /// Each call to `sample()` returns the next value from a counter starting
    /// at `next_token`.  Tests that want EOS-triggered finishes set the EOS
    /// token to a value that the counter will hit.
    struct MockDispatch {
        free_blocks: usize,
        next_token: Cell<u32>,
        prefill_count: Cell<usize>,
        decode_count: Cell<usize>,
    }

    impl MockDispatch {
        fn new(free_blocks: usize, start_token: u32) -> Self {
            Self {
                free_blocks,
                next_token: Cell::new(start_token),
                prefill_count: Cell::new(0),
                decode_count: Cell::new(0),
            }
        }
    }

    impl Dispatch for MockDispatch {
        type SeqState = MockSeqState;

        fn new_seq_state(&self) -> MockSeqState {
            MockSeqState { seq_len: 0 }
        }

        fn free_seq_state(&mut self, _state: &MockSeqState) {
            self.free_blocks += 1;
        }

        fn free_block_count(&self) -> usize {
            self.free_blocks
        }

        fn seq_len(state: &MockSeqState) -> usize {
            state.seq_len
        }

        fn prepare_prefill(
            &mut self,
            _state: &mut MockSeqState,
            _token_count: usize,
        ) -> anyhow::Result<()> {
            self.prefill_count.set(self.prefill_count.get() + 1);
            Ok(())
        }

        fn forward_prefill(&self, _tokens: &[u32], _state: &MockSeqState, _images: &[crate::model::vision::ProcessedImage]) -> anyhow::Result<()> {
            Ok(())
        }

        fn finish_prefill(state: &mut MockSeqState, token_count: usize) {
            state.seq_len += token_count;
        }

        fn prepare_decode(&mut self, _state: &mut MockSeqState) -> anyhow::Result<()> {
            self.decode_count.set(self.decode_count.get() + 1);
            Ok(())
        }

        fn forward_decode(&self, _token: u32, _state: &MockSeqState) -> anyhow::Result<()> {
            Ok(())
        }

        fn finish_decode(state: &mut MockSeqState) {
            state.seq_len += 1;
        }

        fn sample(
            &self,
            _params: &SampleParams,
            _rng: &mut impl rand::Rng,
            _allowed_tokens: Option<&[u32]>,
            _token_counts: &HashMap<u32, u32>,
            _logit_bias: &HashMap<u32, f32>,
        ) -> anyhow::Result<SampleResult> {
            let token = self.next_token.get();
            self.next_token.set(token + 1);
            Ok(SampleResult { token_id: token, logprob: 0.0, top_logprobs: Vec::new() })
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_request(prompt_len: usize, max_gen: usize) -> SequenceRequest {
        SequenceRequest {
            prompt_tokens: vec![1; prompt_len],
            max_gen_tokens: max_gen,
            params: SampleParams::default(),
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        }
    }

    fn make_tokenizer(eos_ids: Vec<u32>) -> Tokenizer {
        Tokenizer::for_test(eos_ids)
    }

    fn setup_scheduler(
        num_blocks: usize,
        max_active: usize,
    ) -> (TestDispatch, Scheduler<SeqKvState<CpuBackend>>) {
        let dispatch = TestDispatch::new(num_blocks);
        let scheduler = Scheduler::new(max_active);
        (dispatch, scheduler)
    }

    // -----------------------------------------------------------------------
    // Scheduler tests — admission, abort, collect, block freeing.
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_request_returns_incrementing_ids() {
        let (_d, mut s) = setup_scheduler(10, 4);
        let id0 = s.add_request(make_request(4, 10));
        let id1 = s.add_request(make_request(4, 10));
        let id2 = s.add_request(make_request(4, 10));
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(s.waiting_count(), 3);
        assert!(s.active.is_empty());
    }

    #[test]
    fn test_schedule_admits_to_active() {
        let (d, mut s) = setup_scheduler(10, 4);
        s.add_request(make_request(4, 10));
        s.add_request(make_request(4, 10));

        let admitted = s.schedule(&d);
        assert_eq!(admitted.len(), 2);
        assert_eq!(s.active.len(), 2);
        assert_eq!(s.waiting_count(), 0);
    }

    #[test]
    fn test_schedule_respects_max_active() {
        let (d, mut s) = setup_scheduler(10, 2);
        s.add_request(make_request(4, 10));
        s.add_request(make_request(4, 10));
        s.add_request(make_request(4, 10));

        let admitted = s.schedule(&d);
        assert_eq!(admitted.len(), 2);
        assert_eq!(s.active.len(), 2);
        assert_eq!(s.waiting_count(), 1);
    }

    #[test]
    fn test_schedule_blocks_on_kv_memory() {
        let (d, mut s) = setup_scheduler(2, 4);
        s.add_request(make_request(48, 10)); // needs ceil(48/16) = 3 blocks > 2 available

        let admitted = s.schedule(&d);
        assert_eq!(admitted.len(), 0);
        assert_eq!(s.waiting_count(), 1);

        // FCFS: blocked first request prevents the second from being tried
        s.add_request(make_request(16, 10));
        let admitted = s.schedule(&d);
        assert_eq!(admitted.len(), 0);
    }

    #[test]
    fn test_abort_waiting_sequence() {
        let (mut d, mut s) = setup_scheduler(10, 4);
        let id0 = s.add_request(make_request(4, 10));
        let _id1 = s.add_request(make_request(4, 10));

        s.abort_sequence(id0, &mut d);
        assert_eq!(s.waiting_count(), 1);
    }

    #[test]
    fn test_abort_active_sequence_frees_blocks() {
        let (mut d, mut s) = setup_scheduler(4, 4);
        s.add_request(make_request(16, 10));
        let admitted = s.schedule(&d);
        let id = admitted[0];

        d.prepare_prefill(&mut s.active.get_mut(&id).unwrap().kv_state, 16)
            .unwrap();

        let free_before = d.free_block_count();
        s.abort_sequence(id, &mut d);
        let free_after = d.free_block_count();

        assert!(s.active.is_empty());
        assert_eq!(free_after, free_before + 1);
    }

    #[test]
    fn test_collect_finished() {
        let (mut d, mut s) = setup_scheduler(10, 4);
        let id0 = s.add_request(make_request(4, 10));
        let id1 = s.add_request(make_request(4, 10));
        s.schedule(&d);

        s.active.get_mut(&id0).unwrap().finished = true;

        let finished = s.collect_finished(&mut d);
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].0, id0);
        assert_eq!(s.active.len(), 1);
        assert!(s.active.contains_key(&id1));
    }

    #[test]
    fn test_collect_finished_frees_blocks() {
        let (mut d, mut s) = setup_scheduler(10, 4);
        s.add_request(make_request(16, 10));
        s.schedule(&d);

        for seq in s.active.values_mut() {
            d.prepare_prefill(&mut seq.kv_state, 16).unwrap();
        }

        let free_before = d.free_block_count();
        for seq in s.active.values_mut() {
            seq.finished = true;
        }
        s.collect_finished(&mut d);
        let free_after = d.free_block_count();

        assert!(s.active.is_empty());
        assert_eq!(free_after, free_before + 1);
    }

    #[test]
    fn test_has_work() {
        let (mut d, mut s) = setup_scheduler(10, 4);
        assert!(!s.has_work());

        s.add_request(make_request(4, 10));
        assert!(s.has_work());

        s.schedule(&d);
        assert!(s.has_work());

        for seq in s.active.values_mut() {
            seq.finished = true;
        }
        s.collect_finished(&mut d);
        assert!(!s.has_work());
    }

    #[test]
    fn test_schedule_admits_after_finish_frees_blocks() {
        let (mut d, mut s) = setup_scheduler(3, 4);
        s.add_request(make_request(20, 10));
        s.schedule(&d);
        assert_eq!(s.active.len(), 1);

        for seq in s.active.values_mut() {
            d.prepare_prefill(&mut seq.kv_state, 20).unwrap();
        }
        assert_eq!(d.free_block_count(), 1);

        // Second request needs 2 blocks but only 1 free
        s.add_request(make_request(20, 10));
        let admitted = s.schedule(&d);
        assert_eq!(admitted.len(), 0);
        assert_eq!(s.waiting_count(), 1);

        // Finish first → frees 2 blocks → second can be admitted
        for seq in s.active.values_mut() {
            seq.finished = true;
        }
        s.collect_finished(&mut d);
        assert_eq!(d.free_block_count(), 3);

        let admitted = s.schedule(&d);
        assert_eq!(admitted.len(), 1);
        assert_eq!(s.active.len(), 1);
        assert_eq!(s.waiting_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Engine step tests — verify the shared run_step() loop.
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_prefill_then_decode() {
        // One sequence: 4 prompt tokens, max 3 generated tokens.
        //
        // After prefill drains the prompt and samples the first token, the
        // sequence's pending_prefill is empty — so it immediately enters the
        // decode phase in the SAME step.  This is the expected behavior:
        //
        // Step 1: admit + prefill (samples 100) + decode (samples 101) = 2 tokens
        // Step 2: decode (samples 102, hits max_tokens=3) → finishes
        let mut dispatch = MockDispatch::new(10, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2, 3, 4],
            max_gen_tokens: 3,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        // Step 1: admit + prefill + decode (2 tokens in one step)
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.tokens.len(), 2);
        assert_eq!(out.tokens[0].1, 100);
        assert_eq!(out.tokens[1].1, 101);
        assert!(out.finished.is_empty());
        assert_eq!(dispatch.prefill_count.get(), 1);
        assert_eq!(dispatch.decode_count.get(), 1);

        // Step 2: decode → hits max_tokens (3), finishes
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.tokens.len(), 1);
        assert_eq!(out.tokens[0].1, 102);
        assert_eq!(out.finished.len(), 1);
        assert_eq!(out.finished[0].tokens, vec![100, 101, 102]);
        assert!(matches!(out.finished[0].reason, FinishReason::MaxTokens));
    }

    #[test]
    fn test_step_eos_stops_generation() {
        // EOS token = 101.  Sequence generates: 100 (prefill), 101 (decode, EOS).
        // Both happen in step 1 because decode runs after prefill in the same step.
        let mut dispatch = MockDispatch::new(10, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![101]);

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2],
            max_gen_tokens: 10,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.tokens.len(), 2);
        assert_eq!(out.tokens[0].1, 100);
        assert_eq!(out.tokens[1].1, 101);
        assert_eq!(out.finished.len(), 1);
        assert!(matches!(out.finished[0].reason, FinishReason::Eos));
    }

    #[test]
    fn test_step_eos_during_prefill() {
        // If the first sampled token (from prefill) is EOS, the sequence
        // should finish immediately (no decode phase).
        let mut dispatch = MockDispatch::new(10, 999);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2, 3],
            max_gen_tokens: 10,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.tokens.len(), 1);
        assert_eq!(out.tokens[0].1, 999);
        assert_eq!(out.finished.len(), 1);
        assert!(matches!(out.finished[0].reason, FinishReason::Eos));
    }

    #[test]
    fn test_step_multiple_concurrent_sequences() {
        // Two sequences, both with short prompts and max 2 tokens.
        // Step 1: both prefill + decode → each gets 2 tokens → both finish.
        let mut dispatch = MockDispatch::new(10, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2],
            max_gen_tokens: 2,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });
        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![3, 4],
            max_gen_tokens: 2,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(dispatch.prefill_count.get(), 2);
        assert_eq!(out.tokens.len(), 4); // 2 prefill + 2 decode
        assert_eq!(out.finished.len(), 2);
    }

    #[test]
    fn test_step_admission_blocked_by_kv_memory() {
        let mut dispatch = MockDispatch::new(0, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2],
            max_gen_tokens: 5,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert!(out.tokens.is_empty());
        assert!(out.finished.is_empty());
        assert!(scheduler.has_work());
    }

    #[test]
    fn test_step_abort_during_decode() {
        let mut dispatch = MockDispatch::new(10, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        let id = scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2],
            max_gen_tokens: 10,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert!(scheduler.has_work());

        scheduler.abort_sequence(id, &mut dispatch);
        assert!(!scheduler.has_work());

        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert!(out.tokens.is_empty());
        assert!(out.finished.is_empty());
    }

    #[test]
    fn test_step_no_work() {
        let mut dispatch = MockDispatch::new(10, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert!(out.tokens.is_empty());
        assert!(out.finished.is_empty());
    }

    #[test]
    fn test_step_max_tokens_one() {
        // max_gen_tokens = 1 → finishes immediately after prefill sample.
        let mut dispatch = MockDispatch::new(10, 42);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1],
            max_gen_tokens: 1,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.tokens.len(), 1);
        assert_eq!(out.tokens[0].1, 42);
        assert_eq!(out.finished.len(), 1);
        assert!(matches!(out.finished[0].reason, FinishReason::MaxTokens));
        assert_eq!(out.finished[0].tokens, vec![42]);
    }

    #[test]
    fn test_step_continuous_batching_slot_reuse() {
        // Sequence A finishes, freeing its slot.  Sequence B (waiting) should
        // then be admitted in the next step.
        let mut dispatch = MockDispatch::new(10, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(1);
        let tokenizer = make_tokenizer(vec![999]);

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1],
            max_gen_tokens: 1,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });
        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![2],
            max_gen_tokens: 1,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        // Step 1: A admitted + prefilled + finished (max_tokens=1)
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.finished.len(), 1);
        assert_eq!(out.finished[0].tokens, vec![100]);
        assert!(scheduler.has_work());

        // Step 2: B admitted + prefilled + finished
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.finished.len(), 1);
        assert_eq!(out.finished[0].tokens, vec![101]);
        assert!(!scheduler.has_work());
    }

    // -----------------------------------------------------------------------
    // Prefix caching integration tests — verify run_step with a real KvPool
    // and PrefixCache, exercising the full lookup → link → suffix-prefill →
    // register → reuse flow.
    // -----------------------------------------------------------------------

    /// Dispatch backed by a real CpuBackend KvPool + PrefixCache.
    ///
    /// Forward passes are no-ops, but block allocation, prefix caching, and
    /// block freeing are all real.  This tests the full lifecycle from the
    /// engine's perspective.
    struct CachingTestDispatch {
        pool: KvPool<CpuBackend>,
        backend: CpuBackend,
        prefix_cache: PrefixCache,
        /// Tokens passed to forward_prefill (accumulated across calls).
        /// Used to verify that cache hits skip prefix tokens.
        prefilled_tokens: std::cell::RefCell<Vec<Vec<u32>>>,
    }

    impl CachingTestDispatch {
        fn new(num_blocks: usize, cache_entries: usize) -> Self {
            let backend = CpuBackend;
            let pool = KvPool::new(&backend, num_blocks, 4, 1, crate::model::turboquant::KvQuantMode::None, 4);
            Self {
                pool,
                backend,
                prefix_cache: PrefixCache::new(cache_entries),
                prefilled_tokens: std::cell::RefCell::new(Vec::new()),
            }
        }
    }

    impl Dispatch for CachingTestDispatch {
        type SeqState = SeqKvState<CpuBackend>;

        fn new_seq_state(&self) -> Self::SeqState {
            self.pool.new_sequence(&self.backend)
        }

        fn free_seq_state(&mut self, state: &Self::SeqState) {
            if let Some(prefix_tokens) = state.shared_prefix_tokens() {
                self.prefix_cache.release(prefix_tokens);
            }
            self.pool.free_sequence(state);
        }

        fn free_block_count(&self) -> usize {
            self.pool.free_block_count()
        }

        fn seq_len(state: &Self::SeqState) -> usize {
            state.seq_len
        }

        fn prepare_prefill(
            &mut self,
            state: &mut Self::SeqState,
            token_count: usize,
        ) -> anyhow::Result<()> {
            state.ensure_slots(&mut self.pool, token_count)?;
            state.sync_block_table(&self.backend);
            Ok(())
        }

        fn forward_prefill(&self, tokens: &[u32], _state: &Self::SeqState, _images: &[crate::model::vision::ProcessedImage]) -> anyhow::Result<()> {
            // Record what was actually prefilled (for cache-hit verification).
            self.prefilled_tokens.borrow_mut().push(tokens.to_vec());
            Ok(())
        }

        fn finish_prefill(state: &mut Self::SeqState, token_count: usize) {
            state.advance_by(token_count);
        }

        fn prepare_decode(&mut self, state: &mut Self::SeqState) -> anyhow::Result<()> {
            state.ensure_slot(&mut self.pool)?;
            state.sync_block_table(&self.backend);
            Ok(())
        }

        fn forward_decode(&self, _token: u32, _state: &Self::SeqState) -> anyhow::Result<()> {
            Ok(())
        }

        fn finish_decode(state: &mut Self::SeqState) {
            state.advance();
        }

        fn sample(
            &self,
            _params: &SampleParams,
            _rng: &mut impl rand::Rng,
            _allowed_tokens: Option<&[u32]>,
            _token_counts: &HashMap<u32, u32>,
            _logit_bias: &HashMap<u32, f32>,
        ) -> anyhow::Result<SampleResult> {
            Ok(SampleResult { token_id: 0, logprob: 0.0, top_logprobs: Vec::new() })
        }

        fn prefix_cache_lookup(&mut self, prompt_tokens: &[u32]) -> Option<(Vec<BlockHandle>, usize)> {
            self.prefix_cache.lookup(prompt_tokens)
        }

        fn prefix_cache_register(&mut self, tokens: &[u32], state: &mut Self::SeqState) {
            let prefix_blocks = tokens.len() / BLOCK_SIZE;
            if prefix_blocks == 0 {
                return;
            }
            let prefix_len = prefix_blocks * BLOCK_SIZE;
            let prefix_tokens = tokens[..prefix_len].to_vec();
            let block_indices = state.block_table_cpu_slice()[..prefix_blocks].to_vec();
            if let Some(evicted) = self.prefix_cache.insert(prefix_tokens.clone(), block_indices) {
                self.pool.free_blocks(&evicted);
            }
            state.mark_prefix_shared(prefix_blocks, prefix_tokens);
        }

        fn link_prefix(
            &self,
            state: &mut Self::SeqState,
            prefix_handles: &[BlockHandle],
            prefix_token_count: usize,
            prefix_tokens: Vec<u32>,
        ) {
            state.link_prefix(prefix_handles, prefix_token_count, prefix_tokens);
        }
    }

    #[test]
    fn test_prefix_cache_miss_registers_then_hit_skips_prefill() {
        // Two sequences with the same 32-token prefix + different suffixes.
        // First sequence: 40 tokens (not block-aligned: 2.5 blocks).
        //   → Registers 32-token block-aligned prefix (2 full blocks).
        // Second sequence: same 32-token prefix + different suffix.
        //   → Cache hit: 32 tokens skipped, only suffix prefilled.
        let mut dispatch = CachingTestDispatch::new(20, 4);
        let mut scheduler: Scheduler<SeqKvState<CpuBackend>> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        // Sequence 1: 40 tokens (2 full blocks + 8 extra in partial block).
        let prompt1: Vec<u32> = (0..40).collect();
        scheduler.add_request(SequenceRequest {
            prompt_tokens: prompt1.clone(),
            max_gen_tokens: 1,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.finished.len(), 1);
        assert_eq!(out.finished[0].cached_tokens, 0); // miss

        // Verify full prompt was prefilled.
        let prefilled = dispatch.prefilled_tokens.borrow().clone();
        assert_eq!(prefilled.len(), 1);
        assert_eq!(prefilled[0].len(), 40);

        // Cache should now have the 32-token block-aligned prefix (2 blocks).
        assert_eq!(dispatch.prefix_cache.len(), 1);

        // Sequence 2: same 32-token prefix + different 8-token suffix.
        dispatch.prefilled_tokens.borrow_mut().clear();
        let mut prompt2: Vec<u32> = (0..32).collect(); // same prefix
        prompt2.extend(100..108); // different suffix (8 tokens)
        scheduler.add_request(SequenceRequest {
            prompt_tokens: prompt2.clone(),
            max_gen_tokens: 1,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.finished.len(), 1);
        assert_eq!(out.finished[0].cached_tokens, 32); // hit: 32 tokens cached

        // Verify only the 8-token suffix was prefilled (not the full 40).
        let prefilled = dispatch.prefilled_tokens.borrow().clone();
        assert_eq!(prefilled.len(), 1);
        assert_eq!(prefilled[0].len(), 8);
        assert_eq!(prefilled[0], (100..108).collect::<Vec<u32>>());
    }

    #[test]
    fn test_prefix_cache_blocks_freed_correctly_after_both_sequences_finish() {
        // Two sequences share a prefix.  After both finish, the shared blocks
        // should remain held by the cache (not double-freed).
        //
        // Prompt: 40 tokens = 2 full blocks (cached) + 8 in partial block (not cached).
        let mut dispatch = CachingTestDispatch::new(20, 4);
        let mut scheduler: Scheduler<SeqKvState<CpuBackend>> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        let prompt: Vec<u32> = (0..40).collect(); // 2 full blocks + 8 extra
        let blocks_before = dispatch.pool.free_block_count();

        // Sequence 1: prefill, register 32-token prefix, generate 1 token, finish.
        scheduler.add_request(SequenceRequest {
            prompt_tokens: prompt.clone(),
            max_gen_tokens: 1,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.finished.len(), 1);
        let blocks_after_seq1 = dispatch.pool.free_block_count();
        // 3 blocks allocated for 40 tokens (ceil(40/16) = 3).
        // 2 prefix blocks held by cache.
        // 1 suffix block (partial, positions 32-39) freed on finish.
        // Plus 1 decode block allocated then freed (sample needs pos 40).
        // Net: 2 blocks held by cache.
        assert_eq!(
            blocks_before - blocks_after_seq1,
            2, // 2 prefix blocks held by cache
            "prefix blocks should be held by cache"
        );

        // Sequence 2: same first 32 tokens + different suffix, cache hit.
        let mut prompt2: Vec<u32> = (0..32).collect();
        prompt2.extend(500..508); // different 8-token suffix
        scheduler.add_request(SequenceRequest {
            prompt_tokens: prompt2.clone(),
            max_gen_tokens: 1,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.finished.len(), 1);
        assert_eq!(out.finished[0].cached_tokens, 32);

        let blocks_after_seq2 = dispatch.pool.free_block_count();
        // Seq2 used cached prefix (no new prefix blocks).
        // Allocated 0 suffix blocks (8 tokens fit in the last partial area
        // of block 1, which is shared — wait, no.  Suffix tokens need their
        // own block.  seq_len=32 after link, ensure_slots(8) needs
        // ceil((32+8)/16) = 3 blocks total.  Already have 2 from prefix,
        // so allocates 1 new block.  That 1 block is freed on finish.
        // Cache still holds 2.
        assert_eq!(
            blocks_before - blocks_after_seq2,
            2,
            "prefix blocks still held by cache after both sequences finish"
        );

        // Verify cache ref count is 0 now (both sequences released).
        // Insert enough entries to trigger eviction of our prefix.
        let evict1: Vec<u32> = vec![200; BLOCK_SIZE];
        let evict2: Vec<u32> = vec![201; BLOCK_SIZE];
        let evict3: Vec<u32> = vec![202; BLOCK_SIZE];
        let evict4: Vec<u32> = vec![203; BLOCK_SIZE];
        let bh = |i, g| BlockHandle { index: i, generation: g };
        dispatch.prefix_cache.insert(evict1.clone(), vec![bh(99, 0)]);
        dispatch.prefix_cache.release(&evict1);
        dispatch.prefix_cache.insert(evict2.clone(), vec![bh(98, 0)]);
        dispatch.prefix_cache.release(&evict2);
        dispatch.prefix_cache.insert(evict3.clone(), vec![bh(97, 0)]);
        dispatch.prefix_cache.release(&evict3);
        // This insert should evict our original prefix (oldest, ref_count=0).
        let evicted = dispatch.prefix_cache.insert(evict4.clone(), vec![bh(96, 0)]);
        assert!(evicted.is_some(), "original prefix should be evictable (ref_count=0)");
    }

    #[test]
    fn test_prefix_cache_short_prompt_no_cache() {
        // Prompts shorter than BLOCK_SIZE should not be cached (can't be block-aligned).
        let mut dispatch = CachingTestDispatch::new(10, 4);
        let mut scheduler: Scheduler<SeqKvState<CpuBackend>> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2, 3],
            max_gen_tokens: 1,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(dispatch.prefix_cache.len(), 0);
    }

    #[test]
    fn test_prefix_cache_seq_len_correct_after_link() {
        // After a cache hit, seq_len should equal prefix_len, and suffix prefill
        // should advance it further.
        let mut dispatch = CachingTestDispatch::new(30, 4);
        let mut scheduler: Scheduler<SeqKvState<CpuBackend>> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        // Seq 1: 40 tokens (2 full blocks cached + 8 in partial).
        let prompt1: Vec<u32> = (0..40).collect();
        scheduler.add_request(SequenceRequest {
            prompt_tokens: prompt1.clone(),
            max_gen_tokens: 1,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();

        // Seq 2: same 32-token prefix + 24-token different suffix = 56 tokens.
        let mut prompt2: Vec<u32> = (0..32).collect();
        prompt2.extend(1000..1024);
        scheduler.add_request(SequenceRequest {
            prompt_tokens: prompt2.clone(),
            max_gen_tokens: 2,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        // Step: admit + cache hit (32 tokens) + prefill suffix (24 tokens)
        //       + sample first token + decode + sample second → finish (max_tokens=2).
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.finished.len(), 1);
        assert_eq!(out.finished[0].cached_tokens, 32);

        // Verify the suffix was prefilled correctly (24 tokens, not the full 56).
        let prefilled = dispatch.prefilled_tokens.borrow();
        // Last prefill call should be the 24-token suffix.
        let last = prefilled.last().unwrap();
        assert_eq!(last.len(), 24);
        assert_eq!(*last, (1000..1024).collect::<Vec<u32>>());
    }

    // -----------------------------------------------------------------------
    // Preemption tests — evict active sequences to free KV blocks.
    // -----------------------------------------------------------------------

    #[test]
    fn test_preempt_evicts_longest_sequence() {
        // Pool with 4 blocks, max 4 active.  Admit two sequences that each
        // use 2 blocks (32-token prompts).  Pool is now full.  A third
        // request should trigger preemption of the sequence with more
        // generated tokens.
        let mut dispatch = MockDispatch::new(4, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        // Seq A: 16-token prompt (1 block), max_gen=10
        let id_a = scheduler.add_request(make_request(16, 10));
        // Seq B: 16-token prompt (1 block), max_gen=10
        let id_b = scheduler.add_request(make_request(16, 10));

        // Step 1: admit both, prefill both, decode both (each generates 1 token).
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(scheduler.active.len(), 2);

        // Step 2: decode again — each generates another token.
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();

        // Both sequences now have 2 generated tokens each.
        // Exhaust free blocks to simulate memory pressure.
        dispatch.free_blocks = 0;

        // Submit a new request that needs 1 block.
        scheduler.add_request(make_request(16, 5));

        // Step 3: schedule can't admit (0 free blocks), preemption kicks in.
        // Should evict one of the active sequences (both have same generated
        // count, so either is valid).
        let active_before = scheduler.active.len();
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();

        // The evicted sequence was re-queued, and the new request or the
        // re-queued request may have been admitted.  The key invariant:
        // preemption happened (we didn't deadlock).
        assert!(
            scheduler.active.len() >= 1,
            "at least one sequence should be active after preemption"
        );
        // The waiting queue should have the evicted sequence if it wasn't
        // re-admitted yet, or be empty if everything fit.
        assert!(
            scheduler.has_work(),
            "engine should still have work after preemption"
        );
    }

    #[test]
    fn test_preempt_does_not_evict_single_active() {
        // With only 1 active sequence and no free blocks, preemption should
        // NOT fire — evicting the only sequence would just re-admit it into
        // the same deadlock.
        let mut dispatch = MockDispatch::new(0, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);

        // Manually admit one sequence.
        dispatch.free_blocks = 2;
        scheduler.add_request(make_request(16, 10));
        scheduler.schedule(&dispatch);
        assert_eq!(scheduler.active.len(), 1);

        // Exhaust blocks and add a waiting request.
        dispatch.free_blocks = 0;
        scheduler.add_request(make_request(16, 5));

        // try_preempt should return None (only 1 active sequence).
        let result = scheduler.try_preempt(&mut dispatch);
        assert!(result.is_none(), "should not evict the only active sequence");
        assert_eq!(scheduler.active.len(), 1);
    }

    #[test]
    fn test_preempt_requeues_at_front() {
        // After preemption, the evicted sequence should be at the front of
        // the waiting queue (it already waited and did work).
        let mut dispatch = MockDispatch::new(4, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        // Admit two sequences.
        scheduler.add_request(make_request(16, 10));
        scheduler.add_request(make_request(16, 10));
        scheduler.schedule(&dispatch);
        assert_eq!(scheduler.active.len(), 2);

        // Generate some tokens so sequences have generated_tokens.
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();

        // Exhaust blocks, add a new waiting request.
        dispatch.free_blocks = 0;
        let new_id = scheduler.add_request(make_request(16, 5));
        assert_eq!(scheduler.waiting_count(), 1);

        // Preempt — the evicted sequence should go to front of waiting queue,
        // BEFORE the new request.
        let evicted = scheduler.try_preempt(&mut dispatch);
        assert!(evicted.is_some());
        assert_eq!(scheduler.waiting_count(), 2);

        // Front of waiting queue should be the re-queued sequence (higher ID
        // than the new request because it gets a fresh ID).
        let front_id = scheduler.waiting.front().unwrap().0;
        assert_ne!(front_id, new_id, "evicted sequence should be at front, not the new request");
    }

    #[test]
    fn test_preempt_preserves_block_accounting() {
        // After preemption + re-admission, block accounting should be consistent.
        // Use the real KvPool-backed TestDispatch for this test.
        let mut dispatch = TestDispatch::new(6); // 6 blocks
        let mut scheduler: Scheduler<SeqKvState<CpuBackend>> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        let blocks_total = dispatch.pool.free_block_count();

        // Admit seq A: 32 tokens = 2 blocks for prefill.
        scheduler.add_request(make_request(32, 5));
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        let blocks_after_a = dispatch.pool.free_block_count();
        assert!(blocks_after_a < blocks_total, "seq A should have consumed blocks");

        // Admit seq B: 16 tokens = 1 block for prefill.
        scheduler.add_request(make_request(16, 5));
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();

        // Record blocks used.
        let blocks_used = blocks_total - dispatch.pool.free_block_count();

        // Add a request that can't fit.
        scheduler.add_request(make_request(48, 5)); // needs 3 blocks

        // Preempt — should free some blocks.
        let evicted = scheduler.try_preempt(&mut dispatch);
        assert!(evicted.is_some(), "should have evicted a sequence");

        let blocks_after_preempt = dispatch.pool.free_block_count();
        assert!(
            blocks_after_preempt > blocks_total - blocks_used,
            "preemption should have freed blocks"
        );
    }

    // -----------------------------------------------------------------------
    // Chunked prefill interleaving tests — verify that long prefills don't
    // starve decoding sequences.
    // -----------------------------------------------------------------------

    /// Mock dispatch with a configurable max_prefill_chunk.
    ///
    /// Used to test chunked prefill interleaving: when a sequence has a prompt
    /// longer than `chunk_size`, each step should prefill at most one chunk,
    /// then yield to let decoding sequences make progress.
    struct ChunkedMockDispatch {
        free_blocks: usize,
        next_token: Cell<u32>,
        prefill_count: Cell<usize>,
        decode_count: Cell<usize>,
        chunk_size: usize,
    }

    impl ChunkedMockDispatch {
        fn new(free_blocks: usize, start_token: u32, chunk_size: usize) -> Self {
            Self {
                free_blocks,
                next_token: Cell::new(start_token),
                prefill_count: Cell::new(0),
                decode_count: Cell::new(0),
                chunk_size,
            }
        }
    }

    impl Dispatch for ChunkedMockDispatch {
        type SeqState = MockSeqState;

        fn new_seq_state(&self) -> MockSeqState {
            MockSeqState { seq_len: 0 }
        }

        fn free_seq_state(&mut self, _state: &MockSeqState) {
            self.free_blocks += 1;
        }

        fn free_block_count(&self) -> usize {
            self.free_blocks
        }

        fn max_prefill_chunk(&self) -> usize {
            self.chunk_size
        }

        fn seq_len(state: &MockSeqState) -> usize {
            state.seq_len
        }

        fn prepare_prefill(
            &mut self,
            _state: &mut MockSeqState,
            _token_count: usize,
        ) -> anyhow::Result<()> {
            self.prefill_count.set(self.prefill_count.get() + 1);
            Ok(())
        }

        fn forward_prefill(&self, _tokens: &[u32], _state: &MockSeqState, _images: &[crate::model::vision::ProcessedImage]) -> anyhow::Result<()> {
            Ok(())
        }

        fn finish_prefill(state: &mut MockSeqState, token_count: usize) {
            state.seq_len += token_count;
        }

        fn prepare_decode(&mut self, _state: &mut MockSeqState) -> anyhow::Result<()> {
            self.decode_count.set(self.decode_count.get() + 1);
            Ok(())
        }

        fn forward_decode(&self, _token: u32, _state: &MockSeqState) -> anyhow::Result<()> {
            Ok(())
        }

        fn finish_decode(state: &mut MockSeqState) {
            state.seq_len += 1;
        }

        fn sample(
            &self,
            _params: &SampleParams,
            _rng: &mut impl rand::Rng,
            _allowed_tokens: Option<&[u32]>,
            _token_counts: &HashMap<u32, u32>,
            _logit_bias: &HashMap<u32, f32>,
        ) -> anyhow::Result<SampleResult> {
            let token = self.next_token.get();
            self.next_token.set(token + 1);
            Ok(SampleResult { token_id: token, logprob: 0.0, top_logprobs: Vec::new() })
        }
    }

    #[test]
    fn test_chunked_prefill_does_not_starve_decoding_sequences() {
        // Scenario: Sequence A has a short prompt (4 tokens) and is already
        // decoding.  Sequence B arrives with a long prompt (20 tokens) and
        // max_prefill_chunk=8.  Without interleaving, B's prefill would run
        // 3 chunks (8+8+4) in one step, blocking A from decoding.
        //
        // With interleaving, each step should:
        //   - Prefill at most ONE chunk for B
        //   - Decode A (producing a token)
        //
        // So A should get a decode token every step, even while B is still
        // prefilling.
        let chunk_size = 8;
        let mut dispatch = ChunkedMockDispatch::new(100, 100, chunk_size);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        // Sequence A: short prompt, will enter decode quickly.
        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2, 3, 4],
            max_gen_tokens: 10,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        // Step 1: Admit A, prefill A (4 tokens, fits in one chunk), decode A.
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert!(out.tokens.iter().any(|(_, _, _)| true), "A should produce tokens");

        // Now add B with a long prompt that requires multiple chunks.
        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![10; 20], // 20 tokens → needs ceil(20/8) = 3 chunks
            max_gen_tokens: 5,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        let prefill_before = dispatch.prefill_count.get();
        let decode_before = dispatch.decode_count.get();

        // Step 2: Should admit B, prefill ONE chunk of B (8 tokens), AND
        // decode A.  B should NOT fully prefill in this step.
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();

        let prefill_calls = dispatch.prefill_count.get() - prefill_before;
        let decode_calls = dispatch.decode_count.get() - decode_before;

        // Key assertion: only ONE prefill chunk should have been processed,
        // not all 3 chunks.  This ensures the long prefill is interleaved
        // with decode steps.
        assert_eq!(
            prefill_calls, 1,
            "should prefill exactly one chunk per step, got {prefill_calls}"
        );

        // A should have decoded (got a token) — not starved by B's prefill.
        assert!(
            decode_calls >= 1,
            "decoding sequence A should get a decode step, got {decode_calls} decode calls"
        );

        // B should still have pending prefill tokens (12 remaining).
        let b_still_prefilling = scheduler.active.values().any(|seq| !seq.pending_prefill.is_empty());
        assert!(
            b_still_prefilling,
            "sequence B should still have pending prefill tokens after one chunk"
        );

        // Step 3: Another chunk of B (8 tokens) + decode A.
        let prefill_before = dispatch.prefill_count.get();
        let decode_before = dispatch.decode_count.get();
        let _out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        let prefill_calls = dispatch.prefill_count.get() - prefill_before;
        let decode_calls = dispatch.decode_count.get() - decode_before;

        assert_eq!(
            prefill_calls, 1,
            "step 3: should prefill exactly one chunk, got {prefill_calls}"
        );
        assert!(
            decode_calls >= 1,
            "step 3: decoding sequence A should still get a decode step"
        );

        // Step 4: Final chunk of B (4 remaining tokens) + sample B's first
        // token + decode A.
        let prefill_before = dispatch.prefill_count.get();
        let _out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        let prefill_calls = dispatch.prefill_count.get() - prefill_before;

        assert_eq!(
            prefill_calls, 1,
            "step 4: should prefill the final chunk, got {prefill_calls}"
        );

        // B should now be fully prefilled — no more pending tokens.
        let any_prefilling = scheduler.active.values().any(|seq| !seq.pending_prefill.is_empty());
        assert!(
            !any_prefilling,
            "all sequences should be done prefilling after 3 chunks"
        );
    }

    #[test]
    fn test_chunked_prefill_yields_partial_pending() {
        // Verify that pending_prefill is only partially drained per step
        // when the prompt is longer than max_prefill_chunk.
        let chunk_size = 8;
        let mut dispatch = ChunkedMockDispatch::new(100, 100, chunk_size);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        // 24-token prompt → 3 chunks of 8.
        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1; 24],
            max_gen_tokens: 5,
            params: SampleParams { temperature: 0.0, ..SampleParams::default() },
            images: Vec::new(),
            seed: None,
            grammar: None,
            logit_bias: HashMap::new(),
        });

        // Step 1: admit + prefill chunk 1 (8 tokens).
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        // Should have 16 tokens remaining.
        let remaining: usize = scheduler
            .active
            .values()
            .map(|s| s.pending_prefill.len())
            .sum();
        assert_eq!(
            remaining, 16,
            "should have 16 tokens remaining after first chunk, got {remaining}"
        );

        // No token should have been sampled yet (prefill not complete).
        let generated: usize = scheduler
            .active
            .values()
            .map(|s| s.generated_tokens.len())
            .sum();
        assert_eq!(
            generated, 0,
            "should not sample until prefill is complete, got {generated} generated tokens"
        );

        // Step 2: prefill chunk 2 (8 tokens), 8 remaining.
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        let remaining: usize = scheduler
            .active
            .values()
            .map(|s| s.pending_prefill.len())
            .sum();
        assert_eq!(remaining, 8, "should have 8 tokens remaining after second chunk");

        // Step 3: prefill chunk 3 (final 8 tokens) + sample first token.
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        let remaining: usize = scheduler
            .active
            .values()
            .map(|s| s.pending_prefill.len())
            .sum();
        assert_eq!(remaining, 0, "all tokens should be prefilled after third chunk");

        let generated: usize = scheduler
            .active
            .values()
            .map(|s| s.generated_tokens.len())
            .sum();
        assert!(generated >= 1, "should have sampled at least one token after prefill completes");
    }
}
