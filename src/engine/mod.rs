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
//   to forward_prefill_paged().  This uses GEMM (mat-mat) instead of mat-vec
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

use self::dispatch::Dispatch;
use crate::gpu::GpuBackend;
use crate::model::kv_cache::{self, BlockHandle, KvPool, PrefixCache, SeqKvState, BLOCK_SIZE};
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
    fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_gen_tokens: usize,
        temperature: f32,
        top_p: f32,
        images: Vec<crate::model::vision::ProcessedImage>,
    ) -> SeqId;

    /// Run one engine step: admit → prefill → decode → sample → collect.
    fn step(&mut self) -> anyhow::Result<StepOutput>;

    /// Abort a sequence, freeing its KV cache blocks.
    fn abort_sequence(&mut self, id: SeqId);

    /// Whether the engine has pending or active work.
    fn has_work(&self) -> bool;

    /// Access the tokenizer (for incremental text decoding in the worker loop).
    fn tokenizer(&self) -> &Tokenizer;
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

/// Output from a single engine step.
pub(crate) struct StepOutput {
    /// Token generated this step for each active sequence (id, token_id).
    pub tokens: Vec<(SeqId, u32)>,
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
    pub temperature: f32,
    pub top_p: f32,
    /// Preprocessed images for vision models (consumed during first prefill).
    pub images: Vec<crate::model::vision::ProcessedImage>,
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
    /// Sampling temperature for this sequence.
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold for this sequence.
    pub top_p: f32,
    /// Whether this sequence has finished (EOS or max_tokens reached).
    pub finished: bool,
    /// Number of prompt tokens served from the prefix cache.
    pub cached_tokens: usize,
    /// Preprocessed images (consumed during first prefill chunk, then empty).
    pub images: Vec<crate::model::vision::ProcessedImage>,
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
            let seq = Sequence {
                pending_prefill: req.prompt_tokens.into(),
                kv_state: dispatch.new_seq_state(),
                generated_tokens: Vec::new(),
                max_gen_tokens: req.max_gen_tokens,
                temperature: req.temperature,
                top_p: req.top_p,
                finished: false,
                cached_tokens: 0,
                images: req.images,
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

    /// Number of waiting requests.
    #[cfg(test)]
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
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
    let mut step_tokens: Vec<(SeqId, u32)> = Vec::new();

    // 1. Admit waiting requests.
    scheduler.schedule(dispatch);

    // 2. Batched prefill: drain all pending tokens for each prefilling sequence.
    //    Each sequence's prompt is processed via GEMM (mat-mat).  Long prompts
    //    are chunked to fit within the prefill buffer allocation (max_chunk).
    //
    //    Prefix caching: before running prefill, check if the prompt's prefix
    //    is already cached.  If so, link the cached blocks and only prefill
    //    the suffix.  After prefill, register the prefix for future reuse.
    let prefilling_ids: Vec<SeqId> = scheduler
        .active
        .iter()
        .filter(|(_, seq)| !seq.pending_prefill.is_empty() && !seq.finished)
        .map(|(&id, _)| id)
        .collect();

    let max_chunk = dispatch.max_prefill_chunk();

    for id in prefilling_ids {
        let seq = scheduler.active.get_mut(&id).unwrap();
        let tokens: Vec<u32> = seq.pending_prefill.drain(..).collect();

        // --- Prefix cache lookup ---
        // Check if a prefix of this prompt is already cached.  If so,
        // link the cached blocks (skip prefill for those tokens) and
        // only run the model on the remaining suffix.
        let cached_prefix_len = if let Some((prefix_blocks, prefix_token_count)) =
            dispatch.prefix_cache_lookup(&tokens)
        {
            let prefix_len = prefix_token_count;
            let prefix_block_count = prefix_blocks.len();

            // Block-aligned prefix tokens for cache release on free.
            let prefix_tokens = tokens[..prefix_len].to_vec();

            dispatch.link_prefix(
                &mut seq.kv_state,
                &prefix_blocks,
                prefix_token_count,
                prefix_tokens,
            );
            seq.cached_tokens = prefix_len;

            eprintln!(
                "  seq {:>3}  |  prefix cache hit: {} tokens ({} blocks), prefilling {} suffix tokens",
                id, prefix_len, prefix_block_count, tokens.len() - prefix_len,
            );

            prefix_len
        } else {
            0
        };

        let suffix_tokens = &tokens[cached_prefix_len..];

        // Chunk the suffix (or full prompt) and run prefill.
        // Images are passed only for the FIRST chunk (vision encoding happens
        // once; subsequent chunks are pure text continuation).
        if !suffix_tokens.is_empty() {
            let mut first_chunk = true;
            for chunk in suffix_tokens.chunks(max_chunk) {
                let chunk_size = chunk.len();
                dispatch.prepare_prefill(&mut seq.kv_state, chunk_size)?;
                let images = if first_chunk {
                    first_chunk = false;
                    std::mem::take(&mut seq.images)
                } else {
                    Vec::new()
                };
                dispatch.forward_prefill(chunk, &seq.kv_state, &images)?;
                D::finish_prefill(&mut seq.kv_state, chunk_size);
            }
        }

        // --- Prefix cache registration ---
        // After prefill, register the block-aligned prefix for future reuse.
        // Only register if we didn't already use a cached prefix (avoid
        // re-registering a subset of an existing entry).
        if cached_prefix_len == 0 {
            dispatch.prefix_cache_register(&tokens, &mut seq.kv_state);
        }

        sample_and_finish(dispatch, seq, id, tokenizer, &mut step_tokens, &mut rng)?;
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
        let mut temperatures = Vec::with_capacity(decoding_ids.len());
        let mut top_ps = Vec::with_capacity(decoding_ids.len());
        for &id in &decoding_ids {
            let seq = scheduler.active.get_mut(&id).unwrap();
            tokens.push(*seq.generated_tokens.last().unwrap());
            temperatures.push(seq.temperature);
            top_ps.push(seq.top_p);
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

        let sampled = dispatch.sample_batch(&temperatures, &top_ps, &mut rng)?;
        for (i, &id) in decoding_ids.iter().enumerate() {
            let seq = scheduler.active.get_mut(&id).unwrap();
            seq.generated_tokens.push(sampled[i]);
            step_tokens.push((id, sampled[i]));
            if tokenizer.is_eos(sampled[i]) || seq.generated_tokens.len() >= seq.max_gen_tokens {
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
fn sample_and_finish<D: Dispatch>(
    dispatch: &D,
    seq: &mut Sequence<D::SeqState>,
    id: SeqId,
    tokenizer: &Tokenizer,
    step_tokens: &mut Vec<(SeqId, u32)>,
    rng: &mut impl rand::Rng,
) -> anyhow::Result<()> {
    let next_token = dispatch.sample(seq.temperature, seq.top_p, rng)?;
    seq.generated_tokens.push(next_token);
    step_tokens.push((id, next_token));

    if tokenizer.is_eos(next_token) || seq.generated_tokens.len() >= seq.max_gen_tokens {
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
    pub kv_pool: KvPool<B>,
    prefill_bufs: PrefillBuffers<B>,
    /// Batched logits buffer: [max_active, vocab_size] bf16.
    /// Used by `forward_decode_batch` to produce logits for N sequences at once.
    /// Allocated even when batched decode is unsupported — the memory cost is
    /// small (32 seqs × 128K vocab × 2 bytes = ~8 MB).
    logits_batch: B::Tensor,
    backend: &'a B,
    /// Prefix cache for sharing KV blocks across sequences with identical prefixes.
    pub prefix_cache: PrefixCache,
}

impl<'a, B: GpuBackend> SingleGpuDispatch<'a, B> {
    pub fn new(
        model: Model<'a, B>,
        kv_pool: KvPool<B>,
        backend: &'a B,
        max_active: usize,
    ) -> Self {
        let prefill_bufs = PrefillBuffers::new(backend, model.config(), 1024);
        let logits_batch = backend.alloc_tensor(
            &[max_active, model.config().vocab_size],
            crate::gpu::TensorDtype::BF16,
        );
        // Default prefix cache: 64 entries.  Each entry holds a system prompt
        // or common prefix — 64 is generous for typical API server workloads
        // where a handful of system prompts dominate.
        let prefix_cache = PrefixCache::new(64);
        Self {
            model,
            kv_pool,
            prefill_bufs,
            logits_batch,
            backend,
            prefix_cache,
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
        self.model
            .forward_prefill_paged(tokens, &self.kv_pool, state, &self.prefill_bufs, images)
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
        self.model.forward_single_paged(token, &self.kv_pool, state)
    }

    fn finish_decode(state: &mut SeqKvState<B>) {
        state.advance();
    }

    fn sample(
        &self,
        temperature: f32,
        top_p: f32,
        rng: &mut impl rand::Rng,
    ) -> anyhow::Result<u32> {
        crate::model::sampler::sample(self.backend, self.model.logits(), temperature, top_p, rng)
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

        if let Some(evicted) = self.prefix_cache.insert(prefix_tokens.clone(), block_indices) {
            // Return evicted blocks to the free list.
            self.kv_pool.free_blocks(&evicted);
        }

        // Mark the prefix blocks as shared on this sequence so free_sequence()
        // won't return them to the pool — they now belong to the cache.
        state.mark_prefix_shared(prefix_blocks, prefix_tokens);
    }

    fn supports_batched_decode(&self) -> bool {
        self.model.arch_supports_batched_decode()
    }

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        positions: &[u32],
        states: &[&SeqKvState<B>],
    ) -> anyhow::Result<()> {
        self.model.forward_decode_batch_paged(
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
        temperatures: &[f32],
        top_ps: &[f32],
        rng: &mut impl rand::Rng,
    ) -> anyhow::Result<Vec<u32>> {
        crate::model::sampler::sample_batch(
            self.backend,
            &self.logits_batch,
            temperatures.len(),
            self.model.config().vocab_size,
            temperatures,
            top_ps,
            rng,
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
        kv_pool: KvPool<B>,
        tokenizer: Tokenizer,
        backend: &'a B,
        max_active: usize,
    ) -> Self {
        let dispatch = SingleGpuDispatch::new(model, kv_pool, backend, max_active);
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
        temperature: f32,
        top_p: f32,
        images: Vec<crate::model::vision::ProcessedImage>,
    ) -> SeqId {
        self.scheduler.add_request(SequenceRequest {
            prompt_tokens,
            max_gen_tokens,
            temperature,
            top_p,
            images,
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
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    use crate::gpu::cpu::CpuBackend;
    use crate::model::kv_cache::{KvPool, SeqKvState};

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
            let pool = KvPool::new(&backend, num_blocks, 4, 1);
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
            _temperature: f32,
            _top_p: f32,
            _rng: &mut impl rand::Rng,
        ) -> anyhow::Result<u32> {
            Ok(0)
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
            _temperature: f32,
            _top_p: f32,
            _rng: &mut impl rand::Rng,
        ) -> anyhow::Result<u32> {
            let token = self.next_token.get();
            self.next_token.set(token + 1);
            Ok(token)
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_request(prompt_len: usize, max_gen: usize) -> SequenceRequest {
        SequenceRequest {
            prompt_tokens: vec![1; prompt_len],
            max_gen_tokens: max_gen,
            temperature: 1.0,
            top_p: 0.9,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
        });
        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![3, 4],
            max_gen_tokens: 2,
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
        });
        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![2],
            max_gen_tokens: 1,
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            let pool = KvPool::new(&backend, num_blocks, 4, 1);
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
            _temperature: f32,
            _top_p: f32,
            _rng: &mut impl rand::Rng,
        ) -> anyhow::Result<u32> {
            Ok(0)
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
        });
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();

        // Seq 2: same 32-token prefix + 24-token different suffix = 56 tokens.
        let mut prompt2: Vec<u32> = (0..32).collect();
        prompt2.extend(1000..1024);
        scheduler.add_request(SequenceRequest {
            prompt_tokens: prompt2.clone(),
            max_gen_tokens: 2,
            temperature: 0.0,
            top_p: 1.0,
            images: Vec::new(),
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
}
