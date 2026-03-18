// ===========================================================================
// Inference engine: drives the continuous batching loop.
//
// LEARNING OVERVIEW
//
// What this file does:
//   The engine owns the model, scheduler, and tokenizer, and drives the
//   generation loop for MULTIPLE concurrent sequences.  Each step:
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
//   some decoding, some finishing.  The scheduler manages the sequence
//   lifecycle, and finished sequences' KV blocks are returned to the
//   free list for reuse.
//
// Dispatch trait:
//   The step loop is generic over the Dispatch trait (dispatch.rs), which
//   abstracts over single-GPU and multi-GPU backends.  run_step() contains
//   the loop logic once; SingleGpuDispatch and MultiGpuDispatch provide
//   the GPU-specific operations.
// ===========================================================================

pub(crate) mod dispatch;
pub(crate) mod multi_gpu;
pub(crate) mod scheduler;

use self::dispatch::Dispatch;
use self::scheduler::{Scheduler, SeqId, SequenceRequest};
use crate::gpu::GpuBackend;
use crate::model::kv_cache::{KvPool, SeqKvState};
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
}

/// Output from a single engine step.
pub(crate) struct StepOutput {
    /// Token generated this step for each active sequence (id, token_id).
    pub tokens: Vec<(SeqId, u32)>,
    /// Sequences that finished this step.
    pub finished: Vec<FinishedSequence>,
}

// ---------------------------------------------------------------------------
// run_step — the shared engine step loop, generic over Dispatch.
//
// This is the single implementation of admit → prefill → decode → sample →
// collect that both single-GPU and multi-GPU engines use.  The Dispatch
// trait provides the GPU-specific operations.
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
    //    Each sequence's entire prompt is processed in one forward pass using
    //    GEMM (mat-mat) instead of individual mat-vec calls per token.
    let prefilling_ids: Vec<SeqId> = scheduler
        .active
        .iter()
        .filter(|(_, seq)| !seq.pending_prefill.is_empty() && !seq.finished)
        .map(|(&id, _)| id)
        .collect();

    for id in prefilling_ids {
        let seq = scheduler.active.get_mut(&id).unwrap();

        // Drain all pending prefill tokens — the full prompt in one shot.
        let tokens: Vec<u32> = seq.pending_prefill.drain(..).collect();
        let chunk_size = tokens.len();
        let temperature = seq.temperature;
        let top_p = seq.top_p;

        // Pre-allocate all KV blocks and sync block table to GPU.
        dispatch.prepare_prefill(&mut seq.kv_state, chunk_size)?;

        // Run batched GEMM forward pass — the entire prompt in one call.
        dispatch.forward_prefill(&tokens, &seq.kv_state)?;
        D::finish_prefill(&mut seq.kv_state, chunk_size);

        // Sample the first generated token from the last token's logits.
        let next_token = dispatch.sample(temperature, top_p, &mut rng)?;
        seq.generated_tokens.push(next_token);
        step_tokens.push((id, next_token));

        if tokenizer.is_eos(next_token) || seq.generated_tokens.len() >= seq.max_gen_tokens {
            seq.finished = true;
        }
    }

    // 3. Decode: run one token per decoding sequence.
    let decoding_ids: Vec<SeqId> = scheduler
        .active
        .iter()
        .filter(|(_, seq)| seq.pending_prefill.is_empty() && !seq.finished)
        .map(|(&id, _)| id)
        .collect();

    for id in decoding_ids {
        let seq = scheduler.active.get_mut(&id).unwrap();
        let token = *seq.generated_tokens.last().unwrap();
        let temperature = seq.temperature;
        let top_p = seq.top_p;

        // Allocate one KV slot and sync block table.
        dispatch.prepare_decode(&mut seq.kv_state)?;

        // Run single-token forward pass.
        dispatch.forward_decode(token, &seq.kv_state)?;
        D::finish_decode(&mut seq.kv_state);
        model::profile::tick();

        // Sample next token.
        let next_token = dispatch.sample(temperature, top_p, &mut rng)?;
        seq.generated_tokens.push(next_token);
        step_tokens.push((id, next_token));

        if tokenizer.is_eos(next_token) || seq.generated_tokens.len() >= seq.max_gen_tokens {
            seq.finished = true;
        }
    }

    // 4. Collect finished sequences.
    let finished_pairs = scheduler.collect_finished(dispatch);
    let mut finished = Vec::new();
    for (id, seq) in finished_pairs {
        let text = tokenizer.decode(&seq.generated_tokens).unwrap_or_default();
        let reason =
            if seq.generated_tokens.last().map_or(false, |&t| tokenizer.is_eos(t)) {
                FinishReason::Eos
            } else {
                FinishReason::MaxTokens
            };
        finished.push(FinishedSequence {
            id,
            tokens: seq.generated_tokens,
            text,
            reason,
        });
    }

    Ok(StepOutput {
        tokens: step_tokens,
        finished,
    })
}

// ---------------------------------------------------------------------------
// SingleGpuDispatch — Dispatch implementation for single-GPU inference.
//
// Wraps Model + KvPool + PrefillBuffers + backend reference.  Each Dispatch
// method delegates to the appropriate model/kv_cache operation.
// ---------------------------------------------------------------------------

/// Single-GPU dispatch: wraps one Model, one KvPool, one set of PrefillBuffers.
pub(crate) struct SingleGpuDispatch<'a, B: GpuBackend> {
    pub model: Model<'a, B>,
    pub kv_pool: KvPool<B>,
    prefill_bufs: PrefillBuffers<B>,
    backend: &'a B,
}

impl<'a, B: GpuBackend> SingleGpuDispatch<'a, B> {
    pub fn new(model: Model<'a, B>, kv_pool: KvPool<B>, backend: &'a B) -> Self {
        let prefill_bufs = PrefillBuffers::new(backend, model.config(), 1024);
        Self {
            model,
            kv_pool,
            prefill_bufs,
            backend,
        }
    }
}

impl<'a, B: GpuBackend> Dispatch for SingleGpuDispatch<'a, B> {
    type SeqState = SeqKvState<B>;

    fn new_seq_state(&self) -> SeqKvState<B> {
        self.kv_pool.new_sequence(self.backend)
    }

    fn free_seq_state(&mut self, state: &SeqKvState<B>) {
        self.kv_pool.free_sequence(state);
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
        state.sync_block_table(self.backend);
        Ok(())
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        state: &SeqKvState<B>,
    ) -> anyhow::Result<()> {
        self.model.forward_prefill_paged(
            tokens,
            &self.kv_pool,
            state,
            &self.prefill_bufs,
        )
    }

    fn finish_prefill(state: &mut SeqKvState<B>, token_count: usize) {
        state.advance_by(token_count);
    }

    fn prepare_decode(&mut self, state: &mut SeqKvState<B>) -> anyhow::Result<()> {
        state.ensure_slot(&mut self.kv_pool)?;
        state.sync_block_table(self.backend);
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
}

// ---------------------------------------------------------------------------
// Engine — the single-GPU inference engine.
//
// Owns a SingleGpuDispatch, a Scheduler, and a Tokenizer.  Implements
// InferenceEngine by delegating to run_step().
// ---------------------------------------------------------------------------

/// The single-GPU inference engine.
pub(crate) struct Engine<'a, B: GpuBackend> {
    pub dispatch: SingleGpuDispatch<'a, B>,
    pub scheduler: Scheduler<SeqKvState<B>>,
    pub tokenizer: Tokenizer,
}

impl<'a, B: GpuBackend> Engine<'a, B> {
    pub fn new(
        model: Model<'a, B>,
        kv_pool: KvPool<B>,
        tokenizer: Tokenizer,
        backend: &'a B,
        max_active: usize,
    ) -> Self {
        let dispatch = SingleGpuDispatch::new(model, kv_pool, backend);
        let scheduler = Scheduler::new(max_active);
        Self {
            dispatch,
            scheduler,
            tokenizer,
        }
    }

    /// Submit a new completion request.  Returns the sequence ID.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_gen_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> SeqId {
        self.scheduler.add_request(SequenceRequest {
            prompt_tokens,
            max_gen_tokens,
            temperature,
            top_p,
        })
    }

    /// Abort a sequence, removing it from the waiting queue or active set.
    pub fn abort_sequence(&mut self, id: SeqId) {
        self.scheduler.abort_sequence(id, &mut self.dispatch);
    }

    /// Run one engine step.  Returns per-step tokens and newly finished sequences.
    pub fn step(&mut self) -> anyhow::Result<StepOutput> {
        run_step(&mut self.dispatch, &mut self.scheduler, &self.tokenizer)
    }

    /// Whether the engine has any work remaining.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }
}

// ---------------------------------------------------------------------------
// Tests — verify the shared run_step() loop using a mock Dispatch.
//
// MockDispatch tracks sequence state as a simple counter (no real GPU ops).
// It returns deterministic tokens: incrementing from 100, with a configurable
// EOS token.  This tests the engine loop logic (admission, prefill, decode,
// EOS detection, max_tokens) without needing any GPU backend.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Minimal sequence state for testing — just a position counter.
    struct MockSeqState {
        seq_len: usize,
    }

    /// Mock dispatch that returns deterministic tokens.
    ///
    /// Each call to `sample()` returns the next value from a counter starting
    /// at `next_token`.  The `eos_token` value is never returned by sample
    /// itself — tests that want EOS-triggered finishes set it to a value
    /// that the counter will hit.
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
            // Return blocks to the free list (one block per 16 positions).
            // Simplified: just add back some blocks.
            self.free_blocks += 1;
        }

        fn free_block_count(&self) -> usize {
            self.free_blocks
        }

        fn prepare_prefill(
            &mut self,
            _state: &mut MockSeqState,
            _token_count: usize,
        ) -> anyhow::Result<()> {
            self.prefill_count.set(self.prefill_count.get() + 1);
            Ok(())
        }

        fn forward_prefill(
            &self,
            _tokens: &[u32],
            _state: &MockSeqState,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn finish_prefill(state: &mut MockSeqState, token_count: usize) {
            state.seq_len += token_count;
        }

        fn prepare_decode(&mut self, _state: &mut MockSeqState) -> anyhow::Result<()> {
            self.decode_count.set(self.decode_count.get() + 1);
            Ok(())
        }

        fn forward_decode(
            &self,
            _token: u32,
            _state: &MockSeqState,
        ) -> anyhow::Result<()> {
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

    fn make_tokenizer(eos_ids: Vec<u32>) -> Tokenizer {
        Tokenizer::for_test(eos_ids)
    }

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
        let tokenizer = make_tokenizer(vec![999]); // EOS token that won't be hit

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2, 3, 4],
            max_gen_tokens: 3,
            temperature: 0.0,
            top_p: 1.0,
        });

        // Step 1: admit + prefill + decode (2 tokens in one step)
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.tokens.len(), 2);
        assert_eq!(out.tokens[0].1, 100); // from prefill
        assert_eq!(out.tokens[1].1, 101); // from decode (same step)
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
        });

        // Step 1: prefill → 100 (not EOS), decode → 101 (EOS) → finishes
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
        // should finish immediately.
        let mut dispatch = MockDispatch::new(10, 999); // 999 = first sampled token
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2, 3],
            max_gen_tokens: 10,
            temperature: 0.0,
            top_p: 1.0,
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
        // Step 1: both prefill + decode (prefill samples token, decode samples another).
        //         Each sequence gets 2 tokens → both hit max_gen_tokens=2 → both finish.
        let mut dispatch = MockDispatch::new(10, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2],
            max_gen_tokens: 2,
            temperature: 0.0,
            top_p: 1.0,
        });
        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![3, 4],
            max_gen_tokens: 2,
            temperature: 0.0,
            top_p: 1.0,
        });

        // Step 1: both admitted, prefilled, decoded, and finished (all in one step)
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(dispatch.prefill_count.get(), 2);
        // 2 prefill tokens + 2 decode tokens = 4 total
        assert_eq!(out.tokens.len(), 4);
        assert_eq!(out.finished.len(), 2);
    }

    #[test]
    fn test_step_admission_blocked_by_kv_memory() {
        // Only 0 free blocks → request can't be admitted.
        let mut dispatch = MockDispatch::new(0, 100);
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(4);
        let tokenizer = make_tokenizer(vec![999]);

        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1, 2],
            max_gen_tokens: 5,
            temperature: 0.0,
            top_p: 1.0,
        });

        // Step: nothing admitted, no tokens generated
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert!(out.tokens.is_empty());
        assert!(out.finished.is_empty());
        assert!(scheduler.has_work()); // still waiting
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
        });

        // Step 1: admit + prefill
        run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert!(scheduler.has_work());

        // Abort the sequence
        scheduler.abort_sequence(id, &mut dispatch);
        assert!(!scheduler.has_work());

        // Step 2: nothing to do
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
        let mut scheduler: Scheduler<MockSeqState> = Scheduler::new(1); // max 1 active
        let tokenizer = make_tokenizer(vec![999]);

        // Submit two requests — only 1 can be active at a time.
        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![1],
            max_gen_tokens: 1,
            temperature: 0.0,
            top_p: 1.0,
        });
        scheduler.add_request(SequenceRequest {
            prompt_tokens: vec![2],
            max_gen_tokens: 1,
            temperature: 0.0,
            top_p: 1.0,
        });

        // Step 1: A admitted + prefilled + finished (max_tokens=1)
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.finished.len(), 1);
        assert_eq!(out.finished[0].tokens, vec![100]);
        assert!(scheduler.has_work()); // B still waiting

        // Step 2: B admitted + prefilled + finished
        let out = run_step(&mut dispatch, &mut scheduler, &tokenizer).unwrap();
        assert_eq!(out.finished.len(), 1);
        assert_eq!(out.finished[0].tokens, vec![101]);
        assert!(!scheduler.has_work()); // done
    }
}

impl<'a, B: GpuBackend> InferenceEngine for Engine<'a, B> {
    fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_gen_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> SeqId {
        Engine::add_request(self, prompt_tokens, max_gen_tokens, temperature, top_p)
    }

    fn step(&mut self) -> anyhow::Result<StepOutput> {
        Engine::step(self)
    }

    fn abort_sequence(&mut self, id: SeqId) {
        Engine::abort_sequence(self, id)
    }

    fn has_work(&self) -> bool {
        Engine::has_work(self)
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}
