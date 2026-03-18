// ===========================================================================
// Continuous batching scheduler.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Manages multiple concurrent sequences (requests) for continuous batching.
//   The scheduler decides which sequences to admit and tracks their state.
//   The engine (engine.rs) drives the actual forward passes and sampling.
//
// Why continuous batching?
//   Without batching, the GPU processes one sequence at a time.  A single
//   token's forward pass involves ~145 mat-vec multiplies that barely utilise
//   the GPU's compute capacity — they're memory-bandwidth bound.  By batching
//   N sequences, the mat-vec becomes a mat-mat with N columns, increasing
//   arithmetic intensity and saturating the GPU.
//
//   "Continuous" means sequences can join and leave mid-batch — when one
//   sequence finishes, its slot is immediately filled by a waiting request.
//   This maximises GPU utilisation compared to static batching where all
//   sequences must start and end together.
//
// Scheduler design:
//   Simple FCFS (first-come, first-served):
//   1. Waiting queue: new requests arrive here.
//   2. Active set: sequences currently generating tokens.
//   3. Each step: admit waiting sequences if KV blocks are available,
//      then the engine builds batches and runs forward passes.
//
// Generic over SeqState:
//   The scheduler doesn't know about GPU backends.  It is parameterised by
//   a SeqState type (the per-sequence KV cache state) which is provided by
//   the Dispatch trait.  Single-GPU uses SeqKvState<B>, multi-GPU uses
//   Vec<SeqKvState<CudaBackend>>.  The scheduler just stores and passes
//   these opaque values through — it never inspects them directly.
//
// Related files:
//   - engine/dispatch.rs  — Dispatch trait providing SeqState + GPU ops
//   - engine/mod.rs       — run_step() drives the scheduler + dispatch
//   - model/kv_cache.rs   — KvPool, SeqKvState (concrete SeqState types)
// ===========================================================================

use std::collections::{HashMap, VecDeque};

use crate::model::kv_cache;

use super::dispatch::Dispatch;

/// Unique identifier for a sequence request.
pub(crate) type SeqId = u64;

/// A new sequence request submitted to the scheduler.
pub(crate) struct SequenceRequest {
    pub prompt_tokens: Vec<u32>,
    pub max_gen_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}

/// State of a single active sequence.
///
/// Generic over `S` — the per-sequence KV cache state provided by the
/// Dispatch implementation.  The scheduler stores this opaquely.
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
}

/// The continuous batching scheduler.
///
/// Generic over `S` — the per-sequence KV cache state.  The scheduler manages
/// the waiting queue and active set, delegating GPU-specific operations (block
/// allocation, sequence creation/freeing) to the Dispatch trait.
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

    /// Try to admit waiting requests into the active set.
    ///
    /// Uses the Dispatch to check free block count (admission heuristic) and
    /// to create new per-sequence KV state.  Blocks are not actually allocated
    /// here — that happens later in prepare_prefill/prepare_decode.
    pub fn schedule(&mut self, dispatch: &impl Dispatch<SeqState = S>) -> Vec<SeqId> {
        let mut admitted = Vec::new();

        while !self.waiting.is_empty() && self.active.len() < self.max_active {
            let (_, req) = self.waiting.front().unwrap();
            // Encapsulate block size arithmetic — the scheduler doesn't need
            // to know the block size to check admission capacity.
            let prompt_blocks = kv_cache::blocks_needed_for(req.prompt_tokens.len());
            if dispatch.free_block_count() < prompt_blocks {
                break;
            }

            let (id, req) = self.waiting.pop_front().unwrap();

            let seq_state = dispatch.new_seq_state();
            let seq = Sequence {
                pending_prefill: req.prompt_tokens.into(),
                kv_state: seq_state,
                generated_tokens: Vec::new(),
                max_gen_tokens: req.max_gen_tokens,
                temperature: req.temperature,
                top_p: req.top_p,
                finished: false,
            };
            self.active.insert(id, seq);
            admitted.push(id);
        }

        admitted
    }

    /// Abort a sequence, removing it from the waiting queue or active set.
    /// Frees KV blocks if the sequence was active.
    pub fn abort_sequence(&mut self, id: SeqId, dispatch: &mut impl Dispatch<SeqState = S>) {
        // Remove from waiting queue if still queued.
        self.waiting.retain(|(wid, _)| *wid != id);
        // Remove from active set and free KV blocks.
        if let Some(seq) = self.active.remove(&id) {
            dispatch.free_seq_state(&seq.kv_state);
        }
    }

    /// Collect and remove finished sequences, freeing their KV blocks.
    /// Returns the full `Sequence` so callers can inspect generated tokens
    /// and determine the finish reason.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::cpu::CpuBackend;
    use crate::model::kv_cache::{KvPool, SeqKvState};

    /// Minimal Dispatch implementation backed by a CpuBackend KvPool,
    /// used to test scheduler admission and sequence lifecycle.
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

        fn prepare_prefill(
            &mut self,
            state: &mut Self::SeqState,
            token_count: usize,
        ) -> anyhow::Result<()> {
            state.ensure_slots(&mut self.pool, token_count)?;
            state.sync_block_table(&self.backend);
            Ok(())
        }

        fn forward_prefill(&self, _tokens: &[u32], _state: &Self::SeqState) -> anyhow::Result<()> {
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

    fn make_request(prompt_len: usize, max_gen: usize) -> SequenceRequest {
        SequenceRequest {
            prompt_tokens: vec![1; prompt_len],
            max_gen_tokens: max_gen,
            temperature: 1.0,
            top_p: 0.9,
        }
    }

    fn setup(num_blocks: usize, max_active: usize) -> (TestDispatch, Scheduler<SeqKvState<CpuBackend>>) {
        let dispatch = TestDispatch::new(num_blocks);
        let scheduler = Scheduler::new(max_active);
        (dispatch, scheduler)
    }

    #[test]
    fn test_add_request_returns_incrementing_ids() {
        let (_d, mut s) = setup(10, 4);
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
        let (d, mut s) = setup(10, 4);
        s.add_request(make_request(4, 10));
        s.add_request(make_request(4, 10));

        let admitted = s.schedule(&d);
        assert_eq!(admitted.len(), 2);
        assert_eq!(s.active.len(), 2);
        assert_eq!(s.waiting_count(), 0);
    }

    #[test]
    fn test_schedule_respects_max_active() {
        let (d, mut s) = setup(10, 2);
        s.add_request(make_request(4, 10));
        s.add_request(make_request(4, 10));
        s.add_request(make_request(4, 10));

        let admitted = s.schedule(&d);
        assert_eq!(admitted.len(), 2);
        assert_eq!(s.active.len(), 2);
        assert_eq!(s.waiting_count(), 1); // one still waiting
    }

    #[test]
    fn test_schedule_blocks_on_kv_memory() {
        // schedule() checks free_block_count but doesn't allocate blocks —
        // it's an admission heuristic.  With 2 blocks, a 20-token prompt
        // (needs 2 blocks) can be admitted, but a second identical request
        // also passes the check because blocks aren't consumed at admit time.
        // Use a request that needs MORE blocks than exist to see rejection.
        let (d, mut s) = setup(2, 4);
        s.add_request(make_request(48, 10)); // needs ceil(48/16) = 3 blocks > 2 available

        let admitted = s.schedule(&d);
        assert_eq!(admitted.len(), 0);
        assert_eq!(s.waiting_count(), 1);

        // A request that fits should be admitted
        s.add_request(make_request(16, 10)); // needs 1 block
        let admitted = s.schedule(&d);
        // The first (48 tokens) still blocks, but second (16 tokens) is behind it in FCFS
        // FCFS means the blocked first request prevents the second from being tried
        assert_eq!(admitted.len(), 0);
    }

    #[test]
    fn test_abort_waiting_sequence() {
        let (mut d, mut s) = setup(10, 4);
        let id0 = s.add_request(make_request(4, 10));
        let _id1 = s.add_request(make_request(4, 10));

        s.abort_sequence(id0, &mut d);
        assert_eq!(s.waiting_count(), 1);
    }

    #[test]
    fn test_abort_active_sequence_frees_blocks() {
        let (mut d, mut s) = setup(4, 4);
        s.add_request(make_request(16, 10));
        let admitted = s.schedule(&d);
        let id = admitted[0];

        // Simulate the engine allocating blocks via dispatch
        d.prepare_prefill(&mut s.active.get_mut(&id).unwrap().kv_state, 16).unwrap();

        let free_before = d.free_block_count();
        s.abort_sequence(id, &mut d);
        let free_after = d.free_block_count();

        assert!(s.active.is_empty());
        assert_eq!(free_after, free_before + 1);
    }

    #[test]
    fn test_collect_finished() {
        let (mut d, mut s) = setup(10, 4);
        let id0 = s.add_request(make_request(4, 10));
        let id1 = s.add_request(make_request(4, 10));
        s.schedule(&d);

        // Mark one as finished
        s.active.get_mut(&id0).unwrap().finished = true;

        let finished = s.collect_finished(&mut d);
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].0, id0);
        assert_eq!(s.active.len(), 1);
        assert!(s.active.contains_key(&id1));
    }

    #[test]
    fn test_collect_finished_frees_blocks() {
        let (mut d, mut s) = setup(10, 4);
        s.add_request(make_request(16, 10));
        s.schedule(&d);

        // Simulate engine allocating blocks
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
        let (mut d, mut s) = setup(10, 4);
        assert!(!s.has_work());

        s.add_request(make_request(4, 10));
        assert!(s.has_work()); // waiting

        s.schedule(&d);
        assert!(s.has_work()); // active

        for seq in s.active.values_mut() {
            seq.finished = true;
        }
        s.collect_finished(&mut d);
        assert!(!s.has_work()); // empty
    }

    #[test]
    fn test_schedule_admits_after_finish_frees_blocks() {
        let (mut d, mut s) = setup(3, 4);
        // First request: 20 tokens, needs ceil(20/16) = 2 blocks
        s.add_request(make_request(20, 10));
        s.schedule(&d);
        assert_eq!(s.active.len(), 1);

        // Simulate engine allocating 2 blocks
        for seq in s.active.values_mut() {
            d.prepare_prefill(&mut seq.kv_state, 20).unwrap();
        }
        assert_eq!(d.free_block_count(), 1);

        // Second request needs 2 blocks but only 1 free → can't admit
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
}
