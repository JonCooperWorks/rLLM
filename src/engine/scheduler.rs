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
// ===========================================================================

use std::collections::{HashMap, VecDeque};

use crate::gpu::GpuBackend;
use crate::model::kv_cache::{BLOCK_SIZE, KvPool, SeqKvState};

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
pub(crate) struct Sequence<B: GpuBackend> {
    /// Tokens remaining to prefill (drained as prefill progresses).
    pub pending_prefill: VecDeque<u32>,
    /// KV cache state for this sequence.
    pub kv_state: SeqKvState<B>,
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
pub(crate) struct Scheduler<B: GpuBackend> {
    /// Waiting queue of new requests (ID pre-assigned at submission time).
    waiting: VecDeque<(SeqId, SequenceRequest)>,
    /// Active sequences currently being processed.
    pub active: HashMap<SeqId, Sequence<B>>,
    /// The shared KV block pool.
    pub kv_pool: KvPool<B>,
    /// Next sequence ID.
    next_id: SeqId,
    /// Maximum concurrent sequences.
    max_active: usize,
}

impl<B: GpuBackend> Scheduler<B> {
    pub fn new(kv_pool: KvPool<B>, max_active: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            active: HashMap::new(),
            kv_pool,
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
    pub fn schedule(&mut self, backend: &B) -> Vec<SeqId> {
        let mut admitted = Vec::new();

        while !self.waiting.is_empty() && self.active.len() < self.max_active {
            let (_, req) = self.waiting.front().unwrap();
            let prompt_blocks = (req.prompt_tokens.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
            if self.kv_pool.free_block_count() < prompt_blocks {
                break;
            }

            let (id, req) = self.waiting.pop_front().unwrap();

            let seq_state = self.kv_pool.new_sequence(backend);
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
    pub fn abort_sequence(&mut self, id: SeqId) {
        // Remove from waiting queue if still queued.
        self.waiting.retain(|(wid, _)| *wid != id);
        // Remove from active set and free KV blocks.
        if let Some(seq) = self.active.remove(&id) {
            self.kv_pool.free_sequence(&seq.kv_state);
        }
    }

    /// Collect and remove finished sequences, freeing their KV blocks.
    /// Returns the full `Sequence` so callers can inspect generated tokens
    /// and determine the finish reason.
    pub fn collect_finished(&mut self) -> Vec<(SeqId, Sequence<B>)> {
        let finished_ids: Vec<SeqId> = self
            .active
            .iter()
            .filter(|(_, seq)| seq.finished)
            .map(|(&id, _)| id)
            .collect();

        let mut results = Vec::new();
        for id in finished_ids {
            if let Some(seq) = self.active.remove(&id) {
                self.kv_pool.free_sequence(&seq.kv_state);
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

    fn make_request(prompt_len: usize, max_gen: usize) -> SequenceRequest {
        SequenceRequest {
            prompt_tokens: vec![1; prompt_len],
            max_gen_tokens: max_gen,
            temperature: 1.0,
            top_p: 0.9,
        }
    }

    fn setup(num_blocks: usize, max_active: usize) -> (CpuBackend, Scheduler<CpuBackend>) {
        let backend = CpuBackend;
        let pool = KvPool::new(&backend, num_blocks, 4, 1); // kv_dim=4, 1 layer
        let scheduler = Scheduler::new(pool, max_active);
        (backend, scheduler)
    }

    #[test]
    fn test_add_request_returns_incrementing_ids() {
        let (_b, mut s) = setup(10, 4);
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
        let (b, mut s) = setup(10, 4);
        s.add_request(make_request(4, 10));
        s.add_request(make_request(4, 10));

        let admitted = s.schedule(&b);
        assert_eq!(admitted.len(), 2);
        assert_eq!(s.active.len(), 2);
        assert_eq!(s.waiting_count(), 0);
    }

    #[test]
    fn test_schedule_respects_max_active() {
        let (b, mut s) = setup(10, 2);
        s.add_request(make_request(4, 10));
        s.add_request(make_request(4, 10));
        s.add_request(make_request(4, 10));

        let admitted = s.schedule(&b);
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
        let (b, mut s) = setup(2, 4);
        s.add_request(make_request(48, 10)); // needs ceil(48/16) = 3 blocks > 2 available

        let admitted = s.schedule(&b);
        assert_eq!(admitted.len(), 0);
        assert_eq!(s.waiting_count(), 1);

        // A request that fits should be admitted
        s.add_request(make_request(16, 10)); // needs 1 block
        let admitted = s.schedule(&b);
        // The first (48 tokens) still blocks, but second (16 tokens) is behind it in FCFS
        // FCFS means the blocked first request prevents the second from being tried
        assert_eq!(admitted.len(), 0);
    }

    #[test]
    fn test_abort_waiting_sequence() {
        let (_b, mut s) = setup(10, 4);
        let id0 = s.add_request(make_request(4, 10));
        let _id1 = s.add_request(make_request(4, 10));

        s.abort_sequence(id0);
        assert_eq!(s.waiting_count(), 1);
    }

    #[test]
    fn test_abort_active_sequence_frees_blocks() {
        let (b, mut s) = setup(4, 4);
        s.add_request(make_request(16, 10));
        let admitted = s.schedule(&b);
        let id = admitted[0];

        // Simulate the engine allocating blocks (schedule doesn't consume blocks)
        s.active.get_mut(&id).unwrap().kv_state.ensure_slots(&mut s.kv_pool, 16).unwrap();

        let free_before = s.kv_pool.free_block_count();
        s.abort_sequence(id);
        let free_after = s.kv_pool.free_block_count();

        assert!(s.active.is_empty());
        assert_eq!(free_after, free_before + 1);
    }

    #[test]
    fn test_collect_finished() {
        let (b, mut s) = setup(10, 4);
        let id0 = s.add_request(make_request(4, 10));
        let id1 = s.add_request(make_request(4, 10));
        s.schedule(&b);

        // Mark one as finished
        s.active.get_mut(&id0).unwrap().finished = true;

        let finished = s.collect_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].0, id0);
        assert_eq!(s.active.len(), 1);
        assert!(s.active.contains_key(&id1));
    }

    #[test]
    fn test_collect_finished_frees_blocks() {
        let (b, mut s) = setup(10, 4);
        s.add_request(make_request(16, 10));
        s.schedule(&b);

        // Simulate engine allocating blocks
        for seq in s.active.values_mut() {
            seq.kv_state.ensure_slots(&mut s.kv_pool, 16).unwrap();
        }

        let free_before = s.kv_pool.free_block_count();
        for seq in s.active.values_mut() {
            seq.finished = true;
        }
        s.collect_finished();
        let free_after = s.kv_pool.free_block_count();

        assert!(s.active.is_empty());
        assert_eq!(free_after, free_before + 1);
    }

    #[test]
    fn test_has_work() {
        let (b, mut s) = setup(10, 4);
        assert!(!s.has_work());

        s.add_request(make_request(4, 10));
        assert!(s.has_work()); // waiting

        s.schedule(&b);
        assert!(s.has_work()); // active

        for seq in s.active.values_mut() {
            seq.finished = true;
        }
        s.collect_finished();
        assert!(!s.has_work()); // empty
    }

    #[test]
    fn test_schedule_admits_after_finish_frees_blocks() {
        let (b, mut s) = setup(3, 4);
        // First request: 20 tokens, needs ceil(20/16) = 2 blocks
        s.add_request(make_request(20, 10));
        s.schedule(&b);
        assert_eq!(s.active.len(), 1);

        // Simulate engine allocating 2 blocks
        for seq in s.active.values_mut() {
            seq.kv_state.ensure_slots(&mut s.kv_pool, 20).unwrap();
        }
        assert_eq!(s.kv_pool.free_block_count(), 1);

        // Second request needs 2 blocks but only 1 free → can't admit
        s.add_request(make_request(20, 10));
        let admitted = s.schedule(&b);
        assert_eq!(admitted.len(), 0);
        assert_eq!(s.waiting_count(), 1);

        // Finish first → frees 2 blocks → second can be admitted
        for seq in s.active.values_mut() {
            seq.finished = true;
        }
        s.collect_finished();
        assert_eq!(s.kv_pool.free_block_count(), 3);

        let admitted = s.schedule(&b);
        assert_eq!(admitted.len(), 1);
        assert_eq!(s.active.len(), 1);
        assert_eq!(s.waiting_count(), 0);
    }
}
