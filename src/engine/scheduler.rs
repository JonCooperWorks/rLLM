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
}
