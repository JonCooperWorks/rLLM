// ===========================================================================
// Paged KV cache for efficient memory management.
//
// LEARNING OVERVIEW
//
// Why paged allocation?
//   A flat [MAX_SEQ_LEN, kv_dim] cache wastes memory when sequences are
//   short.  With paging, memory is allocated in fixed-size blocks on demand.
//   A 100-token sequence uses 7 blocks (ceil(100/16)), not 4096 positions.
//
//   More importantly, paging enables continuous batching: sequences can be
//   added/removed without fragmenting GPU memory, because blocks are
//   interchangeable (any block can hold any logical position from any
//   sequence).
//
// Memory layout:
//   The physical KV pool is a single large GPU buffer per layer per K/V:
//     pool shape: [num_blocks * BLOCK_SIZE, kv_dim]
//   A block table (per sequence) maps logical block indices to physical ones:
//     block_table[logical_block] = physical_block_index
//
//   To find position `pos`:
//     logical_block = pos / BLOCK_SIZE
//     offset_in_block = pos % BLOCK_SIZE
//     physical_block = block_table[logical_block]
//     data_index = (physical_block * BLOCK_SIZE + offset_in_block) * kv_dim
//
// Block size: 16 tokens.
//   Each block stores 16 positions of KV data.  For Llama 3.2 1B (kv_dim=512,
//   bf16), one block is 16 * 512 * 2 = 16 KB per layer per K/V.  This aligns
//   well with Metal's memory access patterns and flash attention tile sizes.
// ===========================================================================

use std::collections::HashMap;

use crate::gpu::{GpuCore, TensorDtype};

/// Number of token positions stored per KV cache block.
pub(crate) const BLOCK_SIZE: usize = 16;

/// Maximum number of logical blocks per sequence (supports up to
/// MAX_BLOCKS_PER_SEQ * BLOCK_SIZE = 8192 * 16 = 131072 tokens = 128K context).
pub(crate) const MAX_BLOCKS_PER_SEQ: usize = 8192;

/// Number of blocks needed to store `token_count` tokens.
///
/// Module-level convenience function that encapsulates the block size so
/// callers (scheduler, commands) don't need to import BLOCK_SIZE directly.
/// This keeps the block size as an internal implementation detail of the
/// KV cache module — if we ever change it, only this module needs updating.
pub(crate) fn blocks_needed_for(token_count: usize) -> usize {
    (token_count + BLOCK_SIZE - 1) / BLOCK_SIZE
}

/// A pool of KV cache blocks shared across all sequences.
///
/// The pool owns the physical GPU memory for all K and V caches across all
/// layers.  Individual sequences get assigned physical blocks from this pool
/// via their block tables.
#[allow(dead_code)]
pub(crate) struct KvPool<B: GpuCore> {
    /// Physical K cache pool: one buffer per layer.
    /// Shape per buffer: [num_physical_blocks * BLOCK_SIZE, kv_dim] in bf16.
    pub k_pool: Vec<B::Tensor>,
    /// Physical V cache pool: one buffer per layer.
    pub v_pool: Vec<B::Tensor>,

    /// Free list of physical block indices (LIFO stack for locality).
    free_blocks: Vec<u32>,
    /// Total number of physical blocks in the pool.
    num_physical_blocks: usize,
    /// KV dimension (num_kv_heads * head_dim).
    kv_dim: usize,
    /// Number of transformer layers.
    num_layers: usize,
}

/// Per-sequence KV cache state: a block table and current length.
///
/// The block table maps logical block indices to physical block indices
/// in the shared KvPool.  It lives on the CPU and is uploaded to GPU
/// when the attention kernel needs it.
///
/// When prefix caching is active, the first N blocks may be shared with
/// other sequences.  `shared_prefix_len` tracks how many leading blocks
/// are borrowed from the PrefixCache (these must NOT be freed when the
/// sequence finishes — they belong to the cache).
pub(crate) struct SeqKvState<B: GpuCore> {
    /// Logical block index -> physical block index.
    /// Length = ceil(seq_len / BLOCK_SIZE), grows as the sequence gets longer.
    block_table_cpu: Vec<u32>,
    /// GPU-resident copy of the block table, uploaded when modified.
    /// Shape: [MAX_BLOCKS_PER_SEQ] u32 (padded with zeros for unused slots).
    pub block_table_gpu: B::Tensor,
    /// Number of tokens currently stored in this sequence's KV cache.
    pub seq_len: usize,
    /// Whether the GPU copy needs updating.
    dirty: bool,
    /// Number of leading blocks borrowed from the prefix cache.
    /// These blocks are read-only and must not be freed by this sequence.
    shared_prefix_blocks: usize,
    /// The prefix tokens that were cached (for release on free).
    /// None if this sequence doesn't use a cached prefix.
    shared_prefix_tokens: Option<Vec<u32>>,
}

// ===========================================================================
// Prefix cache — shares KV blocks across sequences with identical prefixes.
//
// The insight: when many requests share a common prefix (system prompt,
// few-shot examples, tool definitions), the KV computed for that prefix is
// identical.  Rather than re-running prefill for every request, we compute
// it once, keep the blocks alive, and let subsequent sequences point their
// block tables at the same physical blocks.
//
// Mechanism:
//   1. After prefill, the engine hashes the prefix tokens and registers the
//      block table entries in PrefixCache.
//   2. When a new request arrives with matching tokens, the engine copies
//      the cached block indices into the new sequence's block table,
//      advances seq_len past the prefix, and only prefills the suffix.
//   3. Reference counting tracks how many active sequences share each
//      cached prefix.  When the ref count drops to zero AND the entry is
//      evicted, the blocks are returned to the free list.
//
// The cache is keyed by a hash of the token sequence, checked against the
// full token list for collision safety.  Entries are evicted LRU when the
// cache exceeds its capacity.
// ===========================================================================

/// A cached prefix: the token sequence, block indices, and reference count.
pub(crate) struct CachedPrefix {
    /// The token IDs that produced this prefix's KV data.
    /// Stored for collision checking — the hash is not sufficient alone.
    pub tokens: Vec<u32>,
    /// Physical block indices containing this prefix's KV data.
    /// These blocks are NOT on the free list while the entry exists.
    pub block_indices: Vec<u32>,
    /// Number of token positions filled in these blocks.
    /// Always equals `tokens.len()`.
    pub token_count: usize,
    /// Number of active sequences currently using this prefix.
    /// Blocks cannot be freed while ref_count > 0.
    ref_count: usize,
    /// Monotonic counter for LRU eviction (higher = more recent).
    last_used: u64,
}

/// Prefix cache: maps token-sequence hashes to shared KV block sets.
///
/// Lives alongside KvPool and manages the lifecycle of shared prefix blocks.
/// The cache has a fixed capacity (max entries); when full, the least recently
/// used entry with ref_count == 0 is evicted and its blocks freed.
pub(crate) struct PrefixCache {
    /// Hash of prefix tokens → entry index.
    entries: HashMap<u64, CachedPrefix>,
    /// Maximum number of cached prefixes.
    max_entries: usize,
    /// Monotonic clock for LRU ordering.
    clock: u64,
    /// Running stats.
    pub hits: u64,
    pub misses: u64,
}

impl PrefixCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Hash a token sequence.  Uses FNV-1a for speed — we verify against
    /// the stored tokens on hit, so collisions are safe (just a miss).
    fn hash_tokens(tokens: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
        for &t in tokens {
            h ^= t as u64;
            h = h.wrapping_mul(0x100000001b3); // FNV prime
        }
        h
    }

    /// Look up a prefix.  Returns the cached block indices and token count
    /// if found, and increments the ref count.
    ///
    /// The caller provides the full prompt tokens; this method checks all
    /// prefixes that are a prefix of the prompt (longest match wins).
    pub fn lookup(&mut self, prompt_tokens: &[u32]) -> Option<(Vec<u32>, usize)> {
        // Try progressively shorter block-aligned prefixes.
        // Start from the longest block-aligned prefix of the prompt.
        let max_prefix_blocks = prompt_tokens.len() / BLOCK_SIZE;
        for num_blocks in (1..=max_prefix_blocks).rev() {
            let prefix_len = num_blocks * BLOCK_SIZE;
            let prefix = &prompt_tokens[..prefix_len];
            let hash = Self::hash_tokens(prefix);
            if let Some(entry) = self.entries.get_mut(&hash) {
                // Verify tokens match (collision safety).
                if entry.tokens == prefix {
                    entry.ref_count += 1;
                    self.clock += 1;
                    entry.last_used = self.clock;
                    self.hits += 1;
                    return Some((entry.block_indices.clone(), entry.token_count));
                }
            }
        }
        self.misses += 1;
        None
    }

    /// Register a new prefix after prefill completes.
    ///
    /// `tokens` is the prefix token sequence (should be block-aligned).
    /// `block_indices` are the physical blocks holding the KV data.
    /// The entry starts with ref_count = 1 (the sequence that just prefilled).
    ///
    /// If the cache is full, evicts the LRU entry with ref_count == 0.
    /// Returns the evicted blocks (if any) so the caller can free them.
    pub fn insert(
        &mut self,
        tokens: Vec<u32>,
        block_indices: Vec<u32>,
    ) -> Option<Vec<u32>> {
        let hash = Self::hash_tokens(&tokens);
        if self.entries.contains_key(&hash) {
            return None; // Already cached (race between concurrent prefills).
        }

        // Evict if at capacity.
        let evicted_blocks = if self.entries.len() >= self.max_entries {
            self.evict_lru()
        } else {
            None
        };

        let token_count = tokens.len();
        self.clock += 1;
        self.entries.insert(
            hash,
            CachedPrefix {
                tokens,
                block_indices,
                token_count,
                ref_count: 1,
                last_used: self.clock,
            },
        );

        evicted_blocks
    }

    /// Decrement the ref count for a prefix matching `tokens`.
    ///
    /// Called when a sequence finishes or is aborted.  Does NOT free blocks —
    /// the entry stays cached for future reuse until evicted.
    pub fn release(&mut self, tokens: &[u32]) {
        let hash = Self::hash_tokens(tokens);
        if let Some(entry) = self.entries.get_mut(&hash) {
            if entry.tokens == tokens {
                entry.ref_count = entry.ref_count.saturating_sub(1);
            }
        }
    }

    /// Evict the least recently used entry with ref_count == 0.
    /// Returns the freed block indices, or None if nothing is evictable.
    fn evict_lru(&mut self) -> Option<Vec<u32>> {
        let victim = self
            .entries
            .iter()
            .filter(|(_, e)| e.ref_count == 0)
            .min_by_key(|(_, e)| e.last_used)
            .map(|(&hash, _)| hash);

        if let Some(hash) = victim {
            let entry = self.entries.remove(&hash).unwrap();
            Some(entry.block_indices)
        } else {
            None
        }
    }

    /// Number of cached prefixes.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Total blocks held by the cache (not on the free list).
    pub fn blocks_held(&self) -> usize {
        self.entries.values().map(|e| e.block_indices.len()).sum()
    }

    /// Hit rate as a fraction (0.0 to 1.0).
    #[allow(dead_code)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl<B: GpuCore> KvPool<B> {
    /// Allocate a KV pool with room for `num_blocks` physical blocks.
    ///
    /// Total GPU memory = num_blocks * BLOCK_SIZE * kv_dim * 2 bytes * 2 (K+V)
    ///                     * num_layers.
    /// For 256 blocks, 16 layers, kv_dim=512: 256 * 16 * 512 * 2 * 2 * 16 = 128 MB.
    pub fn new(backend: &B, num_blocks: usize, kv_dim: usize, num_layers: usize) -> Self {
        let total_positions = num_blocks * BLOCK_SIZE;
        let mut k_pool = Vec::with_capacity(num_layers);
        let mut v_pool = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_pool.push(backend.alloc_tensor(&[total_positions, kv_dim], TensorDtype::BF16));
            v_pool.push(backend.alloc_tensor(&[total_positions, kv_dim], TensorDtype::BF16));
        }

        // Initialise free list with all block indices (in reverse so pop gives 0, 1, 2, ...).
        let free_blocks: Vec<u32> = (0..num_blocks as u32).rev().collect();

        Self {
            k_pool,
            v_pool,
            free_blocks,
            num_physical_blocks: num_blocks,
            kv_dim,
            num_layers,
        }
    }

    /// Number of free blocks remaining.
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Number of physical blocks in the pool.
    pub fn total_blocks(&self) -> usize {
        self.num_physical_blocks
    }

    /// Maximum number of token positions across the entire pool.
    ///
    /// Callers use this for status messages instead of computing
    /// `num_blocks * BLOCK_SIZE` themselves — keeps the block size
    /// encapsulated within the KV cache module.
    pub fn max_tokens(&self) -> usize {
        self.num_physical_blocks * BLOCK_SIZE
    }

    /// Total GPU memory used by this pool in bytes.
    ///
    /// Accounts for K + V pools across all layers.  Used for memory
    /// reporting in commands and the API server.
    pub fn total_memory_bytes(&self) -> usize {
        // 2 (K+V) × num_layers × num_blocks × BLOCK_SIZE × kv_dim × 2 (bf16)
        2 * self.num_layers * self.num_physical_blocks * BLOCK_SIZE * self.kv_dim * 2
    }

    /// Number of blocks needed to store `token_count` tokens.
    ///
    /// Encapsulates the `ceil(tokens / BLOCK_SIZE)` arithmetic so callers
    /// (scheduler, commands) don't need to know the block size.
    /// Delegates to the module-level `blocks_needed_for()` for convenience.
    #[allow(dead_code)]
    pub fn blocks_needed_for(token_count: usize) -> usize {
        blocks_needed_for(token_count)
    }

    /// Allocate one physical block from the free list.  Returns None if OOM.
    pub fn alloc_block(&mut self) -> Option<u32> {
        self.free_blocks.pop()
    }

    /// Return a physical block to the free list.
    pub fn free_block(&mut self, block_idx: u32) {
        self.free_blocks.push(block_idx);
    }

    /// Create a new empty sequence state.
    pub fn new_sequence(&self, backend: &B) -> SeqKvState<B> {
        // Allocate GPU block table buffer (fixed size, padded with zeros).
        let block_table_gpu = backend.alloc_tensor(
            &[MAX_BLOCKS_PER_SEQ],
            TensorDtype::F32, // Using F32 for u32 storage (same byte size).
        );
        SeqKvState {
            block_table_cpu: Vec::new(),
            block_table_gpu,
            seq_len: 0,
            dirty: false,
            shared_prefix_blocks: 0,
            shared_prefix_tokens: None,
        }
    }

    /// Free blocks belonging to a sequence.
    ///
    /// Prefix-aware: only frees blocks that this sequence owns (allocated
    /// after the shared prefix).  Shared prefix blocks are managed by the
    /// PrefixCache and freed only on eviction.
    pub fn free_sequence(&mut self, seq: &SeqKvState<B>) {
        // Skip the first `shared_prefix_blocks` — those belong to the cache.
        for &block_idx in &seq.block_table_cpu[seq.shared_prefix_blocks..] {
            self.free_block(block_idx);
        }
    }

    /// Return evicted prefix blocks to the free list.
    ///
    /// Called when PrefixCache evicts an entry — its blocks were held out
    /// of the free list while cached, and now need to be returned.
    pub fn free_blocks(&mut self, blocks: &[u32]) {
        for &block_idx in blocks {
            self.free_block(block_idx);
        }
    }

    /// Number of layers.
    #[allow(dead_code)]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// KV dimension.
    #[allow(dead_code)]
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

impl<B: GpuCore> SeqKvState<B> {
    /// Ensure there is room for one more token.  If the current last block
    /// is full (or no blocks allocated yet), allocate a new block from the pool.
    pub fn ensure_slot(&mut self, pool: &mut KvPool<B>) -> anyhow::Result<()> {
        let needed_blocks = (self.seq_len + BLOCK_SIZE) / BLOCK_SIZE; // ceil((seq_len+1)/BLOCK_SIZE)
        if needed_blocks > self.block_table_cpu.len() {
            // Need a new block.
            let block_idx = pool
                .alloc_block()
                .ok_or_else(|| anyhow::anyhow!("KV cache out of memory: no free blocks"))?;
            self.block_table_cpu.push(block_idx);
            self.dirty = true;
        }
        Ok(())
    }

    /// Upload the block table to GPU if it has been modified since last upload.
    pub fn sync_block_table(&mut self, backend: &B) {
        if !self.dirty {
            return;
        }
        // Build padded u32 array.
        let mut table = vec![0u32; MAX_BLOCKS_PER_SEQ];
        for (i, &block_idx) in self.block_table_cpu.iter().enumerate() {
            table[i] = block_idx;
        }
        // Upload via copy_to_tensor (works on unified memory and discrete GPUs).
        let bytes: &[u8] = bytemuck::cast_slice(&table);
        backend.copy_to_tensor(&self.block_table_gpu, bytes);
        self.dirty = false;
    }

    /// Advance seq_len by 1 after writing a token's KV.
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Advance seq_len by `count` after writing multiple tokens' KV (batched prefill).
    ///
    /// After `forward_prefill_paged` writes K/V for the whole prompt chunk
    /// into the paged cache, this records that those positions are now occupied.
    /// Future decode steps will attend to all these positions.
    pub fn advance_by(&mut self, count: usize) {
        self.seq_len += count;
    }

    /// Ensure there is room for `count` more tokens.
    /// Allocates new blocks as needed from the pool.
    ///
    /// For batched prefill: called once before the GEMM forward pass to
    /// pre-allocate ALL blocks needed for the entire prompt.  For a 100-token
    /// prompt with BLOCK_SIZE=16, this allocates ceil(100/16) = 7 blocks
    /// from the shared pool's free list.
    ///
    /// This is the bulk version of `ensure_slot()` (which allocates one
    /// position at a time for decode).  Pre-allocating avoids the overhead
    /// of checking and potentially allocating inside the per-token loop.
    pub fn ensure_slots(&mut self, pool: &mut KvPool<B>, count: usize) -> anyhow::Result<()> {
        let new_len = self.seq_len + count;
        let needed_blocks = (new_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        while self.block_table_cpu.len() < needed_blocks {
            let block_idx = pool
                .alloc_block()
                .ok_or_else(|| anyhow::anyhow!("KV cache out of memory: no free blocks"))?;
            self.block_table_cpu.push(block_idx);
            self.dirty = true;
        }
        Ok(())
    }

    /// Link this sequence to a cached prefix.
    ///
    /// Copies the prefix's physical block indices into this sequence's block
    /// table and advances seq_len to skip the already-computed positions.
    /// The blocks are NOT allocated from the free list — they're borrowed
    /// from the PrefixCache.
    pub fn link_prefix(
        &mut self,
        prefix_blocks: &[u32],
        prefix_token_count: usize,
        prefix_tokens: Vec<u32>,
    ) {
        self.block_table_cpu = prefix_blocks.to_vec();
        self.seq_len = prefix_token_count;
        self.shared_prefix_blocks = prefix_blocks.len();
        self.shared_prefix_tokens = Some(prefix_tokens);
        self.dirty = true;
    }

    /// Mark leading blocks as shared with the prefix cache.
    ///
    /// Called after `prefix_cache_register` inserts this sequence's prefix
    /// blocks into the cache.  Ensures `free_sequence()` won't free them
    /// (they now belong to the cache, not this sequence).
    pub fn mark_prefix_shared(&mut self, num_blocks: usize, prefix_tokens: Vec<u32>) {
        self.shared_prefix_blocks = num_blocks;
        self.shared_prefix_tokens = Some(prefix_tokens);
    }

    /// The prefix tokens this sequence is sharing, if any.
    pub fn shared_prefix_tokens(&self) -> Option<&[u32]> {
        self.shared_prefix_tokens.as_deref()
    }

    /// Read-only access to the CPU block table.
    ///
    /// Used by the prefix cache to record which physical blocks hold a
    /// prefix's KV data after prefill completes.
    pub fn block_table_cpu_slice(&self) -> &[u32] {
        &self.block_table_cpu
    }

    /// Number of logical blocks currently allocated.
    #[allow(dead_code)]
    pub fn num_blocks(&self) -> usize {
        self.block_table_cpu.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::cpu::CpuBackend;

    fn make_pool(num_blocks: usize) -> (CpuBackend, KvPool<CpuBackend>) {
        let backend = CpuBackend;
        let pool = KvPool::new(&backend, num_blocks, 4, 2); // kv_dim=4, 2 layers
        (backend, pool)
    }

    #[test]
    fn test_pool_initial_state() {
        let (_b, pool) = make_pool(8);
        assert_eq!(pool.free_block_count(), 8);
        assert_eq!(pool.total_blocks(), 8);
        assert_eq!(pool.num_layers(), 2);
        assert_eq!(pool.kv_dim(), 4);
        assert_eq!(pool.k_pool.len(), 2);
        assert_eq!(pool.v_pool.len(), 2);
    }

    #[test]
    fn test_alloc_and_free_block() {
        let (_b, mut pool) = make_pool(4);
        let b0 = pool.alloc_block().unwrap();
        assert_eq!(b0, 0); // LIFO: reversed, so first pop = 0
        assert_eq!(pool.free_block_count(), 3);

        let b1 = pool.alloc_block().unwrap();
        assert_eq!(b1, 1);
        assert_eq!(pool.free_block_count(), 2);

        pool.free_block(b0);
        assert_eq!(pool.free_block_count(), 3);

        // Re-allocate should get b0 back (LIFO)
        let b_realloc = pool.alloc_block().unwrap();
        assert_eq!(b_realloc, b0);
    }

    #[test]
    fn test_alloc_block_exhaustion() {
        let (_b, mut pool) = make_pool(2);
        assert!(pool.alloc_block().is_some());
        assert!(pool.alloc_block().is_some());
        assert!(pool.alloc_block().is_none()); // exhausted
    }

    #[test]
    fn test_new_sequence_empty() {
        let (b, pool) = make_pool(4);
        let seq = pool.new_sequence(&b);
        assert_eq!(seq.seq_len, 0);
        assert_eq!(seq.num_blocks(), 0);
    }

    #[test]
    fn test_ensure_slot_allocates_blocks() {
        let (b, mut pool) = make_pool(4);
        let mut seq = pool.new_sequence(&b);

        // First token needs first block
        seq.ensure_slot(&mut pool).unwrap();
        assert_eq!(seq.num_blocks(), 1);
        assert_eq!(pool.free_block_count(), 3);

        // Fill up first block (BLOCK_SIZE = 16 tokens)
        seq.advance();
        for _ in 1..BLOCK_SIZE {
            seq.ensure_slot(&mut pool).unwrap();
            seq.advance();
        }
        assert_eq!(seq.num_blocks(), 1); // still one block
        assert_eq!(seq.seq_len, BLOCK_SIZE);

        // Next token spills into second block
        seq.ensure_slot(&mut pool).unwrap();
        assert_eq!(seq.num_blocks(), 2);
        assert_eq!(pool.free_block_count(), 2);
    }

    #[test]
    fn test_ensure_slots_bulk() {
        let (b, mut pool) = make_pool(10);
        let mut seq = pool.new_sequence(&b);

        // Bulk allocate for 40 tokens: ceil(40/16) = 3 blocks
        seq.ensure_slots(&mut pool, 40).unwrap();
        assert_eq!(seq.num_blocks(), 3);
        assert_eq!(pool.free_block_count(), 7);
    }

    #[test]
    fn test_ensure_slot_oom() {
        let (b, mut pool) = make_pool(1);
        let mut seq = pool.new_sequence(&b);

        // First block ok
        seq.ensure_slot(&mut pool).unwrap();
        // Fill the block
        for _ in 0..BLOCK_SIZE {
            seq.advance();
        }
        // Next block should fail (only 1 block total, already used)
        let result = seq.ensure_slot(&mut pool);
        assert!(result.is_err());
    }

    #[test]
    fn test_free_sequence_returns_blocks() {
        let (b, mut pool) = make_pool(4);
        let mut seq = pool.new_sequence(&b);
        seq.ensure_slots(&mut pool, 20).unwrap(); // 2 blocks
        assert_eq!(pool.free_block_count(), 2);

        pool.free_sequence(&seq);
        assert_eq!(pool.free_block_count(), 4); // all returned
    }

    #[test]
    fn test_sync_block_table() {
        let (b, mut pool) = make_pool(4);
        let mut seq = pool.new_sequence(&b);
        seq.ensure_slot(&mut pool).unwrap();

        // Should be dirty after alloc
        assert!(seq.dirty);
        seq.sync_block_table(&b);
        assert!(!seq.dirty);

        // Second sync should be a no-op
        seq.sync_block_table(&b);
        assert!(!seq.dirty);
    }

    #[test]
    fn test_advance_and_advance_by() {
        let (b, pool) = make_pool(4);
        let mut seq = pool.new_sequence(&b);
        assert_eq!(seq.seq_len, 0);

        seq.advance();
        assert_eq!(seq.seq_len, 1);

        seq.advance_by(10);
        assert_eq!(seq.seq_len, 11);
    }

    // -----------------------------------------------------------------------
    // PrefixCache tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefix_cache_miss_then_hit() {
        let mut cache = PrefixCache::new(4);

        // First lookup: miss (cache empty).
        let prompt: Vec<u32> = (0..32).collect(); // 2 blocks worth
        assert!(cache.lookup(&prompt).is_none());
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 0);

        // Insert a prefix (block-aligned: 32 tokens = 2 blocks).
        let blocks = vec![10, 20];
        cache.insert(prompt.clone(), blocks.clone());

        // Second lookup: hit.
        let result = cache.lookup(&prompt);
        assert!(result.is_some());
        let (found_blocks, token_count) = result.unwrap();
        assert_eq!(found_blocks, vec![10, 20]);
        assert_eq!(token_count, 32);
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn test_prefix_cache_longer_prompt_matches_prefix() {
        let mut cache = PrefixCache::new(4);

        // Cache a 32-token prefix (2 blocks).
        let prefix: Vec<u32> = (0..32).collect();
        cache.insert(prefix.clone(), vec![10, 20]);

        // Look up with a longer prompt that starts with the same 32 tokens.
        let mut long_prompt: Vec<u32> = (0..48).collect(); // 3 blocks
        let result = cache.lookup(&long_prompt);
        assert!(result.is_some());
        let (blocks, count) = result.unwrap();
        assert_eq!(blocks, vec![10, 20]);
        assert_eq!(count, 32);
    }

    #[test]
    fn test_prefix_cache_ref_counting() {
        let mut cache = PrefixCache::new(4);
        let prefix: Vec<u32> = (0..16).collect(); // 1 block

        // Insert with initial ref_count = 1.
        cache.insert(prefix.clone(), vec![5]);

        // Two more lookups → ref_count = 3.
        cache.lookup(&prefix);
        cache.lookup(&prefix);

        // Release one → ref_count = 2.
        cache.release(&prefix);

        // Entry should still exist (ref_count > 0).
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_prefix_cache_eviction_lru() {
        let mut cache = PrefixCache::new(2); // Capacity 2

        // Insert two entries.
        let p1: Vec<u32> = vec![1; BLOCK_SIZE];
        let p2: Vec<u32> = vec![2; BLOCK_SIZE];
        cache.insert(p1.clone(), vec![10]);
        // Release p1's ref so it's evictable.
        cache.release(&p1);

        cache.insert(p2.clone(), vec![20]);
        // Release p2's ref so it's evictable.
        cache.release(&p2);

        // Insert a third — should evict p1 (oldest, ref_count=0).
        let p3: Vec<u32> = vec![3; BLOCK_SIZE];
        let evicted = cache.insert(p3.clone(), vec![30]);
        assert_eq!(evicted, Some(vec![10])); // p1's blocks freed

        // p1 should be gone, p2 and p3 remain.
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(&p1).is_none());
    }

    #[test]
    fn test_prefix_cache_no_evict_if_ref_held() {
        let mut cache = PrefixCache::new(1); // Capacity 1

        // Insert and keep ref (ref_count = 1 from insert).
        let p1: Vec<u32> = vec![1; BLOCK_SIZE];
        cache.insert(p1.clone(), vec![10]);

        // Try to insert another — can't evict p1 (ref_count > 0).
        let p2: Vec<u32> = vec![2; BLOCK_SIZE];
        let evicted = cache.insert(p2.clone(), vec![20]);
        assert_eq!(evicted, None); // Nothing evictable.

        // Both entries now exist (over capacity, but can't evict).
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_link_prefix_skips_shared_blocks_on_free() {
        let (b, mut pool) = make_pool(8);
        let mut seq = pool.new_sequence(&b);

        // Simulate linking 2 prefix blocks (blocks 0, 1) then allocating 1 own block.
        let prefix_blocks = vec![0, 1];
        seq.link_prefix(&prefix_blocks, 32, vec![0; 32]);

        // Allocate one more block for the suffix.
        seq.ensure_slots(&mut pool, 16).unwrap();
        assert_eq!(seq.num_blocks(), 3); // 2 prefix + 1 own

        // Free the sequence — only the suffix block should be freed.
        let before = pool.free_block_count();
        pool.free_sequence(&seq);
        let after = pool.free_block_count();
        assert_eq!(after - before, 1); // Only 1 block freed (not the 2 prefix blocks)
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = PrefixCache::new(4);
        assert_eq!(cache.hit_rate(), 0.0);

        let prefix: Vec<u32> = (0..16).collect();
        cache.insert(prefix.clone(), vec![5]);

        // 1 miss (short prompt), then 2 hits
        let short: Vec<u32> = vec![99; 5]; // too short for any block-aligned match
        cache.lookup(&short); // miss
        cache.lookup(&prefix); // hit
        cache.lookup(&prefix); // hit

        // 2 hits / (2 hits + 1 miss) = 2/3
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-10);
    }
}
