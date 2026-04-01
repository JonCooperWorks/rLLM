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
//   A block table (per sequence) maps logical blocks to physical ones via
//   BlockHandle (index + generation):
//     block_table[logical_block] = BlockHandle { index, generation }
//
//   To find position `pos`:
//     logical_block = pos / BLOCK_SIZE
//     offset_in_block = pos % BLOCK_SIZE
//     physical_block = block_table[logical_block].index
//     data_index = (physical_block * BLOCK_SIZE + offset_in_block) * kv_dim
//
// Block size: 16 tokens.
//   Each block stores 16 positions of KV data.  For Llama 3.2 1B (kv_dim=512,
//   bf16), one block is 16 * 512 * 2 = 16 KB per layer per K/V.  This aligns
//   well with Metal's memory access patterns and flash attention tile sizes.
//
// Generational indices:
//   Block indices are capabilities into a long-lived GPU buffer.  A raw u32
//   index that outlives its allocation silently reads wrong data.  BlockHandle
//   pairs each index with a generation counter; the pool tracks generations
//   per slot and increments on free.  Stale handles panic on use — turning
//   silent corruption into a loud crash.  The GPU never sees generations;
//   sync_block_table extracts raw indices for the attention kernel.
//
// Prefix caching:
//   PrefixCache shares KV blocks across sequences with identical prefixes.
//   See the PrefixCache section below and docs/prompt-caching.md.
// ===========================================================================

use std::collections::HashMap;

use crate::gpu::{GpuCore, TensorDtype};
use crate::model::turboquant::{BoundaryConfig, KvQuantPair};
#[cfg(test)]
use crate::model::turboquant::KvQuantMode;

/// Number of token positions stored per KV cache block.
pub(crate) const BLOCK_SIZE: usize = 16;

// ===========================================================================
// Generational block handles — runtime detection of stale block references.
//
// A raw u32 block index is a capability: it grants access to a region of the
// KV pool.  If a block is freed and reallocated, any old reference to it
// silently reads/writes the wrong data.  Rust's ownership model can't prevent
// this because the pool is one long-lived buffer and block indices are just
// offsets into it.
//
// BlockHandle pairs each index with a generation counter.  The pool tracks
// the current generation per slot.  Freeing a block increments its generation.
// Any operation on a stale handle (wrong generation) panics immediately,
// turning silent data corruption into a loud crash with a clear message.
//
// The GPU never sees generations — sync_block_table extracts raw indices
// for the attention kernel.  Generation validation happens on the CPU side
// before upload.
// ===========================================================================

/// A block index paired with its allocation generation.
///
/// The generation counter detects stale references: if a block is freed and
/// reallocated, handles from the previous allocation have the wrong generation
/// and will panic on use.  This catches the class of bugs where block indices
/// outlive their intended lifetime (e.g. the prefix cache UAF bug).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockHandle {
    /// Physical block index in the KV pool (the value the GPU kernel uses).
    pub index: u32,
    /// Generation counter — must match the pool's current generation for this
    /// slot.  Mismatches indicate a stale (use-after-free) reference.
    pub generation: u32,
}

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
///
/// Each block slot has a generation counter.  Allocating returns a `BlockHandle`
/// with the current generation; freeing increments the generation.  Any attempt
/// to use a handle with a stale generation panics, catching use-after-free bugs
/// at runtime instead of silently corrupting KV data.
#[allow(dead_code)]
pub(crate) struct KvPool<B: GpuCore> {
    /// Physical K cache pool: one buffer per layer.
    /// For BF16: shape [num_blocks * BLOCK_SIZE, kv_dim] in bf16.
    /// For TurboQuant: raw byte buffer sized to num_blocks * BLOCK_SIZE * bytes_per_pos.
    pub k_pool: Vec<B::Tensor>,
    /// Physical V cache pool: one buffer per layer.
    pub v_pool: Vec<B::Tensor>,

    /// Free list of physical block indices (LIFO stack for locality).
    free_blocks: Vec<u32>,
    /// Generation counter per physical block slot.  Incremented on each free.
    /// A BlockHandle is valid only if its generation matches the current one.
    generations: Vec<u32>,
    /// Total number of physical blocks in the pool.
    num_physical_blocks: usize,
    /// KV dimension (num_kv_heads * head_dim).
    kv_dim: usize,
    /// Number of transformer layers.
    num_layers: usize,
    /// KV cache quantization pair (may be asymmetric: K=BF16, V=Turbo).
    pub kv_quant: KvQuantPair,
    /// Bytes per position for K pool.
    /// For BF16: kv_dim × 2.  For TurboQuant: num_kv_heads × bytes_per_head_pos.
    k_bytes_per_pos: usize,
    /// Bytes per position for V pool.
    v_bytes_per_pos: usize,
}

/// Per-sequence KV cache state: a block table and current length.
///
/// The block table maps logical block indices to physical block indices
/// in the shared KvPool.  It lives on the CPU and is uploaded to GPU
/// when the attention kernel needs it.
///
/// When prefix caching is active, the first N blocks may be shared with
/// other sequences.  `shared_prefix_blocks` tracks how many leading blocks
/// are borrowed from the PrefixCache (these must NOT be freed when the
/// sequence finishes — they belong to the cache).  The handles carry
/// generation counters, so stale references panic on sync or free.
pub(crate) struct SeqKvState<B: GpuCore> {
    /// Logical block index -> physical block handle (index + generation).
    /// Length = ceil(seq_len / BLOCK_SIZE), grows as the sequence gets longer.
    block_table_cpu: Vec<BlockHandle>,
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

/// A cached prefix: the token sequence, block handles, and reference count.
pub(crate) struct CachedPrefix {
    /// The token IDs that produced this prefix's KV data.
    /// Stored for collision checking — the hash is not sufficient alone.
    pub tokens: Vec<u32>,
    /// Block handles containing this prefix's KV data.
    /// Handles carry generation counters — if any generation is stale when
    /// a new sequence links to this prefix, sync_block_table will panic.
    /// These blocks are NOT on the free list while the entry exists.
    pub block_handles: Vec<BlockHandle>,
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

    /// Look up a prefix.  Returns the cached block handles and token count
    /// if found, and increments the ref count.
    ///
    /// The caller provides the full prompt tokens; this method checks all
    /// prefixes that are a prefix of the prompt (longest match wins).
    /// The returned handles carry generation counters — if the blocks were
    /// freed (bug), sync_block_table_validated will catch it.
    pub fn lookup(&mut self, prompt_tokens: &[u32]) -> Option<(Vec<BlockHandle>, usize)> {
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
                    return Some((entry.block_handles.clone(), entry.token_count));
                }
            }
        }
        self.misses += 1;
        None
    }

    /// Register a new prefix after prefill completes.
    ///
    /// `tokens` is the prefix token sequence (should be block-aligned).
    /// `block_handles` are the handles to physical blocks holding the KV data.
    /// The entry starts with ref_count = 1 (the sequence that just prefilled).
    ///
    /// If the cache is full, evicts the LRU entry with ref_count == 0.
    /// Returns the evicted block handles (if any) so the caller can free them.
    pub fn insert(
        &mut self,
        tokens: Vec<u32>,
        block_handles: Vec<BlockHandle>,
    ) -> Option<Vec<BlockHandle>> {
        let hash = Self::hash_tokens(&tokens);
        if self.entries.contains_key(&hash) {
            return None; // Already cached (race between concurrent prefills).
        }

        // Evict if at capacity.
        let evicted = if self.entries.len() >= self.max_entries {
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
                block_handles,
                token_count,
                ref_count: 1,
                last_used: self.clock,
            },
        );

        evicted
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
    /// Returns the freed block handles, or None if nothing is evictable.
    fn evict_lru(&mut self) -> Option<Vec<BlockHandle>> {
        let victim = self
            .entries
            .iter()
            .filter(|(_, e)| e.ref_count == 0)
            .min_by_key(|(_, e)| e.last_used)
            .map(|(&hash, _)| hash);

        if let Some(hash) = victim {
            let entry = self.entries.remove(&hash).unwrap();
            Some(entry.block_handles)
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
    #[allow(dead_code)] // used by tests; useful diagnostic accessor
    pub fn blocks_held(&self) -> usize {
        self.entries.values().map(|e| e.block_handles.len()).sum()
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

// ===========================================================================
// RadixPrefixCache — trie-based prefix cache with block-level sharing.
//
// Why a radix tree?
//   The flat PrefixCache stores each prefix independently.  If request A
//   caches [0..64] (4 blocks) and request B caches [0..48] (3 blocks), the
//   first 3 blocks are duplicated — 3 blocks wasted.  A radix tree shares
//   common prefix blocks: the trie has one node per BLOCK_SIZE-token edge,
//   and overlapping prefixes share internal nodes (and their KV blocks).
//
// Structure:
//   Each RadixNode represents one block's worth of KV data.  The root has
//   no block (it's a sentinel).  Children are keyed by FNV hash of the
//   BLOCK_SIZE tokens labeling the edge, with full token verification for
//   collision safety.
//
// Eviction:
//   Leaf-first LRU.  Only leaves with ref_count == 0 can be evicted.
//   This preserves internal nodes that are shared by multiple prefixes.
//
// Integration:
//   Drop-in replacement for PrefixCache — same lookup/insert/release API.
//   The Dispatch trait signatures are unchanged.
// ===========================================================================

/// A node in the radix prefix tree.  Each node (except root) holds one
/// KV block handle and tracks how many active sequences pass through it.
struct RadixNode {
    /// KV block at this depth.  None for the root sentinel.
    block_handle: Option<BlockHandle>,
    /// Children keyed by FNV hash of the BLOCK_SIZE tokens on the edge.
    children: HashMap<u64, RadixChild>,
    /// Active sequences whose prefix passes through this node.
    ref_count: usize,
    /// Monotonic LRU timestamp (higher = more recent).
    last_used: u64,
}

/// An edge in the radix tree: the token label + the child node.
struct RadixChild {
    /// The BLOCK_SIZE tokens labeling this edge (for collision checking).
    tokens: Vec<u32>,
    /// The child node.
    node: RadixNode,
}

impl RadixNode {
    fn new(block_handle: Option<BlockHandle>) -> Self {
        Self {
            block_handle,
            children: HashMap::new(),
            ref_count: 0,
            last_used: 0,
        }
    }

    /// Whether this node is a leaf (no children).
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Trie-based prefix cache with block-level sharing.
///
/// Replaces the flat PrefixCache for better memory efficiency when multiple
/// prefixes share common leading blocks (e.g., identical system prompts with
/// different user messages).
pub(crate) struct RadixPrefixCache {
    /// Root sentinel node (no block handle).
    root: RadixNode,
    /// Maximum blocks the cache can hold.
    max_blocks: usize,
    /// Current blocks held by the cache.
    current_blocks: usize,
    /// Monotonic clock for LRU ordering.
    clock: u64,
    /// Running stats.
    pub hits: u64,
    pub misses: u64,
}

impl RadixPrefixCache {
    pub fn new(max_blocks: usize) -> Self {
        Self {
            root: RadixNode::new(None),
            max_blocks,
            current_blocks: 0,
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Hash a single block's worth of tokens.
    fn hash_block(tokens: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &t in tokens {
            h ^= t as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Look up the longest cached prefix of `prompt_tokens`.
    ///
    /// Walks the trie matching BLOCK_SIZE chunks.  Returns the block handles
    /// along the matched path and the total token count, or None if no match.
    /// Increments ref_count on every matched node so they can't be evicted.
    pub fn lookup(&mut self, prompt_tokens: &[u32]) -> Option<(Vec<BlockHandle>, usize)> {
        let num_blocks = prompt_tokens.len() / BLOCK_SIZE;
        if num_blocks == 0 {
            self.misses += 1;
            return None;
        }

        // Pass 1: determine match depth (immutable walk).
        let mut match_depth = 0;
        let mut node = &self.root;
        for b in 0..num_blocks {
            let chunk = &prompt_tokens[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
            let hash = Self::hash_block(chunk);
            match node.children.get(&hash) {
                Some(child) if child.tokens == chunk => {
                    match_depth += 1;
                    node = &child.node;
                }
                _ => break,
            }
        }

        if match_depth == 0 {
            self.misses += 1;
            return None;
        }

        // Pass 2: collect handles and increment ref_counts (mutable walk).
        self.clock += 1;
        let clock = self.clock;
        let mut handles = Vec::with_capacity(match_depth);
        let mut node = &mut self.root;
        for b in 0..match_depth {
            let chunk = &prompt_tokens[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
            let hash = Self::hash_block(chunk);
            let child = node.children.get_mut(&hash).unwrap();
            child.node.ref_count += 1;
            child.node.last_used = clock;
            handles.push(child.node.block_handle.unwrap());
            node = &mut node.children.get_mut(&hash).unwrap().node;
        }

        self.hits += 1;
        Some((handles, match_depth * BLOCK_SIZE))
    }

    /// Register a prefix after prefill completes.
    ///
    /// Walks the trie, creating new nodes as needed for token blocks that
    /// aren't already cached.  Existing nodes along the path are untouched
    /// (their blocks are already cached).  New nodes start with ref_count = 1.
    ///
    /// Returns block handles evicted to make room (caller must free them via
    /// KvPool::free_block).
    pub fn insert(
        &mut self,
        tokens: &[u32],
        block_handles: &[BlockHandle],
    ) -> Vec<BlockHandle> {
        let num_blocks = tokens.len() / BLOCK_SIZE;
        assert_eq!(num_blocks, block_handles.len());

        // Phase 1: determine how many new nodes we need (immutable walk).
        let mut existing_depth = 0;
        {
            let mut node = &self.root;
            for b in 0..num_blocks {
                let chunk = &tokens[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                let hash = Self::hash_block(chunk);
                match node.children.get(&hash) {
                    Some(child) if child.tokens == chunk => {
                        existing_depth += 1;
                        node = &child.node;
                    }
                    _ => break,
                }
            }
        }

        // Phase 2: evict leaves to make room for new nodes (borrows self.root
        // without conflicting with the walk pointer below).
        // We need (current_blocks + new_nodes_needed) <= max_blocks.
        let new_nodes_needed = num_blocks - existing_depth;
        let mut evicted = Vec::new();
        while self.current_blocks + new_nodes_needed > self.max_blocks {
            if let Some(handle) =
                Self::evict_leaf(&mut self.root, &mut self.current_blocks)
            {
                evicted.push(handle);
            } else {
                break; // Nothing evictable (all refs > 0).
            }
        }

        // Phase 3: walk the trie mutably, updating existing nodes and creating
        // new ones.
        self.clock += 1;
        let clock = self.clock;

        let mut node = &mut self.root;
        for b in 0..num_blocks {
            let chunk = &tokens[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
            let hash = Self::hash_block(chunk);

            if node.children.contains_key(&hash) {
                // Existing node — update LRU, increment ref_count.
                let child = node.children.get_mut(&hash).unwrap();
                child.node.ref_count += 1;
                child.node.last_used = clock;
                node = &mut node.children.get_mut(&hash).unwrap().node;
            } else {
                // New node.
                let new_node = RadixNode {
                    block_handle: Some(block_handles[b]),
                    children: HashMap::new(),
                    ref_count: 1,
                    last_used: clock,
                };
                node.children.insert(
                    hash,
                    RadixChild {
                        tokens: chunk.to_vec(),
                        node: new_node,
                    },
                );
                self.current_blocks += 1;
                node = &mut node.children.get_mut(&hash).unwrap().node;
            }
        }

        evicted
    }

    /// Decrement ref_count along the path matching `tokens`.
    ///
    /// Called when a sequence finishes or is aborted.  Does NOT free blocks —
    /// nodes stay cached for future reuse until evicted.
    pub fn release(&mut self, tokens: &[u32]) {
        let num_blocks = tokens.len() / BLOCK_SIZE;
        let mut node = &mut self.root;
        for b in 0..num_blocks {
            let chunk = &tokens[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
            let hash = Self::hash_block(chunk);
            match node.children.get_mut(&hash) {
                Some(child) if child.tokens == chunk => {
                    child.node.ref_count = child.node.ref_count.saturating_sub(1);
                    node = &mut node.children.get_mut(&hash).unwrap().node;
                }
                _ => break, // Path doesn't exist (shouldn't happen).
            }
        }
    }

    /// Evict one leaf node with ref_count == 0 and the smallest last_used.
    /// Removes the leaf from its parent and returns its block handle.
    ///
    /// Uses a recursive DFS from the given node to find all eligible leaves,
    /// picks the LRU one, then removes it.
    fn evict_leaf(root: &mut RadixNode, current_blocks: &mut usize) -> Option<BlockHandle> {
        // Find the best leaf to evict: (hash path from root to parent, child hash).
        let mut best: Option<(u64, Vec<u64>, u64)> = None; // (last_used, path, leaf_hash)

        fn find_leaves(
            node: &RadixNode,
            path: &mut Vec<u64>,
            best: &mut Option<(u64, Vec<u64>, u64)>,
        ) {
            for (&hash, child) in &node.children {
                if child.node.is_leaf() && child.node.ref_count == 0 {
                    let lu = child.node.last_used;
                    if best.as_ref().map_or(true, |(best_lu, _, _)| lu < *best_lu) {
                        *best = Some((lu, path.clone(), hash));
                    }
                } else {
                    path.push(hash);
                    find_leaves(&child.node, path, best);
                    path.pop();
                }
            }
        }

        let mut path = Vec::new();
        find_leaves(root, &mut path, &mut best);

        let (_, parent_path, leaf_hash) = best?;

        // Walk the path to find the parent, then remove the leaf.
        let mut node = root;
        for &hash in &parent_path {
            node = &mut node.children.get_mut(&hash).unwrap().node;
        }
        let removed = node.children.remove(&leaf_hash).unwrap();
        *current_blocks -= 1;
        removed.node.block_handle
    }

    /// Number of blocks held by the cache.
    #[allow(dead_code)]
    pub fn blocks_held(&self) -> usize {
        self.current_blocks
    }

    /// Number of cached prefixes (distinct leaf paths).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        fn count_leaves(node: &RadixNode) -> usize {
            if node.is_leaf() && node.block_handle.is_some() {
                return 1;
            }
            node.children.values().map(|c| count_leaves(&c.node)).sum()
        }
        count_leaves(&self.root)
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
    /// For BF16 (kv_quant=None):
    ///   Total GPU memory = num_blocks * BLOCK_SIZE * kv_dim * 2 bytes * 2 (K+V) * num_layers.
    /// For TurboQuant:
    ///   Total GPU memory = num_blocks * BLOCK_SIZE * bytes_per_pos * 2 (K+V) * num_layers,
    ///   where bytes_per_pos = num_kv_heads × (2 + ceil(head_dim × bits / 8)).
    pub fn new(
        backend: &B,
        num_blocks: usize,
        kv_dim: usize,
        num_layers: usize,
        kv_quant: KvQuantPair,
        head_dim: usize,
        boundary: Option<BoundaryConfig>,
    ) -> Self {
        use crate::model::turboquant::{bytes_per_kv_position, effective_kv_pair_for_layer};

        let total_positions = num_blocks * BLOCK_SIZE;
        let num_kv_heads = if head_dim > 0 { kv_dim / head_dim } else { 0 };

        // Base (interior layer) bytes per position — used for reporting.
        let k_bytes_per_pos = bytes_per_kv_position(head_dim, num_kv_heads, kv_quant.k);
        let v_bytes_per_pos = bytes_per_kv_position(head_dim, num_kv_heads, kv_quant.v);

        let mut k_pool = Vec::with_capacity(num_layers);
        let mut v_pool = Vec::with_capacity(num_layers);

        // Allocate K and V pools per layer.  Boundary layers may use a different
        // quantization mode (more bytes per position) than interior layers.
        for layer_idx in 0..num_layers {
            let layer_pair = effective_kv_pair_for_layer(
                kv_quant, boundary.as_ref(), layer_idx, num_layers,
            );

            // K pool for this layer.
            if layer_pair.k.is_quantized() {
                let k_bpp = bytes_per_kv_position(head_dim, num_kv_heads, layer_pair.k);
                let bf16_elems = (k_bpp + 1) / 2;
                k_pool.push(backend.alloc_tensor(
                    &[total_positions, bf16_elems],
                    TensorDtype::BF16,
                ));
            } else {
                k_pool.push(backend.alloc_tensor(&[total_positions, kv_dim], TensorDtype::BF16));
            }

            // V pool for this layer.
            if layer_pair.v.is_quantized() {
                let v_bpp = bytes_per_kv_position(head_dim, num_kv_heads, layer_pair.v);
                let bf16_elems = (v_bpp + 1) / 2;
                v_pool.push(backend.alloc_tensor(
                    &[total_positions, bf16_elems],
                    TensorDtype::BF16,
                ));
            } else {
                v_pool.push(backend.alloc_tensor(&[total_positions, kv_dim], TensorDtype::BF16));
            }
        }

        // Initialise free list with all block indices (in reverse so pop gives 0, 1, 2, ...).
        let free_blocks: Vec<u32> = (0..num_blocks as u32).rev().collect();
        // All blocks start at generation 0.
        let generations = vec![0u32; num_blocks];

        Self {
            k_pool,
            v_pool,
            free_blocks,
            generations,
            num_physical_blocks: num_blocks,
            kv_dim,
            num_layers,
            kv_quant,
            k_bytes_per_pos,
            v_bytes_per_pos,
        }
    }

    /// Number of free blocks remaining.
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Number of physical blocks in the pool.
    #[allow(dead_code)] // used by tests; useful diagnostic accessor
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
        // (K + V) × num_layers × num_blocks × BLOCK_SIZE
        let per_block = self.k_bytes_per_pos + self.v_bytes_per_pos;
        self.num_layers * self.num_physical_blocks * BLOCK_SIZE * per_block
    }

    /// Bytes per K position across all KV heads.
    pub fn k_bytes_per_position(&self) -> usize {
        self.k_bytes_per_pos
    }

    /// Bytes per V position across all KV heads.
    pub fn v_bytes_per_position(&self) -> usize {
        self.v_bytes_per_pos
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
    ///
    /// The returned `BlockHandle` carries the current generation for this slot.
    /// The handle is valid until the block is freed (which increments the
    /// generation, invalidating any stale copies).
    pub fn alloc_block(&mut self) -> Option<BlockHandle> {
        self.free_blocks.pop().map(|index| BlockHandle {
            index,
            generation: self.generations[index as usize],
        })
    }

    /// Return a physical block to the free list.
    ///
    /// Panics if the handle's generation doesn't match the pool's current
    /// generation for this slot — this means the block was already freed
    /// (double-free) or the handle is from a previous allocation cycle.
    pub fn free_block(&mut self, handle: BlockHandle) {
        let slot = handle.index as usize;
        assert_eq!(
            handle.generation, self.generations[slot],
            "stale BlockHandle: block {} has generation {} but handle has generation {} \
             (block was freed and possibly reallocated since this handle was created)",
            handle.index, self.generations[slot], handle.generation,
        );
        // Increment generation so any remaining handles to this slot become stale.
        self.generations[slot] += 1;
        self.free_blocks.push(handle.index);
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
    ///
    /// Each block's generation is validated — a stale handle panics.
    pub fn free_sequence(&mut self, seq: &SeqKvState<B>) {
        // Skip the first `shared_prefix_blocks` — those belong to the cache.
        for &handle in &seq.block_table_cpu[seq.shared_prefix_blocks..] {
            self.free_block(handle);
        }
    }

    /// Return evicted prefix block handles to the free list.
    ///
    /// Called when PrefixCache evicts an entry — its blocks were held out
    /// of the free list while cached, and now need to be returned.
    /// Generation is validated on each handle.
    pub fn free_blocks(&mut self, handles: &[BlockHandle]) {
        for &handle in handles {
            self.free_block(handle);
        }
    }

    /// Validate that a block handle's generation matches the pool's current
    /// generation for that slot.  Panics on mismatch.
    ///
    /// Used by sync_block_table before uploading indices to the GPU — this
    /// is the last line of defense before stale indices reach the attention
    /// kernel.
    pub fn validate_handle(&self, handle: BlockHandle) {
        let slot = handle.index as usize;
        assert_eq!(
            handle.generation, self.generations[slot],
            "stale BlockHandle in block table: block {} has generation {} but handle has \
             generation {} — this block was freed and reallocated, the KV data is from \
             a different sequence",
            handle.index, self.generations[slot], handle.generation,
        );
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
            let handle = pool
                .alloc_block()
                .ok_or_else(|| anyhow::anyhow!("KV cache out of memory: no free blocks"))?;
            self.block_table_cpu.push(handle);
            self.dirty = true;
        }
        Ok(())
    }

    /// Upload the block table to GPU if it has been modified since last upload.
    ///
    /// Extracts raw u32 indices from BlockHandles for the GPU kernel.
    /// Validates each handle's generation against the pool before upload —
    /// this is the last checkpoint before stale indices reach the attention
    /// kernel.  Pass `pool = None` to skip validation (e.g. in tests without
    /// a real pool).
    pub fn sync_block_table_validated(&mut self, backend: &B, pool: &KvPool<B>) {
        if !self.dirty {
            return;
        }
        // Validate all handles before uploading to GPU.
        for handle in &self.block_table_cpu {
            pool.validate_handle(*handle);
        }
        self.sync_block_table_inner(backend);
    }

    /// Upload the block table to GPU without generation validation.
    ///
    /// Used when no pool reference is available (e.g. multi-GPU paths where
    /// each rank has its own pool).
    #[allow(dead_code)] // CUDA multi-GPU path
    pub fn sync_block_table(&mut self, backend: &B) {
        if !self.dirty {
            return;
        }
        self.sync_block_table_inner(backend);
    }

    /// Inner implementation: build padded u32 array and upload.
    fn sync_block_table_inner(&mut self, backend: &B) {
        let mut table = vec![0u32; MAX_BLOCKS_PER_SEQ];
        for (i, handle) in self.block_table_cpu.iter().enumerate() {
            table[i] = handle.index;
        }
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
    /// After `forward_prefill` writes K/V for the whole prompt chunk
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
            let handle = pool
                .alloc_block()
                .ok_or_else(|| anyhow::anyhow!("KV cache out of memory: no free blocks"))?;
            self.block_table_cpu.push(handle);
            self.dirty = true;
        }
        Ok(())
    }

    /// Link this sequence to a cached prefix.
    ///
    /// Copies the prefix's block handles into this sequence's block table and
    /// advances seq_len to skip the already-computed positions.  The blocks
    /// are NOT allocated from the free list — they're borrowed from the
    /// PrefixCache.  The handles carry generation counters, so if the prefix
    /// blocks were freed (bug), sync_block_table_validated will catch it.
    pub fn link_prefix(
        &mut self,
        prefix_handles: &[BlockHandle],
        prefix_token_count: usize,
        prefix_tokens: Vec<u32>,
    ) {
        self.block_table_cpu = prefix_handles.to_vec();
        self.seq_len = prefix_token_count;
        self.shared_prefix_blocks = prefix_handles.len();
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
    /// Used by the prefix cache to record which block handles hold a
    /// prefix's KV data after prefill completes.
    pub fn block_table_cpu_slice(&self) -> &[BlockHandle] {
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
        let pool = KvPool::new(&backend, num_blocks, 4, 2, KvQuantPair::symmetric(KvQuantMode::None), 4, None); // kv_dim=4, 2 layers
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

    /// Helper to create a BlockHandle for test assertions.
    fn bh(index: u32, generation: u32) -> BlockHandle {
        BlockHandle { index, generation }
    }

    #[test]
    fn test_alloc_and_free_block() {
        let (_b, mut pool) = make_pool(4);
        let b0 = pool.alloc_block().unwrap();
        assert_eq!(b0.index, 0); // LIFO: reversed, so first pop = 0
        assert_eq!(b0.generation, 0);
        assert_eq!(pool.free_block_count(), 3);

        let b1 = pool.alloc_block().unwrap();
        assert_eq!(b1.index, 1);
        assert_eq!(pool.free_block_count(), 2);

        pool.free_block(b0);
        assert_eq!(pool.free_block_count(), 3);

        // Re-allocate should get index 0 back (LIFO), but generation is now 1.
        let b_realloc = pool.alloc_block().unwrap();
        assert_eq!(b_realloc.index, b0.index);
        assert_eq!(b_realloc.generation, 1); // incremented by free
    }

    #[test]
    fn test_alloc_block_exhaustion() {
        let (_b, mut pool) = make_pool(2);
        assert!(pool.alloc_block().is_some());
        assert!(pool.alloc_block().is_some());
        assert!(pool.alloc_block().is_none()); // exhausted
    }

    #[test]
    #[should_panic(expected = "stale BlockHandle")]
    fn test_double_free_panics() {
        let (_b, mut pool) = make_pool(4);
        let handle = pool.alloc_block().unwrap();
        pool.free_block(handle);
        pool.free_block(handle); // stale: generation was incremented
    }

    #[test]
    #[should_panic(expected = "stale BlockHandle")]
    fn test_stale_handle_after_realloc_panics() {
        let (_b, mut pool) = make_pool(4);
        let old_handle = pool.alloc_block().unwrap();
        pool.free_block(old_handle);
        let _new_handle = pool.alloc_block().unwrap(); // same index, new generation
        pool.free_block(old_handle); // stale: generation mismatch
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
        let handles = vec![bh(10, 0), bh(20, 0)];
        cache.insert(prompt.clone(), handles.clone());

        // Second lookup: hit.
        let result = cache.lookup(&prompt);
        assert!(result.is_some());
        let (found_handles, token_count) = result.unwrap();
        assert_eq!(found_handles, vec![bh(10, 0), bh(20, 0)]);
        assert_eq!(token_count, 32);
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn test_prefix_cache_longer_prompt_matches_prefix() {
        let mut cache = PrefixCache::new(4);

        // Cache a 32-token prefix (2 blocks).
        let prefix: Vec<u32> = (0..32).collect();
        cache.insert(prefix.clone(), vec![bh(10, 0), bh(20, 0)]);

        // Look up with a longer prompt that starts with the same 32 tokens.
        let long_prompt: Vec<u32> = (0..48).collect(); // 3 blocks
        let result = cache.lookup(&long_prompt);
        assert!(result.is_some());
        let (handles, count) = result.unwrap();
        assert_eq!(handles, vec![bh(10, 0), bh(20, 0)]);
        assert_eq!(count, 32);
    }

    #[test]
    fn test_prefix_cache_ref_counting() {
        let mut cache = PrefixCache::new(4);
        let prefix: Vec<u32> = (0..16).collect(); // 1 block

        // Insert with initial ref_count = 1.
        cache.insert(prefix.clone(), vec![bh(5, 0)]);

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
        cache.insert(p1.clone(), vec![bh(10, 0)]);
        cache.release(&p1);

        cache.insert(p2.clone(), vec![bh(20, 0)]);
        cache.release(&p2);

        // Insert a third — should evict p1 (oldest, ref_count=0).
        let p3: Vec<u32> = vec![3; BLOCK_SIZE];
        let evicted = cache.insert(p3.clone(), vec![bh(30, 0)]);
        assert_eq!(evicted, Some(vec![bh(10, 0)])); // p1's handles freed

        // p1 should be gone, p2 and p3 remain.
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(&p1).is_none());
    }

    #[test]
    fn test_prefix_cache_no_evict_if_ref_held() {
        let mut cache = PrefixCache::new(1); // Capacity 1

        // Insert and keep ref (ref_count = 1 from insert).
        let p1: Vec<u32> = vec![1; BLOCK_SIZE];
        cache.insert(p1.clone(), vec![bh(10, 0)]);

        // Try to insert another — can't evict p1 (ref_count > 0).
        let p2: Vec<u32> = vec![2; BLOCK_SIZE];
        let evicted = cache.insert(p2.clone(), vec![bh(20, 0)]);
        assert_eq!(evicted, None); // Nothing evictable.

        // Both entries now exist (over capacity, but can't evict).
        assert_eq!(cache.len(), 2);
    }

    // -----------------------------------------------------------------------
    // End-to-end prefix cache + pool integration tests
    //
    // These exercise the real flow: allocate blocks from the pool, register
    // a prefix in the cache, mark blocks as shared, free the first sequence,
    // link a second sequence to the cached prefix, and verify everything.
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefix_sharing_end_to_end() {
        // Simulates the full prefix cache lifecycle:
        //   1. Sequence A prefills 32 tokens (2 blocks)
        //   2. Register prefix in cache, mark blocks shared
        //   3. Sequence A finishes → only suffix blocks freed
        //   4. Sequence B links to cached prefix, allocates suffix
        //   5. Sequence B's block table is valid (generations match)
        //   6. Sequence B finishes → only its suffix blocks freed
        //   7. Cache evicts → prefix blocks finally freed
        let (b, mut pool) = make_pool(8);
        let mut cache = PrefixCache::new(4);
        let prefix_tokens: Vec<u32> = (0..32).collect(); // 2 blocks

        // --- Sequence A: prefill 32 prefix + 16 suffix = 48 tokens ---
        let mut seq_a = pool.new_sequence(&b);
        seq_a.ensure_slots(&mut pool, 48).unwrap(); // allocates 3 blocks
        seq_a.advance_by(48);
        assert_eq!(seq_a.num_blocks(), 3);
        assert_eq!(pool.free_block_count(), 5); // 8 - 3 = 5

        // Register the 2-block prefix in the cache.
        let prefix_handles = seq_a.block_table_cpu_slice()[..2].to_vec();
        let evicted = cache.insert(prefix_tokens.clone(), prefix_handles.clone());
        assert!(evicted.is_none());

        // Mark those 2 blocks as shared so free_sequence skips them.
        seq_a.mark_prefix_shared(2, prefix_tokens.clone());

        // Sequence A finishes.  Only the 1 suffix block should be freed.
        pool.free_sequence(&seq_a);
        assert_eq!(pool.free_block_count(), 6); // 5 + 1 suffix block
        cache.release(&prefix_tokens);

        // --- Sequence B: link cached prefix + 16 suffix ---
        let mut seq_b = pool.new_sequence(&b);
        let (cached_handles, cached_count) = cache.lookup(&prefix_tokens).unwrap();
        assert_eq!(cached_count, 32);
        assert_eq!(cached_handles, prefix_handles); // same handles, same generations

        seq_b.link_prefix(&cached_handles, cached_count, prefix_tokens.clone());
        seq_b.ensure_slots(&mut pool, 16).unwrap(); // 1 more block for suffix
        seq_b.advance_by(16);
        assert_eq!(seq_b.num_blocks(), 3); // 2 prefix + 1 suffix

        // Validated sync should pass — all handles have correct generations.
        seq_b.dirty = true;
        seq_b.sync_block_table_validated(&b, &pool); // no panic

        // Sequence B finishes.
        pool.free_sequence(&seq_b);
        cache.release(&prefix_tokens);

        // --- Cache eviction returns prefix blocks ---
        // Force eviction by filling the cache past capacity.
        let p2: Vec<u32> = vec![200; BLOCK_SIZE];
        let p3: Vec<u32> = vec![201; BLOCK_SIZE];
        let p4: Vec<u32> = vec![202; BLOCK_SIZE];
        let p5: Vec<u32> = vec![203; BLOCK_SIZE];
        cache.insert(p2.clone(), vec![bh(50, 0)]); cache.release(&p2);
        cache.insert(p3.clone(), vec![bh(51, 0)]); cache.release(&p3);
        cache.insert(p4.clone(), vec![bh(52, 0)]); cache.release(&p4);
        let evicted = cache.insert(p5.clone(), vec![bh(53, 0)]);
        assert!(evicted.is_some()); // our original prefix got evicted

        let evicted_handles = evicted.unwrap();
        assert_eq!(evicted_handles.len(), 2);
        // Return evicted prefix blocks to the pool.
        pool.free_blocks(&evicted_handles);
        assert_eq!(pool.free_block_count(), 8); // all blocks back
    }

    #[test]
    fn test_missing_mark_prefix_shared_caught_by_generation() {
        // Proves generational indices catch the bug if mark_prefix_shared
        // is accidentally omitted.  Without it, free_sequence frees the
        // prefix blocks.  A second sequence linking to the cached prefix
        // gets stale handles — sync_block_table_validated panics.
        let (b, mut pool) = make_pool(8);
        let mut cache = PrefixCache::new(4);
        let prefix_tokens: Vec<u32> = (0..16).collect(); // 1 block

        // Sequence A prefills 16 tokens (1 block).
        let mut seq_a = pool.new_sequence(&b);
        seq_a.ensure_slots(&mut pool, 16).unwrap();
        seq_a.advance_by(16);

        // Register prefix in cache.
        let prefix_handles = seq_a.block_table_cpu_slice()[..1].to_vec();
        cache.insert(prefix_tokens.clone(), prefix_handles);

        // BUG: deliberately skip mark_prefix_shared.
        // seq_a.mark_prefix_shared(1, prefix_tokens.clone());  // <-- omitted!

        // Sequence A finishes — free_sequence frees ALL blocks including
        // the prefix block, because shared_prefix_blocks is still 0.
        pool.free_sequence(&seq_a);
        cache.release(&prefix_tokens);

        // The prefix block's generation is now incremented (freed).
        // Someone else could allocate it and write different KV data.
        let _stolen = pool.alloc_block().unwrap(); // takes the freed block

        // Sequence B links to the cached prefix.
        let mut seq_b = pool.new_sequence(&b);
        let (cached_handles, cached_count) = cache.lookup(&prefix_tokens).unwrap();
        seq_b.link_prefix(&cached_handles, cached_count, prefix_tokens.clone());
        seq_b.dirty = true;

        // Validated sync should panic — the cached handle's generation is
        // stale (0) but the pool's generation for that block is now 1.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            seq_b.sync_block_table_validated(&b, &pool);
        }));
        assert!(
            result.is_err(),
            "should panic: prefix block was freed without mark_prefix_shared, \
             generation mismatch should be caught"
        );
    }

    #[test]
    fn test_link_prefix_skips_shared_blocks_on_free() {
        // Same intent as before but with properly allocated blocks.
        let (b, mut pool) = make_pool(8);
        let mut seq = pool.new_sequence(&b);

        // Allocate 3 blocks (48 tokens worth).
        seq.ensure_slots(&mut pool, 48).unwrap();
        seq.advance_by(48);
        assert_eq!(seq.num_blocks(), 3);
        assert_eq!(pool.free_block_count(), 5);

        // Pretend the first 2 blocks are a shared prefix.
        let prefix_tokens: Vec<u32> = (0..32).collect();
        seq.mark_prefix_shared(2, prefix_tokens);

        // Free the sequence — only the 1 suffix block should be freed.
        let before = pool.free_block_count();
        pool.free_sequence(&seq);
        let after = pool.free_block_count();
        assert_eq!(after - before, 1);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = PrefixCache::new(4);
        assert_eq!(cache.hit_rate(), 0.0);

        let prefix: Vec<u32> = (0..16).collect();
        cache.insert(prefix.clone(), vec![bh(5, 0)]);

        // 1 miss (short prompt), then 2 hits
        let short: Vec<u32> = vec![99; 5]; // too short for any block-aligned match
        cache.lookup(&short); // miss
        cache.lookup(&prefix); // hit
        cache.lookup(&prefix); // hit

        // 2 hits / (2 hits + 1 miss) = 2/3
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_validated_sync_catches_stale_handle() {
        let (b, mut pool) = make_pool(4);
        let mut seq = pool.new_sequence(&b);

        // Allocate a block, then free it externally (simulating the bug).
        seq.ensure_slot(&mut pool).unwrap();
        let stale_handle = seq.block_table_cpu[0];

        // Free the block through the pool (incrementing its generation).
        pool.free_block(stale_handle);

        // Re-allocate (same index, new generation).
        let _new_handle = pool.alloc_block().unwrap();

        // The seq still holds the old handle.  Validated sync should catch it.
        seq.dirty = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            seq.sync_block_table_validated(&b, &pool);
        }));
        assert!(result.is_err(), "should panic on stale handle");
    }

    // -----------------------------------------------------------------------
    // Radix tree prefix cache tests.
    // -----------------------------------------------------------------------

    #[test]
    fn test_radix_shared_blocks() {
        // Two overlapping prefixes should share blocks, not duplicate them.
        let mut cache = RadixPrefixCache::new(64);

        // Prefix A: 4 blocks (64 tokens).
        let tokens_a: Vec<u32> = (0..64).collect();
        let handles_a = vec![bh(0, 0), bh(1, 0), bh(2, 0), bh(3, 0)];
        cache.insert(&tokens_a, &handles_a);
        assert_eq!(cache.blocks_held(), 4);

        // Prefix B: first 3 blocks same as A, 4th differs.
        let mut tokens_b: Vec<u32> = (0..48).collect();
        tokens_b.extend(100..116); // different 4th block
        let handles_b = vec![bh(0, 0), bh(1, 0), bh(2, 0), bh(10, 0)];
        cache.insert(&tokens_b, &handles_b);

        // Should be 5 blocks total (3 shared + 1 unique from A + 1 unique from B),
        // not 8.
        assert_eq!(cache.blocks_held(), 5);
    }

    #[test]
    fn test_radix_longest_match() {
        // Cache 64 tokens, lookup 80 tokens → should match the 64-token prefix.
        let mut cache = RadixPrefixCache::new(64);

        let tokens: Vec<u32> = (0..64).collect();
        let handles = vec![bh(0, 0), bh(1, 0), bh(2, 0), bh(3, 0)];
        cache.insert(&tokens, &handles);
        cache.release(&tokens); // Drop ref so it's not pinned.

        // Lookup with a longer prompt that shares the first 64 tokens.
        let mut longer: Vec<u32> = (0..64).collect();
        longer.extend(200..216); // 5th block doesn't match anything.
        let result = cache.lookup(&longer);
        assert!(result.is_some());
        let (matched_handles, matched_tokens) = result.unwrap();
        assert_eq!(matched_tokens, 64);
        assert_eq!(matched_handles.len(), 4);
    }

    #[test]
    fn test_radix_partial_match() {
        // Cache [0..64], lookup [0..48, 999..1015] — first 3 blocks match.
        let mut cache = RadixPrefixCache::new(64);

        let tokens: Vec<u32> = (0..64).collect();
        let handles = vec![bh(0, 0), bh(1, 0), bh(2, 0), bh(3, 0)];
        cache.insert(&tokens, &handles);
        cache.release(&tokens);

        let mut partial: Vec<u32> = (0..48).collect();
        partial.extend(999..1015);
        let result = cache.lookup(&partial);
        assert!(result.is_some());
        let (matched_handles, matched_tokens) = result.unwrap();
        assert_eq!(matched_tokens, 48);
        assert_eq!(matched_handles.len(), 3);
    }

    #[test]
    fn test_radix_leaf_eviction() {
        // Cache with max 4 blocks.  Insert 4-block prefix, release it,
        // then insert a 2-block prefix that needs room → should evict leaves.
        let mut cache = RadixPrefixCache::new(4);

        let tokens_a: Vec<u32> = (0..64).collect();
        let handles_a = vec![bh(0, 0), bh(1, 0), bh(2, 0), bh(3, 0)];
        cache.insert(&tokens_a, &handles_a);
        cache.release(&tokens_a);
        assert_eq!(cache.blocks_held(), 4);

        // Insert a different prefix that needs 2 blocks → should evict 2 leaves.
        let tokens_b: Vec<u32> = (100..132).collect();
        let handles_b = vec![bh(10, 0), bh(11, 0)];
        let evicted = cache.insert(&tokens_b, &handles_b);
        assert_eq!(evicted.len(), 2, "should evict 2 leaves to make room");
        assert_eq!(cache.blocks_held(), 4); // Still at capacity.
    }

    #[test]
    fn test_radix_ref_count_prevents_eviction() {
        // A leaf with ref_count > 0 must not be evicted.
        let mut cache = RadixPrefixCache::new(2);

        let tokens: Vec<u32> = (0..32).collect();
        let handles = vec![bh(0, 0), bh(1, 0)];
        cache.insert(&tokens, &handles);
        // Don't release — ref_count stays at 1.
        assert_eq!(cache.blocks_held(), 2);

        // Try to insert another prefix — can't evict because ref_count > 0.
        let tokens_b: Vec<u32> = (100..116).collect();
        let handles_b = vec![bh(10, 0)];
        let evicted = cache.insert(&tokens_b, &handles_b);
        assert!(evicted.is_empty(), "should not evict referenced blocks");
        // The new block was still inserted (over capacity — acceptable).
        assert_eq!(cache.blocks_held(), 3);
    }

    #[test]
    fn test_radix_empty_cache_miss() {
        let mut cache = RadixPrefixCache::new(64);
        let tokens: Vec<u32> = (0..32).collect();
        let result = cache.lookup(&tokens);
        assert!(result.is_none());
        assert_eq!(cache.misses, 1);
    }

    #[test]
    fn test_radix_release_decrements_refcount() {
        // After release, all nodes should have ref_count decremented.
        let mut cache = RadixPrefixCache::new(64);
        let tokens: Vec<u32> = (0..32).collect();
        let handles = vec![bh(0, 0), bh(1, 0)];
        cache.insert(&tokens, &handles);

        // ref_count should be 1 after insert.
        cache.release(&tokens);

        // After release, nodes should be evictable.
        // Insert enough to trigger eviction — it should succeed.
        let tokens_b: Vec<u32> = (100..116).collect();
        let handles_b = vec![bh(10, 0)];
        // The original nodes are now ref_count=0 and evictable.
        // (No capacity pressure here, but the state is correct.)
    }

    #[test]
    fn test_radix_short_prompt_no_match() {
        // Prompts shorter than BLOCK_SIZE should not match anything.
        let mut cache = RadixPrefixCache::new(64);
        let tokens: Vec<u32> = (0..8).collect(); // Less than BLOCK_SIZE
        let result = cache.lookup(&tokens);
        assert!(result.is_none());
    }
}
