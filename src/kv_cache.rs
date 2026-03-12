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

use crate::gpu::{GpuBackend, TensorDtype};

/// Number of token positions stored per KV cache block.
pub(crate) const BLOCK_SIZE: usize = 16;

/// Maximum number of logical blocks per sequence (supports up to
/// MAX_BLOCKS_PER_SEQ * BLOCK_SIZE = 256 * 16 = 4096 tokens).
pub(crate) const MAX_BLOCKS_PER_SEQ: usize = 256;

/// A pool of KV cache blocks shared across all sequences.
///
/// The pool owns the physical GPU memory for all K and V caches across all
/// layers.  Individual sequences get assigned physical blocks from this pool
/// via their block tables.
pub(crate) struct KvPool<B: GpuBackend> {
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
pub(crate) struct SeqKvState<B: GpuBackend> {
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
}

impl<B: GpuBackend> KvPool<B> {
    /// Allocate a KV pool with room for `num_blocks` physical blocks.
    ///
    /// Total GPU memory = num_blocks * BLOCK_SIZE * kv_dim * 2 bytes * 2 (K+V)
    ///                     * num_layers.
    /// For 256 blocks, 16 layers, kv_dim=512: 256 * 16 * 512 * 2 * 2 * 16 = 128 MB.
    pub fn new(
        backend: &B,
        num_blocks: usize,
        kv_dim: usize,
        num_layers: usize,
    ) -> Self {
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
        }
    }

    /// Free all blocks belonging to a sequence.
    pub fn free_sequence(&mut self, seq: &SeqKvState<B>) {
        for &block_idx in &seq.block_table_cpu {
            self.free_block(block_idx);
        }
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// KV dimension.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

impl<B: GpuBackend> SeqKvState<B> {
    /// Ensure there is room for one more token.  If the current last block
    /// is full (or no blocks allocated yet), allocate a new block from the pool.
    pub fn ensure_slot(&mut self, pool: &mut KvPool<B>) -> anyhow::Result<()> {
        let needed_blocks = (self.seq_len + BLOCK_SIZE) / BLOCK_SIZE; // ceil((seq_len+1)/BLOCK_SIZE)
        if needed_blocks > self.block_table_cpu.len() {
            // Need a new block.
            let block_idx = pool.alloc_block()
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
            let block_idx = pool.alloc_block()
                .ok_or_else(|| anyhow::anyhow!("KV cache out of memory: no free blocks"))?;
            self.block_table_cpu.push(block_idx);
            self.dirty = true;
        }
        Ok(())
    }

    /// Number of logical blocks currently allocated.
    pub fn num_blocks(&self) -> usize {
        self.block_table_cpu.len()
    }
}

