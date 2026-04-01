// ===========================================================================
// SSD expert streaming for MoE models.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Enables running MoE models that don't fit in GPU memory by streaming
//   expert weights from SSD on-demand.  Only the K active experts selected
//   by the router are loaded per layer, instead of keeping all N experts
//   resident.  For Qwen3.5-35b-a3b (256 experts, 67GB bf16), this reduces
//   expert memory from 60GB to ~15MB (K=8 buffer slots).
//
// How it works:
//   1. During model loading, expert weight tensors are NOT uploaded to GPU.
//      Instead, their file locations (shard path + byte offset) are recorded
//      in an ExpertIndex.
//   2. K pre-allocated GPU buffer slots are created (one per active expert).
//   3. During inference, after the router selects K experts, the streamer
//      pread()s their weights from disk into the GPU buffers.
//   4. The forward pass runs using these temporary buffers, then they're
//      reused for the next layer.
//
// Performance architecture (flash-moe pattern):
//   - **Parallel pread + upload**: K experts are loaded concurrently using
//     std::thread::scope.  Each thread reads one expert's weights via pread()
//     and immediately copies them into the GPU buffer — both phases (disk I/O
//     and GPU upload) run in parallel across experts.  On unified memory
//     backends (Metal), the GPU upload is a direct memcpy to buffer.contents()
//     which is safe to do from multiple threads (no Metal API calls involved).
//   - **Fused gate+up read**: For Qwen3.5's fused format, gate and up
//     projections are contiguous on disk.  Instead of 2 pread() calls, we do
//     1 pread() for the combined block — halving NVMe command overhead.
//   - **OS page cache**: No custom LRU.  Flash-moe tested custom caching and
//     found it 38% slower than trusting the OS page cache (~71% hit rate
//     naturally).
//   - **No pipeline overlap**: Apple Silicon's unified memory controller means
//     GPU compute and NVMe DMA can't truly overlap — serial I/O→compute is
//     optimal.
//
// Why parallel Phase 2?
//   The original code serialized GPU uploads because "Metal command encoding
//   isn't thread-safe."  But copy_to_tensor on Metal is just ptr::copy to
//   buffer.contents() — a plain memory write with no Metal API calls.
//   Writing to different Metal buffer contents pointers from different threads
//   is safe (disjoint memory regions).  By merging Phase 2 into each pread
//   thread, we parallelise both I/O and memcpy across K experts.
//
// Double-buffering (CUDA preparation):
//   Two sets of K GPU buffer slots are allocated (active + inactive).
//   load_experts() writes to the active set synchronously (Metal path).
//   load_experts_async() writes to the inactive set via copy_to_tensor_async,
//   then sync_and_swap() flips them.  On Metal the two paths are identical
//   because copy_to_tensor_async defaults to synchronous copy.  On CUDA,
//   this will enable overlapping DMA with compute via a transfer stream.
//
// Why pread and not mmap?
//   mmap'ing 751GB of shard files (94 × ~8GB for 397B) causes excessive memory
//   pressure.  Each page fault triggers a 16KB kernel I/O operation, while pread()
//   issues 8-16MB reads that the kernel can satisfy with large sequential I/O.
//   For large MoE models, targeted pread gives the kernel better I/O scheduling
//   information than random page faults across hundreds of GB of virtual mappings.
//
// Inspired by flash-moe (github.com/danveloper/flash-moe).
//
// Related files:
//   loader.rs       — builds the ExpertIndex during model loading
//   primitives.rs   — moe_expert_dispatch_streamed() uses the streamer
//   model/mod.rs    — Model holds Option<ExpertStreamer>
// ===========================================================================

use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use tracing::info;

use crate::gpu::{GpuCore, PinnedBuf, TensorDtype};

/// Wrapper to send raw pointers across thread boundaries.
///
/// Safety: the pointed-to memory must outlive the thread and each thread
/// must write to disjoint regions.  Both are guaranteed by ExpertStreamer:
/// GPU buffer slots are pre-allocated and each thread targets a different slot.
#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

// ---------------------------------------------------------------------------
// ExpertLocation — where one expert's weights live on disk.
//
// For Qwen3.5's fused format, gate_up_proj is a single tensor
// [num_experts, 2*moe_inter, hidden] where each expert is a contiguous
// slice along dim 0.  We record the byte offset and size for each
// expert's gate, up, and down projections within their shard files.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct ExpertLocation {
    /// Shard index for gate projection (into shard_files vec).
    pub shard_gate_up: usize,
    /// Shard index for up projection (may differ from gate for per-expert models).
    ///
    /// Mixtral Q4 occasionally splits gate and up across shards at shard
    /// boundaries (e.g. layer 14 expert 1: w1 in shard 1, w3 in shard 2).
    pub shard_up: usize,
    /// Shard index for down_proj tensor.
    pub shard_down: usize,
    /// File byte offset of this expert's gate projection data.
    pub gate_offset: u64,
    /// File byte offset of this expert's up projection data.
    pub up_offset: u64,
    /// File byte offset of this expert's down projection data.
    pub down_offset: u64,
    /// Byte size of gate projection (moe_inter * hidden * 2 for bf16).
    pub gate_bytes: usize,
    /// Byte size of up projection (same as gate for standard models).
    pub up_bytes: usize,
    /// Byte size of down projection (hidden * moe_inter * 2 for bf16).
    pub down_bytes: usize,
}

impl ExpertLocation {
    /// Whether gate and up projections are contiguous on disk.
    ///
    /// True for Qwen3.5's fused gate_up_proj format, where gate and up are
    /// stacked along dim 0 within a single tensor.  When contiguous AND in
    /// the same shard, we can read both in a single pread() instead of two.
    fn gate_up_contiguous(&self) -> bool {
        self.shard_gate_up == self.shard_up
            && self.up_offset == self.gate_offset + self.gate_bytes as u64
    }
}

// ---------------------------------------------------------------------------
// ExpertIndex — full model expert location map.
//
// Built during loading when --stream-experts is set.  Records where every
// expert's weights live on disk without loading any expert data.
//
// shard_files is Arc-wrapped so it can be shared with scoped threads without
// lifetime issues (File is Sync, so &File can be sent to other threads).
// ---------------------------------------------------------------------------

pub(crate) struct ExpertIndex {
    /// Per-layer expert locations: layers[layer_idx][expert_idx].
    /// Non-MoE layers (e.g. Nemotron-H mamba2/attention) have empty vecs.
    pub layers: Vec<Vec<ExpertLocation>>,
    /// Open file handles for each shard (kept alive for pread).
    pub shard_files: Arc<Vec<File>>,
    /// Expert dimensions.
    pub hidden: usize,
    pub moe_inter: usize,
    /// Pre-quantization format of expert data on disk (Q4 or Q8).
    /// When Some, pread reads quantized bytes directly — less I/O than bf16.
    pub quant_format: Option<crate::gpu::ops::quant::QuantFormat>,
    /// Whether experts have a gate projection (SwiGLU models).
    /// False for Nemotron-H which uses relu² (only up_proj + down_proj).
    pub has_gate_proj: bool,
}

// ---------------------------------------------------------------------------
// SlotReadBufs — per-slot CPU buffers for parallel pread.
//
// Each expert slot gets its own read buffers so that K threads can pread()
// concurrently without contention.  The gate_up buffer holds both gate and
// up projections contiguously (matching the fused on-disk layout) so that
// Qwen3.5's fused format can be loaded with a single pread.
//
// On CUDA, buffers are pinned (page-locked) via cuMemAllocHost — required
// for true async HtoD transfers.  On Metal/CPU, regular heap buffers are
// used since unified memory doesn't need pinning.
// ---------------------------------------------------------------------------

/// A staging buffer that may be pinned (for async DMA) or heap-allocated.
///
/// Pinned memory (CUDA): allocated via cuMemAllocHost, enables true async
/// transfers via cuMemcpyHtoDAsync on a dedicated transfer stream.
/// Heap memory (Metal/CPU): regular Vec<u8>, used when pinned isn't needed
/// (unified memory) or unavailable (alloc_pinned_buf returns None).
enum ReadBuf {
    Heap(Vec<u8>),
    Pinned(PinnedBuf),
}

impl ReadBuf {
    fn as_slice(&self) -> &[u8] {
        match self {
            ReadBuf::Heap(v) => v.as_slice(),
            ReadBuf::Pinned(p) => p.as_slice(),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            ReadBuf::Heap(v) => v.as_mut_slice(),
            ReadBuf::Pinned(p) => p.as_mut_slice(),
        }
    }
}

struct SlotReadBufs {
    /// Combined gate+up buffer.  For fused format, read in one pread().
    /// Layout: [gate_bytes | up_bytes] — gate first, then up.
    gate_up: ReadBuf,
    /// Down projection buffer.
    down: ReadBuf,
}

// ---------------------------------------------------------------------------
// ExpertStreamer — manages GPU buffer slots and loads experts on demand.
//
// The streamer holds a pool of GPU buffer slots as an LRU cache.  When
// the router selects K experts, load_experts() checks the cache first:
// cached experts skip the NVMe read + GPU upload entirely.  Only cache
// misses trigger pread + DMA.  This is especially effective on CUDA where
// each expert load requires a PCIe transfer — cache hits avoid both the
// NVMe read AND the PCIe hop.
//
// The cache size defaults to 64 slots (~6 MB × 64 = ~384 MB for Q4
// Qwen3.5-122B), which caches 25% of the 256 experts per layer.
// Cross-layer locality (same expert selected in adjacent layers) and
// cross-token locality (same expert popular across tokens) both benefit.
//
// Performance: parallel pread across cache misses, fused gate+up reads
// for models with contiguous gate/up layout (Qwen3.5).  On unified
// memory backends (Metal), the GPU upload is merged into the same
// parallel pass as the pread.
// ---------------------------------------------------------------------------

/// Default number of expert cache slots.  64 slots × ~6 MB per Q4 expert
/// = ~384 MB GPU memory for Qwen3.5-122B.  Larger caches improve hit
/// rate but consume more VRAM.
const DEFAULT_CACHE_SLOTS: usize = 64;

/// Tracks an in-flight background prefetch operation (CUDA only).
///
/// Between layers, prefetch_start() spawns background pread threads that
/// read the predicted next-layer experts from SSD into CPU pinned buffers.
/// When the next layer's load_experts() is called, it checks if there's a
/// completed prefetch for that layer and uses the prefetched data for hits,
/// falling back to normal pread for mispredictions.
#[cfg(not(target_os = "macos"))]
struct PrefetchState {
    /// Which layer was prefetched.
    layer_idx: usize,
    /// Expert indices that were prefetched (in order matching prefetch_bufs slots).
    experts: Vec<usize>,
    /// Whether the prefetch has been completed (threads joined).
    completed: bool,
}

pub(crate) struct ExpertStreamer<B: GpuCore> {
    /// Expert index (file locations for all experts).
    pub index: ExpertIndex,
    /// LRU cache of GPU-resident expert weight buffers.
    /// Size is configurable (default 64 slots).
    cache: Vec<ExpertSlot<B>>,
    /// Maps (layer_idx, expert_idx) → cache slot index for O(1) lookup.
    cache_map: UnsafeCell<HashMap<(usize, usize), usize>>,
    /// Monotonic counter for LRU tracking.  Each access bumps the slot's
    /// last_used timestamp.  On eviction, the slot with the smallest
    /// timestamp is chosen.
    lru_clock: Cell<u64>,
    /// Per-slot last-used timestamp for LRU eviction.
    lru_timestamps: UnsafeCell<Vec<u64>>,
    /// After load_experts(), maps slot_idx (0..K) → cache slot index.
    /// Used by active_slot() to find the cached expert for each selected slot.
    active_indices: UnsafeCell<Vec<usize>>,
    /// Number of experts selected per token (K).
    #[allow(dead_code)] // stored for diagnostics; active_indices.len() used at runtime
    k: usize,
    /// Per-slot CPU read buffers for parallel pread → GPU upload.
    ///
    /// UnsafeCell because load_experts() needs to write to these buffers
    /// through &self (matching the GPU tensor interior mutability pattern —
    /// inference is single-threaded within a model).  Each slot's buffers
    /// are accessed by exactly one thread during parallel pread.
    read_bufs: UnsafeCell<Vec<SlotReadBufs>>,

    /// Second set of CPU read buffers for background prefetch (CUDA only).
    /// Populated by prefetch_start(), consumed by load_experts() on the next
    /// layer when the prediction matches.  Gated by cfg(not(target_os = "macos"))
    /// because Metal's unified memory controller can't overlap SSD DMA with
    /// GPU compute — serial I/O→compute is optimal there.
    #[cfg(not(target_os = "macos"))]
    prefetch_bufs: UnsafeCell<Vec<SlotReadBufs>>,
    /// State of in-flight prefetch, if any.
    #[cfg(not(target_os = "macos"))]
    prefetch_state: UnsafeCell<Option<PrefetchState>>,
}

/// Safety: ExpertStreamer uses interior mutability (Cell, UnsafeCell) for
/// single-threaded inference — each model instance is accessed by exactly
/// one thread at a time.  The multi-GPU dispatch in multi_gpu.rs assigns
/// one RankState per thread; each rank's ExpertStreamer is never shared
/// across threads concurrently.  This impl is required because CudaBackend
/// is used with thread::scope in multi_gpu.rs, which requires Sync on
/// captured references even though each thread touches a disjoint rank.
unsafe impl<B: GpuCore> Sync for ExpertStreamer<B> {}

/// One expert's worth of GPU-resident weight buffers.
pub(crate) struct ExpertSlot<B: GpuCore> {
    pub gate_proj: B::Tensor,
    pub up_proj: B::Tensor,
    pub down_proj: B::Tensor,
}

impl<B: GpuCore> ExpertStreamer<B> {
    /// Create a new streamer with an LRU cache of GPU buffer slots.
    ///
    /// `k` is the number of experts per token (from the model config).
    /// The cache holds `cache_slots` experts (default 64), so frequently
    /// accessed experts remain GPU-resident across layers and tokens.
    pub fn new(backend: &B, index: ExpertIndex, k: usize) -> Self {
        Self::new_with_cache(backend, index, k, DEFAULT_CACHE_SLOTS)
    }

    /// Create a streamer with a specific cache size.
    pub fn new_with_cache(
        backend: &B,
        index: ExpertIndex,
        k: usize,
        cache_slots: usize,
    ) -> Self {
        // Cache must hold at least K slots (one per active expert).
        let cache_slots = cache_slots.max(k);
        let cache = Self::allocate_slots(backend, &index, cache_slots);

        // CPU read buffers: sized for the on-disk format.  Only K buffers
        // needed — at most K experts are loaded from disk per call (misses).
        let hidden = index.hidden;
        let moe_inter = index.moe_inter;
        // Nemotron (no gate): only up_proj on disk, so gate_up buffer = up_bytes.
        // SwiGLU models: gate_up buffer = gate_bytes + up_bytes = 2 * moe_inter.
        let gate_up_factor = if index.has_gate_proj { moe_inter * 2 } else { moe_inter };
        let (gate_up_bytes, down_bytes) = expert_byte_sizes(
            index.quant_format, gate_up_factor, hidden, moe_inter,
        );
        // Try pinned (page-locked) allocation for CUDA async DMA.
        // Falls back to heap allocation on Metal/CPU or if pinned alloc fails.
        let alloc_buf = |byte_count: usize| -> ReadBuf {
            match backend.alloc_pinned_buf(byte_count) {
                Some(buf) => ReadBuf::Pinned(buf),
                None => ReadBuf::Heap(vec![0u8; byte_count]),
            }
        };
        let read_bufs = UnsafeCell::new(
            (0..k)
                .map(|_| SlotReadBufs {
                    gate_up: alloc_buf(gate_up_bytes),
                    down: alloc_buf(down_bytes),
                })
                .collect(),
        );

        let expert_bytes = gate_up_bytes + down_bytes;
        info!(
            cache_slots = cache_slots,
            cache_mb = cache_slots * expert_bytes / (1024 * 1024),
            experts_per_tok = k,
            mb_per_expert = expert_bytes / (1024 * 1024),
            "expert streaming initialized",
        );

        #[cfg(not(target_os = "macos"))]
        let prefetch_bufs = UnsafeCell::new(
            (0..k)
                .map(|_| SlotReadBufs {
                    gate_up: alloc_buf(gate_up_bytes),
                    down: alloc_buf(down_bytes),
                })
                .collect(),
        );

        ExpertStreamer {
            index,
            cache,
            cache_map: UnsafeCell::new(HashMap::with_capacity(cache_slots)),
            lru_clock: Cell::new(0),
            lru_timestamps: UnsafeCell::new(vec![0u64; cache_slots]),
            active_indices: UnsafeCell::new(vec![0usize; k]),
            k,
            read_bufs,
            #[cfg(not(target_os = "macos"))]
            prefetch_bufs,
            #[cfg(not(target_os = "macos"))]
            prefetch_state: UnsafeCell::new(None),
        }
    }

    /// Allocate GPU buffer slots for expert weights.
    fn allocate_slots(backend: &B, index: &ExpertIndex, n: usize) -> Vec<ExpertSlot<B>> {
        let hidden = index.hidden;
        let moe_inter = index.moe_inter;
        let dtype = match index.quant_format {
            Some(crate::gpu::ops::quant::QuantFormat::Q4) => TensorDtype::Q4,
            Some(crate::gpu::ops::quant::QuantFormat::Q8) => TensorDtype::Q8,
            Some(crate::gpu::ops::quant::QuantFormat::FP8) => TensorDtype::FP8,
            Some(crate::gpu::ops::quant::QuantFormat::TQ3) => TensorDtype::TQ3,
            Some(crate::gpu::ops::quant::QuantFormat::NVFP4) => TensorDtype::NVFP4,
            None => TensorDtype::BF16,
        };

        // Nemotron-H (relu²) has no gate_proj — allocate a dummy 1-element tensor.
        let gate_shape = if index.has_gate_proj {
            vec![moe_inter, hidden]
        } else {
            vec![1]
        };

        (0..n)
            .map(|_| ExpertSlot {
                gate_proj: backend.alloc_tensor(&gate_shape, dtype),
                up_proj: backend.alloc_tensor(&[moe_inter, hidden], dtype),
                down_proj: backend.alloc_tensor(&[hidden, moe_inter], dtype),
            })
            .collect()
    }

    /// Get a cached expert slot after load_experts().
    ///
    /// `idx` is 0..K (the position in the selected experts list).
    /// Returns a reference to the cache slot holding that expert's weights.
    pub fn active_slot(&self, idx: usize) -> &ExpertSlot<B> {
        let active_indices = unsafe { &*self.active_indices.get() };
        &self.cache[active_indices[idx]]
    }

    /// Find the LRU (least recently used) cache slot for eviction.
    fn find_lru_slot(&self) -> usize {
        let timestamps = unsafe { &*self.lru_timestamps.get() };
        let mut min_ts = u64::MAX;
        let mut min_idx = 0;
        for (i, &ts) in timestamps.iter().enumerate() {
            if ts < min_ts {
                min_ts = ts;
                min_idx = i;
            }
        }
        min_idx
    }

    /// Load selected experts into GPU cache slots with LRU caching.
    ///
    /// For each selected expert:
    ///   - Cache hit: skip NVMe read + GPU upload entirely (free).
    ///   - Cache miss: evict LRU slot, pread from SSD, upload to GPU.
    ///
    /// Cache hits avoid both the NVMe read AND the PCIe transfer (on CUDA),
    /// which is the dominant cost for expert streaming on discrete GPUs.
    ///
    /// Safety: takes &self (not &mut) because inference is single-threaded and
    /// GPU tensor writes already use interior mutability.
    pub fn load_experts(
        &self,
        backend: &B,
        layer_idx: usize,
        selected: &[(usize, f32)],
    ) {
        let cache_map = unsafe { &mut *self.cache_map.get() };
        let timestamps = unsafe { &mut *self.lru_timestamps.get() };
        let active_indices = unsafe { &mut *self.active_indices.get() };
        let read_bufs = unsafe { &mut *self.read_bufs.get() };

        // Bump the LRU clock.
        let clock = self.lru_clock.get() + 1;
        self.lru_clock.set(clock);

        // Phase 0: Classify each selected expert as hit or miss.
        // Collect misses that need loading.
        struct MissInfo {
            _slot_idx: usize,     // position in selected[] (0..K)
            expert_idx: usize,    // global expert index
            cache_slot: usize,    // cache slot to load into
        }
        let mut misses: Vec<MissInfo> = Vec::new();

        for (slot_idx, &(expert_idx, _)) in selected.iter().enumerate() {
            let key = (layer_idx, expert_idx);
            if let Some(&cache_slot) = cache_map.get(&key) {
                // Cache hit — just update LRU timestamp.
                timestamps[cache_slot] = clock;
                active_indices[slot_idx] = cache_slot;
            } else {
                // Cache miss — evict LRU slot.
                let cache_slot = self.find_lru_slot();
                // Remove old entry from cache_map.
                cache_map.retain(|_, &mut v| v != cache_slot);
                // Insert new entry.
                cache_map.insert(key, cache_slot);
                timestamps[cache_slot] = clock;
                active_indices[slot_idx] = cache_slot;
                misses.push(MissInfo {
                    _slot_idx: slot_idx,
                    expert_idx,
                    cache_slot,
                });
            }
        }

        // If all hits, nothing to load — early return.
        if misses.is_empty() {
            return;
        }

        // Pre-fetch GPU buffer pointers for missed slots (direct path).
        let gpu_ptrs: Vec<Option<[SendPtr; 3]>> = misses
            .iter()
            .map(|m| {
                let slot = &self.cache[m.cache_slot];
                match (
                    backend.tensor_mut_ptr(&slot.gate_proj),
                    backend.tensor_mut_ptr(&slot.up_proj),
                    backend.tensor_mut_ptr(&slot.down_proj),
                ) {
                    (Some(g), Some(u), Some(d)) => {
                        Some([SendPtr(g), SendPtr(u), SendPtr(d)])
                    }
                    _ => None,
                }
            })
            .collect();
        let direct = gpu_ptrs.first().is_some_and(|p| p.is_some());

        // Phase 1 + Phase 2 (fused per-thread): parallel pread + GPU upload
        // for cache misses only.
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(misses.len());

            let buf_iter = read_bufs.iter_mut();
            for ((miss_idx, miss), slot_bufs) in misses.iter().enumerate().zip(buf_iter) {
                let loc = &self.index.layers[layer_idx][miss.expert_idx];
                let shard_files = &self.index.shard_files;
                let ptrs = gpu_ptrs[miss_idx];

                handles.push(s.spawn(move || -> std::io::Result<()> {
                    let gate_up = slot_bufs.gate_up.as_mut_slice();
                    if loc.gate_bytes == 0 {
                        // No gate projection (Nemotron relu²) — only read up_proj.
                        pread_exact(
                            &shard_files[loc.shard_up],
                            &mut gate_up[..loc.up_bytes],
                            loc.up_offset,
                        )?;
                    } else if loc.gate_up_contiguous() {
                        let total = loc.gate_bytes + loc.up_bytes;
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut gate_up[..total],
                            loc.gate_offset,
                        )?;
                    } else {
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut gate_up[..loc.gate_bytes],
                            loc.gate_offset,
                        )?;
                        pread_exact(
                            &shard_files[loc.shard_up],
                            &mut gate_up[loc.gate_bytes..loc.gate_bytes + loc.up_bytes],
                            loc.up_offset,
                        )?;
                    }

                    pread_exact(
                        &shard_files[loc.shard_down],
                        &mut slot_bufs.down.as_mut_slice()[..loc.down_bytes],
                        loc.down_offset,
                    )?;

                    // Phase 2 (direct): memcpy for unified memory backends.
                    if let Some([g, u, d]) = ptrs {
                        let gate_up = slot_bufs.gate_up.as_slice();
                        if loc.gate_bytes > 0 {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    gate_up.as_ptr(),
                                    g.0,
                                    loc.gate_bytes,
                                );
                                std::ptr::copy_nonoverlapping(
                                    gate_up.as_ptr().add(loc.gate_bytes),
                                    u.0,
                                    loc.up_bytes,
                                );
                            }
                        } else {
                            // No gate — up_proj is at offset 0 in the buffer.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    gate_up.as_ptr(),
                                    u.0,
                                    loc.up_bytes,
                                );
                            }
                        }
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                slot_bufs.down.as_slice().as_ptr(),
                                d.0,
                                loc.down_bytes,
                            );
                        }
                    }
                    Ok(())
                }));
            }

            for h in handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => panic!("expert streaming I/O failed: {e}"),
                    Err(_) => panic!("expert streaming thread panicked"),
                }
            }
        });

        // Fallback Phase 2 (async DMA): CUDA path for cache misses.
        if !direct {
            for (miss_idx, miss) in misses.iter().enumerate() {
                let loc = &self.index.layers[layer_idx][miss.expert_idx];
                let slot = &self.cache[miss.cache_slot];
                let bufs = &read_bufs[miss_idx];
                let gate_up = bufs.gate_up.as_slice();

                if loc.gate_bytes > 0 {
                    backend.copy_to_tensor_async(&slot.gate_proj, &gate_up[..loc.gate_bytes]);
                    backend.copy_to_tensor_async(
                        &slot.up_proj,
                        &gate_up[loc.gate_bytes..loc.gate_bytes + loc.up_bytes],
                    );
                } else {
                    // No gate — up_proj at offset 0.
                    backend.copy_to_tensor_async(&slot.up_proj, &gate_up[..loc.up_bytes]);
                }
                backend.copy_to_tensor_async(&slot.down_proj, &bufs.down.as_slice()[..loc.down_bytes]);
            }
            backend.sync_transfers();
        }
    }

    /// Find the next MoE layer after `from` (handles interleaved non-MoE layers).
    #[cfg(not(target_os = "macos"))]
    fn next_moe_layer(&self, from: usize) -> Option<usize> {
        for i in (from + 1)..self.index.layers.len() {
            if !self.index.layers[i].is_empty() {
                return Some(i);
            }
        }
        None
    }

    /// Start background prefetch of predicted experts for the next MoE layer.
    ///
    /// Spawns pread threads that read expert weights from SSD into CPU pinned
    /// buffers.  The threads are joined when `prefetch_join()` is called.
    /// Prediction heuristic: reuse this layer's selected expert indices.
    ///
    /// CUDA-only — Metal's unified memory can't overlap SSD DMA with GPU compute.
    #[cfg(not(target_os = "macos"))]
    pub fn prefetch_start(
        &self,
        layer_idx: usize,
        predicted_experts: &[(usize, f32)],
    ) {
        let next_layer = match self.next_moe_layer(layer_idx) {
            Some(l) => l,
            None => return, // last MoE layer — nothing to prefetch
        };

        let prefetch_state = unsafe { &mut *self.prefetch_state.get() };
        let k = unsafe { &*self.prefetch_bufs.get() }.len();

        // Collect expert indices to prefetch (same as current layer's selection).
        let experts: Vec<usize> = predicted_experts
            .iter()
            .take(k)
            .map(|&(idx, _)| idx)
            .collect();

        // Validate that the predicted experts exist in the next layer.
        let next_experts = &self.index.layers[next_layer];
        let valid_experts: Vec<usize> = experts
            .iter()
            .filter(|&&e| e < next_experts.len())
            .copied()
            .collect();

        if valid_experts.is_empty() {
            return;
        }

        // Synchronous prefetch: use scoped threads to pread into prefetch_bufs.
        // This blocks until all reads complete, but the GPU is still computing
        // asynchronously on its CUDA stream.  The benefit: when the next layer
        // needs these experts, the data is already in CPU pinned RAM — only
        // the fast PCIe DMA transfer remains.
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(valid_experts.len());

            for (buf_idx, &expert_idx) in valid_experts.iter().enumerate() {
                let loc = &next_experts[expert_idx];
                let shard_files = &self.index.shard_files;
                // Safety: each thread writes to a disjoint prefetch buffer slot.
                let bufs_ptr = self.prefetch_bufs.get();
                let slot_bufs = unsafe { &mut (&mut (*bufs_ptr))[buf_idx] };

                handles.push(s.spawn(move || -> std::io::Result<()> {
                    let gate_up = slot_bufs.gate_up.as_mut_slice();
                    if loc.gate_bytes == 0 {
                        pread_exact(
                            &shard_files[loc.shard_up],
                            &mut gate_up[..loc.up_bytes],
                            loc.up_offset,
                        )?;
                    } else if loc.gate_up_contiguous() {
                        let total = loc.gate_bytes + loc.up_bytes;
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut gate_up[..total],
                            loc.gate_offset,
                        )?;
                    } else {
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut gate_up[..loc.gate_bytes],
                            loc.gate_offset,
                        )?;
                        pread_exact(
                            &shard_files[loc.shard_up],
                            &mut gate_up[loc.gate_bytes..loc.gate_bytes + loc.up_bytes],
                            loc.up_offset,
                        )?;
                    }

                    pread_exact(
                        &shard_files[loc.shard_down],
                        &mut slot_bufs.down.as_mut_slice()[..loc.down_bytes],
                        loc.down_offset,
                    )?;
                    Ok(())
                }));
            }

            for h in handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        tracing::warn!("prefetch I/O error (non-fatal): {e}");
                        return;
                    }
                    Err(_) => {
                        tracing::warn!("prefetch thread panicked (non-fatal)");
                        return;
                    }
                }
            }
        });

        *prefetch_state = Some(PrefetchState {
            layer_idx: next_layer,
            experts: valid_experts,
            completed: true,
        });
    }

    /// Check if there's a completed prefetch for the given layer, and if the
    /// given expert was prefetched, return a reference to its prefetch buffer.
    ///
    /// Returns `Some(buf_idx)` if the expert was prefetched, `None` otherwise.
    #[cfg(not(target_os = "macos"))]
    fn prefetch_hit(&self, layer_idx: usize, expert_idx: usize) -> Option<usize> {
        let state = unsafe { &*self.prefetch_state.get() };
        if let Some(ps) = state {
            if ps.layer_idx == layer_idx && ps.completed {
                return ps.experts.iter().position(|&e| e == expert_idx);
            }
        }
        None
    }

    /// Upload a prefetched expert from prefetch_bufs to GPU via async DMA.
    ///
    /// Called during load_experts() for cache misses that were correctly predicted.
    #[cfg(not(target_os = "macos"))]
    fn upload_prefetched(
        &self,
        backend: &B,
        layer_idx: usize,
        expert_idx: usize,
        cache_slot: usize,
        prefetch_buf_idx: usize,
    ) {
        let prefetch_bufs = unsafe { &*self.prefetch_bufs.get() };
        let loc = &self.index.layers[layer_idx][expert_idx];
        let slot = &self.cache[cache_slot];
        let bufs = &prefetch_bufs[prefetch_buf_idx];
        let gate_up = bufs.gate_up.as_slice();

        if loc.gate_bytes > 0 {
            backend.copy_to_tensor_async(&slot.gate_proj, &gate_up[..loc.gate_bytes]);
            backend.copy_to_tensor_async(
                &slot.up_proj,
                &gate_up[loc.gate_bytes..loc.gate_bytes + loc.up_bytes],
            );
        } else {
            backend.copy_to_tensor_async(&slot.up_proj, &gate_up[..loc.up_bytes]);
        }
        backend.copy_to_tensor_async(&slot.down_proj, &bufs.down.as_slice()[..loc.down_bytes]);
    }

    /// Log prediction accuracy for this layer's prefetch.
    #[cfg(not(target_os = "macos"))]
    pub fn prefetch_log_accuracy(&self, layer_idx: usize, selected: &[(usize, f32)]) {
        let state = unsafe { &*self.prefetch_state.get() };
        if let Some(ps) = state {
            if ps.layer_idx == layer_idx && ps.completed {
                let hits = ps.experts.iter()
                    .filter(|&&e| selected.iter().any(|&(s, _)| s == e))
                    .count();
                tracing::debug!(
                    layer = layer_idx,
                    hits = hits,
                    total = ps.experts.len(),
                    "expert prefetch accuracy"
                );
            }
        }
    }

    /// Clear prefetch state after it's been consumed.
    #[cfg(not(target_os = "macos"))]
    pub fn prefetch_clear(&self) {
        let state = unsafe { &mut *self.prefetch_state.get() };
        *state = None;
    }
}

// ---------------------------------------------------------------------------
// pread — read bytes from a file at a specific offset without seeking.
//
// Uses the Unix pread() syscall which is thread-safe (no shared file
// position) and efficient (single syscall, no lseek + read race).
// ---------------------------------------------------------------------------

fn pread_exact(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

// ---------------------------------------------------------------------------
// Index building helpers — used by loader.rs.
// ---------------------------------------------------------------------------

/// Compute the absolute file offset where tensor data begins in a safetensors file.
///
/// Safetensors layout: [8-byte LE header_len][JSON header (header_len bytes)][tensor data...]
/// The `data_offset` from the JSON header is relative to the start of the tensor data region.
pub(crate) fn safetensors_data_start(mmap: &[u8]) -> u64 {
    assert!(
        mmap.len() >= 8,
        "safetensors file too small ({} bytes, need at least 8)",
        mmap.len()
    );
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap());
    let data_start = 8 + header_len;
    assert!(
        data_start as usize <= mmap.len(),
        "safetensors header_len ({header_len}) exceeds file size ({})",
        mmap.len()
    );
    data_start
}

/// Build an ExpertIndex for fused Qwen3.5-style expert tensors.
///
/// Qwen3.5 stores experts as fused tensors:
///   gate_up_proj: [num_experts, 2 * moe_inter, hidden] — gate and up stacked
///   down_proj:    [num_experts, hidden, moe_inter]
///
/// Each expert is a contiguous slice along dim 0, so expert j's data starts
/// at `tensor_file_offset + j * expert_stride_bytes`.
pub(crate) fn build_fused_expert_index(
    layer_info: Vec<FusedLayerInfo>,
    shard_files: Vec<File>,
    hidden: usize,
    moe_inter: usize,
    num_experts: usize,
    quant_format: Option<crate::gpu::ops::quant::QuantFormat>,
) -> ExpertIndex {
    // Byte sizes depend on whether experts are stored as Q4/Q8 or bf16 on disk.
    let (gate_up_expert_bytes, gate_bytes, down_expert_bytes) = {
        let (gu, d) = expert_byte_sizes(quant_format, moe_inter * 2, hidden, moe_inter);
        let (g, _) = expert_byte_sizes(quant_format, moe_inter, hidden, moe_inter);
        (gu, g, d)
    };

    let layers = layer_info
        .iter()
        .map(|info| {
            (0..num_experts)
                .map(|j| {
                    let gu_base = info.gate_up_file_offset + (j as u64) * (gate_up_expert_bytes as u64);
                    let d_base = info.down_file_offset + (j as u64) * (down_expert_bytes as u64);

                    ExpertLocation {
                        shard_gate_up: info.shard_gate_up,
                        shard_up: info.shard_gate_up, // fused: same tensor → same shard
                        shard_down: info.shard_down,
                        gate_offset: gu_base,
                        up_offset: gu_base + gate_bytes as u64,
                        down_offset: d_base,
                        gate_bytes,
                        up_bytes: gate_bytes,
                        down_bytes: down_expert_bytes,
                    }
                })
                .collect()
        })
        .collect();

    ExpertIndex {
        layers,
        shard_files: Arc::new(shard_files),
        hidden,
        moe_inter,
        quant_format,
        has_gate_proj: true, // Fused format always has gate+up
    }
}

/// Build an ExpertIndex for per-expert tensor format (Qwen3-MoE, Mixtral, Nemotron-H).
///
/// Standard (has_gate=true): 3 tensors per expert (gate_proj, up_proj, down_proj).
/// Nemotron (has_gate=false): 2 tensors per expert (up_proj, down_proj — relu² activation).
/// Non-MoE layers have empty vecs in layer_info (Nemotron-H's mamba2/attention layers).
pub(crate) fn build_per_expert_index(
    layer_info: Vec<Vec<PerExpertInfo>>,
    shard_files: Vec<File>,
    hidden: usize,
    moe_inter: usize,
    quant_format: Option<crate::gpu::ops::quant::QuantFormat>,
    has_gate: bool,
) -> ExpertIndex {
    let (proj_bytes, down_bytes) = expert_byte_sizes(quant_format, moe_inter, hidden, moe_inter);
    let gate_bytes = if has_gate { proj_bytes } else { 0 };

    let layers = layer_info
        .iter()
        .map(|experts| {
            experts
                .iter()
                .map(|info| ExpertLocation {
                    shard_gate_up: info.shard_gate,
                    shard_up: info.shard_up,
                    shard_down: info.shard_down,
                    gate_offset: info.gate_file_offset,
                    up_offset: info.up_file_offset,
                    down_offset: info.down_file_offset,
                    gate_bytes,
                    up_bytes: proj_bytes,
                    down_bytes,
                })
                .collect()
        })
        .collect();

    ExpertIndex {
        layers,
        shard_files: Arc::new(shard_files),
        hidden,
        moe_inter,
        quant_format,
        has_gate_proj: has_gate,
    }
}

/// Compute (gate_up_bytes, down_bytes) for a given quant format.
fn expert_byte_sizes(
    quant_format: Option<crate::gpu::ops::quant::QuantFormat>,
    gate_up_rows: usize,
    hidden: usize,
    moe_inter: usize,
) -> (usize, usize) {
    match quant_format {
        Some(crate::gpu::ops::quant::QuantFormat::Q4) => (
            crate::gpu::q4_byte_count(gate_up_rows, hidden),
            crate::gpu::q4_byte_count(hidden, moe_inter),
        ),
        Some(crate::gpu::ops::quant::QuantFormat::Q8) => (
            crate::gpu::q8_byte_count(gate_up_rows, hidden),
            crate::gpu::q8_byte_count(hidden, moe_inter),
        ),
        Some(crate::gpu::ops::quant::QuantFormat::FP8) => (
            crate::gpu::fp8_byte_count(gate_up_rows, hidden),
            crate::gpu::fp8_byte_count(hidden, moe_inter),
        ),
        Some(crate::gpu::ops::quant::QuantFormat::TQ3) => (
            crate::gpu::tq3_byte_count(gate_up_rows, hidden),
            crate::gpu::tq3_byte_count(hidden, moe_inter),
        ),
        Some(crate::gpu::ops::quant::QuantFormat::NVFP4) => (
            crate::gpu::nvfp4_byte_count(gate_up_rows, hidden),
            crate::gpu::nvfp4_byte_count(hidden, moe_inter),
        ),
        None => (
            gate_up_rows * hidden * 2, // bf16
            hidden * moe_inter * 2,
        ),
    }
}

/// Per-layer info for fused expert tensors (gate_up_proj + down_proj).
pub(crate) struct FusedLayerInfo {
    pub shard_gate_up: usize,
    pub shard_down: usize,
    /// Absolute file offset of the gate_up_proj tensor data.
    pub gate_up_file_offset: u64,
    /// Absolute file offset of the down_proj tensor data.
    pub down_file_offset: u64,
}

/// Per-expert info for separate expert tensors.
pub(crate) struct PerExpertInfo {
    pub shard_gate: usize,
    pub shard_up: usize,
    pub shard_down: usize,
    pub gate_file_offset: u64,
    pub up_file_offset: u64,
    pub down_file_offset: u64,
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_data_start_valid() {
        // 8-byte LE header_len = 16, then 16 bytes of header, then data.
        let mut mmap = vec![0u8; 32];
        mmap[..8].copy_from_slice(&16u64.to_le_bytes());
        assert_eq!(safetensors_data_start(&mmap), 24); // 8 + 16
    }

    #[test]
    fn test_safetensors_data_start_zero_header() {
        // header_len = 0: data starts immediately after the 8-byte length.
        let mut mmap = vec![0u8; 8];
        mmap[..8].copy_from_slice(&0u64.to_le_bytes());
        assert_eq!(safetensors_data_start(&mmap), 8);
    }

    #[test]
    #[should_panic(expected = "safetensors file too small")]
    fn test_safetensors_data_start_too_short() {
        // Only 4 bytes — should panic with a clear message.
        let mmap = vec![0u8; 4];
        safetensors_data_start(&mmap);
    }

    #[test]
    #[should_panic(expected = "header_len")]
    fn test_safetensors_data_start_header_exceeds_file() {
        // header_len = 1000, but file is only 16 bytes.
        let mut mmap = vec![0u8; 16];
        mmap[..8].copy_from_slice(&1000u64.to_le_bytes());
        safetensors_data_start(&mmap);
    }
}
