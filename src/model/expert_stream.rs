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
use std::fs::File;
use std::sync::Arc;

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
    /// Shard index for gate_up_proj tensor (into shard_files vec).
    pub shard_gate_up: usize,
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
    /// stacked along dim 0 within a single tensor.  When contiguous, we can
    /// read both in a single pread() instead of two — halving NVMe commands.
    fn gate_up_contiguous(&self) -> bool {
        self.up_offset == self.gate_offset + self.gate_bytes as u64
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
    pub layers: Vec<Vec<ExpertLocation>>,
    /// Open file handles for each shard (kept alive for pread).
    pub shard_files: Arc<Vec<File>>,
    /// Expert dimensions.
    pub hidden: usize,
    pub moe_inter: usize,
    /// Whether expert data on disk is already Q4 (pre-quantized model).
    /// When true, pread reads Q4 bytes directly — 3.2x less I/O than bf16.
    pub prequantized: bool,
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
// The streamer holds K pre-allocated GPU buffer slots (one per active
// expert).  When the router selects experts, load_experts() reads their
// weights from disk and copies them into the buffer slots.
//
// Performance: parallel pread across K experts, fused gate+up reads for
// models with contiguous gate/up layout (Qwen3.5).  On unified memory
// backends (Metal), the GPU upload is merged into the same parallel pass
// as the pread — each thread does pread→memcpy without a serial Phase 2.
// ---------------------------------------------------------------------------

pub(crate) struct ExpertStreamer<B: GpuCore> {
    /// Expert index (file locations for all experts).
    pub index: ExpertIndex,
    /// Double-buffered GPU slots.  Compute reads from `slots[active]`,
    /// async uploads target `slots[1 - active]`.  On Metal the two sets
    /// behave identically (sync uploads); on CUDA the inactive set
    /// receives DMA transfers while compute runs on the active set.
    slots: [Vec<ExpertSlot<B>>; 2],
    /// Which slot set (0 or 1) is ready for compute.  Uses `Cell` for
    /// interior mutability — inference is single-threaded.
    active: Cell<usize>,
    /// Per-slot CPU read buffers for parallel pread → GPU upload.
    ///
    /// UnsafeCell because load_experts() needs to write to these buffers
    /// through &self (matching the GPU tensor interior mutability pattern —
    /// inference is single-threaded within a model).  Each slot's buffers
    /// are accessed by exactly one thread during parallel pread.
    read_bufs: UnsafeCell<Vec<SlotReadBufs>>,
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
    /// Create a new streamer with K pre-allocated double-buffered slots.
    ///
    /// Each slot is sized for one expert's gate, up, and down projections.
    /// If prequantized, slots are allocated as Q4 tensors (smaller).
    /// Two sets are allocated (active + inactive) for double-buffering.
    pub fn new(backend: &B, index: ExpertIndex, k: usize) -> Self {
        let slots_a = Self::allocate_slots(backend, &index, k);
        let slots_b = Self::allocate_slots(backend, &index, k);

        // CPU read buffers: sized for the on-disk format.
        // Pre-quantized: Q4 sizes (3.2x smaller).  Otherwise: bf16 sizes.
        let hidden = index.hidden;
        let moe_inter = index.moe_inter;
        let (gate_up_bytes, down_bytes) = if index.prequantized {
            (
                crate::gpu::q4_byte_count(moe_inter * 2, hidden),
                crate::gpu::q4_byte_count(hidden, moe_inter),
            )
        } else {
            (
                moe_inter * hidden * 2 * 2, // gate + up combined (bf16)
                hidden * moe_inter * 2,
            )
        };
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

        ExpertStreamer {
            index,
            slots: [slots_a, slots_b],
            active: Cell::new(0),
            read_bufs,
        }
    }

    /// Allocate one set of K GPU buffer slots for expert weights.
    fn allocate_slots(backend: &B, index: &ExpertIndex, k: usize) -> Vec<ExpertSlot<B>> {
        let hidden = index.hidden;
        let moe_inter = index.moe_inter;
        let use_q4 = index.prequantized;
        let dtype = if use_q4 { TensorDtype::Q4 } else { TensorDtype::BF16 };

        (0..k)
            .map(|_| ExpertSlot {
                gate_proj: backend.alloc_tensor(&[moe_inter, hidden], dtype),
                up_proj: backend.alloc_tensor(&[moe_inter, hidden], dtype),
                down_proj: backend.alloc_tensor(&[hidden, moe_inter], dtype),
            })
            .collect()
    }

    /// Get the active (ready-to-compute) expert slots.
    pub fn active_slots(&self) -> &[ExpertSlot<B>] {
        &self.slots[self.active.get()]
    }

    /// Load selected experts from disk into GPU buffer slots.
    ///
    /// Fully parallel approach: each of K threads handles one expert end-to-end:
    ///   1. pread expert weights from SSD (fused gate+up where possible)
    ///   2. copy data into GPU buffer (memcpy to unified memory, or via trait)
    ///
    /// On unified memory backends (Metal/CPU), both phases run in the same
    /// thread — the GPU upload is just a memcpy to buffer.contents(), which
    /// is safe from multiple threads writing to disjoint buffers.  On CUDA
    /// (where tensor_mut_ptr returns None), Phase 2 falls back to serial
    /// copy_to_tensor after all threads join.
    ///
    /// Safety: takes &self (not &mut) because inference is single-threaded and
    /// GPU tensor writes already use interior mutability.  During parallel pread,
    /// each thread accesses only its own slot's buffers — no cross-slot access.
    pub fn load_experts(
        &self,
        backend: &B,
        layer_idx: usize,
        selected: &[(usize, f32)],
    ) {
        // Safety: single-threaded inference — no concurrent access to read_bufs
        // from other model code.  Within this function, each thread gets exclusive
        // access to exactly one slot's buffers via iter_mut().
        let read_bufs = unsafe { &mut *self.read_bufs.get() };
        let active = self.active.get();

        // Pre-fetch GPU buffer pointers before spawning threads.
        // On Metal/CPU: Some([gate_ptr, up_ptr, down_ptr]) — direct memcpy target.
        // On CUDA: None — must use copy_to_tensor after threads join.
        let gpu_ptrs: Vec<Option<[SendPtr; 3]>> = selected
            .iter()
            .enumerate()
            .map(|(slot_idx, _)| {
                let slot = &self.slots[active][slot_idx];
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

        // Phase 1 + Phase 2 (fused per-thread): parallel pread + GPU upload.
        //
        // Each thread reads one expert (gate+up + down).  pread() is thread-safe
        // (no shared file position), and File is Sync so &File is Send.
        // std::thread::scope ensures all threads join before we return.
        //
        // For pre-quantized models, reads are 3.2x smaller (Q4 vs bf16) — the
        // main performance win of pre-quantization.
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(selected.len());

            let iter = read_bufs
                .iter_mut()
                .zip(selected.iter())
                .zip(gpu_ptrs.iter());

            for ((slot_bufs, &(expert_idx, _)), &ptrs) in iter {
                let loc = &self.index.layers[layer_idx][expert_idx];
                let shard_files = &self.index.shard_files;

                handles.push(s.spawn(move || {
                    // Phase 1: pread from disk into CPU staging buffer.
                    // Gate + up: single pread if contiguous (fused format), else two.
                    let gate_up = slot_bufs.gate_up.as_mut_slice();
                    if loc.gate_up_contiguous() {
                        let total = loc.gate_bytes + loc.up_bytes;
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut gate_up[..total],
                            loc.gate_offset,
                        );
                    } else {
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut gate_up[..loc.gate_bytes],
                            loc.gate_offset,
                        );
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut gate_up[loc.gate_bytes..loc.gate_bytes + loc.up_bytes],
                            loc.up_offset,
                        );
                    }

                    // Down projection (always a separate read).
                    pread_exact(
                        &shard_files[loc.shard_down],
                        &mut slot_bufs.down.as_mut_slice()[..loc.down_bytes],
                        loc.down_offset,
                    );

                    // Phase 2 (direct): memcpy from staging buffer to GPU buffer.
                    // On Metal/CPU, buffer.contents() is CPU-writable unified memory.
                    // Each thread writes to its own expert's GPU buffers (disjoint),
                    // so no synchronization needed between threads.
                    if let Some([g, u, d]) = ptrs {
                        let gate_up = slot_bufs.gate_up.as_slice();
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
                            std::ptr::copy_nonoverlapping(
                                slot_bufs.down.as_slice().as_ptr(),
                                d.0,
                                loc.down_bytes,
                            );
                        }
                    }
                }));
            }

            for h in handles {
                h.join().unwrap();
            }
        });

        // Fallback Phase 2 (async DMA): for backends without direct CPU→GPU writes.
        // CUDA enqueues HtoD transfers on a dedicated transfer stream, then waits
        // via event synchronisation.  With pinned staging buffers, all K×3 transfers
        // run asynchronously on the DMA engine.
        if !direct {
            for (slot_idx, &(expert_idx, _)) in selected.iter().enumerate() {
                let loc = &self.index.layers[layer_idx][expert_idx];
                let slot = &self.slots[self.active.get()][slot_idx];
                let bufs = &read_bufs[slot_idx];
                let gate_up = bufs.gate_up.as_slice();

                backend.copy_to_tensor_async(&slot.gate_proj, &gate_up[..loc.gate_bytes]);
                backend.copy_to_tensor_async(
                    &slot.up_proj,
                    &gate_up[loc.gate_bytes..loc.gate_bytes + loc.up_bytes],
                );
                backend.copy_to_tensor_async(&slot.down_proj, &bufs.down.as_slice()[..loc.down_bytes]);
            }
            backend.sync_transfers();
        }
    }

    /// Load experts into the inactive buffer set using async GPU transfers.
    ///
    /// Phase 1: parallel pread from SSD (identical to load_experts).
    /// Phase 2: async GPU upload via copy_to_tensor_async into the INACTIVE slots.
    ///
    /// After calling this, the caller must call `sync_and_swap()` before
    /// computing on the newly loaded data.  On Metal, this behaves identically
    /// to load_experts() + swap because copy_to_tensor_async defaults to sync.
    pub fn load_experts_async(
        &self,
        backend: &B,
        layer_idx: usize,
        selected: &[(usize, f32)],
    ) {
        let read_bufs = unsafe { &mut *self.read_bufs.get() };

        // Phase 1: Parallel pread from disk into per-slot CPU buffers.
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(selected.len());

            for (slot_bufs, &(expert_idx, _)) in read_bufs.iter_mut().zip(selected.iter()) {
                let loc = &self.index.layers[layer_idx][expert_idx];
                let shard_files = &self.index.shard_files;

                handles.push(s.spawn(move || {
                    let gate_up = slot_bufs.gate_up.as_mut_slice();
                    if loc.gate_up_contiguous() {
                        let total = loc.gate_bytes + loc.up_bytes;
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut gate_up[..total],
                            loc.gate_offset,
                        );
                    } else {
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut gate_up[..loc.gate_bytes],
                            loc.gate_offset,
                        );
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut gate_up[loc.gate_bytes..loc.gate_bytes + loc.up_bytes],
                            loc.up_offset,
                        );
                    }

                    pread_exact(
                        &shard_files[loc.shard_down],
                        &mut slot_bufs.down.as_mut_slice()[..loc.down_bytes],
                        loc.down_offset,
                    );
                }));
            }

            for h in handles {
                h.join().unwrap();
            }
        });

        // Phase 2: Async GPU upload into INACTIVE slots.
        let inactive = 1 - self.active.get();
        for (slot_idx, &(expert_idx, _)) in selected.iter().enumerate() {
            let loc = &self.index.layers[layer_idx][expert_idx];
            let slot = &self.slots[inactive][slot_idx];
            let bufs = &read_bufs[slot_idx];
            let gate_up = bufs.gate_up.as_slice();

            backend.copy_to_tensor_async(&slot.gate_proj, &gate_up[..loc.gate_bytes]);
            backend.copy_to_tensor_async(
                &slot.up_proj,
                &gate_up[loc.gate_bytes..loc.gate_bytes + loc.up_bytes],
            );
            backend.copy_to_tensor_async(&slot.down_proj, &bufs.down.as_slice()[..loc.down_bytes]);
        }
    }

    /// Wait for async transfers to complete, then swap active/inactive.
    ///
    /// After this returns, `active_slots()` points to the newly uploaded data.
    /// On Metal, `sync_transfers()` is a no-op (uploads were already synchronous).
    pub fn sync_and_swap(&self, backend: &B) {
        backend.sync_transfers();
        self.active.set(1 - self.active.get());
    }
}

// ---------------------------------------------------------------------------
// pread — read bytes from a file at a specific offset without seeking.
//
// Uses the Unix pread() syscall which is thread-safe (no shared file
// position) and efficient (single syscall, no lseek + read race).
// ---------------------------------------------------------------------------

fn pread_exact(file: &File, buf: &mut [u8], offset: u64) {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
        .expect("pread failed during expert streaming");
}

// ---------------------------------------------------------------------------
// Index building helpers — used by loader.rs.
// ---------------------------------------------------------------------------

/// Compute the absolute file offset where tensor data begins in a safetensors file.
///
/// Safetensors layout: [8-byte LE header_len][JSON header (header_len bytes)][tensor data...]
/// The `data_offset` from the JSON header is relative to the start of the tensor data region.
pub(crate) fn safetensors_data_start(mmap: &[u8]) -> u64 {
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap());
    8 + header_len
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
    prequantized: bool,
) -> ExpertIndex {
    // Byte sizes depend on whether experts are stored as Q4 or bf16 on disk.
    // Q4 quantization is per-row, so fused gate+up Q4 data is still contiguous
    // per expert: [gate Q4 rows | up Q4 rows] within each expert's slice.
    let (gate_up_expert_bytes, gate_bytes, down_expert_bytes) = if prequantized {
        (
            crate::gpu::q4_byte_count(moe_inter * 2, hidden),
            crate::gpu::q4_byte_count(moe_inter, hidden),
            crate::gpu::q4_byte_count(hidden, moe_inter),
        )
    } else {
        (
            moe_inter * 2 * hidden * 2, // bf16
            moe_inter * hidden * 2,
            hidden * moe_inter * 2,
        )
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
        prequantized,
    }
}

/// Build an ExpertIndex for per-expert tensor format (Qwen3-MoE, Mixtral).
///
/// Each expert has separate tensors:
///   experts.{j}.gate_proj: [moe_inter, hidden]
///   experts.{j}.up_proj:   [moe_inter, hidden]
///   experts.{j}.down_proj: [hidden, moe_inter]
pub(crate) fn build_per_expert_index(
    layer_info: Vec<Vec<PerExpertInfo>>,
    shard_files: Vec<File>,
    hidden: usize,
    moe_inter: usize,
    prequantized: bool,
) -> ExpertIndex {
    let (gate_bytes, down_bytes) = if prequantized {
        (
            crate::gpu::q4_byte_count(moe_inter, hidden),
            crate::gpu::q4_byte_count(hidden, moe_inter),
        )
    } else {
        (moe_inter * hidden * 2, hidden * moe_inter * 2)
    };

    let layers = layer_info
        .iter()
        .map(|experts| {
            experts
                .iter()
                .map(|info| ExpertLocation {
                    shard_gate_up: info.shard_gate,
                    shard_down: info.shard_down,
                    gate_offset: info.gate_file_offset,
                    up_offset: info.up_file_offset,
                    down_offset: info.down_file_offset,
                    gate_bytes,
                    up_bytes: gate_bytes,
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
        prequantized,
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
