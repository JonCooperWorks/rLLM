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
//   - **Parallel pread**: K experts are read concurrently using std::thread::scope.
//     Each thread reads one expert's weights independently.  pread() is thread-safe
//     (no shared file position) so multiple threads can read from the same file.
//   - **Fused gate+up read**: For Qwen3.5's fused format, gate and up projections
//     are contiguous on disk.  Instead of 2 pread() calls (gate, up), we do 1
//     pread() for the combined gate_up block — halving NVMe command overhead.
//   - **OS page cache**: No custom LRU.  Flash-moe tested custom caching and found
//     it 38% slower than trusting the OS page cache (~71% hit rate naturally).
//   - **No pipeline overlap**: Apple Silicon's unified memory controller means GPU
//     compute and NVMe DMA can't truly overlap — serial I/O→compute is optimal.
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

use std::cell::UnsafeCell;
use std::fs::File;

use crate::gpu::{GpuCore, TensorDtype, quantize_bf16_to_q4};

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
// ---------------------------------------------------------------------------

pub(crate) struct ExpertIndex {
    /// Per-layer expert locations: layers[layer_idx][expert_idx].
    pub layers: Vec<Vec<ExpertLocation>>,
    /// Open file handles for each shard (kept alive for pread).
    pub shard_files: Vec<File>,
    /// Expert dimensions.
    pub hidden: usize,
    pub moe_inter: usize,
    /// Whether to Q4-quantize experts when loading them into GPU buffers.
    pub quantize: bool,
}

// ---------------------------------------------------------------------------
// SlotReadBufs — per-slot CPU buffers for parallel pread.
//
// Each expert slot gets its own read buffers so that K threads can pread()
// concurrently without contention.  The gate_up buffer holds both gate and
// up projections contiguously (matching the fused on-disk layout) so that
// Qwen3.5's fused format can be loaded with a single pread.
// ---------------------------------------------------------------------------

struct SlotReadBufs {
    /// Combined gate+up buffer.  For fused format, read in one pread().
    /// Layout: [gate_bytes | up_bytes] — gate first, then up.
    gate_up: Vec<u8>,
    /// Down projection buffer.
    down: Vec<u8>,
}

// ---------------------------------------------------------------------------
// ExpertStreamer — manages GPU buffer slots and loads experts on demand.
//
// The streamer holds K pre-allocated GPU buffer slots (one per active
// expert).  When the router selects experts, load_experts() reads their
// weights from disk and copies them into the buffer slots.
//
// Performance: parallel pread across K experts, fused gate+up reads for
// models with contiguous gate/up layout (Qwen3.5).  For K=10 on 397B,
// this saturates NVMe queue depth instead of issuing 30 serial reads.
// ---------------------------------------------------------------------------

pub(crate) struct ExpertStreamer<B: GpuCore> {
    /// Expert index (file locations for all experts).
    pub index: ExpertIndex,
    /// K buffer slots for active experts.  Reused every layer.
    pub slots: Vec<ExpertSlot<B>>,
    /// Per-slot CPU read buffers for parallel pread → GPU upload.
    ///
    /// UnsafeCell because load_experts() needs to write to these buffers
    /// through &self (matching the GPU tensor interior mutability pattern —
    /// inference is single-threaded within a model).  Each slot's buffers
    /// are accessed by exactly one thread during parallel pread.
    read_bufs: UnsafeCell<Vec<SlotReadBufs>>,
}

/// One expert's worth of GPU-resident weight buffers.
pub(crate) struct ExpertSlot<B: GpuCore> {
    pub gate_proj: B::Tensor,
    pub up_proj: B::Tensor,
    pub down_proj: B::Tensor,
}

impl<B: GpuCore> ExpertStreamer<B> {
    /// Create a new streamer with K pre-allocated buffer slots.
    ///
    /// Each slot is sized for one expert's gate, up, and down projections.
    /// If quantize is true, slots are allocated as Q4 tensors (smaller).
    pub fn new(backend: &B, index: ExpertIndex, k: usize) -> Self {
        let hidden = index.hidden;
        let moe_inter = index.moe_inter;
        let quantize = index.quantize;

        let (gate_dtype, up_dtype, down_dtype) = if quantize {
            (TensorDtype::Q4, TensorDtype::Q4, TensorDtype::Q4)
        } else {
            (TensorDtype::BF16, TensorDtype::BF16, TensorDtype::BF16)
        };

        let mut slots = Vec::with_capacity(k);
        for _ in 0..k {
            slots.push(ExpertSlot {
                gate_proj: backend.alloc_tensor(&[moe_inter, hidden], gate_dtype),
                up_proj: backend.alloc_tensor(&[moe_inter, hidden], up_dtype),
                down_proj: backend.alloc_tensor(&[hidden, moe_inter], down_dtype),
            });
        }

        // CPU read buffers: per-slot gate_up + down for parallel pread.
        let gate_up_bytes = moe_inter * hidden * 2 * 2; // gate + up combined (bf16)
        let down_bytes = hidden * moe_inter * 2;
        let read_bufs = UnsafeCell::new(
            (0..k)
                .map(|_| SlotReadBufs {
                    gate_up: vec![0u8; gate_up_bytes],
                    down: vec![0u8; down_bytes],
                })
                .collect(),
        );

        ExpertStreamer {
            index,
            slots,
            read_bufs,
        }
    }

    /// Load selected experts from disk into GPU buffer slots.
    ///
    /// Two-phase approach:
    ///   Phase 1: Parallel pread — K threads read experts concurrently from SSD.
    ///            For fused formats (Qwen3.5), gate+up are read in a single pread.
    ///   Phase 2: Serial GPU upload — copy data into Metal buffers (not thread-safe).
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
        let hidden = self.index.hidden;
        let moe_inter = self.index.moe_inter;
        let quantize = self.index.quantize;

        // Safety: single-threaded inference — no concurrent access to read_bufs
        // from other model code.  Within this function, each thread gets exclusive
        // access to exactly one slot's buffers via iter_mut().
        let read_bufs = unsafe { &mut *self.read_bufs.get() };

        // Phase 1: Parallel pread from disk into per-slot CPU buffers.
        //
        // Each thread reads one expert (gate+up + down).  pread() is thread-safe
        // (no shared file position), and File is Sync so &File is Send.
        // std::thread::scope ensures all threads join before we proceed to GPU upload.
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(selected.len());

            for (slot_bufs, &(expert_idx, _)) in read_bufs.iter_mut().zip(selected.iter()) {
                let loc = &self.index.layers[layer_idx][expert_idx];
                let shard_files = &self.index.shard_files;

                handles.push(s.spawn(move || {
                    // Gate + up: single pread if contiguous (fused format), else two.
                    if loc.gate_up_contiguous() {
                        let total = loc.gate_bytes + loc.up_bytes;
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut slot_bufs.gate_up[..total],
                            loc.gate_offset,
                        );
                    } else {
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut slot_bufs.gate_up[..loc.gate_bytes],
                            loc.gate_offset,
                        );
                        pread_exact(
                            &shard_files[loc.shard_gate_up],
                            &mut slot_bufs.gate_up[loc.gate_bytes..loc.gate_bytes + loc.up_bytes],
                            loc.up_offset,
                        );
                    }

                    // Down projection (always a separate read).
                    pread_exact(
                        &shard_files[loc.shard_down],
                        &mut slot_bufs.down[..loc.down_bytes],
                        loc.down_offset,
                    );
                }));
            }

            for h in handles {
                h.join().unwrap();
            }
        });

        // Phase 2: GPU upload (serial — Metal command encoding isn't thread-safe).
        //
        // Optionally Q4-quantize on CPU before uploading.  The gate_up buffer
        // holds [gate | up] contiguously, so we slice at gate_bytes to split.
        for (slot_idx, &(expert_idx, _)) in selected.iter().enumerate() {
            let loc = &self.index.layers[layer_idx][expert_idx];
            let slot = &self.slots[slot_idx];
            let bufs = &read_bufs[slot_idx];

            if quantize {
                let q4_gate = quantize_bf16_to_q4(
                    &bufs.gate_up[..loc.gate_bytes],
                    moe_inter,
                    hidden,
                );
                backend.copy_to_tensor(&slot.gate_proj, &q4_gate);

                let q4_up = quantize_bf16_to_q4(
                    &bufs.gate_up[loc.gate_bytes..loc.gate_bytes + loc.up_bytes],
                    moe_inter,
                    hidden,
                );
                backend.copy_to_tensor(&slot.up_proj, &q4_up);

                let q4_down = quantize_bf16_to_q4(
                    &bufs.down[..loc.down_bytes],
                    hidden,
                    moe_inter,
                );
                backend.copy_to_tensor(&slot.down_proj, &q4_down);
            } else {
                backend.copy_to_tensor(&slot.gate_proj, &bufs.gate_up[..loc.gate_bytes]);
                backend.copy_to_tensor(
                    &slot.up_proj,
                    &bufs.gate_up[loc.gate_bytes..loc.gate_bytes + loc.up_bytes],
                );
                backend.copy_to_tensor(&slot.down_proj, &bufs.down[..loc.down_bytes]);
            }
        }
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
    quantize: bool,
) -> ExpertIndex {
    let gate_up_expert_bytes = moe_inter * 2 * hidden * 2; // bf16
    let gate_bytes = moe_inter * hidden * 2;
    let down_expert_bytes = hidden * moe_inter * 2;

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
        shard_files,
        hidden,
        moe_inter,
        quantize,
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
    quantize: bool,
) -> ExpertIndex {
    let gate_bytes = moe_inter * hidden * 2;
    let down_bytes = hidden * moe_inter * 2;

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
        shard_files,
        hidden,
        moe_inter,
        quantize,
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
