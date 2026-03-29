// ---------------------------------------------------------------------------
// NCCL communicator initialization for multi-GPU tensor parallelism.
//
// Uses cudarc's raw NCCL bindings to create one communicator per GPU device.
// We use the raw API (ncclComm_t) rather than the safe Comm wrapper because
// the allreduce trait requires in-place operations through shared tensor
// references, which conflicts with the safe API's mutability requirements.
// ---------------------------------------------------------------------------

use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::nccl::sys as nccl_sys;

/// Wrapper around a raw NCCL communicator handle.
///
/// Stores the `ncclComm_t` alongside the device ordinal and stream handle
/// for use in allreduce operations.
#[allow(dead_code)]
pub(crate) struct NcclComm {
    pub(crate) comm: nccl_sys::ncclComm_t,
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
}

// Safety: NCCL communicators are thread-safe for collective operations.
// Each comm is bound to a specific device and used from the backend that
// owns that device.  The raw pointer itself is just an opaque handle.
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

impl Drop for NcclComm {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::nccl::result::comm_abort(self.comm);
        }
    }
}

/// Initialize NCCL communicators for `world_size` GPU devices.
///
/// Creates one raw `ncclComm_t` per device using `ncclCommInitAll` (the
/// single-process multi-device initialization path).
pub(crate) fn init_nccl_comms(world_size: usize) -> anyhow::Result<Vec<Arc<NcclComm>>> {
    // Validate device count.
    let device_count = CudaContext::device_count()
        .map_err(|e| anyhow::anyhow!("failed to query CUDA device count: {e}"))?;
    anyhow::ensure!(
        world_size <= device_count as usize,
        "requested {world_size} GPUs but only {device_count} available"
    );

    // ncclCommInitAll creates all communicators at once for single-process use.
    let ordinals: Vec<i32> = (0..world_size as i32).collect();
    let mut raw_comms = vec![std::ptr::null_mut(); world_size];

    unsafe {
        cudarc::nccl::result::comm_init_all(
            raw_comms.as_mut_ptr(),
            world_size as i32,
            ordinals.as_ptr(),
        )
        .map_err(|e| anyhow::anyhow!("ncclCommInitAll failed: {e:?}"))?;
    }

    let comms: Vec<Arc<NcclComm>> = raw_comms
        .into_iter()
        .enumerate()
        .map(|(rank, comm)| {
            Arc::new(NcclComm {
                comm,
                rank,
                world_size,
            })
        })
        .collect();

    Ok(comms)
}
