mod backend;
mod kernels;
mod tensor;

pub(crate) use backend::MetalBackend;
// Re-exported so gpu::Backend::Tensor resolves through the module boundary.
#[allow(unused_imports)]
pub(crate) use tensor::MetalTensor;
