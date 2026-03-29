  Plan: Finish the rLLM CUDA Backend                                                                                                                                                                              
                                                                                                                                                                                                                     
     Context                                                                                                                                                                                                         
                                                                                                                                                                                                                     
     The rLLM CUDA backend is partially complete — 20 todo!()/unimplemented!() stubs remain across 9 kernel files. The Metal backend is the complete reference. This work unblocks: Nemotron-H (Mamba2), VLMs        
     (vision), KV cache compression (TurboQuant), and Q8 quantized models on CUDA/Linux.                                                                                                                             
                                                                                                                                                                                                                     
     Files Overview                                                                                                                                                                                                  
                                                                                                                                                                                                                     
     Kernel dispatch (Rust): src/gpu/cuda/kernels/{mamba2,vision,turboquant,matmul,moe,attention,elementwise,norm,core}.rs                                                                                           
     Shaders (CUDA C): src/gpu/cuda/shaders/{mamba2,elementwise,rms_norm,matmul,matmul_tc,moe,attention}.cu + new vision.cu, turboquant.cu                                                                           
     Backend init: src/gpu/cuda/backend.rs (struct fields, shader embedding, compilation, function loading)                                                                                                          
     Metal references: src/gpu/metal/shaders/*.metal + src/gpu/metal/kernels/*.rs                                                                                                                                    
                                                                                                                                                                                                                     
     Established Patterns to Follow                                                                                                                                                                                  
                                                                                                                                                                                                                     
     - Param structs: #[repr(C)] in Rust with unsafe impl DeviceRepr, matching C struct in .cu byte-for-byte                                                                                                         
     - Kernel signatures: extern "C" __global__ void name(const Params params, const __nv_bfloat16* __restrict__ ..., __nv_bfloat16* __restrict__ ...)                                                               
     - Warp reduction: __shfl_xor_sync(0xffffffff, val, offset) via warp_sum()/warp_max() helpers                                                                                                                    
     - Cross-warp reduction: __shared__ float shared[32] + __syncthreads()                                                                                                                                           
     - Launch configs: cfg_1d(total, 256), cfg_blocks(n, 256), cfg_2d_smem(gx, gy, bs, smem)                                                                                                                         
     - Backend loading: include_str!() → compile() → load() → func(&mod, "name")                                                                                                                                     
     - bf16: __nv_bfloat16, convert with __bfloat162float() / __float2bfloat16()                                                                                                                                     
                                                                                                                                                                                                                     
     ---                                                                                                                                                                                                             
     Step 1: Trivial Leaf Kernels                                                                                                                                                                                    
                                                                                                                                                                                                                     
     1a. copy_to_tensor_from_host — kernels/core.rs:168                                                                                                                                                              
                                                                                                                                                                                                                     
     - No shader needed. Use cudarc::driver::result::memcpy_htod_async (reverse of copy_to_host at line 172)                                                                                                         
     - Bind context, assert sizes, async copy src: &[u8] → dst.buf                                                                                                                                                   
                                                                                                                                                                                                                     
     1b. gelu — kernels/elementwise.rs:122 + shaders/elementwise.cu                                                                                                                                                  
                                                                                                                                                                                                                     
     - Add gelu_act kernel: 1 thread/element, cfg_1d(size, 256), reuse ElemParams                                                                                                                                    
     - Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))                                                                                                                                              
     - Backend: add fn_gelu_act field, load from mod_elementwise                                                                                                                                                     
                                                                                                                                                                                                                     
     1c. relu_squared — kernels/elementwise.rs:333 + shaders/elementwise.cu                                                                                                                                          
                                                                                                                                                                                                                     
     - Add relu_squared kernel: 1 thread/element, reuse ElemParams                                                                                                                                                   
     - Formula: max(0, x)^2                                                                                                                                                                                          
     - Backend: add fn_relu_squared field                                                                                                                                                                            
                                                                                                                                                                                                                     
     1d. top_k_sigmoid — kernels/elementwise.rs:346 + shaders/elementwise.cu                                                                                                                                         
                                                                                                                                                                                                                     
     - Add top_k_sigmoid kernel: single-thread (cfg_blocks(1, 1)), like existing top_k_softmax                                                                                                                       
     - New params struct TopKSigmoidParams { num_experts, k, scaling_factor, norm_topk_prob }                                                                                                                        
     - Port from metal/shaders/elementwise.metal sigmoid top-k routing logic                                                                                                                                         
     - Backend: add fn_top_k_sigmoid field                                                 
   1e. layer_norm_batch — kernels/norm.rs:93 + shaders/rms_norm.cu                                                                                                                               03:35:21 [169/297]
                                                                                                                                                                                                                     
     - Add layer_norm_batch kernel: 1 block/row (256 threads), like rms_norm_batch                                                                                                                                   
     - Two reductions (mean, variance) instead of one (sum-of-squares). Has weight + bias (unlike RMSNorm)
     - New params struct LayerNormBatchParams { hidden_size, eps, batch_size }
     - Backend: add fn_layer_norm_batch field, load from mod_rms_norm

     ---
     Step 2: Q8 Dequant Kernels

     Q8 block format: 34 bytes = 2-byte bf16 scale + 32 signed int8 values. Dequant: w = float(q) * scale.

     2a. matvec_q8 — shaders/matmul.cu

     - Port from Q4 pattern, simpler: 1 byte/weight (no nibble extraction)
     - 32 threads/row (warp-cooperative), same MatvecParams
     - Update kernels/matmul.rs:46: TensorDtype::Q8 => &self.fn_matvec_q8
     - Backend: add fn_matvec_q8

     2b. gemm_q8 — shaders/matmul.cu

     - Scalar batched GEMM with Q8 dequant, same GemmParams
     - Update kernels/matmul.rs:105: Q8 branch
     - Backend: add fn_gemm_q8

     2c. gemm_q8_tc — shaders/matmul_tc.cu

     - Tensor-core WMMA variant (sm_80+). Dequant Q8 blocks to bf16 tiles, then WMMA
     - Update kernels/matmul.rs:83: Q8 branch
     - Backend: add fn_gemm_q8_tc: Option<CudaFunction>

     2d. fused_gate_up_swiglu_q8 — shaders/moe.cu

     - Follow fused_gate_up_swiglu_q4 pattern with Q8 dequant
     - Update kernels/moe.rs:57: Q8 branch
     - Backend: add fn_fused_gate_up_swiglu_q8

     ---
     Step 3: Mamba2 Kernels (Nemotron-H)

     Replace stub shaders/mamba2.cu with 3 kernels. Port from metal/shaders/mamba2.metal.

     3a. mamba2_conv1d_silu — 1 thread/channel, cfg_1d(dim, 256)

     - Depthwise conv1d + bias + SiLU
     - Params: { dim, kernel_size, input_offset }

     3b. mamba2_ssm_step — 1 block/head (256 threads), cfg_blocks(num_heads, 256)

     - SSM state recurrence: thread 0 computes shared scalars (softplus dt, exp decay), all threads update state matrix
     - __shared__ for dt/dA/D scalars + y_shared[256]
     - Params: { num_heads, head_dim, state_size, n_groups, b_offset, c_offset, dt_offset, eps }

     3c. mamba2_gated_rms_norm — 1 block/group (256 threads), cfg_blocks(d_inner/group_size, 256)

     - Three phases: gated values (SiLU gate) + sum-of-squares, warp+shared reduction, normalize + write
     - Params: { d_inner, group_size, z_offset, eps }

     3d. Update kernels/mamba2.rs with param structs + launch_builder dispatch
     3e. Update backend.rs: embed shaders/mamba2.cu, compile, load 3 functions                                                                                                                                       
                                                                                                                                                                                                                     
     ---                                                                                                                                                                                                             
     Step 4: Vision Kernels (VLMs)                                                                                                                                                                                   
                                                                                                                                                                                                                     
     Create new shaders/vision.cu. Port from metal/shaders/vision.metal.                                                                                                                                             
                                                                                                                                                                                                                     
     4a. spatial_merge — 1 thread/element, cfg_1d(total, 256)                                                                                                                                                        
                                                                                                                                                                                                                     
     - Pure index arithmetic: rearrange 2D patches by merging spatial neighbors                                                                                                                                      
     - Params: { grid_h, grid_w, hidden_dim, merge_size }                                                                                                                                                            

     4b. spatial_merge_norm — 1 block/token (256 threads), cfg_blocks(num_tokens, 256)

     - Fused spatial merge + LayerNorm: gather patches, compute mean+variance via warp reduction, normalize
     - Params: { grid_h, grid_w, hidden_dim, merge_size, eps }

     4c. scatter_vision_tokens — 1 block, 256 threads, cfg_blocks(1, 256)

     - Serial scan over token_ids, parallel copy of matching rows
     - Params: { image_token_id, seq_len, hidden_dim }

     4d. prefill_attention_fused_qkv — shaders/attention.cu + kernels/attention.rs:395

     - 1 block/(token x head), bidirectional attention for ViT
     - QKV interleaved layout, no causal mask
     - Port from metal/shaders/attention.metal fused QKV kernel
     - Params: { chunk_size, num_heads, num_kv_heads, head_dim, attn_scale }
     - Backend: add fn_prefill_attention_fused_qkv (from mod_attn_128)

     4e. Update kernels/vision.rs with param structs + dispatch

     4f. Update backend.rs: embed shaders/vision.cu, compile, load 3 vision functions + 1 attention function

     ---
     Step 5: TurboQuant Kernels (KV Cache Compression)

     Create new shaders/turboquant.cu. Port from metal/shaders/turboquant.metal. Most complex family.

     5a. turbo_quantize_paged — 1 block/KV head, head_dim threads

     - L2 norm → rotate by Pi → nearest centroid quantize → pack codes into paged pool
     - Bit-packing via __shared__ to avoid byte-level data races
     - Params: { pos, num_kv_heads, head_dim, bits, bytes_per_head_pos, block_size, num_centroids }

     5b. turbo_quantize_paged_batch — batch_size * num_kv_heads blocks

     - Same as single but with positions[batch_idx] indexing


     5d. turbo_paged_attention — 1 block/query head, 256 threads

     - Online softmax with inline dequant from packed codes
     - Phases: load centroids+q → strided position loop (K dequant, score, V accumulate) → cross-thread softmax reduction → V reduction + Pi^T inverse rotation
     - Shared memory ~9KB (centroids + q + reduce buffers)
     - Params: { seq_len, num_heads, num_kv_heads, head_dim, bits, bytes_per_head_pos, block_size, num_centroids, window_size, attn_scale, has_sinks }

     5e. Port helpers: extract_code() (sub-byte code extraction), pack_codes_shared() (bit packing)

     5f. Update kernels/turboquant.rs with param structs + dispatch

     5g. Update backend.rs: embed shaders/turboquant.cu, compile, load 4 functions

     ---
     Step 6: Backend Consolidation

     All backend.rs changes across Steps 1-5, summarized:

     New struct fields (19 total):
     fn_gelu_act, fn_relu_squared, fn_top_k_sigmoid, fn_layer_norm_batch,
     fn_matvec_q8, fn_gemm_q8, fn_gemm_q8_tc: Option<CudaFunction>,
     fn_fused_gate_up_swiglu_q8,
     fn_mamba2_conv1d_silu, fn_mamba2_ssm_step, fn_mamba2_gated_rms_norm,
     fn_spatial_merge, fn_spatial_merge_norm, fn_scatter_vision_tokens,
     fn_prefill_attention_fused_qkv,
     fn_turbo_quantize_paged, fn_turbo_quantize_paged_batch,
     fn_turbo_rotate_q, fn_turbo_paged_attention

     New shader sources (2 new files + 1 replaced stub):
     CUDA_SOURCE_MAMBA2 (replace stub), CUDA_SOURCE_VISION (new), CUDA_SOURCE_TURBOQUANT (new)

     New modules: mod_mamba2, mod_vision, mod_turboquant

     ---
     Step 7: Test & Benchmark

     1. cargo build — verify compilation
     2. python tests/test_model_families.py — run full test suite, fix failures
     3. python tests/bench.py — capture benchmark results for all model families including Q8
     4. Update README.md with CUDA benchmark numbers alongside existing Metal numbers
     5. commit and push

     ---
     Verification Plan

 
     ┌────────────────────────────────────────────┬───────────────────────────────────────────────────────────┐
     │                    Test                    │                     What it validates                     │
     ├────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
     │ Build succeeds                             │ All param struct layouts match, all function handles load │
     ├────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
     │ test_model_families.py (dense models)      │ Steps 1-2 (elementwise, norm, Q8 matmul)                  │
     ├────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
     │ test_model_families.py (Nemotron-H)        │ Step 3 (Mamba2 kernels)                                   │
     ├────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
     │ test_model_families.py (Qwen 3.5 VLM)      │ Step 4 (vision + fused QKV attention)                     │
     ├────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
     │ test_model_families.py (with --kv-quant 4) │ Step 5 (TurboQuant)                                       │
     ├────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
     │ bench.py                                   │ Performance sanity check across all families              │
     └────────────────────────────────────────────┴───────────────────────────────────────────────────────────┘
