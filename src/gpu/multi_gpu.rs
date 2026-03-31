// ===========================================================================
// Multi-GPU inference orchestrator for tensor parallelism.
//
// Manages N GPU backends (one per rank), each with its own:
//   - CudaBackend with NCCL communicator
//   - Model weights (sharded according to ShardingPlan)
//   - KV cache (per-rank, with kv_dim / world_size heads)
//   - Scratch buffers (TP-aware sizing)
//
// The forward pass fans out to all ranks in parallel using
// std::thread::scope.  Each rank runs the same model forward pass
// with its own sharded weights.  AllReduce calls (wired into
// primitives.rs) synchronize partial results via NCCL.
//
// Only rank 0's logits are used for sampling — all ranks produce
// identical logits after the final AllReduce.
// ===========================================================================

#[cfg(feature = "cuda")]
pub(crate) mod tp {
    use std::path::Path;

    use crate::gpu::cuda::CudaBackend;
    use crate::gpu::parallel::{DeviceConfig, ParallelStrategy, ShardingPlan};
    use crate::gpu::GpuCore;
    use crate::model::config::ModelConfig;
    use crate::model::forward::ModelForward;
    use crate::model::kv_cache::{KvPool, SeqKvState};
    use crate::model::loader;
    use crate::model::{Model, PrefillBuffers};

    /// Per-rank inference state: backend, model, KV cache, prefill buffers.
    ///
    /// The backend is boxed for a stable heap address (Model borrows it).
    /// Safety: backend outlives model (both in RankState, drop order is
    /// fields top-down in declaration order).
    #[allow(dead_code)]
    pub(crate) struct RankState {
        // Order matters for drop: model and forward must drop before backend.
        pub model: Model<'static, CudaBackend>,
        pub forward: Box<dyn ModelForward<CudaBackend>>,
        pub prefill_bufs: PrefillBuffers<CudaBackend>,
        pub kv_pool: KvPool<CudaBackend>,
        pub seq_state: SeqKvState<CudaBackend>,
        // Backend is kept alive by the 'static lifetime trick below.
        _backend: Box<CudaBackend>,
    }

    /// Multi-GPU inference controller.
    #[allow(dead_code)]
    pub(crate) struct MultiGpuInference {
        pub ranks: Vec<RankState>,
        pub config: ModelConfig,
        pub world_size: usize,
    }

    #[allow(dead_code)]
    impl MultiGpuInference {
        /// Create multi-GPU inference with `world_size` GPUs.
        pub fn new(
            model_dir: &Path,
            config: ModelConfig,
            is_prequantized: bool,
            world_size: usize,
            num_blocks: usize,
        ) -> anyhow::Result<Self> {
            let backends = crate::gpu::create_backends(world_size)?;

            let mut ranks = Vec::with_capacity(world_size);

            for (rank, backend) in backends.into_iter().enumerate() {
                let backend = Box::new(backend);

                // Create sharding plan for this rank.
                // MoE models use Hybrid strategy: TP for attention, EP for MoE FFN.
                // This assigns whole experts to ranks instead of splitting each
                // expert's matrices, requiring only one AllReduce per MoE layer.
                let strategy = if config.is_moe() {
                    ParallelStrategy::Hybrid
                } else {
                    ParallelStrategy::TensorParallel
                };
                let device = DeviceConfig {
                    world_size,
                    rank,
                    strategy,
                };
                let plan = ShardingPlan::derive(&config, device, false)?;

                // Load sharded weights.
                let weights =
                    loader::load_weights(&*backend, model_dir, &config, Some(&plan))?;

                if rank == 0 {
                    tracing::info!(
                        quantized = is_prequantized,
                        rank = 0,
                        world_size = world_size,
                        "weights loaded",
                    );
                }

                // TP-aware KV cache: each rank has kv_dim / world_size heads.
                let kv_dim = (config.num_key_value_heads / world_size) * config.head_dim;
                let num_kv_layers = config.num_kv_layers();
                let kv_pool = KvPool::new(&*backend, num_blocks, kv_dim, num_kv_layers, crate::model::turboquant::KvQuantPair::symmetric(crate::model::turboquant::KvQuantMode::None), config.head_dim);
                let seq_state = kv_pool.new_sequence(&*backend);

                // TP-aware scratch buffers.
                let prefill_bufs = PrefillBuffers::new_tp(&*backend, &config, 4096, world_size);

                // Safety: we transmute the backend lifetime to 'static.
                // The backend is boxed and stored in the same struct, so it
                // outlives the model.  This is the same pattern used by many
                // self-referential struct solutions.
                let backend_ref: &'static CudaBackend =
                    unsafe { &*(&*backend as *const CudaBackend) };

                let mut model = Model::new_tp(config.clone(), weights, backend_ref, world_size)?;

                // Load vision encoder weights (replicated on every rank).
                // Vision is small (~0.6B params) relative to the LLM, so
                // replication is simpler and faster than sharding the ViT.
                if config.vision.is_some() {
                    use std::collections::HashMap;
                    use safetensors::SafeTensors;
                    let (mmaps, weight_map) = loader::load_safetensors_files(model_dir)?;
                    let shards: Vec<SafeTensors> = mmaps
                        .iter()
                        .map(|m| SafeTensors::deserialize(m))
                        .collect::<Result<_, _>>()
                        .map_err(|e| anyhow::anyhow!("failed to parse safetensors for vision: {e}"))?;
                    let store = loader::TensorStore {
                        shards,
                        weight_map,
                        q4_map: HashMap::new(),
                        q8_map: HashMap::new(),
                        fp8_map: HashMap::new(),
                    };
                    if let Some(vw) = loader::load_vision_weights(backend_ref, &store, &config) {
                        if let Some(vc) = &config.vision {
                            let max_patches = vc.max_pixels / (vc.patch_size * vc.patch_size);
                            let bufs = crate::model::vision::alloc_vision_buffers(backend_ref, vc, max_patches);
                            model.vision_weights = Some(vw);
                            model.vision_bufs = Some(bufs);
                            if rank == 0 {
                                tracing::info!(
                                    blocks = vc.depth,
                                    hidden = vc.hidden_size,
                                    max_patches = max_patches,
                                    "vision encoder ready",
                                );
                            }
                        }
                    }
                }

                let use_ep = matches!(strategy, ParallelStrategy::ExpertParallel | ParallelStrategy::Hybrid);
                let forward = crate::engine::loader::create_forward(config.arch()?, &config, backend_ref, world_size, rank, use_ep, None);

                ranks.push(RankState {
                    model,
                    forward,
                    prefill_bufs,
                    kv_pool,
                    seq_state,
                    _backend: backend,
                });
            }

            Ok(Self {
                ranks,
                config,
                world_size,
            })
        }

        /// Run a single-token forward pass on all ranks in parallel.
        pub fn forward_single_paged(&self, token_id: u32) -> anyhow::Result<()> {
            if self.world_size == 1 {
                let r = &self.ranks[0];
                return r.forward.forward_decode(
                    &r.model, token_id, &r.kv_pool, &r.seq_state,
                );
            }

            // Fan out to all ranks.
            std::thread::scope(|s| {
                let handles: Vec<_> = self
                    .ranks
                    .iter()
                    .map(|rank| {
                        s.spawn(move || {
                            rank.forward.forward_decode(
                                &rank.model,
                                token_id,
                                &rank.kv_pool,
                                &rank.seq_state,
                            )
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap()?;
                }
                Ok(())
            })
        }

        /// Run batched prefill on all ranks in parallel.
        pub fn forward_prefill_paged(&self, tokens: &[u32], images: &[crate::model::vision::ProcessedImage]) -> anyhow::Result<()> {
            if self.world_size == 1 {
                let r = &self.ranks[0];
                r.forward.prefill_preamble(
                    &r.model, tokens, &r.seq_state, &r.prefill_bufs, images,
                )?;
                return r.forward.forward_prefill(
                    &r.model, tokens, &r.kv_pool, &r.seq_state, &r.prefill_bufs,
                );
            }

            std::thread::scope(|s| {
                let handles: Vec<_> = self
                    .ranks
                    .iter()
                    .map(|rank| {
                        s.spawn(move || {
                            rank.forward.prefill_preamble(
                                &rank.model, tokens, &rank.seq_state,
                                &rank.prefill_bufs, images,
                            )?;
                            rank.forward.forward_prefill(
                                &rank.model, tokens, &rank.kv_pool,
                                &rank.seq_state, &rank.prefill_bufs,
                            )
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap()?;
                }
                Ok(())
            })
        }

        /// Get rank 0's logits (all ranks have identical logits after AllReduce).
        pub fn logits(&self) -> &<CudaBackend as GpuCore>::Tensor {
            self.ranks[0].model.logits()
        }

        /// Get rank 0's backend (for sampling, etc.).
        pub fn backend(&self) -> &CudaBackend {
            self.ranks[0].model.backend
        }

        /// Ensure KV slots on all ranks.
        pub fn ensure_slot(&mut self) -> anyhow::Result<()> {
            for rank in &mut self.ranks {
                rank.seq_state.ensure_slot(&mut rank.kv_pool)?;
                rank.seq_state.sync_block_table(rank.model.backend);
            }
            Ok(())
        }

        /// Ensure KV slots for prefill on all ranks.
        pub fn ensure_slots(&mut self, count: usize) -> anyhow::Result<()> {
            for rank in &mut self.ranks {
                rank.seq_state.ensure_slots(&mut rank.kv_pool, count)?;
                rank.seq_state.sync_block_table(rank.model.backend);
            }
            Ok(())
        }

        /// Advance KV state on all ranks.
        pub fn advance(&mut self) {
            for rank in &mut self.ranks {
                rank.seq_state.advance();
            }
        }

        /// Advance KV state by count on all ranks.
        pub fn advance_by(&mut self, count: usize) {
            for rank in &mut self.ranks {
                rank.seq_state.advance_by(count);
            }
        }

        /// Reset for a new sequence: free all KV blocks and create fresh state.
        ///
        /// Called between requests in the API server so we can reuse the same
        /// MultiGpuInference instance without reallocating backends/models.
        pub fn reset(&mut self) {
            for rank in &mut self.ranks {
                rank.kv_pool.free_sequence(&rank.seq_state);
                rank.seq_state = rank.kv_pool.new_sequence(rank.model.backend);
            }
        }

        // -------------------------------------------------------------------
        // Multi-sequence methods for continuous batching.
        //
        // The methods above use a single internal seq_state per rank (for
        // `rllm run` which processes one sequence).  The methods below accept
        // external per-rank KV states, allowing the API server to manage
        // multiple concurrent sequences with the same MultiGpuInference.
        // -------------------------------------------------------------------

        /// Create a new per-rank KV state set for a new sequence.
        pub fn new_sequence(&self) -> Vec<SeqKvState<CudaBackend>> {
            self.ranks
                .iter()
                .map(|r| r.kv_pool.new_sequence(r.model.backend))
                .collect()
        }

        /// Free all KV blocks for a sequence across all ranks.
        pub fn free_sequence(&mut self, states: &[SeqKvState<CudaBackend>]) {
            for (rank, state) in self.ranks.iter_mut().zip(states) {
                rank.kv_pool.free_sequence(state);
            }
        }

        /// Allocate one KV slot on all ranks for an external sequence.
        pub fn ensure_slot_for(
            &mut self,
            states: &mut [SeqKvState<CudaBackend>],
        ) -> anyhow::Result<()> {
            for (rank, state) in self.ranks.iter_mut().zip(states.iter_mut()) {
                state.ensure_slot(&mut rank.kv_pool)?;
                state.sync_block_table(rank.model.backend);
            }
            Ok(())
        }

        /// Allocate KV slots for prefill on all ranks for an external sequence.
        pub fn ensure_slots_for(
            &mut self,
            states: &mut [SeqKvState<CudaBackend>],
            count: usize,
        ) -> anyhow::Result<()> {
            for (rank, state) in self.ranks.iter_mut().zip(states.iter_mut()) {
                state.ensure_slots(&mut rank.kv_pool, count)?;
                state.sync_block_table(rank.model.backend);
            }
            Ok(())
        }

        /// Advance KV state on all ranks for an external sequence.
        pub fn advance_for(states: &mut [SeqKvState<CudaBackend>]) {
            for state in states {
                state.advance();
            }
        }

        /// Advance KV state by count on all ranks for an external sequence.
        pub fn advance_by_for(states: &mut [SeqKvState<CudaBackend>], count: usize) {
            for state in states {
                state.advance_by(count);
            }
        }

        /// Run single-token forward pass with external per-rank KV states.
        pub fn forward_single_paged_with(
            &self,
            token_id: u32,
            states: &[SeqKvState<CudaBackend>],
        ) -> anyhow::Result<()> {
            if self.world_size == 1 {
                let r = &self.ranks[0];
                return r.forward.forward_decode(
                    &r.model, token_id, &r.kv_pool, &states[0],
                );
            }

            std::thread::scope(|s| {
                let handles: Vec<_> = self
                    .ranks
                    .iter()
                    .zip(states.iter())
                    .map(|(rank, state)| {
                        s.spawn(move || {
                            rank.forward.forward_decode(
                                &rank.model, token_id, &rank.kv_pool, state,
                            )
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap()?;
                }
                Ok(())
            })
        }

        /// Run prefill forward pass with external per-rank KV states.
        pub fn forward_prefill_paged_with(
            &self,
            tokens: &[u32],
            states: &[SeqKvState<CudaBackend>],
            images: &[crate::model::vision::ProcessedImage],
        ) -> anyhow::Result<()> {
            if self.world_size == 1 {
                let r = &self.ranks[0];
                r.forward.prefill_preamble(
                    &r.model, tokens, &states[0], &r.prefill_bufs, images,
                )?;
                return r.forward.forward_prefill(
                    &r.model, tokens, &r.kv_pool, &states[0], &r.prefill_bufs,
                );
            }

            std::thread::scope(|s| {
                let handles: Vec<_> = self
                    .ranks
                    .iter()
                    .zip(states.iter())
                    .map(|(rank, state)| {
                        s.spawn(move || {
                            rank.forward.prefill_preamble(
                                &rank.model, tokens, state, &rank.prefill_bufs, images,
                            )?;
                            rank.forward.forward_prefill(
                                &rank.model, tokens, &rank.kv_pool, state,
                                &rank.prefill_bufs,
                            )
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap()?;
                }
                Ok(())
            })
        }
    }
}
