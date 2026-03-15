// ===========================================================================
// Forward pass profiling.
//
// When the `RLLM_PROFILE` environment variable is set (e.g. RLLM_PROFILE=1),
// the forward pass logs per-component timing breakdowns.  Each component
// is timed by calling backend.flush() to force GPU completion, then measuring
// wall-clock time.  This adds overhead (~1ms per flush) but gives accurate
// per-component timings.
//
// Usage:  RLLM_PROFILE=1 cargo run -- run --model ...
// ===========================================================================

use std::sync::OnceLock;
use std::time::Instant;

use crate::gpu::GpuCore;

/// Whether profiling is enabled (checked once, cached).
pub(crate) fn is_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("RLLM_PROFILE").is_ok())
}

/// Accumulated timing data for one forward pass.
pub(crate) struct ForwardProfile {
    start: Instant,
    embed_us: u64,
    attn_us: u64,
    ffn_us: u64,
    other_us: u64,
    token_count: u64,
}

static PROFILE: std::sync::Mutex<Option<ForwardProfile>> = std::sync::Mutex::new(None);

impl ForwardProfile {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            embed_us: 0,
            attn_us: 0,
            ffn_us: 0,
            other_us: 0,
            token_count: 0,
        }
    }
}

/// Start timing a component.  Returns the current instant.
pub(crate) fn begin<B: GpuCore>(backend: &B) -> Instant {
    if !is_enabled() {
        return Instant::now();
    }
    backend.flush();
    Instant::now()
}

/// Record elapsed time for a component.
pub(crate) fn record<B: GpuCore>(backend: &B, t: Instant, component: Component) {
    if !is_enabled() {
        return;
    }
    backend.flush();
    let elapsed = t.elapsed().as_micros() as u64;
    let mut guard = PROFILE.lock().unwrap();
    let p = guard.get_or_insert_with(ForwardProfile::new);
    match component {
        Component::Embed => p.embed_us += elapsed,
        Component::Attention => p.attn_us += elapsed,
        Component::Ffn => p.ffn_us += elapsed,
        Component::Other => p.other_us += elapsed,
    }
}

/// Mark one token completed.  Prints summary every 10 tokens.
pub(crate) fn tick() {
    if !is_enabled() {
        return;
    }
    let mut guard = PROFILE.lock().unwrap();
    let p = guard.get_or_insert_with(ForwardProfile::new);
    p.token_count += 1;
    if p.token_count % 10 == 0 {
        let total_us = p.start.elapsed().as_micros() as u64;
        let tok_s = if total_us > 0 {
            p.token_count as f64 / (total_us as f64 / 1_000_000.0)
        } else {
            0.0
        };
        eprintln!(
            "[profile] {} tokens @ {:.1} tok/s | embed: {:.1}ms, attn: {:.1}ms, ffn: {:.1}ms, other: {:.1}ms (per-token avg)",
            p.token_count,
            tok_s,
            p.embed_us as f64 / p.token_count as f64 / 1000.0,
            p.attn_us as f64 / p.token_count as f64 / 1000.0,
            p.ffn_us as f64 / p.token_count as f64 / 1000.0,
            p.other_us as f64 / p.token_count as f64 / 1000.0,
        );
    }
}

pub(crate) enum Component {
    Embed,
    Attention,
    Ffn,
    Other,
}
