// ===========================================================================
// `rllm bench` — Prefix cache benchmark with system prompts.
//
// Exercises prompt prefix caching on real hardware by running multiple
// requests that share the same system prompt.  The first request misses
// the cache and populates it; subsequent requests hit, skipping prefill
// for the cached prefix.  Reports per-request TTFT and aggregate stats.
//
// All model loading and engine setup is handled by load_model_and_run()
// in commands/mod.rs.
// ===========================================================================

use std::time::Instant;

use super::ModelArgs;

/// Diverse user prompts — short questions that exercise different domains.
/// All share the same system prompt, so the tokenized prefix is identical.
const BENCH_PROMPTS: &[&str] = &[
    "What is the capital of France?",
    "Explain quicksort in two sentences.",
    "Write a haiku about the ocean.",
    "What causes a rainbow?",
    "Convert 72°F to Celsius.",
    "Name three prime numbers greater than 50.",
    "What is the difference between a stack and a queue?",
    "Why is the sky blue?",
];

#[derive(clap::Args)]
pub(crate) struct BenchArgs {
    #[command(flatten)]
    model: ModelArgs,

    /// Number of prompts to run (1–8).  Default 3.
    #[arg(long, default_value = "3")]
    prompts: usize,
}

/// Per-request result for reporting.
struct RequestResult {
    prompt_idx: usize,
    prompt_tokens: usize,
    cached_tokens: usize,
    ttft: std::time::Duration,
    gen_tokens: usize,
    gen_duration: std::time::Duration,
}

pub(crate) fn exec(args: BenchArgs) -> anyhow::Result<()> {
    let system = args.model.system.clone();
    let max_tokens = args.model.max_tokens;
    let num_prompts = args.prompts.min(BENCH_PROMPTS.len());

    eprintln!("=== rllm prefix cache benchmark ===");
    eprintln!("model: {}", args.model.model.display());
    eprintln!("system prompt: {:?}", &system[..system.len().min(80)]);
    eprintln!("prompts: {}", num_prompts);
    eprintln!("max_tokens: {}", max_tokens);
    eprintln!();

    super::load_model_and_run(&args.model, 1, |eng, arch, _images| {
        let mut results: Vec<RequestResult> = Vec::new();

        eprintln!("--- per-request results ---");

        for (i, &user_prompt) in BENCH_PROMPTS[..num_prompts].iter().enumerate() {
            let prompt_tokens =
                super::encode_prompt(eng, arch, user_prompt, true, &system, None)?;
            let prompt_len = prompt_tokens.len();

            eng.add_request(prompt_tokens, max_tokens, 0.0, 1.0, Vec::new(), None);

            let start = Instant::now();
            let mut ttft = None;
            let mut gen_count = 0usize;
            let mut cached_tokens = 0usize;

            while eng.has_work() {
                let output = eng.step()?;

                if ttft.is_none() && !output.tokens.is_empty() {
                    ttft = Some(start.elapsed());
                }

                for &(_seq_id, _token_id) in &output.tokens {
                    gen_count += 1;
                }

                for seq in &output.finished {
                    cached_tokens = seq.cached_tokens;
                }
            }

            let total_elapsed = start.elapsed();
            let ttft = ttft.unwrap_or(total_elapsed);
            let gen_duration = total_elapsed.saturating_sub(ttft);

            let hit = if cached_tokens > 0 { "HIT" } else { "MISS" };
            let prefill_tps = prompt_len as f64 / ttft.as_secs_f64();
            let gen_tps = if gen_duration.as_secs_f64() > 0.0 {
                gen_count as f64 / gen_duration.as_secs_f64()
            } else {
                0.0
            };

            eprintln!(
                "  [{}/{}] {} | prompt {} tok, cached {} tok | TTFT {:.1?} ({:.0} tok/s) | gen {} tok {:.1} tok/s | {:?}",
                i + 1,
                num_prompts,
                hit,
                prompt_len,
                cached_tokens,
                ttft,
                prefill_tps,
                gen_count,
                gen_tps,
                &user_prompt[..user_prompt.len().min(40)],
            );

            results.push(RequestResult {
                prompt_idx: i,
                prompt_tokens: prompt_len,
                cached_tokens,
                ttft,
                gen_tokens: gen_count,
                gen_duration,
            });
        }

        // --- Aggregate stats ---
        eprintln!();
        eprintln!("--- aggregate ---");

        let cold: Vec<&RequestResult> = results.iter().filter(|r| r.cached_tokens == 0).collect();
        let warm: Vec<&RequestResult> = results.iter().filter(|r| r.cached_tokens > 0).collect();

        let avg_ttft = |rs: &[&RequestResult]| -> f64 {
            if rs.is_empty() {
                return 0.0;
            }
            rs.iter().map(|r| r.ttft.as_secs_f64()).sum::<f64>() / rs.len() as f64
        };

        let cold_avg = avg_ttft(&cold);
        let warm_avg = avg_ttft(&warm);

        eprintln!("  cold requests (miss): {}", cold.len());
        eprintln!("  warm requests (hit):  {}", warm.len());
        eprintln!("  avg TTFT cold: {:.1}ms", cold_avg * 1000.0);
        if !warm.is_empty() {
            eprintln!("  avg TTFT warm: {:.1}ms", warm_avg * 1000.0);
            if cold_avg > 0.0 {
                eprintln!("  TTFT speedup:  {:.2}x", cold_avg / warm_avg);
            }
        }

        let total_gen: usize = results.iter().map(|r| r.gen_tokens).sum();
        let total_gen_time: f64 = results.iter().map(|r| r.gen_duration.as_secs_f64()).sum();
        if total_gen_time > 0.0 {
            eprintln!(
                "  decode throughput: {:.1} tok/s (avg across {} requests)",
                total_gen as f64 / total_gen_time,
                results.len()
            );
        }

        let hit_rate = warm.len() as f64 / results.len() as f64 * 100.0;
        eprintln!("  cache hit rate: {:.0}%", hit_rate);

        if let Some(first_warm) = warm.first() {
            eprintln!(
                "  prefix tokens cached: {} (of {} prompt tokens)",
                first_warm.cached_tokens, first_warm.prompt_tokens
            );
        }

        // Suppress unused field warning.
        let _ = results.iter().map(|r| r.prompt_idx).count();

        eprintln!();
        eprintln!("=== benchmark complete ===");

        Ok(())
    })
}
