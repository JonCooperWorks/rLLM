// ===========================================================================
// Prometheus metrics for the inference server.
//
// All metrics are registered with the global prometheus registry and exposed
// via the GET /metrics endpoint.  The worker thread (sync) records inference
// metrics directly; HTTP handlers record request counts.
//
// Metric naming follows Prometheus conventions:
//   - Counters end in _total
//   - Histograms use _seconds for durations
//   - Gauges use present-tense nouns
//
// Related files:
//   - api/mod.rs       — worker loop records inference metrics
//   - api/openai.rs    — handlers record request counts
//   - api/anthropic.rs — handlers record request counts
// ===========================================================================

use prometheus::{
    self, Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge, Opts, Registry,
};

/// All server metrics, constructed once at startup and shared via Arc.
///
/// Using a struct (not global statics) keeps metrics testable and makes
/// ownership explicit.  The struct is stored in `ServerState` and passed
/// to the worker thread.
pub(crate) struct Metrics {
    /// End-to-end request duration (from worker registration to completion).
    pub request_duration: Histogram,
    /// Time to first token (prompt submission to first generated token).
    pub time_to_first_token: Histogram,
    /// Decode throughput in tokens per second.
    pub decode_tokens_per_second: Histogram,
    /// Total prompt tokens processed.
    pub prompt_tokens: IntCounter,
    /// Total completion tokens generated.
    pub completion_tokens: IntCounter,
    /// Total requests received, labelled by endpoint and status.
    pub requests_total: IntCounterVec,
    /// Number of sequences currently being processed by the engine.
    pub active_sequences: IntGauge,
    /// Number of sequences waiting to be admitted.
    pub waiting_sequences: IntGauge,
    /// Number of requests that hit the prefix cache.
    pub prefix_cache_hits: IntCounter,
    /// Total inference errors (engine.step() failures).
    pub errors: IntCounter,
    /// The registry these metrics are registered with.
    registry: Registry,
}

impl Metrics {
    /// Create and register all metrics with a new registry.
    pub fn new() -> Self {
        let registry = Registry::new();

        let request_duration = Histogram::with_opts(
            HistogramOpts::new(
                "rllm_request_duration_seconds",
                "End-to-end request duration in seconds",
            )
            .buckets(vec![0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]),
        )
        .unwrap();
        registry.register(Box::new(request_duration.clone())).unwrap();

        let time_to_first_token = Histogram::with_opts(
            HistogramOpts::new(
                "rllm_time_to_first_token_seconds",
                "Time from request submission to first generated token",
            )
            .buckets(vec![0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
        )
        .unwrap();
        registry
            .register(Box::new(time_to_first_token.clone()))
            .unwrap();

        let decode_tokens_per_second = Histogram::with_opts(
            HistogramOpts::new(
                "rllm_decode_tokens_per_second",
                "Decode throughput in tokens per second",
            )
            .buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0]),
        )
        .unwrap();
        registry
            .register(Box::new(decode_tokens_per_second.clone()))
            .unwrap();

        let prompt_tokens = IntCounter::with_opts(Opts::new(
            "rllm_prompt_tokens_total",
            "Total prompt tokens processed",
        ))
        .unwrap();
        registry.register(Box::new(prompt_tokens.clone())).unwrap();

        let completion_tokens = IntCounter::with_opts(Opts::new(
            "rllm_completion_tokens_total",
            "Total completion tokens generated",
        ))
        .unwrap();
        registry
            .register(Box::new(completion_tokens.clone()))
            .unwrap();

        let requests_total = IntCounterVec::new(
            Opts::new("rllm_requests_total", "Total requests received"),
            &["endpoint"],
        )
        .unwrap();
        registry
            .register(Box::new(requests_total.clone()))
            .unwrap();

        let active_sequences = IntGauge::with_opts(Opts::new(
            "rllm_active_sequences",
            "Number of sequences currently being processed",
        ))
        .unwrap();
        registry
            .register(Box::new(active_sequences.clone()))
            .unwrap();

        let waiting_sequences = IntGauge::with_opts(Opts::new(
            "rllm_waiting_sequences",
            "Number of sequences waiting to be admitted",
        ))
        .unwrap();
        registry
            .register(Box::new(waiting_sequences.clone()))
            .unwrap();

        let prefix_cache_hits = IntCounter::with_opts(Opts::new(
            "rllm_prefix_cache_hits_total",
            "Number of requests that hit the prefix cache",
        ))
        .unwrap();
        registry
            .register(Box::new(prefix_cache_hits.clone()))
            .unwrap();

        let errors = IntCounter::with_opts(Opts::new(
            "rllm_errors_total",
            "Total inference errors",
        ))
        .unwrap();
        registry.register(Box::new(errors.clone())).unwrap();

        Self {
            request_duration,
            time_to_first_token,
            decode_tokens_per_second,
            prompt_tokens,
            completion_tokens,
            requests_total,
            active_sequences,
            waiting_sequences,
            prefix_cache_hits,
            errors,
            registry,
        }
    }

    /// Encode all metrics in Prometheus text exposition format.
    pub fn encode(&self) -> String {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

/// GET /metrics — Prometheus scrape endpoint.
pub(crate) async fn metrics_handler(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<super::ServerState>>,
) -> impl axum::response::IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.encode(),
    )
}
