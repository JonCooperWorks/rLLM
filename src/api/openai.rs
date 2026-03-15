// ===========================================================================
// OpenAI-compatible API handlers.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements three OpenAI-compatible endpoints:
//
//   POST /v1/chat/completions — Chat completions (like ChatGPT)
//   POST /v1/completions      — Text completions (raw prompt, no chat template)
//   GET  /v1/models           — List available models
//
// Each POST endpoint supports two response modes:
//
//   1. Non-streaming (default): Collect all generated tokens, return a
//      single JSON response with the complete text and usage stats.
//
//   2. Streaming (stream=true): Return Server-Sent Events (SSE) where each
//      event contains one token as a JSON chunk.  The final event is
//      `data: [DONE]`.  This is how ChatGPT shows text appearing in real-time.
//
// OpenAI SSE format:
//   Each event is: `data: {json}\n\n`
//   The JSON contains a "delta" with the new content (one token).
//   The final event is: `data: [DONE]\n\n`
//
// Why match OpenAI's format?
//   Many tools, libraries, and UIs (like Open WebUI, LiteLLM, Cursor) can
//   talk to any server that speaks the OpenAI API format.  By matching it
//   exactly, rLLM becomes a drop-in replacement — just change the base URL.
// ===========================================================================

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use super::{InferenceEvent, ServerState, StopReason, WorkerRequest};
use crate::model::chat::Message;

// ---------------------------------------------------------------------------
// Request types (deserialized from JSON).
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub(crate) struct ChatCompletionRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
}

#[derive(serde::Deserialize)]
pub(crate) struct CompletionRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    4096
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    0.9
}

// ---------------------------------------------------------------------------
// Response types (serialized to JSON).
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(serde::Serialize)]
struct ChatChoice {
    index: u32,
    message: Message,
    finish_reason: Option<String>,
}

#[derive(serde::Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(serde::Serialize)]
struct CompletionChoice {
    index: u32,
    text: String,
    finish_reason: Option<String>,
}

#[derive(serde::Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Helper: convert StopReason to OpenAI's finish_reason string.
// ---------------------------------------------------------------------------

fn finish_reason_str(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndOfSequence => "stop",
        StopReason::MaxTokens => "length",
    }
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

/// Handle OpenAI chat completion requests (streaming and non-streaming).
pub(crate) async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, StatusCode> {
    // Tokenize on the async handler thread (CPU-only, doesn't block GPU).
    let prompt_tokens = state
        .tokenizer
        .encode_messages(&req.messages, state.arch)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let (response_tx, response_rx) = tokio::sync::mpsc::channel(64);

    let worker_req = WorkerRequest {
        prompt_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        response_tx,
    };

    state
        .request_tx
        .try_send(worker_req)
        .map_err(|e| match e {
            std::sync::mpsc::TrySendError::Full(_) => StatusCode::SERVICE_UNAVAILABLE,
            std::sync::mpsc::TrySendError::Disconnected(_) => StatusCode::INTERNAL_SERVER_ERROR,
        })?;

    if req.stream {
        Ok(chat_completions_stream(state, response_rx).await)
    } else {
        Ok(chat_completions_blocking(state, response_rx).await)
    }
}

/// Non-streaming: collect all tokens, return complete JSON response.
async fn chat_completions_blocking(
    state: Arc<ServerState>,
    mut response_rx: tokio::sync::mpsc::Receiver<InferenceEvent>,
) -> Response {
    let mut text = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";

    while let Some(event) = response_rx.recv().await {
        match event {
            InferenceEvent::Token { text: t } => text.push_str(&t),
            InferenceEvent::Done {
                stop_reason,
                prompt_tokens: pt,
                completion_tokens: ct,
            } => {
                prompt_tokens = pt;
                completion_tokens = ct;
                finish_reason = finish_reason_str(stop_reason);
            }
            InferenceEvent::Error(e) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e);
            }
        }
    }

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", super::generate_id()),
        object: "chat.completion",
        created: super::unix_timestamp(),
        model: state.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: Message {
                role: "assistant".into(),
                content: text,
            },
            finish_reason: Some(finish_reason.to_string()),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response()
}

/// Streaming: return SSE events with one token per chunk.
async fn chat_completions_stream(
    state: Arc<ServerState>,
    mut response_rx: tokio::sync::mpsc::Receiver<InferenceEvent>,
) -> Response {
    let id = format!("chatcmpl-{}", super::generate_id());
    let model = state.model_name.clone();

    let stream = async_stream::stream! {
        while let Some(event) = response_rx.recv().await {
            match event {
                InferenceEvent::Token { text } => {
                    let chunk = serde_json::json!({
                        "id": &id,
                        "object": "chat.completion.chunk",
                        "created": super::unix_timestamp(),
                        "model": &model,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": text },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    yield Ok::<_, std::convert::Infallible>(
                        format!("data: {}\n\n", chunk)
                    );
                }
                InferenceEvent::Done { stop_reason, .. } => {
                    let chunk = serde_json::json!({
                        "id": &id,
                        "object": "chat.completion.chunk",
                        "created": super::unix_timestamp(),
                        "model": &model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason_str(stop_reason)
                        }]
                    });
                    yield Ok(format!("data: {}\n\n", chunk));
                    yield Ok("data: [DONE]\n\n".to_string());
                }
                InferenceEvent::Error(e) => {
                    let err = serde_json::json!({
                        "error": { "message": e, "type": "server_error" }
                    });
                    yield Ok(format!("data: {}\n\n", err));
                }
            }
        }
    };

    Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

// ---------------------------------------------------------------------------
// POST /v1/completions
// ---------------------------------------------------------------------------

/// Handle OpenAI text completion requests (streaming and non-streaming).
///
/// Unlike chat completions, this takes a raw prompt string and does NOT
/// apply a chat template — suitable for base models or custom prompting.
pub(crate) async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, StatusCode> {
    // Tokenize raw prompt (no chat template) on the async handler thread.
    let prompt_tokens = state
        .tokenizer
        .encode(&req.prompt)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let (response_tx, response_rx) = tokio::sync::mpsc::channel(64);

    let worker_req = WorkerRequest {
        prompt_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        response_tx,
    };

    state
        .request_tx
        .try_send(worker_req)
        .map_err(|e| match e {
            std::sync::mpsc::TrySendError::Full(_) => StatusCode::SERVICE_UNAVAILABLE,
            std::sync::mpsc::TrySendError::Disconnected(_) => StatusCode::INTERNAL_SERVER_ERROR,
        })?;

    if req.stream {
        Ok(completions_stream(state, response_rx).await)
    } else {
        Ok(completions_blocking(state, response_rx).await)
    }
}

/// Non-streaming text completions.
async fn completions_blocking(
    state: Arc<ServerState>,
    mut response_rx: tokio::sync::mpsc::Receiver<InferenceEvent>,
) -> Response {
    let mut text = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";

    while let Some(event) = response_rx.recv().await {
        match event {
            InferenceEvent::Token { text: t } => text.push_str(&t),
            InferenceEvent::Done {
                stop_reason,
                prompt_tokens: pt,
                completion_tokens: ct,
            } => {
                prompt_tokens = pt;
                completion_tokens = ct;
                finish_reason = finish_reason_str(stop_reason);
            }
            InferenceEvent::Error(e) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e);
            }
        }
    }

    Json(CompletionResponse {
        id: format!("cmpl-{}", super::generate_id()),
        object: "text_completion",
        created: super::unix_timestamp(),
        model: state.model_name.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: Some(finish_reason.to_string()),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response()
}

/// Streaming text completions (SSE).
async fn completions_stream(
    state: Arc<ServerState>,
    mut response_rx: tokio::sync::mpsc::Receiver<InferenceEvent>,
) -> Response {
    let id = format!("cmpl-{}", super::generate_id());
    let model = state.model_name.clone();

    let stream = async_stream::stream! {
        while let Some(event) = response_rx.recv().await {
            match event {
                InferenceEvent::Token { text } => {
                    let chunk = serde_json::json!({
                        "id": &id,
                        "object": "text_completion",
                        "created": super::unix_timestamp(),
                        "model": &model,
                        "choices": [{
                            "index": 0,
                            "text": text,
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    yield Ok::<_, std::convert::Infallible>(
                        format!("data: {}\n\n", chunk)
                    );
                }
                InferenceEvent::Done { stop_reason, .. } => {
                    let chunk = serde_json::json!({
                        "id": &id,
                        "object": "text_completion",
                        "created": super::unix_timestamp(),
                        "model": &model,
                        "choices": [{
                            "index": 0,
                            "text": "",
                            "finish_reason": finish_reason_str(stop_reason)
                        }]
                    });
                    yield Ok(format!("data: {}\n\n", chunk));
                    yield Ok("data: [DONE]\n\n".to_string());
                }
                InferenceEvent::Error(e) => {
                    let err = serde_json::json!({
                        "error": { "message": e, "type": "server_error" }
                    });
                    yield Ok(format!("data: {}\n\n", err));
                }
            }
        }
    };

    Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

/// List available models.  Since rLLM serves a single model, this returns
/// a one-element list.
pub(crate) async fn list_models(State(state): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": &state.model_name,
            "object": "model",
            "created": super::unix_timestamp(),
            "owned_by": "rllm",
        }]
    }))
}

// ---------------------------------------------------------------------------
// Error helper.
// ---------------------------------------------------------------------------

fn error_response(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
            "code": status.as_u16(),
        }
    });
    (status, Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_deserialization() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": true
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.as_deref(), Some("test-model"));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "Hello");
        assert_eq!(req.max_tokens, 100);
        assert!((req.temperature - 0.7).abs() < 0.001);
        assert!((req.top_p - 0.95).abs() < 0.001);
        assert!(req.stream);
    }

    #[test]
    fn test_chat_request_defaults() {
        let json = r#"{"messages": [{"role": "user", "content": "Hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, None);
        assert_eq!(req.max_tokens, 4096);
        assert!((req.temperature - 1.0).abs() < 0.001);
        assert!((req.top_p - 0.9).abs() < 0.001);
        assert!(!req.stream);
    }

    #[test]
    fn test_completion_request_deserialization() {
        let json = r#"{"prompt": "Once upon a time", "max_tokens": 50}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "Once upon a time");
        assert_eq!(req.max_tokens, 50);
        assert!((req.temperature - 1.0).abs() < 0.001); // default
    }

    #[test]
    fn test_completion_request_defaults() {
        let json = r#"{"prompt": "test"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 4096);
        assert!((req.temperature - 1.0).abs() < 0.001);
        assert!((req.top_p - 0.9).abs() < 0.001);
        assert!(!req.stream);
    }

    #[test]
    fn test_finish_reason_str_values() {
        assert_eq!(finish_reason_str(StopReason::EndOfSequence), "stop");
        assert_eq!(finish_reason_str(StopReason::MaxTokens), "length");
    }

    #[test]
    fn test_chat_response_serialization() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-test123".into(),
            object: "chat.completion",
            created: 1234567890,
            model: "test-model".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: Message {
                    role: "assistant".into(),
                    content: "Hello!".into(),
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 15);
    }

    #[test]
    fn test_completion_response_serialization() {
        let response = CompletionResponse {
            id: "cmpl-test123".into(),
            object: "text_completion",
            created: 1234567890,
            model: "test-model".into(),
            choices: vec![CompletionChoice {
                index: 0,
                text: "world".into(),
                finish_reason: Some("length".into()),
            }],
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 10,
                total_tokens: 13,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["object"], "text_completion");
        assert_eq!(json["choices"][0]["text"], "world");
        assert_eq!(json["usage"]["prompt_tokens"], 3);
    }
}
