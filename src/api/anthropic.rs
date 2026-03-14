// ===========================================================================
// Anthropic-compatible API handler.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements the Anthropic Messages API endpoint:
//
//   POST /v1/messages — Create a message (like Claude)
//
// How the Anthropic API differs from OpenAI:
//
//   1. System prompt:  Anthropic puts it in a top-level "system" field,
//      NOT as a message with role "system".  OpenAI puts it in messages.
//
//   2. Response structure:  Anthropic wraps text in content blocks:
//      {"content": [{"type": "text", "text": "Hello"}]}
//      OpenAI uses a flat message: {"message": {"content": "Hello"}}
//
//   3. Stop reasons:  Anthropic uses "end_turn" and "max_tokens".
//      OpenAI uses "stop" and "length".
//
//   4. Streaming format:  Both use SSE, but Anthropic includes an `event:`
//      field with typed events (message_start, content_block_delta, etc.).
//      OpenAI only uses `data:` lines.
//
// Anthropic SSE event sequence:
//   event: message_start      — metadata (id, model, usage)
//   event: content_block_start — signals a text block is beginning
//   event: content_block_delta — one per token (the actual text)
//   event: content_block_stop  — text block ended
//   event: message_delta       — final stop reason + output token count
//   event: message_stop        — stream is done
//
// Why support both APIs?
//   Different tools and SDKs speak different API dialects.  The Anthropic
//   Python SDK, Claude Desktop, and tools like Cline use the Anthropic
//   format.  By supporting both, rLLM works as a drop-in local backend
//   for any major LLM client.
// ===========================================================================

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use super::{InferenceEvent, ServerState, StopReason, WorkerRequest};
use crate::model::chat::Message;

// ---------------------------------------------------------------------------
// Request types.
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub(crate) struct MessagesRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    /// Anthropic puts the system prompt at the top level, not in messages.
    #[serde(default)]
    pub system: Option<String>,
}

fn default_max_tokens() -> usize {
    4096
}

// ---------------------------------------------------------------------------
// Response types.
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct MessagesResponse {
    id: String,
    #[serde(rename = "type")]
    type_: &'static str,
    role: &'static str,
    content: Vec<ContentBlock>,
    model: String,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(serde::Serialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    type_: &'static str,
    text: String,
}

#[derive(serde::Serialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

// ---------------------------------------------------------------------------
// Helper: convert StopReason to Anthropic's stop_reason string.
// ---------------------------------------------------------------------------

fn stop_reason_str(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndOfSequence => "end_turn",
        StopReason::MaxTokens => "max_tokens",
    }
}

// ---------------------------------------------------------------------------
// POST /v1/messages
// ---------------------------------------------------------------------------

/// Handle Anthropic Messages API requests (streaming and non-streaming).
///
/// The key difference from OpenAI: the system prompt comes as a separate
/// top-level field.  We prepend it as a system message before sending
/// to the inference worker.
pub(crate) async fn messages(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<MessagesRequest>,
) -> Result<Response, StatusCode> {
    // Build message list: prepend system prompt if provided.
    let mut messages = Vec::new();
    if let Some(ref system) = req.system {
        messages.push(Message {
            role: "system".into(),
            content: system.clone(),
        });
    }
    messages.extend(req.messages);

    // Tokenize on the async handler thread (CPU-only, doesn't block GPU).
    let prompt_tokens = state
        .tokenizer
        .encode_messages(&messages, state.arch)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let (response_tx, response_rx) = tokio::sync::mpsc::channel(64);

    let worker_req = WorkerRequest {
        prompt_tokens,
        max_tokens: req.max_tokens,
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(0.9),
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
        Ok(messages_stream(state, response_rx).await)
    } else {
        Ok(messages_blocking(state, response_rx).await)
    }
}

/// Non-streaming: collect all tokens, return complete JSON response.
async fn messages_blocking(
    state: Arc<ServerState>,
    mut response_rx: tokio::sync::mpsc::Receiver<InferenceEvent>,
) -> Response {
    let mut text = String::new();
    let mut input_tokens = 0usize;
    let mut output_tokens = 0usize;
    let mut stop_reason = "max_tokens";

    while let Some(event) = response_rx.recv().await {
        match event {
            InferenceEvent::Token { text: t } => text.push_str(&t),
            InferenceEvent::Done {
                stop_reason: sr,
                prompt_tokens,
                completion_tokens,
            } => {
                input_tokens = prompt_tokens;
                output_tokens = completion_tokens;
                stop_reason = stop_reason_str(sr);
            }
            InferenceEvent::Error(e) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e);
            }
        }
    }

    Json(MessagesResponse {
        id: format!("msg_{}", super::generate_id()),
        type_: "message",
        role: "assistant",
        content: vec![ContentBlock {
            type_: "text",
            text,
        }],
        model: state.model_name.clone(),
        stop_reason: Some(stop_reason.to_string()),
        usage: AnthropicUsage {
            input_tokens,
            output_tokens,
        },
    })
    .into_response()
}

/// Streaming: return SSE events using Anthropic's event protocol.
///
/// Anthropic's SSE differs from OpenAI in that each event has an `event:`
/// field specifying the event type.  The sequence is:
///   message_start → content_block_start → content_block_delta (per token)
///   → content_block_stop → message_delta → message_stop
async fn messages_stream(
    state: Arc<ServerState>,
    mut response_rx: tokio::sync::mpsc::Receiver<InferenceEvent>,
) -> Response {
    let id = format!("msg_{}", super::generate_id());
    let model = state.model_name.clone();

    let stream = async_stream::stream! {
        // 1. message_start — metadata about the message being generated.
        yield Ok::<_, std::convert::Infallible>(format!(
            "event: message_start\ndata: {}\n\n",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": &id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": &model,
                    "stop_reason": serde_json::Value::Null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                }
            })
        ));

        // 2. content_block_start — signals a new text content block.
        yield Ok(format!(
            "event: content_block_start\ndata: {}\n\n",
            serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            })
        ));

        // 3. content_block_delta — one event per generated token.
        while let Some(event) = response_rx.recv().await {
            match event {
                InferenceEvent::Token { text } => {
                    yield Ok(format!(
                        "event: content_block_delta\ndata: {}\n\n",
                        serde_json::json!({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": { "type": "text_delta", "text": text }
                        })
                    ));
                }
                InferenceEvent::Done { stop_reason, completion_tokens, .. } => {
                    // 4. content_block_stop — this text block is done.
                    yield Ok(format!(
                        "event: content_block_stop\ndata: {}\n\n",
                        serde_json::json!({ "type": "content_block_stop", "index": 0 })
                    ));

                    // 5. message_delta — final stop reason and output token count.
                    yield Ok(format!(
                        "event: message_delta\ndata: {}\n\n",
                        serde_json::json!({
                            "type": "message_delta",
                            "delta": { "stop_reason": stop_reason_str(stop_reason) },
                            "usage": { "output_tokens": completion_tokens }
                        })
                    ));

                    // 6. message_stop — stream is complete.
                    yield Ok(
                        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string()
                    );
                }
                InferenceEvent::Error(e) => {
                    yield Ok(format!(
                        "event: error\ndata: {}\n\n",
                        serde_json::json!({
                            "type": "error",
                            "error": { "type": "server_error", "message": e }
                        })
                    ));
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
// Error helper.
// ---------------------------------------------------------------------------

fn error_response(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": "server_error",
            "message": message,
        }
    });
    (status, Json(body)).into_response()
}
