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
// Tool calling (function calling):
//   Clients can pass `tools` (an array of function definitions) in the
//   chat completion request.  Tool definitions are injected into the
//   system prompt using the model's native format (see model/tools.rs).
//   After generation, the output is parsed for tool call markers.  If
//   found, the response includes `tool_calls` on the assistant message
//   and `finish_reason: "tool_calls"`.  The client can then send results
//   back as messages with `role: "tool"`.
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
use crate::model::tools::{self, ToolDefinition};

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
    /// Tool definitions the model may call (OpenAI function calling).
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Controls tool calling behaviour: "auto" (default), "none", or "required".
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
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
    message: ChatResponseMessage,
    finish_reason: Option<String>,
}

/// Response message — differs from the input Message because the OpenAI API
/// uses separate serialisation for the assistant's output (content may be null
/// when tool_calls are present).
#[derive(serde::Serialize)]
struct ChatResponseMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<tools::ToolCall>>,
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
        StopReason::ToolCalls => "tool_calls",
    }
}

// ---------------------------------------------------------------------------
// Tool injection helper: prepend tool definitions to the system message.
// ---------------------------------------------------------------------------

/// Inject tool definitions into the message list by appending to the system
/// message (or creating one if absent).  Returns the modified message list.
fn inject_tools(
    mut messages: Vec<Message>,
    tools: &[ToolDefinition],
    arch: crate::model::config::ModelArch,
) -> Vec<Message> {
    let tool_prompt = tools::format_tool_system_prompt(arch, tools);
    if tool_prompt.is_empty() {
        return messages;
    }

    // Find existing system message and append, or create a new one.
    if let Some(sys_msg) = messages.iter_mut().find(|m| m.role == "system") {
        sys_msg.content.push_str(&tool_prompt);
    } else {
        messages.insert(
            0,
            Message {
                role: "system".into(),
                content: tool_prompt,
                tool_calls: None,
                tool_call_id: None,
            },
        );
    }

    messages
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

/// Handle OpenAI chat completion requests (streaming and non-streaming).
pub(crate) async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, StatusCode> {
    let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());

    // If tool_choice is "none", strip tools so the model won't be prompted.
    let tools_disabled = req
        .tool_choice
        .as_ref()
        .and_then(|v| v.as_str())
        .is_some_and(|s| s == "none");

    // Inject tool definitions into the system message.
    let messages = if has_tools && !tools_disabled {
        inject_tools(req.messages, req.tools.as_ref().unwrap(), state.arch)
    } else {
        req.messages
    };

    // Tokenize on the async handler thread (CPU-only, doesn't block GPU).
    let prompt_tokens = state
        .tokenizer
        .encode_messages(&messages, state.arch)
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

    // For non-streaming with tools, we need to collect all text to detect
    // tool calls.  For streaming without tools, we can stream directly.
    // For streaming WITH tools, we also collect first then emit.
    if req.stream && !has_tools {
        Ok(chat_completions_stream(state, response_rx).await)
    } else {
        Ok(chat_completions_blocking(state, response_rx, has_tools && !tools_disabled).await)
    }
}

/// Non-streaming: collect all tokens, detect tool calls, return complete JSON response.
async fn chat_completions_blocking(
    state: Arc<ServerState>,
    mut response_rx: tokio::sync::mpsc::Receiver<InferenceEvent>,
    check_tools: bool,
) -> Response {
    let mut text = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut stop_reason = StopReason::MaxTokens;

    while let Some(event) = response_rx.recv().await {
        match event {
            InferenceEvent::Token { text: t } => text.push_str(&t),
            InferenceEvent::Done {
                stop_reason: sr,
                prompt_tokens: pt,
                completion_tokens: ct,
            } => {
                prompt_tokens = pt;
                completion_tokens = ct;
                stop_reason = sr;
            }
            InferenceEvent::Error(e) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e);
            }
        }
    }

    // Check for tool calls in the generated text.
    let (content, tool_calls) = if check_tools {
        let (cleaned, calls) = tools::parse_tool_calls(state.arch, &text);
        if !calls.is_empty() {
            (cleaned, Some(calls))
        } else {
            (text, None)
        }
    } else {
        (text, None)
    };

    let finish_reason = if tool_calls.is_some() {
        StopReason::ToolCalls
    } else {
        stop_reason
    };

    // OpenAI convention: content is null when only tool calls are present.
    let content = if tool_calls.is_some() && content.is_empty() {
        None
    } else {
        Some(content)
    };

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", super::generate_id()),
        object: "chat.completion",
        created: super::unix_timestamp(),
        model: state.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant",
                content,
                tool_calls,
            },
            finish_reason: Some(finish_reason_str(finish_reason).to_string()),
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
        assert!(req.tools.is_none());
        assert!(req.tool_choice.is_none());
    }

    #[test]
    fn test_chat_request_with_tools() {
        let json = r#"{
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }],
            "tool_choice": "auto"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(req.tool_choice.unwrap().as_str(), Some("auto"));
    }

    #[test]
    fn test_chat_request_tool_message() {
        let json = r#"{
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}]},
                {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 72F"}
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 3);
        // Assistant message with null content deserializes to empty string.
        assert_eq!(req.messages[1].content, "");
        assert!(req.messages[1].tool_calls.is_some());
        // Tool result message.
        assert_eq!(req.messages[2].role, "tool");
        assert_eq!(req.messages[2].tool_call_id.as_deref(), Some("call_123"));
        assert_eq!(req.messages[2].content, "Sunny, 72F");
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
        assert_eq!(finish_reason_str(StopReason::ToolCalls), "tool_calls");
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
                message: ChatResponseMessage {
                    role: "assistant",
                    content: Some("Hello!".into()),
                    tool_calls: None,
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
        // tool_calls should be absent (not null) when None.
        assert!(json["choices"][0]["message"].get("tool_calls").is_none());
    }

    #[test]
    fn test_chat_response_with_tool_calls() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-test456".into(),
            object: "chat.completion",
            created: 1234567890,
            model: "test-model".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant",
                    content: None,
                    tool_calls: Some(vec![tools::ToolCall {
                        id: "call_abc".into(),
                        type_: "function".into(),
                        function: tools::FunctionCall {
                            name: "get_weather".into(),
                            arguments: "{\"city\":\"SF\"}".into(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
        assert!(json["choices"][0]["message"]["content"].is_null());
        let tc = &json["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_abc");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "get_weather");
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
