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
//   5. Tool use:  Anthropic uses content blocks of type "tool_use" in the
//      response (with id, name, input fields).  Tool results come as
//      "tool_result" content blocks in user messages.  OpenAI uses a
//      separate "tool_calls" field and "tool" role messages.
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
use crate::model::tools::{self, ToolDefinition};

// ---------------------------------------------------------------------------
// Request types.
// ---------------------------------------------------------------------------

/// Anthropic tool definition (slightly different field names from OpenAI).
#[derive(serde::Deserialize)]
struct AnthropicToolDef {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    input_schema: Option<serde_json::Value>,
}

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
    /// Tool definitions (Anthropic format: name, description, input_schema).
    #[serde(default)]
    pub tools: Option<Vec<AnthropicToolDef>>,
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
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
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
        StopReason::ToolCalls => "tool_use",
    }
}

// ---------------------------------------------------------------------------
// Helper: convert Anthropic tool definitions to internal format.
// ---------------------------------------------------------------------------

fn convert_anthropic_tools(tools: &[AnthropicToolDef]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|t| ToolDefinition {
            type_: "function".into(),
            function: tools::FunctionDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect()
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
    let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());

    // Build message list: prepend system prompt if provided.
    let mut messages = Vec::new();

    // Build system content: combine explicit system prompt with tool definitions.
    let mut system_content = req.system.unwrap_or_default();
    if has_tools {
        let internal_tools = convert_anthropic_tools(req.tools.as_ref().unwrap());
        let tool_prompt = tools::format_tool_system_prompt(state.arch, &internal_tools);
        system_content.push_str(&tool_prompt);
    }

    if !system_content.is_empty() {
        messages.push(Message {
            role: "system".into(),
            content: system_content,
            tool_calls: None,
            tool_call_id: None,
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
        endpoint: "anthropic",
    };

    state.request_tx.try_send(worker_req).map_err(|e| match e {
        std::sync::mpsc::TrySendError::Full(_) => StatusCode::SERVICE_UNAVAILABLE,
        std::sync::mpsc::TrySendError::Disconnected(_) => StatusCode::INTERNAL_SERVER_ERROR,
    })?;

    if req.stream && !has_tools {
        Ok(messages_stream(state, response_rx).await)
    } else {
        Ok(messages_blocking(state, response_rx, has_tools).await)
    }
}

/// Non-streaming: collect all tokens, return complete JSON response.
async fn messages_blocking(
    state: Arc<ServerState>,
    mut response_rx: tokio::sync::mpsc::Receiver<InferenceEvent>,
    check_tools: bool,
) -> Response {
    let mut text = String::new();
    let mut input_tokens = 0usize;
    let mut output_tokens = 0usize;
    let mut stop_reason = StopReason::MaxTokens;

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
                stop_reason = sr;
            }
            InferenceEvent::Error(e) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e);
            }
        }
    }

    // Check for tool calls in the generated text.
    let (content_text, tool_calls) = if check_tools {
        let (cleaned, calls) = tools::parse_tool_calls(state.arch, &text);
        if !calls.is_empty() {
            (cleaned, calls)
        } else {
            (text, Vec::new())
        }
    } else {
        (text, Vec::new())
    };

    // Build content blocks.
    let mut content = Vec::new();

    // Add text block if there's content.
    if !content_text.is_empty() {
        content.push(ContentBlock::Text { text: content_text });
    }

    // Add tool_use blocks for each tool call.
    for call in &tool_calls {
        let input: serde_json::Value = serde_json::from_str(&call.function.arguments)
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        content.push(ContentBlock::ToolUse {
            id: call.id.clone(),
            name: call.function.name.clone(),
            input,
        });
    }

    // If no content blocks at all, add an empty text block.
    if content.is_empty() {
        content.push(ContentBlock::Text {
            text: String::new(),
        });
    }

    let final_stop_reason = if !tool_calls.is_empty() {
        StopReason::ToolCalls
    } else {
        stop_reason
    };

    Json(MessagesResponse {
        id: format!("msg_{}", super::generate_id()),
        type_: "message",
        role: "assistant",
        content,
        model: state.model_name.clone(),
        stop_reason: Some(stop_reason_str(final_stop_reason).to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_messages_request_deserialization() {
        let json = r#"{
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 200,
            "temperature": 0.5,
            "top_p": 0.8,
            "stream": false,
            "system": "You are helpful."
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.as_deref(), Some("claude-3"));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.max_tokens, 200);
        assert!((req.temperature.unwrap() - 0.5).abs() < 0.001);
        assert!((req.top_p.unwrap() - 0.8).abs() < 0.001);
        assert!(!req.stream);
        assert_eq!(req.system.as_deref(), Some("You are helpful."));
    }

    #[test]
    fn test_messages_request_defaults() {
        let json = r#"{"messages": [{"role": "user", "content": "Hi"}]}"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, None);
        assert_eq!(req.max_tokens, 4096);
        assert_eq!(req.temperature, None);
        assert_eq!(req.top_p, None);
        assert!(!req.stream);
        assert_eq!(req.system, None);
        assert!(req.tools.is_none());
    }

    #[test]
    fn test_messages_request_system_field() {
        let json = r#"{
            "messages": [{"role": "user", "content": "Hi"}],
            "system": "Be concise."
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.system.as_deref(), Some("Be concise."));
    }

    #[test]
    fn test_messages_request_with_tools() {
        let json = r#"{
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
    }

    #[test]
    fn test_stop_reason_str_values() {
        assert_eq!(stop_reason_str(StopReason::EndOfSequence), "end_turn");
        assert_eq!(stop_reason_str(StopReason::MaxTokens), "max_tokens");
        assert_eq!(stop_reason_str(StopReason::ToolCalls), "tool_use");
    }

    #[test]
    fn test_messages_response_serialization() {
        let response = MessagesResponse {
            id: "msg_test123".into(),
            type_: "message",
            role: "assistant",
            content: vec![ContentBlock::Text {
                text: "Hello!".into(),
            }],
            model: "test-model".into(),
            stop_reason: Some("end_turn".into()),
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "Hello!");
        assert_eq!(json["stop_reason"], "end_turn");
        assert_eq!(json["usage"]["input_tokens"], 10);
        assert_eq!(json["usage"]["output_tokens"], 5);
    }

    #[test]
    fn test_messages_response_with_tool_use() {
        let response = MessagesResponse {
            id: "msg_test456".into(),
            type_: "message",
            role: "assistant",
            content: vec![
                ContentBlock::Text {
                    text: "Let me check.".into(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_123".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"city": "SF"}),
                },
            ],
            model: "test-model".into(),
            stop_reason: Some("tool_use".into()),
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 20,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["stop_reason"], "tool_use");
        assert_eq!(json["content"].as_array().unwrap().len(), 2);
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][1]["type"], "tool_use");
        assert_eq!(json["content"][1]["name"], "get_weather");
        assert_eq!(json["content"][1]["input"]["city"], "SF");
    }

    #[test]
    fn test_convert_anthropic_tools() {
        let tools = vec![AnthropicToolDef {
            name: "test".into(),
            description: Some("A test tool".into()),
            input_schema: Some(serde_json::json!({"type": "object"})),
        }];
        let converted = convert_anthropic_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].function.name, "test");
        assert_eq!(
            converted[0].function.description.as_deref(),
            Some("A test tool")
        );
    }
}
