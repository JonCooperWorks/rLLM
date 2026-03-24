// ===========================================================================
// Tool calling — types, prompt formatting, and output parsing.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Provides everything needed for tool/function calling across model
//   architectures.  There are three concerns:
//
//   1. Types (ToolDefinition, ToolCall, etc.) — shared between API endpoints
//      and the internal pipeline.
//
//   2. Prompt formatting — serialising tool definitions into the system prompt
//      so the model knows which tools are available.  Each architecture has
//      its own format (Llama 3.1 uses JSON in a specific schema, Qwen/ChatML
//      uses <tool_call> markers, Mistral uses [AVAILABLE_TOOLS], etc.).
//
//   3. Output parsing — detecting and extracting tool calls from the model's
//      generated text.  Each architecture has a dedicated parser that only
//      recognises the markers/format its formatter produces:
//        - Llama: bare JSON objects (no markers — matches its format prompt)
//        - Mistral/Mixtral: [TOOL_CALLS] marker
//        - Qwen, Phi, Gemma, GPT-OSS: <tool_call> XML markers (ChatML)
//      Both `format_tool_system_prompt()` and `parse_tool_calls()` use
//      exhaustive matches on `ModelArch` — adding a new architecture forces
//      the developer to choose both a formatter and a parser.
//
// How tool calling works end-to-end:
//   1. Client sends a request with `tools` (list of function definitions).
//   2. The API handler calls `format_tool_system_prompt()` to inject tool
//      definitions into the system message.
//   3. The model generates text that may contain tool call markers.
//   4. After generation, `parse_tool_calls()` extracts any tool calls.
//   5. If tool calls are found, the API returns them structured in the
//      response with finish_reason "tool_calls" (OpenAI) or a tool_use
//      content block (Anthropic).
//   6. The client executes the tools and sends results back as "tool" role
//      messages, which are formatted by the chat template (see chat.rs).
//
// Why per-architecture?
//   Each model family was fine-tuned on its own tool calling format.  Using
//   the wrong format produces unreliable output — the model needs to see
//   the exact markers it was trained on.
//
// Related files:
//   - chat.rs: handles "tool" role in chat templates, Message struct
//   - api/openai.rs: OpenAI tool calling API (tools, tool_choice)
//   - api/anthropic.rs: Anthropic tool use API (tools, tool_use blocks)
// ===========================================================================

use super::config::ModelArch;

// ---------------------------------------------------------------------------
// Tool definition types — what tools are available (from the API request).
// ---------------------------------------------------------------------------

/// A tool the model can call.  Currently only "function" type is supported,
/// matching the OpenAI API convention.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ToolDefinition {
    #[serde(rename = "type", default = "default_tool_type")]
    pub type_: String,
    pub function: FunctionDefinition,
}

/// A function definition: name, description, and JSON Schema for parameters.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

fn default_tool_type() -> String {
    "function".into()
}

// ---------------------------------------------------------------------------
// Tool call types — what the model outputs when it wants to call a tool.
// ---------------------------------------------------------------------------

/// A tool call extracted from model output.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub function: FunctionCall,
}

/// The function name and arguments the model wants to invoke.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct FunctionCall {
    pub name: String,
    /// Arguments as a JSON string (matches OpenAI convention).
    pub arguments: String,
}

/// Generate a tool call ID (OpenAI format: "call_" + random hex).
fn generate_tool_call_id() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    format!("call_{:016x}", rng.random::<u64>())
}

// ---------------------------------------------------------------------------
// System prompt formatting — inject tool definitions for the model.
//
// Each architecture has its own format.  The returned string should be
// prepended to (or replace) the system message before tokenization.
// ---------------------------------------------------------------------------

/// Format tool definitions into a system prompt addition for the given architecture.
///
/// Returns a string that should be appended to the system message content.
/// If there is no system message, the caller should create one with this content.
pub(crate) fn format_tool_system_prompt(arch: ModelArch, tools: &[ToolDefinition]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    match arch {
        ModelArch::Llama => format_tools_llama(tools),
        ModelArch::Mistral | ModelArch::Mixtral => format_tools_mistral(tools),
        ModelArch::Qwen2 | ModelArch::Qwen3Moe | ModelArch::Qwen3_5 | ModelArch::GptOss => {
            format_tools_chatml(tools)
        }
        ModelArch::Phi => format_tools_chatml(tools),
        ModelArch::Gemma3 => format_tools_chatml(tools),
    }
}

/// Llama 3.1+ tool format.
///
/// Llama's instruct fine-tuning expects tool definitions in a specific
/// JSON format within an "Environment: ipython" system message.  The model
/// was trained to output tool calls as JSON after a `<|python_tag|>` token,
/// but since many deployments use a simpler JSON format, we use the
/// "function calling" variant that works with Llama 3.1/3.2/3.3 instruct.
fn format_tools_llama(tools: &[ToolDefinition]) -> String {
    let tool_json: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": &t.function.name,
                    "description": &t.function.description,
                    "parameters": &t.function.parameters,
                }
            })
        })
        .collect();

    let tools_str = serde_json::to_string_pretty(&tool_json).unwrap_or_default();

    format!(
        "\n\nYou have access to the following tools:\n\n{tools_str}\n\n\
         When you need to call a tool, respond with a JSON object in this exact format:\n\
         {{\"name\": \"function_name\", \"arguments\": {{...}}}}\n\n\
         If you need to call multiple tools, output one JSON object per line.\n\
         Only call tools when necessary — otherwise respond normally."
    )
}

/// Mistral/Mixtral tool format.
///
/// Mistral models expect tools listed in [AVAILABLE_TOOLS] markers and
/// output calls in [TOOL_CALLS] markers as a JSON array.
fn format_tools_mistral(tools: &[ToolDefinition]) -> String {
    let tool_json: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": &t.function.name,
                    "description": &t.function.description,
                    "parameters": &t.function.parameters,
                }
            })
        })
        .collect();

    let tools_str = serde_json::to_string(&tool_json).unwrap_or_default();

    format!("\n\n[AVAILABLE_TOOLS]{tools_str}[/AVAILABLE_TOOLS]")
}

/// ChatML-family tool format (Qwen, Phi, Gemma, GPT-OSS).
///
/// Qwen 2.5 instruct models were trained with tool definitions as a
/// structured list in the system prompt.  The model outputs tool calls
/// wrapped in <tool_call> markers.  This format is also used as a
/// reasonable default for other ChatML-based models.
fn format_tools_chatml(tools: &[ToolDefinition]) -> String {
    let mut out = String::from("\n\n# Tools\n\nYou have access to the following tools:\n\n");

    for tool in tools {
        let params = tool
            .function
            .parameters
            .as_ref()
            .map(|p| serde_json::to_string(p).unwrap_or_default())
            .unwrap_or_else(|| "{}".into());

        out.push_str(&format!(
            "## {name}\n\n{desc}\n\nParameters: {params}\n\n",
            name = tool.function.name,
            desc = tool
                .function
                .description
                .as_deref()
                .unwrap_or("No description."),
            params = params,
        ));
    }

    out.push_str(
        "When you need to call a tool, use this format:\n\
         <tool_call>\n{\"name\": \"function_name\", \"arguments\": {...}}\n</tool_call>\n\n\
         You can call multiple tools by using multiple <tool_call> blocks.",
    );

    out
}

// ---------------------------------------------------------------------------
// Tool call parsing — extract tool calls from model-generated text.
//
// Returns (cleaned_text, tool_calls):
//   - cleaned_text: the response text with tool call markers removed
//   - tool_calls: extracted ToolCall structs (empty if no tool calls found)
// ---------------------------------------------------------------------------

/// Parse model output for tool calls, using the appropriate format for the architecture.
///
/// Returns (content_text, tool_calls).  If no tool calls are found,
/// tool_calls is empty and content_text is the full original text.
///
/// This match is intentionally exhaustive (no wildcard arm) so that adding a
/// new `ModelArch` variant forces the developer to choose the correct parser
/// — mirroring the exhaustive match in `format_tool_system_prompt`.
pub(crate) fn parse_tool_calls(arch: ModelArch, text: &str) -> (String, Vec<ToolCall>) {
    match arch {
        ModelArch::Llama => parse_tool_calls_llama(text),
        ModelArch::Mistral | ModelArch::Mixtral => parse_tool_calls_mistral(text),
        ModelArch::Qwen2 | ModelArch::Qwen3Moe | ModelArch::Qwen3_5 | ModelArch::GptOss => {
            parse_tool_calls_chatml(text)
        }
        ModelArch::Phi => parse_tool_calls_chatml(text),
        ModelArch::Gemma3 => parse_tool_calls_chatml(text),
    }
}

/// Parse <tool_call>...</tool_call> markers (Qwen/ChatML style).
///
/// Only recognises explicit `<tool_call>` markers — no bare-JSON fallback.
/// The corresponding formatter (`format_tools_chatml`) instructs the model to
/// use these markers, so anything *without* them is treated as normal text.
fn parse_tool_calls_chatml(text: &str) -> (String, Vec<ToolCall>) {
    if !text.contains("<tool_call>") {
        return (text.to_string(), Vec::new());
    }

    let mut calls = Vec::new();
    let mut cleaned = text.to_string();

    // Extract all <tool_call>...</tool_call> blocks.
    while let Some(start) = cleaned.find("<tool_call>") {
        let end_tag = "</tool_call>";
        let end = cleaned[start..]
            .find(end_tag)
            .map(|i| start + i + end_tag.len());

        if let Some(end) = end {
            let inner_start = start + "<tool_call>".len();
            let inner_end = end - end_tag.len();
            if let Some(block) = cleaned.get(inner_start..inner_end) {
                if let Some(call) = parse_json_tool_call(block.trim()) {
                    calls.push(call);
                }
            }
            cleaned.replace_range(start..end, "");
        } else {
            // Unclosed tag — try to parse the rest as a tool call.
            let inner_start = start + "<tool_call>".len();
            if let Some(block) = cleaned.get(inner_start..) {
                if let Some(call) = parse_json_tool_call(block.trim()) {
                    calls.push(call);
                }
            }
            cleaned.truncate(start);
            break;
        }
    }

    if !calls.is_empty() {
        return (cleaned.trim().to_string(), calls);
    }

    (text.to_string(), Vec::new())
}

/// Parse bare JSON tool calls (Llama format).
///
/// Llama's tool-call format instructs the model to respond with a plain JSON
/// object `{"name": "...", "arguments": {...}}` — no markers.  This parser is
/// only used for Llama so the bare-JSON heuristic doesn't accidentally fire
/// for other architectures.
fn parse_tool_calls_llama(text: &str) -> (String, Vec<ToolCall>) {
    if let Some(calls) = parse_bare_json_tool_calls(text) {
        let cleaned = strip_json_tool_calls(text);
        return (cleaned, calls);
    }
    (text.to_string(), Vec::new())
}

/// Parse [TOOL_CALLS] markers (Mistral format).
///
/// Mistral outputs: [TOOL_CALLS][{"name": "...", "arguments": {...}}]
fn parse_tool_calls_mistral(text: &str) -> (String, Vec<ToolCall>) {
    let marker = "[TOOL_CALLS]";
    if let Some(pos) = text.find(marker) {
        let before = text[..pos].trim().to_string();
        let json_part = text[pos + marker.len()..].trim();

        // Mistral outputs a JSON array of tool calls.
        if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(json_part) {
            let calls: Vec<ToolCall> = arr
                .into_iter()
                .filter_map(|v| {
                    let name = v.get("name")?.as_str()?.to_string();
                    let arguments = v.get("arguments")?;
                    Some(ToolCall {
                        id: generate_tool_call_id(),
                        type_: "function".into(),
                        function: FunctionCall {
                            name,
                            arguments: arguments.to_string(),
                        },
                    })
                })
                .collect();

            if !calls.is_empty() {
                return (before, calls);
            }
        }
    }

    (text.to_string(), Vec::new())
}

/// Try to parse a JSON string as a tool call: {"name": "...", "arguments": {...}}.
fn parse_json_tool_call(json_str: &str) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = v.get("arguments")?;

    Some(ToolCall {
        id: generate_tool_call_id(),
        type_: "function".into(),
        function: FunctionCall {
            name,
            arguments: arguments.to_string(),
        },
    })
}

/// Try to find bare JSON tool calls in text (no markers).
///
/// Looks for JSON objects containing a "name" field at the top level of the text.
/// Used as a fallback when the model doesn't use explicit markers.
fn parse_bare_json_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    let trimmed = text.trim();

    // Try as a single JSON object.
    if trimmed.starts_with('{') {
        if let Some(call) = parse_json_tool_call(trimmed) {
            calls.push(call);
            return Some(calls);
        }
    }

    // Try line-by-line (multiple tool calls, one per line).
    for line in trimmed.lines() {
        let line = line.trim();
        if line.starts_with('{') && line.contains("\"name\"") {
            if let Some(call) = parse_json_tool_call(line) {
                calls.push(call);
            }
        }
    }

    if calls.is_empty() { None } else { Some(calls) }
}

/// Remove JSON tool call objects from text (for cleaning content when
/// bare JSON is detected).
fn strip_json_tool_calls(text: &str) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('{') && trimmed.contains("\"name\"") {
            if parse_json_tool_call(trimmed).is_some() {
                continue; // Skip tool call lines.
            }
        }
        lines.push(line);
    }
    lines.join("\n").trim().to_string()
}

// ---------------------------------------------------------------------------
// Formatting tool results in chat messages.
//
// When the client sends tool results back, they arrive as messages with
// role="tool".  Each architecture formats these differently in its chat
// template.  These helpers are called from the format functions in chat.rs.
// ---------------------------------------------------------------------------

/// Format a tool result message for Llama 3.1+ template.
///
/// Llama uses <|start_header_id|>ipython<|end_header_id|> for tool results
/// (the "ipython" role is what Meta trained the model on).
pub(crate) fn format_tool_result_llama(tool_call_id: &str, content: &str) -> String {
    let _ = tool_call_id; // Llama doesn't use tool_call_id in the template.
    format!("<|start_header_id|>ipython<|end_header_id|>\n\n{content}<|eot_id|>")
}

/// Format a tool result message for ChatML-family templates (Qwen, Phi, etc.).
///
/// Qwen uses <|im_start|>tool with the result content, optionally wrapped
/// in <tool_response> markers.
pub(crate) fn format_tool_result_chatml(content: &str) -> String {
    format!("<|im_start|>tool\n<tool_response>\n{content}\n</tool_response><|im_end|>\n")
}

/// Format a tool result message for Gemma 3.
///
/// Gemma uses <start_of_turn>tool role.
pub(crate) fn format_tool_result_gemma(content: &str) -> String {
    format!("<start_of_turn>tool\n{content}<end_of_turn>\n")
}

/// Format a tool result message for Phi.
pub(crate) fn format_tool_result_phi(content: &str) -> String {
    format!("<|im_start|>tool<|im_sep|>\n{content}<|im_end|>\n")
}

/// Format a tool result for Mistral.
///
/// Mistral uses [TOOL_RESULTS] markers.
pub(crate) fn format_tool_result_mistral(content: &str) -> String {
    format!("[TOOL_RESULTS]{content}[/TOOL_RESULTS]")
}

// ---------------------------------------------------------------------------
// Tool name validation — filter parsed tool calls against defined tools.
//
// The model may hallucinate tool names that don't match any defined tool.
// This function removes invalid calls so clients only receive calls they
// can actually execute.
// ---------------------------------------------------------------------------

/// Filter tool calls to only include those whose function name matches a
/// defined tool.  Returns the filtered list and logs a warning for each
/// removed call so operators can diagnose prompt issues.
pub(crate) fn validate_tool_calls(
    calls: Vec<ToolCall>,
    defined_tools: &[ToolDefinition],
) -> Vec<ToolCall> {
    if defined_tools.is_empty() {
        // No tools defined — nothing to validate against.
        return calls;
    }
    calls
        .into_iter()
        .filter(|call| {
            let valid = defined_tools
                .iter()
                .any(|t| t.function.name == call.function.name);
            if !valid {
                eprintln!(
                    "  warning: model produced tool call '{}' which is not in the defined tools — skipping",
                    call.function.name
                );
            }
            valid
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            type_: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: Some("Get weather for a city.".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"]
                })),
            },
        }]
    }

    #[test]
    fn test_format_tools_llama() {
        let tools = make_tools();
        let prompt = format_tool_system_prompt(ModelArch::Llama, &tools);
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("function_name"));
        assert!(prompt.contains("\"name\""));
    }

    #[test]
    fn test_format_tools_mistral() {
        let tools = make_tools();
        let prompt = format_tool_system_prompt(ModelArch::Mistral, &tools);
        assert!(prompt.contains("[AVAILABLE_TOOLS]"));
        assert!(prompt.contains("[/AVAILABLE_TOOLS]"));
        assert!(prompt.contains("get_weather"));
    }

    #[test]
    fn test_format_tools_chatml() {
        let tools = make_tools();
        let prompt = format_tool_system_prompt(ModelArch::Qwen2, &tools);
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("<tool_call>"));
    }

    #[test]
    fn test_format_tools_empty() {
        let prompt = format_tool_system_prompt(ModelArch::Llama, &[]);
        assert!(prompt.is_empty());
    }

    // -- Parsing tests --

    #[test]
    fn test_parse_tool_call_xml_markers() {
        let text = "Let me check the weather.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"SF\"}}\n</tool_call>";
        let (content, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("SF"));
        assert!(content.contains("check the weather"));
        assert!(!content.contains("tool_call"));
    }

    #[test]
    fn test_parse_tool_call_multiple_xml() {
        let text = "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>";
        let (_, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
    }

    #[test]
    fn test_parse_tool_call_mistral() {
        let text =
            "Sure!\n[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}]";
        let (content, calls) = parse_tool_calls(ModelArch::Mistral, text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(content, "Sure!");
    }

    #[test]
    fn test_parse_tool_call_bare_json() {
        let text = "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"LA\"}}";
        let (_, calls) = parse_tool_calls(ModelArch::Llama, text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_parse_no_tool_calls() {
        let text = "The weather in SF is sunny and 72°F.";
        let (content, calls) = parse_tool_calls(ModelArch::Llama, text);
        assert!(calls.is_empty());
        assert_eq!(content, text);
    }

    #[test]
    fn test_tool_call_has_id() {
        let text = "<tool_call>\n{\"name\": \"test\", \"arguments\": {}}\n</tool_call>";
        let (_, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].type_, "function");
    }

    #[test]
    fn test_tool_definition_deserialize() {
        let json = r#"{"type": "function", "function": {"name": "test", "description": "A test function", "parameters": {"type": "object"}}}"#;
        let tool: ToolDefinition = serde_json::from_str(json).unwrap();
        assert_eq!(tool.function.name, "test");
        assert_eq!(
            tool.function.description.as_deref(),
            Some("A test function")
        );
    }

    #[test]
    fn test_tool_definition_default_type() {
        let json = r#"{"function": {"name": "test"}}"#;
        let tool: ToolDefinition = serde_json::from_str(json).unwrap();
        assert_eq!(tool.type_, "function");
    }

    #[test]
    fn test_bare_json_multiline() {
        let text = "{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n{\"name\": \"b\", \"arguments\": {\"y\": 2}}";
        let (_, calls) = parse_tool_calls(ModelArch::Llama, text);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_chatml_rejects_bare_json() {
        // ChatML archs must NOT parse bare JSON — only <tool_call> markers.
        let text = "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"LA\"}}";
        for arch in [
            ModelArch::Qwen2,
            ModelArch::Phi,
            ModelArch::Gemma3,
            ModelArch::GptOss,
        ] {
            let (content, calls) = parse_tool_calls(arch, text);
            assert!(calls.is_empty(), "arch {arch:?} should not parse bare JSON");
            assert_eq!(content, text);
        }
    }

    // -- Security: bounds-safe string slicing tests --

    #[test]
    fn test_tool_call_extraction_empty_tags() {
        // Empty <tool_call></tool_call> — should not panic, no call extracted.
        // When no calls are found, original text is returned (by design).
        let text = "<tool_call></tool_call>";
        let (_, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_tool_call_extraction_nested_angle_brackets() {
        // Edge case: angle brackets inside the tool_call body.
        let text = "<tool_call>{\"name\": \"f\", \"arguments\": {\"query\": \"a < b > c\"}}</tool_call>";
        let (_, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "f");
    }

    #[test]
    fn test_tool_call_extraction_just_opening_tag() {
        // Only an opening tag with nothing after it — should not panic.
        // When no calls are found, original text is returned (by design).
        let text = "<tool_call>";
        let (_, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_tool_call_extraction_whitespace_only() {
        // Opening and closing tags with only whitespace — should not panic.
        let text = "<tool_call>   </tool_call>";
        let (_, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert!(calls.is_empty());
    }

    // -- Multiple tool call tests (parallel tool calling) --

    #[test]
    fn test_mistral_multiple_tool_calls() {
        let text = "[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}, {\"name\": \"get_time\", \"arguments\": {\"tz\": \"EST\"}}]";
        let (content, calls) = parse_tool_calls(ModelArch::Mistral, text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        assert!(content.is_empty());
    }

    #[test]
    fn test_mistral_multiple_with_text_before() {
        let text = "Let me look that up for you.\n[TOOL_CALLS][{\"name\": \"a\", \"arguments\": {}}, {\"name\": \"b\", \"arguments\": {}}]";
        let (content, calls) = parse_tool_calls(ModelArch::Mistral, text);
        assert_eq!(calls.len(), 2);
        assert_eq!(content, "Let me look that up for you.");
    }

    #[test]
    fn test_chatml_text_between_tool_calls() {
        // Text before, between, and after tool_call blocks should be preserved.
        let text = "First.\n<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\nMiddle.\n<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>\nLast.";
        let (content, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert_eq!(calls.len(), 2);
        assert!(content.contains("First."));
        assert!(content.contains("Middle."));
        assert!(content.contains("Last."));
    }

    #[test]
    fn test_chatml_unclosed_tag_with_valid_json() {
        // Unclosed <tool_call> with valid JSON after it — should still parse.
        let text = "Here you go:\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"rust\"}}";
        let (content, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
        assert!(content.contains("Here you go:"));
    }

    #[test]
    fn test_chatml_malformed_json_skipped() {
        // Malformed JSON inside <tool_call> should be silently skipped.
        let text = "<tool_call>\n{not valid json}\n</tool_call>\n<tool_call>\n{\"name\": \"ok\", \"arguments\": {}}\n</tool_call>";
        let (_, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "ok");
    }

    #[test]
    fn test_chatml_missing_name_field_skipped() {
        // JSON object without "name" should be skipped.
        let text = "<tool_call>\n{\"function\": \"wrong_field\", \"arguments\": {}}\n</tool_call>";
        let (_, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_tool_call_ids_are_unique() {
        let text = "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"c\", \"arguments\": {}}\n</tool_call>";
        let (_, calls) = parse_tool_calls(ModelArch::Qwen2, text);
        assert_eq!(calls.len(), 3);
        // All IDs should be unique.
        let ids: std::collections::HashSet<&str> = calls.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids.len(), 3);
        // All IDs should have the "call_" prefix.
        for call in &calls {
            assert!(call.id.starts_with("call_"));
        }
    }

    #[test]
    fn test_llama_text_before_json_not_parsed() {
        // If there's text before the JSON, Llama parser should not find tool calls
        // (it only parses when the trimmed text starts with '{' or has JSON lines).
        let text = "Sure, here's the answer.\n\nThe weather is nice.";
        let (content, calls) = parse_tool_calls(ModelArch::Llama, text);
        assert!(calls.is_empty());
        assert_eq!(content, text);
    }

    #[test]
    fn test_llama_multiline_with_non_json_lines() {
        // Mix of JSON tool calls and regular text lines — only JSON should be parsed.
        let text = "{\"name\": \"a\", \"arguments\": {\"x\": 1}}\nsome text\n{\"name\": \"b\", \"arguments\": {\"y\": 2}}";
        let (content, calls) = parse_tool_calls(ModelArch::Llama, text);
        assert_eq!(calls.len(), 2);
        assert!(content.contains("some text"));
        assert!(!content.contains("\"name\""));
    }

    #[test]
    fn test_tool_call_with_nested_arguments() {
        // Deeply nested argument objects should be preserved correctly.
        let args = r#"{"query": {"bool": {"must": [{"match": {"field": "value"}}]}}}"#;
        let text = format!(
            "<tool_call>\n{{\"name\": \"search\", \"arguments\": {args}}}\n</tool_call>"
        );
        let (_, calls) = parse_tool_calls(ModelArch::Qwen2, &text);
        assert_eq!(calls.len(), 1);
        assert!(calls[0].function.arguments.contains("bool"));
        assert!(calls[0].function.arguments.contains("must"));
    }

    #[test]
    fn test_mistral_malformed_json_array() {
        // If the JSON array after [TOOL_CALLS] is invalid, no calls extracted.
        let text = "[TOOL_CALLS]{not an array}";
        let (content, calls) = parse_tool_calls(ModelArch::Mistral, text);
        assert!(calls.is_empty());
        assert_eq!(content, text);
    }

    #[test]
    fn test_format_tools_multiple() {
        let tools = vec![
            ToolDefinition {
                type_: "function".into(),
                function: FunctionDefinition {
                    name: "get_weather".into(),
                    description: Some("Get weather".into()),
                    parameters: None,
                },
            },
            ToolDefinition {
                type_: "function".into(),
                function: FunctionDefinition {
                    name: "get_time".into(),
                    description: Some("Get time".into()),
                    parameters: None,
                },
            },
        ];
        // All formatters should include both tool names.
        for arch in [ModelArch::Llama, ModelArch::Mistral, ModelArch::Qwen2] {
            let prompt = format_tool_system_prompt(arch, &tools);
            assert!(
                prompt.contains("get_weather"),
                "{arch:?} missing get_weather"
            );
            assert!(prompt.contains("get_time"), "{arch:?} missing get_time");
        }
    }

    #[test]
    fn test_chatml_no_description() {
        // Tool with no description should use fallback text.
        let tools = vec![ToolDefinition {
            type_: "function".into(),
            function: FunctionDefinition {
                name: "mystery".into(),
                description: None,
                parameters: None,
            },
        }];
        let prompt = format_tool_system_prompt(ModelArch::Qwen2, &tools);
        assert!(prompt.contains("mystery"));
        assert!(prompt.contains("No description."));
    }

    // -- Tool name validation tests --

    #[test]
    fn test_validate_tool_calls_filters_invalid_names() {
        let defs = make_tools(); // has "get_weather"
        let calls = vec![
            ToolCall {
                id: "call_1".into(),
                type_: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: "{}".into(),
                },
            },
            ToolCall {
                id: "call_2".into(),
                type_: "function".into(),
                function: FunctionCall {
                    name: "hallucinated_tool".into(),
                    arguments: "{}".into(),
                },
            },
        ];
        let valid = validate_tool_calls(calls, &defs);
        assert_eq!(valid.len(), 1);
        assert_eq!(valid[0].function.name, "get_weather");
    }

    #[test]
    fn test_validate_tool_calls_empty_defs_passes_all() {
        let calls = vec![ToolCall {
            id: "call_1".into(),
            type_: "function".into(),
            function: FunctionCall {
                name: "anything".into(),
                arguments: "{}".into(),
            },
        }];
        let valid = validate_tool_calls(calls, &[]);
        assert_eq!(valid.len(), 1);
    }

    #[test]
    fn test_validate_tool_calls_all_valid() {
        let defs = make_tools();
        let calls = vec![ToolCall {
            id: "call_1".into(),
            type_: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: "{\"city\":\"NYC\"}".into(),
            },
        }];
        let valid = validate_tool_calls(calls, &defs);
        assert_eq!(valid.len(), 1);
    }

    #[test]
    fn test_validate_tool_calls_all_invalid() {
        let defs = make_tools();
        let calls = vec![ToolCall {
            id: "call_1".into(),
            type_: "function".into(),
            function: FunctionCall {
                name: "nonexistent".into(),
                arguments: "{}".into(),
            },
        }];
        let valid = validate_tool_calls(calls, &defs);
        assert!(valid.is_empty());
    }
}
