// ===========================================================================
// Chat template — formatting messages for instruct-tuned models.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Converts a list of chat messages (system, user, assistant) into the
//   special-token-delimited string format that instruct models expect.
//   Supports Llama 3, ChatML (Qwen), Gemma 3, Phi, and Mistral formats.
//
// Why do instruct models need a chat template?
//   Base models are trained on raw text — they just predict the next token.
//   Instruct models are fine-tuned on conversations that follow a specific
//   format using special tokens to mark role boundaries.  If you feed an
//   instruct model raw text without the template, it doesn't know it's
//   supposed to "answer" — it just continues the text like a base model.
//
// Llama 3 chat format:
//   <|start_header_id|>system<|end_header_id|>
//
//   You are a helpful assistant.<|eot_id|>
//   <|start_header_id|>user<|end_header_id|>
//
//   What is 2+2?<|eot_id|>
//   <|start_header_id|>assistant<|end_header_id|>
//
//   Uses special tokens:
//     <|start_header_id|> → 128006
//     <|end_header_id|>   → 128007
//     <|eot_id|>          → 128009
//
// ChatML format (Qwen 2.5):
//   <|im_start|>system
//   You are a helpful assistant.<|im_end|>
//   <|im_start|>user
//   What is 2+2?<|im_end|>
//   <|im_start|>assistant
//
//   ChatML (Chat Markup Language) was originally created by OpenAI and
//   adopted by many model families including Qwen.  It's simpler than
//   Llama 3's format — just im_start/im_end markers:
//     <|im_start|> → 151644
//     <|im_end|>   → 151645
//
// Base model vs. instruct model:
//   The weights have the same architecture.  The only difference is training
//   data — instruct models saw millions of conversations in their respective
//   format during fine-tuning.  Using the WRONG template produces garbage —
//   the model literally doesn't understand role boundaries if the special
//   tokens don't match what it was trained on.
//
// Why not use a Jinja2 template engine?
//   HuggingFace stores chat templates as Jinja2 strings in tokenizer_config.json.
//   But for Llama 3 and ChatML, the templates are simple enough that string
//   concatenation does the job — no template engine dependency needed.
// ===========================================================================

use base64::Engine;

use super::config::ModelArch;
use super::thinking;
use super::tools::{self, ToolCall};

/// Raw image data attached to a message for vision models.
///
/// Contains the raw bytes of an image file (JPEG, PNG, or WebP).  Images are
/// base64-encoded for JSON serialisation and decoded back to raw bytes on
/// deserialisation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ImageData {
    /// Raw image bytes (JPEG, PNG, or WebP).
    #[serde(with = "base64_bytes")]
    pub data: Vec<u8>,
}

/// Serde helper: serialise `Vec<u8>` as a base64 string.
///
/// This is needed because JSON has no binary type — image bytes must be encoded
/// as base64 strings for transport.  The OpenAI and Anthropic APIs both use
/// base64 for inline image data.
mod base64_bytes {
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(data: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(data))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        base64::engine::general_purpose::STANDARD
            .decode(&s)
            .map_err(serde::de::Error::custom)
    }
}

/// A single message in a chat conversation.
///
/// Roles follow the OpenAI convention used by most LLM APIs:
///   - "system":    instructions for the model's behaviour
///   - "user":      the human's message
///   - "assistant": the model's response (for multi-turn conversations)
///   - "tool":      result from a tool call (paired with tool_call_id)
///
/// For vision models, user messages can include inline images via the `images`
/// field.  The chat template inserts appropriate image placeholder tokens
/// (e.g. `<|vision_start|><|image_pad|><|vision_end|>` for Qwen).
///
/// Custom deserialization:
///   The `content` field can be a plain string, null, or an array of content
///   parts (OpenAI multi-modal format).  When content is an array, text parts
///   are concatenated into `content` and image_url parts with base64 data URLs
///   are decoded into `images`.  The Anthropic API uses a similar array format
///   with `{"type": "image", "source": {"type": "base64", ...}}` blocks.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Message {
    pub role: String,
    /// Message text.  May be empty for assistant tool-call-only messages
    /// (OpenAI sends content=null when the model only produces tool calls).
    pub content: String,
    /// Tool calls made by the assistant (only present on role="assistant" messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// ID of the tool call this message is responding to (only on role="tool" messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Image data for vision models (raw bytes per image, e.g. JPEG/PNG).
    /// When present, the chat template prepends vision placeholder tokens
    /// before the text content so the model knows where image embeddings go.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ImageData>>,
}

/// Custom deserializer for Message that handles both plain-string and
/// multi-modal array content formats.
///
/// OpenAI vision API sends content as an array of typed parts:
/// ```json
/// {"role": "user", "content": [
///   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
///   {"type": "text", "text": "What's in this image?"}
/// ]}
/// ```
///
/// Anthropic vision API sends content as an array with image blocks:
/// ```json
/// {"role": "user", "content": [
///   {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
///   {"type": "text", "text": "What's in this image?"}
/// ]}
/// ```
///
/// This deserializer handles all three forms: string, null, and array.
impl<'de> serde::Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        /// Raw helper struct for initial JSON deserialization.
        /// Content is kept as a raw JSON value so we can inspect its type.
        #[derive(serde::Deserialize)]
        struct RawMessage {
            role: String,
            #[serde(default)]
            content: Option<serde_json::Value>,
            #[serde(default)]
            tool_calls: Option<Vec<ToolCall>>,
            #[serde(default)]
            tool_call_id: Option<String>,
            #[serde(default)]
            images: Option<Vec<ImageData>>,
        }

        let raw = RawMessage::deserialize(deserializer)?;

        let (content, mut images) = match raw.content {
            // Null or absent → empty string (OpenAI sends null for tool-call-only messages).
            None => (String::new(), None),
            Some(serde_json::Value::String(s)) => (s, None),
            Some(serde_json::Value::Array(parts)) => {
                // Multi-modal content array (OpenAI or Anthropic format).
                parse_content_parts(&parts)
            }
            Some(_) => (String::new(), None),
        };

        // Merge images from the content array with any explicitly provided images field.
        if let Some(explicit) = raw.images {
            match &mut images {
                Some(existing) => existing.extend(explicit),
                None => images = Some(explicit),
            }
        }

        Ok(Message {
            role: raw.role,
            content,
            tool_calls: raw.tool_calls,
            tool_call_id: raw.tool_call_id,
            images,
        })
    }
}

/// Parse an array of content parts from either OpenAI or Anthropic vision format.
///
/// Returns (concatenated_text, optional_images).
fn parse_content_parts(parts: &[serde_json::Value]) -> (String, Option<Vec<ImageData>>) {
    let mut text = String::new();
    let mut images: Vec<ImageData> = Vec::new();

    for part in parts {
        let type_ = part.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match type_ {
            // Text content block (shared by OpenAI and Anthropic).
            "text" => {
                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(t);
                }
            }
            // OpenAI image_url block: {"type": "image_url", "image_url": {"url": "data:...;base64,DATA"}}
            "image_url" => {
                if let Some(url) = part
                    .get("image_url")
                    .and_then(|v| v.get("url"))
                    .and_then(|v| v.as_str())
                {
                    if let Some(data) = decode_data_url(url) {
                        images.push(ImageData { data });
                    }
                }
            }
            // Anthropic image block: {"type": "image", "source": {"type": "base64", "data": "..."}}
            "image" => {
                if let Some(source) = part.get("source") {
                    let source_type = source.get("type").and_then(|v| v.as_str());
                    if source_type == Some("base64") {
                        if let Some(b64) = source.get("data").and_then(|v| v.as_str()) {
                            if let Ok(data) =
                                base64::engine::general_purpose::STANDARD.decode(b64)
                            {
                                images.push(ImageData { data });
                            }
                        }
                    }
                }
            }
            _ => {} // Unknown content type — skip gracefully.
        }
    }

    let images = if images.is_empty() { None } else { Some(images) };
    (text, images)
}

/// Decode a `data:image/...;base64,DATA` URL into raw bytes.
///
/// Returns None if the URL is not a valid base64 data URL.  Regular HTTP URLs
/// are not supported (we don't fetch remote images).
fn decode_data_url(url: &str) -> Option<Vec<u8>> {
    // data:image/jpeg;base64,/9j/4AAQ...
    let suffix = url.strip_prefix("data:")?;
    let (_, b64) = suffix.split_once(";base64,")?;
    // Reject oversized payloads before decoding to prevent OOM.
    // 100 MB of base64 ≈ 75 MB decoded — far larger than any reasonable image.
    if b64.len() > 100_000_000 {
        return None;
    }
    base64::engine::general_purpose::STANDARD.decode(b64).ok()
}

/// Return the number of images attached to a message (0 if none).
fn image_count(msg: &Message) -> usize {
    msg.images.as_ref().map_or(0, |imgs| imgs.len())
}

/// Prepend vision placeholder tokens for each image in a message.
///
/// Different model families use different marker formats:
///   - ChatML (Qwen): `<|vision_start|><|image_pad|><|vision_end|>\n` per image
///   - Gemma 3:       `<start_of_image><image_soft_token><end_of_image>\n` per image
///
/// We insert a single `<|image_pad|>` / `<image_soft_token>` per image as a
/// placeholder.  The actual number of vision tokens depends on the processed
/// image resolution and will be expanded during tokenization / embedding scatter.
fn vision_prefix(msg: &Message, arch: ModelArch) -> String {
    let n = image_count(msg);
    if n == 0 {
        return String::new();
    }
    let mut prefix = String::new();
    for _ in 0..n {
        match arch {
            ModelArch::Qwen2 | ModelArch::Qwen3Moe | ModelArch::Qwen3_5 | ModelArch::GptOss => {
                prefix.push_str("<|vision_start|><|image_pad|><|vision_end|>\n");
            }
            ModelArch::Gemma3 => {
                prefix.push_str("<start_of_image><image_soft_token><end_of_image>\n");
            }
            // Other architectures don't have vision support yet — images are
            // silently ignored rather than crashing.
            _ => {}
        }
    }
    prefix
}

/// Format messages using the correct chat template for the model architecture.
///
/// Each model family was fine-tuned on its own chat format.  Using the wrong
/// template produces gibberish — the model literally doesn't understand the
/// role boundaries if the special tokens are wrong.
pub(crate) fn format_chat(arch: ModelArch, messages: &[Message]) -> String {
    format_chat_with_thinking(arch, messages, None)
}

/// Format messages with optional thinking control.
///
/// When `thinking` is Some(true), a thinking prompt tag is appended after
/// the generation prompt so the model begins reasoning.  When Some(false),
/// a suppression tag is appended to prevent thinking.  When None, the
/// default is used: thinking-capable models (Qwen 3, 3.5) get thinking
/// enabled automatically, since they were trained to always produce
/// `<think>...</think>` blocks and loop without the prompt tag.
///
/// See `thinking.rs` for the architecture-specific tags.
pub(crate) fn format_chat_with_thinking(
    arch: ModelArch,
    messages: &[Message],
    thinking_enabled: Option<bool>,
) -> String {
    let mut out = match arch {
        ModelArch::Llama => format_llama3(messages),
        ModelArch::Mistral | ModelArch::Mixtral => format_mistral(messages),
        ModelArch::Qwen2 | ModelArch::Qwen3Moe | ModelArch::Qwen3_5 | ModelArch::GptOss | ModelArch::NemotronH => {
            format_chatml(messages)
        }
        ModelArch::Phi => format_phi(messages),
        ModelArch::Gemma3 => format_gemma3(messages),
    };

    // Resolve thinking: explicit preference wins, otherwise default to enabled
    // for thinking-capable architectures (Qwen 3/3.5 were trained to always
    // produce <think> blocks — without the prompt tag they loop endlessly).
    let enabled = thinking_enabled.unwrap_or_else(|| thinking::supports_thinking(arch));

    if enabled {
        if let Some(tag) = thinking::thinking_prompt_tag(arch, true) {
            out.push_str(tag);
        }
    } else if thinking_enabled == Some(false) {
        if let Some(tag) = thinking::thinking_suppress_tag(arch) {
            out.push_str(tag);
        }
    }

    out
}

/// Format chat messages into the Llama 3 instruct template string.
///
/// The returned string contains special token markers (e.g. `<|start_header_id|>`)
/// as literal text.  The caller must encode this with special-token parsing
/// enabled so the tokenizer maps these markers to their proper token IDs.
///
/// The `<|start_header_id|>assistant<|end_header_id|>` generation prompt is
/// always appended — this tells the model "now it's your turn to speak".
fn format_llama3(messages: &[Message]) -> String {
    let mut out = String::with_capacity(512);

    // Note: we do NOT include <|begin_of_text|> here.  The HF tokenizer's
    // encode(text, add_special_tokens=true) automatically prepends BOS (128000).
    // Including it in the template would produce a duplicate BOS.

    for msg in messages {
        if msg.role == "tool" {
            // Tool results use the "ipython" role in Llama's template.
            out.push_str(&tools::format_tool_result_llama(
                msg.tool_call_id.as_deref().unwrap_or(""),
                &msg.content,
            ));
            continue;
        }

        // Role header: <|start_header_id|>role<|end_header_id|>\n\n
        out.push_str("<|start_header_id|>");
        out.push_str(&msg.role);
        out.push_str("<|end_header_id|>\n\n");

        // Message content followed by end-of-turn marker.
        out.push_str(&msg.content);
        out.push_str("<|eot_id|>");
    }

    // Generation prompt — tells the model to start generating as assistant.
    // The two trailing newlines match Meta's official template.
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

    out
}

/// Format chat messages into the ChatML template string (Qwen 2.5).
///
/// ChatML uses `<|im_start|>` and `<|im_end|>` markers with a single newline
/// separating the role name from the content.  This is simpler than Llama 3's
/// nested header markers.
///
/// Same as Llama 3, the returned string contains special token markers as
/// literal text — the tokenizer parses them into token IDs.
fn format_chatml(messages: &[Message]) -> String {
    let mut out = String::with_capacity(512);

    // Note: no BOS here either — the HF tokenizer adds it automatically
    // when encoding with add_special_tokens=true.

    // Qwen 2.5 models are trained with a mandatory system message.  HuggingFace's
    // `apply_chat_template()` injects "You are Qwen, created by Alibaba Cloud.
    // You are a helpful assistant." when no system message is present.  Omitting
    // it shifts all token positions and degrades output quality — especially at
    // small model sizes (3B) where greedy decoding falls into repetition loops.
    let has_system = messages.iter().any(|m| m.role == "system");
    if !has_system {
        out.push_str("<|im_start|>system\n");
        out.push_str("You are a helpful assistant.<|im_end|>\n");
    }

    for msg in messages {
        if msg.role == "tool" {
            out.push_str(&tools::format_tool_result_chatml(&msg.content));
            continue;
        }

        // <|im_start|>role\n[vision_prefix]content<|im_end|>\n
        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push('\n');
        // Insert vision placeholders before text for user messages with images.
        // Uses the Qwen/ChatML vision token format (shared by all ChatML archs).
        out.push_str(&vision_prefix(msg, ModelArch::Qwen2));
        out.push_str(&msg.content);
        out.push_str("<|im_end|>\n");
    }

    // Generation prompt — model fills in the assistant response.
    out.push_str("<|im_start|>assistant\n");

    out
}

/// Format chat messages into the Gemma 3 instruct template string.
///
/// Gemma 3 uses `<start_of_turn>` and `<end_of_turn>` markers with a newline
/// separating the role name from the content:
///
///   <start_of_turn>user
///   What is 2+2?<end_of_turn>
///   <start_of_turn>model
///
/// Note: Gemma uses "model" instead of "assistant" for the model's role.
///
/// Special tokens:
///   <start_of_turn> → 106
///   <end_of_turn>   → 107
fn format_gemma3(messages: &[Message]) -> String {
    let mut out = String::with_capacity(512);

    for msg in messages {
        if msg.role == "tool" {
            out.push_str(&tools::format_tool_result_gemma(&msg.content));
            continue;
        }

        out.push_str("<start_of_turn>");
        // Gemma uses "model" for the assistant role.
        let role = if msg.role == "assistant" {
            "model"
        } else {
            &msg.role
        };
        out.push_str(role);
        out.push('\n');
        // Insert vision placeholders before text for user messages with images.
        out.push_str(&vision_prefix(msg, ModelArch::Gemma3));
        out.push_str(&msg.content);
        out.push_str("<end_of_turn>\n");
    }

    // Generation prompt — model fills in the response.
    out.push_str("<start_of_turn>model\n");

    out
}

/// Format chat messages into the Phi instruct template string.
///
/// Phi-4 uses a ChatML-like format but with `<|im_sep|>` between the role
/// name and message content (where ChatML uses a newline).
///
///   <|im_start|>system<|im_sep|>
///   You are a helpful assistant.<|im_end|>
///   <|im_start|>user<|im_sep|>
///   What is 2+2?<|im_end|>
///   <|im_start|>assistant<|im_sep|>
///
/// Special tokens:
///   <|im_start|> → 100264
///   <|im_sep|>   → 100266
///   <|im_end|>   → 100265
fn format_phi(messages: &[Message]) -> String {
    let mut out = String::with_capacity(512);

    for msg in messages {
        if msg.role == "tool" {
            out.push_str(&tools::format_tool_result_phi(&msg.content));
            continue;
        }

        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push_str("<|im_sep|>\n");
        out.push_str(&msg.content);
        out.push_str("<|im_end|>\n");
    }

    // Generation prompt.
    out.push_str("<|im_start|>assistant<|im_sep|>\n");

    out
}

/// Format chat messages into the Mistral instruct template string.
///
/// Mistral uses `[INST]` and `[/INST]` markers.  If a system message is present,
/// it's prepended to the first user message (separated by a blank line).
/// Multi-turn conversations alternate `[INST] user [/INST] assistant</s>`.
///
///   <s>[INST] You are helpful.\n\nWhat is 2+2? [/INST]
///
/// The tokenizer handles `<s>` (BOS=1) and `</s>` (EOS=2) as special tokens.
fn format_mistral(messages: &[Message]) -> String {
    let mut out = String::with_capacity(512);

    // Extract optional system message.
    let (system, rest) = if messages.first().map(|m| m.role.as_str()) == Some("system") {
        (Some(messages[0].content.as_str()), &messages[1..])
    } else {
        (None, messages.as_ref())
    };

    let mut first_user = true;
    for msg in rest {
        match msg.role.as_str() {
            "user" => {
                out.push_str(" [INST] ");
                // Prepend system message to the first user message.
                if first_user {
                    if let Some(sys) = system {
                        out.push_str(sys);
                        out.push_str("\n\n");
                    }
                    first_user = false;
                }
                out.push_str(&msg.content);
                out.push_str(" [/INST]");
            }
            "assistant" => {
                out.push(' ');
                out.push_str(&msg.content);
                out.push_str("</s>");
            }
            "tool" => {
                out.push_str(&tools::format_tool_result_mistral(&msg.content));
            }
            _ => {}
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            images: None,
        }
    }

    #[test]
    fn test_llama3_system_and_user() {
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let result = format_chat(ModelArch::Llama, &messages);
        assert_eq!(
            result,
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn test_llama3_multi_turn() {
        let messages = vec![
            msg("system", "Be concise."),
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
            msg("user", "How are you?"),
        ];
        let result = format_chat(ModelArch::Llama, &messages);
        assert!(result.contains(
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>"
        ));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_chatml_format() {
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let result = format_chat(ModelArch::Qwen2, &messages);
        assert_eq!(
            result,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_chatml_used_for_qwen_variants() {
        let messages = vec![msg("user", "Hi")];
        let qwen2 = format_chat(ModelArch::Qwen2, &messages);
        let qwen3_moe = format_chat(ModelArch::Qwen3Moe, &messages);
        let qwen3_5 = format_chat(ModelArch::Qwen3_5, &messages);
        // Qwen 3 MoE and 3.5 support thinking — they get <think>\n appended.
        // The base ChatML structure is the same, just with the thinking prompt tag.
        assert!(qwen3_moe.starts_with(&qwen2[..qwen2.len() - 1]));
        assert!(qwen3_5.starts_with(&qwen2[..qwen2.len() - 1]));
        assert!(qwen3_moe.ends_with("<think>\n"));
        assert!(qwen3_5.ends_with("<think>\n"));
        // Qwen 2 does not support thinking — no tag.
        assert!(!qwen2.contains("<think>"));
    }

    #[test]
    fn test_phi_format() {
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let result = format_chat(ModelArch::Phi, &messages);
        assert_eq!(
            result,
            "<|im_start|>system<|im_sep|>\nYou are helpful.<|im_end|>\n\
             <|im_start|>user<|im_sep|>\nHello<|im_end|>\n\
             <|im_start|>assistant<|im_sep|>\n"
        );
    }

    #[test]
    fn test_gemma3_format() {
        let messages = vec![msg("user", "Hello")];
        let result = format_chat(ModelArch::Gemma3, &messages);
        assert_eq!(
            result,
            "<start_of_turn>user\nHello<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn test_gemma3_assistant_becomes_model() {
        let messages = vec![
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
            msg("user", "Bye"),
        ];
        let result = format_chat(ModelArch::Gemma3, &messages);
        // "assistant" role should be mapped to "model" in Gemma 3
        assert!(result.contains("<start_of_turn>model\nHello!<end_of_turn>"));
        assert!(!result.contains("<start_of_turn>assistant"));
    }

    #[test]
    fn test_mistral_system_and_user() {
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let result = format_chat(ModelArch::Mistral, &messages);
        assert_eq!(result, " [INST] You are helpful.\n\nHello [/INST]");
    }

    #[test]
    fn test_mistral_multi_turn() {
        let messages = vec![
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
            msg("user", "How are you?"),
        ];
        let result = format_chat(ModelArch::Mistral, &messages);
        assert_eq!(
            result,
            " [INST] Hi [/INST] Hello!</s> [INST] How are you? [/INST]"
        );
    }

    #[test]
    fn test_all_formats_end_with_generation_prompt() {
        let messages = vec![msg("user", "Hi")];
        // Every format ends with a prompt for the assistant to generate
        let llama = format_chat(ModelArch::Llama, &messages);
        assert!(llama.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));

        let chatml = format_chat(ModelArch::Qwen2, &messages);
        assert!(chatml.ends_with("<|im_start|>assistant\n"));

        let phi = format_chat(ModelArch::Phi, &messages);
        assert!(phi.ends_with("<|im_start|>assistant<|im_sep|>\n"));

        let gemma = format_chat(ModelArch::Gemma3, &messages);
        assert!(gemma.ends_with("<start_of_turn>model\n"));

        let mistral = format_chat(ModelArch::Mistral, &messages);
        assert!(mistral.ends_with("[/INST]"));
    }

    #[test]
    fn test_empty_messages() {
        let messages: Vec<Message> = vec![];
        // Should just produce the generation prompt
        let llama = format_chat(ModelArch::Llama, &messages);
        assert_eq!(llama, "<|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    // -- Thinking control tests --

    #[test]
    fn test_thinking_enabled_qwen3_5() {
        let messages = vec![msg("user", "Solve 2+2")];
        let result = format_chat_with_thinking(ModelArch::Qwen3_5, &messages, Some(true));
        // Should end with the generation prompt + <think> tag
        assert!(result.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn test_thinking_disabled_qwen3_5() {
        let messages = vec![msg("user", "Solve 2+2")];
        let result = format_chat_with_thinking(ModelArch::Qwen3_5, &messages, Some(false));
        // When explicitly disabled, no thinking tag is added.
        assert!(!result.ends_with("<think>\n"));
    }

    #[test]
    fn test_thinking_none_defaults_to_enabled_for_thinking_models() {
        // Qwen 3.5 supports thinking — None should default to enabled.
        let messages = vec![msg("user", "Hello")];
        let with_none = format_chat_with_thinking(ModelArch::Qwen3_5, &messages, None);
        let with_true = format_chat_with_thinking(ModelArch::Qwen3_5, &messages, Some(true));
        assert_eq!(with_none, with_true, "None should default to thinking enabled for Qwen 3.5");
        assert!(with_none.ends_with("<think>\n"));
    }

    #[test]
    fn test_thinking_none_no_tag_for_non_thinking_models() {
        // Llama doesn't support thinking — None should not add a tag.
        let messages = vec![msg("user", "Hello")];
        let with_none = format_chat_with_thinking(ModelArch::Llama, &messages, None);
        assert!(!with_none.contains("<think>"));
    }

    #[test]
    fn test_format_chat_defaults_thinking_for_qwen() {
        // format_chat() passes None, which should auto-enable thinking for Qwen 3.5.
        let messages = vec![msg("user", "Solve 2+2")];
        let result = format_chat(ModelArch::Qwen3_5, &messages);
        assert!(result.ends_with("<think>\n"), "format_chat should enable thinking for Qwen 3.5");
    }

    #[test]
    fn test_format_chat_no_thinking_for_llama() {
        let messages = vec![msg("user", "Hello")];
        let result = format_chat(ModelArch::Llama, &messages);
        assert!(!result.contains("<think>"), "format_chat should not add thinking for Llama");
    }

    #[test]
    fn test_thinking_unsupported_arch_no_tag() {
        let messages = vec![msg("user", "Hello")];
        let result = format_chat_with_thinking(ModelArch::Llama, &messages, Some(true));
        // Llama doesn't support thinking — should be same as without
        let without = format_chat(ModelArch::Llama, &messages);
        assert_eq!(result, without);
    }

    // -- Security: decode_data_url size limit tests --

    #[test]
    fn test_decode_data_url_normal_image() {
        // Small valid base64 PNG (1x1 transparent pixel).
        let b64 = base64::engine::general_purpose::STANDARD.encode(&[0u8; 32]);
        let url = format!("data:image/png;base64,{b64}");
        assert!(decode_data_url(&url).is_some());
    }

    #[test]
    fn test_decode_data_url_rejects_oversized() {
        // 101 MB of base64 characters — should be rejected before decoding.
        let huge = "A".repeat(101_000_000);
        let url = format!("data:image/png;base64,{huge}");
        assert!(decode_data_url(&url).is_none());
    }

    #[test]
    fn test_decode_data_url_just_under_limit() {
        // 99 MB — should be accepted (though will fail base64 decode with
        // invalid data, that's fine — we're testing the size gate).
        let big = "A".repeat(99_000_000);
        let url = format!("data:image/png;base64,{big}");
        // May return None due to invalid base64, but should NOT be rejected
        // by the size check — the size check only rejects > 100M.
        // Just verify it doesn't panic.
        let _ = decode_data_url(&url);
    }

    // -- Tool message formatting tests --

    #[test]
    fn test_llama_tool_result_formatting() {
        let messages = vec![
            msg("user", "Weather?"),
            Message {
                role: "tool".into(),
                content: "Sunny, 72F".into(),
                tool_calls: None,
                tool_call_id: Some("call_123".into()),
                images: None,
            },
        ];
        let result = format_chat(ModelArch::Llama, &messages);
        assert!(result.contains("ipython"));
        assert!(result.contains("Sunny, 72F"));
    }

    #[test]
    fn test_chatml_tool_result_formatting() {
        let messages = vec![
            msg("user", "Weather?"),
            Message {
                role: "tool".into(),
                content: "Sunny, 72F".into(),
                tool_calls: None,
                tool_call_id: None,
                images: None,
            },
        ];
        let result = format_chat(ModelArch::Qwen2, &messages);
        assert!(result.contains("<tool_response>"));
        assert!(result.contains("Sunny, 72F"));
        assert!(result.contains("</tool_response>"));
    }

    #[test]
    fn test_gemma_tool_result_formatting() {
        let messages = vec![
            msg("user", "Weather?"),
            Message {
                role: "tool".into(),
                content: "{\"temp\": 72}".into(),
                tool_calls: None,
                tool_call_id: None,
                images: None,
            },
        ];
        let result = format_chat(ModelArch::Gemma3, &messages);
        assert!(result.contains("<start_of_turn>tool"));
        assert!(result.contains("{\"temp\": 72}"));
    }

    #[test]
    fn test_mistral_tool_result_formatting() {
        let messages = vec![
            msg("user", "Weather?"),
            Message {
                role: "tool".into(),
                content: "Sunny".into(),
                tool_calls: None,
                tool_call_id: None,
                images: None,
            },
        ];
        let result = format_chat(ModelArch::Mistral, &messages);
        assert!(result.contains("[TOOL_RESULTS]"));
        assert!(result.contains("Sunny"));
        assert!(result.contains("[/TOOL_RESULTS]"));
    }

    #[test]
    fn test_phi_tool_result_formatting() {
        let messages = vec![
            msg("user", "Weather?"),
            Message {
                role: "tool".into(),
                content: "Sunny".into(),
                tool_calls: None,
                tool_call_id: None,
                images: None,
            },
        ];
        let result = format_chat(ModelArch::Phi, &messages);
        assert!(result.contains("<|im_start|>tool<|im_sep|>"));
        assert!(result.contains("Sunny"));
    }

    // -- Content deserialization tests --

    #[test]
    fn test_null_content_deserializes_to_empty() {
        let json = r#"{"role": "assistant", "content": null}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content, "");
    }

    #[test]
    fn test_content_array_text_parts() {
        let json = r#"{"role": "user", "content": [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"}
        ]}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content, "Hello World");
    }

    #[test]
    fn test_content_array_with_image_url() {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&[1u8, 2, 3]);
        let json = format!(
            r#"{{"role": "user", "content": [
                {{"type": "image_url", "image_url": {{"url": "data:image/png;base64,{b64}"}}}},
                {{"type": "text", "text": "Describe this"}}
            ]}}"#
        );
        let msg: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg.content, "Describe this");
        assert_eq!(msg.images.as_ref().unwrap().len(), 1);
        assert_eq!(msg.images.as_ref().unwrap()[0].data, vec![1, 2, 3]);
    }

    #[test]
    fn test_content_array_anthropic_image() {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&[4u8, 5, 6]);
        let json = format!(
            r#"{{"role": "user", "content": [
                {{"type": "image", "source": {{"type": "base64", "data": "{b64}"}}}},
                {{"type": "text", "text": "What is this?"}}
            ]}}"#
        );
        let msg: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg.content, "What is this?");
        assert_eq!(msg.images.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_vision_prefix_qwen_multiple_images() {
        let msg = Message {
            role: "user".into(),
            content: "Describe both.".into(),
            tool_calls: None,
            tool_call_id: None,
            images: Some(vec![
                ImageData { data: vec![1] },
                ImageData { data: vec![2] },
            ]),
        };
        let prefix = vision_prefix(&msg, ModelArch::Qwen2);
        // Should have two vision placeholder blocks.
        assert_eq!(prefix.matches("<|image_pad|>").count(), 2);
    }

    #[test]
    fn test_vision_prefix_gemma() {
        let msg = Message {
            role: "user".into(),
            content: "Describe.".into(),
            tool_calls: None,
            tool_call_id: None,
            images: Some(vec![ImageData { data: vec![1] }]),
        };
        let prefix = vision_prefix(&msg, ModelArch::Gemma3);
        assert!(prefix.contains("<start_of_image>"));
        assert!(prefix.contains("<image_soft_token>"));
    }

    #[test]
    fn test_vision_prefix_no_images() {
        let msg = msg("user", "Hello");
        let prefix = vision_prefix(&msg, ModelArch::Qwen2);
        assert!(prefix.is_empty());
    }

    // -- Default ChatML system message tests --

    #[test]
    fn test_chatml_default_system_when_absent() {
        // When no system message is provided, ChatML should inject a default.
        let messages = vec![msg("user", "Hello")];
        let result = format_chat(ModelArch::Qwen2, &messages);
        assert!(
            result.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"),
            "should inject default system message when absent: {result}"
        );
    }

    #[test]
    fn test_chatml_no_double_system_when_present() {
        // When user provides a system message, don't add a second one.
        let messages = vec![
            msg("system", "You are a pirate."),
            msg("user", "Hello"),
        ];
        let result = format_chat(ModelArch::Qwen2, &messages);
        // Should have exactly one system block.
        assert_eq!(
            result.matches("<|im_start|>system").count(),
            1,
            "should not duplicate system message: {result}"
        );
        assert!(
            result.contains("You are a pirate."),
            "should use user-provided system message: {result}"
        );
    }

    // -- Thinking suppression tests --

    #[test]
    fn test_thinking_suppress_qwen3_5() {
        // thinking=false should inject closed <think> block for Qwen 3.5.
        let messages = vec![msg("user", "Hi")];
        let result = format_chat_with_thinking(ModelArch::Qwen3_5, &messages, Some(false));
        assert!(
            result.ends_with("<|im_start|>assistant\n<think>\n</think>\n"),
            "should inject closed think block to suppress thinking: {result}"
        );
    }

    #[test]
    fn test_thinking_suppress_llama_no_tag() {
        // thinking=false on non-thinking arch should add nothing.
        let messages = vec![msg("user", "Hi")];
        let with_false = format_chat_with_thinking(ModelArch::Llama, &messages, Some(false));
        let without = format_chat(ModelArch::Llama, &messages);
        assert_eq!(with_false, without, "Llama shouldn't get any thinking tag");
    }
}
