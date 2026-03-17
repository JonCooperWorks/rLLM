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

use super::config::ModelArch;
use super::tools::{self, ToolCall};

/// A single message in a chat conversation.
///
/// Roles follow the OpenAI convention used by most LLM APIs:
///   - "system":    instructions for the model's behaviour
///   - "user":      the human's message
///   - "assistant": the model's response (for multi-turn conversations)
///   - "tool":      result from a tool call (paired with tool_call_id)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Message {
    pub role: String,
    /// Message text.  May be empty for assistant tool-call-only messages
    /// (OpenAI sends content=null when the model only produces tool calls).
    #[serde(default, deserialize_with = "deserialize_nullable_string")]
    pub content: String,
    /// Tool calls made by the assistant (only present on role="assistant" messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// ID of the tool call this message is responding to (only on role="tool" messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Deserialize a string that may be null (e.g. OpenAI sends content=null
/// for assistant messages that only contain tool calls).
fn deserialize_nullable_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    Option::<String>::deserialize(deserializer).map(|o| o.unwrap_or_default())
}

/// Format messages using the correct chat template for the model architecture.
///
/// Each model family was fine-tuned on its own chat format.  Using the wrong
/// template produces gibberish — the model literally doesn't understand the
/// role boundaries if the special tokens are wrong.
pub(crate) fn format_chat(arch: ModelArch, messages: &[Message]) -> String {
    match arch {
        ModelArch::Llama => format_llama3(messages),
        // Mistral and Mixtral use [INST]/[/INST] markers with system prepended to first user message.
        ModelArch::Mistral | ModelArch::Mixtral => format_mistral(messages),
        // Qwen 2.5, Qwen 3 MoE, Qwen 3.5, and GPT-OSS all use ChatML format.
        ModelArch::Qwen2 | ModelArch::Qwen3Moe | ModelArch::Qwen3_5 | ModelArch::GptOss => format_chatml(messages),
        // Phi uses a ChatML-like format but with <|im_sep|> between role and content.
        ModelArch::Phi => format_phi(messages),
        // Gemma 3 uses <start_of_turn>/<end_of_turn> markers.
        ModelArch::Gemma3 => format_gemma3(messages),
    }
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

    for msg in messages {
        if msg.role == "tool" {
            out.push_str(&tools::format_tool_result_chatml(&msg.content));
            continue;
        }

        // <|im_start|>role\ncontent<|im_end|>\n
        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push('\n');
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
        let role = if msg.role == "assistant" { "model" } else { &msg.role };
        out.push_str(role);
        out.push('\n');
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
        }
    }

    #[test]
    fn test_llama3_system_and_user() {
        let messages = vec![
            msg("system", "You are helpful."),
            msg("user", "Hello"),
        ];
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
        assert!(result.contains("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_chatml_format() {
        let messages = vec![
            msg("system", "You are helpful."),
            msg("user", "Hello"),
        ];
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
        // All Qwen variants produce ChatML format
        assert_eq!(qwen2, qwen3_moe);
        assert_eq!(qwen2, qwen3_5);
    }

    #[test]
    fn test_phi_format() {
        let messages = vec![
            msg("system", "You are helpful."),
            msg("user", "Hello"),
        ];
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
        let messages = vec![
            msg("user", "Hello"),
        ];
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
        let messages = vec![
            msg("system", "You are helpful."),
            msg("user", "Hello"),
        ];
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
        assert_eq!(result, " [INST] Hi [/INST] Hello!</s> [INST] How are you? [/INST]");
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
}
