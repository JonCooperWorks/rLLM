// ===========================================================================
// Chat template — formatting messages for instruct-tuned models.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Converts a list of chat messages (system, user, assistant) into the
//   special-token-delimited string format that instruct models expect.
//   Supports both Llama 3 and Qwen 2.5 (ChatML) formats.
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

/// A single message in a chat conversation.
///
/// Roles follow the OpenAI convention used by most LLM APIs:
///   - "system":    instructions for the model's behaviour
///   - "user":      the human's message
///   - "assistant": the model's response (for multi-turn conversations)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Message {
    pub role: String,
    pub content: String,
}

/// Format messages using the correct chat template for the model architecture.
///
/// Each model family was fine-tuned on its own chat format.  Using the wrong
/// template produces gibberish — the model literally doesn't understand the
/// role boundaries if the special tokens are wrong.
pub(crate) fn format_chat(arch: ModelArch, messages: &[Message]) -> String {
    match arch {
        ModelArch::Llama => format_llama3(messages),
        // Qwen 2.5, Qwen 3 MoE, and Qwen 3.5 all use ChatML format.
        ModelArch::Qwen2 | ModelArch::Qwen3Moe | ModelArch::Qwen3_5 => format_chatml(messages),
        // Phi uses a ChatML-like format but with <|im_sep|> between role and content.
        ModelArch::Phi => format_phi(messages),
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
