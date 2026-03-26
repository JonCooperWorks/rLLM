// ===========================================================================
// Extended thinking — parsing and control for chain-of-thought reasoning.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Provides parsing and control for "extended thinking" — a feature where
//   models emit internal reasoning wrapped in `<think>...</think>` tags
//   before producing their visible response.  This is similar to how
//   tool calling works (see tools.rs): we detect special markers in the
//   model's output and structure them for the API response.
//
// How thinking works end-to-end:
//   1. Client sends a request with `thinking: { enabled: true }` (Anthropic)
//      or `thinking: true` (OpenAI).
//   2. The chat template injects a control tag (e.g. `/think` for Qwen) so
//      the model knows to produce reasoning output.
//   3. The model generates `<think>reasoning here</think>actual response`.
//   4. After generation, `parse_thinking()` extracts the thinking block.
//   5. The API returns thinking as a separate content block (Anthropic)
//      or as `reasoning_content` on the message (OpenAI).
//
// Which models support thinking?
//   - Qwen 3 / Qwen 3.5 — trained with `/think` and `/no_think` control
//     tags in the user turn.  Generates `<think>...</think>` blocks.
//   - Other architectures may add support in the future.
//
// Why per-architecture control?
//   Like tool calling, each model was trained on its own thinking format.
//   Qwen uses `/think` in the prompt and `<think>` in the output.  Other
//   models (if they add thinking) may use different markers.  The exhaustive
//   match on ModelArch forces the developer to handle each architecture.
//
// Related files:
//   - tools.rs: same pattern — format markers, parse output, extract blocks
//   - chat.rs: injects thinking control into chat templates
//   - api/openai.rs: returns thinking as reasoning_content
//   - api/anthropic.rs: returns thinking as a content block
// ===========================================================================

use super::config::ModelArch;

// ---------------------------------------------------------------------------
// Thinking result — extracted from model output after generation.
// ---------------------------------------------------------------------------

/// The result of parsing model output for thinking blocks.
///
/// Mirrors the (cleaned_text, tool_calls) pattern from `parse_tool_calls()`.
pub(crate) struct ThinkingResult {
    /// The model's visible response with thinking markers removed.
    pub content: String,
    /// The thinking/reasoning text, if the model produced a `<think>` block.
    pub thinking: Option<String>,
}

// ---------------------------------------------------------------------------
// Architecture support — which models support thinking control?
// ---------------------------------------------------------------------------

/// Whether the given architecture supports thinking mode.
///
/// Models that support thinking were fine-tuned to recognise control tags
/// (e.g. `/think`, `/no_think`) and produce structured thinking output.
/// For unsupported architectures, thinking is a no-op — no tags are injected
/// and no parsing is attempted.
pub(crate) fn supports_thinking(arch: ModelArch) -> bool {
    match arch {
        // Qwen 3 MoE and Qwen 3.5 were trained with /think and /no_think tags.
        ModelArch::Qwen3Moe | ModelArch::Qwen3_5 => true,
        // Other architectures do not currently support thinking.
        ModelArch::Llama
        | ModelArch::Qwen2
        | ModelArch::Mistral
        | ModelArch::Mixtral
        | ModelArch::Phi
        | ModelArch::Gemma3
        | ModelArch::GptOss => false,
    }
}

// ---------------------------------------------------------------------------
// Thinking control tag — injected into the generation prompt.
//
// This is analogous to `format_tool_system_prompt()` in tools.rs — it
// modifies the prompt so the model knows whether to produce thinking output.
// ---------------------------------------------------------------------------

/// Return the thinking control tag to append to the generation prompt.
///
/// When thinking is enabled, models that support it get a control tag
/// appended after the final `<|im_start|>assistant\n` prompt.  For Qwen,
/// this is `<think>\n` which tells the model to begin its reasoning block.
///
/// When thinking is disabled (or the model doesn't support it), returns None.
pub(crate) fn thinking_prompt_tag(arch: ModelArch, enabled: bool) -> Option<&'static str> {
    if !enabled || !supports_thinking(arch) {
        return None;
    }

    match arch {
        // Qwen 3 / 3.5: appending `<think>\n` after the assistant prompt
        // causes the model to emit its reasoning inside `<think>...</think>`.
        ModelArch::Qwen3Moe | ModelArch::Qwen3_5 => Some("<think>\n"),
        _ => None,
    }
}

/// Return the thinking suppression tag to append to the generation prompt.
///
/// Qwen 3 / 3.5 models are trained to always produce `<think>` blocks.
/// Simply omitting the `<think>` prompt tag is NOT sufficient — the model
/// generates `<think>` on its own.  To suppress thinking, we inject a
/// closed `<think>\n</think>\n` block so the model sees thinking as
/// "already done" and proceeds directly to the visible response.
pub(crate) fn thinking_suppress_tag(arch: ModelArch) -> Option<&'static str> {
    match arch {
        ModelArch::Qwen3Moe | ModelArch::Qwen3_5 => Some("<think>\n</think>\n"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Output parsing — extract thinking blocks from generated text.
//
// This mirrors `parse_tool_calls()` in tools.rs.  Each architecture has
// its own parser, and the top-level function dispatches via exhaustive match.
// ---------------------------------------------------------------------------

/// Parse model output for thinking blocks.
///
/// Returns a `ThinkingResult` containing the cleaned content (thinking tags
/// removed) and any extracted thinking text.
///
/// Like `parse_tool_calls()`, this uses an exhaustive match on `ModelArch`
/// to force handling each architecture.
pub(crate) fn parse_thinking(arch: ModelArch, text: &str) -> ThinkingResult {
    match arch {
        // Qwen 3 / 3.5 produce <think>...</think> blocks.
        ModelArch::Qwen3Moe | ModelArch::Qwen3_5 => parse_thinking_xml(text),
        // Architectures without thinking support pass through unchanged.
        ModelArch::Llama
        | ModelArch::Qwen2
        | ModelArch::Mistral
        | ModelArch::Mixtral
        | ModelArch::Phi
        | ModelArch::Gemma3
        | ModelArch::GptOss => ThinkingResult {
            content: text.to_string(),
            thinking: None,
        },
    }
}

/// Parse `<think>...</think>` XML markers from model output.
///
/// Qwen 3 / 3.5 models wrap their reasoning in `<think>` tags at the start
/// of the response.  Everything before `</think>` is thinking; everything
/// after is the visible response.
///
/// Handles edge cases:
///   - No `<think>` tag → no thinking, full text is content
///   - `<think>` with no `</think>` → entire text is treated as thinking
///     (model was likely cut off by max_tokens)
///   - Empty thinking block `<think>\n</think>` → no thinking content
///   - `</think>` without opening `<think>` → the opening `<think>` was
///     already injected as a prompt tag, so everything before `</think>`
///     is thinking (common with Qwen 3.5 where `<think>` is a special
///     token consumed by the prompt and the model starts generating
///     thinking content directly).
fn parse_thinking_xml(text: &str) -> ThinkingResult {
    // First, try to find a </think> closing tag.
    if let Some(end_pos) = text.find("</think>") {
        // Check if there's an opening <think> tag before it.
        let think_start = text[..end_pos].find("<think>");

        let thinking_text = if let Some(start) = think_start {
            // Both tags present: extract text between them.
            text[start + "<think>".len()..end_pos].trim()
        } else {
            // Only </think> found — the opening <think> was injected as a
            // prompt tag (not in the model output).  Everything before
            // </think> is thinking content.
            text[..end_pos].trim()
        };

        let content = text[end_pos + "</think>".len()..].trim();

        ThinkingResult {
            content: content.to_string(),
            thinking: if thinking_text.is_empty() {
                None
            } else {
                Some(thinking_text.to_string())
            },
        }
    } else if let Some(think_start) = text.find("<think>") {
        // Opening <think> with no closing tag — model was cut off.
        let thinking_text = text[think_start + "<think>".len()..].trim();

        ThinkingResult {
            content: String::new(),
            thinking: if thinking_text.is_empty() {
                None
            } else {
                Some(thinking_text.to_string())
            },
        }
    } else {
        // No thinking markers at all.
        ThinkingResult {
            content: text.to_string(),
            thinking: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- supports_thinking --

    #[test]
    fn test_qwen3_supports_thinking() {
        assert!(supports_thinking(ModelArch::Qwen3Moe));
        assert!(supports_thinking(ModelArch::Qwen3_5));
    }

    #[test]
    fn test_other_archs_no_thinking() {
        for arch in [
            ModelArch::Llama,
            ModelArch::Qwen2,
            ModelArch::Mistral,
            ModelArch::Mixtral,
            ModelArch::Phi,
            ModelArch::Gemma3,
            ModelArch::GptOss,
        ] {
            assert!(
                !supports_thinking(arch),
                "{arch:?} should not support thinking"
            );
        }
    }

    // -- thinking_prompt_tag --

    #[test]
    fn test_thinking_prompt_tag_enabled() {
        assert_eq!(
            thinking_prompt_tag(ModelArch::Qwen3_5, true),
            Some("<think>\n")
        );
        assert_eq!(
            thinking_prompt_tag(ModelArch::Qwen3Moe, true),
            Some("<think>\n")
        );
    }

    #[test]
    fn test_thinking_prompt_tag_disabled() {
        assert_eq!(thinking_prompt_tag(ModelArch::Qwen3_5, false), None);
    }

    #[test]
    fn test_thinking_prompt_tag_unsupported_arch() {
        assert_eq!(thinking_prompt_tag(ModelArch::Llama, true), None);
        assert_eq!(thinking_prompt_tag(ModelArch::Qwen2, true), None);
    }

    // -- thinking_suppress_tag --

    #[test]
    fn test_thinking_suppress_tag() {
        // No suppression tags needed — omitting the <think> prompt tag
        // is sufficient to prevent thinking output.
        assert_eq!(thinking_suppress_tag(ModelArch::Qwen3_5), None);
        assert_eq!(thinking_suppress_tag(ModelArch::Llama), None);
    }

    // -- parse_thinking --

    #[test]
    fn test_parse_thinking_with_block() {
        let text = "<think>\nLet me reason about this.\nThe answer is 42.\n</think>\nThe answer is 42.";
        let result = parse_thinking(ModelArch::Qwen3_5, text);
        assert_eq!(result.content, "The answer is 42.");
        assert_eq!(
            result.thinking.as_deref(),
            Some("Let me reason about this.\nThe answer is 42.")
        );
    }

    #[test]
    fn test_parse_thinking_no_block() {
        let text = "The answer is 42.";
        let result = parse_thinking(ModelArch::Qwen3_5, text);
        assert_eq!(result.content, "The answer is 42.");
        assert!(result.thinking.is_none());
    }

    #[test]
    fn test_parse_thinking_empty_block() {
        let text = "<think>\n</think>\nThe answer is 42.";
        let result = parse_thinking(ModelArch::Qwen3_5, text);
        assert_eq!(result.content, "The answer is 42.");
        assert!(result.thinking.is_none());
    }

    #[test]
    fn test_parse_thinking_unclosed_tag() {
        // Model was cut off by max_tokens before closing the thinking block.
        let text = "<think>\nLet me think about this...";
        let result = parse_thinking(ModelArch::Qwen3_5, text);
        assert_eq!(result.content, "");
        assert_eq!(
            result.thinking.as_deref(),
            Some("Let me think about this...")
        );
    }

    #[test]
    fn test_parse_thinking_unsupported_arch_passthrough() {
        // Even if text contains <think> tags, unsupported archs pass through.
        let text = "<think>reasoning</think>response";
        let result = parse_thinking(ModelArch::Llama, text);
        assert_eq!(result.content, text);
        assert!(result.thinking.is_none());
    }

    #[test]
    fn test_parse_thinking_multiline() {
        let text = "<think>\nStep 1: Read the question.\nStep 2: Consider the context.\nStep 3: Formulate the answer.\n</think>\nHere is my response.";
        let result = parse_thinking(ModelArch::Qwen3Moe, text);
        assert!(result.thinking.as_ref().unwrap().contains("Step 1"));
        assert!(result.thinking.as_ref().unwrap().contains("Step 3"));
        assert_eq!(result.content, "Here is my response.");
    }

    #[test]
    fn test_parse_thinking_with_leading_content() {
        // Some models might emit text before the <think> tag.
        let text = "Sure! <think>\nreasoning\n</think>\nAnswer.";
        let result = parse_thinking(ModelArch::Qwen3_5, text);
        assert_eq!(result.content, "Answer.");
        assert_eq!(result.thinking.as_deref(), Some("reasoning"));
    }

    #[test]
    fn test_parse_thinking_close_only() {
        // When thinking is prompted via `<think>\n` in the generation prompt,
        // the model output starts with thinking content directly (no opening
        // <think> tag) and ends with </think> before the visible response.
        let text = "Let me reason.\nThe answer is 4.\n</think>\n\n2+2 equals **4**.";
        let result = parse_thinking(ModelArch::Qwen3_5, text);
        assert_eq!(
            result.thinking.as_deref(),
            Some("Let me reason.\nThe answer is 4.")
        );
        assert_eq!(result.content, "2+2 equals **4**.");
    }
}
