// ===========================================================================
// Tokenizer — text ↔ token ID conversion using HuggingFace tokenizers.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Converts between human-readable text and the integer token IDs that the
//   model operates on.  "The capital" → [791, 6864] → model → [315] → "of".
//
// Tokenisation (encoding):
//   The tokenizer splits text into subword pieces using Byte-Pair Encoding
//   (BPE).  Common words like "the" get a single token, while rare words are
//   split into multiple pieces: "unfamiliar" → ["un", "familiar"].  This
//   gives a good balance between vocabulary size and sequence length.
//
// Model-specific special tokens:
//   Different model families use different vocabularies and special token IDs.
//   The tokenizer is configured per-architecture at construction time:
//
//   Llama 3.x (vocab_size=128256):
//     BOS = 128000  <|begin_of_text|>   — prepended to every sequence
//     EOS = 128001  <|end_of_text|>     — signals sequence completion
//     EOT = 128009  <|eot_id|>          — end of turn (chat mode)
//
//   Qwen 2.5 (vocab_size=152064):
//     BOS = 151643  <|endoftext|>       — both BOS and EOS (GPT convention)
//     EOT = 151645  <|im_end|>          — end of turn (ChatML mode)
//
// Why wrap the HF tokenizer?
//   The `tokenizers` crate provides the core BPE implementation.  Our wrapper
//   adds: (1) automatic BOS prepending, (2) EOS detection for generation
//   stopping, (3) model-specific special token configuration.  This keeps
//   tokenizer details out of main.rs.
// ===========================================================================

use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

use super::chat;
use super::config::ModelArch;

#[derive(Clone)]
pub(crate) struct Tokenizer {
    /// The HuggingFace tokenizer (BPE model + merge rules + vocabulary).
    inner: HfTokenizer,
    /// Token IDs that signal the model wants to stop generating.
    eos_token_ids: Vec<u32>,
    /// Beginning-of-sequence token ID, prepended to every prompt.
    /// None for models that don't use BOS (e.g. Qwen 2.5).
    bos_token_id: Option<u32>,
}

impl Tokenizer {
    /// Load a tokenizer and configure special tokens for the model architecture.
    ///
    /// Different model families use different special token IDs because they
    /// have different vocabularies.  Llama 3 has 128256 tokens; Qwen 2.5 has
    /// 152064.  Using the wrong BOS/EOS would produce garbage or fail to stop.
    pub fn from_file(path: &Path, arch: ModelArch) -> anyhow::Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        let (bos_token_id, eos_token_ids) = match arch {
            // Llama 3.x: BOS=128000, stop on EOS (128001) or EOT (128009).
            ModelArch::Llama => (Some(128000), vec![128001, 128009]),

            // Mistral 7B / Mixtral 8x7B: BOS=1, EOS=2 (SentencePiece 32K vocab).
            ModelArch::Mistral | ModelArch::Mixtral => (Some(1), vec![2]),

            // Qwen 2.5: no BOS token.  Stop on <|endoftext|> or <|im_end|>.
            //
            // Learning note: unlike Llama, Qwen doesn't prepend a BOS token.
            // The model was trained without one — adding BOS would shift all
            // positions by 1 and degrade output quality.  In chat mode,
            // <|im_end|> (151645) is the primary stop token.
            ModelArch::Qwen2 => (None, vec![151643, 151645]),

            // Qwen 3 MoE: same convention as Qwen 2.5 — no BOS, ChatML stops.
            // Vocab size differs (151936 vs 152064) but special token IDs are
            // in the same range.
            ModelArch::Qwen3Moe => (None, vec![151643, 151645]),

            // Qwen 3.5: no BOS.  Stop on <|endoftext|> (248044) or <|im_end|>
            // (248046).  ChatML format uses <|im_end|> to terminate assistant
            // turns — without it the model loops through repeated <think> blocks.
            // Vocab size is 248320.
            ModelArch::Qwen3_5 => (None, vec![248044, 248046]),

            // Phi (Microsoft): no BOS.  Stop on <|im_end|> (100265) or
            // <|endoftext|> (100257).  Uses tiktoken-based 100352-token vocab.
            ModelArch::Phi => (None, vec![100257, 100265]),

            // Gemma 3: BOS=2 (<bos>).  Stop on <end_of_turn> (106) or <eos> (1).
            // Gemma uses a 262144-token SentencePiece vocabulary.
            ModelArch::Gemma3 => (Some(2), vec![1, 106]),

            // GPT-OSS-20B: no BOS.  eos_token_id=200002 from config.json.
            // Uses a 201088-token vocabulary.
            ModelArch::GptOss => (None, vec![200002]),

            // Nemotron-H: eos_token_id=2 (</s>) from config.json.  No BOS.
            // The HF tokenizer_config.json sets eos_token to <|im_end|>
            // (token 11) — generation must stop there or it runs past
            // the assistant turn boundary.
            ModelArch::NemotronH => (None, vec![2, 11]),
        };

        Ok(Self {
            inner,
            eos_token_ids,
            bos_token_id,
        })
    }

    /// Encode text into token IDs, prepending BOS if the model uses one.
    ///
    /// Example (Llama): "Hello" → [128000, 9906]  (BOS + "Hello")
    /// Example (Qwen):  "Hello" → [9707]          (no BOS)
    ///
    /// Learning note: the `false` argument to `encode()` disables the
    /// tokenizer's built-in special-token handling — we prepend BOS manually
    /// (when needed) to ensure exactly one BOS at the start.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let mut ids = Vec::new();
        if let Some(bos) = self.bos_token_id {
            ids.push(bos);
        }
        ids.extend_from_slice(encoding.get_ids());
        Ok(ids)
    }

    /// Encode a chat-template-formatted string into token IDs.
    ///
    /// Unlike `encode()`, this does NOT prepend BOS — the HF tokenizer's
    /// `encode(text, add_special_tokens=true)` handles that automatically.
    /// The `true` flag also tells the tokenizer to parse special token syntax
    /// (e.g. `<|start_header_id|>`, `<|im_start|>`) into their actual token
    /// IDs instead of treating them as literal text.
    pub fn encode_chat(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Number of tokens in the vocabulary (including added/special tokens).
    ///
    /// This may be smaller than the model's embedding dimension — some models
    /// pad the embedding matrix to a multiple of 64/128 for GEMM efficiency.
    /// The sampler uses this to clamp logits and avoid sampling padding tokens.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Decode token IDs back into text.
    ///
    /// The `true` argument to `decode()` skips special tokens in the output
    /// (so we don't print "<|begin_of_text|>" literally).
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))
    }

    /// Encode a user prompt, optionally wrapping it in a chat template.
    ///
    /// In chat mode (`system` is Some), builds a [system, user] message list,
    /// formats it with the model's chat template, and encodes with special
    /// tokens.  In raw mode (`system` is None), encodes the prompt directly.
    pub fn encode_prompt(
        &self,
        prompt: &str,
        arch: ModelArch,
        system: Option<&str>,
    ) -> anyhow::Result<Vec<u32>> {
        match system {
            Some(sys) => {
                let messages = vec![
                    chat::Message {
                        role: "system".into(),
                        content: sys.to_string(),
                        tool_calls: None,
                        tool_call_id: None,
                        images: None,
                    },
                    chat::Message {
                        role: "user".into(),
                        content: prompt.to_string(),
                        tool_calls: None,
                        tool_call_id: None,
                        images: None,
                    },
                ];
                let formatted = chat::format_chat(arch, &messages);
                self.encode_chat(&formatted)
            }
            None => self.encode(prompt),
        }
    }

    /// Encode pre-structured chat messages into token IDs.
    ///
    /// Used by the API server where messages arrive already structured
    /// (system/user/assistant roles from the HTTP request).  Delegates to
    /// `encode_messages_with_thinking()` with no thinking control.
    pub fn encode_messages(
        &self,
        messages: &[chat::Message],
        arch: ModelArch,
    ) -> anyhow::Result<Vec<u32>> {
        self.encode_messages_with_thinking(messages, arch, None)
    }

    /// Encode chat messages with optional thinking control.
    ///
    /// Like `encode_messages()` but passes a thinking flag to the chat
    /// template so the model knows whether to produce reasoning output.
    /// See `chat::format_chat_with_thinking()` for details.
    pub fn encode_messages_with_thinking(
        &self,
        messages: &[chat::Message],
        arch: ModelArch,
        thinking: Option<bool>,
    ) -> anyhow::Result<Vec<u32>> {
        let formatted = chat::format_chat_with_thinking(arch, messages, thinking);
        self.encode_chat(&formatted)
    }

    /// Create a minimal tokenizer for testing.
    ///
    /// Uses an empty BPE model (no real vocabulary) — only `is_eos()` works
    /// correctly.  `decode()` returns empty strings, which is fine for engine
    /// step tests that only care about token IDs and sequence lifecycle.
    #[cfg(test)]
    pub fn for_test(eos_token_ids: Vec<u32>) -> Self {
        use tokenizers::models::bpe::BPE;
        let inner = HfTokenizer::new(BPE::default());
        Self {
            inner,
            eos_token_ids,
            bos_token_id: None,
        }
    }

    /// Check if a token ID is an end-of-sequence signal.
    /// Used to stop the generation loop.
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    /// Check if a token ID is a thinking start marker (`<think>`).
    ///
    /// Models like Qwen 3.5 emit `<think>` as a special token (ID 248068)
    /// that gets stripped by `decode(skip_special_tokens=true)`.  The worker
    /// loop uses this to inject the literal `<think>` text back into the
    /// decoded output so the text-level thinking parser can find it.
    pub fn is_think_start(&self, token_id: u32) -> bool {
        // Qwen 3 / 3.5: <think> = 248068
        token_id == 248068
    }

    /// Check if a token ID is a thinking end marker (`</think>`).
    ///
    /// See `is_think_start()` for context.  `</think>` = 248069 in Qwen 3.5.
    pub fn is_think_end(&self, token_id: u32) -> bool {
        // Qwen 3 / 3.5: </think> = 248069
        token_id == 248069
    }

    /// Decode tokens incrementally, simulating the streaming API path.
    ///
    /// Decodes the full `ids` buffer and returns only the text after
    /// `prev_text_len` characters.  This is the pattern used by the API
    /// server to avoid SentencePiece Strip decoder issues.
    #[cfg(test)]
    fn decode_incremental(&self, ids: &[u32], prev_text_len: usize) -> (String, usize) {
        let full = self.decode(ids).unwrap();
        let new_text = full[prev_text_len..].to_string();
        (new_text, full.len())
    }
}

// ===========================================================================
// Tests — EOS token validation and incremental decode round-trip.
//
// These tests load actual model tokenizer files from disk to verify:
//   1. Hardcoded EOS token IDs decode to the expected special token strings
//   2. Incremental (token-by-token) decode preserves spaces correctly
//
// The incremental decode test catches the Mistral space-stripping bug:
// SentencePiece's Strip decoder removes leading spaces when decoding
// individual tokens, so we must decode the full buffer and diff.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Helper: load a tokenizer if the model directory exists on disk.
    fn load_if_exists(model_dir: &str, arch: ModelArch) -> Option<Tokenizer> {
        let path = Path::new("models").join(model_dir).join("tokenizer.json");
        if path.exists() {
            Some(Tokenizer::from_file(&path, arch).unwrap())
        } else {
            None
        }
    }

    // -----------------------------------------------------------------------
    // EOS token validation: verify hardcoded IDs match actual special tokens.
    //
    // For each model, encode the special token string (e.g. "<end_of_turn>")
    // and check that the resulting token ID is in our EOS list.  This catches
    // off-by-one errors like the Gemma 107→106 bug.
    // -----------------------------------------------------------------------

    #[test]
    fn test_eos_tokens_gemma3() {
        let Some(tok) = load_if_exists("gemma-3-4b-it", ModelArch::Gemma3) else {
            return;
        };
        // <end_of_turn> should be token 106, and it should be EOS.
        let ids = tok.inner.encode("<end_of_turn>", true).unwrap();
        let eot_id = ids.get_ids().iter().find(|&&id| id != 2); // skip BOS=2
        assert_eq!(
            eot_id,
            Some(&106),
            "Gemma <end_of_turn> should be token 106"
        );
        assert!(tok.is_eos(106), "Gemma token 106 should be EOS");
        assert!(tok.is_eos(1), "Gemma token 1 (<eos>) should be EOS");
        assert!(!tok.is_eos(107), "Gemma token 107 should NOT be EOS");
    }

    #[test]
    fn test_eos_tokens_llama() {
        let Some(tok) = load_if_exists("llama-3.2-1b-instruct", ModelArch::Llama) else {
            return;
        };
        let ids = tok.inner.encode("<|eot_id|>", true).unwrap();
        assert!(
            ids.get_ids().contains(&128009),
            "Llama <|eot_id|> should be 128009"
        );
        assert!(tok.is_eos(128001), "Llama EOS token");
        assert!(tok.is_eos(128009), "Llama EOT token");
    }

    #[test]
    fn test_eos_tokens_mistral() {
        let Some(tok) = load_if_exists("mistral-7b-instruct", ModelArch::Mistral) else {
            return;
        };
        assert!(tok.is_eos(2), "Mistral EOS=2");
        // Token 2 should decode to </s>.
        let text = tok.decode(&[2]).unwrap();
        assert!(
            text.is_empty() || text == "</s>",
            "Mistral EOS decode: {text:?}"
        );
    }

    #[test]
    fn test_eos_tokens_qwen2() {
        let Some(tok) = load_if_exists("qwen-2.5-3b-instruct", ModelArch::Qwen2) else {
            return;
        };
        let ids = tok.inner.encode("<|im_end|>", true).unwrap();
        assert!(
            ids.get_ids().contains(&151645),
            "Qwen <|im_end|> should be 151645"
        );
        assert!(tok.is_eos(151643), "Qwen endoftext");
        assert!(tok.is_eos(151645), "Qwen im_end");
    }

    #[test]
    fn test_eos_tokens_qwen3_5() {
        let Some(tok) = load_if_exists("qwen3.5-27b", ModelArch::Qwen3_5) else {
            return;
        };
        let ids = tok.inner.encode("<|im_end|>", true).unwrap();
        assert!(
            ids.get_ids().contains(&248046),
            "Qwen 3.5 <|im_end|> should be 248046"
        );
        assert!(tok.is_eos(248044), "Qwen 3.5 endoftext");
        assert!(tok.is_eos(248046), "Qwen 3.5 im_end");
    }

    #[test]
    fn test_eos_tokens_phi() {
        let Some(tok) = load_if_exists("phi-4", ModelArch::Phi) else {
            return;
        };
        assert!(tok.is_eos(100257), "Phi endoftext");
        assert!(tok.is_eos(100265), "Phi im_end");
    }

    // -----------------------------------------------------------------------
    // Incremental decode: simulate the API streaming path.
    //
    // Encode a sentence, then decode one token at a time using the
    // incremental pattern (full-buffer decode, emit new chars).  The
    // concatenated output must match a single-shot decode of all tokens.
    //
    // This catches the Mistral space-stripping bug where single-token
    // decode(&[id]) strips leading spaces due to SentencePiece's Strip
    // decoder step.
    // -----------------------------------------------------------------------

    fn check_incremental_decode(tok: &Tokenizer, text: &str) {
        let ids = tok.inner.encode(text, false).unwrap();
        let token_ids = ids.get_ids();

        // Single-shot decode (ground truth).
        let expected = tok.decode(token_ids).unwrap();

        // Incremental decode: one token at a time, like the API server.
        let mut buf: Vec<u32> = Vec::new();
        let mut prev_len = 0;
        let mut reconstructed = String::new();
        for &id in token_ids {
            buf.push(id);
            let (new_text, new_len) = tok.decode_incremental(&buf, prev_len);
            reconstructed.push_str(&new_text);
            prev_len = new_len;
        }

        assert_eq!(
            reconstructed, expected,
            "incremental decode mismatch for {text:?}"
        );
    }

    #[test]
    fn test_incremental_decode_mistral() {
        let Some(tok) = load_if_exists("mistral-7b-instruct", ModelArch::Mistral) else {
            return;
        };
        // These strings have spaces that SentencePiece Strip would eat.
        check_incremental_decode(&tok, "for number in range(1, 21):");
        check_incremental_decode(&tok, "def hello_world():\n    print(\"Hello World\")");
        check_incremental_decode(&tok, "The quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn test_incremental_decode_llama() {
        let Some(tok) = load_if_exists("llama-3.2-1b-instruct", ModelArch::Llama) else {
            return;
        };
        check_incremental_decode(&tok, "for i in range(1, 21):");
        check_incremental_decode(&tok, "The quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn test_incremental_decode_gemma() {
        let Some(tok) = load_if_exists("gemma-3-4b-it", ModelArch::Gemma3) else {
            return;
        };
        check_incremental_decode(&tok, "for i in range(1, 21):");
        check_incremental_decode(&tok, "The quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn test_incremental_decode_qwen() {
        let Some(tok) = load_if_exists("qwen-2.5-3b-instruct", ModelArch::Qwen2) else {
            return;
        };
        check_incremental_decode(&tok, "for i in range(1, 21):");
        check_incremental_decode(&tok, "The quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn test_incremental_decode_phi() {
        let Some(tok) = load_if_exists("phi-4", ModelArch::Phi) else {
            return;
        };
        check_incremental_decode(&tok, "for i in range(1, 21):");
        check_incremental_decode(&tok, "The quick brown fox jumps over the lazy dog");
    }
}
