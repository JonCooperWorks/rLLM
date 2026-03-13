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

            // Qwen 2.5: no BOS token.  Stop on <|endoftext|> or <|im_end|>.
            //
            // Learning note: unlike Llama, Qwen doesn't prepend a BOS token.
            // The model was trained without one — adding BOS would shift all
            // positions by 1 and degrade output quality.  In chat mode,
            // <|im_end|> (151645) is the primary stop token.
            ModelArch::Qwen2 => (None, vec![151643, 151645]),
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
                    },
                    chat::Message {
                        role: "user".into(),
                        content: prompt.to_string(),
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
    /// (system/user/assistant roles from the HTTP request).
    pub fn encode_messages(
        &self,
        messages: &[chat::Message],
        arch: ModelArch,
    ) -> anyhow::Result<Vec<u32>> {
        let formatted = chat::format_chat(arch, messages);
        self.encode_chat(&formatted)
    }

    /// Check if a token ID is an end-of-sequence signal.
    /// Used to stop the generation loop.
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }
}
