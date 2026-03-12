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
//   Llama 3.2 uses a vocabulary of 128256 tokens, including:
//     - ~128000 regular text tokens (subwords, characters, bytes)
//     - ~256 special tokens (BOS, EOS, tool markers, etc.)
//
// Special tokens:
//   BOS (Beginning of Sequence, ID 128000):
//     Prepended to every prompt.  Tells the model "this is the start of a
//     new sequence".  Without BOS, the model would treat the first token as
//     a continuation of some unknown previous context.
//
//   EOS (End of Sequence, ID 128001):
//     The model generates this when it considers the sequence complete.
//     We stop generation when we see this token.
//
//   EOT (End of Turn, ID 128009):
//     Used in chat-formatted sequences to mark the end of one speaker's turn.
//     We also treat this as a stop signal for the base model.
//
// Why wrap the HF tokenizer?
//   The `tokenizers` crate provides the core BPE implementation.  Our wrapper
//   adds: (1) automatic BOS prepending, (2) EOS detection for generation
//   stopping.  This keeps tokenizer details out of main.rs.
// ===========================================================================

use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

pub(crate) struct Tokenizer {
    /// The HuggingFace tokenizer (BPE model + merge rules + vocabulary).
    inner: HfTokenizer,
    /// Token IDs that signal the model wants to stop generating.
    eos_token_ids: Vec<u32>,
}

// Llama 3.x special token IDs (defined in the tokenizer config).
const BOS_TOKEN_ID: u32 = 128000; // <|begin_of_text|>
const EOS_TOKEN_ID: u32 = 128001; // <|end_of_text|>
const EOT_TOKEN_ID: u32 = 128009; // <|eot_id|>

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    ///
    /// The tokenizer.json contains the full BPE model: vocabulary, merge rules,
    /// pre/post-processing steps, and special token definitions.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        Ok(Self {
            inner,
            eos_token_ids: vec![EOS_TOKEN_ID, EOT_TOKEN_ID],
        })
    }

    /// Encode text into token IDs, prepending the BOS token.
    ///
    /// Example: "Hello" → [128000, 9906]  (BOS + "Hello")
    ///
    /// Learning note: the `false` argument to `encode()` disables the
    /// tokenizer's built-in special-token handling — we prepend BOS manually
    /// because the base model expects exactly one BOS at the start.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let mut ids = vec![BOS_TOKEN_ID];
        ids.extend_from_slice(encoding.get_ids());
        Ok(ids)
    }

    /// Encode a chat-template-formatted string into token IDs.
    ///
    /// Unlike `encode()`, this does NOT prepend BOS — the chat template string
    /// already starts with `<|begin_of_text|>`.  The `true` flag tells the HF
    /// tokenizer to parse special token syntax (e.g. `<|start_header_id|>`)
    /// into their actual token IDs instead of treating them as literal text.
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

    /// Check if a token ID is an end-of-sequence signal.
    /// Used to stop the generation loop.
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }
}
