// ===========================================================================
// Grammar-constrained generation — structured output via token-level masking.
//
// LEARNING OVERVIEW
//
// What this file does:
//   Implements OpenAI's `response_format: {"type": "json_schema"}` by
//   constraining which tokens the sampler can pick at each decode step.
//   Instead of hoping the model produces valid JSON (and validating after),
//   we GUARANTEE it by masking out invalid tokens before sampling.
//
// How it works:
//   1. JSON schema → regex (via outlines-core)
//   2. Regex → DFA (deterministic finite automaton)
//   3. DFA + vocabulary → per-state allowed token sets (precomputed)
//   4. At each decode step: look up current DFA state → get allowed tokens
//      → mask disallowed tokens to -inf → sample from remaining
//   5. After sampling: advance DFA state based on chosen token
//
// Performance:
//   Grammar compilation (steps 1-3) takes 50-500ms depending on schema
//   complexity.  This runs on the async handler thread during request
//   setup, NOT on the GPU worker thread.  The per-token cost (steps 4-5)
//   is O(1) — just a hash lookup and a bitmask application.
//
// outlines-core:
//   We use the `outlines-core` crate (the Rust core of the Outlines library,
//   used by vLLM in production).  It handles the JSON schema → regex → DFA
//   → token mask pipeline.  Key types:
//     - `Vocabulary`: maps token bytes → token IDs
//     - `Index`: precomputed DFA with per-state allowed token sets
//
// Related files:
//   - model/sampler.rs   — applies the token mask before sampling
//   - model/tokenizer.rs — provides vocabulary for grammar compilation
//   - engine/mod.rs      — carries GrammarState per-sequence
//   - api/openai.rs      — parses json_schema from request, compiles grammar
// ===========================================================================

use std::sync::Arc;

use outlines_core::index::Index;
use outlines_core::json_schema;
use outlines_core::vocabulary::Vocabulary;

use super::tokenizer::Tokenizer;

/// Precomputed grammar data for a JSON schema.
///
/// Built once per request during tokenization (on the async handler thread).
/// The `Index` contains per-DFA-state allowed token maps, precomputed from
/// the vocabulary at construction time.  Shared via `Arc` across sequences
/// that use the same schema.
pub(crate) struct CompiledGrammar {
    /// The outlines-core index: maps DFA states to allowed token IDs.
    index: Index,
}

/// Mutable per-sequence grammar tracking state.
///
/// Each sequence with structured output gets its own `GrammarState` that
/// tracks the current position in the DFA.  After each sampled token, the
/// state advances to the next DFA state.
pub(crate) struct GrammarState {
    /// Reference to the precomputed grammar (shared across sequences).
    grammar: Arc<CompiledGrammar>,
    /// Current DFA state ID.
    current_state: u32,
}

impl GrammarState {
    /// Create a new grammar state starting at the DFA's initial state.
    pub fn new(grammar: Arc<CompiledGrammar>) -> Self {
        let initial = grammar.index.initial_state();
        Self {
            grammar,
            current_state: initial,
        }
    }

    /// Get the allowed token IDs for the current DFA state.
    ///
    /// Returns None if the current state has no transitions (should not happen
    /// if the DFA is well-formed and we haven't reached a final state).
    pub fn allowed_tokens(&self) -> Option<Vec<u32>> {
        self.grammar.index.allowed_tokens(&self.current_state)
    }

    /// Advance the DFA state after a token is sampled.
    ///
    /// Returns Ok(()) if the transition is valid, Err if the token is not
    /// allowed in the current state (should not happen if masking was applied).
    pub fn advance(&mut self, token_id: u32) -> anyhow::Result<()> {
        if let Some(next) = self.grammar.index.next_state(&self.current_state, &token_id) {
            self.current_state = next;
            Ok(())
        } else {
            // Token was in the allowed set but maps to a final state with EOS.
            // Check if we're in a final state — EOS tokens return None from
            // next_state but are valid.
            if self.grammar.index.is_final_state(&self.current_state) {
                Ok(())
            } else {
                anyhow::bail!(
                    "grammar: invalid transition from state {} with token {}",
                    self.current_state,
                    token_id
                )
            }
        }
    }

    /// Whether the current state is an accepting (final) state.
    ///
    /// When true, the sequence can stop generating.  The engine checks this
    /// alongside EOS and max_tokens to decide when a sequence is finished.
    pub fn is_complete(&self) -> bool {
        self.grammar.index.is_final_state(&self.current_state)
    }
}

/// Compile a JSON schema into a `CompiledGrammar` using outlines-core.
///
/// This is CPU-intensive (regex compilation + DFA construction + vocabulary scan)
/// and should be called on the handler thread, not the GPU worker thread.
///
/// The `tokenizer` provides the vocabulary mapping (token bytes → token IDs)
/// that outlines-core needs to build per-state token masks.
pub(crate) fn compile_json_schema(
    schema: &serde_json::Value,
    tokenizer: &Tokenizer,
) -> anyhow::Result<Arc<CompiledGrammar>> {
    // Step 1: JSON schema → regex pattern.
    let schema_str = serde_json::to_string(schema)?;
    let regex = json_schema::regex_from_str(&schema_str, None, None)
        .map_err(|e| anyhow::anyhow!("failed to compile JSON schema to regex: {e}"))?;

    // Step 2: Build outlines-core Vocabulary from our tokenizer.
    let (vocab_map, eos_token_id) = tokenizer.get_vocabulary();
    let mut vocabulary = Vocabulary::new(eos_token_id);
    for (token_bytes, token_id) in &vocab_map {
        // Skip EOS token (outlines-core excludes it from normal transitions
        // and adds it explicitly to final states).
        if *token_id == eos_token_id {
            continue;
        }
        vocabulary
            .try_insert(token_bytes.clone(), *token_id)
            .map_err(|e| anyhow::anyhow!("vocabulary insert failed: {e}"))?;
    }

    // Step 3: Build the Index (DFA + per-state token masks).
    let index = Index::new(&regex, &vocabulary)
        .map_err(|e| anyhow::anyhow!("failed to build grammar index: {e}"))?;

    Ok(Arc::new(CompiledGrammar { index }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a minimal vocabulary for testing.
    fn test_vocabulary() -> (Vocabulary, u32) {
        let eos_id = 99;
        let mut vocab = Vocabulary::new(eos_id);
        // Add some basic tokens that can form JSON.
        for (token, id) in [
            ("{", 0),
            ("}", 1),
            ("\"", 2),
            (":", 3),
            (",", 4),
            ("name", 5),
            ("age", 6),
            ("1", 7),
            ("2", 8),
            ("0", 9),
            (" ", 10),
            ("a", 11),
            ("b", 12),
            ("c", 13),
            ("true", 14),
            ("false", 15),
            ("null", 16),
            ("[", 17),
            ("]", 18),
            (".", 19),
            ("-", 20),
        ] {
            vocab.try_insert(token, id).unwrap();
        }
        (vocab, eos_id)
    }

    #[test]
    fn test_integer_regex_index() {
        let (vocab, _eos_id) = test_vocabulary();
        // Simple integer regex.
        let regex = "0|[1-9][0-9]*";
        let index = Index::new(regex, &vocab).unwrap();

        let initial = index.initial_state();
        assert!(!index.is_final_state(&initial));

        let allowed = index.allowed_tokens(&initial).unwrap();
        // Should allow digits that can start an integer.
        assert!(!allowed.is_empty());
    }

    #[test]
    fn test_grammar_state_advance() {
        let (vocab, _eos_id) = test_vocabulary();
        let regex = "0|[1-9][0-9]*";
        let index = Index::new(regex, &vocab).unwrap();
        let grammar = Arc::new(CompiledGrammar { index });

        let mut state = GrammarState::new(grammar);
        let allowed = state.allowed_tokens().unwrap();
        assert!(!allowed.is_empty());

        // Pick any allowed token and advance.
        let token = allowed[0];
        state.advance(token).unwrap();
    }
}
