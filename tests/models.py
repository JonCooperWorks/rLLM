# ---------------------------------------------------------------------------
# models.py — unified model registry for rLLM testing and benchmarking.
#
# Single source of truth for all model configurations.  Both pytest tests
# (test_model_families.py) and benchmark mode (--bench) read from here.
# Adding a new model family requires only one edit: add it to MODEL_REGISTRY.
#
# Related: conftest.py (server management), quality.py (output validation),
#          benchmark.py (measurement engine)
# ---------------------------------------------------------------------------

import json
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Describes one model configuration for testing and benchmarking.

    Fields used by both test and bench modes:
      test_id          — short identifier for pytest parametrize IDs
      model_name       — directory name under models/ (without quant suffix)
      family           — architecture family name
      is_moe           — requires --stream-experts decision
      bf16_size_gb     — estimated bf16 weight size for memory planning
      extra_args       — additional CLI args (e.g., --kv-quant)
      supports_stream_experts — some MoE models don't support it yet
      temperature      — sampling temperature (0 = greedy)
      has_vision       — model supports image input

    Fields used primarily by bench mode:
      is_chat          — None = auto-detect from name, True/False = override
      quality_sensitive — known to be fragile (quality may drop, not a bench failure)
      in_test_suite    — True = included in curated pytest test list
    """

    test_id: str
    model_name: str
    family: str
    is_moe: bool = False
    bf16_size_gb: float = 0
    extra_args: tuple[str, ...] = ()
    supports_stream_experts: bool = True
    temperature: float = 0
    has_vision: bool = False
    is_chat: bool | None = None
    quality_sensitive: bool = False
    in_test_suite: bool = False


# ---------------------------------------------------------------------------
# Model registry — every model rLLM knows about, in one place.
#
# Models with in_test_suite=True are the curated per-family test models.
# All others are bench-only (discovered when scanning the models/ directory).
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelConfig] = {
    # ---- Curated test suite (one per architecture family, smallest available) ----
    "llama-3.2-3b-instruct": ModelConfig(
        "llama-3b", "llama-3.2-3b-instruct", "Llama",
        bf16_size_gb=6, in_test_suite=True,
    ),
    "qwen2.5-3b-instruct": ModelConfig(
        "qwen2-3b", "qwen2.5-3b-instruct", "Qwen2",
        bf16_size_gb=6, in_test_suite=True,
    ),
    "gemma-3-4b-it": ModelConfig(
        "gemma3-4b", "gemma-3-4b-it", "Gemma3",
        bf16_size_gb=8, has_vision=True, in_test_suite=True,
    ),
    "mistral-7b-instruct-v0.3": ModelConfig(
        "mistral-7b", "mistral-7b-instruct-v0.3", "Mistral",
        bf16_size_gb=14, in_test_suite=True,
    ),
    "qwen3.5-9b": ModelConfig(
        "qwen3.5-9b", "qwen3.5-9b", "Qwen3_5",
        bf16_size_gb=18, has_vision=True, is_chat=True, in_test_suite=True,
    ),
    "phi-4": ModelConfig(
        "phi-4", "phi-4", "Phi",
        bf16_size_gb=28, is_chat=True, in_test_suite=True,
    ),
    "mixtral-8x7b-instruct-v0.1": ModelConfig(
        "mixtral-8x7b", "mixtral-8x7b-instruct-v0.1", "Mixtral",
        is_moe=True, bf16_size_gb=93, in_test_suite=True,
    ),
    "qwen3-coder-30b-a3b-instruct": ModelConfig(
        "qwen3moe-30b", "qwen3-coder-30b-a3b-instruct", "Qwen3Moe",
        is_moe=True, bf16_size_gb=60, in_test_suite=True,
    ),
    "qwen3.5-35b-a3b": ModelConfig(
        "qwen3.5-35b", "qwen3.5-35b-a3b", "Qwen3_5M",
        is_moe=True, bf16_size_gb=70, has_vision=True, is_chat=True, in_test_suite=True,
    ),
    "gpt-oss-20b": ModelConfig(
        "gpt-oss-20b", "gpt-oss-20b", "GptOss",
        is_moe=True, bf16_size_gb=40, supports_stream_experts=False,
        temperature=0.6, is_chat=True, quality_sensitive=True, in_test_suite=True,
    ),
    "nemotron-3-30b": ModelConfig(
        "nemotron-h-30b", "nemotron-3-30b", "NemotronH",
        is_moe=True, bf16_size_gb=63, supports_stream_experts=True,
        is_chat=True, in_test_suite=True,
    ),
    "nemotron-3-120b": ModelConfig(
        "nemotron-h-120b", "nemotron-3-120b", "NemotronH",
        is_moe=True, bf16_size_gb=240, is_chat=True, in_test_suite=True,
    ),

    # ---- Bench-only models (not in curated test suite) -----------------------
    "llama-3.2-3b": ModelConfig(
        "llama-3b-base", "llama-3.2-3b", "Llama", bf16_size_gb=6,
    ),
    "qwen2.5-7b-instruct": ModelConfig(
        "qwen2-7b", "qwen2.5-7b-instruct", "Qwen2", bf16_size_gb=14,
    ),
    "llama-3.1-8b-instruct": ModelConfig(
        "llama-8b", "llama-3.1-8b-instruct", "Llama", bf16_size_gb=16,
    ),
    "gemma-3-27b-it": ModelConfig(
        "gemma3-27b", "gemma-3-27b-it", "Gemma3", bf16_size_gb=54,
    ),
    "qwen3.5-27b": ModelConfig(
        "qwen3.5-27b", "qwen3.5-27b", "Qwen3_5",
        bf16_size_gb=54, is_chat=True, in_test_suite=True,
    ),
    "deepseek-r1-distill-qwen-32b": ModelConfig(
        "deepseek-r1-32b", "deepseek-r1-distill-qwen-32b", "Qwen2",
        bf16_size_gb=64, is_chat=True,
    ),
    "llama-3.1-70b-instruct": ModelConfig(
        "llama-70b", "llama-3.1-70b-instruct", "Llama", bf16_size_gb=140,
    ),
    "qwen2.5-72b-instruct": ModelConfig(
        "qwen2-72b", "qwen2.5-72b-instruct", "Qwen2", bf16_size_gb=144,
    ),
    "qwen3.5-122b": ModelConfig(
        "qwen3.5-122b", "qwen3.5-122b", "Qwen3_5M",
        is_moe=True, bf16_size_gb=244, is_chat=True,
    ),
    "qwen3.5-122b-a10b": ModelConfig(
        "qwen3.5-122b-a10b", "qwen3.5-122b-a10b", "Qwen3_5M",
        is_moe=True, bf16_size_gb=244, is_chat=True,
    ),
    "mixtral-8x22b-instruct-v0.1": ModelConfig(
        "mixtral-8x22b", "mixtral-8x22b-instruct-v0.1", "Mixtral",
        is_moe=True, bf16_size_gb=352,
    ),
    "gpt-oss-120b": ModelConfig(
        "gpt-oss-120b", "gpt-oss-120b", "GptOss",
        is_moe=True, bf16_size_gb=240, supports_stream_experts=False,
        is_chat=True, quality_sensitive=True,
    ),
    "qwen3.5-397b": ModelConfig(
        "qwen3.5-397b", "qwen3.5-397b", "Qwen3_5M",
        is_moe=True, bf16_size_gb=794, is_chat=True,
    ),
    "qwen3.5-397b-a27b": ModelConfig(
        "qwen3.5-397b-a27b", "qwen3.5-397b-a27b", "Qwen3_5M",
        is_moe=True, bf16_size_gb=794, is_chat=True,
    ),
}


# ---------------------------------------------------------------------------
# TurboQuant test configurations (KV cache quantization variants).
# Tested on Llama 3B — smallest instruct model.
# ---------------------------------------------------------------------------

TURBOQUANT_CONFIGS = [
    ModelConfig("turbo4-llama", "llama-3.2-3b-instruct", "Llama",
                extra_args=("--kv-quant", "turbo4"), bf16_size_gb=6),
    ModelConfig("turbo2-llama", "llama-3.2-3b-instruct", "Llama",
                extra_args=("--kv-quant", "turbo2"), bf16_size_gb=6),
    ModelConfig("no-kv-quant-llama", "llama-3.2-3b-instruct", "Llama",
                extra_args=("--kv-quant", "none"), bf16_size_gb=6),
    ModelConfig("turbo4-vonly-llama", "llama-3.2-3b-instruct", "Llama",
                extra_args=("--kv-quant", "none:turbo4"), bf16_size_gb=6),
]


# ---------------------------------------------------------------------------
# Prompts — varied prompts that exercise different generation patterns.
# Each model family test uses a different prompt so we're not just testing
# one pattern.  All are factual/instructional — any instruction-tuned model
# should handle them, and the output is easy to validate for coherence.
# ---------------------------------------------------------------------------

PROMPTS = [
    # Short factual — tests basic instruction following.
    "Explain what a hash table is in two sentences.",
    # Longer expository — tests sustained multi-paragraph generation.
    "Write a short paragraph explaining how the internet works, from typing "
    "a URL in a browser to seeing the webpage.  Include DNS, TCP, and HTTP.",
    # Enumeration / structured — tests list generation and formatting.
    "List five common sorting algorithms and give a one-sentence description "
    "of how each one works.",
    # Reasoning / comparison — tests coherent argumentation.
    "Compare and contrast Python and Rust.  What are the strengths and "
    "weaknesses of each language?  Give concrete examples.",
    # Creative but bounded — tests fluency without drifting into gibberish.
    "Write a short story in exactly three sentences about a robot that "
    "discovers it can dream.",
    # Technical explanation — tests domain vocabulary coherence.
    "Explain how public-key cryptography works.  Include the roles of the "
    "public key, private key, and why it is difficult to reverse.",
    # Step-by-step — tests sequential reasoning.
    "Walk me through the steps to deploy a web application to a cloud "
    "provider.  Cover DNS setup, containerisation, and monitoring.",
    # Concise summary — tests compression and accuracy.
    "Summarise the key ideas behind MapReduce in a few sentences.",
    # Multi-part question — tests handling compound prompts.
    "What is gradient descent?  Why is the learning rate important?  "
    "What happens if it is set too high or too low?",
    # Opinion / analysis — tests balanced generation.
    "What are the trade-offs between microservices and monolithic "
    "architectures?  When would you choose one over the other?",
]

# Realistic max_tokens — long enough to exercise sustained generation and
# KV cache behavior (prefill + many decode steps), but not so long that
# tests take forever.  512 tokens is ~400 words, a solid paragraph or two.
MAX_TOKENS = 512

# Default benchmark prompt (shorter, used when not running the full test suite).
DEFAULT_BENCH_PROMPT = "The meaning of life is"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_test_models() -> list[ModelConfig]:
    """Return the curated test models (one per architecture family)."""
    return [m for m in MODEL_REGISTRY.values() if m.in_test_suite]


def get_vision_models() -> list[ModelConfig]:
    """Return test models that support image input."""
    return [m for m in get_test_models() if m.has_vision]


def prompt_for_index(index: int) -> str:
    """Return a prompt from the PROMPTS list, cycling if index exceeds length."""
    return PROMPTS[index % len(PROMPTS)]


def is_base_model(name: str) -> bool:
    """Return True if the model is a base (pretrained) model, not instruct/chat.

    Base models don't have a chat template and need the /v1/completions
    endpoint (raw text prompting) instead of /v1/chat/completions.
    """
    # Strip quant suffixes to check the base name.
    base = name
    for suffix in ("-q4-q8", "-q8", "-q4"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break

    # Check registry first — explicit is_chat overrides heuristics.
    if base in MODEL_REGISTRY:
        cfg = MODEL_REGISTRY[base]
        if cfg.is_chat is not None:
            return not cfg.is_chat

    # Name-based heuristics.
    if "-instruct" in name or name.endswith("-it"):
        return False

    return True


def check_model_shards(model_dir: str) -> tuple[bool, list[str]]:
    """Validate that all safetensors shards referenced by the index exist.

    Returns (ok, missing_files).  For single-file models (no index.json),
    returns (True, []).
    """
    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return True, []

    try:
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = set(weight_map.values())
        missing = [s for s in sorted(shard_files) if not (model_path / s).exists()]
        return len(missing) == 0, missing
    except (json.JSONDecodeError, OSError):
        return True, []
