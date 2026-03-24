# scripts/

Quick-setup scripts for getting rLLM running on a rented GPU.

## Typical setup order

```bash
# 1. Install Rust
bash scripts/install-rust.sh

# 2. Install HuggingFace CLI
bash scripts/install-hf.sh

# 3. Install Claude Code (optional)
bash scripts/install-claude-code.sh

# 4. Download models (set HF_TOKEN first for gated models)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
bash scripts/download-models.sh --medium

# 5. Build and benchmark
bash scripts/benchmark.sh --medium
```

## Scripts

| Script | What it does |
|---|---|
| `install-rust.sh` | Installs stable Rust via rustup |
| `install-hf.sh` | Installs the HuggingFace CLI |
| `install-claude-code.sh` | Installs Node.js (via nvm) + Claude Code |
| `download-models.sh` | Downloads model weights from HuggingFace |
| `benchmark.sh` | Runs each model in bf16/Q4 and prints a Markdown table |

## Model tiers

Both `download-models.sh` and `benchmark.sh` share the same tier system:

| Tier | Flag | Models | Disk |
|---|---|---|---|
| Small | `--small` | 1B–8B (10 models) | ~100 GB |
| Medium | `--medium` | + models up to 35B | ~500 GB |
| Big | `--big` | + 70B–122B models | ~1 TB+ |
| Massive | `--massive` | + 397B models | ~2 TB+ |

Default is `--medium`.
