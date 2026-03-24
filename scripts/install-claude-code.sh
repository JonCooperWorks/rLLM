#!/bin/bash
# ---------------------------------------------------------------------------
# install-claude-code.sh — install Claude Code CLI.
#
# Installs Node.js 22 via nvm if `node` is not found, then installs
# Claude Code globally via npm.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- 1. Ensure Node.js is available --------------------------------------
if ! command -v node &>/dev/null; then
  echo "Node.js not found — installing via nvm..."
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  export NVM_DIR="$HOME/.nvm"
  # shellcheck source=/dev/null
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
  nvm install 22
fi

echo "Node: $(node --version)"

# ---- 2. Install Claude Code ----------------------------------------------
echo "Installing Claude Code..."
npm install -g @anthropic-ai/claude-code

echo "✓ Claude Code installed: $(claude --version)"
