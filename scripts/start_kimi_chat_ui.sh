#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/ui"

export KIMI_UI_API_HOST="${KIMI_UI_API_HOST:-0.0.0.0}"
export KIMI_UI_API_PORT="${KIMI_UI_API_PORT:-5174}"
export KIMI_LLAMA_URL="${KIMI_LLAMA_URL:-http://127.0.0.1:8081}"

npm run api
