#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-/home/ye/ml-experiments/llama.cpp-kimi-linear}"
CUDA_SHIM="${CUDA_SHIM:-$ROOT/runtime/cuda13}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/ymcki-Kimi-Linear-48B-A3B-Instruct-GGUF/Kimi-Linear-48B-A3B-Instruct.MXFP4_MOE.gguf}"
HOST="${KIMI_LLAMA_HOST:-0.0.0.0}"
PORT="${KIMI_LLAMA_PORT:-8081}"
CTX="${KIMI_LLAMA_CTX:-8192}"
GPU_LAYERS="${KIMI_LLAMA_GPU_LAYERS:-100}"

export LD_LIBRARY_PATH="$CUDA_SHIM/lib64:${LD_LIBRARY_PATH:-}"

exec "$LLAMA_ROOT/build/bin/llama-server" \
  -m "$MODEL_PATH" \
  -c "$CTX" \
  -ngl "$GPU_LAYERS" \
  --host "$HOST" \
  --port "$PORT"
