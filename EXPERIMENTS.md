# Kimi Linear reproduction

Mode: reproduction
Source: https://github.com/MoonshotAI/Kimi-Linear
Created: 2026-05-23T23:37:18-07:00

## Goal

Learn Kimi Linear's architectural breakthrough by reproducing tractable mechanism-level findings before building UI. The order is:

1. Reproduce KDA kernel correctness and operator-speed behavior.
2. Reproduce scaled synthetic recall/copy/state-tracking findings.
3. Attempt full-model inference/serving only if the local hardware path is credible.
4. Build a UI after reproduction has clarified the architecture and experiment controls.

## Source Context

- Upstream repo: https://github.com/MoonshotAI/Kimi-Linear
- Paper/model card: https://arxiv.org/abs/2510.26692, https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct, https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base
- Upstream commit/version: `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`
- FLA implementation: `/home/ye/ml-experiments/flash-linear-attention` at `1c403c3a82896ca0dd2ef8a952a85e3bdbffc941`; installed `flash-linear-attention==0.4.0`
- License: MIT

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB VRAM
- Driver/CUDA: driver 580.159.03, torch CUDA runtime 12.8 in conda `kimi-linear`
- Python: 3.11.14 in conda `kimi-linear`
- Framework: torch 2.9.0+cu128, transformers 4.57.1, fla-core 0.4.0, flash-linear-attention 0.4.0, vLLM 0.11.1rc6 dev build
- Current blocker: root-owned `VLLM::EngineCore` was using about 30.7 GiB VRAM at source-review time.

## Milestones

- [x] Source/context captured
- [x] Environment installed
- [x] Smoke test completed
- [ ] Main experiment completed
- [x] Report written

## Run Log

Append notable runs here and keep machine-readable records in `experiments/runs.jsonl`.

- 2026-05-23 23:37 PDT: cloned official MoonshotAI/Kimi-Linear into this workspace at `8c1d85e`, created branch `reproduce/kimi-linear-source-review`, bootstrapped experiment artifacts, captured hardware snapshot, and completed initial source review.
- 2026-05-23 23:44 PDT: `scripts/kda_kernel_smoke.py` passed. `chunk_kda` matched naive recurrent KDA within max error 0.00390625 across outputs/gradients under tolerance 0.05. Artifact: `artifacts/kda_kernel_smoke.json`.
- 2026-05-23 23:46 PDT: tiny backward benchmark at B=1/H=2/D=64, T=256/512 did not reproduce the paper speed direction; DPLR was slightly faster than KDA. Artifact: `artifacts/kda_operator_benchmark_tiny.jsonl`.
- 2026-05-23 23:47 PDT: forward-only benchmark closer to paper shape B=1/H=16/D=128 reproduced KDA speed direction at longer lengths: 1.77x faster than DPLR at 2048 tokens and 1.86x faster at 4096 tokens. Artifacts: `artifacts/kda_operator_benchmark_forward_h16d128.jsonl`, `artifacts/kda_operator_benchmark_forward_h16d128_4096.jsonl`.
- 2026-05-23 23:50 PDT: synthetic palindrome harness ran but did not produce a valid learning result. fp16/lr=1e-3 went NaN for KDA/GDN; Mamba2 OOMed while root-owned vLLM occupied about 30.7 GiB VRAM. Artifact: `artifacts/synthetic_palindrome_smoke.jsonl`.
- 2026-05-24 00:00 PDT: extended the forward-only paper-like operator benchmark to 8192 tokens. KDA ran at `1.0256 ms`; DPLR OOMed under the same occupied-GPU constraint. Artifact: `artifacts/kda_operator_benchmark_forward_h16d128_8192.jsonl`.
- 2026-05-24 00:02 PDT: patched benchmark/synthetic scripts to clear gradients, flush metadata, and fail fast on non-finite loss.
- 2026-05-24 00:04 PDT: tiny no-short-conv bf16 synthetic palindrome smoke completed without NaNs for KDA/GDN, proving the training loop can run in the current constrained GPU state. Artifact: `artifacts/synthetic_palindrome_tiny_noconv_bf16.jsonl`.
- 2026-05-24 00:05 PDT: 100-step tiny no-short-conv bf16 palindrome curve for KDA/GDN stayed finite but remained near chance; not a Figure 4 reproduction. Artifact: `artifacts/synthetic_palindrome_tiny_noconv_bf16_100steps.jsonl`.
