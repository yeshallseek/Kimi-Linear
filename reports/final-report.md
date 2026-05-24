# Kimi Linear reproduction Report

Mode: reproduction
Source: https://github.com/MoonshotAI/Kimi-Linear

## Summary

Initial reproduction is partially successful. I reproduced KDA kernel correctness against FLA's naive recurrent reference and reproduced the operator-speed direction for KDA vs DPLR at a reduced, paper-like forward-only shape through 4096 tokens. At 8192 tokens, KDA still ran while DPLR OOMed under the same occupied-GPU constraint, which is useful evidence for the memory-efficiency side of the claim. I did not yet reproduce the synthetic learning curves from Figure 4.

## What Was Tested

- Official MoonshotAI/Kimi-Linear repo at `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`.
- Installed FLA path through conda `kimi-linear`: `flash-linear-attention==0.4.0`, `fla-core==0.4.0`.
- KDA correctness: `chunk_kda` and `fused_recurrent_kda` vs `naive_recurrent_kda`.
- KDA-vs-DPLR operator latency at two local scales.
- A first synthetic palindrome training harness using tiny KDA/GDN/Mamba2-style mixers, plus a constrained no-conv bf16 smoke variant.

## Commands

```bash
conda run -n kimi-linear python scripts/kda_kernel_smoke.py \
  --output artifacts/kda_kernel_smoke.json \
  --seq-len 64 --heads 1 --head-dim 64 --batch-size 1 \
  --dtype float16 --tolerance 0.05

conda run -n kimi-linear python scripts/kda_operator_benchmark.py \
  --providers kda,dplr --lengths 256,512 \
  --heads 2 --head-dim 64 --dtype float16 \
  --warmup 1 --rep 3 \
  --output artifacts/kda_operator_benchmark_tiny.jsonl

conda run -n kimi-linear python scripts/kda_operator_benchmark.py \
  --providers kda,dplr --lengths 1024,2048 \
  --heads 16 --head-dim 128 --dtype float16 \
  --warmup 1 --rep 3 --forward-only \
  --output artifacts/kda_operator_benchmark_forward_h16d128.jsonl

conda run -n kimi-linear python scripts/kda_operator_benchmark.py \
  --providers kda,dplr --lengths 4096 \
  --heads 16 --head-dim 128 --dtype float16 \
  --warmup 1 --rep 3 --forward-only \
  --output artifacts/kda_operator_benchmark_forward_h16d128_4096.jsonl

conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task palindrome --models kda,gdn,mamba2 \
  --seq-len 128 --steps 10 --eval-every 5 --eval-batches 1 \
  --batch-size 4 --hidden-size 128 --heads 2 --head-dim 64 \
  --mamba-head-dim 64 --mamba-state-size 64 --dtype float16 \
  --output artifacts/synthetic_palindrome_smoke.jsonl

conda run -n kimi-linear python scripts/kda_operator_benchmark.py \
  --providers kda,dplr --lengths 8192 \
  --heads 16 --head-dim 128 --dtype float16 \
  --warmup 2 --rep 5 --forward-only \
  --output artifacts/kda_operator_benchmark_forward_h16d128_8192.jsonl

conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task palindrome --models kda,gdn \
  --seq-len 64 --steps 100 --eval-every 20 --eval-batches 4 \
  --batch-size 2 --hidden-size 64 --heads 1 --head-dim 64 \
  --no-short-conv --mlp-ratio 1 --dtype bfloat16 --lr 5e-4 \
  --output artifacts/synthetic_palindrome_tiny_noconv_bf16_100steps.jsonl
```

## Results

- Kernel smoke: passed. Maximum observed error across output/final-state/gradient comparisons was `0.00390625`, under tolerance `0.05`. Artifact: `artifacts/kda_kernel_smoke.json`.
- Tiny backward operator run, B=1/H=2/D=64:
  - T=256: KDA `0.4916 ms`, DPLR `0.4163 ms`.
  - T=512: KDA `0.4259 ms`, DPLR `0.3917 ms`.
  - This does not reproduce the paper speed direction, but the shape is much smaller than the paper's benchmark regime.
- Forward-only operator run, B=1/H=16/D=128:
  - T=1024: KDA `0.1595 ms`, DPLR `0.1638 ms` (near tie).
  - T=2048: KDA `0.2212 ms`, DPLR `0.3912 ms` (`1.77x` KDA speedup).
  - T=4096: KDA `0.4619 ms`, DPLR `0.8607 ms` (`1.86x` KDA speedup).
  - T=8192: KDA `1.0256 ms`; DPLR OOMed trying to allocate an additional 256 MiB while KDA fit. This is a memory-efficiency datapoint rather than a latency-ratio datapoint.
  - This partially reproduces the paper's operator-efficiency claim direction, at smaller lengths and forward-only due to current VRAM constraints.
- Tiny no-short-conv bf16 synthetic palindrome:
  - 2-step smoke completed for KDA and GDN without NaNs.
  - 100-step run stayed finite but did not learn; final eval accuracy was `0.0078` for KDA and `0.0039` for GDN, near chance for a 128-token vocabulary.
  - This validates the low-memory training loop but does not reproduce Figure 4.

## Failures and Limitations

- Full 48B BF16 model reproduction is not feasible on one 32GB RTX 5090. Official vLLM docs assume 4 or 8 GPUs for the released checkpoint at large context.
- A root-owned vLLM process was using about 30.7 GiB of VRAM, leaving only a few hundred MiB during some runs.
- Synthetic palindrome training is not reproduced yet:
  - KDA/GDN with fp16 and lr `1e-3` produced NaNs quickly.
  - Mamba2 OOMed under current GPU pressure.
  - A tiny no-short-conv bf16 run stayed finite but did not learn, and it intentionally omits short convolution, which the paper says is important.
- The official MoonshotAI/Kimi-Linear repo contains report/model-card assets, not the full private training/eval code or datasets.

## Reproducibility Notes

- Current experiment branch: `reproduce/kimi-linear-source-review`.
- Hardware snapshot: `experiments/hardware-snapshot-20260523-233718.json`.
- Machine-readable run records: `experiments/runs.jsonl`.
- Scripts added:
  - `scripts/kda_kernel_smoke.py`
  - `scripts/kda_operator_benchmark.py`
  - `scripts/synthetic_recall_probe.py`

## Sources Used

Separate official sources from third-party sources and cite URLs/commits used.

- Official paper: https://arxiv.org/abs/2510.26692
- Official repo: https://github.com/MoonshotAI/Kimi-Linear at `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`
- Official Instruct model: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct at `e1df551a447157d4658b573f9a695d57658590e9`
- Official Base model: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base at `3b171c17bfc4ee348599b6781a2ca8715c21c8dc`
- Official FLA KDA implementation: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda; local clone observed at `1c403c3a82896ca0dd2ef8a952a85e3bdbffc941`
- vLLM Kimi-Linear recipe: https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-Linear.html

## Next Experiments

- Free the GPU or move runs to a machine without the current vLLM process, then rerun backward operator benchmarks at H=16/D=128 for 2k-64k lengths.
- Stabilize synthetic probes with bf16/float32, short convolution enabled, and a small LR sweep over `{5e-5, 1e-4, 5e-4, 1e-3}` as in the paper.
- Add an MQAR probe after palindrome runs are stable.
- Build the UI only after the synthetic learning result is credible.
