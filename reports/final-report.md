# Kimi Linear reproduction Report

Mode: reproduction
Source: https://github.com/MoonshotAI/Kimi-Linear

## Summary

Initial reproduction is partially successful. I reproduced KDA kernel correctness against FLA's naive recurrent reference, reproduced the operator-speed direction for KDA vs DPLR at a reduced, paper-like forward-only shape through 4096 tokens, and added a CPU recurrence probe that isolates why channel-wise KDA gates can retain one channel while forgetting another. At 8192 tokens, KDA still ran while DPLR OOMed under the same occupied-GPU constraint, which is useful evidence for the memory-efficiency side of the claim. I also stabilized several tiny bf16 short-conv synthetic runs, but those runs did not reproduce the Figure 4 KDA advantage.

## What Was Tested

- Official MoonshotAI/Kimi-Linear repo at `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`.
- Installed FLA path through conda `kimi-linear`: `flash-linear-attention==0.4.0`, `fla-core==0.4.0`.
- KDA correctness: `chunk_kda` and `fused_recurrent_kda` vs `naive_recurrent_kda`.
- KDA-vs-DPLR operator latency at two local scales.
- A recurrence-level selective retention/forgetting probe comparing channel-wise KDA decay against scalar GDN decay.
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

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/kda_operator_benchmark.py \
  --providers kda,dplr --lengths 2048,4096 \
  --heads 8 --head-dim 64 --dtype float16 \
  --warmup 1 --rep 3 \
  --output artifacts/kda_operator_benchmark_backward_h8d64_2048_4096.jsonl

conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task palindrome --models kda,gdn \
  --seq-len 64 --steps 100 --eval-every 20 --eval-batches 4 \
  --batch-size 2 --hidden-size 64 --heads 1 --head-dim 64 \
  --no-short-conv --mlp-ratio 1 --dtype bfloat16 --lr 5e-4 \
  --output artifacts/synthetic_palindrome_tiny_noconv_bf16_100steps.jsonl

conda run -n kimi-linear python scripts/channel_gate_probe.py

conda run -n kimi-linear python scripts/channel_gate_probe.py \
  --seq-lengths 256 512 1024 2048 \
  --samples 5000 \
  --output artifacts/channel_gate_probe_paper_lengths.json \
  --csv-output artifacts/channel_gate_probe_paper_lengths.csv \
  --figure-output reports/figures/channel_gate_probe_paper_lengths.png

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task palindrome --models kda,gdn \
  --vocab-size 16 --seq-len 32 --steps 1000 --eval-every 100 --eval-batches 8 \
  --batch-size 2 --hidden-size 64 --heads 1 --head-dim 64 \
  --mlp-ratio 1 --dtype bfloat16 --lr 5e-4 \
  --output artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_1000steps.jsonl

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task mqar --models kda,gdn \
  --vocab-size 64 --seq-len 64 --steps 2000 --eval-every 200 --eval-batches 8 \
  --batch-size 2 --hidden-size 64 --heads 1 --head-dim 64 \
  --mlp-ratio 1 --dtype bfloat16 --lr 5e-4 \
  --output artifacts/synthetic_mqar_shortconv_bf16_b2_2000steps.jsonl

for lr in 5e-5 1e-4 5e-4 1e-3; do
  safe=${lr//-/_}
  PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
    --task palindrome --models kda,gdn \
    --vocab-size 16 --seq-len 32 --steps 2000 --eval-every 200 --eval-batches 8 \
    --batch-size 2 --hidden-size 64 --heads 1 --head-dim 64 \
    --mlp-ratio 1 --dtype bfloat16 --lr "$lr" \
    --output "artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_lr${safe}_2000steps.jsonl"
done
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
- Reduced backward operator runs under the same occupied-GPU constraint:
  - H=4/D=64: DPLR was slightly faster at T=1024 and T=2048 (`0.4712 ms` vs KDA `0.5084 ms`; `0.4014 ms` vs KDA `0.4309 ms`).
  - H=8/D=64: DPLR was faster at T=2048 (`0.4643 ms` vs KDA `0.5787 ms`), but KDA became faster at T=4096 (`0.5037 ms` vs DPLR `0.8847 ms`, `1.76x` KDA speedup).
  - These backward runs are directional only because they use low warmup/repetition and run with about 30.7 GiB already occupied by another process. Artifacts: `artifacts/kda_operator_benchmark_backward_h4d64_1024_2048.jsonl`, `artifacts/kda_operator_benchmark_backward_h8d64_2048_4096.jsonl`.
- Channel-gate mechanism probe:
  - Task setup: one channel must preserve a signal from the first token; another channel receives noise and must reset for a final recent signal.
  - Scalar GDN's grid-best decay was `0.0` across sequence lengths 16-256, which forgets noise but also gives up the long signal; empirical MSE stayed about `0.96-1.01`.
  - KDA's grid-best decays were `alpha_long=1.0` and `alpha_short=0.0`, giving zero empirical MSE across the same sequence lengths.
  - Paper-length rerun at 256, 512, 1024, and 2048 tokens preserved the same gap: scalar GDN stayed near MSE `0.99-1.02`, while KDA stayed at zero empirical MSE. Artifact: `artifacts/channel_gate_probe_paper_lengths.json`.
  - This directly supports the channel-wise selective retention/forgetting mechanism, but it is a recurrence-level causal probe, not a Figure 4 training reproduction. Artifact: `artifacts/channel_gate_probe.json`.
- Tiny no-short-conv bf16 synthetic palindrome:
  - 2-step smoke completed for KDA and GDN without NaNs.
  - 100-step run stayed finite but did not learn; final eval accuracy was `0.0078` for KDA and `0.0039` for GDN, near chance for a 128-token vocabulary.
  - This validates the low-memory training loop but does not reproduce Figure 4.
- Tiny short-conv bf16 synthetic probes:
  - Palindrome, vocab 128, seq_len 64, hidden 64, batch 2, lr `1e-4`, 200 steps: finite but near chance (`0.0078` KDA, `0.0039` GDN final eval accuracy).
  - Easier palindrome, vocab 16, seq_len 32 request padded to actual length 65, hidden 64, batch 2, lr `5e-4`, 1000 steps: loss improved, but GDN was slightly ahead (`0.1758` GDN vs `0.1367` KDA final eval accuracy). This is a negative result for KDA advantage at this tiny scale.
  - Same easier palindrome at batch 8 / hidden 96 OOMed in Triton backward for both KDA and GDN while the root-owned vLLM process occupied most VRAM.
  - MQAR, vocab 64, seq_len 64 request padded to actual length 65, hidden 64, batch 2, lr `5e-4`, 2000 steps: both KDA and GDN stayed near chance (`0.0208` final eval accuracy).
  - The synthetic harness now pads sequences shorter than 65 tokens because FLA KDA/GDN switch to fused recurrent mode at `q_len <= 64`, and their training path asserts that chunk mode is required.
- Tiny palindrome LR grid:
  - Ran the paper LR grid `{5e-5, 1e-4, 5e-4, 1e-3}` on the memory-safe easy palindrome setting: vocab 16, requested seq_len 32 padded to actual length 65, hidden 64, batch 2, 2000 steps.
  - This did not reproduce Figure 4's KDA advantage. Best KDA final eval accuracy was `0.3711` at lr `1e-3`; best GDN final eval accuracy was `0.9453` at lr `1e-3`.
  - Final accuracies by LR: KDA `{5e-5: 0.0508, 1e-4: 0.0645, 5e-4: 0.1621, 1e-3: 0.3711}`; GDN `{5e-5: 0.0586, 1e-4: 0.0684, 5e-4: 0.7832, 1e-3: 0.9453}`.
  - This is constrained negative evidence for the tiny setting, not a paper-scale result. Artifacts: `artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_lr*_2000steps.jsonl`.

## Failures and Limitations

- Full 48B BF16 model reproduction is not feasible on one 32GB RTX 5090. Official vLLM docs assume 4 or 8 GPUs for the released checkpoint at large context.
- A root-owned vLLM process was using about 30.7 GiB of VRAM, leaving only a few hundred MiB during some runs.
- Synthetic palindrome training is not reproduced yet:
  - KDA/GDN with fp16 and lr `1e-3` produced NaNs quickly.
  - Mamba2 OOMed under current GPU pressure.
  - A tiny no-short-conv bf16 run stayed finite but did not learn, and it intentionally omits short convolution, which the paper says is important.
  - Tiny short-conv bf16 runs are stable but too small/noisy to show the paper's KDA advantage; the easiest LR-grid palindrome run strongly favored GDN.
- The channel-gate probe is intentionally simpler than the paper's learned synthetic tasks; it validates the recurrence mechanism, not optimization under the paper's training setup.
- The official MoonshotAI/Kimi-Linear repo contains report/model-card assets, not the full private training/eval code or datasets.

## Reproducibility Notes

- Current experiment branch: `reproduce/kimi-linear-source-review`.
- Hardware snapshot: `experiments/hardware-snapshot-20260523-233718.json`.
- Machine-readable run records: `experiments/runs.jsonl`.
- Scripts added:
  - `scripts/kda_kernel_smoke.py`
  - `scripts/kda_operator_benchmark.py`
  - `scripts/synthetic_recall_probe.py`
  - `scripts/channel_gate_probe.py`

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
