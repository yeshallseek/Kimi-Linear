# Kimi Linear reproduction Report

Mode: reproduction
Source: https://github.com/MoonshotAI/Kimi-Linear

## Summary

Initial reproduction is now substantially stronger, but the learned synthetic result is still seed-sensitive and task-sensitive. I reproduced KDA kernel correctness against FLA's naive recurrent reference, reproduced the operator-speed direction for KDA vs DPLR at reduced paper-like shapes, and got paper-shape learning results for palindrome, 64-stack state tracking, and Zoology-style MQAR. Across palindrome seeds `42`, `123`, and `7`, KDA learned reliably (`0.8981-0.9641`, mean final accuracy `0.9286`) while GDN was bimodal (mean `0.6701`). On paper-shape 64-stack at lr `1e-3`, KDA mean final accuracy was `0.9654` vs GDN `0.9424` over three seeds, with KDA higher on two seeds and tied on one; GDN still reached high accuracy earlier. The MQAR result depended on matching FLA full-model recurrent initialization: source-style KDA `dt_bias` initialization changed KDA from near chance to `0.9988` final accuracy on the hard high-vocab Zoology curriculum eval slice. Remaining gaps are the Mamba2 20k baseline and multi-seed/source-style baseline consolidation before UI work.

## What Was Tested

- Official MoonshotAI/Kimi-Linear repo at `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`.
- Installed FLA path through conda `kimi-linear`: `flash-linear-attention==0.4.0`, `fla-core==0.4.0`.
- KDA correctness: `chunk_kda` and `fused_recurrent_kda` vs `naive_recurrent_kda`.
- KDA-vs-DPLR operator latency at two local scales.
- A recurrence-level selective retention/forgetting probe comparing channel-wise KDA decay against scalar GDN decay.
- Synthetic palindrome, MQAR, and stack/state-tracking training harnesses using tiny KDA/GDN/Mamba2-style mixers, plus constrained bf16 smoke variants and source-style KDA recurrent initialization controls.
- Free-GPU paper-shape palindrome, MQAR, and 64-stack probes at 2 layers, hidden 256, 2 heads, head_dim 128, seq_len 256, vocab 128, batch 4.

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

conda run -n kimi-linear python scripts/summarize_synthetic_runs.py \
  artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_lr*_2000steps.jsonl \
  --output artifacts/synthetic_palindrome_easy_lr_sweep_summary.json \
  --csv-output artifacts/synthetic_palindrome_easy_lr_sweep_summary.csv

for lr in 5e-5 1e-4 5e-4 1e-3; do
  safe=${lr//-/_}
  PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
    --task palindrome --models kda,gdn \
    --vocab-size 16 --seq-len 32 --steps 2000 --eval-every 200 --eval-batches 8 \
    --batch-size 2 --hidden-size 64 --heads 1 --head-dim 64 \
    --mlp-ratio 1 --dtype bfloat16 --lr "$lr" --seed 123 \
    --output "artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_seed123_lr${safe}_2000steps.jsonl"
done

conda run -n kimi-linear python scripts/summarize_synthetic_runs.py \
  artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_lr*_2000steps.jsonl \
  artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_seed123_lr*_2000steps.jsonl \
  --output artifacts/synthetic_palindrome_easy_lr_sweep_2seed_summary.json \
  --csv-output artifacts/synthetic_palindrome_easy_lr_sweep_2seed_summary.csv

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task palindrome --models kda,gdn \
  --vocab-size 16 --seq-len 32 --steps 10000 --eval-every 1000 --eval-batches 8 \
  --batch-size 2 --hidden-size 64 --heads 1 --head-dim 64 \
  --mlp-ratio 1 --dtype bfloat16 --lr 1e-3 --seed 42 \
  --output artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_seed42_lr1e_3_10000steps.jsonl

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task palindrome --models kda,gdn \
  --vocab-size 16 --seq-len 32 --steps 10000 --eval-every 1000 --eval-batches 8 \
  --batch-size 2 --hidden-size 64 --heads 1 --head-dim 64 \
  --mlp-ratio 1 --dtype bfloat16 --lr 1e-3 --seed 123 \
  --output artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_seed123_lr1e_3_10000steps.jsonl

conda run -n kimi-linear python scripts/summarize_synthetic_runs.py \
  artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_seed42_lr1e_3_10000steps.jsonl \
  artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_seed123_lr1e_3_10000steps.jsonl \
  --output artifacts/synthetic_palindrome_easy_lr1e3_10000step_2seed_summary.json \
  --csv-output artifacts/synthetic_palindrome_easy_lr1e3_10000step_2seed_summary.csv

for seed in 42 123; do
  PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
    --task stack --models kda,gdn \
    --vocab-size 128 --seq-len 96 --steps 2000 --eval-every 200 --eval-batches 8 \
    --batch-size 2 --hidden-size 64 --heads 1 --head-dim 64 \
    --mlp-ratio 1 --dtype bfloat16 --lr 1e-3 --seed "$seed" --num-stacks 16 \
    --output "artifacts/synthetic_stack_shortconv_bf16_b2_seed${seed}_lr1e_3_2000steps.jsonl"
done

conda run -n kimi-linear python scripts/summarize_synthetic_runs.py \
  artifacts/synthetic_stack_shortconv_bf16_b2_seed42_lr1e_3_2000steps.jsonl \
  artifacts/synthetic_stack_shortconv_bf16_b2_seed123_lr1e_3_2000steps.jsonl \
  --output artifacts/synthetic_stack_shortconv_bf16_b2_lr1e3_2seed_summary.json \
  --csv-output artifacts/synthetic_stack_shortconv_bf16_b2_lr1e3_2seed_summary.csv

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task palindrome --models kda,gdn,mamba2 \
  --vocab-size 128 --seq-len 256 --steps 20 --eval-every 10 --eval-batches 1 \
  --batch-size 1 --hidden-size 256 --heads 2 --head-dim 128 \
  --mamba-head-dim 128 --mamba-state-size 128 --mamba-expand 2 \
  --mlp-ratio 2 --dtype bfloat16 --lr 5e-4 --seed 42 \
  --output artifacts/synthetic_palindrome_paper_shape_bf16_b1_20steps.jsonl

conda run -n kimi-linear python scripts/summarize_synthetic_runs.py \
  artifacts/synthetic_palindrome_paper_shape_bf16_b1_20steps.jsonl \
  --output artifacts/synthetic_palindrome_paper_shape_bf16_b1_20steps_summary.json \
  --csv-output artifacts/synthetic_palindrome_paper_shape_bf16_b1_20steps_summary.csv

for lr in 5e-5 1e-4 5e-4 1e-3; do
  safe=${lr//-/_}
  PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
    --task palindrome --models kda,gdn,mamba2 \
    --vocab-size 128 --seq-len 256 --steps 2000 --eval-every 200 --eval-batches 4 \
    --batch-size 4 --hidden-size 256 --heads 2 --head-dim 128 \
    --mamba-head-dim 128 --mamba-state-size 128 --mamba-expand 2 \
    --mlp-ratio 2 --dtype bfloat16 --lr "$lr" --seed 42 \
    --output "artifacts/synthetic_palindrome_paper_shape_bf16_b4_lr${safe}_2000steps_freegpu.jsonl"
done

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task palindrome --models kda,gdn,mamba2 \
  --vocab-size 128 --seq-len 256 --steps 20000 --eval-every 2000 --eval-batches 4 \
  --batch-size 4 --hidden-size 256 --heads 2 --head-dim 128 \
  --mamba-head-dim 128 --mamba-state-size 128 --mamba-expand 2 \
  --mlp-ratio 2 --dtype bfloat16 --lr 1e-3 --seed 42 \
  --output artifacts/synthetic_palindrome_paper_shape_bf16_b4_lr1e_3_20000steps_freegpu.jsonl

for seed in 123 7; do
  PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
    --task palindrome --models kda,gdn \
    --vocab-size 128 --seq-len 256 --steps 20000 --eval-every 2000 --eval-batches 4 \
    --batch-size 4 --hidden-size 256 --heads 2 --head-dim 128 \
    --mamba-head-dim 128 --mamba-state-size 128 --mamba-expand 2 \
    --mlp-ratio 2 --dtype bfloat16 --lr 1e-3 --seed "$seed" \
    --output "artifacts/synthetic_palindrome_paper_shape_bf16_b4_seed${seed}_lr1e_3_20000steps_freegpu.jsonl"
done

conda run -n kimi-linear python scripts/summarize_synthetic_runs.py \
  artifacts/synthetic_palindrome_paper_shape_bf16_b4_lr1e_3_20000steps_freegpu.jsonl \
  artifacts/synthetic_palindrome_paper_shape_bf16_b4_seed123_lr1e_3_20000steps_freegpu.jsonl \
  artifacts/synthetic_palindrome_paper_shape_bf16_b4_seed7_lr1e_3_20000steps_freegpu.jsonl \
  --output artifacts/synthetic_palindrome_paper_shape_bf16_b4_lr1e3_20000step_3seed_summary.json \
  --csv-output artifacts/synthetic_palindrome_paper_shape_bf16_b4_lr1e3_20000step_3seed_summary.csv

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task mqar --models kda,gdn,mamba2 \
  --vocab-size 128 --seq-len 256 --num-queries 63 \
  --steps 200 --eval-every 50 --eval-batches 4 \
  --batch-size 4 --hidden-size 256 --heads 2 --head-dim 128 \
  --mamba-head-dim 128 --mamba-state-size 128 --mamba-expand 2 \
  --mlp-ratio 2 --dtype bfloat16 --lr 5e-4 --seed 42 \
  --output artifacts/synthetic_mqar_paper_shape_bf16_b4_q63_200steps_freegpu.jsonl

for lr in 5e-5 1e-4 5e-4 1e-3; do
  safe=${lr//-/_}
  PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
    --task mqar --models kda,gdn,mamba2 \
    --vocab-size 128 --seq-len 256 --num-queries 63 \
    --steps 2000 --eval-every 200 --eval-batches 4 \
    --batch-size 4 --hidden-size 256 --heads 2 --head-dim 128 \
    --mamba-head-dim 128 --mamba-state-size 128 --mamba-expand 2 \
    --mlp-ratio 2 --dtype bfloat16 --lr "$lr" --seed 42 \
    --output "artifacts/synthetic_mqar_paper_shape_bf16_b4_q63_lr${safe}_2000steps_freegpu.jsonl"
done

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task mqar --models kda,gdn \
  --vocab-size 128 --seq-len 256 --num-queries 63 \
  --steps 20000 --eval-every 2000 --eval-batches 4 \
  --batch-size 4 --hidden-size 256 --heads 2 --head-dim 128 \
  --mamba-head-dim 128 --mamba-state-size 128 --mamba-expand 2 \
  --mlp-ratio 2 --dtype bfloat16 --lr 1e-3 --seed 42 \
  --output artifacts/synthetic_mqar_paper_shape_bf16_b4_q63_lr1e_3_20000steps_freegpu.jsonl

conda run -n kimi-linear python scripts/summarize_synthetic_runs.py \
  artifacts/synthetic_mqar_paper_shape_bf16_b4_q63_lr1e_3_20000steps_freegpu.jsonl \
  --output artifacts/synthetic_mqar_paper_shape_bf16_b4_q63_lr1e3_20000steps_freegpu_summary.json \
  --csv-output artifacts/synthetic_mqar_paper_shape_bf16_b4_q63_lr1e3_20000steps_freegpu_summary.csv

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task stack --models kda,gdn,mamba2 \
  --vocab-size 128 --seq-len 256 --num-stacks 64 \
  --steps 200 --eval-every 50 --eval-batches 4 \
  --batch-size 4 --hidden-size 256 --heads 2 --head-dim 128 \
  --mamba-head-dim 128 --mamba-state-size 128 --mamba-expand 2 \
  --mlp-ratio 2 --dtype bfloat16 --lr 5e-4 --seed 42 \
  --output artifacts/synthetic_stack_paper_shape_bf16_b4_s64_200steps_freegpu.jsonl

for lr in 5e-5 1e-4 5e-4 1e-3; do
  safe=${lr//-/_}
  PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
    --task stack --models kda,gdn,mamba2 \
    --vocab-size 128 --seq-len 256 --num-stacks 64 \
    --steps 2000 --eval-every 200 --eval-batches 4 \
    --batch-size 4 --hidden-size 256 --heads 2 --head-dim 128 \
    --mamba-head-dim 128 --mamba-state-size 128 --mamba-expand 2 \
    --mlp-ratio 2 --dtype bfloat16 --lr "$lr" --seed 42 \
    --output "artifacts/synthetic_stack_paper_shape_bf16_b4_s64_lr${safe}_2000steps_freegpu.jsonl"
done

for seed in 123 7; do
  PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
    --task stack --models kda,gdn \
    --vocab-size 128 --seq-len 256 --num-stacks 64 \
    --steps 2000 --eval-every 200 --eval-batches 4 \
    --batch-size 4 --hidden-size 256 --heads 2 --head-dim 128 \
    --mamba-head-dim 128 --mamba-state-size 128 --mamba-expand 2 \
    --mlp-ratio 2 --dtype bfloat16 --lr 1e-3 --seed "$seed" \
    --output "artifacts/synthetic_stack_paper_shape_bf16_b4_s64_seed${seed}_lr1e_3_2000steps_freegpu.jsonl"
done

conda run -n kimi-linear python scripts/summarize_synthetic_runs.py \
  artifacts/synthetic_stack_paper_shape_bf16_b4_s64_lr1e_3_2000steps_freegpu.jsonl \
  artifacts/synthetic_stack_paper_shape_bf16_b4_s64_seed123_lr1e_3_2000steps_freegpu.jsonl \
  artifacts/synthetic_stack_paper_shape_bf16_b4_s64_seed7_lr1e_3_2000steps_freegpu.jsonl \
  --output artifacts/synthetic_stack_paper_shape_bf16_b4_s64_lr1e3_2000step_3seed_summary.json \
  --csv-output artifacts/synthetic_stack_paper_shape_bf16_b4_s64_lr1e3_2000step_3seed_summary.csv
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
  - Repeating the LR grid with seed `123` did not reverse the conclusion. Across seeds `42` and `123`, best mean final accuracy was GDN at lr `1e-3` (`0.9053`) vs KDA at lr `1e-3` (`0.4814`). KDA did beat GDN on seed `123` at lr `5e-4`, so the constrained setting is seed/LR sensitive.
  - Extending the best LR `1e-3` to 10,000 steps showed that KDA can improve substantially on seed `123`, but still did not catch GDN on the two-seed mean: final accuracy averaged `0.5322` for KDA vs `0.9521` for GDN. Seed-level final accuracy was KDA `{42: 0.2305, 123: 0.8340}` and GDN `{42: 0.9648, 123: 0.9395}`.
  - This is constrained negative evidence for the tiny setting, not a paper-scale result. Artifacts: `artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_lr*_2000steps.jsonl`, `artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_seed123_lr*_2000steps.jsonl`, `artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_seed42_lr1e_3_10000steps.jsonl`, `artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_seed123_lr1e_3_10000steps.jsonl`, summarized in `artifacts/synthetic_palindrome_easy_lr_sweep_2seed_summary.json` and `artifacts/synthetic_palindrome_easy_lr1e3_10000step_2seed_summary.json`.
- Tiny stack/state-tracking probe:
  - Added a stack generator matching the paper's operation structure: PUSH stores a value for a stack ID; POP asks the model to predict the most recently pushed value for that stack.
  - The local run uses a smaller 16-stack setup, not the paper's 64-stack task: vocab 128, seq_len 96, hidden 64, batch 2, lr `1e-3`, 2000 steps, seeds `42` and `123`.
  - Both KDA and GDN learned, but GDN was higher on the two-seed mean: final eval accuracy `0.9428` for GDN vs `0.8311` for KDA.
  - This expands Figure 4 coverage to the state-tracking task, but remains constrained negative evidence. Artifact: `artifacts/synthetic_stack_shortconv_bf16_b2_lr1e3_2seed_summary.json`.
- Paper-shape synthetic feasibility smoke:
  - Ran the paper's small synthetic model shape at batch 1 for 20 steps: 2 layers, hidden 256, 2 heads, head_dim 128, seq_len 256, vocab 128, bf16, lr `5e-4`.
  - KDA and GDN both OOMed during Triton backward under the current occupied-GPU constraint.
  - Mamba2 completed the smoke with final eval accuracy `0.0079`, essentially chance for vocab 128.
  - This confirms the paper-shape KDA/GDN synthetic reproduction should wait until the root-owned vLLM process frees VRAM. Artifacts: `artifacts/synthetic_palindrome_paper_shape_bf16_b1_20steps.jsonl`, `artifacts/synthetic_palindrome_paper_shape_bf16_b1_20steps_summary.json`.
- Free-GPU paper-shape palindrome reproduction:
  - Stopped root-owned `VLLM::EngineCore` PID `3107306` with user authorization, freeing the RTX 5090 for reproduction.
  - Batch-4, 200-step smoke at the paper small-model shape fit for KDA, GDN, and Mamba2.
  - 2000-step LR grid: KDA's best final accuracy was `0.0389` at lr `1e-3`; GDN's best final accuracy was `0.0143` at lr `1e-3`; Mamba2's best final accuracy was `0.0861` at lr `5e-4`.
  - 20,000-step extension at lr `1e-3`, seed `42`: KDA reached `0.9237` final eval accuracy and GDN reached `0.0389`, reproducing the Figure 4 palindrome direction for that seed.
  - Multi-seed check at the same setting: KDA final accuracy was `{42: 0.9237, 123: 0.8981, 7: 0.9641}`; GDN final accuracy was `{42: 0.0389, 123: 0.9808, 7: 0.9906}`. Mean final accuracy favored KDA (`0.9286`) over GDN (`0.6701`) because GDN collapsed on seed `42`, but GDN won two of three seeds.
  - The current interpretation is KDA is more robust on this harness, while GDN can learn the task very well for some seeds. This is a partial, nuanced reproduction rather than a blanket KDA-over-GDN result.
  - Mamba2's 20k segment and a separate Mamba2-only 20k run stayed GPU-active without producing a first eval record for several minutes, so they were terminated and recorded as errors. Artifacts: `artifacts/synthetic_palindrome_paper_shape_bf16_b4_lr_sweep_2000steps_freegpu_summary.json`, `artifacts/synthetic_palindrome_paper_shape_bf16_b4_20000steps_freegpu_summary.json`.
- Free-GPU paper-shape MQAR reproduction attempt:
  - Used `--num-queries 63`, producing actual sequence length 249, close to the paper's 256-token setting.
  - Batch-4, 200-step smoke fit for KDA, GDN, and Mamba2; all stayed near chance (`0.0151` KDA, `0.0161` GDN, `0.0141` Mamba2 final accuracy).
  - 2000-step LR grid: best final accuracy was KDA `0.0272` at lr `1e-4`, GDN `0.0252` at lr `1e-3`, and Mamba2 `0.0696` at lr `5e-4`.
  - 20,000-step KDA/GDN extension at lr `1e-3`, seed `42`: KDA remained near chance (`0.0131` final accuracy, best `0.0202`), while GDN rose modestly (`0.0917` final accuracy, best `0.0938`) but did not solve the task.
  - This is a negative MQAR reproduction for KDA in the current harness, and a weak positive optimization signal for GDN. Artifacts: `artifacts/synthetic_mqar_paper_shape_bf16_b4_q63_lr_sweep_2000steps_freegpu_summary.json`, `artifacts/synthetic_mqar_paper_shape_bf16_b4_q63_lr1e3_20000steps_freegpu_summary.json`.
- Corrected Zoology-style MQAR audit:
  - Inspected HazyResearch Zoology's `zoology/data/multiquery_ar.py`, the task family cited by the paper. The prior local MQAR generator was a contiguous language-model layout and did not match Zoology's random query-region layout, vocab 8192, upper-half value tokens, one-query-per-key structure, or power-law gap sampling.
  - Patched `scripts/synthetic_recall_probe.py` with `--mqar-layout zoology`, source-style gap/filler controls, label-based evaluation, optional tied embeddings, and optional `std=0.02` initialization.
  - Corrected paper-shape MQAR (`vocab=8192`, `seq_len=256`, `num_pairs=64`) fit on GPU but did not learn retrieval. The 20,000-step lr `1e-3` run ended at KDA `0.00049`, GDN `0.0`, and Mamba2 `0.0` final accuracy, with loss around `log(4096)`, meaning the models learned the value-token half but not key-value lookup.
  - Source-style tied embeddings, `std=0.02` init, MLP ratio 4, high LRs, batch 64, and a `num_pairs=16` high-vocab slice still stayed at chance by 2000 steps.
  - A tiny positive control did learn Zoology-style MQAR: with `vocab=256`, `seq_len=64`, `num_pairs=4`, batch 64, GDN reached `0.9971` final accuracy and KDA reached `0.2612` by 2000 steps. This proves the corrected harness can learn easy MQAR, but the paper-shape/high-vocab MQAR result remains unresolved. Artifact: `artifacts/synthetic_mqar_zoology_tied_diagnostics_summary.json`.
- Zoology curriculum MQAR diagnostic:
  - Added `--mqar-train-curriculum zoology_figure3`, matching the official Zoology train mix weights while evaluating on the hard `seq_len=256, num_pairs=64` target.
  - This made high-vocab MQAR learnable in the local harness, but in the wrong direction for the paper claim: GDN reached `0.9838` final accuracy at 2000 steps, while KDA stayed near chance (`0.0016` final, best `0.0027`) under the same run.
  - KDA-only checks at the higher Zoology LR scale (`1e-3`, `3.16e-3`, `1e-2`, `3.16e-2`) with weight decay `0.1` also stayed near chance.
  - This was a stronger MQAR non-reproduction before the next initialization audit found the local KDA wrapper mismatch. Artifact: `artifacts/synthetic_mqar_zoology_curriculum_diagnostics_summary.json`.
- KDA source-initialization MQAR diagnostic:
  - Audited FLA's full KDA model initialization against the tiny standalone layer wrapper. Full `KDAPreTrainedModel._init_weights` samples KDA `dt_bias` from log-uniform time constants in `[0.001, 0.1]`; the bare `KimiDeltaAttention` layer initializes `dt_bias=0`.
  - Added explicit `--source-init`, `--source-init-scope`, and `--source-param-groups` controls to the synthetic harness.
  - With source-style KDA recurrent initialization, KDA now solves the hard Zoology curriculum eval slice: full source init reached `0.9988` final accuracy and `0.9993` best accuracy at 2000 steps.
  - Ablations isolate the effect to recurrent gate initialization. Recurrent-only source init reached `0.9695` final accuracy at 1000 steps; weights-only source init stayed at `0.00049`; source no-decay grouping without recurrent init stayed at `0.0014`.
  - This changes the MQAR interpretation from “KDA non-reproduction” to “KDA reproduction depends on the full-model recurrent initialization, especially `dt_bias`.” Artifact: `artifacts/synthetic_mqar_zoology_source_init_ablation_summary.json`.
- Free-GPU paper-shape 64-stack reproduction:
  - Used 64 stacks and actual sequence length 255, matching the paper-shape synthetic setting closely within the local generator.
  - Batch-4, 200-step smoke fit for KDA, GDN, and Mamba2. At 200 steps, GDN was ahead (`0.2165`) with KDA lower but learning (`0.1280`) and Mamba2 near chance (`0.0197`).
  - 2000-step LR grid, seed `42`: best final accuracy was KDA `0.9655` at lr `1e-3`, GDN `0.9194` at lr `5e-4`, and Mamba2 `0.1440` at lr `5e-4`.
  - 3-seed KDA/GDN check at lr `1e-3`: KDA final accuracy `{42: 0.9655, 123: 0.9815, 7: 0.9492}`, GDN `{42: 0.9002, 123: 0.9815, 7: 0.9455}`. Mean final accuracy was KDA `0.9654` vs GDN `0.9424`.
  - This partially reproduces a KDA state-tracking quality advantage at the paper small-model shape. It does not reproduce a clean convergence-speed advantage because GDN reached high accuracy earlier in the curves. Artifacts: `artifacts/synthetic_stack_paper_shape_bf16_b4_s64_lr_sweep_2000steps_freegpu_summary.json`, `artifacts/synthetic_stack_paper_shape_bf16_b4_s64_lr1e3_2000step_3seed_summary.json`.

## Failures and Limitations

- Full 48B BF16 model reproduction is not feasible on one 32GB RTX 5090. Official vLLM docs assume 4 or 8 GPUs for the released checkpoint at large context.
- A root-owned vLLM process was using about 30.7 GiB of VRAM, leaving only a few hundred MiB during early runs; it was later stopped with user authorization.
- Synthetic palindrome training is partially reproduced:
  - KDA/GDN with fp16 and lr `1e-3` produced NaNs quickly.
  - Mamba2 OOMed in the early occupied-GPU smoke.
  - A tiny no-short-conv bf16 run stayed finite but did not learn, and it intentionally omits short convolution, which the paper says is important.
  - Tiny short-conv bf16 runs are stable but too small/noisy to show the paper's KDA advantage; the two-seed easy palindrome LR grid favored GDN on the best mean score.
  - The early stack probe used 16 stacks, not the paper's 64 stacks. The later free-GPU 64-stack run is closer to paper shape and gives a KDA final-quality edge over three seeds.
  - The free-GPU paper-shape palindrome run shows robust KDA learning across three seeds, but GDN beats KDA on two of those seeds; the result is not a clean KDA-over-GDN reproduction.
  - The free-GPU paper-shape MQAR run was initially negative for KDA. Generator validation fixed a major mismatch with Zoology MQAR, and the later source-initialization audit fixed the local KDA wrapper mismatch. KDA now solves the hard high-vocab Zoology curriculum eval slice when initialized like the full FLA KDA model. Remaining caveats: this is still a single-seed local reproduction, and the baseline comparison is sensitive to whether GDN uses default layer init or source-style full-model init.
  - The free-GPU paper-shape 64-stack run favors KDA on final accuracy, but GDN converges faster early, so it is not a clean reproduction of a KDA convergence-speed advantage.
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
  - `scripts/summarize_synthetic_runs.py`

## Sources Used

Separate official sources from third-party sources and cite URLs/commits used.

- Official paper: https://arxiv.org/abs/2510.26692
- Official repo: https://github.com/MoonshotAI/Kimi-Linear at `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`
- Official Instruct model: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct at `e1df551a447157d4658b573f9a695d57658590e9`
- Official Base model: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base at `3b171c17bfc4ee348599b6781a2ca8715c21c8dc`
- Official FLA KDA implementation: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda; local clone observed at `1c403c3a82896ca0dd2ef8a952a85e3bdbffc941`
- MQAR task reference: https://github.com/HazyResearch/zoology, inspected `zoology/data/multiquery_ar.py` and `zoology/experiments/paper_configs/arxiv24_based_figure3/configs.py`
- vLLM Kimi-Linear recipe: https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-Linear.html

## Next Experiments

- Diagnose or replace the Mamba2 20k baseline path; 2k works, but 20k hangs before the first eval record.
- Confirm the source-initialized MQAR result across seeds, and decide whether to compare against GDN under source-style full-model init, default standalone layer init, or both.
- Decide whether the current palindrome plus 64-stack evidence is sufficient to start UI design, or whether MQAR needs to be fixed first.
- Rerun backward operator benchmarks at H=16/D=128 for 2k-64k lengths with the GPU free.
- Build the UI only after the synthetic learning result is credible.
