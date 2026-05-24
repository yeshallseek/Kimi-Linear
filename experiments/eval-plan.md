# Evaluation Plan

## Tier 0: Source-Grounded Smoke

Purpose: verify the local environment can execute the official FLA KDA path.

Commands:

```bash
conda run -n kimi-linear python scripts/kda_kernel_smoke.py \
  --output artifacts/kda_kernel_smoke.json
```

Success criteria:

- `chunk_kda` and `fused_recurrent_kda` match naive recurrent KDA on tiny tensors within recorded tolerances.
- The output JSON records package versions, GPU, tensor shape, max absolute errors, and runtime.

## Tier 1: Operator Claim

Purpose: reproduce the direction of the paper's KDA vs DPLR kernel-efficiency claim at smaller lengths.

Command:

```bash
conda run -n kimi-linear python scripts/kda_operator_benchmark.py \
  --providers kda,dplr,gdn,attn \
  --lengths 256,512,1024,2048,4096 \
  --output artifacts/kda_operator_benchmark.jsonl
```

Low-memory backward variant used while the GPU is occupied:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/kda_operator_benchmark.py \
  --providers kda,dplr --lengths 2048,4096 \
  --heads 8 --head-dim 64 --dtype float16 \
  --warmup 1 --rep 3 \
  --output artifacts/kda_operator_benchmark_backward_h8d64_2048_4096.jsonl
```

Success criteria:

- KDA runs successfully for all selected lengths.
- KDA median latency is lower than DPLR for at least the longer selected lengths.
- If FlashAttention or DPLR fails due to install or memory, the failure is recorded with traceback and the benchmark continues.

## Tier 1.5: Recurrence Mechanism Probe

Purpose: isolate the selective retention/forgetting mechanism without depending on GPU memory or full synthetic training stability.

Command:

```bash
conda run -n kimi-linear python scripts/channel_gate_probe.py
```

Paper-length variant:

```bash
conda run -n kimi-linear python scripts/channel_gate_probe.py \
  --seq-lengths 256 512 1024 2048 \
  --output artifacts/channel_gate_probe_paper_lengths.json \
  --csv-output artifacts/channel_gate_probe_paper_lengths.csv
```

Success criteria:

- Scalar GDN has one best decay for both channels and incurs nonzero error when a task requires both long retention and short forgetting.
- KDA can choose separate long-channel and short-channel decays and reduce the recurrence-level error to zero.
- Results are labeled as mechanism evidence, not a replacement for Tier 2 Figure 4 training reproduction.

## Tier 2: Synthetic Learning Probe

Purpose: reproduce a scaled version of Figure 4 before attempting any full-model or UI work.

Initial command:

```bash
conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task palindrome \
  --models kda,gdn,mamba2 \
  --seq-len 256 \
  --steps 500 \
  --output artifacts/synthetic_palindrome_256.jsonl
```

Constrained LR-grid command used while the GPU is occupied:

```bash
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

for seed in 42 123; do
  PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
    --task palindrome --models kda,gdn \
    --vocab-size 16 --seq-len 32 --steps 10000 --eval-every 1000 --eval-batches 8 \
    --batch-size 2 --hidden-size 64 --heads 1 --head-dim 64 \
    --mlp-ratio 1 --dtype bfloat16 --lr 1e-3 --seed "$seed" \
    --output "artifacts/synthetic_palindrome_easy_shortconv_bf16_b2_seed${seed}_lr1e_3_10000steps.jsonl"
done

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
```

Paper-shape feasibility smoke under the occupied-GPU constraint:

```bash
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
```

Observed result: KDA and GDN OOMed during backward; Mamba2 completed the 20-step smoke at chance accuracy. This occupied-GPU result was superseded by the free-GPU runs below.

Free-GPU paper-shape palindrome LR grid and 20k extension:

```bash
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
```

Observed result: the 20k extension reproduced the KDA-vs-GDN palindrome direction for seed `42` (`0.9237` KDA final accuracy vs `0.0389` GDN). Mamba2 learned modestly in the 2k grid but its 20k segment hung before the first eval record and needs a separate fix or replacement baseline command.

Multi-seed confirmation command:

```bash
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
```

Observed result: KDA learned reliably across all three seeds (`0.8981-0.9641`, mean `0.9286`), while GDN was bimodal (`0.0389`, `0.9808`, `0.9906`, mean `0.6701`). Treat this as evidence for KDA robustness on this harness, not a per-seed KDA win.

Free-GPU paper-shape MQAR commands:

```bash
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
```

Observed result: the MQAR generator produced actual length 249 with 63 query pairs. The 2000-step LR grid did not show a KDA learning signal: best final accuracy was KDA `0.0272`, GDN `0.0252`, and Mamba2 `0.0696`. The 20,000-step KDA/GDN extension at lr `1e-3` stayed negative for KDA (`0.0131` final accuracy, best `0.0202`) and only modestly above chance for GDN (`0.0917` final accuracy, best `0.0938`). Treat this as an unresolved/negative MQAR reproduction for the current harness.

Corrected Zoology-style MQAR audit and follow-up:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task mqar --mqar-layout zoology --models kda,gdn,mamba2 \
  --vocab-size 8192 --seq-len 256 --num-pairs 64 \
  --steps 20000 --eval-every 2000 --eval-batches 8 \
  --batch-size 4 --hidden-size 256 --heads 2 --head-dim 128 \
  --mamba-head-dim 128 --mamba-state-size 128 --mamba-expand 2 \
  --mlp-ratio 2 --dtype bfloat16 --lr 1e-3 --seed 42 \
  --output artifacts/synthetic_mqar_zoology_shape_bf16_b4_v8192_p64_lr1e_3_20000steps_freegpu.jsonl

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task mqar --mqar-layout zoology --tie-embeddings --init-std 0.02 --models kda,gdn \
  --vocab-size 8192 --seq-len 256 --num-pairs 64 \
  --steps 2000 --eval-every 200 --eval-batches 2 \
  --batch-size 64 --hidden-size 256 --heads 2 --head-dim 128 \
  --mlp-ratio 4 --dtype bfloat16 --lr 1e-3 --seed 42 \
  --output artifacts/synthetic_mqar_zoology_tied_init002_mlp4_bf16_b64_v8192_p64_lr1e_3_2000steps_freegpu.jsonl

PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task mqar --mqar-layout zoology --tie-embeddings --init-std 0.02 --models kda,gdn \
  --vocab-size 256 --seq-len 64 --num-pairs 4 \
  --steps 2000 --eval-every 200 --eval-batches 8 \
  --batch-size 64 --hidden-size 256 --heads 2 --head-dim 128 \
  --mlp-ratio 4 --dtype bfloat16 --lr 1e-3 --seed 42 \
  --output artifacts/synthetic_mqar_zoology_tied_init002_mlp4_bf16_b64_v256_p4_lr1e_3_2000steps_freegpu.jsonl
```

Observed result: the original local MQAR layout was not faithful to Zoology. The corrected `--mqar-layout zoology` path uses vocab 8192, upper-half value tokens, power-law query gaps, random non-query fillers, and one query per key. At paper task shape (`vocab=8192`, `seq_len=256`, `num_pairs=64`), KDA/GDN/Mamba2 stayed at the value-vocabulary baseline through 20,000 steps: final accuracy was KDA `0.00049`, GDN `0.0`, and Mamba2 `0.0`. Source-style tied embeddings, `std=0.02` init, MLP ratio 4, higher LRs, batch 64, and `num_pairs=16` still did not produce a high-vocab retrieval signal by 2000 steps. The tiny positive control (`vocab=256`, `seq_len=64`, `num_pairs=4`, batch 64) did learn: GDN reached `0.9971` final accuracy and KDA reached `0.2612`. Treat MQAR as unresolved rather than reproduced.

Zoology train-mix curriculum diagnostic:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task mqar --mqar-layout zoology --mqar-train-curriculum zoology_figure3 \
  --tie-embeddings --init-std 0.02 --models kda,gdn \
  --vocab-size 8192 --seq-len 256 --num-pairs 64 \
  --steps 2000 --eval-every 200 --eval-batches 2 \
  --batch-size 128 --hidden-size 256 --heads 2 --head-dim 128 \
  --mlp-ratio 4 --dtype bfloat16 --lr 1e-3 --seed 42 \
  --output artifacts/synthetic_mqar_zoology_curriculum_tied_init002_mlp4_bf16_b128_v8192_evalp64_lr1e_3_2000steps_freegpu.jsonl
```

Observed result: the curriculum matches Zoology's train slice weights but evaluates on the hard `seq_len=256, num_pairs=64` target. This made the high-vocab task learnable for GDN: final eval accuracy was `0.9838` at 2000 steps. KDA stayed near chance in the same run (`0.0016` final, best `0.0027`). A KDA-only LR sweep at `1e-3`, `3.16e-3`, `1e-2`, and `3.16e-2` with source-style weight decay `0.1` also stayed near chance. This was a KDA non-reproduction until the source-initialization audit below found the wrapper mismatch. Artifact: `artifacts/synthetic_mqar_zoology_curriculum_diagnostics_summary.json`.

KDA source-initialization diagnostic:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task mqar --mqar-layout zoology --mqar-train-curriculum zoology_figure3 \
  --tie-embeddings --init-std 0.02 --source-init --source-param-groups \
  --models kda,gdn \
  --vocab-size 8192 --seq-len 256 --num-pairs 64 \
  --steps 2000 --eval-every 200 --eval-batches 2 \
  --batch-size 128 --hidden-size 256 --heads 2 --head-dim 128 \
  --mlp-ratio 4 --dtype bfloat16 --lr 1e-3 --seed 42 \
  --output artifacts/synthetic_mqar_zoology_curriculum_sourceinit_sourcewd_bf16_b128_v8192_evalp64_lr1e_3_2000steps_freegpu.jsonl
```

Observed result: the KDA failure above was traced to a wrapper-level initialization mismatch. FLA's full `KDAPreTrainedModel._init_weights` initializes KDA `dt_bias` from log-uniform time constants, but the bare `KimiDeltaAttention` layer starts with `dt_bias=0`. Adding source-style recurrent initialization makes KDA solve the hard eval slice: full source init reached `0.9988` final / `0.9993` best accuracy at 2000 steps. Ablations show the effect is recurrent-gate-specific: `--source-init-scope recurrent` reached `0.9695` final accuracy at 1000 steps, while `--source-init-scope weights` stayed at `0.00049`, and source-style no-decay groups without recurrent init stayed at `0.0014`. Artifact: `artifacts/synthetic_mqar_zoology_source_init_ablation_summary.json`.

Free-GPU paper-shape 64-stack commands:

```bash
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

Observed result: actual sequence length was 255. In the seed-42 LR grid, KDA's best final accuracy was `0.9655` at lr `1e-3`, GDN's best final accuracy was `0.9194` at lr `5e-4`, and Mamba2's best final accuracy was `0.1440` at lr `5e-4`. At lr `1e-3` over seeds `42`, `123`, and `7`, KDA mean final accuracy was `0.9654` vs GDN `0.9424`; KDA was higher on seeds `42` and `7`, and tied GDN on seed `123`. GDN reached high accuracy earlier in the curves, so the current evidence is a KDA final-quality edge rather than a convergence-speed win.

Controls:

- Same sequence length, batch size, optimizer, seed, hidden size, layer count, and head count.
- Only the mixer/model family changes.
- Repeat with at least two seeds before treating a result as meaningful.
- For KDA/GDN training, pad generated sequences shorter than 65 tokens so FLA uses chunk mode; the recurrent mode path is inference-only for these layers.

## Tier 3: Full-Model Inference

Purpose: test released Kimi Linear behavior after mechanism-level reproduction.

Constraints:

- BF16 48B checkpoint is not expected to fit on one 32GB RTX 5090.
- Official vLLM recipe assumes 4 or 8 GPUs for 1M context.
- Third-party AWQ checkpoints can be tried later, but results must be labeled as quantized-inference checks, not paper reproduction.

## Tier 4: UI

Start only after Tier 0-2 results exist. The UI should expose the mechanism learned from reproduction: task generator, model family selector, sequence length, seed, training curve, operator latency table, and result notebook links.
