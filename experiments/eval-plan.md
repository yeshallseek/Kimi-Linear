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
```

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
