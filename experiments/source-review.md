# Source Review: Kimi Linear reproduction

Mode: reproduction
Initial source: https://github.com/MoonshotAI/Kimi-Linear

## Source Map

| Priority | Source | Version/Commit | Why It Matters |
| --- | --- | --- | --- |
| Official | https://arxiv.org/abs/2510.26692 and local `tech_report.pdf` | arXiv v2; local PDF created 2025-10-31; 28 pages | Primary technical report for KDA recurrence, chunk algorithm, 3:1 hybrid architecture, synthetic tasks, scaling, and benchmark claims. |
| Official | https://github.com/MoonshotAI/Kimi-Linear | `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3` | Official project README and released report assets. This repo does not contain runnable training/eval code. |
| Official model | https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct | `e1df551a447157d4658b573f9a695d57658590e9`, last modified 2025-12-16 | Released instruction checkpoint and remote Transformers implementation. Config confirms 49B params, 1M max length, 20 KDA layers, 7 full-attention layers, 256 experts, 8 experts/token. |
| Official model | https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base | `3b171c17bfc4ee348599b6781a2ca8715c21c8dc`, last modified 2026-01-30 | Released base checkpoint for reproduction of base-model behavior, if enough GPU memory is available. |
| Official kernel | https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda | local `/home/ye/ml-experiments/flash-linear-attention` at `1c403c3a82896ca0dd2ef8a952a85e3bdbffc941`; upstream HEAD observed `abfa403de2146b9a2ab762a603f8fdb61cc3c166` | KDA implementation, tests, and `benchmarks/ops/benchmark_kda.py`. The installed env has `flash-linear-attention==0.4.0` and `fla-core==0.4.0`. |
| Maintainer docs | https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-Linear.html | vLLM recipe dated 2026-04-28 | Current serving guidance; warns to avoid vLLM 0.12.0 and assumes 4-GPU or 8-GPU tensor parallel for the full checkpoint. |
| Local prior attempt | `/home/ye/ml-experiments/Kimi-Linear` | branch `play` at `1a4ebd9058878a3f053d8b06cd55ebd2ca68355e`; dirty: untracked `run_inference.py` | Existing fork with user/local attempts using `cyankiwi/Kimi-Linear-48B-A3B-Instruct-AWQ-4bit`. Useful as prior context, but not used as the clean experiment repo. |
| Current experiment repo | `/home/ye/ml-experiments/kimi-linear-attention` | branch `reproduce/kimi-linear-source-review` from official `8c1d85e` | Clean lab notebook and reproduction harness location. |

## Source Priority

- Official paper/project/repo/model/data sources: arXiv tech report, MoonshotAI/Kimi-Linear, MoonshotAI Hugging Face Base/Instruct checkpoints, FLA KDA implementation.
- Maintainer or dependency docs: vLLM Kimi-Linear recipe; FLA tests and benchmarks in the local flash-linear-attention clone.
- Third-party reproductions, forks, blogs, or forum notes: `cyankiwi` AWQ model and the older local fork are useful only as single-GPU inference context. They are not primary evidence for the architecture claims.
- Unresolved candidate sources: the paper mentions vLLM implementation, but the official Kimi-Linear GitHub repo only documents vLLM usage; concrete vLLM support lives in vLLM releases and remote model code. Need verify with an actual vLLM smoke run later.

## Objective

Learn the architectural mechanism behind Kimi Linear before building any UI, then reproduce the most tractable findings locally:

1. Verify KDA kernel correctness against a naive recurrent implementation.
2. Reproduce the paper's operator-level speed direction: KDA should be materially faster than a general DPLR delta-rule formulation at useful sequence lengths.
3. Reproduce a scaled version of the synthetic-task result from Figure 4: KDA should learn copy/recall/state-tracking probes better or faster than GDN/Mamba2 under matched tiny-model settings.
4. Only after those pass, attempt full-model inference or long-context serving and then build a UI around the experiment controls/results.

## Key Claims or Features

- KDA extends Gated DeltaNet by replacing a coarse head-wise forget gate with channel-wise, diagonal gating over the recurrent state.
- KDA is a constrained DPLR transition where the low-rank factors are tied to the key, reducing unstable reciprocal-decay work and several matrix multiplications versus general DPLR.
- Kimi Linear interleaves KDA with full global MLA layers at a 3:1 ratio. The released 27-layer config uses 20 KDA layers plus 7 full-attention layers.
- Claimed results:
  - Synthetic probes: KDA achieves the best accuracy as sequence length grows from 256 to 2048 and converges faster than GDN on palindrome/MQAR. The paper uses 2 layers, 2 attention heads, head dimension 128, up to 20,000 steps, and an LR grid over `{5e-5, 1e-4, 5e-4, 1e-3}`.
  - Operator efficiency: KDA is roughly 2x faster than the general DPLR formulation up to 64k in the paper's kernel benchmark.
  - Full model: 48B total / 3B active MoE, trained with the same recipe as MLA/GDN-H baselines; Kimi Linear beats MLA/GDN-H on short-context, long-context, and RL-style evaluations.
  - Serving: KV cache use falls by up to 75%; decoding throughput rises up to 6x at 1M context due to fixed-size KDA state plus only periodic full attention.

## Repo Architecture

- `README.md`: official usage and claims; links to paper/model and recommends Python >=3.10, torch >=2.6, `fla-core >=0.4.0`.
- `tech_report.pdf`: paper bundled in repo.
- `figures/`: architecture and performance images only.
- No official training scripts, datasets, synthetic-task implementation, or benchmark runner are present in MoonshotAI/Kimi-Linear.
- `/home/ye/ml-experiments/flash-linear-attention/fla/ops/kda/`: KDA chunk, recurrent, gate, and naive implementations.
- `/home/ye/ml-experiments/flash-linear-attention/benchmarks/ops/benchmark_kda.py`: compares GDN, Comba, KDA, DPLR, and FlashAttention over sequence lengths.
- `/home/ye/ml-experiments/flash-linear-attention/tests/ops/test_kda.py`: correctness tests versus naive recurrent/chunk KDA.
- `/home/ye/ml-experiments/flash-linear-attention/fla/models/kda/`: tiny KDA language model config and implementation usable for synthetic probes.
- `/home/ye/ml-experiments/flash-linear-attention/fla/models/gated_deltanet/` and `fla/models/mamba2/`: local baselines for scaled synthetic reproduction.

## Execution Path

Smallest faithful command or code path to run first:

```bash
conda run -n kimi-linear python scripts/kda_kernel_smoke.py \
  --output artifacts/kda_kernel_smoke.json
```

Then, if GPU memory is available:

```bash
conda run -n kimi-linear python scripts/kda_operator_benchmark.py \
  --providers kda,dplr,gdn,attn \
  --lengths 256,512,1024,2048,4096 \
  --output artifacts/kda_operator_benchmark.jsonl
```

For the first learning probe:

```bash
conda run -n kimi-linear python scripts/synthetic_recall_probe.py \
  --task palindrome \
  --models kda,gdn,mamba2 \
  --seq-len 256 \
  --steps 500 \
  --output artifacts/synthetic_palindrome_256.jsonl
```

For a CPU-safe recurrence-level mechanism probe while GPU memory is constrained:

```bash
conda run -n kimi-linear python scripts/channel_gate_probe.py
```

To run the same mechanism probe at the paper's Figure 4 length range:

```bash
conda run -n kimi-linear python scripts/channel_gate_probe.py \
  --seq-lengths 256 512 1024 2048 \
  --output artifacts/channel_gate_probe_paper_lengths.json \
  --csv-output artifacts/channel_gate_probe_paper_lengths.csv
```

## Minimal Causal Experiment

- Simplest faithful experiment: run FLA KDA forward/backward against naive recurrent KDA on tiny random tensors, then benchmark KDA vs DPLR/GDN/FlashAttention for the same tensor shapes.
- Mechanism probe: construct a two-channel recurrence where one channel must preserve a long-lived signal and the other must forget noisy short-lived writes. KDA can assign different channel decays; scalar GDN cannot.
- First causal learning probe: train same-size tiny causal language models on a synthetic palindrome or MQAR task, changing only the sequence mixer (`KDAConfig`, `GatedDeltaNetConfig`, `Mamba2Config`).
- Proposed control runs: GDN controls for delta rule with scalar/head-wise gating; Mamba2 controls for multiplicative decay without delta-rule memory; full attention or FlashAttention controls operator throughput but not model quality.
- Effect being isolated: whether channel-wise gated delta memory improves selective retention/forgetting and whether the constrained DPLR/KDA formulation improves hardware efficiency.
- Success: KDA correctness matches naive implementation within FLA test tolerances; KDA operator median latency is lower than DPLR at the selected lengths; KDA reaches higher accuracy or the same accuracy in fewer steps than GDN/Mamba2 on at least one synthetic task.
- Falsification or caveat: KDA fails correctness on this GPU/env; KDA is not faster than DPLR under identical local shapes; synthetic training shows no accuracy/convergence advantage after matched seeds and learning-rate sweeps.
- Current synthetic status: memory-safe vocab-16, hidden-64 probes did not reproduce KDA's Figure 4 advantage and should be treated as constrained negative controls. After reclaiming the GPU, the paper-shape palindrome setup (2 layers, 2 heads, head_dim 128, hidden 256, seq_len 256, vocab 128, batch 4) showed robust KDA learning across seeds `42`, `123`, and `7` at 20,000 steps, with mean final accuracy `0.9286`. GDN was seed-sensitive: it failed on seed `42` (`0.0389`) but reached `0.9808` and `0.9906` on seeds `123` and `7`, giving mean final accuracy `0.6701`. This supports KDA robustness but does not reproduce a per-seed KDA win over GDN. The 20k Mamba2 baseline is unresolved because the Mamba2 long-run segment stayed GPU-active without producing a first eval record and was terminated; Mamba2 did learn modestly in the 2000-step LR grid.

## RTX 5090 Fit

- Hardware snapshot: NVIDIA GeForce RTX 5090, 32,607 MiB VRAM, driver 580.159.03. At snapshot time a root-owned `VLLM::EngineCore` process used about 30,674 MiB. The process was stopped with user authorization on 2026-05-24 00:53 PDT, restoring enough VRAM for paper-shape synthetic runs.
- Environment: conda `kimi-linear`, Python 3.11.14, torch 2.9.0+cu128, CUDA runtime 12.8, transformers 4.57.1, flash-linear-attention 0.4.0, fla-core 0.4.0, vLLM 0.11.1rc6 dev build.
- Expected VRAM:
  - KDA kernel smoke: small, likely <2 GiB plus kernel compile overhead.
  - Operator benchmark at 4k-64k, B=1/H=16/D=128: grows quickly; run short lengths first while GPU is occupied.
  - Tiny synthetic training: configurable; start with hidden_size 256, 2 layers, 2 heads, seq_len 256.
  - Full 48B BF16 checkpoint: not feasible on one 32GB GPU. Even AWQ 4-bit may be tight with KV/state/cache overhead.
- Precision: start with float16/bfloat16 GPU kernels; use float32 only for naive correctness where FLA tests do.
- Scaling changes from paper: no 1.4T/5.7T pretraining, no 48B matched baselines, no 128k/1M benchmark suite initially. The local reproduction targets mechanism-level claims first.

## Risks and Ambiguities

- Official project repo is mostly a report/model card, not a runnable reproduction package.
- The full benchmark data and training corpora are not public; exact reproduction of Tables 3-5 and RL Figure 6 is not locally feasible.
- vLLM docs assume 4 or 8 GPUs for full 1M context serving; current machine has one RTX 5090.
- The local FLA clone is behind upstream HEAD. Installed package is `0.4.0`; should record exact installed version for runs, and only update after a baseline run or if a bug blocks reproduction.
- The previously occupying root-owned vLLM process was stopped with user authorization. Future runs can reclaim the GPU for this experiment when needed, recording the action.
- The paper's synthetic setup gives model sizes and learning-rate grid but not full dataset-generation code. Our synthetic reproduction must clearly mark deviations.
- FLA KDA/GDN layers switch to fused recurrent mode at `q_len <= 64`, but training asserts that only chunk mode is supported. The local synthetic harness pads shorter generated tasks to length 65 so tiny training probes use the supported path.
- Third-party AWQ quantization may alter model behavior and is not evidence for the paper's architecture claims.

## Decision

Proceed with caveats. Start with official FLA kernel correctness and a scaled operator benchmark; then add synthetic task reproduction. Defer full-model inference and UI until these mechanism-level reproductions are credible.
