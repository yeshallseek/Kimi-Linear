export const projectMeta = {
  branch: "reproduce/kimi-linear-source-review",
  commit: "5a8d5c6",
  gpu: "RTX 5090",
  environment: "conda kimi-linear, torch 2.9.0+cu128, FLA 0.4.0",
};

export const headlineStats = [
  {
    label: "KDA kernel max error",
    value: "0.0039",
    detail: "against naive recurrent reference",
  },
  {
    label: "KDA vs DPLR forward",
    value: "1.86x",
    detail: "H=16 D=128 T=4096",
  },
  {
    label: "MQAR step 400",
    value: "0.863",
    detail: "KDA mean, recurrent init baseline",
  },
  {
    label: "MQAR final mean",
    value: "0.993",
    detail: "KDA over 3 seeds",
  },
];

export const taskSummaries = [
  {
    id: "mqar",
    label: "Zoology MQAR",
    status: "Reproduced with source recurrent init",
    primaryMetric: "0.993",
    secondaryMetric: "0.989",
    metricLabel: "mean final accuracy",
    kdaNote: "KDA learns earlier; step-400 mean 0.863",
    gdnNote: "GDN solves by 2k; step-400 mean 0.141",
    artifact: "artifacts/synthetic_mqar_zoology_recurrent_init_fair_baseline_3seed_summary.json",
  },
  {
    id: "palindrome",
    label: "Palindrome",
    status: "Robust KDA, seed-sensitive GDN",
    primaryMetric: "0.929",
    secondaryMetric: "0.670",
    metricLabel: "mean final accuracy",
    kdaNote: "KDA learned on all three seeds",
    gdnNote: "GDN failed on seed 42, solved seeds 123 and 7",
    artifact: "artifacts/synthetic_palindrome_paper_shape_bf16_b4_lr1e3_20000step_3seed_summary.json",
  },
  {
    id: "stack",
    label: "64-stack tracking",
    status: "KDA final-quality edge",
    primaryMetric: "0.965",
    secondaryMetric: "0.942",
    metricLabel: "mean final accuracy",
    kdaNote: "Higher on two seeds and tied on one",
    gdnNote: "Converged earlier in several curves",
    artifact: "artifacts/synthetic_stack_paper_shape_bf16_b4_s64_lr1e3_2000step_3seed_summary.json",
  },
];

export const mqarCurve = [
  { step: 400, kda: 0.8634440104166666, gdn: 0.14056396484375 },
  { step: 600, kda: 0.96319580078125, gdn: 0.8677164713541666 },
  { step: 1000, kda: 0.9872233072916666, gdn: 0.96246337890625 },
  { step: 2000, kda: 0.9931844075520834, gdn: 0.9892781575520834 },
];

export const ablationRows = [
  {
    label: "KDA source recurrent",
    scope: "recurrent",
    model: "kda",
    accuracy: 0.9695,
    steps: 1000,
    note: "Only A_log and dt_bias source init",
  },
  {
    label: "KDA source full",
    scope: "full",
    model: "kda",
    accuracy: 0.9988,
    steps: 2000,
    note: "Full source init plus no-decay groups",
  },
  {
    label: "KDA source weights",
    scope: "weights",
    model: "kda",
    accuracy: 0.0005,
    steps: 1000,
    note: "Projection/conv normal init only",
  },
  {
    label: "KDA no recurrent fix",
    scope: "partial",
    model: "kda",
    accuracy: 0.0014,
    steps: 1000,
    note: "No-decay groups without dt init",
  },
  {
    label: "GDN default layer",
    scope: "default",
    model: "gdn",
    accuracy: 0.9838,
    steps: 2000,
    note: "Standalone GDN layer init",
  },
];

export const operatorRows = [
  { length: 1024, kdaMs: 0.1595, dplrMs: 0.1638, speedup: 1.03, mode: "forward" },
  { length: 2048, kdaMs: 0.2212, dplrMs: 0.3912, speedup: 1.77, mode: "forward" },
  { length: 4096, kdaMs: 0.4619, dplrMs: 0.8607, speedup: 1.86, mode: "forward" },
  { length: 4096, kdaMs: 0.5037, dplrMs: 0.8847, speedup: 1.76, mode: "backward reduced" },
];

export const seedResults = {
  mqar: [
    { seed: 42, kda: 0.9827, gdn: 0.9875 },
    { seed: 123, kda: 0.9977, gdn: 0.9937 },
    { seed: 7, kda: 0.9991, gdn: 0.9867 },
  ],
  palindrome: [
    { seed: 42, kda: 0.9237, gdn: 0.0389 },
    { seed: 123, kda: 0.8981, gdn: 0.9808 },
    { seed: 7, kda: 0.9641, gdn: 0.9906 },
  ],
  stack: [
    { seed: 42, kda: 0.9655, gdn: 0.9002 },
    { seed: 123, kda: 0.9815, gdn: 0.9815 },
    { seed: 7, kda: 0.9492, gdn: 0.9455 },
  ],
};

export const commandDefaults = {
  task: "mqar",
  models: "kda,gdn",
  seed: 42,
  steps: 2000,
  batchSize: 128,
  lr: 0.001,
  sourceInitScope: "recurrent",
  sourceParamGroups: true,
  seqLen: 256,
  vocabSize: 8192,
  numPairs: 64,
};

export const artifacts = [
  {
    name: "Final report",
    path: "reports/final-report.md",
    kind: "report",
  },
  {
    name: "Run ledger",
    path: "experiments/runs.jsonl",
    kind: "ledger",
  },
  {
    name: "MQAR fair baseline",
    path: "artifacts/synthetic_mqar_zoology_recurrent_init_fair_baseline_3seed_summary.json",
    kind: "summary",
  },
  {
    name: "MQAR step summary",
    path: "artifacts/synthetic_mqar_zoology_recurrent_init_fair_baseline_step_summary.json",
    kind: "summary",
  },
  {
    name: "Source-init ablation",
    path: "artifacts/synthetic_mqar_zoology_source_init_ablation_summary.json",
    kind: "summary",
  },
];
