import {
  Activity,
  BarChart3,
  Clipboard,
  Database,
  ExternalLink,
  Gauge,
  GitBranch,
  MessageSquare,
  Play,
  RefreshCw,
  Send,
  Server,
  SlidersHorizontal,
} from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import {
  ablationRows,
  artifacts,
  commandDefaults,
  headlineStats,
  mqarCurve,
  operatorRows,
  projectMeta,
  seedResults,
  taskSummaries,
} from "./data";

const modelColors = {
  kda: "#059669",
  gdn: "#2877c9",
};

const taskOptions = [
  { value: "mqar", label: "Zoology MQAR" },
  { value: "palindrome", label: "Palindrome" },
  { value: "stack", label: "64-stack tracking" },
];

const initOptions = [
  { value: "recurrent", label: "Recurrent" },
  { value: "full", label: "Full" },
  { value: "none", label: "Legacy" },
];

const apiBase =
  import.meta.env.VITE_KIMI_API_BASE ||
  `${window.location.protocol}//${window.location.hostname || "127.0.0.1"}:5174`;

const viewOptions = [
  { value: "chat", label: "Pretrained chat" },
  { value: "reproduction", label: "Reproduction" },
];

const defaultSystemPrompt =
  "You are Kimi Linear 48B-A3B Instruct running locally through a GGUF quant. Answer directly and note uncertainty.";

function formatAccuracy(value) {
  return value.toFixed(3);
}

function buildCommand(config) {
  const taskArgs =
    config.task === "mqar"
      ? [
          "--mqar-layout zoology",
          "--mqar-train-curriculum zoology_figure3",
          `--num-pairs ${config.numPairs}`,
          "--tie-embeddings",
          "--init-std 0.02",
        ]
      : config.task === "stack"
        ? ["--num-stacks 64"]
        : [];

  const sourceArgs =
    config.sourceInitScope === "none"
      ? []
      : [
          "--source-init",
          `--source-init-scope ${config.sourceInitScope}`,
          config.sourceParamGroups ? "--source-param-groups" : "",
        ].filter(Boolean);

  return [
    "conda run -n kimi-linear python scripts/synthetic_recall_probe.py",
    `  --task ${config.task}`,
    `  --models ${config.models}`,
    `  --dtype bfloat16`,
    `  --seed ${config.seed}`,
    `  --vocab-size ${config.vocabSize}`,
    `  --seq-len ${config.seqLen}`,
    ...taskArgs.map((arg) => `  ${arg}`),
    `  --batch-size ${config.batchSize}`,
    `  --steps ${config.steps}`,
    `  --eval-every ${config.steps >= 2000 ? 200 : 50}`,
    `  --eval-batches 2`,
    `  --lr ${config.lr}`,
    `  --weight-decay 0.01`,
    `  --layers 2`,
    `  --hidden-size 256`,
    `  --heads 2`,
    `  --head-dim 128`,
    `  --mlp-ratio 4`,
    ...sourceArgs.map((arg) => `  ${arg}`),
    `  --output artifacts/ui_${config.task}_${config.models.replace(",", "_")}_seed${config.seed}.jsonl`,
  ].join(" \\\n");
}

function Header({ view, setView }) {
  return (
    <header className="border-b border-zinc-950/10 bg-white/82 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
        <div className="min-w-0">
          <p className="font-mono text-base text-zinc-600 sm:text-sm">Kimi Linear model lab</p>
          <h1 className="mt-1 max-w-[18ch] text-2xl font-semibold tracking-tight text-balance text-zinc-950 sm:text-3xl">
            Local Kimi console
          </h1>
        </div>
        <div className="flex min-w-0 flex-wrap items-center gap-2 text-base text-zinc-700 sm:text-sm">
          <SegmentControl value={view} onChange={setView} options={viewOptions} ariaLabel="View" />
          <span className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-zinc-950/10 bg-white px-2.5 py-1.5">
            <GitBranch className="size-4 shrink-0 stroke-zinc-500" />
            <span className="truncate">{projectMeta.branch}</span>
          </span>
          <span className="rounded-md border border-zinc-950/10 bg-white px-2.5 py-1.5 font-mono">
            {projectMeta.commit}
          </span>
          <span className="rounded-md border border-signal-600/20 bg-signal-50 px-2.5 py-1.5 text-signal-700">
            {projectMeta.gpu}
          </span>
        </div>
      </div>
    </header>
  );
}

function StatGrid() {
  return (
    <section className="@container border-b border-zinc-950/10 bg-white/72">
      <div className="mx-auto grid max-w-7xl grid-cols-1 divide-y divide-zinc-950/10 px-4 sm:grid-cols-2 sm:divide-x sm:divide-y-0 sm:px-6 lg:grid-cols-4 lg:px-8 [&>*:first-child]:pl-0 [&>*:last-child]:pr-0">
        {headlineStats.map((stat) => (
          <div key={stat.label} className="py-5 sm:px-5">
            <p className="truncate text-base text-zinc-600 sm:text-sm">{stat.label}</p>
            <p className="mt-1 text-3xl font-semibold tracking-tight text-zinc-950">{stat.value}</p>
            <p className="mt-1 truncate text-base text-zinc-500 sm:text-sm">{stat.detail}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function SegmentControl({ value, onChange, options, ariaLabel }) {
  return (
    <div
      className="flex w-full flex-wrap gap-1 rounded-lg border border-zinc-950/10 bg-white p-1 sm:inline-flex sm:w-auto"
      aria-label={ariaLabel}
    >
      {options.map((option) => {
        const active = value === option.value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={[
              "min-w-0 flex-1 rounded-md px-3 py-2 text-base font-medium sm:flex-none sm:text-sm",
              active ? "bg-zinc-950 text-white" : "text-zinc-700",
            ].join(" ")}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

function LineChart({ points }) {
  const width = 640;
  const height = 260;
  const padding = { top: 20, right: 24, bottom: 34, left: 42 };
  const xMin = Math.min(...points.map((point) => point.step));
  const xMax = Math.max(...points.map((point) => point.step));
  const xScale = (step) =>
    padding.left + ((step - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);
  const yScale = (value) => padding.top + (1 - value) * (height - padding.top - padding.bottom);
  const lineFor = (model) => points.map((point) => `${xScale(point.step)},${yScale(point[model])}`).join(" ");

  return (
    <div className="@container">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img" aria-label="MQAR accuracy by step">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line
              x1={padding.left}
              x2={width - padding.right}
              y1={yScale(tick)}
              y2={yScale(tick)}
              stroke="rgba(24,24,27,0.08)"
            />
            <text x={12} y={yScale(tick) + 5} className="fill-zinc-500 text-sm">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        {points.map((point) => (
          <g key={point.step}>
            <line
              x1={xScale(point.step)}
              x2={xScale(point.step)}
              y1={padding.top}
              y2={height - padding.bottom}
              stroke="rgba(24,24,27,0.05)"
            />
            <text x={xScale(point.step) - 16} y={height - 9} className="fill-zinc-500 text-sm">
              {point.step}
            </text>
          </g>
        ))}
        <polyline fill="none" stroke={modelColors.kda} strokeWidth="4" strokeLinecap="round" points={lineFor("kda")} />
        <polyline fill="none" stroke={modelColors.gdn} strokeWidth="4" strokeLinecap="round" points={lineFor("gdn")} />
        {points.map((point) => (
          <g key={`dots-${point.step}`}>
            <circle cx={xScale(point.step)} cy={yScale(point.kda)} r="5" fill={modelColors.kda} />
            <circle cx={xScale(point.step)} cy={yScale(point.gdn)} r="5" fill={modelColors.gdn} />
          </g>
        ))}
      </svg>
    </div>
  );
}

function MechanismPanel() {
  const kdaBars = [0.96, 0.82, 0.64, 0.38, 0.18, 0.74, 0.52, 0.91];
  return (
    <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">Recurrent gate shape</h2>
          <p className="mt-2 max-w-[58ch] text-base text-pretty text-zinc-600 sm:text-sm">
            KDA's source-initialized time constants let channels specialize before the synthetic lookup task saturates.
          </p>
        </div>
        <Activity className="size-4 shrink-0 stroke-signal-600" />
      </div>
      <div className="mt-6 grid gap-5 md:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-lg border border-zinc-950/10 bg-zinc-50 p-4">
          <div className="mb-3 flex items-center justify-between text-base sm:text-sm">
            <span className="font-medium text-zinc-800">KDA channel gates</span>
            <span className="font-mono text-zinc-500">dt_bias source init</span>
          </div>
          <div className="grid grid-cols-8 gap-2">
            {kdaBars.map((height, index) => (
              <div key={index} className="flex h-28 items-end rounded-md bg-white ring-1 ring-zinc-950/10">
                <div
                  className="w-full rounded-b-md rounded-t-[3px] bg-signal-600"
                  style={{ height: `${Math.max(10, height * 100)}%` }}
                />
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-lg border border-zinc-950/10 bg-zinc-50 p-4">
          <div className="mb-3 flex items-center justify-between text-base sm:text-sm">
            <span className="font-medium text-zinc-800">GDN head gate</span>
            <span className="font-mono text-zinc-500">shared decay</span>
          </div>
          <div className="flex h-28 items-end rounded-md bg-white ring-1 ring-zinc-950/10">
            <div className="w-full rounded-b-md rounded-t-[3px] bg-cobalt-600" style={{ height: "67%" }} />
          </div>
        </div>
      </div>
    </section>
  );
}

function TaskPanel({ selectedTask, setSelectedTask }) {
  const selected = taskSummaries.find((task) => task.id === selectedTask);
  const seeds = seedResults[selectedTask];

  return (
    <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">Task outcomes</h2>
          <p className="mt-2 max-w-[62ch] text-base text-pretty text-zinc-600 sm:text-sm">
            {selected.status}
          </p>
        </div>
        <SegmentControl value={selectedTask} onChange={setSelectedTask} options={taskOptions} ariaLabel="Task" />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <div className="min-w-0">
          <div className="grid grid-cols-2 gap-3">
            <MetricPill label="KDA" value={selected.primaryMetric} color="signal" />
            <MetricPill label="GDN" value={selected.secondaryMetric} color="cobalt" />
          </div>
          <div className="mt-5 space-y-3 text-base text-zinc-700 sm:text-sm">
            <p>{selected.kdaNote}</p>
            <p>{selected.gdnNote}</p>
            <p className="break-all font-mono text-zinc-500">{selected.artifact}</p>
          </div>
        </div>
        <div className="-mx-4 -my-2 overflow-x-auto whitespace-nowrap sm:-mx-5">
          <div className="inline-block min-w-full px-4 py-2 align-middle sm:px-5">
            <table className="w-full">
              <thead>
                <tr className="border-b border-zinc-950/10 text-left text-base text-zinc-500 sm:text-sm">
                  <th className="whitespace-nowrap py-2 pr-4 font-medium">Seed</th>
                  <th className="whitespace-nowrap px-4 py-2 font-medium">KDA</th>
                  <th className="whitespace-nowrap px-4 py-2 font-medium">GDN</th>
                  <th className="whitespace-nowrap py-2 pl-4 font-medium">Delta</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-950/10 text-base sm:text-sm">
                {seeds.map((row) => (
                  <tr key={row.seed}>
                    <td className="py-3 pr-4 font-mono text-zinc-700">{row.seed}</td>
                    <td className="px-4 py-3 text-zinc-900">{formatAccuracy(row.kda)}</td>
                    <td className="px-4 py-3 text-zinc-900">{formatAccuracy(row.gdn)}</td>
                    <td className="py-3 pl-4 font-mono text-zinc-600">{(row.kda - row.gdn).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}

function MetricPill({ label, value, color }) {
  const classes =
    color === "signal"
      ? "border-signal-600/20 bg-signal-50 text-signal-700"
      : "border-cobalt-600/20 bg-cobalt-50 text-cobalt-700";
  return (
    <div className={`rounded-lg border p-4 ${classes}`}>
      <p className="text-base sm:text-sm">{label}</p>
      <p className="mt-1 text-3xl font-semibold tracking-tight">{value}</p>
    </div>
  );
}

function CurvePanel() {
  return (
    <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">MQAR convergence</h2>
          <p className="mt-2 max-w-[64ch] text-base text-pretty text-zinc-600 sm:text-sm">
            Recurrent-only source initialization, 3-seed mean, hard 256-token / 64-pair eval slice.
          </p>
        </div>
        <div className="flex gap-4 text-base sm:text-sm">
          <span className="inline-flex items-center gap-2 text-zinc-700">
            <span className="size-3 rounded-sm bg-signal-600" />
            KDA
          </span>
          <span className="inline-flex items-center gap-2 text-zinc-700">
            <span className="size-3 rounded-sm bg-cobalt-600" />
            GDN
          </span>
        </div>
      </div>
      <div className="mt-5">
        <LineChart points={mqarCurve} />
      </div>
    </section>
  );
}

function AblationPanel() {
  return (
    <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">Initialization ablation</h2>
          <p className="mt-2 max-w-[66ch] text-base text-pretty text-zinc-600 sm:text-sm">
            The reproduction turns on the recurrent gate parameters, not the projection weight distribution.
          </p>
        </div>
        <Gauge className="size-4 shrink-0 stroke-ember-600" />
      </div>
      <div className="mt-5 space-y-4">
        {ablationRows.map((row) => (
          <div key={row.label}>
            <div className="mb-1 flex items-baseline justify-between gap-3 text-base sm:text-sm">
              <span className="font-medium text-zinc-800">{row.label}</span>
              <span className="font-mono text-zinc-600">
                {row.accuracy < 0.01 ? row.accuracy.toFixed(4) : row.accuracy.toFixed(3)}
              </span>
            </div>
            <div className="h-2 rounded-sm bg-zinc-100">
              <div
                className={[
                  "h-2 rounded-sm",
                  row.model === "kda" ? "bg-signal-600" : "bg-cobalt-600",
                ].join(" ")}
                style={{ width: `${Math.max(0.8, row.accuracy * 100)}%` }}
              />
            </div>
            <p className="mt-1 text-base text-zinc-500 sm:text-sm">{row.note}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function OperatorPanel() {
  return (
    <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">Operator speed</h2>
          <p className="mt-2 max-w-[60ch] text-base text-pretty text-zinc-600 sm:text-sm">
            Local RTX 5090 reduced-shape checks reproduce the KDA-over-DPLR speed direction at useful lengths.
          </p>
        </div>
        <BarChart3 className="size-4 shrink-0 stroke-cobalt-600" />
      </div>
      <div className="mt-5 -mx-5 -my-2 overflow-x-auto whitespace-nowrap">
        <div className="inline-block min-w-full px-5 py-2 align-middle">
          <table className="w-full">
            <thead>
              <tr className="border-b border-zinc-950/10 text-left text-base text-zinc-500 sm:text-sm">
                <th className="whitespace-nowrap py-2 pr-4 font-medium">Length</th>
                <th className="whitespace-nowrap px-4 py-2 font-medium">KDA ms</th>
                <th className="whitespace-nowrap px-4 py-2 font-medium">DPLR ms</th>
                <th className="whitespace-nowrap px-4 py-2 font-medium">Speedup</th>
                <th className="whitespace-nowrap py-2 pl-4 font-medium">Mode</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-950/10 text-base sm:text-sm">
              {operatorRows.map((row) => (
                <tr key={`${row.length}-${row.mode}`}>
                  <td className="py-3 pr-4 font-mono text-zinc-700">{row.length}</td>
                  <td className="px-4 py-3 text-zinc-900">{row.kdaMs.toFixed(4)}</td>
                  <td className="px-4 py-3 text-zinc-900">{row.dplrMs.toFixed(4)}</td>
                  <td className="px-4 py-3 font-medium text-signal-700">{row.speedup.toFixed(2)}x</td>
                  <td className="py-3 pl-4 text-zinc-600">{row.mode}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function CommandPanel() {
  const [config, setConfig] = useState(commandDefaults);
  const [copied, setCopied] = useState(false);
  const command = useMemo(() => buildCommand(config), [config]);

  const update = (key, value) => {
    setConfig((current) => ({ ...current, [key]: value }));
    setCopied(false);
  };

  const copyCommand = async () => {
    await navigator.clipboard.writeText(command);
    setCopied(true);
  };

  return (
    <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">Experiment command</h2>
          <p className="mt-2 max-w-[62ch] text-base text-pretty text-zinc-600 sm:text-sm">
            The command mirrors the validated GPU path and writes a JSONL artifact into `artifacts/`.
          </p>
        </div>
        <button
          type="button"
          onClick={copyCommand}
          className="inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-signal-600 py-2 pr-3 pl-2 text-sm font-medium text-white ring-1 ring-signal-600 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal-600"
        >
          <Clipboard className="size-4 shrink-0 stroke-white" />
          {copied ? "Copied" : "Copy command"}
        </button>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-4">
        <Control label="Task" id="task">
          <select
            id="task"
            name="task"
            value={config.task}
            onChange={(event) => update("task", event.target.value)}
            className="w-full rounded-lg bg-white px-3 py-2.5 text-base ring-1 ring-zinc-950/10 sm:py-2 sm:text-sm"
          >
            {taskOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </Control>
        <Control label="Models" id="models">
          <select
            id="models"
            name="models"
            value={config.models}
            onChange={(event) => update("models", event.target.value)}
            className="w-full rounded-lg bg-white px-3 py-2.5 text-base ring-1 ring-zinc-950/10 sm:py-2 sm:text-sm"
          >
            <option value="kda,gdn">KDA, GDN</option>
            <option value="kda">KDA</option>
            <option value="gdn">GDN</option>
            <option value="mamba2">Mamba2</option>
          </select>
        </Control>
        <Control label="Seed" id="seed">
          <input
            id="seed"
            name="seed"
            type="number"
            value={config.seed}
            onChange={(event) => update("seed", Number(event.target.value))}
            className="w-full rounded-lg bg-white px-3 py-2.5 text-base ring-1 ring-zinc-950/10 sm:py-2 sm:text-sm"
          />
        </Control>
        <Control label="Init scope" id="init-scope">
          <select
            id="init-scope"
            name="initScope"
            value={config.sourceInitScope}
            onChange={(event) => update("sourceInitScope", event.target.value)}
            className="w-full rounded-lg bg-white px-3 py-2.5 text-base ring-1 ring-zinc-950/10 sm:py-2 sm:text-sm"
          >
            {initOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </Control>
        <Control label="Steps" id="steps">
          <input
            id="steps"
            name="steps"
            type="number"
            min="20"
            step="100"
            value={config.steps}
            onChange={(event) => update("steps", Number(event.target.value))}
            className="w-full rounded-lg bg-white px-3 py-2.5 text-base ring-1 ring-zinc-950/10 sm:py-2 sm:text-sm"
          />
        </Control>
        <Control label="Batch" id="batch">
          <input
            id="batch"
            name="batch"
            type="number"
            min="1"
            value={config.batchSize}
            onChange={(event) => update("batchSize", Number(event.target.value))}
            className="w-full rounded-lg bg-white px-3 py-2.5 text-base ring-1 ring-zinc-950/10 sm:py-2 sm:text-sm"
          />
        </Control>
        <Control label="Learning rate" id="lr">
          <input
            id="lr"
            name="lr"
            type="number"
            min="0.00001"
            step="0.0001"
            value={config.lr}
            onChange={(event) => update("lr", Number(event.target.value))}
            className="w-full rounded-lg bg-white px-3 py-2.5 text-base ring-1 ring-zinc-950/10 sm:py-2 sm:text-sm"
          />
        </Control>
        <div className="flex items-end">
          <label htmlFor="source-param-groups" className="flex items-center gap-3 text-base text-zinc-700 sm:text-sm">
            <span className="group inline-grid size-5 grid-cols-1 sm:size-4">
              <input
                id="source-param-groups"
                name="sourceParamGroups"
                type="checkbox"
                checked={config.sourceParamGroups}
                onChange={(event) => update("sourceParamGroups", event.target.checked)}
                className="col-start-1 row-start-1 appearance-none rounded-sm border border-zinc-300 bg-white checked:border-signal-600 checked:bg-signal-600 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal-600"
              />
              <span className="pointer-events-none col-start-1 row-start-1 hidden size-2 self-center justify-self-center rounded-[2px] bg-white group-has-checked:block" />
            </span>
            Source param groups
          </label>
        </div>
      </div>

      <pre className="mt-5 max-w-full overflow-x-auto rounded-lg bg-zinc-950 p-4 text-base text-zinc-100 sm:text-sm">
        <code>{command}</code>
      </pre>
    </section>
  );
}

function ChatLab() {
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [messages, setMessages] = useState([]);
  const [systemPrompt, setSystemPrompt] = useState(defaultSystemPrompt);
  const [prompt, setPrompt] = useState("Write a concise checklist for testing a local LLM inference server.");
  const [settings, setSettings] = useState({ temperature: 0.7, topP: 0.9, maxTokens: 256 });
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");

  const refreshStatus = async () => {
    try {
      const [statusResponse, historyResponse] = await Promise.all([
        fetch(`${apiBase}/api/status`),
        fetch(`${apiBase}/api/history`),
      ]);
      setStatus(await statusResponse.json());
      setHistory((await historyResponse.json()).runs || []);
    } catch (requestError) {
      setStatus({ backend: false, error: requestError.message });
    }
  };

  useEffect(() => {
    refreshStatus();
    const timer = window.setInterval(refreshStatus, 8000);
    return () => window.clearInterval(timer);
  }, []);

  const updateSetting = (key, value) => {
    setSettings((current) => ({ ...current, [key]: value }));
  };

  const submitPrompt = async () => {
    const userPrompt = prompt.trim();
    if (!userPrompt || isRunning) return;

    const nextMessages = [
      ...(systemPrompt.trim() ? [{ role: "system", content: systemPrompt.trim() }] : []),
      ...messages,
      { role: "user", content: userPrompt },
    ];
    setMessages((current) => [...current, { role: "user", content: userPrompt }]);
    setPrompt("");
    setIsRunning(true);
    setError("");

    try {
      const response = await fetch(`${apiBase}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: nextMessages, ...settings }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Generation failed");
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: payload.output || "",
          runId: payload.runId,
          latencyMs: payload.latencyMs,
        },
      ]);
      refreshStatus();
    } catch (requestError) {
      setError(requestError.message);
      setMessages((current) => [
        ...current,
        { role: "assistant", content: `Error: ${requestError.message}`, error: true },
      ]);
    } finally {
      setIsRunning(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError("");
  };

  const llamaReady = Boolean(status?.llama?.ok);

  return (
    <main className="mx-auto grid max-w-7xl gap-5 px-4 py-6 sm:px-6 lg:grid-cols-[minmax(0,1fr)_360px] lg:px-8">
      <section className="min-w-0 rounded-lg border border-zinc-950/10 bg-white">
        <div className="border-b border-zinc-950/10 p-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-base text-zinc-600 sm:text-sm">
                <MessageSquare className="size-4 shrink-0 stroke-signal-600" />
                <span>ymcki GGUF / MXFP4_MOE</span>
              </div>
              <h2 className="mt-2 text-2xl font-semibold tracking-tight text-balance text-zinc-950">
                Pretrained Kimi chat
              </h2>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-base sm:text-sm">
              <span
                className={[
                  "rounded-md border px-2.5 py-1.5",
                  llamaReady
                    ? "border-signal-600/20 bg-signal-50 text-signal-700"
                    : "border-ember-600/20 bg-ember-50 text-ember-700",
                ].join(" ")}
              >
                {llamaReady ? "llama-server ready" : "llama-server offline"}
              </span>
              <button
                type="button"
                onClick={refreshStatus}
                className="inline-flex h-9 items-center justify-center gap-2 rounded-lg bg-white py-2 pr-3 pl-2 text-sm font-medium text-zinc-800 ring-1 ring-zinc-950/10"
              >
                <RefreshCw className="size-4 shrink-0 stroke-zinc-500" />
                Refresh
              </button>
            </div>
          </div>
        </div>

        <div className="min-h-[420px] space-y-4 p-5">
          {messages.length === 0 ? (
            <div className="rounded-lg border border-dashed border-zinc-950/15 bg-zinc-50 p-5">
              <p className="max-w-[62ch] text-base text-pretty text-zinc-600 sm:text-sm">No turns in this session.</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className={[
                  "max-w-[82ch] rounded-lg border p-4",
                  message.role === "user"
                    ? "ml-auto border-cobalt-600/20 bg-cobalt-50"
                    : message.error
                      ? "border-ember-600/20 bg-ember-50"
                      : "border-zinc-950/10 bg-zinc-50",
                ].join(" ")}
              >
                <div className="mb-2 flex items-center justify-between gap-3 text-base sm:text-sm">
                  <span className="font-medium text-zinc-900">{message.role === "user" ? "You" : "Kimi"}</span>
                  {message.latencyMs ? (
                    <span className="font-mono text-zinc-500">{(message.latencyMs / 1000).toFixed(1)}s</span>
                  ) : null}
                </div>
                <p className="whitespace-pre-wrap text-base text-zinc-700 sm:text-sm">{message.content}</p>
                {message.runId ? <p className="mt-3 break-all font-mono text-base text-zinc-500 sm:text-sm">{message.runId}</p> : null}
              </div>
            ))
          )}
        </div>

        <div className="border-t border-zinc-950/10 p-5">
          {error ? <p className="mb-3 text-base text-ember-700 sm:text-sm">{error}</p> : null}
          <div className="grid gap-3">
            <textarea
              id="kimi-prompt"
              name="prompt"
              rows={4}
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if ((event.metaKey || event.ctrlKey) && event.key === "Enter") submitPrompt();
              }}
              aria-label="Prompt"
              className="w-full resize-y rounded-lg bg-white p-3 text-base text-zinc-950 ring-1 ring-zinc-950/10 focus:outline-2 focus:-outline-offset-1 focus:outline-signal-600 sm:text-sm"
            />
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <button
                type="button"
                onClick={submitPrompt}
                disabled={isRunning || !prompt.trim()}
                className="inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-signal-600 py-2 pr-3 pl-2 text-sm font-medium text-white ring-1 ring-signal-600 disabled:cursor-not-allowed disabled:bg-zinc-300 disabled:ring-zinc-300"
              >
                <Send className="size-4 shrink-0 stroke-white" />
                {isRunning ? "Running" : "Run prompt"}
              </button>
              <button
                type="button"
                onClick={clearChat}
                className="inline-flex h-9 items-center justify-center rounded-lg bg-white px-3 py-2 text-sm font-medium text-zinc-800 ring-1 ring-zinc-950/10"
              >
                Clear
              </button>
            </div>
          </div>
        </div>
      </section>

      <aside className="min-w-0 space-y-5">
        <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
          <div className="flex items-start gap-3">
            <Server className="mt-1 size-4 shrink-0 stroke-cobalt-600" />
            <div className="min-w-0">
              <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">Runtime</h2>
              <dl className="mt-4 space-y-3 text-base sm:text-sm">
                <div>
                  <dt className="font-medium text-zinc-900">Model</dt>
                  <dd className="mt-1 break-all font-mono text-zinc-500">MXFP4_MOE.gguf</dd>
                </div>
                <div>
                  <dt className="font-medium text-zinc-900">Backend</dt>
                  <dd className="mt-1 break-all font-mono text-zinc-500">{apiBase}</dd>
                </div>
                <div>
                  <dt className="font-medium text-zinc-900">llama.cpp</dt>
                  <dd className="mt-1 break-all font-mono text-zinc-500">
                    {status?.llamaUrl || "http://127.0.0.1:8081"}
                  </dd>
                </div>
              </dl>
            </div>
          </div>
        </section>

        <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
          <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">Settings</h2>
          <div className="mt-4 space-y-4">
            <Control label="System" id="system-prompt">
              <textarea
                id="system-prompt"
                name="systemPrompt"
                rows={5}
                value={systemPrompt}
                onChange={(event) => setSystemPrompt(event.target.value)}
                className="w-full resize-y rounded-lg bg-white p-3 text-base text-zinc-950 ring-1 ring-zinc-950/10 focus:outline-2 focus:-outline-offset-1 focus:outline-signal-600 sm:text-sm"
              />
            </Control>
            <Control label="Temperature" id="temperature">
              <input
                id="temperature"
                name="temperature"
                type="number"
                min="0"
                max="2"
                step="0.05"
                value={settings.temperature}
                onChange={(event) => updateSetting("temperature", Number(event.target.value))}
                className="w-full rounded-lg bg-white px-3 py-2.5 text-base ring-1 ring-zinc-950/10 sm:py-2 sm:text-sm"
              />
            </Control>
            <Control label="Top-p" id="top-p">
              <input
                id="top-p"
                name="topP"
                type="number"
                min="0.05"
                max="1"
                step="0.05"
                value={settings.topP}
                onChange={(event) => updateSetting("topP", Number(event.target.value))}
                className="w-full rounded-lg bg-white px-3 py-2.5 text-base ring-1 ring-zinc-950/10 sm:py-2 sm:text-sm"
              />
            </Control>
            <Control label="Max tokens" id="max-tokens">
              <input
                id="max-tokens"
                name="maxTokens"
                type="number"
                min="16"
                max="2048"
                step="16"
                value={settings.maxTokens}
                onChange={(event) => updateSetting("maxTokens", Number(event.target.value))}
                className="w-full rounded-lg bg-white px-3 py-2.5 text-base ring-1 ring-zinc-950/10 sm:py-2 sm:text-sm"
              />
            </Control>
          </div>
        </section>

        <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
          <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">Recent runs</h2>
          <div className="mt-4 divide-y divide-zinc-950/10">
            {history.length === 0 ? (
              <p className="text-base text-zinc-500 sm:text-sm">No chat runs yet.</p>
            ) : (
              history.slice(0, 6).map((run) => (
                <div key={run.run_id} className="py-3">
                  <div className="flex items-center justify-between gap-3 text-base sm:text-sm">
                    <span className={run.status === "ok" ? "text-signal-700" : "text-ember-700"}>{run.status}</span>
                    <span className="font-mono text-zinc-500">
                      {run.results?.latency_ms ? `${(run.results.latency_ms / 1000).toFixed(1)}s` : ""}
                    </span>
                  </div>
                  <p className="mt-1 break-all font-mono text-base text-zinc-500 sm:text-sm">{run.run_id}</p>
                  {run.results?.output_preview ? (
                    <p className="mt-1 line-clamp-3 text-base text-zinc-600 sm:text-sm">{run.results.output_preview}</p>
                  ) : null}
                </div>
              ))
            )}
          </div>
        </section>
      </aside>
    </main>
  );
}

function Control({ label, id, children }) {
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-base font-medium text-zinc-700 sm:text-sm">
        {label}
      </label>
      {children}
    </div>
  );
}

function ArtifactPanel() {
  return (
    <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">Artifact index</h2>
          <p className="mt-2 max-w-[62ch] text-base text-pretty text-zinc-600 sm:text-sm">
            Each UI number is backed by one of these repo artifacts.
          </p>
        </div>
        <Database className="size-4 shrink-0 stroke-zinc-600" />
      </div>
      <div className="mt-5 divide-y divide-zinc-950/10">
        {artifacts.map((artifact) => (
          <div key={artifact.path} className="flex items-center justify-between gap-4 py-3">
            <div className="min-w-0">
              <p className="text-base font-medium text-zinc-900 sm:text-sm">{artifact.name}</p>
              <p className="mt-1 break-all font-mono text-base text-zinc-500 sm:text-sm">{artifact.path}</p>
            </div>
            <ExternalLink className="size-4 shrink-0 stroke-zinc-400" />
          </div>
        ))}
      </div>
    </section>
  );
}

export default function App() {
  const [selectedTask, setSelectedTask] = useState("mqar");
  const [view, setView] = useState("chat");

  return (
    <div className="min-h-screen text-zinc-950">
      <Header view={view} setView={setView} />
      {view === "chat" ? (
        <ChatLab />
      ) : (
        <>
          <StatGrid />
          <main className="mx-auto grid max-w-7xl gap-5 px-4 py-6 sm:px-6 lg:grid-cols-[minmax(0,1fr)_360px] lg:px-8">
            <div className="min-w-0 space-y-5">
              <MechanismPanel />
              <CurvePanel />
              <TaskPanel selectedTask={selectedTask} setSelectedTask={setSelectedTask} />
              <OperatorPanel />
              <CommandPanel />
            </div>
            <aside className="min-w-0 space-y-5">
              <AblationPanel />
              <ArtifactPanel />
              <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
                <div className="flex items-start gap-3">
                  <Play className="mt-1 size-4 shrink-0 stroke-signal-600" />
                  <div className="min-w-0">
                    <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">Next run focus</h2>
                    <p className="mt-2 text-base text-pretty text-zinc-600 sm:text-sm">
                      Vary `source-init-scope`, sequence length, and seed first; those are the levers that changed the
                      reproduction outcome.
                    </p>
                  </div>
                </div>
              </section>
              <section className="rounded-lg border border-zinc-950/10 bg-white p-5">
                <div className="flex items-start gap-3">
                  <SlidersHorizontal className="mt-1 size-4 shrink-0 stroke-cobalt-600" />
                  <div className="min-w-0">
                    <h2 className="text-xl font-semibold tracking-tight text-balance text-zinc-950">Hardware path</h2>
                    <p className="mt-2 break-words text-base text-pretty text-zinc-600 sm:text-sm">
                      {projectMeta.environment}
                    </p>
                  </div>
                </div>
              </section>
            </aside>
          </main>
        </>
      )}
    </div>
  );
}
