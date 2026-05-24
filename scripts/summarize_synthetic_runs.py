#!/usr/bin/env python3
"""Summarize synthetic probe JSONL files into a compact comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def summarize_file(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    metadata = rows[0] if rows[0].get("type") == "metadata" else {}
    args = metadata.get("args", {})
    models = sorted({row.get("model") for row in rows if row.get("model")})
    summaries: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row for row in rows if row.get("model") == model]
        errors = [row for row in model_rows if row.get("type") == "error"]
        eval_rows = [row for row in model_rows if row.get("type") == "train_eval"]
        if errors and not eval_rows:
            err = errors[-1]
            summaries.append(
                {
                    "file": str(path),
                "task": metadata.get("task") or err.get("task"),
                "model": model,
                "lr": args.get("lr"),
                "seed": args.get("seed"),
                "status": "error",
                "error_type": err.get("error_type"),
                "error": err.get("error"),
                },
            )
            continue
        if not eval_rows:
            continue
        final = max(eval_rows, key=lambda row: row.get("step", -1))
        best = max(eval_rows, key=lambda row: row.get("eval_accuracy", float("-inf")))
        summaries.append(
            {
                "file": str(path),
                "task": metadata.get("task") or final.get("task"),
                "model": model,
                "lr": args.get("lr"),
                "weight_decay": args.get("weight_decay"),
                "seed": args.get("seed"),
                "status": "ok",
                "requested_seq_len": args.get("seq_len"),
                "actual_seq_len": final.get("actual_seq_len"),
                "vocab_size": args.get("vocab_size"),
                "hidden_size": args.get("hidden_size"),
                "batch_size": args.get("batch_size"),
                "dtype": args.get("dtype"),
                "steps": args.get("steps"),
                "mqar_train_curriculum": args.get("mqar_train_curriculum"),
                "source_init": args.get("source_init"),
                "source_init_scope": args.get("source_init_scope"),
                "source_param_groups": args.get("source_param_groups"),
                "param_count": final.get("param_count"),
                "final_step": final.get("step"),
                "final_eval_accuracy": final.get("eval_accuracy"),
                "final_eval_loss": final.get("eval_loss"),
                "final_train_loss": final.get("train_loss"),
                "best_eval_accuracy": best.get("eval_accuracy"),
                "best_eval_step": best.get("step"),
                "best_eval_loss": best.get("eval_loss"),
            },
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row.get("task"),
            row.get("model"),
            row.get("lr"),
            row.get("weight_decay"),
            row.get("requested_seq_len"),
            row.get("actual_seq_len"),
            row.get("vocab_size"),
            row.get("hidden_size"),
            row.get("batch_size"),
            row.get("dtype"),
            row.get("mqar_train_curriculum"),
            row.get("source_init"),
            row.get("source_init_scope"),
            row.get("source_param_groups"),
        )
        groups.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        (
            task,
            model,
            lr,
            weight_decay,
            requested_seq_len,
            actual_seq_len,
            vocab_size,
            hidden_size,
            batch_size,
            dtype,
            mqar_train_curriculum,
            source_init,
            source_init_scope,
            source_param_groups,
        ) = key
        final_acc = [row["final_eval_accuracy"] for row in group if row.get("final_eval_accuracy") is not None]
        best_acc = [row["best_eval_accuracy"] for row in group if row.get("best_eval_accuracy") is not None]
        final_loss = [row["final_eval_loss"] for row in group if row.get("final_eval_loss") is not None]
        aggregates.append(
            {
                "task": task,
                "model": model,
                "lr": lr,
                "weight_decay": weight_decay,
                "requested_seq_len": requested_seq_len,
                "actual_seq_len": actual_seq_len,
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "batch_size": batch_size,
                "dtype": dtype,
                "mqar_train_curriculum": mqar_train_curriculum,
                "source_init": source_init,
                "source_init_scope": source_init_scope,
                "source_param_groups": source_param_groups,
                "num_seeds": len({row.get("seed") for row in group}),
                "seeds": sorted(row.get("seed") for row in group),
                "mean_final_eval_accuracy": mean_or_none(final_acc),
                "min_final_eval_accuracy": min(final_acc) if final_acc else None,
                "max_final_eval_accuracy": max(final_acc) if final_acc else None,
                "mean_best_eval_accuracy": mean_or_none(best_acc),
                "mean_final_eval_loss": mean_or_none(final_loss),
            },
        )
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("artifacts/synthetic_summary.json"))
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for path in args.inputs:
        summaries.extend(summarize_file(path))
    result = {
        "inputs": [str(path) for path in args.inputs],
        "results": summaries,
        "aggregates": aggregate_rows(summaries),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if args.csv_output:
        write_csv(args.csv_output, summaries)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
