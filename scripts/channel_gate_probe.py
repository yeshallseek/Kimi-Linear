#!/usr/bin/env python3
"""Mechanism probe for KDA channel-wise gating vs GDN scalar gating.

This is not a replacement for the paper's Figure 4 training curves. It is a
small causal probe for the core architectural claim: KDA's per-channel decay can
keep long-lived information while forgetting short-lived/noisy channels, whereas
a scalar/head-wise gate must use one decay rate for both.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import platform
from pathlib import Path

import torch


RATIO_EPSILON = 1e-12


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def dumps_record(record: dict) -> str:
    return json.dumps(record, indent=2, sort_keys=True, allow_nan=False)


def alpha_grid(size: int, device: torch.device) -> torch.Tensor:
    if size < 2:
        raise ValueError("--grid-size must be at least 2")
    # Include 1.0 exactly so KDA can express the ideal long-memory channel.
    return torch.cat(
        [
            torch.linspace(0.0, 0.999, steps=size - 1, device=device, dtype=torch.float64),
            torch.tensor([1.0], device=device, dtype=torch.float64),
        ],
    )


def theoretical_mse(alpha: torch.Tensor, seq_len: int, signal_var: float, noise_var: float) -> torch.Tensor:
    powers = torch.arange(1, seq_len, device=alpha.device, dtype=torch.float64)
    signal_error = (alpha ** (seq_len - 1) - 1.0).square() * signal_var
    noise_error = (alpha[:, None] ** (2.0 * powers)).sum(dim=1) * noise_var
    return signal_error + noise_error


def generate_batch(
    seq_len: int,
    samples: int,
    signal_std: float,
    noise_std: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    writes = torch.zeros(samples, seq_len, 2, device=device, dtype=torch.float64)
    long_signal = torch.randn(samples, device=device, generator=generator, dtype=torch.float64) * signal_std
    recent_signal = torch.randn(samples, device=device, generator=generator, dtype=torch.float64) * signal_std
    noise = torch.randn(samples, max(seq_len - 1, 1), device=device, generator=generator, dtype=torch.float64) * noise_std
    writes[:, 0, 0] = long_signal
    writes[:, : seq_len - 1, 1] = noise[:, : seq_len - 1]
    writes[:, seq_len - 1, 1] = recent_signal
    target = long_signal + recent_signal
    return writes, target


def empirical_mse(writes: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor) -> float:
    alpha = alpha.to(device=writes.device, dtype=torch.float64)
    if alpha.numel() == 1:
        alpha = alpha.expand(writes.shape[-1])
    seq_len = writes.shape[1]
    long_signal = writes[:, 0, 0]
    recent_signal = writes[:, seq_len - 1, 1]
    long_contribution = long_signal * (alpha[0] ** (seq_len - 1))
    if seq_len > 1:
        powers = torch.arange(seq_len - 1, 0, -1, device=writes.device, dtype=torch.float64)
        noise_contribution = writes[:, : seq_len - 1, 1] @ (alpha[1] ** powers)
    else:
        noise_contribution = torch.zeros_like(long_signal)
    pred = long_contribution + noise_contribution + recent_signal
    return float((pred - target).square().mean().cpu())


def learn_alphas(
    writes: torch.Tensor,
    target: torch.Tensor,
    mode: str,
    steps: int,
    lr: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    if mode == "gdn":
        logits = torch.nn.Parameter(torch.zeros(1, device=writes.device, dtype=torch.float64))
    elif mode == "kda":
        logits = torch.nn.Parameter(torch.zeros(2, device=writes.device, dtype=torch.float64))
    else:
        raise ValueError(mode)
    optimizer = torch.optim.AdamW([logits], lr=lr, weight_decay=0.0)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        alpha = torch.sigmoid(logits)
        decay = alpha.expand(writes.shape[-1]) if mode == "gdn" else alpha
        seq_len = writes.shape[1]
        long_signal = writes[:, 0, 0]
        recent_signal = writes[:, seq_len - 1, 1]
        long_contribution = long_signal * (decay[0] ** (seq_len - 1))
        if seq_len > 1:
            powers = torch.arange(seq_len - 1, 0, -1, device=writes.device, dtype=torch.float64)
            noise_contribution = writes[:, : seq_len - 1, 1] @ (decay[1] ** powers)
        else:
            noise_contribution = torch.zeros_like(long_signal)
        pred = long_contribution + noise_contribution + recent_signal
        loss = (pred - target).square().mean()
        loss.backward()
        optimizer.step()
    alpha = torch.sigmoid(logits).detach().cpu()
    return {
        "mode": mode,
        "learned_alpha": [float(x) for x in alpha.reshape(-1)],
        "learned_mse": empirical_mse(writes, target, alpha.to(writes.device)),
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    grid = alpha_grid(args.grid_size, device)
    rows: list[dict] = []
    for seq_len in args.seq_lengths:
        theory = theoretical_mse(
            grid,
            seq_len=seq_len,
            signal_var=args.signal_std**2,
            noise_var=args.noise_std**2,
        )
        gdn_idx = int(torch.argmin(theory).cpu())
        gdn_alpha = float(grid[gdn_idx].cpu())

        # KDA can independently choose the best long and short channel decays.
        long_signal_error = (grid ** (seq_len - 1) - 1.0).square() * (args.signal_std**2)
        short_noise_error = theoretical_mse(
            grid,
            seq_len=seq_len,
            signal_var=0.0,
            noise_var=args.noise_std**2,
        )
        kda_long_alpha = float(grid[int(torch.argmin(long_signal_error).cpu())].cpu())
        kda_short_alpha = float(grid[int(torch.argmin(short_noise_error).cpu())].cpu())

        writes, target = generate_batch(
            seq_len=seq_len,
            samples=args.samples,
            signal_std=args.signal_std,
            noise_std=args.noise_std,
            seed=args.seed + seq_len,
            device=device,
        )
        gdn_empirical = empirical_mse(writes, target, torch.tensor([gdn_alpha], dtype=torch.float64, device=device))
        kda_empirical = empirical_mse(
            writes,
            target,
            torch.tensor([kda_long_alpha, kda_short_alpha], dtype=torch.float64, device=device),
        )
        ratio = gdn_empirical / max(kda_empirical, RATIO_EPSILON)
        learned_gdn = learn_alphas(writes, target, "gdn", args.learn_steps, args.learn_lr, args.seed) if args.learn_steps > 0 else None
        learned_kda = learn_alphas(writes, target, "kda", args.learn_steps, args.learn_lr, args.seed) if args.learn_steps > 0 else None
        rows.append(
            {
                "seq_len": seq_len,
                "gdn_best_alpha": gdn_alpha,
                "gdn_theory_mse": float(theory[gdn_idx].cpu()),
                "gdn_empirical_mse": gdn_empirical,
                "kda_best_alpha_long": kda_long_alpha,
                "kda_best_alpha_short": kda_short_alpha,
                "kda_empirical_mse": kda_empirical,
                "kda_empirical_zero": kda_empirical == 0.0,
                "empirical_mse_ratio_gdn_over_kda": ratio,
                "learned_gdn_alpha": learned_gdn["learned_alpha"] if learned_gdn else None,
                "learned_gdn_mse": learned_gdn["learned_mse"] if learned_gdn else None,
                "learned_kda_alpha": learned_kda["learned_alpha"] if learned_kda else None,
                "learned_kda_mse": learned_kda["learned_mse"] if learned_kda else None,
            },
        )

    return {
        "summary": {
            "claim_tested": "channel-wise KDA decay can retain a long-memory channel while forgetting a noisy short-memory channel; scalar GDN cannot satisfy both with one alpha",
            "interpretation": "This is a recurrence-level causal probe, not a full Figure 4 reproduction.",
            "ratio_note": f"Ratios use max(kda_empirical_mse, {RATIO_EPSILON}) because the grid-best KDA setting can be exactly zero-error.",
            "best_ratio_gdn_over_kda": max(row["empirical_mse_ratio_gdn_over_kda"] for row in rows),
        },
        "config": {
            "seq_lengths": args.seq_lengths,
            "samples": args.samples,
            "grid_size": args.grid_size,
            "signal_std": args.signal_std,
            "noise_std": args.noise_std,
            "learn_steps": args.learn_steps,
            "learn_lr": args.learn_lr,
            "seed": args.seed,
            "device": str(device),
        },
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flash_linear_attention": package_version("flash-linear-attention"),
            "fla_core": package_version("fla-core"),
        },
        "results": rows,
    }


def write_csv(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seq_len",
        "gdn_best_alpha",
        "gdn_theory_mse",
        "gdn_empirical_mse",
        "kda_best_alpha_long",
        "kda_best_alpha_short",
        "kda_empirical_mse",
        "kda_empirical_zero",
        "empirical_mse_ratio_gdn_over_kda",
        "learned_gdn_alpha",
        "learned_gdn_mse",
        "learned_kda_alpha",
        "learned_kda_mse",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in results:
            serialized = dict(row)
            serialized["learned_gdn_alpha"] = json.dumps(serialized["learned_gdn_alpha"])
            serialized["learned_kda_alpha"] = json.dumps(serialized["learned_kda_alpha"])
            writer.writerow(serialized)


def write_plot(path: Path, results: list[dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [row["seq_len"] for row in results]
    gdn = [row["gdn_empirical_mse"] for row in results]
    kda = [max(row["kda_empirical_mse"], 1e-12) for row in results]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, gdn, marker="o", label="GDN scalar gate (grid best)")
    plt.plot(xs, kda, marker="o", label="KDA channel gates (grid best)")
    if all(row["learned_gdn_mse"] is not None for row in results):
        learned_gdn = [row["learned_gdn_mse"] for row in results]
        plt.plot(xs, learned_gdn, marker="x", linestyle="--", label="GDN learned")
    if all(row["learned_kda_mse"] is not None for row in results):
        learned_kda = [row["learned_kda_mse"] for row in results]
        plt.plot(xs, learned_kda, marker="x", linestyle="--", label="KDA learned")
    plt.yscale("log")
    plt.xlabel("Sequence length")
    plt.ylabel("Final recall MSE (log)")
    plt.title("Channel-wise decay separates retention from forgetting")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/channel_gate_probe.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("artifacts/channel_gate_probe.csv"))
    parser.add_argument("--figure-output", type=Path, default=Path("reports/figures/channel_gate_probe.png"))
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--signal-std", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--learn-steps", type=int, default=0)
    parser.add_argument("--learn-lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps_record(result) + "\n", encoding="utf-8")
    write_csv(args.csv_output, result["results"])
    plotted = write_plot(args.figure_output, result["results"])
    result["artifacts"] = {
        "json": str(args.output),
        "csv": str(args.csv_output),
        "figure": str(args.figure_output) if plotted else None,
    }
    args.output.write_text(dumps_record(result) + "\n", encoding="utf-8")
    print(dumps_record(result))


if __name__ == "__main__":
    main()
