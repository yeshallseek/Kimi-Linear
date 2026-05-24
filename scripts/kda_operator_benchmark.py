#!/usr/bin/env python3
"""Reduced KDA operator benchmark for local reproduction runs."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import traceback
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule
from fla.ops.kda import chunk_kda

try:
    from flash_attn import flash_attn_func
except Exception:  # pragma: no cover - recorded at runtime
    flash_attn_func = None


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def dumps_record(record: dict) -> str:
    return json.dumps(record, sort_keys=True, allow_nan=False)


def run_one(args: argparse.Namespace, provider: str, T: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLA/Triton operator benchmarking.")

    os.environ["FLA_USE_TMA"] = "1" if args.tma else "0"
    torch.manual_seed(args.seed + T + sum(ord(c) for c in provider))
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    B, H, D = args.batch_size, args.heads, args.head_dim
    do = torch.randn(B, T, H, D, dtype=dtype, device=device)

    def bench(fn: Callable[[], torch.Tensor]) -> tuple[float, float, float]:
        sync()
        result = triton.testing.do_bench(
            fn,
            quantiles=[0.5, 0.2, 0.8],
            warmup=args.warmup,
            rep=args.rep,
        )
        sync()
        return tuple(float(x) for x in result)

    if provider == "gdn":
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        g = F.logsigmoid(torch.randn(B, T, H, dtype=dtype, device=device)).requires_grad_(True)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)

        def fn() -> torch.Tensor:
            if args.backward:
                clear_grads(q, k, v, g, beta)
            out = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)[0]
            if args.backward:
                out.backward(do, retain_graph=False)
            return out

    elif provider == "kda":
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device)).requires_grad_(True)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid().requires_grad_(True)

        def fn() -> torch.Tensor:
            if args.backward:
                clear_grads(q, k, v, g, beta)
            out = chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)[0]
            if args.backward:
                out.backward(do, retain_graph=False)
            return out

    elif provider == "dplr":
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        a = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        b = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device)).requires_grad_(True)

        def fn() -> torch.Tensor:
            if args.backward:
                clear_grads(q, k, a, b, v, g)
            out = chunk_dplr_delta_rule(q=q, k=k, v=v, a=a, b=b, gk=g)[0]
            if args.backward:
                out.backward(do, retain_graph=False)
            return out

    elif provider == "attn":
        if flash_attn_func is None:
            raise RuntimeError("flash-attn is not importable.")
        q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)

        def fn() -> torch.Tensor:
            if args.backward:
                clear_grads(q, k, v)
            out = flash_attn_func(q, k, v)
            if args.backward:
                out.backward(do, retain_graph=False)
            return out

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    p50, p20, p80 = bench(fn)
    free, total = torch.cuda.mem_get_info()
    return {
        "ok": True,
        "provider": provider,
        "seq_len": T,
        "median_ms": p50,
        "p20_ms": p20,
        "p80_ms": p80,
        "shape": {
            "batch_size": B,
            "heads": H,
            "head_dim": D,
            "dtype": args.dtype,
            "backward": args.backward,
            "tma": args.tma,
        },
        "gpu_free_mib_after": free // 2**20,
        "gpu_total_mib": total // 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", default="kda,dplr,gdn,attn")
    parser.add_argument("--lengths", default="256,512,1024,2048,4096")
    parser.add_argument("--output", type=Path, default=Path("artifacts/kda_operator_benchmark.jsonl"))
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--tma", action="store_true")
    parser.add_argument("--forward-only", dest="backward", action="store_false")
    parser.set_defaults(backward=True)
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "type": "metadata",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flash_linear_attention": package_version("flash-linear-attention"),
        "fla_core": package_version("fla-core"),
        "triton": package_version("triton"),
        "flash_attn": package_version("flash-attn"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    with args.output.open("w", encoding="utf-8") as f:
        f.write(dumps_record(header) + "\n")
        f.flush()
        for T in lengths:
            for provider in providers:
                clear_memory()
                try:
                    record = run_one(args, provider, T)
                except Exception as exc:  # continue through install/OOM/provider issues
                    record = {
                        "ok": False,
                        "provider": provider,
                        "seq_len": T,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                print(dumps_record(record))
                f.write(dumps_record(record) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
