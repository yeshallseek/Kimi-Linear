#!/usr/bin/env python3
"""Small KDA correctness smoke test against FLA's naive recurrent reference."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.naive import naive_recurrent_kda


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().cpu())


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args: argparse.Namespace) -> dict:
    os.environ["FLA_USE_TMA"] = "1" if args.tma else "0"
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    if device.type != "cuda":
        raise RuntimeError("FLA KDA Triton kernels require CUDA for this smoke test.")

    B, T, H, D = args.batch_size, args.seq_len, args.heads, args.head_dim
    scale = args.scale

    q = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_(True)
    g = (F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32, device=device)) / args.gate_logit_normalizer)
    g.requires_grad_(True)
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    beta.requires_grad_(True)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device).requires_grad_(True)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    sync_if_cuda(device)
    start = time.perf_counter()
    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_grads = {
        "dq": q.grad.detach().clone(),
        "dk": k.grad.detach().clone(),
        "dv": v.grad.detach().clone(),
        "dg": g.grad.detach().clone(),
        "dbeta": beta.grad.detach().clone(),
        "dh0": h0.grad.detach().clone(),
    }
    sync_if_cuda(device)
    ref_ms = (time.perf_counter() - start) * 1000

    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    sync_if_cuda(device)
    start = time.perf_counter()
    chunk, chunk_ht = chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ((chunk * do).sum() + (chunk_ht * dht).sum()).backward(retain_graph=True)
    chunk_grads = {
        "dq": q.grad.detach().clone(),
        "dk": k.grad.detach().clone(),
        "dv": v.grad.detach().clone(),
        "dg": g.grad.detach().clone(),
        "dbeta": beta.grad.detach().clone(),
        "dh0": h0.grad.detach().clone(),
    }
    sync_if_cuda(device)
    chunk_ms = (time.perf_counter() - start) * 1000

    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    sync_if_cuda(device)
    start = time.perf_counter()
    recurrent, recurrent_ht = fused_recurrent_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    sync_if_cuda(device)
    recurrent_ms = (time.perf_counter() - start) * 1000

    errors = {
        "chunk_o": max_abs(ref, chunk),
        "chunk_final_state": max_abs(ref_ht, chunk_ht),
        "recurrent_o": max_abs(ref, recurrent),
        "recurrent_final_state": max_abs(ref_ht, recurrent_ht),
    }
    for name, ref_grad in ref_grads.items():
        errors[f"chunk_grad_{name}"] = max_abs(ref_grad, chunk_grads[name])

    tolerance = args.tolerance
    ok = all(value <= tolerance for value in errors.values())

    gpu = None
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_mib": total // 2**20,
            "free_memory_mib": free // 2**20,
        }

    return {
        "ok": ok,
        "tolerance": tolerance,
        "errors": errors,
        "timing_ms": {
            "naive_recurrent_forward_backward": ref_ms,
            "chunk_forward_backward": chunk_ms,
            "fused_recurrent_forward": recurrent_ms,
        },
        "shape": {
            "batch_size": B,
            "seq_len": T,
            "heads": H,
            "head_dim": D,
            "dtype": args.dtype,
            "scale": scale,
            "tma": args.tma,
        },
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flash_linear_attention": package_version("flash-linear-attention"),
            "fla_core": package_version("fla-core"),
            "triton": package_version("triton"),
            "gpu": gpu,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/kda_kernel_smoke.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--gate-logit-normalizer", type=float, default=1.0)
    parser.add_argument("--tma", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()

    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
