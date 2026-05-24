#!/usr/bin/env python3
"""Scaled synthetic copy/recall probes for KDA, GDN, and Mamba2."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import platform
import time
import traceback
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.kda import KimiDeltaAttention
from fla.layers.mamba2 import Mamba2


PAD = 0
BOS = 1
SEP = 2
DATA_START = 3
PUSH = DATA_START
POP = DATA_START + 1


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def sanitize_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    return value


def dumps_record(record: dict) -> str:
    return json.dumps(sanitize_json(record), sort_keys=True, allow_nan=False)


def flush_record(handle, record: dict) -> None:
    line = dumps_record(record)
    print(line)
    handle.write(line + "\n")
    handle.flush()


def parse_models(raw: str) -> list[str]:
    models = [m.strip().lower() for m in raw.split(",") if m.strip()]
    allowed = {"kda", "gdn", "mamba2"}
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    return models


def make_palindrome_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload_len = max(32, (args.seq_len - 1) // 2)
    x = torch.randint(DATA_START, args.vocab_size, (args.batch_size, payload_len), device=device)
    sep = torch.full((args.batch_size, 1), SEP, dtype=torch.long, device=device)
    seq = torch.cat([x, sep, torch.flip(x, dims=[1])], dim=1)
    labels = seq.clone()
    labels[:, : payload_len + 1] = -100
    return seq, labels


def make_mqar_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key_space = min(args.key_space, max(8, (args.vocab_size - DATA_START) // 2))
    value_start = DATA_START + key_space
    if value_start >= args.vocab_size:
        raise ValueError("vocab_size is too small for MQAR key/value split.")

    max_pairs_by_len = max(4, (args.seq_len - 1) // 4)
    num_pairs = min(args.num_pairs or max_pairs_by_len, key_space, max_pairs_by_len)
    num_queries = min(args.num_queries or max(2, num_pairs // 4), num_pairs)
    seq_len = num_pairs * 2 + 1 + num_queries * 2

    seq = torch.full((args.batch_size, seq_len), PAD, dtype=torch.long, device=device)
    labels = torch.full_like(seq, -100)
    sep_pos = num_pairs * 2
    seq[:, sep_pos] = SEP

    for b in range(args.batch_size):
        keys = torch.randperm(key_space, device=device)[:num_pairs] + DATA_START
        values = torch.randint(value_start, args.vocab_size, (num_pairs,), device=device)
        pairs = torch.stack([keys, values], dim=1).flatten()
        query_indices = torch.randperm(num_pairs, device=device)[:num_queries]
        qkeys = keys[query_indices]
        qvals = values[query_indices]
        queries = torch.stack([qkeys, qvals], dim=1).flatten()
        seq[b, : sep_pos] = pairs
        seq[b, sep_pos + 1 :] = queries
        value_positions = torch.arange(sep_pos + 2, seq_len, 2, device=device)
        labels[b, value_positions] = seq[b, value_positions]

    return seq, labels


def make_stack_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    num_stacks = min(args.num_stacks, max(1, args.vocab_size - DATA_START - 3))
    stack_start = DATA_START + 2
    value_start = stack_start + num_stacks
    if value_start >= args.vocab_size:
        raise ValueError("vocab_size is too small for stack IDs and values.")
    num_values = args.vocab_size - value_start
    num_ops = max(4, args.seq_len // 3)
    seq_len = num_ops * 3

    seq = torch.full((args.batch_size, seq_len), PAD, dtype=torch.long, device=device)
    labels = torch.full_like(seq, -100)
    for b in range(args.batch_size):
        stacks: list[list[int]] = [[] for _ in range(num_stacks)]
        target_count = 0
        for op_idx in range(num_ops):
            pos = op_idx * 3
            nonempty = [idx for idx, stack in enumerate(stacks) if stack]
            must_pop = op_idx == num_ops - 1 and target_count == 0 and bool(nonempty)
            do_push = not nonempty or (not must_pop and float(torch.rand((), device=device).cpu()) < args.stack_push_prob)
            if do_push:
                stack_id = int(torch.randint(0, num_stacks, (), device=device).cpu())
                value = int(torch.randint(0, num_values, (), device=device).cpu())
                stacks[stack_id].append(value)
                seq[b, pos : pos + 3] = torch.tensor(
                    [PUSH, stack_start + stack_id, value_start + value],
                    dtype=torch.long,
                    device=device,
                )
            else:
                stack_id = nonempty[int(torch.randint(0, len(nonempty), (), device=device).cpu())]
                value = stacks[stack_id].pop()
                seq[b, pos : pos + 3] = torch.tensor(
                    [POP, stack_start + stack_id, value_start + value],
                    dtype=torch.long,
                    device=device,
                )
                labels[b, pos + 2] = value_start + value
                target_count += 1

    return seq, labels


def pad_to_chunk_training_minimum(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    minimum_seq_len: int = 65,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FLA KDA/GDN training requires chunk mode; q_len <= 64 selects recurrent mode."""
    pad_len = minimum_seq_len - input_ids.shape[1]
    if pad_len <= 0:
        return input_ids, labels
    input_pad = torch.full((input_ids.shape[0], pad_len), PAD, dtype=input_ids.dtype, device=input_ids.device)
    label_pad = torch.full((labels.shape[0], pad_len), -100, dtype=labels.dtype, device=labels.device)
    return torch.cat([input_ids, input_pad], dim=1), torch.cat([labels, label_pad], dim=1)


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if args.task == "palindrome":
        return pad_to_chunk_training_minimum(*make_palindrome_batch(args, device))
    if args.task == "mqar":
        return pad_to_chunk_training_minimum(*make_mqar_batch(args, device))
    if args.task == "stack":
        return pad_to_chunk_training_minimum(*make_stack_batch(args, device))
    raise ValueError(f"Unsupported task: {args.task}")


class TinyMixerBlock(nn.Module):
    def __init__(self, model_name: str, args: argparse.Namespace, layer_idx: int):
        super().__init__()
        self.model_name = model_name
        self.norm1 = nn.RMSNorm(args.hidden_size, eps=1e-5)
        self.norm2 = nn.RMSNorm(args.hidden_size, eps=1e-5)
        if model_name == "kda":
            self.mixer = KimiDeltaAttention(
                hidden_size=args.hidden_size,
                head_dim=args.head_dim,
                num_heads=args.heads,
                mode="chunk",
                use_short_conv=not args.no_short_conv,
                layer_idx=layer_idx,
            )
        elif model_name == "gdn":
            self.mixer = GatedDeltaNet(
                hidden_size=args.hidden_size,
                expand_v=1.0,
                head_dim=args.head_dim,
                num_heads=args.heads,
                mode="chunk",
                use_short_conv=not args.no_short_conv,
                layer_idx=layer_idx,
            )
        elif model_name == "mamba2":
            num_heads = max(1, int(args.mamba_expand * args.hidden_size / args.mamba_head_dim))
            self.mixer = Mamba2(
                num_heads=num_heads,
                hidden_size=args.hidden_size,
                head_dim=args.mamba_head_dim,
                state_size=args.mamba_state_size,
                expand=args.mamba_expand,
                chunk_size=args.mamba_chunk_size,
                layer_idx=layer_idx,
                rms_norm=False,
            )
        else:
            raise ValueError(model_name)

        mlp_hidden = max(args.hidden_size, int(args.hidden_size * args.mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_size, mlp_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(mlp_hidden, args.hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        if self.model_name == "mamba2":
            y = self.mixer(y)
        else:
            y = self.mixer(y, use_cache=False)[0]
        x = x + y
        return x + self.mlp(self.norm2(x))


class TinyMixerLM(nn.Module):
    def __init__(self, model_name: str, args: argparse.Namespace):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size, padding_idx=PAD)
        self.layers = nn.ModuleList([TinyMixerBlock(model_name, args, i) for i in range(args.layers)])
        self.norm = nn.RMSNorm(args.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **_: object):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            shifted = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], -100)), dim=1)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), shifted.reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)


def build_model(name: str, args: argparse.Namespace):
    return TinyMixerLM(name, args)


@torch.no_grad()
def evaluate(model, args: argparse.Namespace, device: torch.device, batches: int) -> dict:
    model.eval()
    total_correct = 0
    total_targets = 0
    total_loss = 0.0
    for _ in range(batches):
        input_ids, labels = make_batch(args, device)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        total_loss += float(out.loss.detach().cpu())
        logits = model(input_ids=input_ids, use_cache=False).logits
        target_mask = labels[:, 1:] != -100
        pred = logits[:, :-1].argmax(dim=-1)
        target = input_ids[:, 1:]
        total_correct += int((pred[target_mask] == target[target_mask]).sum().detach().cpu())
        total_targets += int(target_mask.sum().detach().cpu())
    model.train()
    return {
        "eval_loss": total_loss / batches,
        "eval_accuracy": total_correct / max(1, total_targets),
        "eval_targets": total_targets,
    }


def train_one(model_name: str, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> list[dict]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(model_name, args)
    param_count = sum(p.numel() for p in model.parameters())
    model.to(device=device, dtype=dtype)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    records: list[dict] = []
    start = time.perf_counter()

    for step in range(1, args.steps + 1):
        input_ids, labels = make_batch(args, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss.detach().cpu())}")
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, args, device, args.eval_batches)
            record = {
                "type": "train_eval",
                "model": model_name,
                "task": args.task,
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "elapsed_s": time.perf_counter() - start,
                "param_count": param_count,
                "actual_seq_len": int(input_ids.shape[1]),
                **metrics,
            }
            records.append(record)

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["palindrome", "mqar", "stack"], default="palindrome")
    parser.add_argument("--models", default="kda,gdn,mamba2")
    parser.add_argument("--output", type=Path, default=Path("artifacts/synthetic_probe.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--no-short-conv", action="store_true")
    parser.add_argument("--fuse-modules", action="store_true")
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--key-space", type=int, default=64)
    parser.add_argument("--num-pairs", type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--num-stacks", type=int, default=16)
    parser.add_argument("--stack-push-prob", type=float, default=0.6)
    parser.add_argument("--mamba-head-dim", type=int, default=128)
    parser.add_argument("--mamba-state-size", type=int, default=128)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-chunk-size", type=int, default=256)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Synthetic probe is intended for CUDA; FLA training kernels are GPU-oriented.")

    dtype = getattr(torch, args.dtype)
    torch.set_float32_matmul_precision("high")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata_record = {
        "type": "metadata",
        "task": args.task,
        "models": parse_models(args.models),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flash_linear_attention": package_version("flash-linear-attention"),
        "fla_core": package_version("fla-core"),
        "gpu": torch.cuda.get_device_name(0),
    }

    with args.output.open("w", encoding="utf-8") as f:
        f.write(dumps_record(metadata_record) + "\n")
        f.flush()
        for model_name in parse_models(args.models):
            try:
                records = train_one(model_name, args, device, dtype)
            except Exception as exc:
                record = {
                    "type": "error",
                    "model": model_name,
                    "task": args.task,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
                records = [record]
                torch.cuda.empty_cache()
            for record in records:
                flush_record(f, record)


if __name__ == "__main__":
    main()
