#!/usr/bin/env python3
"""Compare two saved diffusion outputs (H3_TP2_SAVE_LATENTS .pt files): bit-equality and numeric distance per tensor.

    python scripts/latents_diff.py a.pt b.pt

The first request of a boot gives a different latents digest on every boot (2026-11-02), so cross-boot A/Bs of a
lossless lever are judged here against the stock-vs-stock cross-boot floor, not by sha256.
"""
import sys

import torch


def tensors(obj, prefix=""):
    if isinstance(obj, torch.Tensor):
        yield prefix or "t", obj
    elif isinstance(obj, (tuple, list)):
        for i, x in enumerate(obj):
            yield from tensors(x, f"{prefix}[{i}]")
    elif isinstance(obj, dict):
        for k, x in obj.items():
            yield from tensors(x, f"{prefix}.{k}")


def main():
    a = dict(tensors(torch.load(sys.argv[1], map_location="cpu")))
    b = dict(tensors(torch.load(sys.argv[2], map_location="cpu")))
    for k in a:
        if k not in b:
            print(f"{k}: only in a"); continue
        x, y = a[k].float(), b[k].float()
        if x.shape != y.shape:
            print(f"{k}: shape {tuple(x.shape)} vs {tuple(y.shape)}"); continue
        d = (x - y).abs()
        eq = torch.equal(a[k], b[k])
        rel = (d.norm() / x.norm().clamp_min(1e-12)).item()
        print(f"{k}: shape {tuple(x.shape)} bit-equal {eq} max|d| {d.max().item():.3e} mean|d| {d.mean().item():.3e} rel-l2 {rel:.3e} "
              f"frac-differing {(d > 0).float().mean().item():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
