#!/usr/bin/env python3
"""Copy 5 of the Ulysses exact exchange vs the receive-buffer quant (``H3_TP2_ULYSSES_EXACT=2``, h3_tp2/ulysses_outq.py).

Per attention layer and rank, at M local rows (P = 2, 28 local heads x 128, K = 7168):

  copy5        ulysses_exact's reverse unpack: receive [P, M, 28, 128] -> new [M, 7168] (alloc + copy)
  stock_quant  the out-projection's QuantFP8 as serving runs it (forward_native compiled by Inductor, dynamic) on that
               contiguous tensor
  stock        copy5 + stock_quant, timed as one sequence (what mode 1 runs)
  recv_quant   ulysses_outq.quant_from_recv on the receive buffer (what mode 2 runs instead of ``stock``), for a few
               num_warps
  contig_quant the same kernel on the contiguous tensor (the strided-vs-contiguous read cost)
  v_pack       for reference, the forward-side copy that mode 1 still makes when the qkv GEMM is the CuTe-DSL one
               (H3_TP2_FP8_GEMM=cutlass: no consumer-fused producer): stock's pack of v, the strided [M, 56, 128] view
               of the bf16 qkv output (row stride 21504), into send_v [P, M, 28, 128]

and checks that the FP8 tensor and the scales of ``recv_quant`` equal ``stock``'s bit for bit.  Timing: CUDA events,
median over --iters after --warmup, one stream.  Before each row count the script checks that --require-file exists
(the queue's "GPU free" marker) and stops if it does not.

Usage (Spark_06, from a scratch copy of the tree, NOT the serving tree):
  ~/h3-long/.venv/bin/python scripts/ulysses_outq_bench.py --m 10240 18880 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from h3_tp2 import ulysses_outq as oq  # noqa: E402

WS, HL, D = 2, 28, 128
K = WS * HL * D


def serving_quant():
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    cfg = VllmConfig(compilation_config=CompilationConfig(), device_config=DeviceConfig(device=torch.device("cuda")))
    with set_current_vllm_config(cfg):
        q = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN, num_token_padding=None)
    assert hasattr(q._forward_method, "_torchdynamo_orig_callable"), "QuantFP8 is not the compiled forward_native"
    return q


def timed(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def peak_mb(fn) -> float:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return peak / 2**20


def bits(t):
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[10240, 18880])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--warps", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--layers", type=int, default=50)
    ap.add_argument("--require-file", default=os.path.expanduser("~/h3-tp2-results/gpu-free"))
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    quant = serving_quant()
    rows = []
    for m in args.m:
        if args.require_file and not os.path.exists(args.require_file):
            print(f"STOP: {args.require_file} is gone", flush=True)
            break
        g = torch.Generator(device="cuda").manual_seed(m)
        x = (torch.randn(m, K, device="cuda", generator=g) * torch.rand(m, 1, device="cuda", generator=g) * 4).to(torch.bfloat16)
        recv = x.view(m, WS, HL, D).transpose(0, 1).contiguous()      # the reverse all-to-all's receive buffer
        del x
        contig = oq.unpack_rows(recv)

        def stock():
            return quant(oq.unpack_rows(recv), None, None)

        ref = stock()
        got = oq.quant_from_recv(recv)
        equal = all(a.shape == b.shape and a.stride() == b.stride() and torch.equal(bits(a), bits(b)) for a, b in zip(ref, got))
        row = {"m": m, "bitwise_equal": equal,
               "copy5_ms": timed(lambda: oq.unpack_rows(recv), args.iters, args.warmup),
               "stock_quant_ms": timed(lambda: quant(contig, None, None), args.iters, args.warmup),
               "stock_ms": timed(stock, args.iters, args.warmup)}
        for w in args.warps:
            row[f"recv_quant_w{w}_ms"] = timed(lambda: oq.quant_from_recv(recv, warps=w), args.iters, args.warmup)
            row[f"contig_quant_w{w}_ms"] = timed(lambda: oq.quant_rows(contig, m, WS, HL * D, HL * D, K, warps=w),
                                                 args.iters, args.warmup)
        best = min(args.warps, key=lambda w: row[f"recv_quant_w{w}_ms"])
        row["best_warps"] = best
        row["recv_quant_ms"] = row[f"recv_quant_w{best}_ms"]
        row["recv_quant_default_ms"] = row[f"recv_quant_w{oq._WARPS}_ms"] if oq._WARPS in args.warps else None
        dflt = row["recv_quant_default_ms"] or row["recv_quant_ms"]
        row["saved_per_layer_ms"] = row["stock_ms"] - dflt
        row["saved_per_forward_ms"] = row["saved_per_layer_ms"] * args.layers
        row["copy5_gbps"] = 2 * recv.numel() * 2 / row["copy5_ms"] / 1e6
        from h3_tp2 import ulysses_exact as ux

        qkv = torch.empty((m, 3 * WS * HL * D), dtype=torch.bfloat16, device="cuda").normal_()
        v_view = qkv[:, 2 * K:].view(m, WS * HL, D)
        send_v = torch.empty(ux.send_shape(m, WS * HL, D, WS), dtype=torch.bfloat16, device="cuda")
        row["v_pack_ms"] = timed(lambda: ux.pack_into(send_v, v_view, WS), args.iters, args.warmup)
        del qkv, v_view, send_v
        row["peak_mb_stock"] = peak_mb(stock)
        row["peak_mb_recv"] = peak_mb(lambda: oq.quant_from_recv(recv))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        rows.append(row)
        del recv, contig, ref, got
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0 if rows and all(r["bitwise_equal"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
