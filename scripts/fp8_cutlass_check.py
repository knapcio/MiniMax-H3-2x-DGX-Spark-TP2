#!/usr/bin/env python3
"""Checks and timings for ``H3_TP2_FP8_GEMM=cutlass`` (h3_tp2/fp8_cutlass.py) on the GB10.

  host     per-call host cost of the launcher: from_dlpack for the five tensors, the compiled call, the whole
           ``fp8_cutlass.scaled_mm`` (what the dispatcher pays), each against torch._scaled_mm on the same tiny problem
  shapes   per DiT shape (qkv, out, fc1, fc2) and M: DSL vs two-pass bitwise on the probe's realistic inputs
           (scripts/fp8_epilogue_probe.make_inputs), ms of each (min of --iters CUDA-event runs after 2 warm-ups)
  swiglu   fc1: DSL bf16 output + the MLP's own SiluAndMul (vLLM-Omni, forward_cuda = torch.ops._C.silu_and_mul)
           against fused-SwiGLU level 1 (fp8_swiglu.scale_swiglu on the cuBLASLt fp32 product), bitwise, plus the time
           of both fc1 paths

Usage (Spark_06; the serving head may stay up: peak ~10 GB at M=42848):
  ~/h3-long/.venv/bin/python scripts/fp8_cutlass_check.py --m 21952 42848 --jsonl out.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from fp8_epilogue_probe import SHAPES, bench, compare, make_inputs, verdict  # noqa: E402
from h3_tp2 import fp8_cutlass, fp8_gemm, fp8_swiglu  # noqa: E402

DEV = "cuda"


def host_cost(n=2000):
    g = fp8_cutlass._state["gemm"]
    M, N, K = 128, 128, 64
    a = torch.zeros(M, K, device=DEV).to(torch.float8_e4m3fn)
    w = torch.zeros(N, K, device=DEV).to(torch.float8_e4m3fn)
    sa, sb = torch.ones(M, device=DEV), torch.ones(N, device=DEV)
    y = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    one = torch.ones((), device=DEV)
    F8, BF = g._cutlass.Float8E4M3FN, g._cutlass.BFloat16
    res = {}

    def per_call(fn):
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        t = (time.perf_counter() - t0) / n * 1e6
        torch.cuda.synchronize()
        return round(t, 1)

    res["from_dlpack_5_tensors_us"] = per_call(lambda: (g._t2(a, F8), g._t2(w, F8), g._t2(y, BF), g._t1(sa), g._t1(sb)))
    res["from_dlpack_3_tensors_us"] = per_call(lambda: (g._t2(a, F8), g._t2(y, BF), g._t1(sa)))
    args = (g._t2(a, F8), g._t2(w, F8), g._t2(y, BF), g._t1(sa), g._t1(sb))
    stream = g._cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    res["compiled_call_us"] = per_call(lambda: g.compiled(*args, stream))
    res["run_us"] = per_call(lambda: g.run(a, w, sa, sb, y))
    res["scaled_mm_us"] = per_call(lambda: fp8_cutlass.scaled_mm(a, w.t(), sa.view(-1, 1), sb, torch.bfloat16))
    res["torch_scaled_mm_us"] = per_call(lambda: torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, nargs="+", default=[21952, 42848])
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--parts", default="host,shapes,swiglu")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    out_f = open(args.jsonl, "a") if args.jsonl else None

    def emit(r):
        print(json.dumps(r), flush=True)
        if out_f:
            out_f.write(json.dumps(r) + "\n")
            out_f.flush()

    assert fp8_gemm.probe_triton(DEV)
    t0 = time.time()
    ok = fp8_cutlass.probe(DEV)
    emit({"probe": ok, "reason": fp8_cutlass._state["reason"], "probe_s": round(time.time() - t0, 2),
          "device": torch.cuda.get_device_name(), "torch": torch.__version__})
    if not ok:
        return 1
    if "host" in parts:
        emit({"host": host_cost()})
    act = None
    if "swiglu" in parts:
        from vllm_omni.diffusion.layers.activation import SiluAndMul
        act = SiluAndMul()
        emit({"silu_and_mul_forward": act._forward_method.__name__})
    one = torch.ones((), device=DEV)
    for M in args.m:
        for shape in args.shapes:
            N, K = SHAPES[shape]
            a, w, sa, sb = make_inputs(M, N, K, "random", seed=sum(map(ord, shape)) * 100003 + M)
            r = {"M": M, "shape": shape, "N": N, "K": K}
            if "shapes" in parts:
                ref = fp8_gemm.scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16)
                y = fp8_cutlass.scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
                torch.cuda.synchronize()
                r["cmp"] = compare(y, ref)
                r["verdict"] = verdict(r["cmp"])
                del y, ref
                torch.cuda.empty_cache()
                r["two_pass_ms"] = bench(lambda: fp8_gemm.scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16), args.iters)[0]
                r["dsl_ms"] = bench(lambda: fp8_cutlass.scaled_mm(a, w.t(), sa, sb, torch.bfloat16), args.iters)[0]
                r["dsl_tflops"] = round(2.0 * M * N * K / r["dsl_ms"] / 1e9, 1)
            if "swiglu" in parts and shape == "fc1":
                s_a, s_b = fp8_gemm._vec_scales(sa, sb, M, N)

                def level1():
                    x32 = torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
                    return fp8_swiglu.scale_swiglu(x32, s_a, s_b, None)

                def dsl_act():
                    return act(fp8_cutlass.scaled_mm(a, w.t(), sa, sb, torch.bfloat16))

                ref = level1()
                y = dsl_act()
                torch.cuda.synchronize()
                r["swiglu_cmp"] = compare(y, ref)
                r["swiglu_verdict"] = verdict(r["swiglu_cmp"])
                del y, ref
                torch.cuda.empty_cache()
                r["fc1_level1_ms"] = bench(level1, args.iters)[0]
                r["fc1_dsl_silu_ms"] = bench(dsl_act, args.iters)[0]
            emit(r)
            a = w = sa = sb = None
            torch.cuda.empty_cache()
    if out_f:
        out_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
