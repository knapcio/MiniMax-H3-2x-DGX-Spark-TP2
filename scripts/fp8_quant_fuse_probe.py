"""Probe for H3_TP2_FP8_QUANT_FUSE (speed plan item 6): per site and M, the stock pair (upstream producer + the serving
per-token QuantFP8) against the fused producer (h3_tp2/fp8_quant_fuse.py): ms per call (min of N CUDA-event runs after
warm-ups), bit identity of the fp8 bytes / scales / by-products, peak memory; plus the quant arithmetic checks that
explain fused SwiGLU level 2 (fp8_swiglu's in-kernel quant and its test reference against the serving op).

Usage (Spark_06, GPU free): ~/h3-long/.venv/bin/python scripts/fp8_quant_fuse_probe.py --m 10240 18880 \
    --jsonl ~/h3-tp2-results/fp8-quant-fuse-probe/probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3_tp2 import fp8_quant_fuse as fq  # noqa: E402

BF16 = torch.bfloat16
HIDDEN, INNER, FFN, HEADS, HL, D = 5376, 7168, 14336, 56, 28, 128
LAYERS = 50


def serving_quant():
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(),
                                            device_config=DeviceConfig(device="cuda"))):
        q = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
    return q, (lambda h: q(h, None, None))


def bench(fn, reps: int, warm: int = 2) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b))
    return best


def bits_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    v = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[x.element_size()]
    return bool(torch.equal(x.contiguous().view(v), y.contiguous().view(v)))


def mismatch_count(x: torch.Tensor, y: torch.Tensor) -> int:
    v = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[x.element_size()]
    return int((x.contiguous().view(v) != y.contiguous().view(v)).sum())


def site_inputs(site: str, M: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if site in ("qkv", "fc1"):
        mag = torch.exp(torch.randn(M, 1, device="cuda", generator=g) * 1.5)
        x = (torch.randn(M, HIDDEN, device="cuda", generator=g) * mag).to(BF16)
        w = (1.0 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=g)).to(BF16)
        ada = (torch.randn(6, 6 * HIDDEN, device="cuda", generator=g)).to(BF16)
        idx = torch.randint(0, 6, (M,), device="cuda", generator=g)
        branch = (torch.randn(M, HIDDEN, device="cuda", generator=g) * 3).to(BF16) if site == "fc1" else None
        return dict(x=x, w=w, shift=ada[:, :HIDDEN], scale=ada[:, HIDDEN:2 * HIDDEN], gate=ada[:, 2 * HIDDEN:3 * HIDDEN],
                    idx=idx, branch=branch)
    if site == "out":
        recv = (torch.randn(2, M, HL, D, device="cuda", generator=g)
                * torch.exp(torch.randn(1, M, 1, 1, device="cuda", generator=g))).to(BF16)
        return dict(recv=recv)
    h = (torch.randn(M, 2 * FFN, device="cuda", generator=g) * 4).to(BF16)
    return dict(h=h)


def run_site(site: str, M: int, quant, reps: int) -> dict:
    from vllm_omni.diffusion.layers import indexed_modulation as im
    from vllm_omni.diffusion.layers.activation import SiluAndMul

    inp = site_inputs(site, M, seed=M * 13 + len(site))
    eps = 1e-6
    act = SiluAndMul()
    extra_ok = True
    if site == "qkv":
        x, w, sh, sc, idx = inp["x"], inp["w"], inp["shift"], inp["scale"], inp["idx"]
        producer = lambda: im.rms_norm_indexed_scale_shift(x, w, sh, sc, idx, eps)  # noqa: E731
        fused = lambda: fq.rms_quant(x, w, sh, sc, idx, eps)  # noqa: E731
        k = HIDDEN
    elif site == "fc1":
        x, w, sh, sc, gt, idx, br = (inp[n] for n in ("x", "w", "shift", "scale", "gate", "idx", "branch"))
        producer = lambda: im.indexed_gate_rms_norm_scale_shift(x, gt, br, w, sh, sc, idx, eps)  # noqa: E731
        fused = lambda: fq.gate_rms_quant(x, gt, br, w, sh, sc, idx, eps)  # noqa: E731
        k = HIDDEN
    elif site == "out":
        recv = inp["recv"]
        producer = lambda: fq.unpack_stock(recv, HEADS)  # noqa: E731
        fused = lambda: fq.unpack_quant(recv)  # noqa: E731
        k = INNER
    else:
        h = inp["h"]
        producer = lambda: act(h)  # noqa: E731
        fused = lambda: fq.silu_mul_quant(h)  # noqa: E731
        k = FFN

    def stock():  # noqa: ANN202
        p = producer()
        bf = p[1] if site == "fc1" else p
        q, s = quant(bf.reshape(-1, k))
        return (p[0], q, s) if site == "fc1" else (q, s)

    ref = stock()
    out = fused()
    if site == "fc1":
        extra_ok = bits_equal(out[0], ref[0])
        out, ref = out[1:], ref[1:]
    q_ok, s_ok = bits_equal(out[0], ref[0]), bits_equal(out[1], ref[1])
    rec = dict(site=site, M=M, K=k, fp8_identical=q_ok, scale_identical=s_ok, byproduct_identical=extra_ok,
               fp8_mismatch=0 if q_ok else mismatch_count(out[0], ref[0]),
               scale_mismatch=0 if s_ok else mismatch_count(out[1], ref[1]))
    del ref, out
    bf_once = producer()
    bf_once = (bf_once[1] if site == "fc1" else bf_once).reshape(-1, k)
    rec["producer_ms"] = bench(producer, reps)
    rec["quant_ms"] = bench(lambda: quant(bf_once), reps)
    rec["stock_ms"] = bench(stock, reps)
    rec["fused_ms"] = bench(fused, reps)
    del bf_once
    rec["saving_ms"] = rec["stock_ms"] - rec["fused_ms"]
    rec["saving_s_per_step"] = rec["saving_ms"] * LAYERS / 1000.0
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    stock()
    rec["stock_peak_gb"] = (torch.cuda.max_memory_allocated() - base) / 1e9
    torch.cuda.reset_peak_memory_stats()
    fused()
    rec["fused_peak_gb"] = (torch.cuda.max_memory_allocated() - base) / 1e9
    del inp
    torch.cuda.empty_cache()
    return rec


def level2_diagnosis(quant) -> dict:
    """Is fp8_swiglu level 2's in-kernel quant the serving quant, and is its test reference (dynamic=False) the
    serving op?  Exhaustive over the row maximum (every positive finite bf16) plus random rows."""
    from h3_tp2 import fp8_swiglu

    bits = torch.arange(1, 0x7F80, dtype=torch.int32)
    amax = bits.to(torch.int16).view(BF16).cuda()
    g = torch.Generator(device="cuda").manual_seed(11)
    h = ((torch.rand(amax.numel(), 512, device="cuda", generator=g) * 2 - 1) * amax.float()[:, None]).to(BF16)
    h[:, 0] = amax
    serving = quant(h)
    ours = fq.quant(h)
    static_ref = fp8_swiglu.quant_reference(h)
    res = dict(rows=int(amax.numel()),
               fused_vs_serving_scale=mismatch_count(ours[1], serving[1]),
               fused_vs_serving_fp8=mismatch_count(ours[0], serving[0]),
               level2_reference_vs_serving_scale=mismatch_count(static_ref[1], serving[1]),
               level2_reference_vs_serving_fp8=mismatch_count(static_ref[0], serving[0]))
    try:
        from vllm import _custom_ops as ops
        q_cuda, s_cuda = ops.scaled_fp8_quant(h, None, use_per_token_if_dynamic=True)
        res["forward_cuda_vs_serving_scale"] = mismatch_count(s_cuda.view(-1, 1), serving[1])
        res["forward_cuda_vs_serving_fp8"] = mismatch_count(q_cuda, serving[0])
    except Exception as exc:  # noqa: BLE001
        res["forward_cuda"] = f"{type(exc).__name__}: {exc}"
    # level 2's kernel itself on serving-like fc1 products (unit scales -> the SwiGLU of the fp32 values)
    M, width = 2048, 2 * FFN
    x32 = torch.randn(M, width, device="cuda", generator=g) * 3000
    sa = torch.rand(M, device="cuda", generator=g) * 0.01 + 0.001
    sb = torch.rand(width, device="cuda", generator=g) * 0.01 + 0.001
    hh = fp8_swiglu.scale_swiglu_reference(x32, sa, sb, None)
    q_ref, s_ref = quant(hh)
    q2, s2 = fp8_swiglu.scale_swiglu_quant(x32, sa, sb, None)
    res["level2_kernel_vs_serving_scale"] = mismatch_count(s2.view(-1, 1), s_ref)
    res["level2_kernel_vs_serving_fp8"] = mismatch_count(q2, q_ref)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[10240, 18880])
    ap.add_argument("--sites", nargs="+", default=list(fq.SITES))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-level2", action="store_true")
    ap.add_argument("--jsonl", type=str, default="")
    args = ap.parse_args()
    out = open(args.jsonl, "a") if args.jsonl else None   # noqa: SIM115

    def emit(rec: dict) -> None:
        line = json.dumps(rec)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()

    qobj, quant = serving_quant()
    emit(dict(kind="env", torch=torch.__version__, device=torch.cuda.get_device_name(), time=time.strftime("%F %T"),
              quant_forward=getattr(qobj._forward_method, "__name__", str(qobj._forward_method))))
    for M in args.m:
        total = 0.0
        for site in args.sites:
            rec = run_site(site, M, quant, args.reps)
            total += rec["saving_s_per_step"]
            emit(dict(kind="site", **rec))
        emit(dict(kind="total", M=M, saving_s_per_step=total))
    if not args.no_level2:
        emit(dict(kind="level2", **level2_diagnosis(quant)))


if __name__ == "__main__":
    main()
