#!/usr/bin/env python3
"""Speed-plan item 6b probe: the FP8 scale pass fused into the consumers of the qkv / out / fc2 GEMMs
(``H3_TP2_FP8_CONSUMER_FUSE``, h3_tp2/fp8_consumers.py).

Run on ONE idle GB10 with the head stopped (and the host compacted), in the serving venv (peak ~9 GB at M = 42848 on
qkv).  For each site and row count it builds the fp32 GEMM product the serving path makes (scalar-scale cuBLASLt GEMM
of per-token / per-channel quantised random operands, ``torch._scaled_mm``), then times and compares

  old    = the Triton scale pass into bf16 (fp8_gemm._launch) + the upstream consumer from vLLM-Omni
           (fused_qk_norm_rope / indexed_gate_rms_norm_scale_shift / indexed_gate), what serving runs today
  pass   = the scale pass alone            cons = the upstream consumer alone (on the bf16 tensor)
  fused  = fp8_consumers' fused consumer on the fp32 product (qkv: q/k kernels + the v column pass)

and checks bit identity of every output (q, k and the strided v view; residual_out and modulated_out; the gate-add
output) on the timed data and on a directed set (zeros, signed zeros, subnormals, overflow to inf, NaN).  It also
prints the registers / spills Triton compiled for the stock and fused kernels and the peak memory of each path.

Usage (on Spark_06, head stopped):
  ~/h3-long/.venv/bin/python scripts/fp8_consumer_probe.py --quick
  ~/h3-long/.venv/bin/python scripts/fp8_consumer_probe.py --json ~/h3-tp2-results/fp8-consumer-probe/probe.json
Options: --m 42848 21952  --sites qkv out fc2  --iters 20  --warmup 3  --layers 50  --bw 235
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
from h3_tp2 import fp8_consumers as fc  # noqa: E402
from h3_tp2 import fp8_gemm  # noqa: E402

HIDDEN, HEADS, HEAD_DIM, ROT, FFN = 5376, 56, 128, 96, 14336
SITES = {"qkv": (3 * HEADS * HEAD_DIM, HIDDEN), "out": (HIDDEN, HEADS * HEAD_DIM), "fc2": (HIDDEN, FFN)}  # (N, K)
SAVED_COLS = {"qkv": 2 * HEADS * HEAD_DIM, "out": HIDDEN, "fc2": HIDDEN}   # bf16 columns no longer written + read
E4M3 = torch.float8_e4m3fn
BF16 = torch.bfloat16
EPS = 1e-6
U = 6   # AdaLN rows (2 conditions x 3 modalities)


def quantised_operands(M: int, N: int, K: int, g: torch.Generator):
    """vLLM-style online FP8: activations per token (log-normal row magnitudes), weights per channel (N(0, 0.02))."""
    x = torch.randn(M, K, device="cuda", generator=g, dtype=BF16) * torch.exp(torch.randn(M, 1, device="cuda", generator=g)).to(BF16)
    sa = (x.float().abs().amax(dim=1) / 448.0).clamp(min=1.0 / (448.0 * 512.0))
    a = (x.float() / sa[:, None]).clamp(-448, 448).to(E4M3)
    del x
    w = torch.randn(N, K, device="cuda", generator=g) * 0.02
    sb = w.abs().amax(dim=1).clamp(min=1e-8) / 448.0
    wq = (w / sb[:, None]).clamp(-448, 448).to(E4M3)
    del w
    return a, wq.t(), sa.contiguous(), sb.contiguous()


def product(a, b):  # noqa: ANN001
    one = torch.ones((), device="cuda")
    return torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32)


def directed_product(M: int, N: int) -> torch.Tensor:
    vals = torch.tensor([0.0, -0.0, 1e-40, -1e-40, 1.0, -1.0, 3e38, -3e38, 1e4, -1e4, 0.5000001, 1.9999999,
                         float("inf"), -float("inf"), float("nan")], device="cuda")
    x = torch.randn(M, N, device="cuda") * 1000
    x[0] = vals.repeat(N // vals.numel() + 1)[:N]            # one row of special values (it NaNs its own row only)
    x[1, ::7] = 0.0
    x[2, ::5] = -0.0
    return x


def rope_table(M: int, g: torch.Generator) -> torch.Tensor:
    """H3's packed [cos(f[:48]) | sin(f[:48])] in bf16, from random (t, h, w) positions (MiniMaxH3Rope layout)."""
    pos = torch.stack([torch.randint(0, 70, (M,), device="cuda", generator=g),
                       torch.randint(0, 48, (M,), device="cuda", generator=g),
                       torch.randint(0, 84, (M,), device="cuda", generator=g)], dim=1).float()
    inv = 10000.0 ** (-torch.arange(0, 32, 2, device="cuda", dtype=torch.float32) / 32)
    half = (pos[:, :, None] * inv[None, None, :]).reshape(M, 48)
    return torch.cat((torch.cos(half), torch.sin(half)), dim=-1).to(BF16)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, int]:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, -1
    na, nb = torch.isnan(a), torch.isnan(b)
    if not torch.equal(na, nb):
        return False, int((na != nb).sum())
    ia, ib = a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
    diff = int(((ia != ib) & ~na).sum())
    return diff == 0, diff


def time_ms(fn, warmup: int, iters: int) -> float:  # noqa: ANN001
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def peak_gb(fn) -> float:  # noqa: ANN001
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e9


def kernel_regs(jit) -> list[str]:  # noqa: ANN001
    """n_regs / n_spills of every compiled specialisation of a Triton JITFunction (best effort, Triton 3.x)."""
    out = []
    try:
        for cache in getattr(jit, "device_caches", {}).values():
            for k in cache[0].values():
                out.append(f"regs={getattr(k, 'n_regs', '?')} spills={getattr(k, 'n_spills', '?')}")
    except Exception as exc:  # noqa: BLE001
        out.append(f"unavailable ({exc})")
    return out


def make_site(site: str, M: int, x32: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, g: torch.Generator):
    """(old_fn, pass_fn, cons_fn, fused_fn, compare_fn) closures for one site on the given product."""
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fq
    from vllm_omni.diffusion.layers import indexed_modulation as im

    N = x32.shape[1]
    P = torch.empty((M, N), device="cuda", dtype=BF16)

    def scale_pass(y=P):  # noqa: ANN001,ANN202
        fp8_gemm._launch(x32, sa, sb, None, y, N)
        return y

    def stash():  # noqa: ANN202
        return fc.Stash(x32, sa, sb, None, torch.empty((M, N), device="cuda", dtype=BF16))

    if site == "qkv":
        H, D = HEADS, HEAD_DIM
        wq = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(BF16)
        wk = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(BF16)
        rope = rope_table(M, g)

        def consumer(p):  # noqa: ANN001,ANN202
            q, k, v = p.split([H * D, H * D, H * D], dim=-1)
            oq, ok = fq.fused_qk_norm_rope(q.view(M, H, D), k.view(M, H, D), wq, wk, rope, EPS)
            return oq, ok, v.view(M, H, D)

        def fused():  # noqa: ANN202
            return fc.qk_norm_rope_fused(stash(), H, H, D, wq, wk, rope, EPS)

        names = ("q", "k", "v")
    else:
        g6 = (torch.randn(U, 6 * HIDDEN, device="cuda", generator=g) * 0.1).to(BF16).chunk(6, dim=-1)
        idx = torch.randint(0, U, (M,), device="cuda", generator=g)
        residual = torch.randn(M, HIDDEN, device="cuda", generator=g).to(BF16)
        if site == "out":
            weight = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=g)).to(BF16)

            def consumer(p):  # noqa: ANN001,ANN202
                return im.indexed_gate_rms_norm_scale_shift(residual, g6[2], p, weight, g6[3], g6[4], idx, EPS)

            def fused():  # noqa: ANN202
                return fc.gate_rms_fused(stash(), residual, g6[2], weight, g6[3], g6[4], idx, EPS)

            names = ("residual_out", "modulated_out")
        else:
            def consumer(p):  # noqa: ANN001,ANN202
                return (im.indexed_gate(residual, g6[5], p, idx),)

            def fused():  # noqa: ANN202
                return (fc.gate_fused(stash(), residual, g6[5], idx),)

            names = ("out",)

    def old():  # noqa: ANN202
        return consumer(scale_pass(torch.empty((M, N), device="cuda", dtype=BF16)))

    def compare() -> dict:
        ref, got = old(), fused()
        res = {}
        for n, r, f in zip(names, ref, got):
            ok, diff = bits_equal(f, r)
            res[n] = {"bit_identical": ok, "diff_elems": diff}
        return res

    scale_pass()
    return old, scale_pass, (lambda: consumer(P)), fused, compare


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, nargs="+", default=[42848, 21952])
    ap.add_argument("--sites", nargs="+", default=list(SITES), choices=list(SITES))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--layers", type=int, default=50)
    ap.add_argument("--bw", type=float, default=235.0, help="GB/s used for the byte-count projection")
    ap.add_argument("--quick", action="store_true", help="M=21952 only, 5 iterations")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.quick:
        args.m, args.iters = [21952], 5
    assert torch.cuda.is_available(), "needs the GPU"
    if not fp8_gemm.probe_triton("cuda") or not fc._HAVE_TRITON:
        sys.exit("Triton scale pass / fused consumers unavailable in this process")
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name()}  sites {args.sites}  M {args.m}")
    results = []
    g = torch.Generator(device="cuda").manual_seed(0)
    for M in args.m:
        for site in args.sites:
            N, K = SITES[site]
            a, b, sa, sb = quantised_operands(M, N, K, g)
            x32 = product(a, b)
            del a, b
            old, spass, cons, fused, compare = make_site(site, M, x32, sa, sb, g)
            check = compare()
            t_old = time_ms(old, args.warmup, args.iters)
            t_pass = time_ms(spass, args.warmup, args.iters)
            t_cons = time_ms(cons, args.warmup, args.iters)
            t_fused = time_ms(fused, args.warmup, args.iters)
            pk_old, pk_fused = peak_gb(old), peak_gb(fused)
            del old, spass, cons, fused, compare
            # directed special values on a slice of rows (same site, same kernels)
            md = min(M, 1024)
            xd = directed_product(md, N)
            *_, compare_d = make_site(site, md, xd, sa[:md].contiguous(), sb, g)
            check_d = compare_d()
            del compare_d, xd, x32
            torch.cuda.empty_cache()
            saved_bytes = 4 * M * SAVED_COLS[site]
            row = {
                "site": site, "M": M, "N": N, "K": K,
                "old_ms": round(t_old, 3), "scale_pass_ms": round(t_pass, 3), "consumer_ms": round(t_cons, 3),
                "fused_ms": round(t_fused, 3), "saving_ms": round(t_old - t_fused, 3),
                "saving_s_per_step": round((t_old - t_fused) * args.layers / 1000, 3),
                "bytes_saved_per_call_GB": round(saved_bytes / 1e9, 3),
                "projected_ms_at_bw": round(saved_bytes / (args.bw * 1e9) * 1e3, 3),
                "peak_old_GB": round(pk_old, 2), "peak_fused_GB": round(pk_fused, 2),
                "random": check, "directed": check_d,
            }
            ok = all(v["bit_identical"] for v in check.values()) and all(v["bit_identical"] for v in check_d.values())
            row["bit_identical"] = ok
            results.append(row)
            print(f"{site:4s} M={M:6d} N={N:5d}  old {t_old:8.3f} ms (pass {t_pass:.3f} + consumer {t_cons:.3f})  "
                  f"fused {t_fused:8.3f} ms  saving {t_old - t_fused:+.3f} ms = {row['saving_s_per_step']:+.3f} s/step "
                  f"(bytes say {row['projected_ms_at_bw']:.3f} ms)  extra peak over the product {pk_old:.2f} -> {pk_fused:.2f} GB  "
                  f"bit-identical {'YES' if ok else 'NO ' + json.dumps({**check, **{'directed': check_d}})}", flush=True)
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fq
    from vllm_omni.diffusion.layers import indexed_modulation as im
    regs = {"stock _rms_norm_rope_kernel": kernel_regs(fq._rms_norm_rope_kernel),
            "fused _scale_rms_norm_rope_kernel": kernel_regs(fc._scale_rms_norm_rope_kernel),
            "stock _indexed_gate_rms_norm_scale_shift_kernel": kernel_regs(im._indexed_gate_rms_norm_scale_shift_kernel),
            "fused _scale_indexed_gate_rms_norm_scale_shift_kernel": kernel_regs(fc._scale_indexed_gate_rms_norm_scale_shift_kernel),
            "stock _indexed_gate_kernel": kernel_regs(im._indexed_gate_kernel),
            "fused _scale_indexed_gate_kernel": kernel_regs(fc._scale_indexed_gate_kernel),
            "fused _scale_cols_kernel (v)": kernel_regs(fc._scale_cols_kernel)}
    for k, v in regs.items():
        print(f"  {k}: {', '.join(v) or 'not compiled'}")
    for M in args.m:
        tot = sum(r["saving_s_per_step"] for r in results if r["M"] == M)
        print(f"M={M}: measured saving {tot:+.3f} s/step over {args.layers} layers (sites {args.sites})")
    print("ALL BIT-IDENTICAL" if all(r["bit_identical"] for r in results) else "NOT BIT-IDENTICAL: see rows above")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "results": results, "regs": regs}, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
