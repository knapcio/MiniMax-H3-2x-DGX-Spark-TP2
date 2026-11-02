#!/usr/bin/env python3
"""Speed-plan item 4 probe: can the per-row/per-column scaled FP8 GEMM of the DiT run without the fp32 round trip?

Run on ONE idle GB10 with the head stopped and the host compacted (peak ~20 GB at M = 42848 on fc1).  For the four
DiT GEMMs (qkv N=21504 K=5376, out N=5376 K=7168, fc1 N=28672 K=5376, fc2 N=5376 K=14336) at the per-rank row counts
of the 1344x768 (M=42848) and 960x544 (M=21952) cells it times and compares against the current two-pass path:

  a_two_pass       scalar-scale cuBLASLt GEMM with fp32 output + the Triton scale pass (what serving runs; reference)
  a_gemm_fp32      its GEMM alone            a_scale_pass   its scale pass alone
  a_swiglu         fc1 only: fp32 GEMM + fused epilogue/SwiGLU level 1 (serving's fc1; reference for *_swiglu)
  ceil_bf16_scalar cuBLASLt with scalar scales and bf16 output: the speed ceiling, NOT the model's numerics
  stock_cutlass    vLLM's cutlass_scaled_mm (per-row epilogue, ~59 TFLOPS on sm121), if vLLM imports
  b_torch_rowwise  torch._scaled_mm with scale_a [M,1], scale_b [1,N], bf16 out (try; the error is reported)
  b_F_scaled_mm    torch.nn.functional.scaled_mm with ScalingType.RowWise (the v2 entry point; try)
  c0_triton_acc    our Triton FP8 GEMM writing its raw fp32 accumulator: bitwise vs a_gemm_fp32 decides whether
                   (c) can be bit-identical to (a) on real data (same K order and MMA rounding as cuBLASLt or not)
  c_triton_fused   Triton FP8 GEMM with the scale epilogue in registers (H3_TP2_FP8_GEMM=fused), per tile config
  c_triton_swiglu  fc1 only: Triton FP8 GEMM + epilogue + SwiGLU in one kernel, per tile config
  d_lt_scalar_f32  cuBLASLt via ctypes, scalar scales, fp32 out: control that the ctypes path matches torch's kernel
  d_lt_outer_vec   cuBLASLt via ctypes with CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F (per-row A and per-column B
                   scales in cuBLASLt's own epilogue), bf16 out, every heuristic algorithm; "_nosplitk" restricts the
                   heuristic to algorithms without a split-K reduction; "+silu" adds vLLM's silu_and_mul for fc1
  (c) "scalar _scaled_mm + a fused Triton epilogue that reads the fp32 product once" is exactly (a) / a_swiglu: it
  cannot remove the fp32 write, so the fused variants put the epilogue inside the GEMM instead.

Inputs: "random" = bf16 data quantised per token / per channel as vLLM does (activation rows with a log-normal spread
of magnitudes, weights N(0, 0.02)), so the scales span realistic ranges; "exact" = integer fp8 values whose fp32
accumulators are exact in any summation order: there every difference to (a) is the epilogue's (order of the scale
multiplies, rounding), and the c variants MUST be bit-identical (a failure there is a bug).  Bit-identity on random
inputs additionally needs the same reduction order; it is reported, not assumed.

Usage (on Spark_06, head stopped):
  ~/h3-long/.venv/bin/python scripts/fp8_epilogue_probe.py --quick                 # smoke: M=21952, default tiles
  ~/h3-long/.venv/bin/python scripts/fp8_epilogue_probe.py --json results/20261103-fp8-epilogue-probe/probe.json
Options: --m 42848 21952  --shapes qkv out fc1 fc2  --inputs random exact  --iters 5  --configs "BM,BN,BK,w,s;..."
         --swiglu-configs "..."  --lt-algos 8  --lt-workspace-mib 32  --kernel-names  --skip a,b,c,d
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from h3_tp2 import fp8_gemm, fp8_swiglu  # noqa: E402

SHAPES = {"qkv": (21504, 5376), "out": (5376, 7168), "fc1": (28672, 5376), "fc2": (5376, 14336)}
E4M3 = torch.float8_e4m3fn
DEFAULT_CFGS = [(128, 128, 64, 8, 3), (128, 128, 128, 8, 2), (128, 128, 64, 4, 4), (128, 256, 64, 8, 3),
                (256, 128, 64, 8, 3), (128, 256, 128, 8, 2), (64, 256, 128, 4, 3), (128, 64, 128, 4, 3)]
# (two accumulators: 128x128 per half spills on sm_121 with Triton 3.8, so BN stays <= 64 at BM 128)
DEFAULT_SWIGLU_CFGS = [(128, 64, 64, 8, 3), (128, 64, 128, 8, 2), (128, 64, 64, 8, 4), (64, 64, 128, 4, 3),
                       (64, 128, 64, 4, 3), (256, 64, 64, 8, 3)]
DEV = "cuda"


# ----------------------------------------------------------------------------------------------------------------
# inputs

def make_inputs(M, N, K, kind, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    if kind == "exact":
        a = torch.randint(-4, 5, (M, K), device=DEV, generator=g).to(E4M3)
        w = torch.randint(-2, 3, (N, K), device=DEV, generator=g).to(E4M3)      # |acc| <= 8 K < 2^24: exact
        sa = torch.exp(torch.empty(M, 1, device=DEV).uniform_(-9.0, -2.0, generator=g))
        sb = torch.exp(torch.empty(N, device=DEV).uniform_(-9.5, -7.5, generator=g))
        return a, w, sa, sb

    def quant_rows(rows, cols, std_fn):
        q = torch.empty(rows, cols, device=DEV, dtype=E4M3)
        s = torch.empty(rows, device=DEV, dtype=torch.float32)
        for r0 in range(0, rows, 4096):
            r1 = min(rows, r0 + 4096)
            x = (torch.randn(r1 - r0, cols, device=DEV, generator=g) * std_fn(r1 - r0)).to(torch.bfloat16).float()
            sc = torch.clamp(x.abs().amax(dim=1) / 448.0, min=1.0 / (448.0 * 512.0))
            q[r0:r1] = torch.clamp(x * (1.0 / sc)[:, None], -448.0, 448.0).to(E4M3)
            s[r0:r1] = sc
        return q, s

    a, sa = quant_rows(M, K, lambda n: torch.exp(torch.randn(n, 1, device=DEV, generator=g) * 1.5))
    w, sb = quant_rows(N, K, lambda n: 0.02)
    return a, w, sa.view(M, 1), sb


# ----------------------------------------------------------------------------------------------------------------
# comparison and timing

def _key(bits):
    if bits.dtype == torch.int16:
        b = bits.to(torch.int32)
        return torch.where(b < 0, -(b & 0x7FFF), b)
    b = bits.to(torch.int64)
    return torch.where(b < 0, -(b & 0x7FFFFFFF), b)


def compare(out, ref, chunk=2048):
    """Bitwise and ulp comparison (ulps of the output dtype; +0 == -0 counts as 1 differing bit pattern, 0 ulp)."""
    if out.shape != ref.shape or out.dtype != ref.dtype:
        return {"note": f"not comparable: {tuple(out.shape)} {out.dtype} vs {tuple(ref.shape)} {ref.dtype}"}
    ib = torch.int16 if out.element_size() == 2 else torch.int32
    n_diff = 0
    max_ulp = 0
    hist = {"1": 0, "2": 0, ">2": 0}
    max_abs = 0.0
    ref_max = 0.0
    nan_mismatch = 0
    for r0 in range(0, out.shape[0], chunk):
        o, r = out[r0:r0 + chunk], ref[r0:r0 + chunk]
        bo, br = o.contiguous().view(ib), r.contiguous().view(ib)
        ne = bo != br
        nd = int(ne.sum())
        rf = r.float()
        ref_max = max(ref_max, float(torch.nan_to_num(rf, nan=0.0, posinf=0.0, neginf=0.0).abs().max()))
        if nd:
            n_diff += nd
            u = (_key(bo) - _key(br)).abs()
            max_ulp = max(max_ulp, int(u.max()))
            hist["1"] += int((u == 1).sum())
            hist["2"] += int((u == 2).sum())
            hist[">2"] += int((u > 2).sum())
            of = o.float()
            nan_mismatch += int((torch.isnan(of) != torch.isnan(rf)).sum())
            d = torch.nan_to_num((of - rf).abs(), nan=0.0)
            max_abs = max(max_abs, float(d.max()))
    total = out.numel()
    return {"bitwise": n_diff == 0, "n_diff": n_diff, "frac_diff": n_diff / total, "max_ulp": max_ulp, "ulp_hist": hist,
            "max_abs": max_abs, "rel_max_abs": max_abs / ref_max if ref_max else 0.0, "nan_mismatch": nan_mismatch}


def verdict(c):
    if not c or "bitwise" not in c:
        return "-"
    if c["bitwise"]:
        return "BIT-IDENTICAL"
    if c["max_ulp"] <= 1 and not c["nan_mismatch"]:
        return f"<=1 ulp ({c['frac_diff'] * 100:.4f} % differ)"
    return f"max {c['max_ulp']} ulp ({c['frac_diff'] * 100:.3f} % differ, >2 ulp: {c['ulp_hist']['>2']})"


def bench(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    names = []
    for ev in p.key_averages():
        t = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
        if t and ev.key not in names:
            names.append(ev.key)
    return names


# ----------------------------------------------------------------------------------------------------------------
# cuBLASLt through ctypes (the library torch already loaded)

class LtError(RuntimeError):
    pass


class _Heur(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_uint64 * 8), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


def _load_lt():
    names = ["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"]
    for n in names:                                   # already mapped by torch: dlopen returns that instance
        try:
            return ctypes.CDLL(n, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    dirs = [os.path.join(os.path.dirname(torch.__file__), "lib")]
    try:
        import nvidia
        for p in nvidia.__path__:
            dirs += [os.path.join(p, "cu13", "lib"), os.path.join(p, "cublas", "lib")]
    except ImportError:
        pass
    for d in dirs:
        for n in names:
            if os.path.exists(os.path.join(d, n)):
                return ctypes.CDLL(os.path.join(d, n), mode=ctypes.RTLD_GLOBAL)
    raise LtError("libcublasLt not found")


class Lt:
    COMPUTE_32F, R_32F, R_16F, R_16BF, R_8F_E4M3 = 68, 0, 2, 14, 28
    OP_N, OP_T = 0, 1
    D_TRANSA, D_TRANSB, D_EPILOGUE, D_BIAS_POINTER = 3, 4, 7, 8
    D_A_SCALE_POINTER, D_B_SCALE_POINTER, D_FAST_ACCUM, D_BIAS_DATA_TYPE = 17, 18, 25, 26
    D_A_SCALE_MODE, D_B_SCALE_MODE = 31, 32
    SCALE_SCALAR, SCALE_OUTER_VEC = 0, 3
    EPI_BIAS = 4
    P_MAX_WS, P_RED_MASK = 1, 3
    DTYPES = {torch.float32: 0, torch.float16: 2, torch.bfloat16: 14}

    def __init__(self, ws_bytes):
        L = self.L = _load_lt()
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        for fn, args in {
            "cublasLtCreate": [ctypes.POINTER(vp)],
            "cublasLtMatmulDescCreate": [ctypes.POINTER(vp), ctypes.c_int, ctypes.c_int],
            "cublasLtMatmulDescSetAttribute": [vp, ctypes.c_int, vp, sz],
            "cublasLtMatmulDescDestroy": [vp],
            "cublasLtMatrixLayoutCreate": [ctypes.POINTER(vp), ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64],
            "cublasLtMatrixLayoutDestroy": [vp],
            "cublasLtMatmulPreferenceCreate": [ctypes.POINTER(vp)],
            "cublasLtMatmulPreferenceSetAttribute": [vp, ctypes.c_int, vp, sz],
            "cublasLtMatmulPreferenceDestroy": [vp],
            "cublasLtMatmulAlgoGetHeuristic": [vp, vp, vp, vp, vp, vp, vp, ctypes.c_int, vp, ctypes.POINTER(ctypes.c_int)],
            "cublasLtMatmul": [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, sz, vp],
            "cublasLtMatmulAlgoConfigGetAttribute": [vp, ctypes.c_int, vp, sz, ctypes.POINTER(sz)],
        }.items():
            f = getattr(L, fn)
            f.argtypes = args
            f.restype = ctypes.c_int
        L.cublasLtGetStatusString.argtypes = [ctypes.c_int]
        L.cublasLtGetStatusString.restype = ctypes.c_char_p
        L.cublasLtGetVersion.restype = ctypes.c_size_t
        self.version = int(L.cublasLtGetVersion())
        self.handle = vp()
        self._chk(L.cublasLtCreate(ctypes.byref(self.handle)), "create")
        self.ws_bytes = ws_bytes
        self.ws = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device=DEV)

    def _chk(self, st, what):
        if st != 0:
            raise LtError(f"{what}: status {st} {self.L.cublasLtGetStatusString(st).decode()}")

    def _set(self, fn, obj, attr, cval, what):
        self._chk(fn(obj, attr, ctypes.addressof(cval), ctypes.sizeof(cval)), what)

    def plan(self, a, w, d, scale_a_ptr, scale_b_ptr, mode, bias=None, red_mask=None, n_algos=8):
        """Row-major D[M, N] = A[M, K] W[N, K]^T as column-major D^T = W A^T: cuBLASLt's A is the weight (m = N,
        TRANSA = T), its B the activation (n = M); so its A scale is our per-column scale_b and its B scale our
        per-row scale_a.  Returns a plan with the heuristic's algorithms."""
        L = self.L
        M, K = a.shape
        N = w.shape[0]
        p = {"keep": []}
        desc = ctypes.c_void_p()
        self._chk(L.cublasLtMatmulDescCreate(ctypes.byref(desc), self.COMPUTE_32F, self.R_32F), "desc")
        p["desc"] = desc
        S = L.cublasLtMatmulDescSetAttribute
        self._set(S, desc, self.D_TRANSA, ctypes.c_int32(self.OP_T), "transa")
        self._set(S, desc, self.D_TRANSB, ctypes.c_int32(self.OP_N), "transb")
        self._set(S, desc, self.D_A_SCALE_POINTER, ctypes.c_void_p(scale_b_ptr), "a scale ptr")
        self._set(S, desc, self.D_B_SCALE_POINTER, ctypes.c_void_p(scale_a_ptr), "b scale ptr")
        if mode != self.SCALE_SCALAR:
            self._set(S, desc, self.D_A_SCALE_MODE, ctypes.c_int32(mode), "a scale mode")
            self._set(S, desc, self.D_B_SCALE_MODE, ctypes.c_int32(mode), "b scale mode")
        if bias is not None:
            self._set(S, desc, self.D_EPILOGUE, ctypes.c_uint32(self.EPI_BIAS), "epilogue")
            self._set(S, desc, self.D_BIAS_POINTER, ctypes.c_void_p(bias.data_ptr()), "bias ptr")
            self._set(S, desc, self.D_BIAS_DATA_TYPE, ctypes.c_int32(self.DTYPES[bias.dtype]), "bias type")
        lay = []
        for dt, rows, cols, ld in [(self.R_8F_E4M3, K, N, K), (self.R_8F_E4M3, K, M, K),
                                   (self.DTYPES[d.dtype], N, M, N), (self.DTYPES[d.dtype], N, M, N)]:
            h = ctypes.c_void_p()
            self._chk(L.cublasLtMatrixLayoutCreate(ctypes.byref(h), dt, rows, cols, ld), "layout")
            lay.append(h)
        p["A"], p["B"], p["C"], p["D"] = lay
        pref = ctypes.c_void_p()
        self._chk(L.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "pref")
        p["pref"] = pref
        self._set(L.cublasLtMatmulPreferenceSetAttribute, pref, self.P_MAX_WS, ctypes.c_uint64(self.ws_bytes), "ws")
        if red_mask is not None:
            self._set(L.cublasLtMatmulPreferenceSetAttribute, pref, self.P_RED_MASK, ctypes.c_uint32(red_mask), "red mask")
        res = (_Heur * n_algos)()
        cnt = ctypes.c_int(0)
        st = L.cublasLtMatmulAlgoGetHeuristic(self.handle, desc, p["A"], p["B"], p["C"], p["D"], pref, n_algos,
                                              ctypes.addressof(res), ctypes.byref(cnt))
        self._chk(st, "heuristic")
        if cnt.value == 0:
            raise LtError("heuristic returned no algorithm")
        p["res"] = res
        p["n"] = cnt.value
        return p

    def algo_info(self, p, i):
        out = {"ws": int(p["res"][i].workspaceSize), "waves": round(float(p["res"][i].wavesCount), 3)}
        for name, attr, ct in [("id", 0, ctypes.c_int32), ("tile", 1, ctypes.c_uint32), ("splitk", 2, ctypes.c_int32),
                               ("reduction", 3, ctypes.c_uint32), ("stages", 6, ctypes.c_uint32)]:
            v, wr = ct(0), ctypes.c_size_t(0)
            if self.L.cublasLtMatmulAlgoConfigGetAttribute(ctypes.addressof(p["res"][i].algo), attr, ctypes.addressof(v),
                                                           ctypes.sizeof(v), ctypes.byref(wr)) == 0:
                out[name] = int(v.value)
        return out

    def run(self, p, i, a, w, d):
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        if p["res"][i].workspaceSize > self.ws_bytes:
            raise LtError("algorithm needs more workspace")
        st = self.L.cublasLtMatmul(self.handle, p["desc"], ctypes.addressof(alpha), w.data_ptr(), p["A"], a.data_ptr(), p["B"],
                                   ctypes.addressof(beta), d.data_ptr(), p["C"], d.data_ptr(), p["D"],
                                   ctypes.addressof(p["res"][i].algo), self.ws.data_ptr(), self.ws_bytes,
                                   torch.cuda.current_stream().cuda_stream)
        self._chk(st, "matmul")
        return d

    def free(self, p):
        L = self.L
        for k in ("A", "B", "C", "D"):
            L.cublasLtMatrixLayoutDestroy(p[k])
        L.cublasLtMatmulPreferenceDestroy(p["pref"])
        L.cublasLtMatmulDescDestroy(p["desc"])


# ----------------------------------------------------------------------------------------------------------------

def parse_cfgs(s, default):
    if not s:
        return default
    return [tuple(int(v) for v in c.split(",")) for c in s.split(";") if c.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, nargs="+", default=[42848, 21952])
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    ap.add_argument("--inputs", nargs="+", default=["random", "exact"], choices=["random", "exact"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--time-exact", action="store_true", help="also time on the exact inputs (default: compare only)")
    ap.add_argument("--configs", default="", help='Triton tiles "BM,BN,BK,warps,stages;..." for c_triton_fused')
    ap.add_argument("--swiglu-configs", default="", help="tiles for c_triton_swiglu (BN = columns per half)")
    ap.add_argument("--lt-algos", type=int, default=8)
    ap.add_argument("--lt-workspace-mib", type=int, default=32)
    ap.add_argument("--kernel-names", action="store_true", help="record the CUDA kernel names of each variant (profiler)")
    ap.add_argument("--skip", default="", help="comma list of families to skip: a,b,c,d,stock")
    ap.add_argument("--quick", action="store_true", help="M=21952, random inputs, default tiles only, 3 iterations")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.quick:
        args.m, args.inputs, args.iters = [21952], ["random"], 3
        args.configs = args.configs or ";".join(",".join(map(str, c)) for c in DEFAULT_CFGS[:1])
        args.swiglu_configs = args.swiglu_configs or ";".join(",".join(map(str, c)) for c in DEFAULT_SWIGLU_CFGS[:1])
    skip = set(x.strip() for x in args.skip.split(",") if x.strip())
    cfgs = parse_cfgs(args.configs, DEFAULT_CFGS)
    sw_cfgs = parse_cfgs(args.swiglu_configs, DEFAULT_SWIGLU_CFGS)

    import triton
    props = torch.cuda.get_device_properties(0)
    env = {"device": props.name, "cc": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
           "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__}
    triton_ok = fp8_gemm.probe_triton(DEV)
    fp8_gemm._state["fused"] = True
    lt = None
    if "d" not in skip:
        try:
            lt = Lt(args.lt_workspace_mib << 20)
            env["cublasLt"] = lt.version
        except Exception as exc:  # noqa: BLE001
            env["cublasLt_error"] = repr(exc)
    ops = None
    if "stock" not in skip:
        try:
            from vllm import _custom_ops as ops  # noqa: F811
        except Exception as exc:  # noqa: BLE001
            env["vllm_error"] = repr(exc)
    have_silu = ops is not None and hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "silu_and_mul")
    env["triton_scale_pass"] = triton_ok
    print(json.dumps(env))
    print("columns: shape M inputs variant | ms min / median | TFLOPS | vs reference | note")
    one = torch.ones((), device=DEV, dtype=torch.float32)
    rows = []

    def silu_mul(h):
        if have_silu:
            o = torch.empty(h.shape[0], h.shape[1] // 2, device=DEV, dtype=h.dtype)
            torch.ops._C.silu_and_mul(o, h)
            return o
        d = h.shape[1] // 2
        return torch.nn.functional.silu(h[:, :d]) * h[:, d:]

    for kind in args.inputs:
        Ms = args.m if (kind == "random" or args.time_exact) else args.m[:1]
        for M in Ms:
            for shape in args.shapes:
                N, K = SHAPES[shape]
                flop = 2.0 * M * N * K
                a, w, sa, sb = make_inputs(M, N, K, kind, seed=sum(map(ord, shape)) * 100003 + M)
                wt = w.t()
                sa_v, sb_v = sa.view(-1).contiguous(), sb.contiguous()
                timed = kind == "random" or args.time_exact
                x32 = torch._scaled_mm(a, wt, scale_a=one, scale_b=one, out_dtype=torch.float32)
                ref = fp8_gemm.scaled_mm_tensor32(a, wt, sa, sb, torch.bfloat16)
                ref_sw = fp8_swiglu.scale_swiglu(x32, sa_v, sb_v, None) if shape == "fc1" else None

                def record(variant, fn, against=None, note="", is_ref=False, out=None):
                    r = {"shape": shape, "M": M, "N": N, "K": K, "inputs": kind, "variant": variant, "note": note}
                    try:
                        if out is None:
                            out = fn()
                            torch.cuda.synchronize()
                        if against is not None:
                            r["cmp"] = compare(out, against)
                        del out
                        if timed and fn is not None:
                            r["ms_min"], r["ms_med"] = bench(fn, args.iters)
                            r["tflops"] = flop / r["ms_min"] / 1e9
                        if args.kernel_names and fn is not None:
                            r["kernels"] = kernel_names(fn)
                    except Exception as exc:  # noqa: BLE001
                        r["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
                    rows.append(r)
                    t = f"{r['ms_min']:8.2f} / {r['ms_med']:8.2f} ms {r['tflops']:6.1f} TF" if "ms_min" in r else " " * 33
                    v = "reference" if is_ref else (verdict(r.get("cmp")) if "cmp" in r else "")
                    e = r.get("error", "")
                    print(f"{shape:4s} M={M:6d} {kind:6s} {variant:34s} | {t} | {v:42s} | {note}{e}", flush=True)
                    torch.cuda.empty_cache()
                    return r

                def fail(variant, exc, prefix=""):
                    rows.append({"shape": shape, "M": M, "inputs": kind, "variant": variant, "error": repr(exc)})
                    print(f"{shape:4s} M={M:6d} {kind:6s} {variant:34s} | {prefix}{exc}", flush=True)

                if "a" not in skip:
                    record("a_two_pass", lambda: fp8_gemm.scaled_mm_tensor32(a, wt, sa, sb, torch.bfloat16), is_ref=True, out=ref)
                    record("a_gemm_fp32", lambda: torch._scaled_mm(a, wt, scale_a=one, scale_b=one, out_dtype=torch.float32),
                           is_ref=True, out=x32)
                    y_tmp = torch.empty_like(ref)
                    record("a_scale_pass", lambda: fp8_gemm._launch(x32, sa_v, sb_v, None, y_tmp, N), note="pass alone", out=ref)
                    del y_tmp
                    if shape == "fc1":
                        record("a_swiglu (serving fc1)", lambda: fp8_swiglu.scale_swiglu(
                            torch._scaled_mm(a, wt, scale_a=one, scale_b=one, out_dtype=torch.float32), sa_v, sb_v, None),
                            is_ref=True, out=ref_sw)
                    record("ceil_bf16_scalar", lambda: torch._scaled_mm(a, wt, scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
                           note="speed ceiling only, other numerics")
                if ops is not None:
                    record("stock_cutlass", lambda: ops.cutlass_scaled_mm(a, wt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16),
                           against=ref)
                if "b" not in skip:
                    record("b_torch_rowwise", lambda: torch._scaled_mm(a, wt, scale_a=sa, scale_b=sb.view(1, N),
                                                                        out_dtype=torch.bfloat16), against=ref)
                    F = torch.nn.functional
                    if hasattr(F, "scaled_mm"):
                        RW = F.ScalingType.RowWise
                        forms = [("scale_b [1,N]", sb.view(1, N)), ("scale_b [N,1]", sb.view(N, 1)), ("scale_b [N]", sb)]
                        for form, sbx in forms:   # the v2 entry point's expected scale shape varies
                            r = record("b_F_scaled_mm_rowwise", lambda sbx=sbx: F.scaled_mm(a, wt, sa, RW, sbx, RW,
                                                                                            output_dtype=torch.bfloat16),
                                       against=ref, note=form + " ")
                            if "error" not in r:
                                break
                if "c" not in skip:
                    for cfg in cfgs:
                        tag = ",".join(map(str, cfg))
                        acc = torch.empty_like(x32)
                        record(f"c0_triton_acc [{tag}]", lambda cfg=cfg: fp8_gemm.launch_fused(a, wt, sa_v, sb_v, None, acc, cfg),
                               against=x32, note="raw fp32 accumulator vs cuBLASLt's")
                        del acc
                        record(f"c_triton_fused [{tag}]", lambda cfg=cfg: fp8_gemm.launch_fused(
                            a, wt, sa_v, sb_v, None, torch.empty_like(ref), cfg), against=ref)
                    if shape == "fc1":
                        for cfg in sw_cfgs:
                            tag = ",".join(map(str, cfg))
                            record(f"c_triton_swiglu [{tag}]", lambda cfg=cfg: fp8_swiglu.gemm_swiglu(a, wt, sa_v, sb_v, None, cfg),
                                   against=ref_sw)
                if lt is not None:
                    try:
                        p = lt.plan(a, w, x32, one.data_ptr(), one.data_ptr(), Lt.SCALE_SCALAR, n_algos=1)
                        d32 = torch.empty_like(x32)
                        record("d_lt_scalar_f32 (control)", lambda: lt.run(p, 0, a, w, d32), against=x32,
                               note=json.dumps(lt.algo_info(p, 0)))
                        del d32
                        lt.free(p)
                    except Exception as exc:  # noqa: BLE001
                        fail("d_lt_scalar_f32 (control)", exc)
                    for label, mask in (("d_lt_outer_vec", None), ("d_lt_outer_vec_nosplitk", 0)):
                        try:
                            yb = torch.empty_like(ref)
                            p = lt.plan(a, w, yb, sa_v.data_ptr(), sb_v.data_ptr(), Lt.SCALE_OUTER_VEC, red_mask=mask,
                                        n_algos=args.lt_algos)
                        except Exception as exc:  # noqa: BLE001
                            fail(label, exc, "NOT AVAILABLE: ")
                            continue
                        for i in range(p["n"]):
                            r = record(f"{label} algo{i}", lambda i=i: lt.run(p, i, a, w, yb), against=ref,
                                       note=json.dumps(lt.algo_info(p, i)))
                            if shape == "fc1" and i == 0 and "error" not in r:
                                record(f"{label} algo0 +silu", lambda: silu_mul(lt.run(p, 0, a, w, yb)), against=ref_sw,
                                       note="vLLM silu_and_mul" if have_silu else "torch silu*mul")
                        lt.free(p)
                        del yb
                a = w = wt = x32 = ref = ref_sw = sa = sb = sa_v = sb_v = None   # free before the next shape
                torch.cuda.empty_cache()

    summarize(rows, args.m)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"env": env, "args": vars(args), "rows": rows, "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, f, indent=1)
        print(f"wrote {args.json}")


def summarize(rows, Ms):
    """Projected GEMM seconds per denoise step and rank: 50 layers x (qkv, out, fc1 incl. SwiGLU, fc2) at each M."""
    print("\nper-step projection (50 layers; fc1 includes its SwiGLU; best timed config/algorithm per family; "
          "random inputs):")
    for M in Ms:
        def best(shape, prefixes):
            ts = [(r["ms_min"], r["variant"]) for r in rows if r.get("M") == M and r.get("shape") == shape
                  and r.get("inputs") == "random" and "ms_min" in r and "error" not in r
                  and any(r["variant"].startswith(p) for p in prefixes)]
            return min(ts) if ts else None
        fams = {
            "a  serving (two-pass, fc1 level 1)": {"qkv": ["a_two_pass"], "out": ["a_two_pass"], "fc2": ["a_two_pass"],
                                                   "fc1": ["a_swiglu"]},
            "c  Triton fused (H3_TP2_FP8_GEMM=fused)": {"qkv": ["c_triton_fused"], "out": ["c_triton_fused"],
                                                        "fc2": ["c_triton_fused"], "fc1": ["c_triton_swiglu"]},
            "d  cuBLASLt outer-vector (+silu on fc1)": {"qkv": ["d_lt_outer_vec"], "out": ["d_lt_outer_vec"],
                                                        "fc2": ["d_lt_outer_vec"], "fc1": ["d_lt_outer_vec algo0 +silu",
                                                                                           "d_lt_outer_vec_nosplitk algo0 +silu"]},
            "ceiling: scalar bf16 cuBLASLt (no scales)": {s: ["ceil_bf16_scalar"] for s in SHAPES},
        }
        base = None
        for name, sel in fams.items():
            parts = {s: best(s, p) for s, p in sel.items()}
            if any(v is None for v in parts.values()):
                missing = [s for s, v in parts.items() if v is None]
                print(f"  M={M:6d} {name:44s}: incomplete (no timing for {', '.join(missing)})")
                continue
            step = 50 * sum(v[0] for v in parts.values()) / 1000.0
            if base is None:
                base = step
            picks = "  ".join(f"{s}={v[0]:.1f}ms[{v[1].split(' ', 1)[-1] if ' ' in v[1] else ''}]" for s, v in parts.items())
            print(f"  M={M:6d} {name:44s}: {step:6.2f} s/step ({step - base:+6.2f} s)   {picks}")


if __name__ == "__main__":
    main()
