"""Is the scalar-scale cuBLASLt FP8 GEMM (torch._scaled_mm, fp32 out: the two-pass path's first pass) bit-stable?

PyTorch's scaled_gemm sets no CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, so cuBLASLt may pick a split-K algorithm
that reduces in place with atomics for some shapes; such a kernel returns different bits for the same inputs.  This
sweeps the DiT GEMM shapes over a range of M (rows per rank) and reports every (shape, M) whose output changes between
repeats, plus the same check for vLLM's CUTLASS kernel (per-row scales) as the control.

Usage: fp8-gemm-determinism.py [M ...]   (default: a sweep around the 960x576 and 1344x768 5 s cells)
"""
import sys
import time

import torch

shapes = [("qkv", 21504, 5376), ("out", 5376, 7168), ("fc1", 28672, 5376), ("fc2", 5376, 14336)]
Ms = [int(x) for x in sys.argv[1:]] or [36864, 40960, 49152, 57344, 61440, 65536, 67584, 69632, 71680, 73728]
reps = 6
dev = "cuda"
try:
    from vllm import _custom_ops as ops
    have_cutlass = hasattr(ops, "cutlass_scaled_mm")
except Exception:  # noqa: BLE001
    have_cutlass = False

torch.manual_seed(0)
bad = []
for name, N, K in shapes:
    b = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02).to(torch.float8_e4m3fn).t()  # K x N, column-major
    sb_row = torch.rand(N, device=dev) * 0.01 + 0.001
    for M in Ms:
        a = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn)
        sa = torch.ones((), device=dev)
        sbs = torch.ones((), device=dev)
        outs = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            outs.append(torch._scaled_mm(a, b, scale_a=sa, scale_b=sbs, out_dtype=torch.float32))
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps * 1e3
        same = all(torch.equal(outs[0], o) for o in outs[1:])
        maxdiff = max((outs[0] - o).abs().max().item() for o in outs[1:])
        ctl = "n/a"
        if have_cutlass:
            sa_row = torch.rand(M, 1, device=dev) * 0.01 + 0.001
            c = [ops.cutlass_scaled_mm(a, b, scale_a=sa_row, scale_b=sb_row, out_dtype=torch.bfloat16) for _ in range(3)]
            ctl = "same" if all(torch.equal(c[0], o) for o in c[1:]) else "DIFFERS"
        tag = "same" if same else f"DIFFERS max {maxdiff:.3g}"
        print(f"{name:4s} N={N:5d} K={K:5d} M={M:6d}: cublasLt scalar fp32-out {tag:22s} {dt:7.1f} ms | cutlass per-row {ctl}", flush=True)
        if not same:
            bad.append((name, M, maxdiff))
        del outs, a
        torch.cuda.empty_cache()
print("non-deterministic (shape, M):", bad if bad else "none")
