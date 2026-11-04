"""Two-pass FP8 GEMM for the per-token / per-channel scaled linears (``H3_TP2_FP8_GEMM=tensor32``).

vLLM's FP8 linear path on the GB10 runs ``cutlass_scaled_mm`` with per-token activation scales (M, 1) and per-channel
weight scales (N,) applied in the kernel epilogue; on sm121 that kernel reaches ~59 TFLOPS while the same GEMM with
scalar scales reaches ~189 TFLOPS (results/20261028-fp8-gemm-bench/).  This module uses the same operations as vLLM's
fused epilogue (fp32 accumulation, scale_b then scale_a in fp32, bias folded with an fma, one rounding to the output
dtype) but splits them: a scalar-scale ``torch._scaled_mm`` with fp32 output (cuBLASLt, fp32 compute), then one Triton
pass for the scales, the bias and the rounding.  The GEMM's reduction order differs from the CUTLASS kernel's (max abs
diff 3.8e-6 on the DiT shapes), so the output is not bit-identical to stock; over a 20-step render the frames differ
(SSIM ~0.81 against stock, same look).  The fp32 temporary is M x N x 4 bytes (4.2 GB for fc1 at M = 36864).

``H3_TP2_FP8_GEMM=fused`` (opt-in, speed-plan item 4) replaces both passes with one Triton FP8 GEMM whose epilogue
applies exactly the same operations to the fp32 accumulator in registers (scale_b * acc, then scale_a, bias as one fma,
one rounding): the fp32 product is never written or read.  The epilogue is bit-identical to the scale pass by
construction (tests/test_fp8_fused_epilogue_cpu.py, and on order-independent integer inputs on the GPU); the
accumulator is bit-identical to cuBLASLt's only if both reduce K in the same order with the same MMA instruction,
which scripts/fp8_epilogue_probe.py measures.  Tile config: ``H3_TP2_FP8_FUSED_CFG=BM,BN,BK,warps,stages``.

``H3_TP2_FP8_GEMM=cutlass`` (the default since 2026-11-04; ``tensor32`` = two-pass only) keeps this two-pass path for every call and adds a router ahead of the stash hooks:
the qkv, out and fc1 GEMMs of the running DiT block go to a CuTe-DSL sm120 FP8 GEMM with the same epilogue in
registers (``h3_tp2/fp8_cutlass.py``; bit-identical to two-pass where its accumulator equals cuBLASLt's, checked at
install), falling back to two-pass if its import, compile or install check fails.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("vllm_omni.h3_tp2.fp8_gemm")   # under vllm_omni so the serving process shows it

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _scale_bias_cast(x_ptr, sa_ptr, sb_ptr, bias_ptr, y_ptr, N, HAS_BIAS: tl.constexpr, OUT_BF16: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)          # M*N exceeds 2^31 on 1344x768 and continuation windows
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < N
        off = row * N + cols
        x = tl.load(x_ptr + off, mask=m, other=0.0)
        # vLLM's CUTLASS epilogue order: t = scale_b * acc, then scale_a * t, bias folded as fma(scale_a, t, bias)
        t = tl.load(sb_ptr + cols, mask=m, other=0.0) * x
        sa = tl.load(sa_ptr + row)
        if HAS_BIAS:
            y = tl.math.fma(sa, t, tl.load(bias_ptr + cols, mask=m, other=0.0).to(tl.float32))
        else:
            y = sa * t
        if OUT_BF16:
            tl.store(y_ptr + off, y.to(tl.bfloat16), mask=m)
        else:
            tl.store(y_ptr + off, y.to(tl.float16), mask=m)

    @triton.jit
    def _grouped_tile(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
        # GROUP_M row tiles share their B column tiles in L2 (the Triton matmul tutorial's swizzle); mirrors tile_coords()
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    @triton.jit
    def _fp8_mm_fused(a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, y_ptr, M, N, K, stride_am, stride_bk, stride_bn, stride_ym,
                      HAS_BIAS: tl.constexpr, OUT: tl.constexpr, EVEN_K: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
        # y[M, N] = epilogue(A[M, K] @ B[K, N]); A row-major fp8, B = W^T with W [N, K] row-major (stride_bk = 1).
        # OUT 0: the raw fp32 accumulator (probe only, scales ignored); 1: bf16; 2: fp16.
        pid_m, pid_n = _grouped_tile(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        ram = (offs_m % M).to(tl.int64)          # rows/cols past the edge reread valid data; the store masks them
        rbn = (offs_n % N).to(tl.int64)
        a_ptrs = a_ptr + ram[:, None] * stride_am + rk[None, :]
        b_ptrs = b_ptr + rbn[None, :] * stride_bn + rk[:, None] * stride_bk
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):  # K in ascending order, one accumulator, no split-K
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(a_ptrs, mask=rk[None, :] < K - k * BLOCK_K, other=0.0)
                b = tl.load(b_ptrs, mask=rk[:, None] < K - k * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk
        m_ok = offs_m < M
        n_ok = offs_n < N
        mask = m_ok[:, None] & n_ok[None, :]
        y_off = offs_m.to(tl.int64)[:, None] * stride_ym + offs_n[None, :]
        if OUT == 0:
            tl.store(y_ptr + y_off, acc, mask=mask)
        else:
            # the scale pass's arithmetic, element for element: t = scale_b * acc, y = scale_a * t (or fma with the bias)
            t = tl.load(sb_ptr + offs_n, mask=n_ok, other=0.0)[None, :] * acc
            sa = tl.broadcast_to(tl.load(sa_ptr + offs_m, mask=m_ok, other=0.0)[:, None], (BLOCK_M, BLOCK_N))
            if HAS_BIAS:
                bias = tl.load(bias_ptr + offs_n, mask=n_ok, other=0.0).to(tl.float32)
                y = tl.math.fma(sa, t, tl.broadcast_to(bias[None, :], (BLOCK_M, BLOCK_N)))
            else:
                y = sa * t
            if OUT == 1:
                tl.store(y_ptr + y_off, y.to(tl.bfloat16), mask=mask)
            else:
                tl.store(y_ptr + y_off, y.to(tl.float16), mask=mask)

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    _HAVE_TRITON = False
_state = {"triton": _HAVE_TRITON, "level": "tensor32", "fused": False}


def scale_bias_cast_torch(x32: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None, out_dtype: torch.dtype) -> torch.Tensor:
    """Eager reference for the Triton pass in the same order (scale_b first, then scale_a, then the bias) with one
    rounding to ``out_dtype``; the bias add is a separate fp32 rounding here (the kernel uses one fma)."""
    y = (sb.view(1, -1).float() * x32) * sa.view(-1, 1).float()
    if bias is not None:
        y = y + bias.view(1, -1).float()
    return y.to(out_dtype)


_UNIT: dict = {}


def unit_scale(device) -> torch.Tensor:  # noqa: ANN001
    """The fp32 scalar 1.0 the scalar-scale GEMMs pass to cuBLASLt, made once per device and synchronised before its
    first use.  A scale tensor filled right before ``torch._scaled_mm`` on the same stream was read as stale memory by
    some tiles of the cuBLASLt sm121 kernel when its 512-byte block came from fresh memory (after a few FP8 GEMMs and
    ``torch.cuda.empty_cache()``; Spark_06 2026-11-03: every wrong element off by the same factor, the old contents of
    that memory; correct with a synchronised or long-lived scale; results/20261103-cutlass-sm120/hazard.py), so the
    product never depends on a just-written scale."""
    key = (device.type, device.index) if isinstance(device, torch.device) else str(device)
    one = _UNIT.get(key)
    if one is None:
        one = torch.ones((), device=device, dtype=torch.float32)
        if one.is_cuda and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize(one.device)
        _UNIT[key] = one
    return one


def scaled_mm_tensor32(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_dtype: torch.dtype, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in for ``ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)`` on the row/column-scaled case."""
    M, N = a.shape[0], b.shape[1]
    one = unit_scale(a.device)
    x32 = torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32)
    sa = scale_a.reshape(-1).float()
    sb = scale_b.reshape(-1).float()
    if sa.numel() == 1:
        sa = sa.expand(M).contiguous()
    if sb.numel() == 1:
        sb = sb.expand(N).contiguous()
    if not _state["triton"] or out_dtype not in (torch.bfloat16, torch.float16):
        return scale_bias_cast_torch(x32, sa, sb, bias, out_dtype)
    y = torch.empty((M, N), device=a.device, dtype=out_dtype)
    _launch(x32, sa, sb, bias, y, N)
    return y


def _launch(x32, sa, sb, bias, y, N):  # noqa: ANN001
    block = 2048
    bias_t = bias if bias is not None else y  # unused when HAS_BIAS is False
    _scale_bias_cast[(x32.shape[0], triton.cdiv(N, block))](x32, sa.contiguous(), sb.contiguous(), bias_t, y, N,
                                                              HAS_BIAS=bias is not None, OUT_BF16=y.dtype is torch.bfloat16, BLOCK=block)


def epilogue_torch(acc: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None, out_dtype: torch.dtype) -> torch.Tensor:
    """Eager mirror of the fused kernel's in-register epilogue (and of the Triton scale pass): t = scale_b * acc and
    y = scale_a * t in fp32, or y = fma(scale_a, t, bias) with a single rounding (emulated in float64: the fp32 product
    is exact there; the sum is rounded twice, which differs from one fma only on exact fp32 midpoints), then one
    rounding to ``out_dtype``.  Without a bias it equals ``scale_bias_cast_torch`` bit for bit."""
    t = sb.view(1, -1).float() * acc.float()
    sa2 = sa.view(-1, 1).float()
    if bias is None:
        y = sa2 * t
    else:
        y = (sa2.double() * t.double() + bias.view(1, -1).double()).float()
    return y.to(out_dtype)


def tile_coords(pid: int, M: int, N: int, block_m: int, block_n: int, group_m: int) -> tuple[int, int]:
    """Python mirror of the kernel's ``_grouped_tile`` (program id -> (row tile, column tile))."""
    num_pid_m = -(-M // block_m)
    num_pid_n = -(-N // block_n)
    in_group = group_m * num_pid_n
    first = (pid // in_group) * group_m
    size_m = min(num_pid_m - first, group_m)
    return first + (pid % in_group) % size_m, (pid % in_group) // size_m


_DEFAULT_CFG = (128, 128, 64, 8, 3)   # BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages; 48 KB of smem per stage set
_GROUP_M = 8


def fused_config() -> tuple[int, int, int, int, int]:
    """``H3_TP2_FP8_FUSED_CFG=BM,BN,BK,warps,stages`` (from the probe's tile sweep) or the default."""
    env = os.environ.get("H3_TP2_FP8_FUSED_CFG", "").strip()
    if not env:
        return _DEFAULT_CFG
    vals = tuple(int(v) for v in env.split(","))
    if len(vals) != 5:
        raise ValueError(f"H3_TP2_FP8_FUSED_CFG needs 5 integers BM,BN,BK,warps,stages, got {env!r}")
    return vals  # type: ignore[return-value]


def launch_fused(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None,
                 y: torch.Tensor, cfg=None) -> torch.Tensor:  # noqa: ANN001
    """One-kernel FP8 GEMM + epilogue into ``y`` ([M, N]; fp32 = raw accumulator, bf16/fp16 = scaled output).
    ``a`` [M, K] row-major fp8, ``b`` [K, N] fp8 whose transpose is contiguous (vLLM's weight layout), ``sa`` [M] and
    ``sb`` [N] fp32 contiguous."""
    bm, bn, bk, warps, stages = cfg or fused_config()
    M, K = a.shape
    N = b.shape[1]
    out = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[y.dtype]
    bias_t = bias if bias is not None else y
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _fp8_mm_fused[grid](a, b, sa, sb, bias_t, y, M, N, K, a.stride(0), b.stride(0), b.stride(1), y.stride(0),
                        HAS_BIAS=bias is not None, OUT=out, EVEN_K=(K % bk == 0),
                        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=_GROUP_M, num_warps=warps, num_stages=stages)
    return y


def _vec_scales(scale_a: torch.Tensor, scale_b: torch.Tensor, M: int, N: int):
    sa = scale_a.reshape(-1).float()
    sb = scale_b.reshape(-1).float()
    if sa.numel() == 1:
        sa = sa.expand(M)
    if sb.numel() == 1:
        sb = sb.expand(N)
    return sa.contiguous(), sb.contiguous()


def scaled_mm_fused(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_dtype: torch.dtype, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in for ``scaled_mm_tensor32`` without the fp32 temporary: one Triton FP8 GEMM with the epilogue in
    registers.  Falls back to the two-pass path when the fused kernel is not usable in this process."""
    if not _state["fused"] or not a.is_cuda or out_dtype not in (torch.bfloat16, torch.float16):
        return scaled_mm_tensor32(a, b, scale_a, scale_b, out_dtype, bias)
    M, N = a.shape[0], b.shape[1]
    sa, sb = _vec_scales(scale_a, scale_b, M, N)
    y = torch.empty((M, N), device=a.device, dtype=out_dtype)
    return launch_fused(a, b, sa, sb, bias, y)


def probe_fused(device) -> bool:  # noqa: ANN001
    """Compile the fused GEMM for the configured tile and check it on (1) integer-valued inputs whose fp32 accumulator
    is exact in any summation order, where it must equal the two-pass path bit for bit (this isolates the epilogue,
    the masks and the tile mapping), and (2) random inputs, where it must agree to reduction-order noise.  On any
    failure the dispatcher keeps the two-pass path for the process."""
    if not _state["triton"]:
        return False
    try:
        g = torch.Generator(device=device).manual_seed(0)
        M, N, K = 200, 400, 512                      # partial row and column tiles; K even, as on the DiT shapes
        a = torch.randint(-4, 5, (M, K), device=device, generator=g).to(torch.float8_e4m3fn)
        w = torch.randint(-2, 3, (N, K), device=device, generator=g).to(torch.float8_e4m3fn)
        sa = torch.rand(M, 1, device=device, generator=g) * 0.01 + 1e-4
        sb = torch.rand(N, device=device, generator=g) * 0.01 + 1e-4
        _state["fused"] = True
        ok = torch.equal(scaled_mm_fused(a, w.t(), sa, sb, torch.bfloat16), scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16))
        a = (torch.randn(M, K, device=device, generator=g) * 0.5).to(torch.float8_e4m3fn)
        w = (torch.randn(N, K, device=device, generator=g) * 0.05).to(torch.float8_e4m3fn)
        y = scaled_mm_fused(a, w.t(), sa, sb, torch.bfloat16).float()
        r = scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16).float()
        ok = ok and bool(torch.isfinite(y).all()) and (y - r).abs().max().item() <= 2e-5 * r.abs().max().item() + 1e-5
        torch.cuda.synchronize(device)
        if not ok:
            logger.warning("h3_tp2: fused FP8 GEMM failed its install check; keeping the two-pass path")
    except Exception as exc:
        logger.warning("h3_tp2: fused FP8 GEMM unavailable (%s); keeping the two-pass path", exc)
        ok = False
    _state["fused"] = ok
    return ok


def probe_triton(device) -> bool:  # noqa: ANN001
    """Compile and run the scale pass once on a tiny input at install time; on failure (e.g. Triton cannot build its
    runtime helpers because the Python headers are missing) the module keeps the eager torch pass for the process."""
    if not _state["triton"]:
        return False
    try:
        x = torch.randn(4, 32, device=device); sa = torch.rand(4, device=device); sb = torch.rand(32, device=device)
        y = torch.empty((4, 32), device=device, dtype=torch.bfloat16)
        _launch(x, sa, sb, None, y, 32)
        torch.cuda.synchronize(device)
        ok = torch.equal(y, scale_bias_cast_torch(x, sa, sb, None, torch.bfloat16))
    except Exception as exc:
        logger.warning("h3_tp2: Triton scale pass unavailable (%s); using the eager torch pass", exc)
        ok = False
    _state["triton"] = ok
    return ok


_stash_hooks: list = []
_routers: list = []      # level "cutlass": fn(a, b, scale_a, scale_b, out_dtype, bias) -> tensor or None, before the hooks


def _route_summary() -> str:
    """Level "cutlass": how many of the two-pass-eligible calls the DSL kernel took (appended to the dispatch log)."""
    if _state["level"] != "cutlass":
        return ""
    from . import fp8_cutlass
    st = fp8_cutlass.stats()
    return f" (of the two-pass calls: CuTe-DSL {st['dsl']}, not taken {st['fallthrough']})"


def set_stash_hook(fn) -> None:  # noqa: ANN001
    """Register a hook that may take over a two-pass call and return a placeholder: fp8_swiglu (fc1 of the MLP) and
    fp8_consumers (qkv / out / fc2 of a DiT block).  Hooks run in registration order; each fires only for its own
    GEMMs (fp8_swiglu on a thread-local request, fp8_consumers on the block's weight tensors), so at most one does."""
    if fn not in _stash_hooks:
        _stash_hooks.append(fn)


def set_router(fn) -> None:  # noqa: ANN001
    """Register a router that may compute an eligible call itself (``fp8_cutlass.route``); it runs before the stash
    hooks and returns None for the calls it does not take."""
    if fn not in _routers:
        _routers.append(fn)


def route(a, b, scale_a, scale_b, out_dtype, bias=None, two_pass=None):  # noqa: ANN001,ANN201
    """The dispatch of a call that passed the dispatcher's eligibility test: routers, then stash hooks, then the fused
    or two-pass GEMM (``two_pass`` replaces ``scaled_mm_tensor32``: the CPU fakes in tests/ pass their eager mirror)."""
    for router in _routers:
        y = router(a, b, scale_a, scale_b, out_dtype, bias)
        if y is not None:
            return y
    for hook in _stash_hooks:
        placeholder = hook(a, b, scale_a, scale_b, out_dtype, bias)
        if placeholder is not None:
            return placeholder
    if _state["fused"]:
        return scaled_mm_fused(a, b, scale_a, scale_b, out_dtype, bias)
    return (two_pass or scaled_mm_tensor32)(a, b, scale_a, scale_b, out_dtype, bias)


def install(level: str = "tensor32") -> None:
    """Route the row/column-scaled FP8 GEMMs of vLLM's linear path through the two-pass kernel (``level`` "tensor32"),
    the fused one-kernel GEMM ("fused", falls back to two-pass if its install check fails) or two-pass plus the CuTe-DSL
    kernel for the DiT block's qkv / out / fc1 ("cutlass", same fallback; the caller then wires the block context with
    ``fp8_cutlass.install_sites``); scalar-scaled calls and layouts ``torch._scaled_mm`` cannot take (non-row-major A,
    B whose transpose is not contiguous) keep the stock kernel.  ``_state["level"]`` is the level actually running."""
    from vllm import _custom_ops as ops

    if level not in ("tensor32", "fused", "cutlass"):
        raise ValueError(f"unknown FP8 GEMM level {level!r}")

    orig = ops.cutlass_scaled_mm
    if getattr(orig, "_h3_tp2_tensor32", False):
        return
    _state["level"] = level
    stats = {"two_pass": 0, "stock": 0}
    seen: set = set()

    def _mm(a, b, scale_a, scale_b, out_dtype, bias=None):  # noqa: ANN001
        M, N = a.shape[0], b.shape[-1]
        if (a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[0]
                or (scale_a.numel() == 1 and scale_b.numel() == 1)          # per-tensor: stock is already fast
                or scale_a.numel() not in (1, M) or scale_b.numel() not in (1, N)   # block-scaled schemes: stock
                or scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32
                or a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn
                or a.device != b.device or out_dtype not in (torch.bfloat16, torch.float16)
                or (bias is not None and (bias.dim() != 1 or bias.numel() != N))
                or not a.is_contiguous() or not b.t().is_contiguous() or a.shape[1] % 16 or N % 16):
            stats["stock"] += 1
            return orig(a, b, scale_a, scale_b, out_dtype, bias)
        stats["two_pass"] += 1
        n = stats["two_pass"] + stats["stock"]
        if len(seen) < 64 and (M, N, a.shape[1]) not in seen:   # each distinct shape once, for the determinism sweep
            seen.add((M, N, a.shape[1]))
            logger.info("h3_tp2: FP8 GEMM two-pass shape M=%d N=%d K=%d", M, N, a.shape[1])
        if n == 2000 or n % 20000 == 0:
            logger.info("h3_tp2: FP8 GEMM dispatch so far: two-pass %d, stock %d%s", stats["two_pass"], stats["stock"],
                        _route_summary())
        return route(a, b, scale_a, scale_b, out_dtype, bias)

    _mm._h3_tp2_tensor32 = True
    _mm.stats = stats
    ops.cutlass_scaled_mm = _mm
    triton_ok = probe_triton("cuda") if torch.cuda.is_available() else False
    if level == "cutlass":
        from . import fp8_cutlass

        if triton_ok and fp8_cutlass.probe("cuda"):
            set_router(fp8_cutlass.route)
            _state["level"] = "cutlass"
            logger.warning("h3_tp2: FP8 GEMM cutlass (CuTe-DSL sm120 FP8 GEMM with the scale epilogue in registers for "
                           "the DiT qkv / out / fc1, two-pass for the rest; %s) ON; bit-identical to two-pass on the "
                           "install check", fp8_cutlass._state["reason"])
            return
        _state["level"] = "tensor32"
        logger.warning("h3_tp2: FP8 GEMM cutlass level unavailable (%s); two-pass path (%s)",
                       fp8_cutlass._state["reason"] if triton_ok else "Triton scale pass not usable",
                       "Triton pass" if triton_ok else "eager torch pass")
        return
    if level == "fused" and triton_ok and probe_fused("cuda"):
        logger.warning("h3_tp2: FP8 GEMM fused (one Triton FP8 GEMM, scale epilogue in registers, tile %s) ON; "
                       "not bit-identical to the stock kernel, bit-identical to two-pass only where the accumulators are "
                       "(scripts/fp8_epilogue_probe.py)", ",".join(map(str, fused_config())))
        return
    logger.warning("h3_tp2: FP8 GEMM two-pass (scalar-scale GEMM + scale epilogue, %s) ON%s; not bit-identical to the stock kernel",
                   "Triton pass" if triton_ok else "eager torch pass",
                   " (fused level requested but unavailable)" if level == "fused" else "")
