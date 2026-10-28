"""Two-pass FP8 GEMM for the per-token / per-channel scaled linears (``H3_TP2_FP8_GEMM=tensor32``).

vLLM's FP8 linear path on the GB10 runs ``cutlass_scaled_mm`` with per-token activation scales (M, 1) and per-channel
weight scales (N,) applied in the kernel epilogue; on sm121 that kernel reaches ~59 TFLOPS while the same GEMM with
scalar scales reaches ~189 TFLOPS (results/20261028-fp8-gemm-bench/).  This module uses the same operations as vLLM's
fused epilogue (fp32 accumulation, scale_b then scale_a in fp32, bias folded with an fma, one rounding to the output
dtype) but splits them: a scalar-scale ``torch._scaled_mm`` with fp32 output (cuBLASLt, fp32 compute), then one Triton
pass for the scales, the bias and the rounding.  The GEMM's reduction order differs from the CUTLASS kernel's (max abs
diff 3.8e-6 on the DiT shapes), so the output is not bit-identical to stock; over a 20-step render the frames differ
(SSIM ~0.81 against stock, same look).  The fp32 temporary is M x N x 4 bytes (4.2 GB for fc1 at M = 36864).
"""
from __future__ import annotations

import logging

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

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    _HAVE_TRITON = False
_state = {"triton": _HAVE_TRITON}


def scale_bias_cast_torch(x32: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None, out_dtype: torch.dtype) -> torch.Tensor:
    """Eager reference for the Triton pass in the same order (scale_b first, then scale_a, then the bias) with one
    rounding to ``out_dtype``; the bias add is a separate fp32 rounding here (the kernel uses one fma)."""
    y = (sb.view(1, -1).float() * x32) * sa.view(-1, 1).float()
    if bias is not None:
        y = y + bias.view(1, -1).float()
    return y.to(out_dtype)


def scaled_mm_tensor32(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_dtype: torch.dtype, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in for ``ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)`` on the row/column-scaled case."""
    M, N = a.shape[0], b.shape[1]
    one = torch.ones((), device=a.device, dtype=torch.float32)
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


_stash_hook = None


def set_stash_hook(fn) -> None:  # noqa: ANN001
    """fp8_swiglu registers a hook that may take over a two-pass call (fc1 of the MLP) and return a placeholder."""
    global _stash_hook
    _stash_hook = fn


def install() -> None:
    """Route the row/column-scaled FP8 GEMMs of vLLM's linear path through the two-pass kernel; scalar-scaled calls and
    layouts ``torch._scaled_mm`` cannot take (non-row-major A, B whose transpose is not contiguous) keep the stock kernel."""
    from vllm import _custom_ops as ops

    orig = ops.cutlass_scaled_mm
    if getattr(orig, "_h3_tp2_tensor32", False):
        return
    stats = {"two_pass": 0, "stock": 0}

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
        if n == 2000 or n % 20000 == 0:
            logger.info("h3_tp2: FP8 GEMM dispatch so far: two-pass %d, stock %d", stats["two_pass"], stats["stock"])
        if _stash_hook is not None:
            placeholder = _stash_hook(a, b, scale_a, scale_b, out_dtype, bias)
            if placeholder is not None:
                return placeholder
        return scaled_mm_tensor32(a, b, scale_a, scale_b, out_dtype, bias)

    _mm._h3_tp2_tensor32 = True
    _mm.stats = stats
    ops.cutlass_scaled_mm = _mm
    triton_ok = probe_triton("cuda") if torch.cuda.is_available() else False
    logger.warning("h3_tp2: FP8 GEMM two-pass (scalar-scale GEMM + scale epilogue, %s) ON; not bit-identical to the stock kernel",
                   "Triton pass" if triton_ok else "eager torch pass")
