"""Two-pass FP8 GEMM for the per-token / per-channel scaled linears (``H3_TP2_FP8_GEMM=tensor32``).

vLLM's FP8 linear path on the GB10 runs ``cutlass_scaled_mm`` with per-token activation scales (M, 1) and per-channel
weight scales (N,) applied in the kernel epilogue; on sm121 that kernel reaches ~59 TFLOPS while the same GEMM with
scalar scales reaches ~189 TFLOPS (results/20261028-fp8-gemm-bench/).  This module keeps the arithmetic of the fused
epilogue (fp32 accumulation, scales applied in fp32, bias in fp32, one rounding to the output dtype) but splits it:
a scalar-scale ``torch._scaled_mm`` with fp32 output, then one Triton pass that applies the scales (and the bias) and
rounds.  The only difference from the fused kernel is the accumulation order inside the GEMM (max abs diff 3.8e-6 on
the DiT shapes, the same as cuBLASLt rowwise vs CUTLASS).  Not bit-identical to the stock output.
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
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
        y = x * tl.load(sa_ptr + row) * tl.load(sb_ptr + cols, mask=m, other=0.0)
        if HAS_BIAS:
            y = y + tl.load(bias_ptr + cols, mask=m, other=0.0).to(tl.float32)
        if OUT_BF16:
            tl.store(y_ptr + row * N + cols, y.to(tl.bfloat16), mask=m)
        else:
            tl.store(y_ptr + row * N + cols, y.to(tl.float16), mask=m)

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    _HAVE_TRITON = False
_state = {"triton": _HAVE_TRITON}


def scale_bias_cast_torch(x32: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None, out_dtype: torch.dtype) -> torch.Tensor:
    """Reference for the Triton pass: fp32 scale, fp32 bias, one rounding."""
    y = x32 * sa.view(-1, 1).float() * sb.view(1, -1).float()
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
    block = 2048
    bias_t = bias if bias is not None else y  # unused when HAS_BIAS is False
    try:
        _scale_bias_cast[(M, triton.cdiv(N, block))](x32, sa, sb, bias_t, y, N, HAS_BIAS=bias is not None, OUT_BF16=out_dtype is torch.bfloat16, BLOCK=block)
    except Exception as exc:  # e.g. Triton cannot build its runtime helpers (no Python headers): keep the torch pass
        _state["triton"] = False
        logger.warning("h3_tp2: Triton scale pass unavailable (%s); using the torch pass (same values, ~3 passes)", exc)
        return scale_bias_cast_torch(x32, sa, sb, bias, out_dtype)
    return y


def install() -> None:
    """Route the row/column-scaled FP8 GEMMs of vLLM's linear path through the two-pass kernel; scalar-scaled calls and
    layouts ``torch._scaled_mm`` cannot take (non-row-major A, B whose transpose is not contiguous) keep the stock kernel."""
    from vllm import _custom_ops as ops

    orig = ops.cutlass_scaled_mm
    if getattr(orig, "_h3_tp2_tensor32", False):
        return
    stats = {"two_pass": 0, "stock": 0}

    def _mm(a, b, scale_a, scale_b, out_dtype, bias=None):  # noqa: ANN001
        if (scale_a.numel() == 1 and scale_b.numel() == 1) or a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn \
                or not a.is_contiguous() or not b.t().is_contiguous() or a.shape[1] % 16 or b.shape[1] % 16:
            stats["stock"] += 1
            return orig(a, b, scale_a, scale_b, out_dtype, bias)
        stats["two_pass"] += 1
        return scaled_mm_tensor32(a, b, scale_a, scale_b, out_dtype, bias)

    _mm._h3_tp2_tensor32 = True
    _mm.stats = stats
    ops.cutlass_scaled_mm = _mm
    logger.warning("h3_tp2: FP8 GEMM two-pass (scalar-scale GEMM + exact scale epilogue) ON; not bit-identical to the stock kernel")
