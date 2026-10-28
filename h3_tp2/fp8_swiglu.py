"""Fused FP8 epilogue + SwiGLU for the DiT MLP (``H3_TP2_FP8_FUSE_SWIGLU=1``, needs ``H3_TP2_FP8_GEMM=tensor32``).

With the two-pass GEMM the MLP's fc1 does: scalar-scale GEMM -> fp32 [M, 2*ffn] -> scale/bias/round pass -> bf16
[M, 2*ffn] -> ``silu_and_mul`` -> bf16 [M, ffn].  fc1 is the largest GEMM (its fp32 output is 4.2 GB at M = 36864), so
the bf16 intermediate costs a write and a read of 2.1 GB per layer.  This module reads the fp32 product once and
writes the SwiGLU result directly: for every column c of the gate half and c + ffn of the up half it applies vLLM's
epilogue (scale_b * acc, then scale_a, bias as one fma), rounds each to bf16 exactly as the separate pass would, then
computes ``silu(gate)`` in fp32 with the same arithmetic as vLLM's ``silu_and_mul`` CUDA kernel (libdevice
``expf``, IEEE division), rounds it to bf16 as that kernel does, multiplies by ``up`` and rounds once more.  The result is meant to be bit-identical to the two-pass path followed by
``silu_and_mul``; the test checks that on random and directed inputs for this build.  The bf16 [M, 2*ffn] intermediate is
no longer written or read, though a placeholder of that size is still allocated (the layer returns it), and the fp32
product lives until the fused kernel has run.
"""
from __future__ import annotations

import logging
import threading

import torch

from . import fp8_gemm

logger = logging.getLogger("vllm_omni.h3_tp2.fp8_swiglu")

_tls = threading.local()

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _round_bf16(x):
        # round-to-nearest-even to bf16 precision, in the integer domain so the compiler cannot fold it away;
        # NaNs are passed through unchanged (the integer add would turn them into infinities or signed zeros)
        bits = x.to(tl.int32, bitcast=True)
        bits = bits + 0x7FFF + ((bits >> 16) & 1)
        rounded = (bits & -65536).to(tl.float32, bitcast=True)
        return tl.where(x != x, x, rounded)

    @triton.jit
    def _scale_swiglu(x_ptr, sa_ptr, sb_ptr, bias_ptr, y_ptr, half, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < half
        width = half * 2
        sa = tl.load(sa_ptr + row)
        g_acc = tl.load(x_ptr + row * width + cols, mask=m, other=0.0)
        u_acc = tl.load(x_ptr + row * width + half + cols, mask=m, other=0.0)
        g = tl.load(sb_ptr + cols, mask=m, other=0.0) * g_acc
        u = tl.load(sb_ptr + half + cols, mask=m, other=0.0) * u_acc
        if HAS_BIAS:
            g = tl.math.fma(sa, g, tl.load(bias_ptr + cols, mask=m, other=0.0).to(tl.float32))
            u = tl.math.fma(sa, u, tl.load(bias_ptr + half + cols, mask=m, other=0.0).to(tl.float32))
        else:
            g = sa * g
            u = sa * u
        g = _round_bf16(g)          # the bf16 fc1 output the stock path would have stored
        u = _round_bf16(u)
        u = tl.where(u == 0.0, tl.abs(u), u)   # vLLM's kernel yields a zero with the sign of silu(gate) when up is -0.0
        # vLLM act_and_mul_kernel: silu_kernel returns (T)(x / (1.0f + expf(-x))), i.e. the silu is rounded to bf16,
        # then the bf16 product silu * up is computed in fp32 and rounded once more
        s = _round_bf16(tl.math.div_rn(g, 1.0 + libdevice.exp(-g)))
        tl.store(y_ptr + row * half + cols, (s * u).to(tl.bfloat16), mask=m)

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


def scale_swiglu_reference(x32: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """The unfused path: the two-pass Triton epilogue to bf16 (fma for the bias, like the fused kernel), then vLLM's
    silu_and_mul kernel (or its native form off the GPU)."""
    if x32.is_cuda and fp8_gemm._state["triton"]:
        h = torch.empty(x32.shape, device=x32.device, dtype=torch.bfloat16)
        fp8_gemm._launch(x32, sa.reshape(-1).float(), sb.reshape(-1).float(), bias, h, x32.shape[1])
    else:
        h = fp8_gemm.scale_bias_cast_torch(x32, sa, sb, bias, torch.bfloat16)
    d = h.shape[-1] // 2
    if h.is_cuda and hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "silu_and_mul"):
        out = torch.empty(h.shape[:-1] + (d,), dtype=h.dtype, device=h.device)
        torch.ops._C.silu_and_mul(out, h)
        return out
    return torch.nn.functional.silu(h[..., :d]) * h[..., d:]


def scale_swiglu(x32: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    M, width = x32.shape
    half = width // 2
    y = torch.empty((M, half), device=x32.device, dtype=torch.bfloat16)
    block = 1024
    bias_t = bias if bias is not None else y
    _scale_swiglu[(M, triton.cdiv(half, block))](x32, sa.contiguous(), sb.contiguous(), bias_t, y, half,
                                                 HAS_BIAS=bias is not None, BLOCK=block)
    return y


def install() -> None:
    """Patch ``MiniMaxH3MLP.forward``: fc1 runs through the two-pass GEMM with its fp32 product stashed (the layer
    receives an uninitialised bf16 placeholder of the right shape), then the fused kernel produces the SwiGLU output
    straight from the stash.  If the dispatch sent fc1 to the stock kernel (nothing stashed) the stock activation runs."""
    if not _HAVE_TRITON or not fp8_gemm._state["triton"]:
        logger.warning("h3_tp2: fused FP8 SwiGLU unavailable (Triton pass not usable in this process); skipped")
        return
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as _tr

    cls = _tr.MiniMaxH3MLP
    if getattr(cls, "_h3_tp2_fused_swiglu", False):
        return
    orig_forward = cls.forward

    def forward(self, x):  # noqa: ANN001
        _tls.stash_next = True
        _tls.stash = None
        _tls.expect_n = int(getattr(self.fc1, "output_size_per_partition", 0) or 0)
        stash = None
        try:
            hidden, _ = self.fc1(x)
            stash = _tls.stash
        finally:
            _tls.stash_next = False
            _tls.stash = None
            _tls.expect_n = 0
        if stash is None:
            hidden = self.act_fn(hidden)
        else:
            lead = hidden.shape[:-1]
            del hidden                                   # the placeholder was never written; free it before the kernel
            x32, sa, sb, bias = stash
            del stash
            hidden = scale_swiglu(x32, sa, sb, bias).view(*lead, x32.shape[1] // 2)
            del x32, sa, sb, bias                        # the 4.2 GB product goes before fc2 allocates
        out, _ = self.fc2(hidden)
        return out

    cls.forward = forward
    cls._h3_tp2_fused_swiglu = True
    fp8_gemm.set_stash_hook(_stash_hook)
    logger.warning("h3_tp2: fused FP8 epilogue + SwiGLU for the MLP ON (matches two-pass + silu_and_mul on the tested inputs and build)")


def _stash_hook(a, b, scale_a, scale_b, out_dtype, bias):  # noqa: ANN001
    """Called by fp8_gemm's dispatcher before the scale pass; returns a placeholder when a stash was requested."""
    if not getattr(_tls, "stash_next", False) or out_dtype is not torch.bfloat16:
        return None
    M, N = a.shape[0], b.shape[1]
    expect = getattr(_tls, "expect_n", 0)
    if N % 2 or (expect and N != expect):              # padded fc1 (N != the logical width): leave it to the two-pass
        _tls.stash_next = False
        return None
    one = torch.ones((), device=a.device, dtype=torch.float32)
    x32 = torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32)
    sa = scale_a.reshape(-1).float()
    sb = scale_b.reshape(-1).float()
    if sa.numel() == 1:
        sa = sa.expand(M).contiguous()
    if sb.numel() == 1:
        sb = sb.expand(N).contiguous()
    _tls.stash = (x32, sa, sb, bias)
    _tls.stash_next = False
    return torch.empty((M, N), device=a.device, dtype=out_dtype)   # placeholder, never read
