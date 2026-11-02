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

With ``H3_TP2_FP8_GEMM=fused`` (opt-in) level 1 goes one step further: the dispatcher's hook stashes fc1's operands
instead of computing the fp32 product, and ``_fp8_mm_swiglu`` runs the GEMM with two accumulators (gate column n and
up column ffn + n share each A tile) and this epilogue on the registers, so the fp32 [M, 2*ffn] product (4.9 GB at
M = 42848) is never written.  Bit-identical to the stash path when the accumulators are (see fp8_gemm).
"""
from __future__ import annotations

import logging
import os
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

    @triton.jit
    def _scale_swiglu_quant(x_ptr, sa_ptr, sb_ptr, bias_ptr, q_ptr, s_ptr, half, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
        # one program per row: the SwiGLU row stays in registers, its abs-max gives fc2's per-token FP8 scale
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
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
        g = _round_bf16(g)
        u = _round_bf16(u)
        u = tl.where(u == 0.0, tl.abs(u), u)
        s = _round_bf16(tl.math.div_rn(g, 1.0 + libdevice.exp(-g)))
        h = (s * u).to(tl.bfloat16)                       # the bf16 silu_and_mul output, never written out
        h32 = tl.where(m, h.to(tl.float32), 0.0)
        # vLLM QuantFP8.forward_native (compiled): amax over the bf16 row, scale = max(amax/448, 1/(448*512)),
        # q = clamp(x * (1/scale), -448, 448) -> fp8 e4m3 round-to-nearest-even
        amax = tl.max(tl.abs(h32), axis=0)
        scale = tl.maximum(amax / 448.0, 1.0 / (448.0 * 512.0))
        q = h32 * (1.0 / scale)
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        tl.store(q_ptr + row * half + cols, q.to(tl.float8e4nv, fp_downcast_rounding="rtne"), mask=m)
        tl.store(s_ptr + row, scale)

    from .fp8_gemm import _grouped_tile

    @triton.jit
    def _fp8_mm_swiglu(a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, y_ptr, M, half, K, stride_am, stride_bk, stride_bn,
                       HAS_BIAS: tl.constexpr, EVEN_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
        # fc1 GEMM (H3_TP2_FP8_GEMM=fused): one A tile feeds two accumulators, gate columns n and up columns half + n
        # of vLLM's merged [gate | up] weight; the epilogue is _scale_swiglu's arithmetic on the registers
        pid_m, pid_n = _grouped_tile(tl.program_id(0), M, half, BLOCK_M, BLOCK_N, GROUP_M)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        ram = (offs_m % M).to(tl.int64)
        rbn = (offs_n % half).to(tl.int64)
        a_ptrs = a_ptr + ram[:, None] * stride_am + rk[None, :]
        g_ptrs = b_ptr + rbn[None, :] * stride_bn + rk[:, None] * stride_bk
        u_ptrs = g_ptrs + half * stride_bn
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            if EVEN_K:
                a = tl.load(a_ptrs)
                bg = tl.load(g_ptrs)
                bu = tl.load(u_ptrs)
            else:
                km = K - k * BLOCK_K
                a = tl.load(a_ptrs, mask=rk[None, :] < km, other=0.0)
                bg = tl.load(g_ptrs, mask=rk[:, None] < km, other=0.0)
                bu = tl.load(u_ptrs, mask=rk[:, None] < km, other=0.0)
            acc_g = tl.dot(a, bg, acc_g)
            acc_u = tl.dot(a, bu, acc_u)
            a_ptrs += BLOCK_K
            g_ptrs += BLOCK_K * stride_bk
            u_ptrs += BLOCK_K * stride_bk
        m_ok = offs_m < M
        n_ok = offs_n < half
        sa = tl.broadcast_to(tl.load(sa_ptr + offs_m, mask=m_ok, other=0.0)[:, None], (BLOCK_M, BLOCK_N))
        g = tl.load(sb_ptr + offs_n, mask=n_ok, other=0.0)[None, :] * acc_g
        u = tl.load(sb_ptr + half + offs_n, mask=n_ok, other=0.0)[None, :] * acc_u
        if HAS_BIAS:
            bg_ = tl.load(bias_ptr + offs_n, mask=n_ok, other=0.0).to(tl.float32)
            bu_ = tl.load(bias_ptr + half + offs_n, mask=n_ok, other=0.0).to(tl.float32)
            g = tl.math.fma(sa, g, tl.broadcast_to(bg_[None, :], (BLOCK_M, BLOCK_N)))
            u = tl.math.fma(sa, u, tl.broadcast_to(bu_[None, :], (BLOCK_M, BLOCK_N)))
        else:
            g = sa * g
            u = sa * u
        g = _round_bf16(g)
        u = _round_bf16(u)
        u = tl.where(u == 0.0, tl.abs(u), u)
        s = _round_bf16(tl.math.div_rn(g, 1.0 + libdevice.exp(-g)))
        y_off = offs_m.to(tl.int64)[:, None] * half + offs_n[None, :]
        tl.store(y_ptr + y_off, (s * u).to(tl.bfloat16), mask=m_ok[:, None] & n_ok[None, :])

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


def scale_swiglu_quant(x32: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None):
    """SwiGLU output quantised per token for fc2: returns (fp8 [M, ffn], fp32 scales [M, 1])."""
    M, width = x32.shape
    half = width // 2
    q = torch.empty((M, half), device=x32.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((M, 1), device=x32.device, dtype=torch.float32)
    bias_t = bias if bias is not None else scale
    _scale_swiglu_quant[(M,)](x32, sa.contiguous(), sb.contiguous(), bias_t, q, scale, half,
                              HAS_BIAS=bias is not None, BLOCK=triton.next_power_of_2(half), num_warps=16)
    return q, scale


def swiglu_epilogue_torch(acc: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Eager mirror of the in-register SwiGLU epilogue of ``_scale_swiglu`` / ``_fp8_mm_swiglu`` on an fp32 accumulator
    [M, 2*half]: the fp8_gemm epilogue rounded to bf16 for gate and up, -0.0 up -> +0.0 (vLLM's kernel), silu in fp32
    rounded to bf16, the product rounded to bf16.  ``exp`` is torch's here and libdevice's in the kernel."""
    h = fp8_gemm.epilogue_torch(acc, sa, sb, bias, torch.bfloat16).float()
    half = h.shape[1] // 2
    g, u = h[:, :half], h[:, half:]
    u = torch.where(u == 0, u.abs(), u)
    s = (g / (1.0 + torch.exp(-g))).to(torch.bfloat16).float()
    return (s * u).to(torch.bfloat16)


class GemmOperands:
    """Stashed in place of fc1's fp32 product when the fused GEMM + SwiGLU kernel will compute it from the operands."""
    __slots__ = ("a", "b")

    def __init__(self, a: torch.Tensor, b: torch.Tensor):
        self.a, self.b = a, b

    @property
    def shape(self):  # noqa: ANN201
        return (self.a.shape[0], self.b.shape[1])


_state = {"gemm_swiglu": False}


def gemm_swiglu(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None, cfg=None) -> torch.Tensor:  # noqa: ANN001
    """fc1 + epilogue + SwiGLU in one kernel: bf16 [M, N/2] from fp8 a [M, K], b [K, N] (b.t() contiguous)."""
    bm, bn, bk, warps, stages = cfg or swiglu_config()
    M, K = a.shape
    half = b.shape[1] // 2
    y = torch.empty((M, half), device=a.device, dtype=torch.bfloat16)
    bias_t = bias if bias is not None else y
    grid = (triton.cdiv(M, bm) * triton.cdiv(half, bn),)
    _fp8_mm_swiglu[grid](a, b, sa.contiguous(), sb.contiguous(), bias_t, y, M, half, K, a.stride(0), b.stride(0), b.stride(1),
                         HAS_BIAS=bias is not None, EVEN_K=(K % bk == 0), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                         GROUP_M=fp8_gemm._GROUP_M, num_warps=warps, num_stages=stages)
    return y


_SWIGLU_DEFAULT_CFG = (128, 64, 64, 8, 3)   # BLOCK_N per half: two 128x64 accumulators, 16 KB of smem per stage


def swiglu_config() -> tuple[int, int, int, int, int]:
    """``H3_TP2_FP8_FUSED_SWIGLU_CFG=BM,BN,BK,warps,stages`` (from the probe) or the default."""
    env = os.environ.get("H3_TP2_FP8_FUSED_SWIGLU_CFG", "").strip()
    if not env:
        return _SWIGLU_DEFAULT_CFG
    vals = tuple(int(v) for v in env.split(","))
    if len(vals) != 5:
        raise ValueError(f"H3_TP2_FP8_FUSED_SWIGLU_CFG needs 5 integers BM,BN,BK,warps,stages, got {env!r}")
    return vals  # type: ignore[return-value]


def probe_gemm_swiglu(device) -> bool:  # noqa: ANN001
    """Install check for the fused fc1 kernel: on integer inputs (exact accumulators in any order) it must equal the
    two-pass GEMM + _scale_swiglu bit for bit; on random inputs it must stay within reduction-order noise."""
    try:
        g = torch.Generator(device=device).manual_seed(0)
        M, width, K = 200, 2 * 320, 512
        a = torch.randint(-4, 5, (M, K), device=device, generator=g).to(torch.float8_e4m3fn)
        w = torch.randint(-2, 3, (width, K), device=device, generator=g).to(torch.float8_e4m3fn)
        sa = torch.rand(M, device=device, generator=g) * 0.01 + 1e-3
        sb = torch.rand(width, device=device, generator=g) * 0.01 + 1e-3
        one = torch.ones((), device=device)
        x32 = torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
        ok = torch.equal(gemm_swiglu(a, w.t(), sa, sb, None).view(torch.int16), scale_swiglu(x32, sa, sb, None).view(torch.int16))
        a = (torch.randn(M, K, device=device, generator=g) * 2).to(torch.float8_e4m3fn)
        w = (torch.randn(width, K, device=device, generator=g) * 0.5).to(torch.float8_e4m3fn)
        x32 = torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
        y = gemm_swiglu(a, w.t(), sa, sb, None).float()
        r = scale_swiglu(x32, sa, sb, None).float()
        ok = ok and bool(torch.isfinite(y).all()) and (y - r).abs().max().item() <= 1e-2 * r.abs().max().item() + 1e-5
        torch.cuda.synchronize(device)
        if not ok:
            logger.warning("h3_tp2: fused FP8 GEMM + SwiGLU failed its install check; fc1 keeps the fp32 stash")
    except Exception as exc:
        logger.warning("h3_tp2: fused FP8 GEMM + SwiGLU unavailable (%s); fc1 keeps the fp32 stash", exc)
        ok = False
    _state["gemm_swiglu"] = ok
    return ok


def quant_reference(h: torch.Tensor):
    """vLLM's own per-token dynamic FP8 quant (compiled native form, as the serving path runs it)."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    with set_current_vllm_config(VllmConfig()):
        q = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
    return torch.compile(q.forward_native, dynamic=False)(h)


def _fc2_direct(fc2, q: torch.Tensor, scale: torch.Tensor, lead) -> torch.Tensor | None:  # noqa: ANN001
    """fc2 on a pre-quantised input through vLLM's own scaled-mm kernel (its apply_weights would quantise again).
    Only for the plain case this model uses: tp 1, no bias, an FP8 scaled-mm kernel; otherwise None (caller falls back)."""
    try:
        from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearKernel
    except ImportError:
        return None
    kernel = getattr(getattr(fc2, "quant_method", None), "fp8_linear", None)
    if not isinstance(kernel, FP8ScaledMMLinearKernel) or getattr(fc2, "tp_size", 1) != 1 \
            or getattr(fc2, "bias", None) is not None or getattr(fc2, "skip_bias_add", False):
        return None
    w, w_s, _, _ = kernel._get_layer_params(fc2)
    if w.shape[0] != q.shape[1]:
        return None
    return kernel.apply_scaled_mm(A=q, B=w, out_dtype=torch.bfloat16, As=scale, Bs=w_s, bias=None,
                                  output_shape=[*lead, w.shape[1]])


def install(level: int = 1) -> None:
    """Patch ``MiniMaxH3MLP.forward``: fc1 runs through the two-pass GEMM with its fp32 product stashed (the layer
    receives an uninitialised bf16 placeholder of the right shape), then the fused kernel produces the SwiGLU output
    straight from the stash.  If the dispatch sent fc1 to the stock kernel (nothing stashed) the stock activation runs.
    ``level`` 2 also quantises the SwiGLU output per token inside the kernel and feeds fc2's scaled-mm kernel directly
    (vLLM's compiled per-token quant reproduced bit for bit; see the tests)."""
    if not _HAVE_TRITON or not fp8_gemm._state["triton"]:
        logger.warning("h3_tp2: fused FP8 SwiGLU unavailable (Triton pass not usable in this process); skipped")
        return
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as _tr
    try:
        from vllm.lora.layers import BaseLayerWithLoRA
    except ImportError:
        BaseLayerWithLoRA = ()  # type: ignore[assignment,misc]  # LoRA support is absent in this vLLM build

    cls = _tr.MiniMaxH3MLP
    if not _install_on(cls, level=level, lora_layer_type=BaseLayerWithLoRA):
        return
    if level == 1 and fp8_gemm._state.get("fused") and torch.cuda.is_available():
        probe_gemm_swiglu("cuda")     # H3_TP2_FP8_GEMM=fused: fc1 runs as one GEMM + SwiGLU kernel, no fp32 product
    fp8_gemm.set_stash_hook(_stash_hook)
    logger.warning("h3_tp2: fused FP8 epilogue + SwiGLU for the MLP ON, level %d%s%s (matches two-pass + silu_and_mul%s on the tested inputs and build); LoRA-wrapped MLPs use the ordinary path",
                   level, " (+ fc2 input quant)" if level >= 2 else "",
                   ", fc1 GEMM fused into the kernel (tile %s)" % ",".join(map(str, swiglu_config())) if _state["gemm_swiglu"] else "",
                   " + vLLM's per-token quant" if level >= 2 else "")


def _install_on(cls, *, level: int, lora_layer_type) -> bool:  # noqa: ANN001
    """Install the MLP wrapper; the caller supplies vLLM's LoRA base class."""
    if getattr(cls, "_h3_tp2_fused_swiglu", False):
        return False
    orig_forward = cls.forward
    fuse_quant = level >= 2

    def forward(self, x):  # noqa: ANN001
        # The fused path deliberately discards fc1's returned placeholder.
        # A LoRA wrapper adds its delta to that result, so discarding it would
        # discard the adapter as well. Level 2 can similarly bypass fc2's
        # wrapper through _fc2_direct. Use the ordinary MLP for either case,
        # including inactive wrappers whose lifecycle we do not control.
        if isinstance(self.fc1, lora_layer_type) or isinstance(self.fc2, lora_layer_type):
            return orig_forward(self, x)
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
            if fuse_quant:
                q, scale = scale_swiglu_quant(x32, sa, sb, bias)
                del x32, sa, sb, bias
                out = _fc2_direct(self.fc2, q, scale, lead)
                if out is not None:
                    return out
                del q, scale
                # unsupported fc2 layout: the fp32 product is gone, redo fc1 through the stock path (correct, slower)
                hidden, _ = self.fc1(x)
                hidden = self.act_fn(hidden)
            elif isinstance(x32, GemmOperands):          # H3_TP2_FP8_GEMM=fused: no fp32 product was ever made
                hidden = gemm_swiglu(x32.a, x32.b, sa, sb, bias).view(*lead, x32.shape[1] // 2)
                del x32, sa, sb, bias
            else:
                hidden = scale_swiglu(x32, sa, sb, bias).view(*lead, x32.shape[1] // 2)
                del x32, sa, sb, bias                    # the 4.2 GB product goes before fc2 allocates
        out, _ = self.fc2(hidden)
        return out

    cls.forward = forward
    cls._h3_tp2_fused_swiglu = True
    return True


def _stash_hook(a, b, scale_a, scale_b, out_dtype, bias):  # noqa: ANN001
    """Called by fp8_gemm's dispatcher before the scale pass; returns a placeholder when a stash was requested."""
    if not getattr(_tls, "stash_next", False) or out_dtype is not torch.bfloat16:
        return None
    M, N = a.shape[0], b.shape[1]
    expect = getattr(_tls, "expect_n", 0)
    if N % 2 or (expect and N != expect):              # padded fc1 (N != the logical width): leave it to the two-pass
        _tls.stash_next = False
        return None
    sa = scale_a.reshape(-1).float()
    sb = scale_b.reshape(-1).float()
    if sa.numel() == 1:
        sa = sa.expand(M).contiguous()
    if sb.numel() == 1:
        sb = sb.expand(N).contiguous()
    if _state["gemm_swiglu"]:                                     # level 1 under H3_TP2_FP8_GEMM=fused only
        _tls.stash = (GemmOperands(a, b), sa, sb, bias)          # the forward runs GEMM + SwiGLU in one kernel
    else:
        one = torch.ones((), device=a.device, dtype=torch.float32)
        x32 = torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32)
        _tls.stash = (x32, sa, sb, bias)
    _tls.stash_next = False
    return torch.empty((M, N), device=a.device, dtype=out_dtype)   # placeholder, never read
