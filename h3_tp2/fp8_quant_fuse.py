"""Per-token FP8 activation quant fused into the producing kernels (``H3_TP2_FP8_QUANT_FUSE``, off by default;
speed plan 2026-11-02 item 6).

Every FP8 linear of a DiT block (vLLM-Omni v0.30.0 ``a8576ccb7``) quantises its bf16 input per token before the GEMM:
``FP8ScaledMMLinearKernel.apply_weights`` -> ``self.quant_fp8(x_2d)`` (``QuantFP8``, dynamic, per token).  With the
diffusion vLLM config (``CompilationConfig()``: mode VLLM_COMPILE, custom ops "none") the op runs its
``forward_native`` compiled by Inductor with dynamic shapes: one Triton reduction kernel per call that reads the bf16
row twice and writes the fp8 row and an fp32 scale (``triton_red_fused__to_copy_abs_clamp_max_mul_reciprocal``).  Its
arithmetic, per row x (bf16) of K elements:

    amax  = max_k |x_k|                    (exact: max is order independent)
    scale = maximum(amax / 448.0, 1/(448*512))   (fp32; the 448 is a runtime fp32 operand of the division)
    q_k   = fp8_e4m3_rn(clamp(float(x_k) * (1.0 / scale), -448, 448))

The four quantised activations of a block and the kernel that produces each of them today (the default level,
``H3_TP2_FP8_GEMM=cutlass``):

========  ====================================================  =====================================  ==============
site      producer of the GEMM input                            one program sees the whole row?        width K
========  ====================================================  =====================================  ==============
qkv       ``rms_norm_indexed_scale_shift`` (Triton, block :950)  yes: one program per row (RMS over it)   5376
out       Ulysses reverse unpack copy (``ulysses_exact``)        yes once the copy is a per-row kernel    7168
fc1       ``indexed_gate_rms_norm_scale_shift`` (Triton, :970)   yes: one program per row (RMS over it)   5376
fc2       ``SiluAndMul`` on fc1's bf16 output (``silu_and_mul``)  yes once SwiGLU is a per-row kernel      14336
========  ====================================================  =====================================  ==============

None of the GEMM epilogues is a producer of a quantised tensor (the DSL kernel's qkv / out / fc1 outputs go to norm,
RoPE, attention and modulation kernels first), and a GEMM tile never sees a whole row, so the fusion sits in the four
producers above.  Each fused producer computes exactly the bf16 value the stock producer stores (same kernel source,
same launch shape, the same ``.to(bf16)`` rounding), keeps it in registers and applies the quant arithmetic above to
the whole row: it writes the fp8 row and the scale instead of the bf16 row.  The layer still receives a bf16 tensor
of the right shape, an uninitialised placeholder; the per-token quant of that layer (the hook on ``QuantFP8.forward``)
recognises the placeholder (same QuantFP8 instance as planned, same data pointer, shape and strides, no static scale)
and returns the stashed (fp8, scale) instead of quantising.  Saved per element: the bf16 write and the quant kernel's
two bf16 reads and its fp8 write, against one fp8 write in the producer (fc2: also the bf16 round trip of the SwiGLU
output), i.e. 4 B per element of the four activations.

Bit identity: the producers are copies of the upstream Triton kernels (the stock ones run unchanged everywhere else),
the quant is the Inductor kernel's operations in its order (fp32 ``/`` with a runtime divisor, ``1.0 / scale``, the
NaN-propagating maximum/minimum of ``torch._inductor.runtime.triton_helpers``, fp8 conversion round to nearest even).
What the CPU cannot check -- that the Triton compiler gives the copied RMS-norm reduction the same reduction tree
next to the extra fp8 store, and the division instruction -- is checked on the GPU: on the first call per site and
width the fused producer and the stock pair (stock producer + the layer's own QuantFP8 instance) run on the first
rows and are compared bit for bit (fp8 bytes, scale bits, by-products); a mismatch disables the site for the process.
``H3_TP2_FP8_QUANT_FUSE_VERIFY=1`` (diagnosis, slow) also compares every fused call with the stock pair.

Safety: a site is planned only for a DiT block whose layers are FP8 linears with a dynamic per-token ``QuantFP8``
(no static or upper-bound scale, no token padding), tp 1, not LoRA-wrapped, with the reviewed upstream sources
(``fp8_consumers.EXPECTED`` fingerprints: block, attention and MLP forwards, the modulation kernels and launchers),
no VSA gate (it would read the qkv input a second time), and with ``VLLM_BATCH_INVARIANT`` off.  If a stash is not
consumed by its quant (any other quant call while it is pending, the layer's forward returning without consuming it,
the block ending), it is "materialised" first -- the stock producer is run into the placeholder -- and a layer that
ran on the placeholder is run again, so the fallback is the stock path bit for bit.

Sites (``H3_TP2_FP8_QUANT_FUSE``): ``1``/``all`` = qkv,out,fc1,fc2; a comma list selects (for bisecting); ``0`` off.
qkv needs nothing else; out needs ``H3_TP2_ULYSSES_EXACT=1`` (its reverse unpack is the producer; with ``=2`` there is
no unpack copy, ``h3_tp2/ulysses_outq.py`` serves the out-projection quant from the receive buffer and the out site is
skipped automatically, logged once, also when it was listed explicitly); fc1 needs the
stock ``indexed_gate_rms_norm_scale_shift`` (not fp8_consumers' ``out`` site, i.e. the cutlass level); fc2 needs
fc1's bf16 output to reach the MLP's ``SiluAndMul`` (the cutlass level; at tensor32 fused SwiGLU level 1 owns it).
"""
from __future__ import annotations

import functools
import logging
import os
import sys
import threading
from typing import Any, Callable

import torch

from . import fp8_gemm

logger = logging.getLogger("vllm_omni.h3_tp2.fp8_quant_fuse")

SITES = ("qkv", "out", "fc1", "fc2")
FP8_MAX = 448.0
MIN_SCALE = 1.0 / (FP8_MAX * 512.0)        # vLLM's _FP8_MIN_SCALING_FACTOR
_SELFCHECK_ROWS = 64
_ROW_WARPS = 8                              # upstream indexed_modulation launchers
_SWIGLU_WARPS = 16
E4M3 = torch.float8_e4m3fn

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    from .fp8_swiglu import _round_bf16

    @triton.jit
    def _nan_maximum(a, b):
        # torch._inductor.runtime.triton_helpers.maximum (NaN propagates)
        mask = (a > b) | (a != a)
        return tl.where(mask, a, b)

    @triton.jit
    def _nan_minimum(a, b):
        # torch._inductor.runtime.triton_helpers.minimum (NaN propagates)
        mask = (a < b) | (a != a)
        return tl.where(mask, a, b)

    @triton.jit
    def _quant_values(h32, scale, MAXV: tl.constexpr):
        # the elementwise half of Inductor's QuantFP8.forward_native kernel: x * (1 / scale), clamp, fp8 RNE
        inv = 1.0 / scale
        y = h32 * inv
        y = _nan_maximum(y, -MAXV)
        y = _nan_minimum(y, MAXV)
        return y.to(tl.float8e4nv)

    @triton.jit
    def _row_scale(amax, fp8_max, MIN_SF: tl.constexpr):
        # scale = maximum(amax / fp8_max, MIN_SF); fp8_max is a runtime fp32 operand, as in the serving kernel
        return _nan_maximum(amax / fp8_max, MIN_SF)

    @triton.jit
    def _abs_max(h32, mask):
        # amax over the valid lanes (masked lanes 0: |x| >= 0, so the maximum is unchanged), NaN propagating
        return tl.reduce(tl.where(mask, tl.abs(h32), 0.0), 0, _nan_maximum)

    @triton.jit
    def _vec8(columns):
        # the same column indices with a provable contiguity of 8 only: Triton coalesces the fp8 store of the row to
        # 8 elements per thread (64-bit stores) instead of 16, so every memory op of the kernel asks for upstream's
        # layout (8 bf16 per thread) and the RMS reduction keeps upstream's per-thread order and shuffle tree
        return (columns // 8) * 8 + columns % 8

    @triton.jit
    def _bf16_row(output, ph_ptrs, mask, RELOAD: tl.constexpr):
        # The bf16 value stock stores, as fp32.  RELOAD 0: rounded in registers (the bf16 row is never written).
        # RELOAD 1: upstream's own store of the row into the placeholder, a CTA barrier, then the row read back, so the
        # part of the kernel before the store is upstream's code with upstream's only memory operations (if the
        # extra quant work made the compiler pick another layout for the RMS reduction, this mode still matches).
        if RELOAD:
            tl.store(ph_ptrs, output, mask=mask)
            tl.debug_barrier()
            return tl.load(ph_ptrs, mask=mask, other=0.0).to(tl.float32)
        else:
            return output.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _rms_norm_indexed_scale_shift_quant_kernel(
        q_ptr,
        s_ptr,
        ph_ptr,
        x_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        stride_x_row,
        stride_shift_row,
        stride_scale_row,
        stride_indices,
        row_offset,
        fp8_max,
        block_n: tl.constexpr,
        MIN_SF: tl.constexpr,
        MAXV: tl.constexpr,
        RELOAD: tl.constexpr,
    ):
        # Copy of vLLM-Omni a8576ccb7 _rms_norm_indexed_scale_shift_kernel (indexed_modulation.py:91-126, Apache-2.0)
        # up to ``output``; the bf16 store becomes the bf16 rounding + the per-token FP8 quant of the row.
        row = tl.program_id(0) + row_offset
        columns = tl.arange(0, block_n)
        mask = columns < hidden_size
        index = tl.load(indices_ptr + row * stride_indices)

        x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / hidden_size
        # Keep RMSNorm and AdaLN intermediates in FP32 on every architecture.
        # Rounding the normalized value to BF16 before the affine transform
        # changes the fused computation, including values at rounding boundaries.
        normalized = x * tl.rsqrt(variance + eps) * weight
        shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)
        output = normalized * (1.0 + scale) + shift

        h32 = _bf16_row(output, ph_ptr + row * hidden_size + columns, mask, RELOAD)   # the value stock stores
        row_scale = _row_scale(_abs_max(h32, mask), fp8_max, MIN_SF)
        q = _quant_values(h32, row_scale, MAXV)
        tl.store(q_ptr + row.to(tl.int64) * hidden_size + _vec8(columns), q, mask=mask)
        tl.store(s_ptr + row, row_scale)

    @triton.jit
    def _indexed_gate_rms_norm_scale_shift_quant_kernel(
        residual_out_ptr,
        q_ptr,
        s_ptr,
        ph_ptr,
        residual_ptr,
        gate_ptr,
        branch_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        stride_residual_row,
        stride_gate_row,
        stride_branch_row,
        stride_shift_row,
        stride_scale_row,
        stride_indices,
        row_offset,
        fp8_max,
        block_n: tl.constexpr,
        MIN_SF: tl.constexpr,
        MAXV: tl.constexpr,
        RELOAD: tl.constexpr,
    ):
        # Copy of vLLM-Omni a8576ccb7 _indexed_gate_rms_norm_scale_shift_kernel (indexed_modulation.py:129-173,
        # Apache-2.0); residual_out is stored as upstream, the modulated bf16 store becomes the per-token FP8 quant.
        row = tl.program_id(0) + row_offset
        columns = tl.arange(0, block_n)
        mask = columns < hidden_size
        index = tl.load(indices_ptr + row * stride_indices)

        residual = tl.load(residual_ptr + row * stride_residual_row + columns, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + row * stride_branch_row + columns, mask=mask, other=0.0).to(tl.float32)
        # Match the BF16 residual value consumed by the unfused RMSNorm path.
        updated = (residual + gate * branch).to(tl.bfloat16).to(tl.float32)
        tl.store(residual_out_ptr + row * hidden_size + columns, updated, mask=mask)

        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(updated * updated, axis=0) / hidden_size
        normalized = updated * tl.rsqrt(variance + eps) * weight
        shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)
        output = normalized * (1.0 + scale) + shift

        h32 = _bf16_row(output, ph_ptr + row * hidden_size + columns, mask, RELOAD)
        row_scale = _row_scale(_abs_max(h32, mask), fp8_max, MIN_SF)
        q = _quant_values(h32, row_scale, MAXV)
        tl.store(q_ptr + row.to(tl.int64) * hidden_size + _vec8(columns), q, mask=mask)
        tl.store(s_ptr + row, row_scale)

    @triton.jit
    def _unpack_quant_kernel(recv_ptr, q_ptr, s_ptr, rows, C, fp8_max, P: tl.constexpr, BLOCK_C: tl.constexpr,
                             MIN_SF: tl.constexpr, MAXV: tl.constexpr):
        # Ulysses reverse unpack + quant: the receive buffer [P, rows, C] (C = local heads x head_dim) holds row m of
        # source p at recv[p, m]; the out-projection input row m is [recv[0, m], ..., recv[P-1, m]] (P*C wide).
        row = tl.program_id(0).to(tl.int64)
        p = tl.arange(0, P)
        c = tl.arange(0, BLOCK_C)
        mask = (c[None, :] < C) & (p[:, None] < P)
        src = p[:, None].to(tl.int64) * rows * C + row * C + c[None, :]
        h32 = tl.load(recv_ptr + src, mask=mask, other=0.0).to(tl.float32)
        amax = tl.reduce(tl.reduce(tl.where(mask, tl.abs(h32), 0.0), 1, _nan_maximum), 0, _nan_maximum)
        row_scale = _row_scale(amax, fp8_max, MIN_SF)
        q = _quant_values(h32, row_scale, MAXV)
        tl.store(q_ptr + row * (P * C) + p[:, None] * C + c[None, :], q, mask=mask)
        tl.store(s_ptr + row, row_scale)

    @triton.jit
    def _silu_mul_quant_kernel(x_ptr, q_ptr, s_ptr, d, stride_x, fp8_max, BLOCK: tl.constexpr,
                               MIN_SF: tl.constexpr, MAXV: tl.constexpr):
        # silu_and_mul on a bf16 [rows, 2d] row (the arithmetic of vLLM's act_and_mul_kernel, as fp8_swiglu level 1
        # reproduces it: silu in fp32 rounded to bf16, the bf16 product), then the per-token FP8 quant of the row.
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        m = cols < d
        g = tl.load(x_ptr + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + row * stride_x + d + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.where(u == 0.0, tl.abs(u), u)   # vLLM's kernel yields a zero with the sign of silu(gate) when up is -0.0
        s = _round_bf16(tl.math.div_rn(g, 1.0 + libdevice.exp(-g)))
        h32 = (s * u).to(tl.bfloat16).to(tl.float32)        # the bf16 silu_and_mul output, never written out
        row_scale = _row_scale(_abs_max(h32, m), fp8_max, MIN_SF)
        q = _quant_values(h32, row_scale, MAXV)
        tl.store(q_ptr + row * d + cols, q, mask=m)
        tl.store(s_ptr + row, row_scale)

    @triton.jit
    def _quant_kernel(x_ptr, q_ptr, s_ptr, K, stride_x, fp8_max, BLOCK: tl.constexpr, MIN_SF: tl.constexpr,
                      MAXV: tl.constexpr):
        # the per-token quant alone on a bf16 [rows, K] tensor (the arithmetic every fused site ends with; probe/tests)
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        m = cols < K
        h32 = tl.load(x_ptr + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
        row_scale = _row_scale(_abs_max(h32, m), fp8_max, MIN_SF)
        tl.store(q_ptr + row * K + cols, _quant_values(h32, row_scale, MAXV), mask=m)
        tl.store(s_ptr + row, row_scale)

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    _HAVE_TRITON = False


# --------------------------------------------------------------------------------------------------------------------
# eager mirror of the quant (CPU tests) and the launchers
# --------------------------------------------------------------------------------------------------------------------


def quant_torch(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``QuantFP8.forward_native`` (dynamic, per token) in eager torch: (fp8 [M, K], fp32 scale [M, 1])."""
    x_max = h.abs().max(dim=-1)[0].unsqueeze(-1).to(torch.float32)
    scale = (x_max / FP8_MAX).clamp(min=MIN_SCALE)
    out = (h.to(torch.float32) * scale.reciprocal()).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
    return out, scale


def _alloc(rows: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN001
    return torch.empty((rows, k), device=device, dtype=E4M3), torch.empty((rows, 1), device=device, dtype=torch.float32)


def _gpu(t: torch.Tensor) -> bool:
    return t.is_cuda and _HAVE_TRITON


def quant(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-token quant alone (Triton on the GPU, the eager mirror on the CPU) on a bf16 [M, K] tensor."""
    if not _gpu(h):
        return quant_torch(h)
    M, K = h.shape
    q, s = _alloc(M, K, h.device)
    if M:
        _quant_kernel[(M,)](h, q, s, K, h.stride(0), FP8_MAX, BLOCK=triton.next_power_of_2(K), MIN_SF=MIN_SCALE,
                            MAXV=FP8_MAX, num_warps=_ROW_WARPS if K <= 8192 else _SWIGLU_WARPS)
    return q, s


def rms_quant(x, weight, shift, scale, indices, eps, stock=None, ph=None):  # noqa: ANN001,ANN201
    """``rms_norm_indexed_scale_shift`` + the per-token quant of its output: (fp8 [M, H], scale [M, 1]).  With ``ph``
    (bf16, like ``x``) the kernel also stores stock's bf16 output there and quantises the row read back (RELOAD)."""
    if not _gpu(x):
        h = stock(x, weight, shift, scale, indices, eps)
        if ph is not None:
            ph.copy_(h)
        return quant_torch(h)
    rows, hidden = x.shape
    q, s = _alloc(rows, hidden, x.device)
    if rows:
        _rms_norm_indexed_scale_shift_quant_kernel[(rows,)](
            q, s, ph if ph is not None else q, x, weight, shift, scale, indices, hidden, eps,
            x.stride(0), shift.stride(0), scale.stride(0), indices.stride(0), 0, FP8_MAX,
            block_n=triton.next_power_of_2(hidden), MIN_SF=MIN_SCALE, MAXV=FP8_MAX, RELOAD=ph is not None,
            num_warps=_ROW_WARPS,
        )
    return q, s


def gate_rms_quant(residual, gate, branch, weight, shift, scale, indices, eps, stock=None, ph=None):  # noqa: ANN001,ANN201
    """``indexed_gate_rms_norm_scale_shift`` with the modulated output quantised per token:
    (residual_out bf16 [M, H], fp8 [M, H], scale [M, 1]); ``ph`` as in ``rms_quant``."""
    if not _gpu(residual):
        res, mod = stock(residual, gate, branch, weight, shift, scale, indices, eps)
        if ph is not None:
            ph.copy_(mod)
        return (res, *quant_torch(mod))
    rows, hidden = residual.shape
    residual_out = torch.empty_like(residual)
    q, s = _alloc(rows, hidden, residual.device)
    if rows:
        _indexed_gate_rms_norm_scale_shift_quant_kernel[(rows,)](
            residual_out, q, s, ph if ph is not None else q, residual, gate, branch, weight, shift, scale, indices,
            hidden, eps, residual.stride(0), gate.stride(0), branch.stride(0), shift.stride(0), scale.stride(0),
            indices.stride(0), 0, FP8_MAX, block_n=triton.next_power_of_2(hidden), MIN_SF=MIN_SCALE, MAXV=FP8_MAX,
            RELOAD=ph is not None, num_warps=_ROW_WARPS,
        )
    return residual_out, q, s


def unpack_stock(recv: torch.Tensor, heads: int) -> torch.Tensor:
    """ulysses_exact.exchange_back's unpack: receive [P, M, hl, d] -> [1, M, H, d] contiguous."""
    ws, m, hl, d = recv.shape
    result = torch.empty((1, m, heads, d), dtype=recv.dtype, device=recv.device)
    result[0].view(m, ws, hl, d).copy_(recv.transpose(0, 1))
    return result


def unpack_quant(recv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The unpack + the per-token quant of the [M, P*hl*d] out-projection input."""
    ws, m, hl, d = recv.shape
    if not _gpu(recv):
        return quant_torch(unpack_stock(recv, ws * hl)[0].reshape(m, ws * hl * d))
    C = hl * d
    q, s = _alloc(m, ws * C, recv.device)
    if m:
        _unpack_quant_kernel[(m,)](recv, q, s, m, C, FP8_MAX, P=ws, BLOCK_C=triton.next_power_of_2(C),
                                   MIN_SF=MIN_SCALE, MAXV=FP8_MAX, num_warps=_ROW_WARPS)
    return q, s


def silu_mul_quant(x: torch.Tensor, stock=None) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN001
    """silu_and_mul on [M, 2d] bf16 + the per-token quant of the [M, d] result."""
    if not _gpu(x):
        return quant_torch(stock(x))
    rows, width = x.shape
    d = width // 2
    q, s = _alloc(rows, d, x.device)
    if rows:
        _silu_mul_quant_kernel[(rows,)](x, q, s, d, x.stride(0), FP8_MAX, BLOCK=triton.next_power_of_2(d),
                                        MIN_SF=MIN_SCALE, MAXV=FP8_MAX, num_warps=_SWIGLU_WARPS)
    return q, s


# --------------------------------------------------------------------------------------------------------------------
# the stash, the block context, the quant hook
# --------------------------------------------------------------------------------------------------------------------

_tls = threading.local()   # ctx: _Ctx of the DiT block whose forward is running in this thread
_state: dict[str, Any] = {"sites": (), "cpu_mirror": False, "checked": {}, "disabled": set(), "orig_quant": None,
                          "quant_cls": None, "act_cls": None, "verify": False, "modes": (0, 1)}
_stats: dict[str, int] = {"fused_qkv": 0, "fused_out": 0, "fused_fc1": 0, "fused_fc2": 0, "consumed": 0,
                          "materialize": 0, "rerun": 0, "selfcheck_pass": 0, "selfcheck_fail": 0,
                          "verify_calls": 0, "verify_mismatch": 0}
_warned: set[str] = set()


def _count(kind: str) -> None:
    _stats[kind] += 1
    if _stats[kind] == 1:
        logger.info("h3_tp2 fp8 quant fuse: first %s", kind)
    elif kind == "consumed" and _stats[kind] % 20000 == 0:
        logger.info("h3_tp2 fp8 quant fuse: %s", " ".join(f"{k}={v}" for k, v in _stats.items() if v))


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("h3_tp2 fp8 quant fuse: " + msg, *args)


class Pending:
    """A producer's (fp8, scale) waiting for the quant of ``layer``; ``ph`` is the bf16 placeholder the model holds and
    ``k`` the GEMM input width (the quant sees ``ph`` as a contiguous [numel / k, k] view)."""
    __slots__ = ("site", "ph", "k", "q", "scale", "quant", "layer", "fill")

    def __init__(self, site: str, ph: torch.Tensor, k: int, q: torch.Tensor, scale: torch.Tensor, quant, layer,  # noqa: ANN001
                 fill: Callable[[torch.Tensor], None]):
        self.site, self.ph, self.k, self.q, self.scale = site, ph, k, q, scale
        self.quant, self.layer, self.fill = quant, layer, fill

    def matches(self, quant_obj, x: torch.Tensor) -> bool:  # noqa: ANN001
        ph, k = self.ph, self.k
        return (quant_obj is self.quant and x.dim() == 2 and x.dtype == ph.dtype and x.device == ph.device
                and x.data_ptr() == ph.data_ptr() and tuple(x.shape) == (ph.numel() // k, k)
                and x.stride() == (k, 1) and ph.is_contiguous())

    def materialize(self) -> torch.Tensor:
        self.fill(self.ph)       # the stock producer's bf16 output into the placeholder
        return self.ph


class _Ctx:
    __slots__ = ("block", "plan", "pending")

    def __init__(self, block, plan: dict):  # noqa: ANN001
        self.block = block
        self.plan = plan            # site -> (layer, its QuantFP8 instance)
        self.pending: Pending | None = None


def _materialize(ctx: _Ctx, reason: str) -> None:
    p = ctx.pending
    if p is None:
        return
    ctx.pending = None
    p.materialize()
    _count("materialize")
    _warn_once("mat-" + reason, "%s: %s stash materialised (stock producer into the placeholder)", reason, p.site)


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _selfcheck(key, fused_fn, stock_fn, *, last: bool = True) -> bool:  # noqa: ANN001
    """Once per key (site, width, ..., mode): fused vs stock pair on the first rows, bit for bit.  A mismatch of the
    ``last`` candidate disables the site for the process."""
    ok = _state["checked"].get(key)
    if ok is not None:
        return ok
    try:
        fused, stock = fused_fn(), stock_fn()
        ok = len(fused) == len(stock) and all(
            f.shape == s.shape and f.dtype == s.dtype and torch.equal(_bits(f), _bits(s)) for f, s in zip(fused, stock))
    except Exception as exc:  # noqa: BLE001
        logger.warning("h3_tp2 fp8 quant fuse: self-check %s raised %s: %s", key, type(exc).__name__, exc)
        ok = False
    _state["checked"][key] = ok
    if ok:
        _count("selfcheck_pass")
        logger.info("h3_tp2 fp8 quant fuse: self-check %s bit-identical", key)
    else:
        _count("selfcheck_fail")
        if last:
            _state["disabled"].add(key[0])
        logger.warning("h3_tp2 fp8 quant fuse: self-check %s NOT bit-identical to the stock producer + QuantFP8; %s",
                       key, f"site {key[0]} disabled for this process" if last else "trying the next mode")
    return ok


def _pick_mode(key, fused_for, stock_fn):  # noqa: ANN001,ANN202
    """The RMS-norm sites: mode 0 (bf16 row only in registers) if it passes the self-check, else mode 1 (upstream's
    bf16 store kept, row read back); None when neither is bit-identical (site disabled).  ``stock_fn`` returns the
    stock outputs with the bf16 row last (compared in mode 1 only, where the fused kernel writes it too)."""
    modes = _state["modes"]
    for m in modes:
        if _selfcheck(key + (m,), lambda: fused_for(m), lambda: stock_fn()[:None if m else -1], last=m == modes[-1]):
            return m
    return None


def _stock_quant(quant_obj, h: torch.Tensor):  # noqa: ANN001,ANN202
    """The layer's own per-token quant (the unwrapped ``QuantFP8.forward``), called as ``apply_weights`` calls it."""
    return _state["orig_quant"](quant_obj, h, None, None)


def _verify(p: Pending) -> None:
    """H3_TP2_FP8_QUANT_FUSE_VERIFY=1: the stock pair on the full tensor, compared with the stash (logged)."""
    h = p.materialize()
    k = p.k
    q_ref, s_ref = _stock_quant(p.quant, h.view(-1, k))
    _stats["verify_calls"] += 1
    nq = int((_bits(p.q) != _bits(q_ref)).sum())
    ns = int((_bits(p.scale) != _bits(s_ref)).sum())
    if nq or ns:
        _stats["verify_mismatch"] += 1
        if _stats["verify_mismatch"] <= 20:
            logger.warning("h3_tp2 fp8 quant fuse VERIFY: %s M=%d K=%d: %d fp8 bytes and %d scales differ from the "
                           "stock producer + QuantFP8", p.site, h.numel() // k, k, nq, ns)
    elif _stats["verify_calls"] % 1000 == 0:
        logger.info("h3_tp2 fp8 quant fuse VERIFY: %d calls checked, %d mismatching", _stats["verify_calls"],
                    _stats["verify_mismatch"])


def _outq_placeholder(x: torch.Tensor) -> bool:
    """True for ulysses_outq's deferred placeholder or a view of it (only loaded with H3_TP2_ULYSSES_EXACT=2)."""
    oq = sys.modules.get(__package__ + ".ulysses_outq")
    if oq is None or torch.compiler.is_compiling():     # placeholders exist only in eager forwards
        return False
    return oq.is_placeholder(x)


def pending_placeholder(x: torch.Tensor) -> bool:
    """True when ``x`` is this thread's pending stash placeholder (for ulysses_outq's guard: a placeholder of this
    module must never reach its out-projection quant)."""
    ctx = getattr(_tls, "ctx", None)
    p = ctx.pending if ctx is not None else None
    return p is not None and x.data_ptr() == p.ph.data_ptr()


def make_quant_forward(orig):  # noqa: ANN001,ANN201
    """Wrap ``QuantFP8.forward``: the pending stash's (fp8, scale) for exactly its planned instance and placeholder.
    Every other call goes to ``orig`` with its arguments untouched."""
    def forward(self, x, *args, **kwargs):  # noqa: ANN001,ANN202
        if isinstance(x, torch.Tensor) and _outq_placeholder(x):
            raise RuntimeError("h3_tp2 fp8 quant fuse: an H3_TP2_ULYSSES_EXACT=2 placeholder (stride 0, no data) "
                               "reached QuantFP8 outside ulysses_outq's out-projection quant")
        ctx = getattr(_tls, "ctx", None)
        p = ctx.pending if ctx is not None else None
        if p is not None:
            scale = args[0] if args else kwargs.get("scale")
            scale_ub = args[1] if len(args) > 1 else kwargs.get("scale_ub")
            if scale is None and scale_ub is None and isinstance(x, torch.Tensor) and p.matches(self, x):
                ctx.pending = None
                if _state["verify"]:
                    _verify(p)
                _count("consumed")
                return p.q, p.scale
            _materialize(ctx, "quant of another tensor")
        return orig(self, x, *args, **kwargs)

    forward._h3_tp2_quant_fuse = True
    forward.__wrapped__ = orig
    return forward


def _layer_post_hook(module, args, output):  # noqa: ANN001,ANN202
    """Safety net: a planned layer whose forward returned without its quant consuming the stash ran on the
    placeholder; make the placeholder real and run the layer again (never expected with the reviewed sources)."""
    ctx = getattr(_tls, "ctx", None)
    p = ctx.pending if ctx is not None else None
    if p is None or p.layer is not module:
        return None
    _materialize(ctx, f"{p.site} layer returned without consuming")
    _count("rerun")
    return module.forward(*args)


def _quant_of(layer):  # noqa: ANN001,ANN202
    """The layer's dynamic per-token QuantFP8 instance, or None."""
    kernel = getattr(getattr(layer, "quant_method", None), "fp8_linear", None)
    q = getattr(kernel, "quant_fp8", None)
    cls = _state["quant_cls"]
    if cls is None or not isinstance(q, cls):
        return None
    if getattr(q, "static", True) or not getattr(q, "use_per_token_if_dynamic", False) \
            or getattr(q, "is_group_quant", True) or getattr(q, "num_token_padding", 0) is not None:
        return None
    if getattr(layer, "input_scale", None) is not None or getattr(layer, "input_scale_ub", None) is not None:
        return None
    return q


def _lora_types():  # noqa: ANN202
    try:
        from vllm.lora.layers import BaseLayerWithLoRA
        return (BaseLayerWithLoRA,)
    except ImportError:
        return ()


def block_plan(block, sites, lora_types=()) -> tuple[dict | None, str]:  # noqa: ANN001
    """{site: (layer, QuantFP8 instance)} for this block, or (None, reason)."""
    try:
        if torch.compiler.is_compiling():
            return None, "torch.compile tracing"
    except AttributeError:
        pass
    attn, mlp = getattr(block, "attn", None), getattr(block, "mlp", None)
    layers = {"qkv": getattr(attn, "qkv_proj", None), "out": getattr(attn, "out_proj", None),
              "fc1": getattr(mlp, "fc1", None), "fc2": getattr(mlp, "fc2", None)}
    plan: dict = {}
    for site in sites:
        if site in _state["disabled"]:
            continue
        layer = layers[site]
        if layer is None or (lora_types and isinstance(layer, lora_types)):
            return None, f"{site}: missing or LoRA-wrapped layer"
        if getattr(layer, "tp_size", 1) != 1:
            return None, f"{site}: tp_size != 1"
        q = _quant_of(layer)
        if q is None:
            return None, f"{site}: not a dynamic per-token FP8 linear"
        if site == "qkv" and getattr(attn, "to_gate_compress", None) is not None:
            continue                              # the VSA gate reads the qkv input a second time
        if site == "fc2" and _state["act_cls"] is not None and not isinstance(getattr(mlp, "act_fn", None),
                                                                              _state["act_cls"]):
            continue
        plan[site] = (layer, q)
        if not getattr(layer, "_h3_tp2_quant_fuse_hook", False):
            layer.register_forward_hook(_layer_post_hook)
            layer._h3_tp2_quant_fuse_hook = True
    return (plan or None), "no site left"


def make_block_forward(orig_forward, sites, lora_types):  # noqa: ANN001,ANN201
    @functools.wraps(orig_forward)       # keeps __wrapped__: fp8_consumers' source fingerprints see upstream's forward
    def forward(self, *args, **kwargs):  # noqa: ANN001,ANN202
        plan, reason = block_plan(self, sites, lora_types)
        if plan is None:
            _warn_once("plan-" + reason, "block not fused (%s); stock path", reason)
            return orig_forward(self, *args, **kwargs)
        prev = getattr(_tls, "ctx", None)
        ctx = _Ctx(self, plan)
        _tls.ctx = ctx
        ok = False
        try:
            out = orig_forward(self, *args, **kwargs)
            ok = True
            return out
        finally:
            _tls.ctx = prev
            if ctx.pending is not None:
                if ok:      # cannot happen with the reviewed forwards; make the placeholder real for whoever holds it
                    _materialize(ctx, "stash left at the end of the block")
                ctx.pending = None

    forward._h3_tp2_quant_fuse = True
    return forward


def _take_site(site: str):  # noqa: ANN202
    """(ctx, layer, quant) when ``site`` is planned in the running block and has no pending stash in front of it."""
    ctx = getattr(_tls, "ctx", None)
    if ctx is None or site not in ctx.plan or site in _state["disabled"]:
        return None
    if ctx.pending is not None:
        _materialize(ctx, f"{site} producer while another stash was pending")
    return ctx


def _device_ok(*ts: torch.Tensor) -> bool:
    dev = ts[0].device
    if any(t.device != dev for t in ts):
        return False
    return (dev.type == "cuda" and _HAVE_TRITON) or (dev.type == "cpu" and _state["cpu_mirror"])


def _rows_ok(x: torch.Tensor, k: int, *others: torch.Tensor) -> bool:
    return (x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[0] > 0 and x.shape[1] == k
            and all(o.dim() == 2 and o.stride(1) == 1 for o in others))


def _k_of(layer) -> int:  # noqa: ANN001
    """The layer's input width (weight [K, N] as vLLM stores it)."""
    w = getattr(layer, "weight", None)
    return int(w.shape[0]) if isinstance(w, torch.Tensor) and w.dim() == 2 else -1


def make_rms_wrapper(orig):  # noqa: ANN001,ANN201
    """``rms_norm_indexed_scale_shift`` (block :950): the qkv site."""
    def rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps):  # noqa: ANN001,ANN202
        ctx = _take_site("qkv")
        if ctx is not None:
            layer, qobj = ctx.plan["qkv"]
            blk = ctx.block
            if weight is getattr(getattr(blk, "norm1", None), "weight", None) and _rows_ok(x, _k_of(layer), shift, scale) \
                    and indices.dim() == 1 and indices.shape[0] == x.shape[0] and _device_ok(x, weight, shift, scale, indices):
                r = min(_SELFCHECK_ROWS, x.shape[0])
                key = ("qkv", x.shape[1], shift.dtype, weight.dtype)

                def _fused_sub(mode):  # noqa: ANN001,ANN202
                    ph_r = torch.empty_like(x[:r]) if mode else None
                    q_r, s_r = rms_quant(x[:r], weight, shift, scale, indices[:r], eps, orig, ph_r)
                    return (q_r, s_r) + ((ph_r,) if mode else ())

                def _stock_sub():  # noqa: ANN202
                    h = orig(x[:r], weight, shift, scale, indices[:r], eps)
                    return (*_stock_quant(qobj, h), h)

                mode = _pick_mode(key, _fused_sub, _stock_sub)
                if mode is not None:
                    ph = torch.empty_like(x)
                    q, s = rms_quant(x, weight, shift, scale, indices, eps, orig, ph if mode else None)

                    def fill(t, _a=(x, weight, shift, scale, indices, eps)):  # noqa: ANN001,ANN202
                        t.copy_(orig(*_a))
                    ctx.pending = Pending("qkv", ph, x.shape[1], q, s, qobj, layer, (lambda t: None) if mode else fill)
                    _count("fused_qkv")
                    return ph
        return orig(x, weight, shift, scale, indices, eps)

    rms_norm_indexed_scale_shift._h3_tp2_quant_fuse = True
    return rms_norm_indexed_scale_shift


def make_gate_rms_wrapper(orig):  # noqa: ANN001,ANN201
    """``indexed_gate_rms_norm_scale_shift`` (block :970): the fc1 site (its modulated output feeds fc1)."""
    def indexed_gate_rms_norm_scale_shift(residual, gate, branch, weight, shift, scale, indices, eps):  # noqa: ANN001,ANN202
        ctx = _take_site("fc1")
        if ctx is not None:
            layer, qobj = ctx.plan["fc1"]
            blk = ctx.block
            if weight is getattr(getattr(blk, "norm2", None), "weight", None) \
                    and _rows_ok(residual, _k_of(layer), gate, branch, shift, scale) \
                    and branch.shape == residual.shape and branch.dtype == residual.dtype \
                    and indices.dim() == 1 and indices.shape[0] == residual.shape[0] \
                    and _device_ok(residual, gate, branch, weight, shift, scale, indices):
                r = min(_SELFCHECK_ROWS, residual.shape[0])
                key = ("fc1", residual.shape[1], gate.dtype, weight.dtype)

                def _fused_sub(mode):  # noqa: ANN001,ANN202
                    ph_r = torch.empty_like(residual[:r]) if mode else None
                    out = gate_rms_quant(residual[:r], gate, branch[:r], weight, shift, scale, indices[:r], eps, orig,
                                         ph_r)
                    return out + ((ph_r,) if mode else ())

                def _stock_sub():  # noqa: ANN202
                    res, mod = orig(residual[:r], gate, branch[:r], weight, shift, scale, indices[:r], eps)
                    return (res, *_stock_quant(qobj, mod), mod)

                mode = _pick_mode(key, _fused_sub, _stock_sub)
                if mode is not None:
                    ph = torch.empty_like(residual)
                    res, q, s = gate_rms_quant(residual, gate, branch, weight, shift, scale, indices, eps, orig,
                                               ph if mode else None)

                    def fill(t, _a=(residual, gate, branch, weight, shift, scale, indices, eps)):  # noqa: ANN001,ANN202
                        t.copy_(orig(*_a)[1])
                    ctx.pending = Pending("fc1", ph, residual.shape[1], q, s, qobj, layer, (lambda t: None) if mode else fill)
                    _count("fused_fc1")
                    return res, ph
        return orig(residual, gate, branch, weight, shift, scale, indices, eps)

    indexed_gate_rms_norm_scale_shift._h3_tp2_quant_fuse = True
    return indexed_gate_rms_norm_scale_shift


def unpack_hook(recv: torch.Tensor, heads: int) -> torch.Tensor | None:
    """``ulysses_exact.exchange_back``'s unpack (the out site): the placeholder [1, M, H, d], or None for stock."""
    ctx = _take_site("out")
    if ctx is None:
        return None
    layer, qobj = ctx.plan["out"]
    ws, m, hl, d = recv.shape
    if (recv.dtype != torch.bfloat16 or not recv.is_contiguous() or m == 0 or ws * hl != heads or ws & (ws - 1)
            or heads * d != _k_of(layer) or not _device_ok(recv)):
        return None
    r = min(_SELFCHECK_ROWS, m)
    key = ("out", ws, hl * d)
    if not _selfcheck(key, lambda: unpack_quant(recv[:, :r].contiguous()),
                      lambda: _stock_quant(qobj, unpack_stock(recv[:, :r], heads)[0].reshape(r, heads * d))):
        return None
    q, s = unpack_quant(recv)
    ph = torch.empty((1, m, heads, d), dtype=recv.dtype, device=recv.device)

    def fill(t, _recv=recv):  # noqa: ANN001,ANN202
        t[0].view(m, ws, hl, d).copy_(_recv.transpose(0, 1))
    ctx.pending = Pending("out", ph, heads * d, q, s, qobj, layer, fill)
    _count("fused_out")
    return ph


def make_act_forward(orig):  # noqa: ANN001,ANN201
    """``SiluAndMul.forward`` (the MLP's act_fn on fc1's bf16 output): the fc2 site."""
    def forward(self, x):  # noqa: ANN001,ANN202
        ctx = _take_site("fc2")
        if ctx is not None and self is getattr(getattr(ctx.block, "mlp", None), "act_fn", None):
            layer, qobj = ctx.plan["fc2"]
            if isinstance(x, torch.Tensor) and _rows_ok(x, 2 * _k_of(layer)) and _device_ok(x):
                r = min(_SELFCHECK_ROWS, x.shape[0])
                key = ("fc2", x.shape[1])
                if _selfcheck(key, lambda: silu_mul_quant(x[:r], lambda t: orig(self, t)),
                              lambda: _stock_quant(qobj, orig(self, x[:r]))):
                    q, s = silu_mul_quant(x, lambda t: orig(self, t))
                    ph = torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)

                    def fill(t, _x=x, _m=self):  # noqa: ANN001,ANN202
                        t.copy_(orig(_m, _x))
                    ctx.pending = Pending("fc2", ph, x.shape[1] // 2, q, s, qobj, layer, fill)
                    _count("fused_fc2")
                    return ph
        return orig(self, x)

    forward._h3_tp2_quant_fuse = True
    return forward


# --------------------------------------------------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------------------------------------------------


def parse_sites(value: str | None) -> tuple[str, ...]:
    """``H3_TP2_FP8_QUANT_FUSE``: "1"/"all" = every site; a comma list of qkv,out,fc1,fc2; else off."""
    v = (value or "").strip().lower()
    if v in ("1", "all", "on", "true"):
        return SITES
    if not v or v in ("0", "off", "false"):
        return ()
    sel = {p.strip() for p in v.split(",")}
    return tuple(s for s in SITES if s in sel)


def is_explicit(value: str | None) -> bool:
    """True when ``H3_TP2_FP8_QUANT_FUSE`` names its sites (a comma list) rather than "1"/"all"."""
    v = (value or "").strip().lower()
    return v not in ("", "0", "off", "false", "1", "all", "on", "true")


OUT_MODE2 = ("H3_TP2_ULYSSES_EXACT=2 makes no reverse unpack copy to fuse into: the out-projection quant reads the "
             "a2a receive buffer there (h3_tp2/ulysses_outq.py); the other sites are unaffected")


# the upstream code between each producer and its quant (fp8_consumers.EXPECTED names plus the ones below)
SITE_SOURCES = {
    "qkv": ("block_forward", "attn_forward", "rms_kernel", "rms", "launch_row_chunks", "col_forward",
            "apply_weights"),
    "out": ("block_forward", "attn_forward", "run_packed", "attn_layer_forward", "row_forward", "apply_weights"),
    "fc1": ("block_forward", "mlp_forward", "gate_rms_kernel", "gate_rms", "launch_row_chunks", "col_forward",
            "apply_weights"),
    "fc2": ("block_forward", "mlp_forward", "silu_forward_cuda", "row_forward", "apply_weights"),
}
# sha256 (first 16 hex) of the stripped source lines (step_cache.fingerprint_source), vLLM-Omni v0.30.0 a8576ccb7 and
# vLLM 0.30.0 in ~/h3-long/.venv on Spark_06 (2026-11-04), for the code fp8_consumers.EXPECTED does not cover
EXPECTED_EXTRA = {
    "rms_kernel": "b791a0e3dec74b2e",          # indexed_modulation._rms_norm_indexed_scale_shift_kernel
    "rms": "373f4b3620d23b43",                 # indexed_modulation.rms_norm_indexed_scale_shift
    "run_packed": "89368785eed0feec",          # MiniMaxH3Attention._run_packed_attention (returns .squeeze(0))
    "attn_layer_forward": "101199f5a077548b",  # vllm_omni.diffusion.attention.layer.Attention.forward
    "silu_forward_cuda": "cc7021d08b7cfd95",   # vllm_omni.diffusion.layers.activation.SiluAndMul.forward_cuda
    "col_forward": "af4d0eb5573d0dc6",         # vllm ColumnParallelLinear.forward (qkv: QKV, fc1: MergedColumn)
    "row_forward": "693d179dcadc13ff",         # vllm RowParallelLinear.forward (out, fc2)
    "apply_weights": "54ad02f4f844454d",       # vllm FP8ScaledMMLinearKernel.apply_weights (x -> quant_fp8 -> GEMM)
}


def _fingerprints(tr) -> dict[str, str | None]:  # noqa: ANN001
    from . import fp8_consumers
    from .step_cache import fingerprint

    fps = fp8_consumers._fingerprints(tr)
    fps["run_packed"] = fingerprint(getattr(tr.MiniMaxH3Attention, "_run_packed_attention", None))
    srcs: dict[str, Any] = {}
    try:
        from vllm_omni.diffusion.layers import indexed_modulation as im
        srcs["rms_kernel"] = getattr(im._rms_norm_indexed_scale_shift_kernel, "fn",
                                     im._rms_norm_indexed_scale_shift_kernel)
        srcs["rms"] = im.rms_norm_indexed_scale_shift
        if tr.rms_norm_indexed_scale_shift is not im.rms_norm_indexed_scale_shift:
            srcs["rms"] = None
    except ImportError:
        pass
    try:
        from vllm_omni.diffusion.attention import layer as layer_mod
        srcs["attn_layer_forward"] = layer_mod.Attention.forward
    except ImportError:
        pass
    try:
        from vllm_omni.diffusion.layers.activation import SiluAndMul
        srcs["silu_forward_cuda"] = SiluAndMul.forward_cuda
    except ImportError:
        pass
    try:
        from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearKernel
        from vllm.model_executor.layers import linear as lin
        srcs.update(col_forward=lin.ColumnParallelLinear.forward, row_forward=lin.RowParallelLinear.forward,
                    apply_weights=FP8ScaledMMLinearKernel.apply_weights)
    except ImportError:
        pass
    for name, fn in srcs.items():
        fps[name] = fingerprint(fn) if fn is not None else "wrapped"
    return fps


def _expected(name: str) -> str | None:
    from . import fp8_consumers
    return EXPECTED_EXTRA[name] if name in EXPECTED_EXTRA else fp8_consumers.EXPECTED.get(name)


def install(sites: tuple[str, ...] = SITES, tr=None, *, quant_cls=None, act_cls=None, check_source: bool = True,  # noqa: ANN001
            lora_types=None, ulysses_mode: str | None = None, explicit: bool = False) -> list[str]:
    """Patch the transformer module ``tr`` (default: vLLM-Omni's MiniMax H3), ``QuantFP8.forward`` and
    ``SiluAndMul.forward`` for the given sites.  Returns the installed sites; a site whose preconditions or upstream
    sources do not hold is skipped with a warning (the stock path stays).  ``explicit``: the sites were named (a comma
    list), only changes the wording of the out site's mode-2 line."""
    sites = tuple(s for s in SITES if s in sites)
    if not sites:
        return []
    if not _state["cpu_mirror"] and not _HAVE_TRITON:
        logger.warning("h3_tp2 fp8 quant fuse: Triton not usable in this process; skipped")
        return []
    if os.environ.get("VLLM_BATCH_INVARIANT", "0").strip() not in ("", "0"):
        logger.warning("h3_tp2 fp8 quant fuse: VLLM_BATCH_INVARIANT changes the FP8 linear path; skipped")
        return []
    if tr is None:
        from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as tr
    if quant_cls is None:
        from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8 as quant_cls
    if act_cls is None and "fc2" in sites:
        from vllm_omni.diffusion.layers.activation import SiluAndMul as act_cls
    block_cls = tr.MiniMaxH3DiTBlock
    if getattr(block_cls.forward, "_h3_tp2_quant_fuse", False):
        return list(_state["sites"])
    from . import fp8_consumers

    skip: dict[str, str] = {}
    if "out" in sites:
        mode = ulysses_mode
        if mode is None:
            try:
                from vllm_omni.diffusion.attention import layer as layer_mod
                mode = getattr(layer_mod.Attention, "_h3_tp2_exact", None)
            except ImportError:
                mode = None
        if mode == "2":
            logger.warning("h3_tp2 fp8 quant fuse: site out %s: %s",
                           "requested but ignored" if explicit else "skipped automatically", OUT_MODE2)
            sites = tuple(s for s in sites if s != "out")
            if not sites:
                return []
        elif mode != "1":
            skip["out"] = f"needs H3_TP2_ULYSSES_EXACT=1 (installed: {mode})"
    if "fc1" in sites and ("out" in fp8_consumers._state.get("sites", ()) or
                           getattr(tr.indexed_gate_rms_norm_scale_shift, "_h3_tp2_consumer_fuse", False)):
        skip["fc1"] = "fp8_consumers' out site owns indexed_gate_rms_norm_scale_shift (tensor32 level)"
    if "fc2" in sites and fp8_gemm._state.get("level") != "cutlass" and not _state["cpu_mirror"]:
        skip["fc2"] = "fc1's output reaches SiluAndMul only at H3_TP2_FP8_GEMM=cutlass"
    if check_source:
        fps = _fingerprints(tr)
        for site in sites:
            bad = [n for n in SITE_SOURCES[site] if fps.get(n) != _expected(n)]
            if bad and site not in skip:
                skip[site] = "upstream source changed: " + ", ".join(f"{n} {fps.get(n)} (reviewed {_expected(n)})"
                                                                     for n in bad)
    for site, why in skip.items():
        logger.warning("h3_tp2 fp8 quant fuse: site %s skipped (%s)", site, why)
    sites = tuple(s for s in sites if s not in skip)
    if not sites:
        return []
    _state["quant_cls"] = quant_cls
    _state["act_cls"] = act_cls if "fc2" in sites else None
    if not getattr(quant_cls.forward, "_h3_tp2_quant_fuse", False):
        _state["orig_quant"] = quant_cls.forward
        quant_cls.forward = make_quant_forward(quant_cls.forward)
    if "qkv" in sites:
        tr.rms_norm_indexed_scale_shift = make_rms_wrapper(tr.rms_norm_indexed_scale_shift)
    if "fc1" in sites:
        tr.indexed_gate_rms_norm_scale_shift = make_gate_rms_wrapper(tr.indexed_gate_rms_norm_scale_shift)
    if "fc2" in sites and not getattr(act_cls.forward, "_h3_tp2_quant_fuse", False):
        act_cls.forward = make_act_forward(act_cls.forward)
    if "out" in sites:
        from . import ulysses_exact
        ulysses_exact.set_unpack_hook(unpack_hook)
    block_cls.forward = make_block_forward(block_cls.forward, sites, _lora_types() if lora_types is None else lora_types)
    _state["sites"] = sites
    _state["verify"] = os.environ.get("H3_TP2_FP8_QUANT_FUSE_VERIFY", "0").strip() == "1"
    mode = os.environ.get("H3_TP2_FP8_QUANT_FUSE_MODE", "auto").strip().lower()
    _state["modes"] = {"0": (0,), "1": (1,)}.get(mode, (0, 1))
    return list(sites)


def stats() -> dict[str, int]:
    return dict(_stats)


if os.environ.get("H3_TP2_FP8_QUANT_FUSE_CPU_MIRROR") == "1":   # tests only: route CPU tensors through the eager mirrors
    _state["cpu_mirror"] = True
