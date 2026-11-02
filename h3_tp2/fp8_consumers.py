"""FP8 scale pass fused into the kernels that consume the GEMM output (``H3_TP2_FP8_CONSUMER_FUSE=1``, off by default;
speed plan 2026-11-02 item 6b).  Needs ``H3_TP2_FP8_GEMM=tensor32`` (the default two-pass path).

Today (vLLM-Omni v0.30.0 ``a8576ccb7``) each DiT block runs four two-pass FP8 GEMMs.  For three of them the scale pass
(``fp8_gemm._scale_bias_cast``) writes a bf16 [M, N] tensor (``fp8_gemm.scaled_mm_tensor32``: ``y = torch.empty``) that
exactly one upstream kernel reads back:

* qkv (N = 21504 = q | k | v, ``minimax_h3_transformer.py:655``): q and k (strided views of the bf16 output, ``:658-660``)
  go to ``fused_qk_norm_rope`` (``:666``, Triton ``_rms_norm_rope_kernel``, ``fused_qk_norm_rope.py:108``, launched
  once for q and once for k); v (``:661``) stays a strided view of the same tensor and is read by the attention layer;
* out projection (N = 5376, ``:703``): returned by the attention, read as ``branch`` by
  ``indexed_gate_rms_norm_scale_shift`` (``:970``, ``indexed_modulation.py:129``);
* fc2 (N = 5376, ``:741``): returned by the MLP, read as ``other`` by ``indexed_gate`` (``:982``,
  ``indexed_modulation.py:58``).
(fc1 is already fused with SwiGLU, ``fp8_swiglu``; ``rms_norm_indexed_scale_shift`` (``:950``) reads the residual stream,
not a GEMM output.)  No copy sits between producer and consumer: the linear layers (tp 1) return the scale pass's
tensor itself (``RowParallelLinear.forward`` without all-reduce, ``apply_scaled_mm`` only views it).

With this module the GEMM dispatcher keeps the fp32 product of those three GEMMs (a "stash") and hands the layer an
uninitialised bf16 placeholder of the right shape; the consumer is replaced by a copy of the upstream Triton kernel that
reads the fp32 product and applies, per element, exactly the scale pass's arithmetic -- ``t = scale_b[col] * acc``, then
``y = scale_a[row] * t`` (or ``fma(scale_a, t, bias)``), then ONE rounding to bf16 (``_round_bf16``, round to nearest
even, the same result as the pass's ``y.to(tl.bfloat16)``) -- and then continues with the consumer's own arithmetic,
verbatim, on that fp32 value, which is exactly the value the consumer's ``tl.load(bf16).to(tl.float32)`` produced.  So
the rounding happens at the same point as before and every later operation sees the same operands; the bf16 [M, N]
intermediate is neither written nor read (v excepted: its columns are still scale-passed into the placeholder, because
the attention reads v as a strided view of the qkv tensor).  Saved per block: 4 bytes per element of q, k, out and fc2
(write + read of bf16): 4.3 GB per layer at M = 42848.

Bit-identity on the GPU rests on two more conditions that the CPU cannot check: the compiler must give the copied
consumer arithmetic the same code (fma contraction, reduction tree of ``tl.sum``) as the upstream kernel.  The copies
keep the upstream launch shapes (heads per program, block_n, num_warps) and keep at least one bf16 access of the same
shape in each kernel, so Triton's coalescing picks the same per-thread layout; on the first eligible call per site and
shape the module runs the fused kernel and the stock pair (scale pass + upstream kernel) on the first 64 rows and
compares the bits: a mismatch disables that site for the process (logged), so an unverified site can never serve.
``scripts/fp8_consumer_probe.py`` measures the same on full shapes plus the timing.

Safety: a stash is made only for the qkv/out/fc2 weights of the DiT block whose forward is running (identified by the
weight tensor), only if the block's forward, the attention and MLP forwards and the upstream kernels and launchers have
the reviewed source (fingerprints in ``EXPECTED``), the layers are not LoRA-wrapped, tp == 1, the GEMM's N is the
layer's logical width and a rope table is passed.  Any consumer call that does not match the pending stash exactly
(pointer, shape, strides, dtypes, devices, flags) first "materialises" it -- runs the ordinary scale pass into the
placeholder -- and then calls the stock consumer, so the fallback is the stock path bit for bit.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

import torch

from . import fp8_gemm

logger = logging.getLogger("vllm_omni.h3_tp2.fp8_consumers")

SITES = ("qkv", "out", "fc2")
_SELFCHECK_ROWS = 64

# sha256 (first 16 hex) of the stripped, non-empty, non-comment, non-decorator source lines (step_cache.fingerprint_source)
# of the upstream code each fused site relies on, vLLM-Omni v0.30.0 a8576ccb7 (~/h3-long/vllm-omni on Spark_06,
# 2026-11-03).  The copies below were made from exactly these sources.
EXPECTED = {
    "attn_forward": "b2a25e0b45bea674",  # minimax_h3_transformer.py:630-704
    "block_forward": "c052863b435652b7",  # minimax_h3_transformer.py:919-982
    "mlp_forward": "ac9fa6ab2762a4bb",  # minimax_h3_transformer.py:738-742 (or fp8_swiglu's forward)
    "rms_norm_rope_kernel": "dde848323d52afbd",  # fused_qk_norm_rope.py:108-168
    "launch_fused_rms_norm_rope": "06631d16d022e0ef",  # fused_qk_norm_rope.py:433-464
    "fused_qk_norm_rope_impl": "dbc570960f4b3a99",  # fused_qk_norm_rope.py:530-570
    "fused_qk_norm_rope": "38d2d10a3af2d594",  # fused_qk_norm_rope.py:632-707
    "fused_cuda_supported": "979429af70cdbf7b",  # fused_qk_norm_rope.py:389-410
    "indexed_gate_kernel": "9273a7807bd876ab",  # indexed_modulation.py:58-87
    "indexed_gate": "0c16c223345767c7",  # indexed_modulation.py:209-240
    "gate_rms_kernel": "7d239874d6fcab35",  # indexed_modulation.py:129-173
    "gate_rms": "b0d503b521d61c64",  # indexed_modulation.py:284-334
    "launch_row_chunks": "9260a692db6402e3",  # indexed_modulation.py:19-24
}
# which fingerprints each site needs (the block forward for all of them)
SITE_SOURCES = {
    "qkv": ("block_forward", "attn_forward", "rms_norm_rope_kernel", "launch_fused_rms_norm_rope",
            "fused_qk_norm_rope_impl", "fused_qk_norm_rope", "fused_cuda_supported"),
    "out": ("block_forward", "attn_forward", "gate_rms_kernel", "gate_rms", "launch_row_chunks"),
    "fc2": ("block_forward", "mlp_forward", "indexed_gate_kernel", "indexed_gate", "launch_row_chunks"),
}

# upstream launch parameters the copies keep (fused_qk_norm_rope.py:44 and :461-462, indexed_modulation.py launchers)
_HEADS_PER_PROGRAM = 8
_ROPE_WARPS = 8
_ROW_WARPS = 8
_HEAD_DIM = 128
_ROTARY_DIM = 96
_VPASS_BLOCK = 2048

try:
    import triton
    import triton.language as tl

    from .fp8_swiglu import _round_bf16

    @triton.jit
    def _epilogue_bf16(acc, sa, sb_ptr, bias_ptr, cols, mask, HAS_BIAS: tl.constexpr):
        # fp8_gemm._scale_bias_cast's arithmetic on one element: t = scale_b * acc, y = scale_a * t (bias: one fma),
        # then the pass's single rounding to bf16 (RNE), kept in fp32 = what the consumer's bf16 load + .to(fp32) gave
        t = tl.load(sb_ptr + cols, mask=mask, other=0.0) * acc
        if HAS_BIAS:
            y = tl.math.fma(sa, t, tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32))
        else:
            y = sa * t
        return _round_bf16(y)

    @triton.jit
    def _scale_cols_kernel(x_ptr, sa_ptr, sb_ptr, bias_ptr, y_ptr, ncols, stride_x, stride_y,
                           HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
        # the scale pass on a column range of the product (v of the qkv GEMM), written into the placeholder's columns
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < ncols
        x = tl.load(x_ptr + row * stride_x + cols, mask=m, other=0.0)
        t = tl.load(sb_ptr + cols, mask=m, other=0.0) * x
        sa = tl.load(sa_ptr + row)
        if HAS_BIAS:
            y = tl.math.fma(sa, t, tl.load(bias_ptr + cols, mask=m, other=0.0).to(tl.float32))
        else:
            y = sa * t
        tl.store(y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=m)

    @triton.jit
    def _scale_rms_norm_rope_kernel(
        acc_ptr,
        sa_ptr,
        sb_ptr,
        bias_ptr,
        weight_ptr,
        rope_table_ptr,
        out_ptr,
        x_stride_t,
        x_stride_h,
        x_stride_d,
        rope_stride_t,
        out_stride_t,
        out_stride_h,
        out_stride_d,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_half: tl.constexpr,
        eps: tl.constexpr,
        heads_per_program: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        # Copy of vLLM-Omni a8576ccb7 _rms_norm_rope_kernel (fused_qk_norm_rope.py:108-168, Apache-2.0).  Only the two
        # loads of x change: they read the fp32 GEMM product and apply the scale pass's epilogue + bf16 rounding.
        token = tl.program_id(0).to(tl.int64)
        head_group = tl.program_id(1)
        heads = head_group * heads_per_program + tl.arange(0, heads_per_program)
        dims = tl.arange(0, head_dim)
        mask = heads[:, None] < num_heads
        cols = heads[:, None] * x_stride_h + dims[None, :] * x_stride_d   # product column within this q/k view
        offsets = token * x_stride_t + cols
        sa = tl.load(sa_ptr + token)

        x = _epilogue_bf16(tl.load(acc_ptr + offsets, mask=mask, other=0.0), sa, sb_ptr, bias_ptr, cols, mask, HAS_BIAS)
        weight = tl.load(weight_ptr + dims).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
        normalized = (x * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        rotary_dim = rotary_half * 2
        pair_dims = tl.where(
            dims < rotary_half,
            dims + rotary_half,
            tl.where(dims < rotary_dim, dims - rotary_half, dims),
        )
        pair_cols = heads[:, None] * x_stride_h + pair_dims[None, :] * x_stride_d
        pair_offsets = token * x_stride_t + pair_cols
        pair_x = _epilogue_bf16(tl.load(acc_ptr + pair_offsets, mask=mask, other=0.0), sa, sb_ptr, bias_ptr,
                                pair_cols, mask, HAS_BIAS)
        pair_weight = tl.load(weight_ptr + pair_dims).to(tl.float32)
        pair_normalized = (pair_x * inv_rms[:, None] * pair_weight[None, :]).to(tl.bfloat16)

        freq_dims = tl.where(
            dims < rotary_half,
            dims,
            tl.where(dims < rotary_dim, dims - rotary_half, 0),
        )
        table_offsets = token * rope_stride_t + freq_dims
        cos = tl.load(rope_table_ptr + table_offsets).to(tl.float32)
        sin = tl.load(rope_table_ptr + table_offsets + rotary_half).to(tl.float32)
        first = normalized.to(tl.float32) * cos - pair_normalized.to(tl.float32) * sin
        second = normalized.to(tl.float32) * cos + pair_normalized.to(tl.float32) * sin
        output = tl.where(
            dims < rotary_dim,
            tl.where(dims < rotary_half, first, second),
            normalized.to(tl.float32),
        )

        out_offsets = token * out_stride_t + heads[:, None] * out_stride_h + dims[None, :] * out_stride_d
        tl.store(out_ptr + out_offsets, output, mask=mask)

    @triton.jit
    def _scale_indexed_gate_kernel(
        output_ptr,
        x_ptr,
        gate_ptr,
        acc_ptr,
        sa_ptr,
        sb_ptr,
        bias_ptr,
        indices_ptr,
        hidden_size,
        stride_output_row,
        stride_x_row,
        stride_gate_row,
        stride_acc_row,
        stride_indices,
        row_offset,
        HAS_BIAS: tl.constexpr,
        block_n: tl.constexpr,
    ):
        # Copy of vLLM-Omni a8576ccb7 _indexed_gate_kernel (indexed_modulation.py:58-87, Apache-2.0); ``other`` (the
        # fc2 output) comes from the fp32 product through the scale pass's epilogue + bf16 rounding.
        row = tl.program_id(0) + row_offset
        columns = tl.arange(0, block_n)
        mask = columns < hidden_size
        index = tl.load(indices_ptr + row * stride_indices)

        x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0).to(tl.float32)
        acc = tl.load(acc_ptr + row * stride_acc_row + columns, mask=mask, other=0.0)
        other = _epilogue_bf16(acc, tl.load(sa_ptr + row), sb_ptr, bias_ptr, columns, mask, HAS_BIAS)

        tl.store(
            output_ptr + row * stride_output_row + columns,
            x + gate * other,
            mask=mask,
        )

    @triton.jit
    def _scale_indexed_gate_rms_norm_scale_shift_kernel(
        residual_out_ptr,
        modulated_out_ptr,
        residual_ptr,
        gate_ptr,
        acc_ptr,
        sa_ptr,
        sb_ptr,
        bias_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        stride_residual_row,
        stride_gate_row,
        stride_acc_row,
        stride_shift_row,
        stride_scale_row,
        stride_indices,
        row_offset,
        HAS_BIAS: tl.constexpr,
        block_n: tl.constexpr,
    ):
        # Copy of vLLM-Omni a8576ccb7 _indexed_gate_rms_norm_scale_shift_kernel (indexed_modulation.py:129-173,
        # Apache-2.0); ``branch`` (the out-projection output) comes from the fp32 product through the scale pass's
        # epilogue + bf16 rounding.  Masked lanes are forced to 0 as the stock masked load gives (they enter tl.sum).
        row = tl.program_id(0) + row_offset
        columns = tl.arange(0, block_n)
        mask = columns < hidden_size
        index = tl.load(indices_ptr + row * stride_indices)

        residual = tl.load(residual_ptr + row * stride_residual_row + columns, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0).to(tl.float32)
        acc = tl.load(acc_ptr + row * stride_acc_row + columns, mask=mask, other=0.0)
        branch = _epilogue_bf16(acc, tl.load(sa_ptr + row), sb_ptr, bias_ptr, columns, mask, HAS_BIAS)
        branch = tl.where(mask, branch, 0.0)
        # Match the BF16 residual value consumed by the unfused RMSNorm path.
        updated = (residual + gate * branch).to(tl.bfloat16).to(tl.float32)
        tl.store(residual_out_ptr + row * hidden_size + columns, updated, mask=mask)

        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(updated * updated, axis=0) / hidden_size
        normalized = updated * tl.rsqrt(variance + eps) * weight
        shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)
        output = normalized * (1.0 + scale) + shift
        tl.store(
            modulated_out_ptr + row * hidden_size + columns,
            output,
            mask=mask,
        )

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    _HAVE_TRITON = False


# --------------------------------------------------------------------------------------------------------------------
# eager mirrors (CPU tests; they follow the kernels' index arithmetic, the GPU runs the Triton copies above)
# --------------------------------------------------------------------------------------------------------------------


def epilogue_bf16_torch(acc: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """``_epilogue_bf16`` on broadcastable fp32 tensors: t = sb * acc, y = sa * t (bias: fma, emulated in float64 as in
    ``fp8_gemm.epilogue_torch``), rounded to bf16 (RNE) and returned as fp32."""
    t = sb * acc
    if bias is None:
        y = sa * t
    else:
        y = (sa.double() * t.double() + bias.double()).float()
    return y.to(torch.bfloat16).float()


def _qk_rope_torch(x32, col0, sa, sb, bias, weight, rope_table, eps, heads, head_dim, strides, out):  # noqa: ANN001
    """Mirror of ``_scale_rms_norm_rope_kernel`` over all tokens, with the kernel's pointer arithmetic: the product,
    scale_b and bias pointers start at column ``col0`` (the q or k block) of the contiguous fp32 product ``x32``;
    ``strides`` = (x_stride_t, x_stride_h, x_stride_d) as the kernel receives them."""
    T = x32.shape[0]
    st, sh, sd = strides
    dims = torch.arange(head_dim)
    hs = torch.arange(heads)
    rotary_half = rope_table.shape[-1] // 2
    rotary_dim = 2 * rotary_half
    flat_acc = x32.reshape(-1)
    tok = torch.arange(T).view(T, 1, 1)
    s_a = sa.view(T, 1, 1).float()

    def epi(dsel):  # noqa: ANN001,ANN202
        cols = hs.view(1, -1, 1) * sh + dsel.view(1, 1, -1) * sd          # column within the q/k block
        acc = flat_acc[col0 + tok * st + cols]
        bias_c = None if bias is None else bias[col0 + cols].float()
        return epilogue_bf16_torch(acc, s_a, sb[col0 + cols], bias_c)

    x = epi(dims)
    w = weight.float()
    inv_rms = torch.rsqrt((x * x).sum(dim=2) / head_dim + eps)
    normalized = (x * inv_rms[..., None] * w).to(torch.bfloat16)
    pair_dims = torch.where(dims < rotary_half, dims + rotary_half, torch.where(dims < rotary_dim, dims - rotary_half, dims))
    pair_x = epi(pair_dims)
    pair_normalized = (pair_x * inv_rms[..., None] * w[pair_dims]).to(torch.bfloat16)
    freq_dims = torch.where(dims < rotary_half, dims, torch.where(dims < rotary_dim, dims - rotary_half, torch.zeros_like(dims)))
    cos = rope_table[:, freq_dims].float().unsqueeze(1)
    sin = rope_table[:, freq_dims + rotary_half].float().unsqueeze(1)
    n32, p32 = normalized.float(), pair_normalized.float()
    first = n32 * cos - p32 * sin
    second = n32 * cos + p32 * sin
    output = torch.where(dims < rotary_dim, torch.where(dims < rotary_half, first, second), n32)
    out.copy_(output.to(out.dtype))
    return out


def _gate_rms_torch(residual, gate, x32, sa, sb, bias, weight, shift, scale, indices, eps, residual_out, modulated_out):  # noqa: ANN001
    """Mirror of ``_scale_indexed_gate_rms_norm_scale_shift_kernel``."""
    H = residual.shape[1]
    branch = epilogue_bf16_torch(x32, sa.view(-1, 1).float(), sb.view(1, -1), None if bias is None else bias.view(1, -1).float())
    g = gate.index_select(0, indices).float()
    updated = (residual.float() + g * branch).to(torch.bfloat16).float()
    residual_out.copy_(updated.to(residual_out.dtype))
    variance = (updated * updated).sum(dim=1, keepdim=True) / H
    normalized = updated * torch.rsqrt(variance + eps) * weight.float()
    output = normalized * (1.0 + scale.index_select(0, indices).float()) + shift.index_select(0, indices).float()
    modulated_out.copy_(output.to(modulated_out.dtype))


def _gate_torch(x, gate, x32, sa, sb, bias, indices, output):  # noqa: ANN001
    """Mirror of ``_scale_indexed_gate_kernel``."""
    other = epilogue_bf16_torch(x32, sa.view(-1, 1).float(), sb.view(1, -1), None if bias is None else bias.view(1, -1).float())
    output.copy_((x.float() + gate.index_select(0, indices).float() * other).to(output.dtype))


def scale_pass_into(x32: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None, y: torch.Tensor) -> torch.Tensor:
    """The ordinary scale pass into ``y`` (the stock path of ``scaled_mm_tensor32``): Triton on the GPU, its eager mirror
    (fma for the bias) on the CPU."""
    if x32.is_cuda and fp8_gemm._state["triton"]:
        fp8_gemm._launch(x32, sa, sb, bias, y, x32.shape[1])
    else:
        y.copy_(fp8_gemm.epilogue_torch(x32, sa, sb, bias, y.dtype))
    return y


# --------------------------------------------------------------------------------------------------------------------
# launchers (fused consumers on an explicit stash; the probe and the tests call these directly)
# --------------------------------------------------------------------------------------------------------------------


class Stash:
    """The fp32 product of one GEMM and its epilogue operands, kept in place of the scale pass's output ``ph``."""
    __slots__ = ("x32", "sa", "sb", "bias", "ph")

    def __init__(self, x32: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, bias: torch.Tensor | None, ph: torch.Tensor):
        self.x32, self.sa, self.sb, self.bias, self.ph = x32, sa, sb, bias, ph

    def materialize(self) -> torch.Tensor:
        return scale_pass_into(self.x32, self.sa, self.sb, self.bias, self.ph)

    def rows(self, r: int) -> "Stash":
        return Stash(self.x32[:r], self.sa[:r], self.sb, self.bias, torch.empty_like(self.ph[:r]))


def _use_triton(t: torch.Tensor) -> bool:
    return t.is_cuda and _HAVE_TRITON


def qk_norm_rope_fused(st: Stash, num_heads: int, num_kv_heads: int, head_dim: int, q_weight: torch.Tensor,
                       k_weight: torch.Tensor, rope_table: torch.Tensor, eps: float):
    """q, k = ``fused_qk_norm_rope`` of the qkv GEMM output, read from the fp32 product; v's columns are scale-passed
    into the placeholder ``st.ph`` (so ``ph[:, q+k:]`` viewed [T, kv_heads, d] is stock's v).  Returns (q, k, v)."""
    x32, sa, sb, bias, ph = st.x32, st.sa, st.sb, st.bias, st.ph
    T, N = x32.shape
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    out_q = torch.empty((T, num_heads, head_dim), dtype=ph.dtype, device=ph.device)
    out_k = torch.empty((T, num_kv_heads, head_dim), dtype=ph.dtype, device=ph.device)
    q_weight, k_weight, rope_table = q_weight.contiguous(), k_weight.contiguous(), rope_table.contiguous()
    v0 = q_size + kv_size
    for col0, heads, w, out in ((0, num_heads, q_weight, out_q), (q_size, num_kv_heads, k_weight, out_k)):
        xv, sbv = x32[:, col0:], sb[col0:]
        bv = None if bias is None else bias[col0:]
        if T == 0:
            continue
        if _use_triton(x32):
            grid = (T, triton.cdiv(heads, _HEADS_PER_PROGRAM))
            _scale_rms_norm_rope_kernel[grid](
                xv, sa, sbv, bv if bv is not None else out, w, rope_table, out,
                x32.stride(0), head_dim, 1, rope_table.stride(0), out.stride(0), out.stride(1), out.stride(2),
                num_heads=heads, head_dim=head_dim, rotary_half=rope_table.shape[-1] // 2, eps=eps,
                heads_per_program=_HEADS_PER_PROGRAM, HAS_BIAS=bv is not None, num_warps=_ROPE_WARPS,
            )
        else:
            _qk_rope_torch(x32, col0, sa, sb, bias, w, rope_table, eps, heads, head_dim, (x32.stride(0), head_dim, 1), out)
    if T:
        xv, sbv, yv = x32[:, v0:], sb[v0:], ph[:, v0:]
        bv = None if bias is None else bias[v0:]
        if _use_triton(x32):
            _scale_cols_kernel[(T, triton.cdiv(kv_size, _VPASS_BLOCK))](
                xv, sa, sbv, bv if bv is not None else yv, yv, kv_size, x32.stride(0), ph.stride(0),
                HAS_BIAS=bv is not None, BLOCK=_VPASS_BLOCK)
        else:
            yv.copy_(fp8_gemm.epilogue_torch(xv, sa, sbv, bv, ph.dtype))
    v = ph[:, v0:].view(T, num_kv_heads, head_dim)
    return out_q, out_k, v


def gate_rms_fused(st: Stash, residual: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, shift: torch.Tensor,
                   scale: torch.Tensor, indices: torch.Tensor, eps: float, reuse_placeholder: bool = True):
    """``indexed_gate_rms_norm_scale_shift(residual, gate, branch, ...)`` with branch = the out-projection output read
    from the fp32 product.  ``residual_out`` is written into the placeholder (same shape, dtype and layout as stock's
    ``empty_like(residual)`` for a contiguous residual)."""
    x32, sa, sb, bias, ph = st.x32, st.sa, st.sb, st.bias, st.ph
    rows, hidden = residual.shape
    residual_out = ph if reuse_placeholder else torch.empty_like(residual)
    modulated_out = torch.empty_like(residual)
    if rows == 0:
        return residual_out, modulated_out
    if _use_triton(x32):
        _scale_indexed_gate_rms_norm_scale_shift_kernel[(rows,)](
            residual_out, modulated_out, residual, gate, x32, sa, sb, bias if bias is not None else ph,
            weight, shift, scale, indices, hidden, eps,
            residual.stride(0), gate.stride(0), x32.stride(0), shift.stride(0), scale.stride(0), indices.stride(0), 0,
            HAS_BIAS=bias is not None, block_n=triton.next_power_of_2(hidden), num_warps=_ROW_WARPS,
        )
    else:
        _gate_rms_torch(residual, gate, x32, sa, sb, bias, weight, shift, scale, indices, eps, residual_out, modulated_out)
    return residual_out, modulated_out


def gate_fused(st: Stash, x: torch.Tensor, gate: torch.Tensor, indices: torch.Tensor, reuse_placeholder: bool = True) -> torch.Tensor:
    """``indexed_gate(x, gate, other, indices)`` with other = the fc2 output read from the fp32 product; the result is
    written into the placeholder when ``x`` is contiguous (stock: ``empty_like(x)``)."""
    x32, sa, sb, bias, ph = st.x32, st.sa, st.sb, st.bias, st.ph
    output = ph if (reuse_placeholder and x.is_contiguous()) else torch.empty_like(x)
    rows, hidden = x.shape
    if rows == 0:
        return output
    if _use_triton(x32):
        _scale_indexed_gate_kernel[(rows,)](
            output, x, gate, x32, sa, sb, bias if bias is not None else ph, indices, hidden,
            output.stride(0), x.stride(0), gate.stride(0), x32.stride(0), indices.stride(0), 0,
            HAS_BIAS=bias is not None, block_n=triton.next_power_of_2(hidden), num_warps=_ROW_WARPS,
        )
    else:
        _gate_torch(x, gate, x32, sa, sb, bias, indices, output)
    return output


# --------------------------------------------------------------------------------------------------------------------
# dispatch: the stash hook, the block context and the consumer wrappers
# --------------------------------------------------------------------------------------------------------------------

_tls = threading.local()   # ctx: the _Ctx of the DiT block whose forward is running in this thread
_state: dict[str, Any] = {"sites": (), "cpu_mirror": False, "checked": {}, "disabled": set()}
_stats = {"stash": 0, "fused_qkv": 0, "fused_out": 0, "fused_fc2": 0, "materialize": 0, "selfcheck_pass": 0,
          "selfcheck_fail": 0}
_warned: set[str] = set()


def _count(kind: str) -> None:
    _stats[kind] += 1
    if _stats[kind] == 1:
        logger.info("h3_tp2 fp8 consumer fuse: first %s", kind)


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("h3_tp2 fp8 consumer fuse: " + msg, *args)


class _Ctx:
    __slots__ = ("want", "pending")

    def __init__(self, want: dict):
        self.want = want          # (weight data_ptr, weight shape) -> site
        self.pending: dict[str, Stash] = {}


def _product_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The two-pass GEMM's fp32 product (the same call as ``scaled_mm_tensor32``); fp32 matmul on the CPU (tests)."""
    if a.is_cuda:
        one = torch.ones((), device=a.device, dtype=torch.float32)
        return torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32)
    return a.float() @ b.float()


def _vec(scale: torch.Tensor, n: int) -> torch.Tensor:
    s = scale.reshape(-1).float()
    return (s.expand(n) if s.numel() == 1 else s).contiguous()


def stash_hook(a, b, scale_a, scale_b, out_dtype, bias):  # noqa: ANN001
    """fp8_gemm dispatcher hook (called for two-pass-eligible GEMMs only): when ``b`` is the qkv/out/fc2 weight of the
    running DiT block, keep the fp32 product and return an uninitialised placeholder for the consumer wrapper."""
    ctx = getattr(_tls, "ctx", None)
    if ctx is None or out_dtype is not torch.bfloat16 or fp8_gemm._state.get("fused"):
        return None
    site = ctx.want.pop((b.data_ptr(), tuple(b.shape)), None)
    if site is None:
        return None
    if ctx.pending:                       # a stash that was never consumed: make it real before the next one
        _materialize_all(ctx, f"unconsumed stash before {site}")
    M, N = a.shape[0], b.shape[1]
    ph = torch.empty((M, N), device=a.device, dtype=out_dtype)
    ctx.pending[site] = Stash(_product_fp32(a, b), _vec(scale_a, M), _vec(scale_b, N), bias, ph)
    _count("stash")
    return ph


def _materialize_all(ctx: _Ctx, reason: str) -> None:
    for site, st in list(ctx.pending.items()):
        st.materialize()
        _count("materialize")
        _warn_once("mat-" + reason, "%s: stash of %s materialised, stock consumer", reason, site)
    ctx.pending.clear()


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _selfcheck(key, fused_fn, stock_fn) -> bool:  # noqa: ANN001
    """Once per (site, shape): fused vs stock pair on the first rows, bit for bit; a mismatch disables the site."""
    ok = _state["checked"].get(key)
    if ok is not None:
        return ok
    try:
        fused, stock = fused_fn(), stock_fn()
        ok = all(f.shape == s.shape and torch.equal(_bits(f), _bits(s)) for f, s in zip(fused, stock))
    except Exception as exc:  # noqa: BLE001
        logger.warning("h3_tp2 fp8 consumer fuse: self-check %s raised %s", key, exc)
        ok = False
    _state["checked"][key] = ok
    if ok:
        _count("selfcheck_pass")
        logger.info("h3_tp2 fp8 consumer fuse: self-check %s bit-identical", key)
    else:
        _count("selfcheck_fail")
        _state["disabled"].add(key[0])
        logger.warning("h3_tp2 fp8 consumer fuse: self-check %s NOT bit-identical to scale pass + stock kernel; "
                       "site %s disabled for this process", key, key[0])
    return ok


def _device_ok(*ts: torch.Tensor) -> bool:
    dev = ts[0].device
    if any(t.device != dev for t in ts):
        return False
    return (dev.type == "cuda" and _HAVE_TRITON and fp8_gemm._state["triton"]) or (dev.type == "cpu" and _state["cpu_mirror"])


def qk_mismatch(st: Stash, q, k, q_weight, k_weight, rope_table, head_dim, rotary_dim, interleaved) -> str | None:  # noqa: ANN001
    """None when (q, k) are exactly the q/k views of the stash's placeholder and the stock call would run the Triton
    half-split kernel (``_fused_cuda_supported``), else the reason."""
    ph = st.ph
    if interleaved:
        return "interleaved"
    if q.dim() != 3 or k.dim() != 3 or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return "q/k rank or dtype"
    T, H, D = q.shape
    Hk = k.shape[1]
    N = ph.shape[1]
    if T == 0 or ph.dim() != 2 or ph.shape[0] != T or k.shape != (T, Hk, D) or N != (H + 2 * Hk) * D:
        return "shapes"
    if not ph.is_contiguous() or not st.x32.is_contiguous() or st.x32.shape != ph.shape:
        return "placeholder/product layout"
    eq, ek = ph[:, :H * D].view(T, H, D), ph[:, H * D:(H + Hk) * D].view(T, Hk, D)   # upstream's split + view
    if q.data_ptr() != eq.data_ptr() or k.data_ptr() != ek.data_ptr() or q.stride() != eq.stride() \
            or k.stride() != ek.stride():
        return "q/k are not the split views of the qkv placeholder"
    if D != _HEAD_DIM or head_dim not in (None, D):
        return "head_dim"
    rot = rope_table.shape[-1] if rope_table.dim() == 2 else -1
    if rot != _ROTARY_DIM or rotary_dim not in (None, rot) or rope_table.shape != (T, rot):
        return "rotary"
    if rope_table.dtype not in (torch.bfloat16, torch.float32) or q_weight.shape != (D,) or k_weight.shape != (D,):
        return "rope table or norm weight dtype/shape"
    if not _device_ok(q, k, q_weight, k_weight, rope_table, st.x32):
        return "device"
    return None


def rows_mismatch(st: Stash, x: torch.Tensor, branch: torch.Tensor, *others: torch.Tensor) -> str | None:
    """None when ``branch`` is exactly the stash's placeholder and ``x`` (the residual) a contiguous bf16 tensor of the
    same shape on the same device as the other operands."""
    ph = st.ph
    if branch.data_ptr() != ph.data_ptr() or branch.shape != ph.shape or branch.stride() != ph.stride():
        return "branch is not the placeholder"
    if x.dim() != 2 or x.shape != ph.shape or x.dtype != torch.bfloat16 or not x.is_contiguous() or not ph.is_contiguous() \
            or not st.x32.is_contiguous() or st.x32.shape != ph.shape:
        return "residual shape/dtype/layout"
    if x.shape[0] == 0:
        return "no rows"
    if any(o.dim() != 2 or o.stride(1) != 1 for o in others[:-1]):
        return "gate/shift/scale layout"
    if others[-1].dim() != 1 or others[-1].shape[0] != x.shape[0]:
        return "indices"
    if not _device_ok(x, st.x32, *others):
        return "device"
    return None


def _take(site: str) -> tuple[_Ctx | None, Stash | None]:
    ctx = getattr(_tls, "ctx", None)
    if ctx is None or not ctx.pending:
        return ctx, None
    st = ctx.pending.get(site)
    if st is None:
        _materialize_all(ctx, f"{site} consumer saw another site's stash")
    return ctx, st


def make_qk_wrapper(orig):  # noqa: ANN001,ANN201
    def fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps, *, head_dim=None, rotary_dim=None, interleaved=False):  # noqa: ANN001,ANN202
        ctx, st = _take("qkv")
        if st is not None:
            reason = "qkv disabled by self-check" if "qkv" in _state["disabled"] else \
                qk_mismatch(st, q, k, q_weight, k_weight, rope_table, head_dim, rotary_dim, interleaved)
            if reason is None:
                H, Hk, D = q.shape[1], k.shape[1], q.shape[-1]
                r = min(_SELFCHECK_ROWS, q.shape[0])

                def _fused_sub():  # noqa: ANN202
                    return qk_norm_rope_fused(st.rows(r), H, Hk, D, q_weight, k_weight, rope_table[:r], eps)

                def _stock_sub():  # noqa: ANN202
                    sub = st.rows(r)
                    p = sub.materialize()
                    qs = p[:, :H * D].view(r, H, D)
                    ks = p[:, H * D:(H + Hk) * D].view(r, Hk, D)
                    oq, ok_ = orig(qs, ks, q_weight, k_weight, rope_table[:r], eps, head_dim=head_dim,
                                   rotary_dim=rotary_dim, interleaved=interleaved)
                    return oq, ok_, p[:, (H + Hk) * D:].view(r, Hk, D)

                if not _selfcheck(("qkv", tuple(st.ph.shape[1:]), rope_table.dtype), _fused_sub, _stock_sub):
                    reason = "qkv disabled by self-check"
            if reason is None:
                del ctx.pending["qkv"]
                out_q, out_k, _ = qk_norm_rope_fused(st, H, Hk, D, q_weight, k_weight, rope_table, eps)
                _count("fused_qkv")
                return out_q, out_k
            _materialize_all(ctx, "qkv: " + reason)
        return orig(q, k, q_weight, k_weight, rope_table, eps, head_dim=head_dim, rotary_dim=rotary_dim, interleaved=interleaved)

    fused_qk_norm_rope._h3_tp2_consumer_fuse = True
    return fused_qk_norm_rope


def make_gate_rms_wrapper(orig):  # noqa: ANN001,ANN201
    def indexed_gate_rms_norm_scale_shift(residual, gate, branch, weight, shift, scale, indices, eps):  # noqa: ANN001,ANN202
        ctx, st = _take("out")
        if st is not None:
            reason = "out disabled by self-check" if "out" in _state["disabled"] else \
                rows_mismatch(st, residual, branch, gate, shift, scale, indices)
            if reason is None and (weight.shape != (residual.shape[1],) or weight.device != residual.device):
                reason = "norm weight shape/device"
            if reason is None:
                r = min(_SELFCHECK_ROWS, residual.shape[0])

                def _fused_sub():  # noqa: ANN202
                    return gate_rms_fused(st.rows(r), residual[:r], gate, weight, shift, scale, indices[:r], eps)

                def _stock_sub():  # noqa: ANN202
                    p = st.rows(r).materialize()
                    return orig(residual[:r], gate, p, weight, shift, scale, indices[:r], eps)

                if not _selfcheck(("out", tuple(residual.shape[1:]), gate.dtype, weight.dtype), _fused_sub, _stock_sub):
                    reason = "out disabled by self-check"
            if reason is None:
                del ctx.pending["out"]
                res = gate_rms_fused(st, residual, gate, weight, shift, scale, indices, eps)
                _count("fused_out")
                return res
            _materialize_all(ctx, "out: " + reason)
        return orig(residual, gate, branch, weight, shift, scale, indices, eps)

    indexed_gate_rms_norm_scale_shift._h3_tp2_consumer_fuse = True
    return indexed_gate_rms_norm_scale_shift


def make_gate_wrapper(orig):  # noqa: ANN001,ANN201
    def indexed_gate(x, gate, other, indices):  # noqa: ANN001,ANN202
        ctx, st = _take("fc2")
        if st is not None:
            reason = "fc2 disabled by self-check" if "fc2" in _state["disabled"] else \
                rows_mismatch(st, x, other, gate, indices)
            if reason is None:
                r = min(_SELFCHECK_ROWS, x.shape[0])

                def _fused_sub():  # noqa: ANN202
                    return (gate_fused(st.rows(r), x[:r].contiguous(), gate, indices[:r], reuse_placeholder=False),)

                def _stock_sub():  # noqa: ANN202
                    p = st.rows(r).materialize()
                    return (orig(x[:r].contiguous(), gate, p, indices[:r]),)

                if not _selfcheck(("fc2", tuple(x.shape[1:]), gate.dtype), _fused_sub, _stock_sub):
                    reason = "fc2 disabled by self-check"
            if reason is None:
                del ctx.pending["fc2"]
                out = gate_fused(st, x, gate, indices)
                _count("fused_fc2")
                return out
            _materialize_all(ctx, "fc2: " + reason)
        return orig(x, gate, other, indices)

    indexed_gate._h3_tp2_consumer_fuse = True
    return indexed_gate


def _lora_types():  # noqa: ANN202
    try:
        from vllm.lora.layers import BaseLayerWithLoRA
        return (BaseLayerWithLoRA,)
    except ImportError:
        return ()


def block_plan(block, rope_table, sites, lora_types=()) -> tuple[dict | None, str]:  # noqa: ANN001
    """{(weight data_ptr, weight shape): site} for this block's eligible GEMMs, or (None, reason)."""
    if rope_table is None:
        return None, "no rope table (q/k norm without fused rope)"
    try:
        if torch.compiler.is_compiling():
            return None, "torch.compile tracing"
    except AttributeError:
        pass
    layers = {"qkv": getattr(getattr(block, "attn", None), "qkv_proj", None),
              "out": getattr(getattr(block, "attn", None), "out_proj", None),
              "fc2": getattr(getattr(block, "mlp", None), "fc2", None)}
    want: dict = {}
    for site in sites:
        if site in _state["disabled"]:
            continue
        layer = layers[site]
        if layer is None or (lora_types and isinstance(layer, lora_types)):
            return None, f"{site}: missing or LoRA-wrapped layer"
        if getattr(layer, "tp_size", 1) != 1:
            return None, f"{site}: tp_size != 1 (an all-reduce would read the placeholder)"
        w = getattr(layer, "weight", None)
        n = int(getattr(layer, "output_size_per_partition", 0) or 0)
        if not isinstance(w, torch.Tensor) or w.dtype != torch.float8_e4m3fn or w.dim() != 2 or w.shape[1] != n:
            return None, f"{site}: weight is not an unpadded FP8 [K, N] tensor"
        want[(w.data_ptr(), tuple(w.shape))] = site
    return (want or None), "no site left"


def make_block_forward(orig_forward, sites, lora_types):  # noqa: ANN001,ANN201
    def forward(self, *args, **kwargs):  # noqa: ANN001,ANN202
        want, reason = block_plan(self, kwargs.get("rope_table"), sites, lora_types)
        if want is None:
            _warn_once("plan-" + reason, "block not fused (%s); stock path", reason)
            return orig_forward(self, *args, **kwargs)
        prev = getattr(_tls, "ctx", None)
        ctx = _Ctx(want)
        _tls.ctx = ctx
        ok = False
        try:
            out = orig_forward(self, *args, **kwargs)
            ok = True
            return out
        finally:
            _tls.ctx = prev
            if ctx.pending:
                if ok:   # cannot happen with the reviewed forwards; make the placeholder real for whoever holds it
                    _materialize_all(ctx, "stash left at the end of the block")
                ctx.pending.clear()

    forward._h3_tp2_consumer_fuse = True
    return forward


def parse_sites(value: str | None) -> tuple[str, ...]:
    """``H3_TP2_FP8_CONSUMER_FUSE``: "1"/"all" = every site; a comma list of qkv,out,fc2 for bisecting; else off."""
    v = (value or "").strip().lower()
    if v in ("1", "all", "on", "true"):
        return SITES
    if not v or v in ("0", "off", "false"):
        return ()
    sel = tuple(s for s in SITES if s in {p.strip() for p in v.split(",")})
    return sel


def _fingerprints(tr) -> dict[str, str | None]:  # noqa: ANN001
    from .step_cache import fingerprint

    fps = {"attn_forward": fingerprint(tr.MiniMaxH3Attention.forward),
           "block_forward": fingerprint(tr.MiniMaxH3DiTBlock.forward)}
    mlp_fwd = tr.MiniMaxH3MLP.forward
    fps["mlp_forward"] = EXPECTED["mlp_forward"] if getattr(tr.MiniMaxH3MLP, "_h3_tp2_fused_swiglu", False) \
        else fingerprint(mlp_fwd)   # fp8_swiglu's forward returns fc2's output unchanged as well
    try:
        from vllm_omni.diffusion.layers import fused_qk_norm_rope as fq
        from vllm_omni.diffusion.layers import indexed_modulation as im
    except ImportError:
        return fps
    src = {"rms_norm_rope_kernel": getattr(fq._rms_norm_rope_kernel, "fn", fq._rms_norm_rope_kernel),
           "launch_fused_rms_norm_rope": fq._launch_fused_rms_norm_rope,
           "fused_qk_norm_rope_impl": fq._fused_qk_norm_rope_impl,
           "fused_qk_norm_rope": fq.fused_qk_norm_rope,
           "fused_cuda_supported": fq._fused_cuda_supported,
           "indexed_gate_kernel": getattr(im._indexed_gate_kernel, "fn", im._indexed_gate_kernel),
           "indexed_gate": im.indexed_gate,
           "gate_rms_kernel": getattr(im._indexed_gate_rms_norm_scale_shift_kernel, "fn",
                                      im._indexed_gate_rms_norm_scale_shift_kernel),
           "gate_rms": im.indexed_gate_rms_norm_scale_shift,
           "launch_row_chunks": im._launch_row_chunks}
    fps.update({k: fingerprint(v) for k, v in src.items()})
    # the transformer must still call the upstream functions themselves (not someone else's wrapper)
    if tr.fused_qk_norm_rope is not fq.fused_qk_norm_rope:
        fps["fused_qk_norm_rope"] = "wrapped"
    if tr.indexed_gate is not im.indexed_gate:
        fps["indexed_gate"] = "wrapped"
    if tr.indexed_gate_rms_norm_scale_shift is not im.indexed_gate_rms_norm_scale_shift:
        fps["gate_rms"] = "wrapped"
    return fps


def install(sites: tuple[str, ...] = SITES, tr=None, *, check_source: bool = True, lora_types=None) -> list[str]:  # noqa: ANN001
    """Patch the transformer module ``tr`` (default: vLLM-Omni's MiniMax H3 transformer) for the given sites and register
    the stash hook.  Returns the installed sites; a site whose upstream source is not the reviewed one, or whose
    preconditions fail, is skipped with a warning (the stock path stays)."""
    sites = tuple(s for s in SITES if s in sites)
    if not sites:
        return []
    if fp8_gemm._state.get("level") != "tensor32" or fp8_gemm._state.get("fused"):
        logger.warning("h3_tp2 fp8 consumer fuse: needs H3_TP2_FP8_GEMM=tensor32 (two-pass); skipped")
        return []
    if not _state["cpu_mirror"] and not (_HAVE_TRITON and fp8_gemm._state.get("triton")):
        logger.warning("h3_tp2 fp8 consumer fuse: Triton scale pass not usable in this process; skipped")
        return []
    if tr is None:
        from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as tr
    block_cls = tr.MiniMaxH3DiTBlock
    if getattr(block_cls.forward, "_h3_tp2_consumer_fuse", False):
        return list(_state["sites"])
    try:
        from vllm_omni.diffusion.attention import layer as layer_mod
        if "qkv" in sites and getattr(layer_mod.Attention, "_h3_tp2_nocopy", 0):
            logger.warning("h3_tp2 fp8 consumer fuse: Ulysses nocopy wraps the rope producer; qkv site skipped")
            sites = tuple(s for s in sites if s != "qkv")
    except ImportError:
        pass
    if check_source:
        fps = _fingerprints(tr)
        keep = []
        for site in sites:
            bad = [n for n in SITE_SOURCES[site] if fps.get(n) != EXPECTED[n]]
            if bad:
                logger.warning("h3_tp2 fp8 consumer fuse: site %s skipped, upstream source changed: %s", site,
                               ", ".join(f"{n} {fps.get(n)} (reviewed {EXPECTED[n]})" for n in bad))
            else:
                keep.append(site)
        sites = tuple(keep)
    if not sites:
        return []
    if "qkv" in sites:
        tr.fused_qk_norm_rope = make_qk_wrapper(tr.fused_qk_norm_rope)
    if "out" in sites:
        tr.indexed_gate_rms_norm_scale_shift = make_gate_rms_wrapper(tr.indexed_gate_rms_norm_scale_shift)
    if "fc2" in sites:
        tr.indexed_gate = make_gate_wrapper(tr.indexed_gate)
    block_cls.forward = make_block_forward(block_cls.forward, sites, _lora_types() if lora_types is None else lora_types)
    fp8_gemm.set_stash_hook(stash_hook)
    _state["sites"] = sites
    return list(sites)


def stats() -> dict[str, int]:
    return dict(_stats)


if os.environ.get("H3_TP2_FP8_CONSUMER_CPU_MIRROR") == "1":   # tests only: route CPU tensors through the eager mirrors
    _state["cpu_mirror"] = True
