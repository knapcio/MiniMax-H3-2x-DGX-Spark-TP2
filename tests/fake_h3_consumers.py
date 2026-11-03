"""Fake vLLM-Omni MiniMax H3 DiT pieces for ``tests/test_fp8_consumer_fuse.py`` (CPU, no vLLM, no Triton, no GPU).

VERBATIM sections are copied unchanged from vLLM-Omni v0.30.0 (a8576ccb7, Apache-2.0): ``MiniMaxH3Attention.forward``
(``minimax_h3_transformer.py:630-704``), ``MiniMaxH3MLP.forward`` (``:738-742``) and ``MiniMaxH3DiTBlock.forward``
(``:919-982``), so the producer -> consumer call structure the fused path relies on is upstream's; and, as text only
(``UPSTREAM_SOURCES``), the Triton kernels and launchers whose copies live in ``h3_tp2/fp8_consumers.py``
(``fused_qk_norm_rope.py`` and ``indexed_modulation.py``), so the test can check the fingerprints in
``fp8_consumers.EXPECTED`` were taken over this exact source.

The stock consumers below are eager torch mirrors of those Triton kernels, written from the kernel code (upstream's own
CPU branches round differently and are not what the GPU runs).  The FP8 linear layers quantise per token and go
through the same hook list as ``fp8_gemm``'s dispatcher, then the two-pass path (fp32 product + the scale pass's eager
mirror with an fma for the bias).  Load a fresh copy with ``load()``.
"""

from __future__ import annotations

import importlib.util
import math

import torch
from torch import nn

from h3_tp2 import fp8_consumers, fp8_gemm

BF16 = torch.bfloat16
E4M3 = torch.float8_e4m3fn


# ---- stock consumers: mirrors of the upstream Triton kernels -------------------------------------------------------
def _rms_norm_rope_stock(x, weight, rope_table, eps):  # noqa: ANN001,ANN202
    """_rms_norm_rope_kernel (fused_qk_norm_rope.py:108-168) on a bf16 [T, H, D] view (any strides)."""
    T, H, Dh = x.shape
    rotary_half = rope_table.shape[-1] // 2
    rotary_dim = rotary_half * 2
    dims = torch.arange(Dh)
    xf = x.float()
    w = weight.float()
    inv_rms = torch.rsqrt((xf * xf).sum(dim=-1) / Dh + eps)
    normalized = (xf * inv_rms[..., None] * w).to(BF16)
    pair_dims = torch.where(dims < rotary_half, dims + rotary_half, torch.where(dims < rotary_dim, dims - rotary_half, dims))
    pair_x = xf[..., pair_dims]
    pair_normalized = (pair_x * inv_rms[..., None] * w[pair_dims]).to(BF16)
    freq_dims = torch.where(dims < rotary_half, dims, torch.where(dims < rotary_dim, dims - rotary_half, torch.zeros_like(dims)))
    cos = rope_table[:, freq_dims].float()[:, None, :]
    sin = rope_table[:, freq_dims + rotary_half].float()[:, None, :]
    first = normalized.float() * cos - pair_normalized.float() * sin
    second = normalized.float() * cos + pair_normalized.float() * sin
    output = torch.where(dims < rotary_dim, torch.where(dims < rotary_half, first, second), normalized.float())
    return output.to(x.dtype).contiguous()


def fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps, *, head_dim=None, rotary_dim=None, interleaved=False):  # noqa: ANN001,ANN202
    """The CUDA half-split path of fused_qk_norm_rope (two launches of _rms_norm_rope_kernel)."""
    CALLS.append("fused_qk_norm_rope")
    assert not interleaved and q.shape[-1] == 128 and rope_table.shape[-1] == 96
    return _rms_norm_rope_stock(q, q_weight, rope_table, eps), _rms_norm_rope_stock(k, k_weight, rope_table, eps)


def indexed_gate(x, gate, other, indices):  # noqa: ANN001,ANN202
    """_indexed_gate_kernel (indexed_modulation.py:58-87)."""
    CALLS.append("indexed_gate")
    return (x.float() + gate.index_select(0, indices).float() * other.float()).to(x.dtype)


def indexed_gate_rms_norm_scale_shift(residual, gate, branch, weight, shift, scale, indices, eps):  # noqa: ANN001,ANN202
    """_indexed_gate_rms_norm_scale_shift_kernel (indexed_modulation.py:129-173)."""
    CALLS.append("indexed_gate_rms_norm_scale_shift")
    H = residual.shape[1]
    updated = (residual.float() + gate.index_select(0, indices).float() * branch.float()).to(BF16).float()
    variance = (updated * updated).sum(dim=1, keepdim=True) / H
    normalized = updated * torch.rsqrt(variance + eps) * weight.float()
    output = normalized * (1.0 + scale.index_select(0, indices).float()) + shift.index_select(0, indices).float()
    return updated.to(residual.dtype), output.to(residual.dtype)


def rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps):  # noqa: ANN001,ANN202
    """_rms_norm_indexed_scale_shift_kernel (indexed_modulation.py:91-126)."""
    H = x.shape[1]
    xf = x.float()
    variance = (xf * xf).sum(dim=1, keepdim=True) / H
    normalized = xf * torch.rsqrt(variance + eps) * weight.float()
    output = normalized * (1.0 + scale.index_select(0, indices).float()) + shift.index_select(0, indices).float()
    return output.to(x.dtype)


CALLS: list[str] = []


# ---- fake layers ---------------------------------------------------------------------------------------------------
def _two_pass_cpu(a, b, scale_a, scale_b, out_dtype, bias=None):  # noqa: ANN001,ANN202
    """``scaled_mm_tensor32`` off the GPU: fp32 product + the scale pass (eager mirror, an fma for the bias)."""
    M, N = a.shape[0], b.shape[1]
    y = torch.empty((M, N), dtype=out_dtype)
    return fp8_consumers.scale_pass_into(fp8_consumers._product_fp32(a, b), fp8_consumers._vec(scale_a, M),
                                         fp8_consumers._vec(scale_b, N), bias, y)


def dispatch(a, b, scale_a, scale_b, out_dtype, bias):  # noqa: ANN001,ANN202
    """fp8_gemm's dispatcher, two-pass branch (``fp8_gemm.route``: routers, then the hooks, then two-pass), with the
    CPU two-pass mirror."""
    return fp8_gemm.route(a, b, scale_a, scale_b, out_dtype, bias, two_pass=_two_pass_cpu)


class FakeFP8Linear(nn.Module):
    """Online-FP8 linear: weight [K, N] fp8 (the transpose of a row-major [N, K]), per-channel scales, tp 1."""

    def __init__(self, k: int, n: int, gen: torch.Generator, *, bias: bool = False, tp_size: int = 1, pad_n: int = 0):
        super().__init__()
        w = torch.randn(n + pad_n, k, generator=gen) * 0.05
        ws = w.abs().amax(dim=1).clamp(min=1e-6) / 448.0
        self.weight = (w / ws[:, None]).clamp(-448, 448).to(E4M3).t()
        self.weight_scale = ws.float()
        self.output_size_per_partition = n
        self.tp_size = tp_size
        self.bias = (torch.randn(n + pad_n, generator=gen) * 0.1).to(BF16) if bias else None

    def forward(self, x):  # noqa: ANN001,ANN201
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).float()
        s = (x2.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1.0 / (448.0 * 512.0))
        a = (x2 / s).clamp(-448, 448).to(E4M3)
        out = dispatch(a, self.weight, s, self.weight_scale, BF16, self.bias)
        n = self.output_size_per_partition
        if out.shape[1] != n:
            out = out[:, :n].contiguous()
        return out.view(*lead, n), None


class RMSNorm(nn.Module):
    def __init__(self, n: int, gen: torch.Generator, eps: float = 1e-6):
        super().__init__()
        self.weight = (1.0 + 0.1 * torch.randn(n, generator=gen)).to(BF16)
        self.variance_epsilon = eps

    def forward(self, x):  # noqa: ANN001,ANN201
        xf = x.float()
        return (xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + self.variance_epsilon) * self.weight.float()).to(x.dtype)


class SiluAndMul(nn.Module):
    def forward(self, h):  # noqa: ANN001,ANN201
        d = h.shape[-1] // 2
        g = h[..., :d].float()
        return ((g / (1.0 + torch.exp(-g))).to(BF16).float() * h[..., d:].float()).to(BF16)


class AdalnProj(nn.Module):
    """Six [U, H] chunk views of one [U, 6H] tensor (row stride 6H), like MiniMaxH3AdalnProj."""

    def __init__(self, hidden: int, gen: torch.Generator):
        super().__init__()
        self.hidden = hidden
        self.w = torch.randn(16, 6 * hidden, generator=gen) * 0.05

    def forward(self, t_emb):  # noqa: ANN001,ANN201
        x = (t_emb.float() @ self.w).to(BF16)
        return tuple(x.chunk(6, dim=-1))


class MiniMaxH3Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, gen: torch.Generator, *, bias: bool = False):
        super().__init__()
        self.head_dim = 128
        self.num_heads = heads
        self.num_kv_heads = heads
        self.softmax_scale = self.head_dim ** -0.5
        self.qkv_proj = FakeFP8Linear(hidden, 3 * heads * 128, gen, bias=bias)
        self.out_proj = FakeFP8Linear(heads * 128, hidden, gen, bias=bias)
        self.q_norm = RMSNorm(128, gen)
        self.k_norm = RMSNorm(128, gen)
        self.to_gate_compress = None
        self.seen_v: list[torch.Tensor] = []

    def _run_packed_attention(self, q, k, v, *, cu_seqlens, max_seqlen, packed_total, num_requests=1,  # noqa: ANN001,ANN202
                              video_layout=None, vsa_prefix_segments=(), gate_compress=None):
        self.seen_v.append(v)                       # the strided v view of the qkv output, as the attention gets it
        qf, kf, vf = (t.float().transpose(0, 1) for t in (q, k, v))
        p = torch.softmax(qf @ kf.transpose(1, 2) * self.softmax_scale, dim=-1)
        return (p @ vf).transpose(0, 1).to(BF16)

    # ---- VERBATIM minimax_h3_transformer.py:630-704 ----
    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_table: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_total: int | None = None,
        num_requests: int = 1,
        sp_seq_lens: list[int] | None = None,
        video_layout: VideoTokenLayout | None = None,
        vsa_prefix_segments: tuple[int, ...] = (),
    ) -> torch.Tensor:
        """x: [T, hidden] packed thd rows -> [T, hidden].

        Operation order: fused qkv projection -> per-head q/k RMSNorm -> RoPE
        on q/k -> variable-length non-causal flash attention -> output projection.

        With Ulysses sequence parallelism, x holds this rank's row shard;
        qkv/norm/RoPE run locally, an all-to-all trades sequence for heads.
        Each rank attends the full sequence with heads/world_size local heads,
        so cu_seqlens retains global packed-document semantics. The inverse
        all-to-all restores the row shard before the output projection.
        """
        total = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_kv_heads, self.head_dim)
        v = v.view(total, self.num_kv_heads, self.head_dim)
        if rope_table is None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        else:
            q, k = fused_qk_norm_rope(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                rope_table,
                self.q_norm.variance_epsilon,
            )

        # The gate is projected from the same local rows as Q. Pure Ulysses
        # reshards it alongside Q/K/V in UlyssesParallelAttention so each VSA
        # rank receives the full sequence for its local head shard.
        gate_compress = None
        if self.to_gate_compress is not None:
            gate_result = self.to_gate_compress(x)
            gate_compress = gate_result[0] if isinstance(gate_result, tuple) else gate_result
            gate_compress = gate_compress.view(total, self.num_heads, self.head_dim)

        # Each request contributes a document for its rows plus one for any
        # nonempty alignment padding. Local/Ulysses backends unpad it, while
        # Ring keeps aligned rows for fixed-size P2P buffers.
        out = self._run_packed_attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            # Before Ulysses, q contains only this rank's row shard. The
            # backend receives the global sequence after all-to-all, so carry
            # its Python length explicitly instead of inferring it from q.
            packed_total=packed_total if packed_total is not None else q.shape[0],
            num_requests=num_requests,
            video_layout=video_layout,
            vsa_prefix_segments=vsa_prefix_segments,
            gate_compress=gate_compress,
        )
        out = out.reshape(total, self.num_heads * self.head_dim)
        out, _ = self.out_proj(out)
        return out


class MiniMaxH3MLP(nn.Module):
    def __init__(self, hidden: int, ffn: int, gen: torch.Generator, *, bias: bool = False):
        super().__init__()
        self.fc1 = FakeFP8Linear(hidden, 2 * ffn, gen, bias=bias)
        self.act_fn = SiluAndMul()
        self.fc2 = FakeFP8Linear(ffn, hidden, gen, bias=bias)

    # ---- VERBATIM minimax_h3_transformer.py:738-742 ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.fc1(x)
        hidden = self.act_fn(hidden)
        out, _ = self.fc2(hidden)
        return out


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, ffn: int, seed: int = 0, *, bias: bool = False):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.norm1 = RMSNorm(hidden, gen)
        self.norm2 = RMSNorm(hidden, gen)
        self.attn = MiniMaxH3Attention(hidden, heads, gen, bias=bias)
        self.mlp = MiniMaxH3MLP(hidden, ffn, gen, bias=bias)
        self.adaln_proj = AdalnProj(hidden, gen)

    # ---- VERBATIM minimax_h3_transformer.py:919-982 ----
    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_table: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_total: int,
        num_requests: int = 1,
        sp_seq_lens: list[int] | None = None,
        video_layout: VideoTokenLayout | None = None,
        vsa_prefix_segments: tuple[int, ...] = (),
    ) -> torch.Tensor:
        """x: [T, H]; t_emb: [M, t_dim]; combined_indices: [T]
        (= inverse_indices * modality_num + token_tags.clamp(min=0)).

        Each block computes AdaLN parameters once, then applies
        norm1 -> scale/shift -> attention -> gated residual, followed by
        norm2 -> scale/shift -> MLP -> gated residual.
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_proj(t_emb)

        residual = x
        h = rms_norm_indexed_scale_shift(
            x,
            self.norm1.weight,
            shift_msa,
            scale_msa,
            combined_indices,
            self.norm1.variance_epsilon,
        )
        h = self.attn(
            h,
            rope_table=rope_table,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            packed_total=packed_total,
            num_requests=num_requests,
            sp_seq_lens=sp_seq_lens,
            video_layout=video_layout,
            vsa_prefix_segments=vsa_prefix_segments,
        )
        x, h = indexed_gate_rms_norm_scale_shift(
            residual,
            gate_msa,
            h,
            self.norm2.weight,
            shift_mlp,
            scale_mlp,
            combined_indices,
            self.norm2.variance_epsilon,
        )
        residual = x
        h = self.mlp(h)
        return indexed_gate(residual, gate_mlp, h, combined_indices)


def block_inputs(block: MiniMaxH3DiTBlock, rows: int, seed: int = 1, *, with_rope: bool = True) -> dict:
    """Packed rows, two condition embeddings, an H3-like rope table [rows, 96] (bf16)."""
    gen = torch.Generator().manual_seed(seed)
    hidden = block.norm1.weight.shape[0]
    x = torch.randn(rows, hidden, generator=gen).to(BF16)
    t_emb = torch.randn(2, 16, generator=gen)
    idx = torch.randint(0, 2, (rows,), generator=gen)
    pos = torch.arange(rows, dtype=torch.float32)[:, None] * torch.exp(-torch.arange(48) / 8.0)[None, :]
    rope = torch.cat((torch.cos(pos), torch.sin(pos)), dim=-1).to(BF16)
    return dict(x=x, t_emb=t_emb, combined_indices=idx, rope_table=rope if with_rope else None,
                cu_seqlens=torch.tensor([0, rows]), max_seqlen=rows, packed_total=rows)


# ---- VERBATIM sources (text only) of the kernels and launchers the fused copies were made from ----------------------
UPSTREAM_SOURCES = {
    'rms_norm_rope_kernel': r'''
    @triton.jit
    def _rms_norm_rope_kernel(
        x_ptr,
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
    ):
        # Long-video Q/K views can span more than 2**31 elements of fused QKV storage.
        token = tl.program_id(0).to(tl.int64)
        head_group = tl.program_id(1)
        heads = head_group * heads_per_program + tl.arange(0, heads_per_program)
        dims = tl.arange(0, head_dim)
        mask = heads[:, None] < num_heads
        offsets = token * x_stride_t + heads[:, None] * x_stride_h + dims[None, :] * x_stride_d

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + dims).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
        normalized = (x * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        rotary_dim = rotary_half * 2
        pair_dims = tl.where(
            dims < rotary_half,
            dims + rotary_half,
            tl.where(dims < rotary_dim, dims - rotary_half, dims),
        )
        pair_offsets = token * x_stride_t + heads[:, None] * x_stride_h + pair_dims[None, :] * x_stride_d
        pair_x = tl.load(x_ptr + pair_offsets, mask=mask, other=0.0).to(tl.float32)
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
''',
    'launch_fused_rms_norm_rope': r'''
def _launch_fused_rms_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    tokens, heads, head_dim = x.shape
    rotary_half = rope_table.shape[-1] // 2
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    grid = (tokens, triton.cdiv(heads, _HEADS_PER_PROGRAM))
    _rms_norm_rope_kernel[grid](
        x,
        weight,
        rope_table,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        rope_table.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads=heads,
        head_dim=head_dim,
        rotary_half=rotary_half,
        eps=eps,
        heads_per_program=_HEADS_PER_PROGRAM,
        num_warps=8,
    )
    return out
''',
    'fused_qk_norm_rope_impl': r'''
def _fused_qk_norm_rope_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _fused_npu_supported(q, k, head_dim, rotary_dim, interleaved):
        return _npu_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            rotary_dim,
        )
    if not _fused_cuda_supported(q, k, head_dim, rotary_dim, interleaved):
        return _eager_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            head_dim,
            rotary_dim,
            interleaved,
        )
    if interleaved:
        return _launch_fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps, interleaved=True)
    # Half-split production traffic keeps the historical per-tensor kernel;
    # _qk_norm_rope_kernel's half-split mode awaits the maintainers' call.
    return (
        _launch_fused_rms_norm_rope(q, q_weight, rope_table, eps),
        _launch_fused_rms_norm_rope(k, k_weight, rope_table, eps),
    )
''',
    'fused_qk_norm_rope': r'''
def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    *,
    head_dim: int | None = None,
    rotary_dim: int | None = None,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Q/K RMSNorm and packed RoPE (half-split or adjacent pairs)."""
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"q and k must be [tokens, heads, head_dim], got {q.shape} and {k.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k shapes are incompatible: {q.shape} and {k.shape}")
    if q.dtype != k.dtype or q.device != k.device:
        raise ValueError("q and k must have the same dtype and device")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"Fused QK RMSNorm/RoPE requires floating inputs, got {q.dtype}")

    head_dim = q.shape[-1] if head_dim is None else head_dim
    rotary_dim = rope_table.shape[-1] if rotary_dim is None else rotary_dim
    if q.shape[-1] != head_dim:
        raise ValueError(f"Expected q/k head_dim={head_dim}, got {q.shape[-1]}")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and in [2, {head_dim}], got {rotary_dim}")
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(f"Expected norm weights [{head_dim}], got {tuple(q_weight.shape)} and {tuple(k_weight.shape)}")
    if q_weight.device != q.device or k_weight.device != q.device:
        raise ValueError("Q/K norm weights must be on the activation device")
    if rope_table.device != q.device or rope_table.dtype not in (
        q.dtype,
        torch.float32,
    ):
        raise ValueError("rope_table must be on q/k's device with their dtype or float32")
    if rope_table.shape != (q.shape[0], rotary_dim):
        raise ValueError(f"Expected rope_table [{q.shape[0]}, {rotary_dim}], got {tuple(rope_table.shape)}")

    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    rope_table = rope_table.contiguous()
    if _fused_npu_supported(q, k, head_dim, rotary_dim, interleaved):
        return _npu_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            rotary_dim,
        )
    if not _fused_cuda_supported(q, k, head_dim, rotary_dim, interleaved):
        return _fused_qk_norm_rope_impl(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            head_dim,
            rotary_dim,
            interleaved,
        )
    return torch.ops.vllm_omni.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        eps,
        head_dim,
        rotary_dim,
        interleaved,
    )
''',
    'fused_cuda_supported': r'''
def _fused_cuda_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool = False,
) -> bool:
    if not (
        HAS_TRITON
        and current_platform.is_cuda()
        and q.is_cuda
        and k.is_cuda
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
    ):
        return False
    if interleaved:
        return rotary_dim % 2 == 0 and 2 <= rotary_dim <= head_dim <= _FUSED_MAX_HEAD_DIM
    # Half-split traffic keeps the pre-existing per-tensor kernel and its
    # exact MiniMax-H3 geometry contract (that kernel is untouched by the
    # interleaved extension).
    return head_dim == _FUSED_HEAD_DIM and rotary_dim == _FUSED_ROTARY_DIM
''',
    'indexed_gate_kernel': r'''
@triton.jit
def _indexed_gate_kernel(
    output_ptr,
    x_ptr,
    gate_ptr,
    other_ptr,
    indices_ptr,
    hidden_size,
    stride_output_row,
    stride_x_row,
    stride_gate_row,
    stride_other_row,
    stride_indices,
    row_offset,
    block_n: tl.constexpr,
):
    row = tl.program_id(0) + row_offset
    columns = tl.arange(0, block_n)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0).to(tl.float32)
    other = tl.load(other_ptr + row * stride_other_row + columns, mask=mask, other=0.0).to(tl.float32)

    tl.store(
        output_ptr + row * stride_output_row + columns,
        x + gate * other,
        mask=mask,
    )
''',
    'indexed_gate': r'''
def indexed_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Return ``x + gate[indices] * other`` without indexed temporaries."""
    if x.is_cpu:
        return (x + gate.index_select(0, indices) * other).to(x.dtype)
    output = torch.empty_like(x)
    rows, hidden_size = x.shape
    if rows == 0:
        return output
    _launch_row_chunks(
        _indexed_gate_kernel,
        rows,
        x.device.type,
        output,
        x,
        gate,
        other,
        indices,
        hidden_size,
        output.stride(0),
        x.stride(0),
        gate.stride(0),
        other.stride(0),
        indices.stride(0),
        block_n=triton.next_power_of_2(hidden_size),
        num_warps=8,
    )
    return output
''',
    'gate_rms_kernel': r'''
@triton.jit
def _indexed_gate_rms_norm_scale_shift_kernel(
    residual_out_ptr,
    modulated_out_ptr,
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
    block_n: tl.constexpr,
):
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
    tl.store(
        modulated_out_ptr + row * hidden_size + columns,
        output,
        mask=mask,
    )
''',
    'gate_rms': r'''
def indexed_gate_rms_norm_scale_shift(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse gated residual, the following RMSNorm, and AdaLN affine."""
    if residual.is_cpu:
        input_dtype = residual.dtype
        residual_out = (residual + gate.index_select(0, indices) * branch).to(input_dtype)
        normalized = residual_out.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + eps)
        normalized = (weight.float() * normalized).to(input_dtype)
        modulated_out = (normalized * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(
            input_dtype
        )
        return residual_out, modulated_out
    residual_out = torch.empty_like(residual)
    modulated_out = torch.empty_like(residual)
    rows, hidden_size = residual.shape
    if rows:
        _launch_row_chunks(
            _indexed_gate_rms_norm_scale_shift_kernel,
            rows,
            residual.device.type,
            residual_out,
            modulated_out,
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            hidden_size,
            eps,
            residual.stride(0),
            gate.stride(0),
            branch.stride(0),
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            block_n=triton.next_power_of_2(hidden_size),
            num_warps=8,
        )
    return residual_out, modulated_out
''',
    'launch_row_chunks': r'''
def _launch_row_chunks(kernel, rows: int, device_type: str, *args, **kwargs) -> None:
    if device_type != "npu":
        kernel[(rows,)](*args, 0, **kwargs)
        return
    for row_offset, chunk_rows in _iter_row_chunks(rows):
        kernel[(chunk_rows,)](*args, row_offset, **kwargs)
''',
}


def load():  # noqa: ANN201
    """A fresh, independent copy of this module (each test patches its own)."""
    spec = importlib.util.spec_from_file_location(f"_fake_h3_consumers_{id(object())}", __file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ = math  # (kept for parity with upstream imports)
