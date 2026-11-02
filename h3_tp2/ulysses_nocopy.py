"""Ulysses exchange without permute copies (``H3_TP2_ULYSSES_NOCOPY=1``, off by default; speed plan 2026-11-02 item 5).

Stock strict Ulysses (vLLM-Omni v0.30.0 ``a8576ccb7``, ``SeqAllToAll4D`` -> ``comm.all_to_all_4D``) moves q, k and v
with three separate all-to-alls and packs each one first: ``comm.py:44`` ``reshape(...).transpose(0, 2).contiguous()``
copies the whole [M, 56, 128] tensor (v is the strided split view of the qkv GEMM output, k and q the outputs of
``fused_qk_norm_rope``), so three ~615 MB copies at 1344x768.  The receive side (``comm.py:60``) is free for batch 1.
The reverse exchange copies twice: ``comm.py:74-80`` packs the attention output by destination and ``comm.py:96``
unpacks the receive into [M, 56, 128].  Five copies of ~615 MB per layer.

This module, at batch 1:

* lets the q/k producer write straight into ONE send buffer ``send[group, dest, M_local, 3, heads_local, d]``: the
  stock ``_rms_norm_rope_kernel`` (the Triton kernel ``fused_qk_norm_rope`` runs for H3) is launched once per
  destination block with its output pointer and strides aimed at ``send[g, p, :, 0]`` / ``[:, 1]``.  v is copied
  into slot 2 once (stock copies it once as well).  If the producer cannot be redirected (no rope table, not the
  Triton path, VSA gate, ...) q, k and v are packed with one ``torch.stack`` into the same buffer instead;
* runs ONE all-to-all per head group for q, k and v together; the receive buffer ``recv[dest-1, M_local, 3, hl, d]``
  flattened to [S, 3, hl, d] gives q, k, v as strided views [1, S, hl, d] (row stride 3*hl*d, the packed-QKV layout)
  that go to cuDNN with no further copy;
* sends the attention output itself (dense [S, hl, d]: its row blocks already are the per-destination chunks) and
  unpacks the receive with one copy straight into the [1, M_local, heads, d] tensor the out-projection reads.

Five copies become two (the v pack and the reverse unpack) with the producer redirected; with the fallback pack
(same bytes as the three stock packs) only the reverse pack is saved.  The same elements meet the same
arithmetic: the norm+RoPE kernel computes every (token, head) row independently and attention is per head, so only
copies and layouts change.  Two things are NOT proven on the CPU and must be checked on the GPU before this is enabled
(see the README row): that cuDNN SDPA picks a plan with identical results for the strided q/k/v views, and that the
per-destination launches of the Triton kernel reproduce its single-launch rows bit for bit.

With ``H3_TP2_A2A_OVERLAP=<groups>`` set as well, the same layout is used per head group and the exchanges run on the
overlap module's comm stream (same memory rule as ``ulysses_overlap``: every buffer that one stream writes and the
other reads is allocated on the compute stream and stays referenced until the compute stream has joined the comm
stream; no ``record_stream``).  Anything outside strict Ulysses, batch 1, no ring, no joint tensors, no masks, no
paged KV, no VSA gate, or heads not divisible by groups*world falls back to the stock forward.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import torch
import torch.distributed as dist

logger = logging.getLogger("vllm_omni.h3_tp2.nocopy")

# Per thread (one rank per process in production; the CPU tests run two ranks as threads):
#   armed   - the MiniMaxH3Attention whose forward is running (set by the forward wrapper)
#   pending - q/k already written into a send buffer by the redirected producer, waiting for the attention layer
_tls = threading.local()
_stats = {"redirect": 0, "pack": 0, "fallback": 0, "materialize": 0, "out_copy": 0}


def _count(kind: str) -> None:
    """Per-process counters; the first occurrence of each path is logged so a GPU run shows which one it took."""
    _stats[kind] += 1
    if _stats[kind] == 1:
        log = logger.warning if kind in ("materialize", "out_copy") else logger.info
        log("h3_tp2 nocopy: first %s exchange", kind)


def _a2a(out: torch.Tensor, inp: torch.Tensor, group=None) -> None:  # noqa: ANN001
    # looked up at call time so h3_tp2.a2a_timing (which patches torch.distributed.all_to_all_single) still sees it
    dist.all_to_all_single(out, inp, group=group)


# ---------------------------------------------------------------------------------------------------------------
# layout helpers (pure torch; tested on the CPU)
# ---------------------------------------------------------------------------------------------------------------


def send_shape(s_loc: int, heads: int, d: int, ws: int, groups: int) -> tuple[int, ...]:
    return (groups, ws, s_loc, 3, heads // (groups * ws), d)


def _by_dest(x: torch.Tensor, ws: int, groups: int) -> torch.Tensor:
    """[M, H, d] (any strides) -> view [G, P, M, hl, d]; head h = g*hg + p*hl + j."""
    m, h, d = x.shape
    return x.unflatten(1, (groups, ws, h // (groups * ws))).permute(1, 2, 0, 3, 4)


def pack_qkv(send: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ws: int, groups: int) -> None:
    """Fallback producer: q, k, v [M, H, d] -> send [G, P, M, 3, hl, d] in one stack."""
    torch.stack((_by_dest(q, ws, groups), _by_dest(k, ws, groups), _by_dest(v, ws, groups)), dim=3, out=send)


def fill_slot(send: torch.Tensor, x: torch.Tensor, slot: int, ws: int, groups: int) -> None:
    send[:, :, :, slot].copy_(_by_dest(x, ws, groups))


def rope_into_send(send, q, k, q_weight, k_weight, rope_table, eps, ws, groups, launch) -> None:  # noqa: ANN001
    """Redirected producer: ``launch(x_block, weight, rope_table, eps, out_view)`` writes norm+RoPE of the heads that go
    to one destination straight into its slot of the send buffer.  ``x_block`` is [M, hl, d] of the original q/k."""
    hl = send.shape[4]
    hg = hl * ws
    for g in range(groups):
        for p in range(ws):
            h0 = g * hg + p * hl
            launch(q[:, h0 : h0 + hl], q_weight, rope_table, eps, send[g, p, :, 0])
            launch(k[:, h0 : h0 + hl], k_weight, rope_table, eps, send[g, p, :, 1])


def qkv_views(recv_g: torch.Tensor):  # noqa: ANN201
    """recv [P, M, 3, hl, d] (contiguous) -> q, k, v strided views [1, P*M, hl, d] (row stride 3*hl*d)."""
    p, m, _, hl, d = recv_g.shape
    flat = recv_g.view(p * m, 3, hl, d)
    return flat[:, 0].unsqueeze(0), flat[:, 1].unsqueeze(0), flat[:, 2].unsqueeze(0)


def back_into(o: torch.Tensor, recv_b: torch.Tensor, result: torch.Tensor, g: int, ws: int, a2a, pg) -> None:  # noqa: ANN001
    """Attention output o [1, S, hl, d] of head group g -> result[0, :, g*hg:(g+1)*hg] with one a2a and one copy."""
    o3 = o[0]
    if not o3.is_contiguous():  # cuDNN returns dense [S, hl, d] (q-matching layout); anything else costs a copy
        _count("out_copy")
        o3 = o3.contiguous()
    a2a(recv_b, o3, group=pg)  # dim 0 of o3 is S: chunk p = the rows of rank p's sequence shard
    _, s_loc, hl, d = recv_b.shape
    hg = hl * ws
    result[0, :, g * hg : (g + 1) * hg].unflatten(1, (ws, hl)).copy_(recv_b.permute(1, 0, 2, 3))


def nocopy_forward(query, key, value, run_attn, *, ws, groups, pg=None, a2a=None, send_box=None, comm=None):  # noqa: ANN001,ANN201
    """Ulysses strict forward for batch 1 without permute copies.

    query/key/value [1, M, H, d] (any strides; ignored for q/k when ``send_box`` holds a buffer the producer already
    filled with q and k).  ``run_attn(q, k, v)`` is the local attention over [1, S, hl, d].  ``send_box`` is a list so
    the buffer can be dropped right after the exchange.  ``comm``: a CUDA stream for grouped overlap, or None to run
    everything on the current stream.  Returns [1, M, H, d] contiguous."""
    a2a = a2a or _a2a
    _, s_loc, heads, d = query.shape
    dt, dev = query.dtype, query.device
    if send_box:
        send = send_box.pop()
        fill_slot(send, value[0], 2, ws, groups)
    else:
        send = torch.empty(send_shape(s_loc, heads, d, ws, groups), dtype=dt, device=dev)
        pack_qkv(send, query[0], key[0], value[0], ws, groups)
    hl = send.shape[4]
    recv = torch.empty_like(send)
    recv_b = torch.empty((groups, ws, s_loc, hl, d), dtype=dt, device=dev)
    result = torch.empty((1, s_loc, heads, d), dtype=dt, device=dev)

    if comm is None:
        for g in range(groups):
            a2a(recv[g], send[g], group=pg)
        del send  # same stream: free the 3x[M, H, d] send buffer before attention
        for g in range(groups):
            o = run_attn(*qkv_views(recv[g]))
            back_into(o, recv_b[g], result, g, ws, a2a, pg)
        return result

    # grouped overlap: every buffer above was allocated on the compute stream and stays referenced to the end
    cur = torch.cuda.current_stream()
    comm.wait_stream(cur)
    try:
        ready = []
        with torch.cuda.stream(comm):
            for g in range(groups):
                a2a(recv[g], send[g], group=pg)
                ev = torch.cuda.Event()
                ev.record(comm)
                ready.append(ev)
        outs, back_ev = [], []
        for g in range(groups):
            cur.wait_event(ready[g])
            o = run_attn(*qkv_views(recv[g]))
            outs.append(o)  # read by the comm stream below; kept alive until the join
            done = torch.cuda.Event()
            done.record(cur)
            with torch.cuda.stream(comm):
                comm.wait_event(done)
                back_into(o, recv_b[g], result, g, ws, a2a, pg)
                ev = torch.cuda.Event()
                ev.record(comm)
            back_ev.append(ev)
        for ev in back_ev:
            cur.wait_event(ev)
        return result
    except Exception:
        cur.wait_stream(comm)
        raise


# ---------------------------------------------------------------------------------------------------------------
# producer redirect: placeholders + pending send buffer
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class _Pending:
    send: torch.Tensor  # [G, P, M, 3, hl, d], q in slot 0, k in slot 1, slot 2 empty
    q_ph: torch.Tensor  # stride-0 placeholders handed back to the model in place of q and k
    k_ph: torch.Tensor
    ws: int
    groups: int

    def matches(self, query: torch.Tensor, key: torch.Tensor) -> bool:
        return (
            query.ndim == 4
            and query.data_ptr() == self.q_ph.data_ptr()
            and key.data_ptr() == self.k_ph.data_ptr()
            and tuple(query.shape[1:]) == tuple(self.q_ph.shape)
        )

    def materialize(self):  # noqa: ANN201
        """Real q, k [1, M, H, d] from the send buffer (used only when the consumer falls back to stock)."""
        _count("materialize")
        out = []
        for slot in (0, 1):
            x = self.send[:, :, :, slot]  # [G, P, M, hl, d]
            g, p, m, hl, d = x.shape
            out.append(x.permute(2, 0, 1, 3, 4).reshape(m, g * p * hl, d).unsqueeze(0))
        return out[0], out[1]


def _placeholder(like: torch.Tensor) -> torch.Tensor:
    """A [M, H, d] tensor that owns one element: nothing may read it (the consumer takes q/k from the send buffer)."""
    return like.new_empty(1).as_strided(tuple(like.shape), (0,) * like.ndim)


def _is_placeholder(t: torch.Tensor) -> bool:
    return t.numel() > 1 and all(s == 0 for s in t.stride())


def take_pending(query: torch.Tensor, key: torch.Tensor) -> "_Pending | None":
    pend = getattr(_tls, "pending", None)
    if pend is not None and pend.matches(query, key):
        _tls.pending = None
        return pend
    if _is_placeholder(query) or _is_placeholder(key):
        raise RuntimeError("h3_tp2 nocopy: placeholder q/k reached attention without their send buffer")
    return None


def eligible_shape(query, key, value, ws: int, groups: int) -> bool:  # noqa: ANN001
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        return False
    if query.shape[0] != 1 or ws < 2 or groups < 1:
        return False
    heads = query.shape[2]
    if heads % (groups * ws) != 0:
        return False
    return key.dtype == query.dtype == value.dtype and key.device == query.device == value.device


def make_forward_impl(orig, groups: int, strategy_info, comm_stream=None, a2a=None):  # noqa: ANN001,ANN201
    """Wrap ``Attention._forward_impl``.  ``strategy_info(layer, attn_metadata)`` -> (pg, ws) when the stock forward
    would run strict Ulysses with nothing this module does not handle, else None."""

    def _forward_impl(self, query, key, value, attn_metadata=None):  # noqa: ANN001
        pend = take_pending(query, key)
        info = strategy_info(self, attn_metadata)
        ok = info is not None and eligible_shape(query, key, value, info[1], groups)  # placeholders carry the real shape
        if pend is not None and (not ok or pend.ws != info[1] or pend.groups != groups):
            query, key = pend.materialize()
            pend = None
        if not ok:
            _count("fallback")
            return orig(self, query, key, value, attn_metadata)
        pg, ws = info
        attn_metadata = self._with_kv_cache_dtype(attn_metadata)

        def run(q, k, v):  # noqa: ANN001,ANN202
            return self._run_local_attention(q, k, v, attn_metadata)

        if pend is not None:
            box = [pend.send]
            pend.send = None
            pend = None
            _count("redirect")
        else:
            box = None
            _count("pack")
        comm = comm_stream() if (groups > 1 and comm_stream is not None and query.is_cuda) else None
        return nocopy_forward(query, key, value, run, ws=ws, groups=groups, pg=pg, a2a=a2a, send_box=box, comm=comm)

    return _forward_impl


def make_rope(orig, groups: int, strategy_info, launch, supported):  # noqa: ANN001,ANN201
    """Wrap the transformer's ``fused_qk_norm_rope``: while a MiniMaxH3Attention forward is armed and its attention
    would take the nocopy path, write q/k into a send buffer and return placeholders.  ``supported(q, k, rope_table,
    interleaved)`` is the stock dispatch test for the Triton path; anything else runs the stock function."""

    def fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps, *, head_dim=None, rotary_dim=None, interleaved=False):  # noqa: ANN001,ANN202
        owner = getattr(_tls, "armed", None)
        _tls.armed = None  # one redirect per armed forward
        info = None
        if owner is not None and getattr(owner, "to_gate_compress", None) is None:
            info = strategy_info(owner.attention, None)
        if (
            info is None
            or q.ndim != 3
            or q.shape != k.shape
            or q.shape[0] == 0
            or head_dim not in (None, q.shape[-1])
            or rotary_dim not in (None, rope_table.shape[-1])
            or rope_table.shape != (q.shape[0], rope_table.shape[-1])
            or not eligible_shape(q.unsqueeze(0), k.unsqueeze(0), k.unsqueeze(0), info[1], groups)
            or not supported(q, k, rope_table, interleaved)
        ):
            return orig(q, k, q_weight, k_weight, rope_table, eps, head_dim=head_dim, rotary_dim=rotary_dim, interleaved=interleaved)
        _, ws = info
        m, heads, d = q.shape
        send = torch.empty(send_shape(m, heads, d, ws, groups), dtype=q.dtype, device=q.device)
        rope_into_send(send, q, k, q_weight.contiguous(), k_weight.contiguous(), rope_table.contiguous(), eps, ws, groups, launch)
        pend = _Pending(send=send, q_ph=_placeholder(q), k_ph=_placeholder(k), ws=ws, groups=groups)
        _tls.pending = pend
        return pend.q_ph, pend.k_ph

    return fused_qk_norm_rope


def make_armed_forward(orig_fwd):  # noqa: ANN001,ANN201
    """Wrap ``MiniMaxH3Attention.forward`` so the rope wrapper knows which attention its q/k feed."""

    def forward(self, *a, **kw):  # noqa: ANN001,ANN202
        prev = getattr(_tls, "armed", None)
        _tls.armed = self
        try:
            return orig_fwd(self, *a, **kw)
        finally:
            _tls.armed = prev
            _tls.pending = None  # never leak a send buffer past the forward that made it

    return forward


# ---------------------------------------------------------------------------------------------------------------
# vLLM-Omni wiring
# ---------------------------------------------------------------------------------------------------------------


def _vllm_strategy_info(layer, attn_metadata):  # noqa: ANN001,ANN202
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
    from vllm_omni.diffusion.forward_context import get_ulysses_mode

    strategy = layer._get_active_parallel_strategy()
    if not isinstance(strategy, UlyssesParallelAttention) or getattr(layer, "use_ring", False):
        return None
    if get_ulysses_mode(default="strict") != "strict" or (strategy._scatter_idx, strategy._gather_idx) != (2, 1):
        return None
    if strategy._use_sync or getattr(strategy, "_ulysses_a2a_permute", False) or strategy._sp_group.ring_world_size > 1:
        return None
    if getattr(layer, "_scheduler_paged_kv", False) or getattr(layer, "_has_custom_attention", False):
        return None  # keep the stock checks and paths for paged KV and custom attention
    if layer._active_paged_kv_adapter() is not None:
        return None
    if attn_metadata is not None and (
        attn_metadata.attn_mask is not None
        or attn_metadata.joint_query is not None
        or attn_metadata.joint_key is not None
        or attn_metadata.joint_value is not None
        or attn_metadata.joint_attn_mask is not None
        or "gate_compress" in attn_metadata.extra
    ):
        return None
    pg = strategy._ulysses_pg
    ws = dist.get_world_size(pg)
    return (pg, ws) if ws > 1 else None


def _comm_stream():  # noqa: ANN202
    """The overlap module's comm stream (created here with its priority rule if the overlap module has not yet)."""
    from h3_tp2 import ulysses_overlap as ov

    if ov._comm_stream is None:
        prio = int(os.environ.get("H3_TP2_A2A_STREAM_PRIORITY", "0").strip() or 0)
        ov._comm_stream = torch.cuda.Stream(priority=prio)
    return ov._comm_stream


def _triton_launch(x, weight, rope_table, eps, out):  # noqa: ANN001,ANN202
    """``_launch_fused_rms_norm_rope`` of vLLM-Omni a8576ccb7 with the output tensor supplied (strides passed through)."""
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fq

    tokens, heads, head_dim = x.shape
    grid = (tokens, fq.triton.cdiv(heads, fq._HEADS_PER_PROGRAM))
    fq._rms_norm_rope_kernel[grid](
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
        rotary_half=rope_table.shape[-1] // 2,
        eps=eps,
        heads_per_program=fq._HEADS_PER_PROGRAM,
        num_warps=8,
    )


def _triton_supported(q, k, rope_table, interleaved) -> bool:  # noqa: ANN001
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fq

    head_dim, rotary_dim = q.shape[-1], rope_table.shape[-1]
    return (
        not interleaved
        and rope_table.device == q.device
        and rope_table.dtype in (q.dtype, torch.float32)
        and not fq._fused_npu_supported(q, k, head_dim, rotary_dim, False)
        and fq._fused_cuda_supported(q, k, head_dim, rotary_dim, False)
    )


def install(groups: int = 1) -> None:
    from vllm_omni.diffusion.attention import layer as layer_mod
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as tr

    cls = layer_mod.Attention
    if getattr(cls, "_h3_tp2_overlap", 0):
        raise RuntimeError("h3_tp2: the Ulysses overlap is installed; nocopy replaces it (install only one)")
    prev = getattr(cls, "_h3_tp2_nocopy", 0)
    if prev:
        if prev != groups:
            raise RuntimeError(f"h3_tp2 nocopy already installed with groups={prev}, cannot change to {groups}")
        return
    cls._forward_impl = make_forward_impl(cls._forward_impl, groups, _vllm_strategy_info, comm_stream=_comm_stream)
    cls._h3_tp2_nocopy = groups

    # producer redirect: the transformer calls the module-level name it imported
    tr.fused_qk_norm_rope = make_rope(tr.fused_qk_norm_rope, groups, _vllm_strategy_info, _triton_launch, _triton_supported)
    att_cls = tr.MiniMaxH3Attention

    att_cls.forward = make_armed_forward(att_cls.forward)
    logger.info("h3_tp2: Ulysses nocopy exchange installed (groups=%d)", groups)
