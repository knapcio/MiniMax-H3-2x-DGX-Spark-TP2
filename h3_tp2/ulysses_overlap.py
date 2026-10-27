"""Overlap the Ulysses all-to-alls with attention compute (``H3_TP2_A2A_OVERLAP=<groups>``, off by default).

Stock vLLM-Omni Ulysses runs, per attention layer: all-to-all q, k, v (sequence -> heads), the attention kernel over
all local heads, all-to-all back.  The GPU idles during the collectives.  Attention is independent per head, so the
local heads are split into ``groups`` head groups: the forward all-to-all of group g+1 and the backward all-to-all of
group g-1 run on a second CUDA stream while the kernel works on group g.  Every byte still moves, every head is
computed by the same kernel; the arithmetic per head is unchanged but a kernel may pick a different plan or tiling for
fewer heads, so identity is NOT claimed by construction: it is verified by the same-seed frame digest for the
configurations listed in the README and nothing else.  Only the strict Ulysses mode without joint (text) tensors and without Ring is
handled; anything else falls back to the stock forward.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger("vllm_omni.h3_tp2.overlap")

_comm_stream: torch.cuda.Stream | None = None


def _a2a_seq_to_heads(x: torch.Tensor, ws: int, pg, recv: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """(B, S/P, hg, D) view -> (B, S, hg/P, D) into ``res``; the same bytes all_to_all_4D(scatter 2, gather 1) moves.

    ``recv`` and ``res`` are allocated by the caller on the compute stream (see ``_forward_impl``); only the packed
    send buffer lives in the comm stream's pool."""
    bs, s_loc, hg, d = x.shape
    x_t = x.reshape(bs, s_loc, ws, hg // ws, d).transpose(0, 2).contiguous()  # (P, S/P, B, hg/P, D)
    dist.all_to_all_single(recv, x_t, group=pg)
    res.copy_(recv.reshape(s_loc * ws, bs, hg // ws, d).transpose(0, 1))
    return res


def _a2a_heads_to_seq(x: torch.Tensor, ws: int, pg, recv: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """(B, S, hg/P, D) -> (B, S/P, hg, D) into ``res``; the same bytes all_to_all_4D(scatter 1, gather 2) moves."""
    bs, s, shard_hc, d = x.shape
    s_loc = s // ws
    x_t = x.reshape(bs, ws, s_loc, shard_hc, d).transpose(0, 3).transpose(0, 1).contiguous()
    x_t = x_t.reshape(ws, shard_hc, s_loc, bs, d)
    dist.all_to_all_single(recv, x_t, group=pg)
    res.copy_(recv.reshape(shard_hc * ws, s_loc, bs, d).transpose(0, 2))
    return res


def install(groups: int) -> None:
    from vllm_omni.diffusion.attention import layer as layer_mod
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
    from vllm_omni.diffusion.forward_context import get_ulysses_mode

    cls = layer_mod.Attention
    prev = getattr(cls, "_h3_tp2_overlap", 0)
    if prev:
        if prev != groups:
            raise RuntimeError(f"h3_tp2 overlap already installed with groups={prev}, cannot change to {groups}")
        return
    orig = cls._forward_impl

    def _forward_impl(self, query, key, value, attn_metadata=None):  # noqa: ANN001
        global _comm_stream
        strategy = self._get_active_parallel_strategy()
        joint = attn_metadata is not None and attn_metadata.joint_query is not None
        if query.ndim != 4 or not (query.is_cuda and key.is_cuda and value.is_cuda):
            return orig(self, query, key, value, attn_metadata)
        if (
            not isinstance(strategy, UlyssesParallelAttention)
            or getattr(self, "use_ring", False)
            or joint
            or get_ulysses_mode(default="strict") != "strict"
        ):
            return orig(self, query, key, value, attn_metadata)
        pg = strategy._ulysses_pg
        ws = dist.get_world_size(pg)
        heads = query.shape[2]
        if ws == 1 or heads % (groups * ws) != 0 or key.shape[2] != heads or value.shape[2] != heads:
            return orig(self, query, key, value, attn_metadata)
        if attn_metadata is not None and (
            attn_metadata.attn_mask is not None
            or attn_metadata.joint_attn_mask is not None
            or attn_metadata.joint_key is not None
            or attn_metadata.joint_value is not None
        ):
            return orig(self, query, key, value, attn_metadata)  # masks and joint tensors: stock path only
        if (strategy._scatter_idx, strategy._gather_idx) != (2, 1):
            return orig(self, query, key, value, attn_metadata)

        if _comm_stream is None:
            _comm_stream = torch.cuda.Stream()
        comm = _comm_stream
        cur = torch.cuda.current_stream()
        hg = heads // groups
        attn_metadata = self._with_kv_cache_dtype(attn_metadata)

        # Memory rule: every buffer that one stream writes and the other reads is allocated on the COMPUTE stream's
        # pool and never record_stream'ed.  Reuse is then ordered by the compute stream, which waits for every comm
        # event before this function returns (and the caller frees anything only after that).  record_stream would
        # instead pin each block until the GPU passes the event while the CPU runs tens of layers ahead: the working
        # set grew from one layer to dozens and ran the head out of memory (2026-10-27 10:24).
        comm.wait_stream(cur)
        try:
            return _grouped(self, query, key, value, attn_metadata, pg, ws, hg, cur, comm)
        except Exception:
            cur.wait_stream(comm)  # join the outstanding communication before the caller's cleanup runs
            raise

    def _grouped(self, query, key, value, attn_metadata, pg, ws, hg, cur, comm):  # noqa: ANN001
        bs, s_loc, _, d = query.shape
        s = s_loc * ws
        hl = hg // ws  # local heads per group after the exchange
        dt, dev = query.dtype, query.device
        # buffers on the compute stream (current here)
        recv_f = torch.empty((ws, s_loc, bs, hl, d), dtype=dt, device=dev)  # shared scratch for the forward receives
        parts = [tuple(torch.empty((bs, s, hl, d), dtype=dt, device=dev) for _ in range(3)) for _ in range(groups)]
        recv_b = [torch.empty((ws, hl, s_loc, bs, d), dtype=dt, device=dev) for _ in range(groups)]
        backs = [torch.empty((bs, s_loc, hg, d), dtype=dt, device=dev) for _ in range(groups)]

        # 1) forward all-to-alls, all groups, on the comm stream (group g+1 overlaps attention of group g)
        ready = []
        with torch.cuda.stream(comm):
            for g in range(groups):
                sl = slice(g * hg, (g + 1) * hg)
                qg, kg, vg = parts[g]
                _a2a_seq_to_heads(query[:, :, sl], ws, pg, recv_f, qg)
                _a2a_seq_to_heads(key[:, :, sl], ws, pg, recv_f, kg)
                _a2a_seq_to_heads(value[:, :, sl], ws, pg, recv_f, vg)
                ev = torch.cuda.Event()
                ev.record(comm)
                ready.append(ev)

        # 2) attention per group on the compute stream; its backward all-to-all goes to the comm stream right away
        back_ev = []
        outs = []
        for g in range(groups):
            cur.wait_event(ready[g])
            qg, kg, vg = parts[g]
            og = self._run_local_attention(qg, kg, vg, attn_metadata)
            outs.append(og)  # keep it alive until the comm stream is done with it (joined below)
            done = torch.cuda.Event()
            done.record(cur)
            with torch.cuda.stream(comm):
                comm.wait_event(done)
                _a2a_heads_to_seq(og, ws, pg, recv_b[g], backs[g])
                ev = torch.cuda.Event()
                ev.record(comm)
            back_ev.append(ev)

        for ev in back_ev:
            cur.wait_event(ev)
        return torch.cat(backs, dim=2)

    cls._forward_impl = _forward_impl
    cls._h3_tp2_overlap = groups
    logger.info("h3_tp2: Ulysses all-to-all overlap installed (groups=%d)", groups)
