"""Overlap the Ulysses all-to-alls with attention compute (``H3_TP2_A2A_OVERLAP=<groups>``, off by default).

Stock vLLM-Omni Ulysses runs, per attention layer: all-to-all q, k, v (sequence -> heads), the attention kernel over
all local heads, all-to-all back.  The GPU idles during the collectives.  Attention is independent per head, so the
local heads are split into ``groups`` head groups: the forward all-to-all of group g+1 and the backward all-to-all of
group g-1 run on a second CUDA stream while the kernel works on group g.  Every byte still moves, every head is
computed by the same kernel on the same data: the output is bit-identical to the stock path (verified with the
same-seed frame digest; see README).  Only the strict Ulysses mode without joint (text) tensors and without Ring is
handled; anything else falls back to the stock forward.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger("vllm_omni.h3_tp2.overlap")

_comm_stream: torch.cuda.Stream | None = None


def _a2a_seq_to_heads(x: torch.Tensor, ws: int, pg) -> torch.Tensor:
    """(B, S/P, hg, D) view -> (B, S, hg/P, D), the same bytes all_to_all_4D(scatter 2, gather 1) moves."""
    bs, s_loc, hg, d = x.shape
    x_t = x.reshape(bs, s_loc, ws, hg // ws, d).transpose(0, 2).contiguous()  # (P, S/P, B, hg/P, D)
    out = torch.empty_like(x_t)
    dist.all_to_all_single(out, x_t, group=pg)
    return out.reshape(s_loc * ws, bs, hg // ws, d).transpose(0, 1).contiguous()


def _a2a_heads_to_seq(x: torch.Tensor, ws: int, pg) -> torch.Tensor:
    """(B, S, hg/P, D) -> (B, S/P, hg, D), the same bytes all_to_all_4D(scatter 1, gather 2) moves."""
    bs, s, shard_hc, d = x.shape
    s_loc = s // ws
    x_t = x.reshape(bs, ws, s_loc, shard_hc, d).transpose(0, 3).transpose(0, 1).contiguous()
    x_t = x_t.reshape(ws, shard_hc, s_loc, bs, d)
    out = torch.empty_like(x_t)
    dist.all_to_all_single(out, x_t, group=pg)
    return out.reshape(shard_hc * ws, s_loc, bs, d).transpose(0, 2).contiguous().reshape(bs, s_loc, shard_hc * ws, d)


def install(groups: int) -> None:
    from vllm_omni.diffusion.attention import layer as layer_mod
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
    from vllm_omni.diffusion.forward_context import get_ulysses_mode

    cls = layer_mod.Attention
    if getattr(cls, "_h3_tp2_overlap", False):
        return
    orig = cls._forward_impl

    def _forward_impl(self, query, key, value, attn_metadata=None):  # noqa: ANN001
        global _comm_stream
        strategy = self._get_active_parallel_strategy()
        joint = attn_metadata is not None and attn_metadata.joint_query is not None
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
        if attn_metadata is not None and attn_metadata.attn_mask is not None and attn_metadata.attn_mask.ndim == 2:
            return orig(self, query, key, value, attn_metadata)  # the stock path may rewrite the mask in place

        if _comm_stream is None:
            _comm_stream = torch.cuda.Stream()
        comm = _comm_stream
        cur = torch.cuda.current_stream()
        hg = heads // groups
        attn_metadata = self._with_kv_cache_dtype(attn_metadata)

        # 1) forward all-to-alls, all groups, on the comm stream (group g+1 overlaps attention of group g)
        comm.wait_stream(cur)
        parts, ready = [], []
        with torch.cuda.stream(comm):
            for g in range(groups):
                sl = slice(g * hg, (g + 1) * hg)
                qg = _a2a_seq_to_heads(query[:, :, sl], ws, pg)
                kg = _a2a_seq_to_heads(key[:, :, sl], ws, pg)
                vg = _a2a_seq_to_heads(value[:, :, sl], ws, pg)
                for t in (qg, kg, vg):
                    t.record_stream(cur)
                ev = torch.cuda.Event()
                ev.record(comm)
                parts.append((qg, kg, vg))
                ready.append(ev)

        # 2) attention per group on the compute stream; its backward all-to-all goes to the comm stream right away
        backs, back_ev = [], []
        for g in range(groups):
            cur.wait_event(ready[g])
            qg, kg, vg = parts[g]
            og = self._run_local_attention(qg, kg, vg, attn_metadata)
            og.record_stream(comm)
            done = torch.cuda.Event()
            done.record(cur)
            with torch.cuda.stream(comm):
                comm.wait_event(done)
                bg = _a2a_heads_to_seq(og, ws, pg)
                bg.record_stream(cur)
                ev = torch.cuda.Event()
                ev.record(comm)
            backs.append(bg)
            back_ev.append(ev)

        for ev in back_ev:
            cur.wait_event(ev)
        return torch.cat(backs, dim=2)

    cls._forward_impl = _forward_impl
    cls._h3_tp2_overlap = True
    logger.info("h3_tp2: Ulysses all-to-all overlap installed (groups=%d)", groups)
