"""Ulysses exchange without the removable permute copies, every cuDNN and NCCL operand in stock's layout
(``H3_TP2_ULYSSES_EXACT``, default ``1`` since 2026-11-04, ``0`` = stock; speed plan 2026-11-02 item 5, second attempt
after ``ulysses_nocopy``).

Stock strict Ulysses at batch 1 (vLLM-Omni v0.30.0 ``a8576ccb7``, ``SeqAllToAll4D`` -> ``comm.all_to_all_4D``) per
attention layer and rank, with M local rows, P = 2 ranks, H = 56 heads, hl = H / P, d = 128:

1. q pack ``comm.py:44``: rope output [M, H, d] -> ``input_t`` [P, M, 1, hl, d] (copy, 14336*M bytes);
2. k pack, the same;
3. v pack, the same from the strided v view of the qkv GEMM output;
   (the three receives ``comm.py:57-60`` are views for batch 1: q/k/v reach cuDNN as contiguous [1, P*M, hl, d])
4. reverse pack ``comm.py:74-80``: attention output [1, P*M, hl, d] -> [P, hl, M, 1, d] (copy);
5. reverse unpack ``comm.py:93-96``: receive [P, hl, M, d] -> [1, M, H, d] (copy).

This module keeps the three forward all-to-alls and their buffers exactly as stock has them (``send`` = stock's
``input_t`` [P, M, hl, d], contiguous; ``recv`` viewed [1, P*M, hl, d] = stock's q/k/v, contiguous, same strides), so
cuDNN gets bit-for-bit the operands it gets today, and removes the copies by moving the producers:

* the q/k producer writes straight into ``send_q`` / ``send_k``: one launch of the same kernel per destination block
  (heads p*hl..), output pointer and strides aimed at ``send[p]`` (removes 1 and 2).  Two producers are handled:
  the H3_TP2_FP8_CONSUMER_FUSE qkv site (``fp8_consumers.qk_norm_rope_fused``, the serving default since
  2026-11-03: q/k from the fp32 GEMM product) and, without it, the upstream ``_rms_norm_rope_kernel``;
* with the consumer-fused producer, v's scale pass is also written per destination into ``send_v`` (removes 3); with
  the upstream producer v is packed by stock's copy;
* the attention output goes to the reverse all-to-all as it is (contiguous [P*M, hl, d]: its row blocks are the
  per-destination chunks; the bytes inside a chunk are in row-major instead of head-major order) and the receive is
  unpacked with one copy straight into the [1, M, H, d] tensor the out-projection reads (removes 4; 5 stays: the
  out-projection's row-major input interleaves the sources, which all_to_all_single cannot produce).

Every value is produced by the same kernel arithmetic on the same operands (the norm+RoPE kernels and the scale pass
are independent per (token, head) row / per element) and only moved by copies, so the result is meant to be bit
identical; ``tests/test_ulysses_exact_gpu.py`` checks the producers, the exchange and cuDNN bitwise on the GPU and the
serving digest is the proof (results/20261103-ulysses-exact).  Removed per layer and rank: 3 (upstream producer) or 4
(consumer-fused producer) x 14336*M bytes written and as many read (4 x 614 MB at M = 42848).

Modes (``H3_TP2_ULYSSES_EXACT``): ``1`` = all of the above; ``fwd`` = forward side only, the reverse runs stock's
``all_to_all_4D`` (bisects a digest mismatch); ``verify`` = diagnostic: every call also runs the stock exchange,
attention and reverse on the same q/k/v (producers stay stock), compares the received q, k, v, the attention output
and the result bitwise (logged), and returns the STOCK result.  ``H3_TP2_A2A_OVERLAP`` is not combined with this module (the plugin installs only this one).

Wiring: ``Attention._forward_impl`` is wrapped at install; the producer hooks and the ``MiniMaxH3Attention.forward``
wrapper that tells them which attention their q/k feed are attached on the first call of the wrapped forward (by
then the fp8 sections of the plugin have installed their own wrappers and source checks, which this does not alter).
A producer that writes into send buffers hands the model stride-0 placeholders for q/k (and leaves the model's v view
of the consumer-fused placeholder unwritten); the wrapped attention forward takes the buffers instead.  Anything not
eligible (not strict Ulysses, batch != 1, masks, joint tensors, ring, paged KV, VSA gate, GQA, heads % P) runs the
stock forward; if a redirected layer's attention turns out not eligible, q/k/v are first materialised exactly.
The strategy checks and the upstream rope launcher are those of ``ulysses_nocopy`` (its exchange is a measured
negative, these helpers are not: see the results README).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import torch
import torch.distributed as dist

logger = logging.getLogger("vllm_omni.h3_tp2.exact")

MODES = ("1", "fwd", "verify")
_SELFCHECK_ROWS = 64  # fp8_consumers' self-check runs the producer on this many rows: never redirect those calls

_tls = threading.local()  # armed: (MiniMaxH3Attention, rows) of the running forward; pending: _Pending
_stats = {k: 0 for k in ("redirect_fused", "redirect_rope", "pack", "fallback", "materialize", "out_copy",
                         "verify_equal", "verify_differs")}
_hooked = {"done": False}


def _count(kind: str) -> None:
    _stats[kind] += 1
    if _stats[kind] == 1:
        (logger.warning if kind in ("materialize", "out_copy", "verify_differs") else logger.info)(
            "h3_tp2 ulysses exact: first %s", kind)
    if kind in ("redirect_fused", "redirect_rope", "pack") and \
            (_stats["redirect_fused"] + _stats["redirect_rope"] + _stats["pack"]) % 2000 == 0:
        logger.info("h3_tp2 ulysses exact: calls so far %s", " ".join(f"{k}={v}" for k, v in _stats.items() if v))


def stats() -> dict[str, int]:
    return dict(_stats)


def _a2a(out: torch.Tensor, inp: torch.Tensor, group=None) -> None:  # noqa: ANN001
    # looked up at call time so h3_tp2.a2a_timing (which patches torch.distributed.all_to_all_single) still sees it
    dist.all_to_all_single(out, inp, group=group)


def _stock_a2a_4d(x: torch.Tensor, scatter_idx: int, gather_idx: int, pg) -> torch.Tensor:  # noqa: ANN001
    from vllm_omni.diffusion.distributed.comm import all_to_all_4D

    return all_to_all_4D(x, scatter_idx, gather_idx, group=pg)


# ---------------------------------------------------------------------------------------------------------------
# layout (pure torch; CPU tests)
# ---------------------------------------------------------------------------------------------------------------


def send_shape(m: int, heads: int, d: int, ws: int) -> tuple[int, int, int, int]:
    """Stock's ``input_t`` for batch 1: (P, M, 1, hl, d) without the unit dim."""
    return (ws, m, heads // ws, d)


def by_dest(x: torch.Tensor, ws: int) -> torch.Tensor:
    """[M, H, d] (any strides) -> view [P, M, hl, d]; head h = p*hl + j goes to rank p (``comm.py:44``)."""
    m, h, d = x.shape
    return x.unflatten(1, (ws, h // ws)).transpose(0, 1)


def pack_into(send: torch.Tensor, x: torch.Tensor, ws: int) -> None:
    """Stock's pack copy (``comm.py:44``) into a given buffer: same bytes, same layout."""
    send.copy_(by_dest(x, ws))


def unpack(send: torch.Tensor) -> torch.Tensor:
    """[P, M, hl, d] -> [M, H, d] contiguous (the inverse of ``pack_into``; used to materialise)."""
    p, m, hl, d = send.shape
    return send.transpose(0, 1).reshape(m, p * hl, d)


def exchange_fwd(send: torch.Tensor, a2a, pg=None) -> torch.Tensor:  # noqa: ANN001
    """One stock forward all-to-all; returns the receive buffer viewed [1, P*M, hl, d] (= ``comm.py:57-60``, batch 1)."""
    recv = torch.empty_like(send)
    a2a(recv, send, group=pg)
    p, m, hl, d = recv.shape
    return recv.view(1, p * m, hl, d)


def exchange_back(o: torch.Tensor, heads: int, ws: int, a2a, pg=None) -> torch.Tensor:  # noqa: ANN001
    """Attention output o [1, P*M, hl, d] -> [1, M, H, d] contiguous: one all-to-all of o itself, one unpack copy."""
    o3 = o[0]
    if not o3.is_contiguous():  # cuDNN returns dense BSHD for contiguous BSHD q; anything else costs stock's copy
        _count("out_copy")
        o3 = o3.contiguous()
    s, hl, d = o3.shape
    m = s // ws
    recv = torch.empty((ws, m, hl, d), dtype=o3.dtype, device=o3.device)
    a2a(recv, o3, group=pg)  # chunk p of o3 = the rows of rank p's sequence shard
    result = torch.empty((1, m, heads, d), dtype=o3.dtype, device=o3.device)
    result[0].view(m, ws, hl, d).copy_(recv.transpose(0, 1))
    return result


def exact_forward(query, key, value, run_attn, *, ws, pg=None, a2a=None, sends=None, back="exact",  # noqa: ANN001,ANN201
                  stock_back=None):
    """Strict Ulysses at batch 1.  ``sends``: a list [send_q, send_k] or [send_q, send_k, send_v] filled by a
    redirected producer (popped here so the buffers die after their exchange; q/k/v arguments are then placeholders
    for what the list holds), or None to pack from query/key/value.  ``back``: "exact" or "stock"."""
    a2a = a2a or _a2a
    _, m, heads, d = value.shape
    shape = send_shape(m, heads, d, ws)
    dt, dev = value.dtype, value.device
    if sends:
        send_q, send_k = sends[0], sends[1]
        send_v = sends[2] if len(sends) > 2 else None
        sends.clear()
    else:
        send_q, send_k, send_v = torch.empty(shape, dtype=dt, device=dev), torch.empty(shape, dtype=dt, device=dev), None
        pack_into(send_q, query[0], ws)
        pack_into(send_k, key[0], ws)
    q = exchange_fwd(send_q, a2a, pg)
    del send_q
    k = exchange_fwd(send_k, a2a, pg)
    del send_k
    if send_v is None:
        send_v = torch.empty(shape, dtype=dt, device=dev)
        pack_into(send_v, value[0], ws)
    v = exchange_fwd(send_v, a2a, pg)
    del send_v
    o = run_attn(q, k, v)
    del q, k, v
    if back == "stock":
        return (stock_back or (lambda t: _stock_a2a_4d(t, 1, 2, pg)))(o)
    return exchange_back(o, heads, ws, a2a, pg)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality (NaN patterns included) for same-shape, same-dtype tensors."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.element_size() == 2:
        return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
    return torch.equal(a, b)


def verify_forward(query, key, value, run_attn, *, ws, pg=None, a2a=None, stock_fwd=None, stock_back=None):  # noqa: ANN001,ANN201
    """Both paths on the same real q/k/v [1, M, H, d]; returns (stock result, {tensor: bitwise equal})."""
    a2a = a2a or _a2a
    stock_fwd = stock_fwd or (lambda t: _stock_a2a_4d(t, 2, 1, pg))
    stock_back = stock_back or (lambda t: _stock_a2a_4d(t, 1, 2, pg))
    _, m, heads, d = value.shape
    eq = {}
    sq, sk, sv = stock_fwd(query), stock_fwd(key), stock_fwd(value)
    so = run_attn(sq, sk, sv)
    sres = stock_back(so)
    shape = send_shape(m, heads, d, ws)
    ex = []
    for name, x, ref in (("q", query, sq), ("k", key, sk), ("v", value, sv)):
        send = torch.empty(shape, dtype=x.dtype, device=x.device)
        pack_into(send, x[0], ws)
        r = exchange_fwd(send, a2a, pg)
        eq[name] = r.stride() == ref.stride() and _same(r, ref)
        ex.append(r)
    del sq, sk, sv
    o = run_attn(*ex)
    del ex
    eq["attn"] = _same(o, so)
    res = exchange_back(o, heads, ws, a2a, pg)
    eq["result"] = res.is_contiguous() and _same(res, sres)
    return sres, eq


# ---------------------------------------------------------------------------------------------------------------
# producers: write q/k (and v) into the send buffers
# ---------------------------------------------------------------------------------------------------------------


def rope_into_send(send_q, send_k, q, k, q_weight, k_weight, rope_table, eps, launch) -> None:  # noqa: ANN001
    """Upstream producer: ``launch(x_block, weight, rope_table, eps, out)`` = ``_rms_norm_rope_kernel`` on the heads of
    one destination ([M, hl, d] view of q/k) into ``send[p]`` ([M, hl, d] contiguous)."""
    ws, _, hl, _ = send_q.shape
    for p in range(ws):
        h = slice(p * hl, (p + 1) * hl)
        launch(q[:, h], q_weight, rope_table, eps, send_q[p])
        launch(k[:, h], k_weight, rope_table, eps, send_k[p])


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def fused_into_send(fc, st, heads: int, head_dim: int, q_weight, k_weight, rope_table, eps, sends) -> None:  # noqa: ANN001
    """Consumer-fused producer: ``fc.qk_norm_rope_fused``'s three launches (``fp8_consumers.py:405-441``), one per
    destination block, into send_q / send_k / send_v [P, T, hl, d] instead of out_q / out_k / the placeholder's v
    columns.  Same kernels, launch shapes and constexprs; only the column offset, head count and output view change."""
    x32, sa, sb, bias = st.x32, st.sa, st.sb, st.bias
    t = x32.shape[0]
    ws, _, hl, d = sends[0].shape
    q_size = heads * head_dim
    blk = hl * head_dim
    for slot, (col0, w) in enumerate(((0, q_weight), (q_size, k_weight))):
        for p in range(ws):
            c = col0 + p * blk
            out = sends[slot][p]
            bv = None if bias is None else bias[c:]
            fc._scale_rms_norm_rope_kernel[(t, _cdiv(hl, fc._HEADS_PER_PROGRAM))](
                x32[:, c:], sa, sb[c:], bv if bv is not None else out, w, rope_table, out,
                x32.stride(0), head_dim, 1, rope_table.stride(0), out.stride(0), out.stride(1), out.stride(2),
                num_heads=hl, head_dim=head_dim, rotary_half=rope_table.shape[-1] // 2, eps=eps,
                heads_per_program=fc._HEADS_PER_PROGRAM, HAS_BIAS=bv is not None, num_warps=fc._ROPE_WARPS,
            )
    v0 = 2 * q_size
    for p in range(ws):
        c = v0 + p * blk
        yv = sends[2][p].view(t, blk)
        bv = None if bias is None else bias[c:]
        fc._scale_cols_kernel[(t, _cdiv(blk, fc._VPASS_BLOCK))](
            x32[:, c:], sa, sb[c:], bv if bv is not None else yv, yv, blk, x32.stride(0), yv.stride(0),
            HAS_BIAS=bv is not None, BLOCK=fc._VPASS_BLOCK)


# ---------------------------------------------------------------------------------------------------------------
# placeholders, pending buffers, arming
# ---------------------------------------------------------------------------------------------------------------


def _placeholder(like: torch.Tensor) -> torch.Tensor:
    """A tensor of ``like``'s shape that owns one element (stride 0): nothing may read it."""
    return like.new_empty(1).as_strided(tuple(like.shape), (0,) * like.ndim)


def _is_placeholder(t: torch.Tensor) -> bool:
    return t.numel() > 1 and all(s == 0 for s in t.stride())


@dataclass
class _Pending:
    sends: list  # [send_q, send_k] or [send_q, send_k, send_v], each [P, M, hl, d]
    q_ph: torch.Tensor
    k_ph: torch.Tensor
    ws: int
    v_view: torch.Tensor | None = None  # the model's (unwritten) v view when send_v holds v

    def matches(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
        ok = (query.ndim == 4 and query.data_ptr() == self.q_ph.data_ptr() and key.data_ptr() == self.k_ph.data_ptr()
              and tuple(query.shape[1:]) == tuple(self.q_ph.shape))
        if ok and self.v_view is not None:
            ok = value.ndim == 4 and value.data_ptr() == self.v_view.data_ptr() and \
                tuple(value.shape[1:]) == tuple(self.v_view.shape) and value[0].stride() == self.v_view.stride()
        return ok

    def materialize(self):  # noqa: ANN201
        """Real q, k [1, M, H, d] (and v written into the model's view) for a stock fallback."""
        _count("materialize")
        q, k = unpack(self.sends[0]).unsqueeze(0), unpack(self.sends[1]).unsqueeze(0)
        if self.v_view is not None:
            self.v_view.copy_(unpack(self.sends[2]))
        self.sends = []
        return q, k


def take_pending(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> "_Pending | None":
    pend = getattr(_tls, "pending", None)
    if pend is not None and pend.matches(query, key, value):
        _tls.pending = None
        return pend
    if _is_placeholder(query) or _is_placeholder(key):
        raise RuntimeError("h3_tp2 ulysses exact: placeholder q/k reached attention without their send buffers")
    return None


def _take_arm(rows: int):  # noqa: ANN202
    """The attention whose forward is running, once, if ``rows`` is its row count (not a self-check sub-call)."""
    arm = getattr(_tls, "armed", None)
    if arm is None or rows != arm[1] or rows <= _SELFCHECK_ROWS:
        return None
    _tls.armed = None
    owner = arm[0]
    return owner if getattr(owner, "to_gate_compress", None) is None else None


def eligible_shape(query, key, value, ws: int) -> bool:  # noqa: ANN001
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        return False
    if query.shape[0] != 1 or ws < 2 or query.shape[2] % ws != 0:
        return False
    return key.dtype == query.dtype == value.dtype and key.device == query.device == value.device


def make_armed_forward(orig_fwd):  # noqa: ANN001,ANN201
    """Wrap ``MiniMaxH3Attention.forward(self, x, ...)`` so the producers know which attention their q/k feed."""

    def forward(self, x, *a, **kw):  # noqa: ANN001,ANN202
        prev = getattr(_tls, "armed", None)
        _tls.armed = (self, int(x.shape[0]) if hasattr(x, "shape") and x.ndim >= 1 else -1)
        try:
            return orig_fwd(self, x, *a, **kw)
        finally:
            _tls.armed = prev
            _tls.pending = None  # never leak send buffers past the forward that made them

    forward._h3_tp2_exact_armed = True
    return forward


def make_rope(orig, strategy_info, launch, supported):  # noqa: ANN001,ANN201
    """Wrap the transformer's ``fused_qk_norm_rope`` (upstream producer, used when the consumer-fused qkv site is not
    installed)."""

    def fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps, *, head_dim=None, rotary_dim=None, interleaved=False):  # noqa: ANN001,ANN202
        owner = _take_arm(int(q.shape[0])) if q.ndim == 3 else None
        info = strategy_info(owner.attention, None) if owner is not None else None
        if (
            info is None
            or q.shape != k.shape
            or head_dim not in (None, q.shape[-1])
            or rotary_dim not in (None, rope_table.shape[-1])
            or rope_table.shape != (q.shape[0], rope_table.shape[-1])
            or not eligible_shape(q.unsqueeze(0), k.unsqueeze(0), k.unsqueeze(0), info[1])
            or not supported(q, k, rope_table, interleaved)
        ):
            return orig(q, k, q_weight, k_weight, rope_table, eps, head_dim=head_dim, rotary_dim=rotary_dim, interleaved=interleaved)
        ws = info[1]
        m, heads, d = q.shape
        sends = [torch.empty(send_shape(m, heads, d, ws), dtype=q.dtype, device=q.device) for _ in range(2)]
        rope_into_send(sends[0], sends[1], q, k, q_weight.contiguous(), k_weight.contiguous(), rope_table.contiguous(),
                       eps, launch)
        _tls.pending = _Pending(sends=sends, q_ph=_placeholder(q), k_ph=_placeholder(k), ws=ws)
        _count("redirect_rope")
        return _tls.pending.q_ph, _tls.pending.k_ph

    fused_qk_norm_rope._h3_tp2_exact = True
    return fused_qk_norm_rope


def make_fused_producer(orig, fc, strategy_info):  # noqa: ANN001,ANN201
    """Wrap ``fp8_consumers.qk_norm_rope_fused`` (called by the consumer-fused qkv wrapper after its own checks)."""

    def qk_norm_rope_fused(st, num_heads, num_kv_heads, head_dim, q_weight, k_weight, rope_table, eps):  # noqa: ANN001,ANN202
        t = int(st.x32.shape[0])
        owner = _take_arm(t)
        info = strategy_info(owner.attention, None) if owner is not None else None
        if (
            info is None
            or num_kv_heads != num_heads
            or num_heads % info[1] != 0
            or not fc._use_triton(st.x32)
            or st.ph.dim() != 2
            or st.ph.shape != (t, 3 * num_heads * head_dim)
            or not st.ph.is_contiguous()
            or not st.x32.is_contiguous()
        ):
            return orig(st, num_heads, num_kv_heads, head_dim, q_weight, k_weight, rope_table, eps)
        ws = info[1]
        shape = send_shape(t, num_heads, head_dim, ws)
        sends = [torch.empty(shape, dtype=st.ph.dtype, device=st.ph.device) for _ in range(3)]
        fused_into_send(fc, st, num_heads, head_dim, q_weight.contiguous(), k_weight.contiguous(),
                        rope_table.contiguous(), eps, sends)
        like = st.ph[:, : num_heads * head_dim].view(t, num_heads, head_dim)
        v_view = st.ph[:, 2 * num_heads * head_dim:].view(t, num_kv_heads, head_dim)  # what the model's v view is
        pend = _Pending(sends=sends, q_ph=_placeholder(like), k_ph=_placeholder(like), ws=ws, v_view=v_view)
        _tls.pending = pend
        _count("redirect_fused")
        return pend.q_ph, pend.k_ph, v_view

    qk_norm_rope_fused._h3_tp2_exact = True
    return qk_norm_rope_fused


# ---------------------------------------------------------------------------------------------------------------
# the attention forward
# ---------------------------------------------------------------------------------------------------------------


def make_forward_impl(orig, mode: str, strategy_info, *, a2a=None, stock_fwd=None, stock_back=None, on_first=None):  # noqa: ANN001,ANN201
    """Wrap ``Attention._forward_impl``.  ``strategy_info(layer, attn_metadata)`` -> (pg, ws) when the stock forward
    would run strict Ulysses with nothing this module does not handle, else None.  ``on_first()`` runs once, on the
    first call (attaches the producer hooks)."""
    if mode not in MODES:
        raise ValueError(f"H3_TP2_ULYSSES_EXACT must be one of {MODES}, got {mode!r}")

    def _forward_impl(self, query, key, value, attn_metadata=None):  # noqa: ANN001
        if on_first is not None and not _hooked["done"]:
            _hooked["done"] = True
            on_first()
        pend = take_pending(query, key, value)
        info = strategy_info(self, attn_metadata)
        ok = info is not None and eligible_shape(query, key, value, info[1])  # placeholders carry the real shape
        if pend is not None and (not ok or pend.ws != info[1]):
            query, key = pend.materialize()
            pend = None
        if not ok:
            _count("fallback")
            return orig(self, query, key, value, attn_metadata)
        pg, ws = info
        attn_metadata = self._with_kv_cache_dtype(attn_metadata)

        def run(q, k, v):  # noqa: ANN001,ANN202
            return self._run_local_attention(q, k, v, attn_metadata)

        if mode == "verify":
            res, eq = verify_forward(query, key, value, run, ws=ws, pg=pg, a2a=a2a, stock_fwd=stock_fwd,
                                     stock_back=stock_back)
            kind = "verify_equal" if all(eq.values()) else "verify_differs"
            _count(kind)
            if kind == "verify_differs" or (_stats[kind] % 500 == 0):
                logger.warning("h3_tp2 ulysses exact verify: %s (%d equal, %d differ so far)",
                               " ".join(f"{n}={'eq' if e else 'DIFF'}" for n, e in eq.items()),
                               _stats["verify_equal"], _stats["verify_differs"])
            return res
        sends = None
        if pend is not None:
            sends = pend.sends
            pend.sends = []
        else:
            _count("pack")
        return exact_forward(query, key, value, run, ws=ws, pg=pg, a2a=a2a, sends=sends,
                             back="stock" if mode == "fwd" else "exact", stock_back=stock_back)

    return _forward_impl


# ---------------------------------------------------------------------------------------------------------------
# vLLM-Omni wiring
# ---------------------------------------------------------------------------------------------------------------


def _hook_producers(strategy_info) -> None:  # noqa: ANN001
    """Attach the arming wrapper and the producer hook (called on the first wrapped attention call)."""
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as tr

    from h3_tp2 import ulysses_nocopy as nc

    att_cls = tr.MiniMaxH3Attention
    if not getattr(att_cls.forward, "_h3_tp2_exact_armed", False):
        att_cls.forward = make_armed_forward(att_cls.forward)
    if getattr(tr.fused_qk_norm_rope, "_h3_tp2_consumer_fuse", False):
        from h3_tp2 import fp8_consumers as fc

        if not getattr(fc.qk_norm_rope_fused, "_h3_tp2_exact", False):
            fc.qk_norm_rope_fused = make_fused_producer(fc.qk_norm_rope_fused, fc, strategy_info)
        logger.warning("h3_tp2 ulysses exact: producer = consumer-fused qkv (q, k and v written per destination)")
    elif not getattr(tr.fused_qk_norm_rope, "_h3_tp2_exact", False):
        tr.fused_qk_norm_rope = make_rope(tr.fused_qk_norm_rope, strategy_info, nc._triton_launch, nc._triton_supported)
        logger.warning("h3_tp2 ulysses exact: producer = upstream norm+RoPE (q and k written per destination)")


def install(mode: str = "1") -> None:
    from vllm_omni.diffusion.attention import layer as layer_mod

    from h3_tp2 import ulysses_nocopy as nc

    mode = mode.strip().lower()
    cls = layer_mod.Attention
    if getattr(cls, "_h3_tp2_overlap", 0) or getattr(cls, "_h3_tp2_nocopy", 0):
        raise RuntimeError("h3_tp2: another Ulysses exchange (overlap / nocopy) is installed; exact replaces it")
    prev = getattr(cls, "_h3_tp2_exact", None)
    if prev:
        if prev != mode:
            raise RuntimeError(f"h3_tp2 ulysses exact already installed with mode {prev}, cannot change to {mode}")
        return
    info = nc._vllm_strategy_info
    # verify compares the exchanges on real q/k/v: the producers stay stock there
    hook = None if mode == "verify" else (lambda: _hook_producers(info))
    cls._forward_impl = make_forward_impl(cls._forward_impl, mode, info, on_first=hook)
    cls._h3_tp2_exact = mode
    logger.info("h3_tp2: Ulysses exact exchange installed (mode %s)", mode)
