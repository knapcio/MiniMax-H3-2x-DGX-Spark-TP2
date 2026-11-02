"""CPU checks for h3_tp2.ulysses_nocopy against the stock strict-Ulysses exchange (two in-process ranks).

``dist.all_to_all_single`` is simulated over two threads (one per rank) with a barrier; like NCCL it requires
contiguous buffers.  The stock reference is ``all_to_all_4D`` of vLLM-Omni v0.30.0 (a8576ccb7,
``vllm_omni/diffusion/distributed/comm.py:36-98``, Apache-2.0, after DeepSpeed / yunchang) with the collective
injected.  Every comparison is ``torch.equal`` (bitwise).  What the CPU cannot show: cuDNN's plan for strided q/k/v
and the per-destination launches of the Triton norm+RoPE kernel (see the module docstring)."""

import threading
import unittest

import torch
import torch.nn.functional as F

from h3_tp2 import ulysses_nocopy as nc

WS, HEADS, D = 2, 56, 128
DTYPES = (torch.bfloat16, torch.float32)
S_LOCS = (64, 96)


class FakeA2A:
    """all_to_all_single over ``ws`` threads: rank r's output = concat over sources of their r-th dim-0 chunk."""

    def __init__(self, ws):
        self.ws = ws
        self.barrier = threading.Barrier(ws, timeout=30)
        self.slots = [None] * ws
        self.calls = [0] * ws

    def bind(self, rank):
        def a2a(out, inp, group=None):  # noqa: ARG001
            assert out.is_contiguous() and inp.is_contiguous(), "NCCL all_to_all_single needs contiguous buffers"
            assert out.numel() == inp.numel() and inp.shape[0] % self.ws == 0 and out.shape[0] % self.ws == 0
            assert out.dtype == inp.dtype
            self.calls[rank] += 1
            self.slots[rank] = inp
            self.barrier.wait()
            pieces = [self.slots[src].reshape(-1).chunk(self.ws)[rank] for src in range(self.ws)]
            out.view(-1).copy_(torch.cat(pieces))
            self.barrier.wait()  # every rank has read the slots before they are reused

        return a2a


def run_ranks(fn, ws=WS):
    """fn(rank, a2a) on ws threads; returns the per-rank results, re-raises the first failure."""
    fake = FakeA2A(ws)
    res, err = [None] * ws, [None] * ws

    def body(r):
        try:
            res[r] = fn(r, fake.bind(r))
        except BaseException as e:  # noqa: BLE001
            err[r] = e
            fake.barrier.abort()

    ts = [threading.Thread(target=body, args=(r,)) for r in range(ws)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for e in err:
        if e is not None and not isinstance(e, threading.BrokenBarrierError):
            raise e
    for e in err:
        if e is not None:
            raise e
    return res, fake


# --- stock reference (comm.py:36-98 with the collective injected) ------------------------------------------------


def stock_all_to_all_4d(x, scatter_idx, gather_idx, ws, a2a):
    if scatter_idx == 2 and gather_idx == 1:
        bs, shard_seqlen, hc, hs = x.shape
        seqlen = shard_seqlen * ws
        shard_hc = hc // ws
        input_t = x.reshape(bs, shard_seqlen, ws, shard_hc, hs).transpose(0, 2).contiguous()
        output = torch.empty_like(input_t)
        a2a(output, input_t)
        output = output.reshape(seqlen, bs, shard_hc, hs)
        return output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
    bs, seqlen, shard_hc, hs = x.shape
    hc = shard_hc * ws
    shard_seqlen = seqlen // ws
    input_t = (
        x.reshape(bs, ws, shard_seqlen, shard_hc, hs)
        .transpose(0, 3)
        .transpose(0, 1)
        .contiguous()
        .reshape(ws, shard_hc, shard_seqlen, bs, hs)
    )
    output = torch.empty_like(input_t)
    a2a(output, input_t)
    output = output.reshape(hc, shard_seqlen, bs, hs)
    return output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)


def stock_forward(q, k, v, run_attn, ws, a2a):
    """UlyssesParallelAttention strict pre_attention -> attention -> post_attention."""
    qh, kh, vh = (stock_all_to_all_4d(t, 2, 1, ws, a2a) for t in (q, k, v))
    return stock_all_to_all_4d(run_attn(qh, kh, vh), 1, 2, ws, a2a)


def sdpa(q, k, v):
    """The cuDNN backend's call shape (BSHD -> BHSD permute, SDPA, permute back) on the CPU math kernel."""
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v, scale=D**-0.5).permute(0, 2, 1, 3)


def ref_rope(x, w, table, eps):
    """Eager stand-in for _rms_norm_rope_kernel (H3 half-split, per (token, head) row, fp32 math)."""
    xf = x.float()
    inv = torch.rsqrt((xf * xf).sum(-1, keepdim=True) / x.shape[-1] + eps)
    n = (xf * inv * w.float()).to(torch.bfloat16).float()
    half = table.shape[-1] // 2
    cos, sin = table[:, None, :half].float(), table[:, None, half:].float()
    a, b = n[..., :half], n[..., half : 2 * half]
    return torch.cat((a * cos - b * sin, b * cos + a * sin, n[..., 2 * half :]), dim=-1).to(x.dtype)


def eager_launch(x, w, table, eps, out):
    out.copy_(ref_rope(x, w, table, eps))


def make_inputs(s_loc, dtype, seed):
    """One rank's qkv GEMM output [M, 3*H*d] and the model's q, k, v views of it (v strided, as in production)."""
    g = torch.Generator().manual_seed(seed)
    qkv = torch.randn(s_loc, 3 * HEADS * D, generator=g).to(dtype)
    q, k, v = (t.view(s_loc, HEADS, D) for t in qkv.split([HEADS * D] * 3, dim=-1))
    w_q = (1 + 0.1 * torch.randn(D, generator=g)).to(dtype)
    w_k = (1 + 0.1 * torch.randn(D, generator=g)).to(dtype)
    table = torch.randn(s_loc, 96, generator=g)
    return q, k, v, w_q, w_k, table


# --- tests --------------------------------------------------------------------------------------------------------


class ExchangeLayoutTest(unittest.TestCase):
    def test_forward_views_equal_stock(self):
        for dtype in DTYPES:
            for s_loc in S_LOCS:
                with self.subTest(dtype=dtype, s_loc=s_loc):
                    ins = [make_inputs(s_loc, dtype, 10 + r)[:3] for r in range(WS)]

                    def stock(r, a2a):
                        q, k, v = ins[r]
                        q, k = q.contiguous(), k.contiguous()  # fused_qk_norm_rope returns fresh tensors
                        return [stock_all_to_all_4d(t.unsqueeze(0), 2, 1, WS, a2a) for t in (q, k, v)]

                    def ours(r, a2a):
                        q, k, v = ins[r]
                        send = torch.empty(nc.send_shape(s_loc, HEADS, D, WS, 1), dtype=dtype)
                        nc.pack_qkv(send, q, k, v, WS, 1)
                        recv = torch.empty_like(send)
                        a2a(recv[0], send[0])
                        return list(nc.qkv_views(recv[0]))

                    ref, _ = run_ranks(stock)
                    got, fake = run_ranks(ours)
                    self.assertEqual(fake.calls, [1, 1])  # one exchange for q, k and v
                    hl = HEADS // WS
                    for r in range(WS):
                        for name, a, b in zip("qkv", ref[r], got[r]):
                            self.assertEqual(b.shape, (1, s_loc * WS, hl, D))
                            self.assertEqual(b.stride()[1:], (3 * hl * D, D, 1), "strided packed-QKV view")
                            self.assertTrue(torch.equal(a, b), f"rank {r} {name} differs")

    def test_reverse_equal_stock(self):
        for dtype in DTYPES:
            for s_loc in S_LOCS:
                with self.subTest(dtype=dtype, s_loc=s_loc):
                    g = torch.Generator().manual_seed(7)
                    outs = [torch.randn(1, s_loc * WS, HEADS // WS, D, generator=g).to(dtype) for _ in range(WS)]

                    def stock(r, a2a):
                        return stock_all_to_all_4d(outs[r], 1, 2, WS, a2a)

                    def ours(r, a2a):
                        recv_b = torch.empty(WS, s_loc, HEADS // WS, D, dtype=dtype)
                        result = torch.empty(1, s_loc, HEADS, D, dtype=dtype)
                        nc.back_into(outs[r], recv_b, result, 0, WS, a2a, None)
                        return result

                    ref, _ = run_ranks(stock)
                    got, _ = run_ranks(ours)
                    for r in range(WS):
                        self.assertTrue(got[r].is_contiguous())
                        self.assertTrue(torch.equal(ref[r], got[r]), f"rank {r} output differs")

    def test_producer_redirect_fills_send_like_pack(self):
        for dtype in DTYPES:
            for groups in (1, 2):
                with self.subTest(dtype=dtype, groups=groups):
                    q, k, v, w_q, w_k, table = make_inputs(96, dtype, 3)
                    a = torch.empty(nc.send_shape(96, HEADS, D, WS, groups), dtype=dtype)
                    nc.pack_qkv(a, ref_rope(q, w_q, table, 1e-6), ref_rope(k, w_k, table, 1e-6), v, WS, groups)
                    b = torch.full_like(a, float("nan"))
                    nc.rope_into_send(b, q, k, w_q, w_k, table, 1e-6, WS, groups, eager_launch)
                    nc.fill_slot(b, v, 2, WS, groups)
                    self.assertTrue(torch.equal(a, b))


class EndToEndTest(unittest.TestCase):
    def _compare(self, dtype, s_loc, groups, redirect):
        ins = [make_inputs(s_loc, dtype, 20 + r) for r in range(WS)]
        eps = 1e-6

        def stock(r, a2a):
            q, k, v, w_q, w_k, table = ins[r]
            q, k = ref_rope(q, w_q, table, eps), ref_rope(k, w_k, table, eps)
            return stock_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), sdpa, WS, a2a)

        def ours(r, a2a):
            q, k, v, w_q, w_k, table = ins[r]
            box = None
            if redirect:
                send = torch.empty(nc.send_shape(s_loc, HEADS, D, WS, groups), dtype=dtype)
                nc.rope_into_send(send, q, k, w_q, w_k, table, eps, WS, groups, eager_launch)
                box = [send]
                del send
            else:
                q, k = ref_rope(q, w_q, table, eps), ref_rope(k, w_k, table, eps)
            # sdpa reads the strided q/k/v views directly (the CPU analogue of handing them to cuDNN)
            return nc.nocopy_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), sdpa, ws=WS, groups=groups, a2a=a2a, send_box=box)

        ref, _ = run_ranks(stock)
        got, fake = run_ranks(ours)
        self.assertEqual(fake.calls, [2 * groups] * WS)  # one forward + one reverse exchange per head group
        for r in range(WS):
            self.assertEqual(got[r].shape, (1, s_loc, HEADS, D))
            self.assertTrue(got[r].is_contiguous())
            self.assertTrue(torch.equal(ref[r], got[r]), f"rank {r} differs")

    def test_groups1_pack(self):
        for dtype in DTYPES:
            for s_loc in S_LOCS:
                with self.subTest(dtype=dtype, s_loc=s_loc):
                    self._compare(dtype, s_loc, 1, redirect=False)

    def test_groups1_redirect(self):
        for dtype in DTYPES:
            for s_loc in S_LOCS:
                with self.subTest(dtype=dtype, s_loc=s_loc):
                    self._compare(dtype, s_loc, 1, redirect=True)

    def test_groups2(self):
        # head groups compute each head with the same per-head attention; the final layout must equal stock's
        for dtype in DTYPES:
            for redirect in (False, True):
                with self.subTest(dtype=dtype, redirect=redirect):
                    self._compare(dtype, 64, 2, redirect=redirect)

    def test_groups2_views_equal_grouped_stock(self):
        # per group, the received q/k/v equal stock all_to_all_4D of that group's head slice (what ulysses_overlap does)
        dtype, s_loc, groups = torch.bfloat16, 64, 2
        hg = HEADS // groups
        ins = [make_inputs(s_loc, dtype, 30 + r)[:3] for r in range(WS)]

        def stock(r, a2a):
            return [
                [stock_all_to_all_4d(t[:, g * hg : (g + 1) * hg].unsqueeze(0), 2, 1, WS, a2a) for t in ins[r]]
                for g in range(groups)
            ]

        def ours(r, a2a):
            send = torch.empty(nc.send_shape(s_loc, HEADS, D, WS, groups), dtype=dtype)
            nc.pack_qkv(send, *ins[r], WS, groups)
            recv = torch.empty_like(send)
            for g in range(groups):
                a2a(recv[g], send[g])
            return [list(nc.qkv_views(recv[g])) for g in range(groups)]

        ref, _ = run_ranks(stock)
        got, _ = run_ranks(ours)
        for r in range(WS):
            for g in range(groups):
                for a, b in zip(ref[r][g], got[r][g]):
                    self.assertTrue(torch.equal(a, b), f"rank {r} group {g} differs")


class _Owner:
    """Stand-in for MiniMaxH3Attention: what the rope wrapper reads from it."""

    def __init__(self, attention):
        self.attention = attention
        self.to_gate_compress = None


class _Layer:
    """Stand-in for vLLM-Omni's Attention: the two methods the nocopy forward calls."""

    def _with_kv_cache_dtype(self, md):
        return md

    def _run_local_attention(self, q, k, v, md):  # noqa: ARG002
        return sdpa(q, k, v)


class WrapperTest(unittest.TestCase):
    def _wiring(self, groups, attn_info, stock_calls):
        """make_rope + make_forward_impl as install() wires them; ``attn_info`` decides the attention-time check."""
        layer = _Layer()

        def orig_forward_impl(self_, q, k, v, md=None):  # the stock forward, single-rank stand-in
            stock_calls.append((q, k, v))
            return "stock"

        def orig_rope(q, k, w_q, w_k, table, eps, **kw):  # noqa: ARG001
            return ref_rope(q, w_q, table, eps), ref_rope(k, w_k, table, eps)

        rope = nc.make_rope(orig_rope, groups, lambda lay, md: (None, WS), eager_launch, lambda *a: True)
        fwd = nc.make_forward_impl(orig_forward_impl, groups, attn_info)
        return layer, rope, fwd

    def test_odd_heads_fall_back_to_stock(self):
        # 56 heads, groups 3: 56 % (3 * 2) != 0 -> the rope runs stock and the attention takes the stock forward
        calls = []
        layer, rope, fwd = self._wiring(3, lambda lay, md: (None, WS), calls)
        q, k, v, w_q, w_k, table = make_inputs(64, torch.bfloat16, 1)
        self.assertFalse(nc.eligible_shape(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), WS, 3))
        qr, kr = nc.make_armed_forward(lambda self_: rope(q, k, w_q, w_k, table, 1e-6))(_Owner(layer))
        self.assertFalse(nc._is_placeholder(qr))
        self.assertTrue(torch.equal(qr, ref_rope(q, w_q, table, 1e-6)))
        self.assertEqual(fwd(layer, qr.unsqueeze(0), kr.unsqueeze(0), v.unsqueeze(0)), "stock")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2].data_ptr(), v.data_ptr())

    def test_batch2_and_one_rank_fall_back(self):
        q = torch.randn(2, 8, HEADS, D)
        self.assertFalse(nc.eligible_shape(q, q, q, WS, 1))
        self.assertFalse(nc.eligible_shape(q[:1], q[:1], q[:1], 1, 1))
        self.assertTrue(nc.eligible_shape(q[:1], q[:1], q[:1], WS, 1))
        self.assertTrue(nc.eligible_shape(q[:1], q[:1], q[:1], WS, 2))
        self.assertTrue(nc.eligible_shape(q[:1], q[:1], q[:1], WS, 4))

    def test_placeholders_materialize_when_attention_falls_back(self):
        # rope redirected, but the attention-time check refuses (e.g. a mask appeared): stock gets the real q/k
        calls = []
        layer, rope, fwd = self._wiring(1, lambda lay, md: None, calls)
        q, k, v, w_q, w_k, table = make_inputs(64, torch.bfloat16, 2)

        def body(self_):
            qp, kp = rope(q, k, w_q, w_k, table, 1e-6)
            self.assertTrue(nc._is_placeholder(qp) and nc._is_placeholder(kp))
            return fwd(layer, qp.unsqueeze(0), kp.unsqueeze(0), v.unsqueeze(0))

        self.assertEqual(nc.make_armed_forward(body)(_Owner(layer)), "stock")
        (qs, ks, vs), = calls
        self.assertTrue(torch.equal(qs[0], ref_rope(q, w_q, table, 1e-6)))
        self.assertTrue(torch.equal(ks[0], ref_rope(k, w_k, table, 1e-6)))
        self.assertEqual(vs.data_ptr(), v.data_ptr())
        self.assertIsNone(getattr(nc._tls, "pending", None))

    def test_placeholder_without_send_buffer_raises(self):
        q = torch.randn(64, HEADS, D)
        ph = nc._placeholder(q).unsqueeze(0)
        with self.assertRaises(RuntimeError):
            nc.take_pending(ph, ph)

    def test_unarmed_rope_is_stock(self):
        calls = []
        layer, rope, fwd = self._wiring(1, lambda lay, md: (None, WS), calls)
        q, k, v, w_q, w_k, table = make_inputs(64, torch.bfloat16, 4)
        qr, kr = rope(q, k, w_q, w_k, table, 1e-6)  # no MiniMaxH3Attention forward running
        self.assertFalse(nc._is_placeholder(qr))

    def test_full_wiring_two_ranks_equals_stock(self):
        # rope redirect -> placeholders -> attention wrapper -> one exchange each way, two ranks, bitwise vs stock
        for dtype in DTYPES:
            for groups in (1, 2):
                with self.subTest(dtype=dtype, groups=groups):
                    s_loc = 96 if groups == 1 else 64
                    ins = [make_inputs(s_loc, dtype, 40 + r) for r in range(WS)]
                    layer = _Layer()

                    def stock(r, a2a):
                        q, k, v, w_q, w_k, table = ins[r]
                        q, k = ref_rope(q, w_q, table, 1e-6), ref_rope(k, w_k, table, 1e-6)
                        return stock_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), sdpa, WS, a2a)

                    def ours(r, a2a):
                        q, k, v, w_q, w_k, table = ins[r]
                        info = lambda lay, md: (None, WS)  # noqa: E731
                        rope = nc.make_rope(lambda *a, **kw: self.fail("stock rope ran"), groups, info, eager_launch, lambda *a: True)
                        fwd = nc.make_forward_impl(lambda *a: self.fail("stock forward ran"), groups, info, a2a=a2a)

                        def body(self_):
                            qp, kp = rope(q, k, w_q, w_k, table, 1e-6)
                            return fwd(layer, qp.unsqueeze(0), kp.unsqueeze(0), v.unsqueeze(0))

                        return nc.make_armed_forward(body)(_Owner(layer))

                    ref, _ = run_ranks(stock)
                    got, fake = run_ranks(ours)
                    self.assertEqual(fake.calls, [2 * groups] * WS)
                    for r in range(WS):
                        self.assertTrue(torch.equal(ref[r], got[r]), f"rank {r} differs")


if __name__ == "__main__":
    unittest.main()
