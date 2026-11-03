"""CPU checks for h3_tp2.ulysses_exact against the stock strict-Ulysses exchange (two in-process ranks).

``dist.all_to_all_single`` is simulated over two threads with a barrier; like NCCL it needs contiguous buffers and
splits dim 0 equally.  The stock reference is ``all_to_all_4D`` of vLLM-Omni v0.30.0 (a8576ccb7,
``vllm_omni/diffusion/distributed/comm.py:36-98``, Apache-2.0, after DeepSpeed / yunchang) with the collective
injected.  Every comparison is ``torch.equal``.  Unlike ``ulysses_nocopy`` the exchange hands the attention q/k/v with
stock's exact shape AND strides (asserted here); the real kernels (Triton producers, cuDNN) are checked on the GPU in
tests/test_ulysses_exact_gpu.py."""

import threading
import types
import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from h3_tp2 import fp8_gemm
from h3_tp2 import ulysses_exact as ux

WS, HEADS, D = 2, 56, 128
DTYPES = (torch.bfloat16, torch.float32)


class FakeA2A:
    """all_to_all_single over ``ws`` threads: rank r's output = concat over sources of their r-th dim-0 chunk."""

    def __init__(self, ws=WS):
        self.ws = ws
        self.barrier = threading.Barrier(ws, timeout=30)
        self.slots = [None] * ws
        self.calls = [0] * ws
        self.sizes = [[] for _ in range(ws)]

    def bind(self, rank):
        def a2a(out, inp, group=None):  # noqa: ARG001
            assert out.is_contiguous() and inp.is_contiguous(), "NCCL all_to_all_single needs contiguous buffers"
            assert out.numel() == inp.numel() and inp.shape[0] % self.ws == 0 and out.dtype == inp.dtype
            self.calls[rank] += 1
            self.sizes[rank].append(inp.numel())
            self.slots[rank] = inp
            self.barrier.wait()
            out.view(-1).copy_(torch.cat([self.slots[s].reshape(-1).chunk(self.ws)[rank] for s in range(self.ws)]))
            self.barrier.wait()

        return a2a


def run_ranks(fn, ws=WS):
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


def stock_all_to_all_4d(x, scatter_idx, gather_idx, ws, a2a):
    """comm.py:36-98 with the collective injected."""
    if scatter_idx == 2 and gather_idx == 1:
        bs, shard_seqlen, hc, hs = x.shape
        seqlen, shard_hc = shard_seqlen * ws, hc // ws
        input_t = x.reshape(bs, shard_seqlen, ws, shard_hc, hs).transpose(0, 2).contiguous()
        output = torch.empty_like(input_t)
        a2a(output, input_t)
        output = output.reshape(seqlen, bs, shard_hc, hs)
        return output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
    bs, seqlen, shard_hc, hs = x.shape
    hc, shard_seqlen = shard_hc * ws, seqlen // ws
    input_t = (
        x.reshape(bs, ws, shard_seqlen, shard_hc, hs).transpose(0, 3).transpose(0, 1).contiguous()
        .reshape(ws, shard_hc, shard_seqlen, bs, hs)
    )
    output = torch.empty_like(input_t)
    a2a(output, input_t)
    output = output.reshape(hc, shard_seqlen, bs, hs)
    return output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)


def stock_forward(q, k, v, run_attn, ws, a2a):
    qh, kh, vh = (stock_all_to_all_4d(t, 2, 1, ws, a2a) for t in (q, k, v))
    return stock_all_to_all_4d(run_attn(qh, kh, vh), 1, 2, ws, a2a)


def sdpa(q, k, v):
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
    with sdpa_kernel([SDPBackend.MATH]):  # vLLM-Omni's import (GPU tests in the same session) disables math SDPA globally
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
    g = torch.Generator().manual_seed(seed)
    qkv = torch.randn(s_loc, 3 * HEADS * D, generator=g).to(dtype)
    q, k, v = (t.view(s_loc, HEADS, D) for t in qkv.split([HEADS * D] * 3, dim=-1))
    w_q = (1 + 0.1 * torch.randn(D, generator=g)).to(dtype)
    w_k = (1 + 0.1 * torch.randn(D, generator=g)).to(dtype)
    table = torch.randn(s_loc, 96, generator=g)
    return q, k, v, w_q, w_k, table


def stock_back_fn(a2a):
    return lambda o: stock_all_to_all_4d(o, 1, 2, WS, a2a)


class ExchangeTest(unittest.TestCase):
    def test_forward_views_are_stocks_tensors(self):
        for dtype in DTYPES:
            for s_loc in (64, 96):
                with self.subTest(dtype=dtype, s_loc=s_loc):
                    ins = [make_inputs(s_loc, dtype, 10 + r)[:3] for r in range(WS)]

                    def stock(r, a2a):
                        q, k, v = ins[r]
                        return [stock_all_to_all_4d(t.unsqueeze(0), 2, 1, WS, a2a) for t in (q.contiguous(), k.contiguous(), v)]

                    def ours(r, a2a):
                        out = []
                        for t in ins[r]:
                            send = torch.empty(ux.send_shape(s_loc, HEADS, D, WS), dtype=dtype)
                            ux.pack_into(send, t, WS)
                            out.append(ux.exchange_fwd(send, a2a))
                        return out

                    ref, _ = run_ranks(stock)
                    got, fake = run_ranks(ours)
                    self.assertEqual(fake.calls, [3, 3])  # three exchanges, as stock
                    for r in range(WS):
                        for name, a, b in zip("qkv", ref[r], got[r]):
                            self.assertEqual(b.shape, a.shape)
                            self.assertEqual(b.stride(), a.stride(), f"{name}: stock's strides (contiguous)")
                            self.assertTrue(b.is_contiguous())
                            self.assertTrue(torch.equal(a, b), f"rank {r} {name} differs")

    def test_send_buffer_is_stocks_input_t(self):
        q = make_inputs(64, torch.bfloat16, 1)[0]
        send = torch.empty(ux.send_shape(64, HEADS, D, WS), dtype=q.dtype)
        ux.pack_into(send, q, WS)
        input_t = q.unsqueeze(0).reshape(1, 64, WS, HEADS // WS, D).transpose(0, 2).contiguous()  # comm.py:44
        self.assertTrue(torch.equal(send, input_t.view(send.shape)))
        self.assertTrue(torch.equal(ux.unpack(send), q))

    def test_reverse_equal_stock(self):
        for dtype in DTYPES:
            g = torch.Generator().manual_seed(7)
            outs = [torch.randn(1, 96 * WS, HEADS // WS, D, generator=g).to(dtype) for _ in range(WS)]
            ref, _ = run_ranks(lambda r, a2a: stock_all_to_all_4d(outs[r], 1, 2, WS, a2a))
            got, fake = run_ranks(lambda r, a2a: ux.exchange_back(outs[r], HEADS, WS, a2a))
            self.assertEqual(fake.calls, [1, 1])
            for r in range(WS):
                self.assertTrue(got[r].is_contiguous() and got[r].shape == ref[r].shape)
                self.assertTrue(torch.equal(ref[r], got[r]), f"rank {r} differs")

    def test_upstream_producer_fills_send_like_stock_pack(self):
        for dtype in DTYPES:
            q, k, _, w_q, w_k, table = make_inputs(96, dtype, 3)
            sq, sk = (torch.full(ux.send_shape(96, HEADS, D, WS), float("nan"), dtype=dtype) for _ in range(2))
            ux.rope_into_send(sq, sk, q, k, w_q, w_k, table, 1e-6, eager_launch)
            self.assertTrue(torch.equal(sq, ux.by_dest(ref_rope(q, w_q, table, 1e-6), WS)))
            self.assertTrue(torch.equal(sk, ux.by_dest(ref_rope(k, w_k, table, 1e-6), WS)))


class EndToEndTest(unittest.TestCase):
    def _compare(self, dtype, s_loc, producer, back):
        ins = [make_inputs(s_loc, dtype, 20 + r) for r in range(WS)]
        eps = 1e-6

        def stock(r, a2a):
            q, k, v, w_q, w_k, table = ins[r]
            q, k = ref_rope(q, w_q, table, eps), ref_rope(k, w_k, table, eps)
            return stock_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), sdpa, WS, a2a)

        def ours(r, a2a):
            q, k, v, w_q, w_k, table = ins[r]
            sends = None
            if producer == "pack":
                q, k = ref_rope(q, w_q, table, eps), ref_rope(k, w_k, table, eps)
            else:
                sends = [torch.empty(ux.send_shape(s_loc, HEADS, D, WS), dtype=dtype) for _ in range(2)]
                ux.rope_into_send(sends[0], sends[1], q, k, w_q, w_k, table, eps, eager_launch)
                if producer == "qkv":
                    sends.append(torch.empty_like(sends[0]))
                    ux.pack_into(sends[2], v, WS)
                    v = torch.full_like(v, float("nan"))  # the model's v view is not read when send_v holds v
            return ux.exact_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), sdpa, ws=WS, a2a=a2a, sends=sends,
                                    back=back, stock_back=stock_back_fn(a2a))

        ref, sfake = run_ranks(stock)
        got, fake = run_ranks(ours)
        self.assertEqual(fake.calls, [4, 4])
        self.assertEqual(fake.sizes, sfake.sizes)  # same messages as stock, in the same order
        for r in range(WS):
            self.assertEqual(got[r].shape, (1, s_loc, HEADS, D))
            self.assertTrue(got[r].is_contiguous())
            self.assertTrue(torch.equal(ref[r], got[r]), f"rank {r} differs")

    def test_all_producers_and_reverses(self):
        for dtype in DTYPES:
            for producer in ("pack", "qk", "qkv"):
                for back in ("exact", "stock"):
                    with self.subTest(dtype=dtype, producer=producer, back=back):
                        self._compare(dtype, 64, producer, back)

    def test_verify_returns_stock_and_all_equal(self):
        ins = [make_inputs(64, torch.bfloat16, 50 + r) for r in range(WS)]

        def stock(r, a2a):
            q, k, v = ins[r][:3]
            return stock_forward(q.contiguous().unsqueeze(0), k.contiguous().unsqueeze(0), v.unsqueeze(0), sdpa, WS, a2a)

        def ours(r, a2a):
            q, k, v = ins[r][:3]
            return ux.verify_forward(q.contiguous().unsqueeze(0), k.contiguous().unsqueeze(0), v.unsqueeze(0), sdpa,
                                     ws=WS, a2a=a2a, stock_fwd=lambda t: stock_all_to_all_4d(t, 2, 1, WS, a2a),
                                     stock_back=stock_back_fn(a2a))

        ref, _ = run_ranks(stock)
        got, fake = run_ranks(ours)
        self.assertEqual(fake.calls, [8, 8])
        for r in range(WS):
            res, eq = got[r]
            self.assertTrue(torch.equal(res, ref[r]))
            self.assertEqual(eq, {"q": True, "k": True, "v": True, "attn": True, "result": True})

    def test_verify_reports_a_wrong_exchange(self):
        ins = [make_inputs(64, torch.bfloat16, 60 + r) for r in range(WS)]

        def ours(r, a2a):
            q, k, v = ins[r][:3]

            def bad_back(t):  # a reverse that swaps the two head halves: verify must flag it
                out = stock_all_to_all_4d(t, 1, 2, WS, a2a)
                return torch.cat(out.chunk(2, dim=2)[::-1], dim=2)

            return ux.verify_forward(q.contiguous().unsqueeze(0), k.contiguous().unsqueeze(0), v.unsqueeze(0), sdpa,
                                     ws=WS, a2a=a2a, stock_fwd=lambda t: stock_all_to_all_4d(t, 2, 1, WS, a2a),
                                     stock_back=bad_back)

        got, _ = run_ranks(ours)
        for r in range(WS):
            self.assertEqual(got[r][1], {"q": True, "k": True, "v": True, "attn": True, "result": False})


# --- consumer-fused producer: fp8_consumers' kernels replaced by eager stand-ins with the same pointer arithmetic ---


class _FakeKernel:
    def __init__(self, fn):
        self.fn = fn
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(grid)
        return self.fn


def _fake_rope_kernel(acc, sa, sb, bias, w, table, out, sx_t, sx_h, sx_d, s_rope, so_t, so_h, so_d, *, num_heads, head_dim,
                      rotary_half, eps, heads_per_program, HAS_BIAS, num_warps):  # noqa: ARG001
    n = num_heads * head_dim
    x = fp8_gemm.epilogue_torch(acc[:, :n], sa, sb[:n], bias[:n] if HAS_BIAS else None, torch.bfloat16)
    out.copy_(ref_rope(x.view(-1, num_heads, head_dim), w, table, eps))


def _fake_cols_kernel(x, sa, sb, bias, y, ncols, stride_x, stride_y, *, HAS_BIAS, BLOCK):  # noqa: ARG001
    y[:, :ncols].copy_(fp8_gemm.epilogue_torch(x[:, :ncols], sa, sb[:ncols], bias[:ncols] if HAS_BIAS else None, y.dtype))


def fake_fc():
    return types.SimpleNamespace(_scale_rms_norm_rope_kernel=_FakeKernel(_fake_rope_kernel),
                                 _scale_cols_kernel=_FakeKernel(_fake_cols_kernel), _HEADS_PER_PROGRAM=8,
                                 _ROPE_WARPS=8, _VPASS_BLOCK=2048, _use_triton=lambda t: True)


def fused_stash(t, with_bias, seed):
    g = torch.Generator().manual_seed(seed)
    n = 3 * HEADS * D
    st = types.SimpleNamespace(x32=torch.randn(t, n, generator=g) * 30, sa=torch.rand(t, generator=g) * 0.01 + 1e-3,
                               sb=torch.rand(n, generator=g) * 0.01 + 1e-3,
                               bias=torch.randn(n, generator=g) * 0.1 if with_bias else None,
                               ph=torch.empty(t, n, dtype=torch.bfloat16))
    w_q = (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
    w_k = (1 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
    return st, w_q, w_k, torch.randn(t, 96, generator=g)


class FusedProducerTest(unittest.TestCase):
    def test_per_destination_launches_equal_full_launches(self):
        for with_bias in (False, True):
            with self.subTest(bias=with_bias):
                t = 80
                st, w_q, w_k, table = fused_stash(t, with_bias, 5)
                fc = fake_fc()
                # reference: the full-width launches fp8_consumers.qk_norm_rope_fused makes, then stock's packs
                full = torch.empty(t, HEADS, D, dtype=torch.bfloat16)
                refs = []
                for col0, w in ((0, w_q), (HEADS * D, w_k)):
                    _fake_rope_kernel(st.x32[:, col0:], st.sa, st.sb[col0:], None if st.bias is None else st.bias[col0:],
                                      w, table, full, 0, D, 1, 0, 0, 0, 0, num_heads=HEADS, head_dim=D, rotary_half=48,
                                      eps=1e-6, heads_per_program=8, HAS_BIAS=st.bias is not None, num_warps=8)
                    refs.append(ux.by_dest(full.clone(), WS))
                v0 = 2 * HEADS * D
                v = fp8_gemm.epilogue_torch(st.x32[:, v0:], st.sa, st.sb[v0:], None if st.bias is None else st.bias[v0:],
                                            torch.bfloat16).view(t, HEADS, D)
                refs.append(ux.by_dest(v, WS))
                sends = [torch.full(ux.send_shape(t, HEADS, D, WS), float("nan"), dtype=torch.bfloat16) for _ in range(3)]
                ux.fused_into_send(fc, st, HEADS, D, w_q, w_k, table, 1e-6, sends)
                for name, a, b in zip("qkv", refs, sends):
                    self.assertTrue(torch.equal(a, b), f"{name} differs")
                self.assertEqual(fc._scale_rms_norm_rope_kernel.grids, [(t, 4)] * 4)  # 28 heads / 8 per program
                self.assertEqual(fc._scale_cols_kernel.grids, [(t, 2)] * 2)  # 3584 columns / 2048


# --- wiring: producers -> placeholders -> attention wrapper ---


class _Owner:
    def __init__(self, attention):
        self.attention = attention
        self.to_gate_compress = None


class _Layer:
    def _with_kv_cache_dtype(self, md):
        return md

    def _run_local_attention(self, q, k, v, md):  # noqa: ARG002
        return sdpa(q, k, v)


INFO = lambda lay, md: (None, WS)  # noqa: E731


class WiringTest(unittest.TestCase):
    def tearDown(self):
        ux._tls.armed = None
        ux._tls.pending = None

    def test_rope_redirect_two_ranks_equals_stock(self):
        for mode in ("1", "fwd"):
            with self.subTest(mode=mode):
                ins = [make_inputs(96, torch.bfloat16, 40 + r) for r in range(WS)]

                def stock(r, a2a):
                    q, k, v, w_q, w_k, table = ins[r]
                    q, k = ref_rope(q, w_q, table, 1e-6), ref_rope(k, w_k, table, 1e-6)
                    return stock_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), sdpa, WS, a2a)

                def ours(r, a2a):
                    q, k, v, w_q, w_k, table = ins[r]
                    layer = _Layer()
                    rope = ux.make_rope(lambda *a, **kw: self.fail("stock rope ran"), INFO, eager_launch, lambda *a: True)
                    fwd = ux.make_forward_impl(lambda *a: self.fail("stock forward ran"), mode, INFO, a2a=a2a,
                                               stock_back=stock_back_fn(a2a))

                    def body(self_, x):
                        qp, kp = rope(q, k, w_q, w_k, table, 1e-6)
                        assert ux._is_placeholder(qp) and ux._is_placeholder(kp)
                        return fwd(layer, qp.unsqueeze(0), kp.unsqueeze(0), v.unsqueeze(0))

                    return ux.make_armed_forward(body)(_Owner(layer), torch.empty(96, 8))

                ref, _ = run_ranks(stock)
                got, fake = run_ranks(ours)
                self.assertEqual(fake.calls, [4, 4])
                for r in range(WS):
                    self.assertTrue(torch.equal(ref[r], got[r]), f"rank {r} differs")
                self.assertIsNone(getattr(ux._tls, "pending", None))

    def test_fused_producer_two_ranks_equals_stock(self):
        # the consumer-fused producer writes q, k AND v; the model's v view stays unwritten and is not read
        t = 72
        stashes = [fused_stash(t, False, 70 + r) for r in range(WS)]

        def real(r):
            st, w_q, w_k, table = stashes[r]
            x = fp8_gemm.epilogue_torch(st.x32, st.sa, st.sb, None, torch.bfloat16)  # the scale pass
            q, k, v = (c.view(t, HEADS, D) for c in x.split([HEADS * D] * 3, dim=-1))
            return ref_rope(q, w_q, table, 1e-6), ref_rope(k, w_k, table, 1e-6), v

        def stock(r, a2a):
            q, k, v = real(r)
            return stock_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), sdpa, WS, a2a)

        def ours(r, a2a):
            st, w_q, w_k, table = stashes[r]
            layer = _Layer()
            prod = ux.make_fused_producer(lambda *a: self.fail("stock producer ran"), fake_fc(), INFO)
            fwd = ux.make_forward_impl(lambda *a: self.fail("stock forward ran"), "1", INFO, a2a=a2a)

            def body(self_, x):
                qp, kp, v = prod(st, HEADS, HEADS, D, w_q, w_k, table, 1e-6)
                model_v = st.ph[:, 2 * HEADS * D:].view(t, HEADS, D)  # the model's split view of the placeholder
                assert v.data_ptr() == model_v.data_ptr()
                return fwd(layer, qp.unsqueeze(0), kp.unsqueeze(0), model_v.unsqueeze(0))

            return ux.make_armed_forward(body)(_Owner(layer), torch.empty(t, 8))

        ref, _ = run_ranks(stock)
        got, fake = run_ranks(ours)
        self.assertEqual(fake.calls, [4, 4])
        for r in range(WS):
            self.assertTrue(torch.equal(ref[r], got[r]), f"rank {r} differs")

    def test_selfcheck_sized_call_is_not_redirected(self):
        # fp8_consumers' self-check runs the producer on <= 64 rows inside the armed forward: must stay stock and must
        # not consume the arm (the full call that follows is redirected)
        st, w_q, w_k, table = fused_stash(80, False, 9)
        calls = []
        prod = ux.make_fused_producer(lambda st_, *a: calls.append(st_.x32.shape[0]) or ("q", "k", "v"), fake_fc(), INFO)

        def body(self_, x):
            sub = types.SimpleNamespace(x32=st.x32[:64], sa=st.sa[:64], sb=st.sb, bias=None, ph=st.ph[:64].clone())
            self.assertEqual(prod(sub, HEADS, HEADS, D, w_q, w_k, table[:64], 1e-6), ("q", "k", "v"))
            qp, _, _ = prod(st, HEADS, HEADS, D, w_q, w_k, table, 1e-6)
            self.assertTrue(ux._is_placeholder(qp))
            return calls

        self.assertEqual(ux.make_armed_forward(body)(_Owner(_Layer()), torch.empty(80, 8)), [64])

    def test_fallback_materializes_q_k_and_v(self):
        t = 72
        st, w_q, w_k, table = fused_stash(t, False, 11)
        stock_calls = []
        prod = ux.make_fused_producer(lambda *a: self.fail("stock producer ran"), fake_fc(), INFO)
        fwd = ux.make_forward_impl(lambda self_, q, k, v, md=None: stock_calls.append((q, k, v)) or "stock", "1",
                                   lambda lay, md: None)

        def body(self_, x):
            qp, kp, v = prod(st, HEADS, HEADS, D, w_q, w_k, table, 1e-6)
            return fwd(_Layer(), qp.unsqueeze(0), kp.unsqueeze(0), v.unsqueeze(0))

        self.assertEqual(ux.make_armed_forward(body)(_Owner(_Layer()), torch.empty(t, 8)), "stock")
        (q, k, v), = stock_calls
        x = fp8_gemm.epilogue_torch(st.x32, st.sa, st.sb, None, torch.bfloat16)
        rq, rk, rv = (c.view(t, HEADS, D) for c in x.split([HEADS * D] * 3, dim=-1))
        self.assertTrue(torch.equal(q[0], ref_rope(rq, w_q, table, 1e-6)))
        self.assertTrue(torch.equal(k[0], ref_rope(rk, w_k, table, 1e-6)))
        self.assertTrue(torch.equal(v[0], rv), "v written back into the model's view")

    def test_unarmed_and_ineligible_calls_are_stock(self):
        q, k, v, w_q, w_k, table = make_inputs(96, torch.bfloat16, 4)
        rope = ux.make_rope(lambda *a, **kw: "stock", INFO, eager_launch, lambda *a: True)
        self.assertEqual(rope(q, k, w_q, w_k, table, 1e-6), "stock")  # no MiniMaxH3Attention forward running
        owner = _Owner(_Layer())
        owner.to_gate_compress = object()  # VSA gate
        self.assertEqual(ux.make_armed_forward(lambda s, x: rope(q, k, w_q, w_k, table, 1e-6))(owner, torch.empty(96, 8)), "stock")
        rope_odd = ux.make_rope(lambda *a, **kw: "stock", lambda lay, md: (None, 3), eager_launch, lambda *a: True)
        self.assertEqual(ux.make_armed_forward(lambda s, x: rope_odd(q, k, w_q, w_k, table, 1e-6))(_Owner(_Layer()), torch.empty(96, 8)),
                         "stock")  # 56 % 3 != 0
        self.assertFalse(ux.eligible_shape(torch.randn(2, 8, HEADS, D), torch.randn(2, 8, HEADS, D), torch.randn(2, 8, HEADS, D), WS))

    def test_placeholder_without_buffers_raises(self):
        ph = ux._placeholder(torch.randn(64, HEADS, D)).unsqueeze(0)
        with self.assertRaises(RuntimeError):
            ux.take_pending(ph, ph, ph)

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            ux.make_forward_impl(lambda *a: None, "2", INFO)


if __name__ == "__main__":
    unittest.main()
