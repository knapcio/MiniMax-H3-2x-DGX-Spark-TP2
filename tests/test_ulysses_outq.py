"""CPU checks for h3_tp2.ulysses_outq (``H3_TP2_ULYSSES_EXACT=2``): copy 5 of the exact exchange left to the
out-projection's per-token FP8 quant, which reads the reverse all-to-all's receive buffer.

The quant runs here through its eager mirror, which reads the receive buffer with the Triton kernel's own addressing
(``as_strided`` with the kernel's strides); the reference is the stock path: copy 5 (``unpack_rows``, the same copy
``ulysses_exact.exchange_back`` makes) followed by ``QuantFP8.forward_native``'s arithmetic.  Every comparison is
bitwise.  The kernel's GPU arithmetic against the real compiled ``QuantFP8`` is checked in
tests/test_ulysses_outq_gpu.py."""

import types
import unittest

import torch

from h3_tp2 import ulysses_exact as ux
from h3_tp2 import ulysses_outq as oq
from tests.test_ulysses_exact import (D, HEADS, INFO, WS, _Layer, eager_launch, make_inputs, ref_rope, run_ranks, sdpa,
                                      stock_all_to_all_4d, stock_forward)

K = HEADS * D


def _bits(t):
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int32)


def same(a, b):
    return a.shape == b.shape and a.stride() == b.stride() and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def reset():
    oq._state.update(disabled=None, checked=False, sources=None)
    for k in oq._stats:
        oq._stats[k] = 0
    oq._tls.owner = None
    oq._tls.pending = None
    ux._tls.armed = None
    ux._tls.pending = None


def directed_rows(m, seed):
    """[m, K] bf16 with zero, tiny, huge, mixed-sign, -0.0, inf and nan rows mixed into attention-like values."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, K, generator=g) * torch.rand(m, 1, generator=g) * 3).to(torch.bfloat16)
    x[0] = 0
    x[1] = -0.0
    x[2] = torch.randn(K, generator=g) * 1e-30
    x[3] = torch.randn(K, generator=g) * 1e30
    x[4, 17] = float("inf")
    x[5, 4000] = float("nan")
    x[6] = torch.finfo(torch.bfloat16).tiny
    x[7, 3583] = 500.0   # the row max at the last column of segment 0 / first of segment 1
    x[8, 3584] = -500.0
    return x


def recv_from_rows(x, ws=WS):
    """The receive buffer [P, M, hl, d] whose copy 5 is x [M, K]."""
    m = x.shape[0]
    return x.view(m, ws, HEADS // ws, D).transpose(0, 1).contiguous()


class FakeQuant:
    """Stand-in for the layer's QuantFP8: forward_native's arithmetic in eager torch; records its calls."""

    def __init__(self):
        self.calls = []
        self.static = False

    def __call__(self, x, scale=None, scale_ub=None):
        self.calls.append((tuple(x.shape), x.is_contiguous(), scale, scale_ub))
        return oq.quant_reference_eager(x)


def fake_owner(m_heads=HEADS):
    """A MiniMaxH3Attention stand-in whose out_proj is already prepared (prepare() reads the cached flag)."""
    quant = FakeQuant()
    kernel = types.SimpleNamespace(quant_fp8=oq._OutQuant(quant))
    layer = types.SimpleNamespace(_h3_tp2_outq=True, quant_method=types.SimpleNamespace(fp8_linear=kernel))
    return types.SimpleNamespace(out_proj=layer, num_heads=m_heads, head_dim=D, attention=_Layer(),
                                 to_gate_compress=None), kernel, quant


def out_proj_quant(kernel, out, m):
    """MiniMaxH3Attention.forward after the attention (.squeeze(0).reshape(total, H*d)) + apply_weights' view + quant."""
    x = out.squeeze(0).reshape(m, K)
    x_2d = x.view(-1, x.shape[-1])
    return kernel.quant_fp8(x_2d, None, None)


class QuantMirrorTest(unittest.TestCase):
    def test_recv_addressing_equals_copy5_then_quant(self):
        for m in (64, 37):
            for seed in (1, 2):
                with self.subTest(m=m, seed=seed):
                    x = directed_rows(m, seed)
                    recv = recv_from_rows(x)
                    self.assertTrue(torch.equal(oq.unpack_rows(recv).view(torch.int16), x.view(torch.int16)))
                    ref = oq.quant_reference_eager(oq.unpack_rows(recv))
                    got = oq.quant_from_recv(recv)
                    for a, b in zip(ref, got):
                        self.assertTrue(same(a, b))
                    self.assertEqual(got[0].shape, (m, K))
                    self.assertEqual(got[1].shape, (m, 1))

    def test_contiguous_rows_through_the_same_addressing(self):
        x = directed_rows(48, 3)
        ref = oq.quant_reference_eager(x)
        got = oq.quant_rows(x, 48, WS, K // WS, K // WS, K)   # a contiguous row as two segments
        for a, b in zip(ref, got):
            self.assertTrue(same(a, b))

    def test_mirror_equals_vllm_forward_native_on_cpu(self):
        try:
            from vllm.config import VllmConfig, set_current_vllm_config
            from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
            from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
        except Exception:  # noqa: BLE001
            self.skipTest("vLLM not importable here")
        with set_current_vllm_config(VllmConfig()):
            q = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
        x = directed_rows(40, 4)
        for a, b in zip(q.forward_native(x), oq.quant_reference_eager(x)):
            self.assertTrue(same(a, b))

    def test_reviewed_sources(self):
        try:
            from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as tr
        except Exception:  # noqa: BLE001
            self.skipTest("vLLM-Omni not importable here")
        reset()
        self.assertEqual(oq.source_fingerprints(tr.MiniMaxH3Attention.forward), oq.EXPECTED)
        self.assertIsNone(oq.check_sources(tr.MiniMaxH3Attention.forward))


class DeferTest(unittest.TestCase):
    def setUp(self):
        reset()

    tearDown = setUp

    def _two_ranks(self, m, seed, calls=2):
        outs = [torch.randn(1, m * WS, HEADS // WS, D, generator=torch.Generator().manual_seed(seed + r)).to(torch.bfloat16)
                for r in range(WS)]
        ref, _ = run_ranks(lambda r, a2a: oq.quant_reference_eager(stock_all_to_all_4d(outs[r], 1, 2, WS, a2a).reshape(m, K)))

        def ours(r, a2a):
            reset_tls()
            owner, kernel, quant = fake_owner()
            res = []
            for _ in range(calls):
                prev = oq.arm(owner, m)
                out = ux.exchange_back(outs[r], HEADS, WS, a2a, defer=True)
                ph = oq._is_placeholder(out)
                res.append((ph, out_proj_quant(kernel, out, m)))
                oq.disarm(prev, check=True)
            return res, quant.calls

        got, fake = run_ranks(ours)
        return ref, got, fake

    def test_two_ranks_equal_stock_copy_and_quant(self):
        for m in (64, 40):
            with self.subTest(m=m):
                reset()
                ref, got, fake = self._two_ranks(m, 30)
                self.assertEqual(fake.calls, [2, 2])   # one reverse all-to-all per call, as stock
                for r in range(WS):
                    res, qcalls = got[r]
                    for ph, (q, s) in res:
                        self.assertTrue(ph, "the exchange returned a placeholder, not the copy")
                        self.assertTrue(same(q, ref[r][0]) and same(s, ref[r][1]), f"rank {r} differs")
                self.assertTrue(oq._state["checked"] and oq._state["disabled"] is None)
                # the self-check ran stock (copy 5 + the layer's quant) once per process (each rank is its own
                # process in serving; the two thread-ranks here may both start it), on the contiguous [M, K] tensor
                self.assertIn(sum(len(got[r][1]) for r in range(WS)), (1, 2))
                self.assertTrue(all(c == ((m, K), True, None, None) for r in range(WS) for c in got[r][1]))
                self.assertGreaterEqual(oq._stats["quant_recv"], 2)

    def test_selfcheck_mismatch_disables_and_returns_stock(self):
        m = 32
        recv = recv_from_rows(directed_rows(m, 9))
        stock = oq.quant_reference_eager(oq.unpack_rows(recv))
        owner, kernel, quant = fake_owner()
        orig = oq.quant_from_recv
        oq.quant_from_recv = lambda r: (orig(r)[0], orig(r)[1] * 2)   # a wrong scale
        try:
            prev = oq.arm(owner, m)
            ph = oq.defer(recv, HEADS)
            self.assertIsNotNone(ph)
            q, s = out_proj_quant(kernel, ph, m)
            oq.disarm(prev, check=True)
        finally:
            oq.quant_from_recv = orig
        self.assertTrue(same(q, stock[0]) and same(s, stock[1]), "the mismatching call returns stock's result")
        self.assertIn("self-check", oq._state["disabled"])
        self.assertEqual(oq._stats["selfcheck_fail"], 1)
        prev = oq.arm(owner, m)
        self.assertIsNone(oq.defer(recv, HEADS), "disabled: copy 5 is made again")
        oq.disarm(prev, check=True)
        x = oq.unpack_rows(recv)
        q2, s2 = kernel.quant_fp8(x, None, None)    # a real tensor goes to the layer's own quant
        self.assertTrue(same(q2, stock[0]) and same(s2, stock[1]))
        self.assertEqual(quant.calls[-1], ((m, K), True, None, None))

    def test_selfcheck_exception_disables(self):
        m = 16
        recv = recv_from_rows(directed_rows(m, 10))
        owner, kernel, _ = fake_owner()
        orig = oq.quant_from_recv

        def boom(r):
            raise RuntimeError("compile failed")

        oq.quant_from_recv = boom
        try:
            prev = oq.arm(owner, m)
            q, s = out_proj_quant(kernel, oq.defer(recv, HEADS), m)
            oq.disarm(prev, check=True)
        finally:
            oq.quant_from_recv = orig
        stock = oq.quant_reference_eager(oq.unpack_rows(recv))
        self.assertTrue(same(q, stock[0]) and same(s, stock[1]))
        self.assertIn("raised", oq._state["disabled"])

    def test_defer_refuses_unarmed_and_mismatched(self):
        recv = recv_from_rows(directed_rows(16, 11))
        self.assertIsNone(oq.defer(recv, HEADS), "no armed attention")
        owner, _, _ = fake_owner()
        prev = oq.arm(owner, 17)
        self.assertIsNone(oq.defer(recv, HEADS), "rows of the armed forward differ")
        oq.disarm(prev, check=True)
        prev = oq.arm(owner, 16)
        self.assertIsNone(oq.defer(recv.float(), HEADS), "not bf16")
        self.assertIsNone(oq.defer(recv, HEADS + 2), "head count differs")
        oq.disarm(prev, check=True)
        owner.out_proj._h3_tp2_outq = False   # prepare() said no
        prev = oq.arm(owner, 16)
        self.assertIsNone(oq.defer(recv, HEADS))
        oq.disarm(prev, check=True)

    def test_arm_consumed_once(self):
        recv = recv_from_rows(directed_rows(16, 12))
        owner, kernel, _ = fake_owner()
        prev = oq.arm(owner, 16)
        ph = oq.defer(recv, HEADS)
        self.assertIsNone(oq.defer(recv, HEADS), "one deferral per armed forward")
        out_proj_quant(kernel, ph, 16)
        oq.disarm(prev, check=True)

    def test_passthrough_and_guards(self):
        _, kernel, quant = fake_owner()
        x = directed_rows(16, 13)[:8].contiguous()
        q, s = kernel.quant_fp8(x, None, None)
        self.assertEqual(quant.calls, [((8, K), True, None, None)])
        self.assertTrue(same(q, oq.quant_reference_eager(x)[0]))
        self.assertFalse(quant.static)   # attribute access reaches the wrapped QuantFP8
        with self.assertRaises(RuntimeError):   # a placeholder without its receive buffer never reaches a quant
            kernel.quant_fp8(oq._placeholder((8, K), x), None, None)
        owner, kernel, _ = fake_owner()
        recv = recv_from_rows(x)
        prev = oq.arm(owner, 8)
        ph = oq.defer(recv, HEADS)
        with self.assertRaises(RuntimeError):   # the placeholder in another shape
            kernel.quant_fp8(ph.reshape(8 * K)[None].expand(1, 8 * K), None, None)
        with self.assertRaises(RuntimeError):   # never consumed: the forward must not return normally
            oq.disarm(prev, check=True)
        prev = oq.arm(owner, 8)
        oq.defer(recv, HEADS)
        oq.disarm(prev, check=False)            # an exception is already propagating: no second one
        self.assertIsNone(getattr(oq._tls, "pending", None))

    def test_placeholder_survives_the_model_views(self):
        recv = recv_from_rows(directed_rows(24, 14))
        owner, _, _ = fake_owner()
        prev = oq.arm(owner, 24)
        ph = oq.defer(recv, HEADS)
        self.assertEqual(ph.shape, (1, 24, HEADS, D))
        x = ph.squeeze(0).reshape(24, K).view(-1, K)
        self.assertEqual(x.data_ptr(), ph.data_ptr())
        self.assertEqual(x.stride(), (0, 0))
        self.assertTrue(oq._tls.pending.matches(x))
        oq._tls.pending = None
        oq.disarm(prev, check=True)


def reset_tls():
    oq._tls.owner = None
    oq._tls.pending = None
    ux._tls.armed = None
    ux._tls.pending = None


class ModeTwoWiringTest(unittest.TestCase):
    """The whole mode-2 attention on two thread-ranks: redirected rope producer -> exact exchange -> placeholder ->
    the model's views -> out-projection quant, against stock's exchange + copy + quant."""

    def setUp(self):
        reset()

    tearDown = setUp

    def test_mode2_equals_stock(self):
        m = 96
        ins = [make_inputs(m, torch.bfloat16, 80 + r) for r in range(WS)]

        def stock(r, a2a):
            q, k, v, w_q, w_k, table = ins[r]
            q, k = ref_rope(q, w_q, table, 1e-6), ref_rope(k, w_k, table, 1e-6)
            out = stock_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), sdpa, WS, a2a)
            return oq.quant_reference_eager(out.squeeze(0).reshape(m, K))

        def ours(r, a2a):
            reset_tls()
            q, k, v, w_q, w_k, table = ins[r]
            owner, kernel, _ = fake_owner()
            layer = owner.attention
            rope = ux.make_rope(lambda *a, **kw: self.fail("stock rope ran"), INFO, eager_launch, lambda *a: True)
            fwd = ux.make_forward_impl(lambda *a: self.fail("stock forward ran"), "2", INFO, a2a=a2a)

            def body(self_, x):
                qp, kp = rope(q, k, w_q, w_k, table, 1e-6)
                out = fwd(layer, qp.unsqueeze(0), kp.unsqueeze(0), v.unsqueeze(0))
                assert oq._is_placeholder(out)
                return out_proj_quant(kernel, out, m)

            return ux.make_armed_forward(body, outq=True)(owner, torch.empty(m, 8))

        ref, _ = run_ranks(stock)
        got, fake = run_ranks(ours)
        self.assertEqual(fake.calls, [4, 4])
        for r in range(WS):
            self.assertTrue(same(got[r][0], ref[r][0]) and same(got[r][1], ref[r][1]), f"rank {r} differs")
        self.assertEqual(oq._stats["deferred"], 2)

    def test_mode2_unconsumed_deferral_raises(self):
        m = 96   # > 64: smaller calls are fp8_consumers' self-check size and are not redirected
        q, k, v, w_q, w_k, table = make_inputs(m, torch.bfloat16, 90)
        owner, _, _ = fake_owner()
        rope = ux.make_rope(lambda *a, **kw: None, INFO, eager_launch, lambda *a: True)

        def a2a(out, inp, group=None):  # noqa: ARG001  (one rank talking to itself: shapes are all that matter here)
            out.view(-1).copy_(inp.reshape(-1))

        fwd = ux.make_forward_impl(lambda *a: None, "2", INFO, a2a=a2a)

        def body(self_, x):
            qp, kp = rope(q, k, w_q, w_k, table, 1e-6)
            return fwd(owner.attention, qp.unsqueeze(0), kp.unsqueeze(0), v.unsqueeze(0))   # nobody quantises

        with self.assertRaises(RuntimeError):
            ux.make_armed_forward(body, outq=True)(owner, torch.empty(m, 8))
        self.assertIsNone(getattr(oq._tls, "pending", None))

    def test_mode1_is_unchanged(self):
        out = torch.randn(1, 2 * 16, HEADS // WS, D).to(torch.bfloat16)
        owner, _, _ = fake_owner()
        prev = oq.arm(owner, 16)

        def a2a(o, i, group=None):  # noqa: ARG001
            o.view(-1).copy_(i.reshape(-1))

        res = ux.exchange_back(out, HEADS, WS, a2a)   # defer=False: the copy, even when armed
        self.assertTrue(res.is_contiguous() and not oq._is_placeholder(res))
        oq.disarm(prev, check=False)


if __name__ == "__main__":
    unittest.main()
