"""CPU checks for the combination of the two out-projection levers: ``H3_TP2_ULYSSES_EXACT=2`` (copy 5 left to the
out-projection quant, h3_tp2/ulysses_outq.py) and ``H3_TP2_FP8_QUANT_FUSE`` (the per-token quant fused into the
producers, h3_tp2/fp8_quant_fuse.py).

Rule: in mode 2 the fuse's ``out`` site (the fused unpack + quant of mode 1) is skipped automatically, logged once
(``skipped automatically``, or ``requested but ignored`` when the site was named); qkv / fc1 / fc2 stay fused, and
``=qkv,fc1,fc2`` behaves the same.  Mode 1 + fuse and mode 2 without the fuse are unchanged (their own suites).  A
placeholder of one lever reaching the other lever's quant raises.  Off the GPU both levers run their eager mirrors,
so this checks the wiring and the hand-offs; the kernels are checked by each lever's GPU suite."""
import logging
import types

import pytest

torch = pytest.importorskip("torch")

from h3_tp2 import fp8_quant_fuse as fq  # noqa: E402
from h3_tp2 import ulysses_exact as ux  # noqa: E402
from h3_tp2 import ulysses_outq as oq  # noqa: E402
from tests import test_fp8_quant_fuse as tq  # noqa: E402
from tests.test_fp8_quant_fuse import _clean  # noqa: E402,F401  (autouse: fuse state reset per test)

BF16 = torch.bfloat16
WS = tq.WS


@pytest.fixture(autouse=True)
def _clean_outq():
    def reset():
        oq._state.update(disabled=None, checked=False, sources=None)
        for k in oq._stats:
            oq._stats[k] = 0
        oq._tls.owner = None
        oq._tls.pending = None
        ux._tls.armed = None
        ux._tls.pending = None
    reset()
    yield
    reset()


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):  # noqa: ANN001
        self.lines.append(record.getMessage())


@pytest.fixture
def fuse_log():
    lg = logging.getLogger("vllm_omni.h3_tp2.fp8_quant_fuse")
    h, level = _Lines(), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    yield h.lines
    lg.removeHandler(h)
    lg.setLevel(level)


def _run_packed_defer(self, q, k, v, **kw):  # noqa: ANN001
    """tests.test_fp8_quant_fuse's simulated Ulysses reverse exchange, in mode 2 (``back="defer"``)."""
    o = tq._ORIG_RUN_PACKED(self, q, k, v, **kw)
    m, h, d = o.shape
    o3 = o.view(m, WS, h // WS, d).transpose(0, 1).reshape(WS * m, h // WS, d)
    res = ux.exchange_back(o3.unsqueeze(0), h, WS, lambda recv, inp, group=None: recv.copy_(inp.view_as(recv)),
                           defer=True)
    return res.squeeze(0)


def _load_mode2(quant_cls):
    mod = tq._load(quant_cls, ulysses=False)
    mod.MiniMaxH3Attention._run_packed_attention = _run_packed_defer
    mod.MiniMaxH3Attention.forward = ux.make_armed_forward(mod.MiniMaxH3Attention.forward, outq=True)
    return mod


def _prepare_out(blk):
    """What ulysses_outq.prepare does for an eligible out-projection (its vLLM checks need the real classes)."""
    lin = blk.attn.out_proj
    kernel = lin.quant_method.fp8_linear
    kernel.quant_fp8 = oq._OutQuant(kernel.quant_fp8)
    lin._h3_tp2_outq = True
    return kernel.quant_fp8._orig


# ---------------------------------------------------------------------------------------------------------------
# install: the out site in mode 2
# ---------------------------------------------------------------------------------------------------------------


def test_mode2_skips_out_automatically(fuse_log):
    mod = tq._load(tq._make_quant_cls())
    assert tq._install(mod, fq.parse_sites("1"), ulysses_mode="2", explicit=fq.is_explicit("1")) == \
        ["qkv", "fc1", "fc2"]
    hits = [ln for ln in fuse_log if "site out" in ln]
    assert len(hits) == 1 and "skipped automatically" in hits[0] and "H3_TP2_ULYSSES_EXACT=2" in hits[0]
    assert not any("needs H3_TP2_ULYSSES_EXACT=1" in ln for ln in fuse_log)
    assert ux._unpack_hook["fn"] is None                       # the unpack hook is not installed in mode 2


def test_mode2_named_sites_without_out_identical(fuse_log):
    mod = tq._load(tq._make_quant_cls())
    v = "qkv,fc1,fc2"
    assert tq._install(mod, fq.parse_sites(v), ulysses_mode="2", explicit=fq.is_explicit(v)) == ["qkv", "fc1", "fc2"]
    assert not any("site out" in ln for ln in fuse_log)
    assert ux._unpack_hook["fn"] is None


@pytest.mark.parametrize("value, left", [("out,qkv", ["qkv"]), ("qkv,out,fc1,fc2", ["qkv", "fc1", "fc2"]),
                                         ("out", [])])
def test_mode2_explicit_out_is_ignored(fuse_log, value, left):
    mod = tq._load(tq._make_quant_cls())
    assert fq.is_explicit(value)
    assert tq._install(mod, fq.parse_sites(value), ulysses_mode="2", explicit=True) == left
    hits = [ln for ln in fuse_log if "site out" in ln]
    assert len(hits) == 1 and "requested but ignored" in hits[0]
    assert ux._unpack_hook["fn"] is None


def test_mode1_keeps_out():
    mod = tq._load(tq._make_quant_cls())
    assert tq._install(mod, fq.SITES, ulysses_mode="1") == list(fq.SITES)
    assert ux._unpack_hook["fn"] is fq.unpack_hook


def test_is_explicit():
    assert not any(fq.is_explicit(v) for v in ("1", "all", "ON", " true ", "", "0", None))
    assert fq.is_explicit("qkv,fc1") and fq.is_explicit("out")


# ---------------------------------------------------------------------------------------------------------------
# the whole block: mode 2 + fuse (qkv, fc1, fc2) bit-identical to the stock path
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [37, 130])
def test_block_mode2_with_fuse_bit_identical(rows):
    ref = tq._reference(rows=rows)
    mod = _load_mode2(tq._make_quant_cls())
    assert tq._install(mod, fq.SITES, ulysses_mode="2") == ["qkv", "fc1", "fc2"]
    blk, inp = tq._block(mod, rows=rows)
    out_quant = _prepare_out(blk)
    for call in (1, 2):
        assert tq._same(blk(**inp), ref)
        s, o = fq.stats(), oq.stats()
        assert (s["fused_qkv"], s["fused_out"], s["fused_fc1"], s["fused_fc2"]) == (call, 0, call, call)
        assert s["consumed"] == 3 * call and s["materialize"] == 0 and s["rerun"] == 0 and s["selfcheck_fail"] == 0
        assert o["deferred"] == call and o["selfcheck_pass"] == 1 and o["selfcheck_fail"] == 0
        assert o["quant_recv"] == call - 1                     # the first deferral is the self-check (stock result)
    assert out_quant.calls == 1                                # the out quant ran once: copy 5 + quant in the check
    assert fq._tls.ctx is None and getattr(oq._tls, "pending", None) is None


def test_block_mode2_outq_disabled_falls_back_to_copy5():
    """outq's self-check fails: copy 5 + the stock out quant; the fuse's out site does not take over."""
    ref = tq._reference()
    mod = _load_mode2(tq._make_quant_cls())
    assert tq._install(mod, fq.SITES, ulysses_mode="2") == ["qkv", "fc1", "fc2"]
    blk, inp = tq._block(mod)
    _prepare_out(blk)
    oq._state["disabled"] = "test"
    assert tq._same(blk(**inp), ref)
    assert fq.stats()["fused_out"] == 0 and oq.stats()["deferred"] == 0 and fq.stats()["consumed"] == 3


# ---------------------------------------------------------------------------------------------------------------
# the placeholders never cross
# ---------------------------------------------------------------------------------------------------------------


def _deferral(m=6, heads=2, d=128):
    recv = torch.randn(WS, m, heads // WS, d).to(BF16)
    ph = oq._placeholder((1, m, heads, d), recv)
    return recv, ph, ph.squeeze(0).reshape(m, heads * d).view(-1, heads * d)


def test_outq_placeholder_raises_in_the_fuse_quant():
    mod = tq._load(tq._make_quant_cls())
    tq._install(mod, fq.SITES, ulysses_mode="2")
    quant = mod.quant_cls()
    recv, ph, x = _deferral()
    oq._tls.pending = oq._Deferred(recv, ph)                       # pending deferral: the pointer identifies it
    with pytest.raises(RuntimeError, match="ULYSSES_EXACT=2 placeholder"):
        quant(x, None, None)
    oq._tls.pending = None                                         # consumed or stale: the storage identifies it
    with pytest.raises(RuntimeError, match="ULYSSES_EXACT=2 placeholder"):
        quant(x, None, None)
    assert quant.calls == 0
    real = torch.randn(6, 256).to(BF16)
    assert quant(real, None, None)[0].shape == (6, 256) and quant.calls == 1
    wide = torch.randn(6, 1).to(BF16).expand(6, 256)               # a broadcast view of real data is not one
    assert not oq.is_placeholder(wide) and quant(wide, None, None)[0].shape == (6, 256)


def test_fuse_placeholder_raises_in_the_outq_quant():
    mod = tq._load(tq._make_quant_cls())
    tq._install(mod, fq.SITES, ulysses_mode="2")
    inner = mod.quant_cls()
    outq = oq._OutQuant(inner)
    ph = torch.empty((6, 256), dtype=BF16)
    q, s = fq.quant_torch(torch.randn(6, 256).to(BF16))
    ctx = fq._Ctx(types.SimpleNamespace(), {})
    ctx.pending = fq.Pending("qkv", ph, 256, q, s, inner, None, lambda t: None)
    fq._tls.ctx = ctx
    try:
        with pytest.raises(RuntimeError, match="QUANT_FUSE placeholder"):
            outq(ph, None, None)
        assert ctx.pending is not None and inner.calls == 0    # not consumed, not computed
        got = inner(ph, None, None)                              # its own planned quant still takes it
        assert got[0] is q and ctx.pending is None
    finally:
        fq._tls.ctx = None


def test_exchange_back_defer_never_calls_the_unpack_hook():
    def hook(recv, heads):  # noqa: ANN001,ARG001
        raise AssertionError("unpack hook called on the deferring path")
    ux.set_unpack_hook(hook)
    o = torch.randn(1, WS * 5, 1, 128).to(BF16)

    def a2a(out, inp, group=None):  # noqa: ARG001
        out.view(-1).copy_(inp.reshape(-1))
    res = ux.exchange_back(o, 2, WS, a2a, defer=True)            # unarmed: refused, copy 5
    assert res.is_contiguous() and not oq._is_placeholder(res) and res.shape == (1, 5, 2, 128)
    with pytest.raises(AssertionError, match="unpack hook"):
        ux.exchange_back(o, 2, WS, a2a)                          # mode 1 path: the hook is the producer
