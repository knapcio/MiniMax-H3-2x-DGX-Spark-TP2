"""CPU checks for H3_TP2_FP8_QUANT_FUSE (speed plan item 6): the per-token FP8 quant fused into the producers of the
qkv, out, fc1 and fc2 GEMM inputs (h3_tp2/fp8_quant_fuse.py): the plumbing, not the kernels.

Upstream's verbatim DiT block, attention and MLP forwards (tests/fake_h3_consumers.py) run with FP8 linears whose
forward quantises through a ``QuantFP8``-like module, as vLLM's ``apply_weights`` does (``quant_fp8(x_2d, None,
None)``), with and without the fusion, and every comparison is ``torch.equal`` on the block output's bf16 bits.  Off
the GPU the fused producers are the stock producer followed by the eager quant, so these tests check what the GPU
cannot show in isolation: which producer calls are taken (the running block's, nothing else), that the layer's quant
returns the stash and nothing else, the placeholder hand-off through the upstream forwards, the Ulysses reverse
unpack hook, and every fallback (self-check mismatch, a quant of another tensor, a layer that never quantises, VSA,
LoRA, static or padded quant, an unplanned block).  The kernels are checked bit for bit on the GPU:
``tests/test_fp8_quant_fuse_gpu.py``, ``scripts/fp8_quant_fuse_probe.py`` and the in-process self-check.
"""
import types

import pytest

torch = pytest.importorskip("torch")

from h3_tp2 import fp8_consumers as fc  # noqa: E402
from h3_tp2 import fp8_gemm, ulysses_exact  # noqa: E402
from h3_tp2 import fp8_quant_fuse as fq  # noqa: E402
from tests import fake_h3_consumers as fake  # noqa: E402

BF16 = torch.bfloat16
WS = 2   # simulated Ulysses world size for the out site


def _make_quant_cls():
    """A fresh QuantFP8 stand-in per test (install patches the class's forward)."""
    class FakeQuantFP8(torch.nn.Module):
        def __init__(self, static=False, per_token=True, padding=None):
            super().__init__()
            self.static = static
            self.use_per_token_if_dynamic = per_token
            self.is_group_quant = False
            self.num_token_padding = padding
            self.calls = 0

        def forward(self, x, scale=None, scale_ub=None, use_triton=False):  # noqa: ANN001
            assert scale is None and scale_ub is None
            self.calls += 1
            return fq.quant_torch(x)
    return FakeQuantFP8


def _qforward(self, x):  # noqa: ANN001
    """FakeFP8Linear through a QuantFP8 module, the way vLLM's FP8ScaledMMLinearKernel.apply_weights does it."""
    lead = x.shape[:-1]
    x2 = x.view(-1, x.shape[-1])
    a, s = self.quant_method.fp8_linear.quant_fp8(x2, None, None)
    out = fake.dispatch(a, self.weight, s, self.weight_scale, BF16, self.bias)
    n = self.output_size_per_partition
    if out.shape[1] != n:
        out = out[:, :n].contiguous()
    return out.view(*lead, n), None


_ORIG_RUN_PACKED = fake.MiniMaxH3Attention._run_packed_attention


def _run_packed_ulysses(self, q, k, v, **kw):  # noqa: ANN001
    """The fake attention, then a simulated Ulysses reverse exchange through ulysses_exact.exchange_back (WS ranks of
    which this process plays all: the a2a hands each source block straight to the receive buffer)."""
    o = _ORIG_RUN_PACKED(self, q, k, v, **kw)                                  # [M, H, d]
    m, h, d = o.shape
    o3 = o.view(m, WS, h // WS, d).transpose(0, 1).reshape(WS * m, h // WS, d)   # as the attention output would be
    res = ulysses_exact.exchange_back(o3.unsqueeze(0), h, WS, lambda recv, inp, group=None: recv.copy_(inp.view_as(recv)))
    return res.squeeze(0)


def _load(quant_cls, *, ulysses=True):
    mod = fake.load()
    mod.FakeFP8Linear.forward = _qforward
    if ulysses:
        mod.MiniMaxH3Attention._run_packed_attention = _run_packed_ulysses
    mod.quant_cls = quant_cls
    return mod


def _block(mod, *, rows=37, hidden=200, heads=2, ffn=96, seed=0, quant_kw=None):
    blk = mod.MiniMaxH3DiTBlock(hidden, heads, ffn, seed=seed)
    for m in blk.modules():
        if isinstance(m, mod.FakeFP8Linear):
            m.quant_method = types.SimpleNamespace(fp8_linear=types.SimpleNamespace(
                quant_fp8=mod.quant_cls(**(quant_kw or {}))))
    return blk, mod.block_inputs(blk, rows)


@pytest.fixture(autouse=True)
def _clean():
    fq._state.update(cpu_mirror=True, checked={}, disabled=set(), sites=(), verify=False)
    for k in fq._stats:
        fq._stats[k] = 0
    fq._tls.ctx = None
    gstate = dict(fp8_gemm._state)
    fake.CALLS.clear()
    yield
    fq._tls.ctx = None
    fq._state.update(cpu_mirror=False, checked={}, disabled=set(), sites=(), verify=False, orig_quant=None,
                     quant_cls=None, act_cls=None)
    ulysses_exact.set_unpack_hook(None)
    fp8_gemm._state.clear()
    fp8_gemm._state.update(gstate)


def _same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _install(mod, sites=fq.SITES, **kw):
    return fq.install(sites, tr=mod, quant_cls=mod.quant_cls, act_cls=mod.SiluAndMul, check_source=False,
                      lora_types=kw.pop("lora_types", ()), ulysses_mode=kw.pop("ulysses_mode", "1"), **kw)


def _reference(rows=37, seed=0, **kw):
    qcls = _make_quant_cls()
    mod = _load(qcls)
    blk, inp = _block(mod, rows=rows, seed=seed, **kw)
    return blk(**inp)


@pytest.mark.parametrize("rows", [1, 37, 130])
def test_block_all_sites_bit_identical(rows):
    ref = _reference(rows=rows)
    qcls = _make_quant_cls()
    mod = _load(qcls)
    assert _install(mod) == list(fq.SITES)
    blk, inp = _block(mod, rows=rows)
    out = blk(**inp)
    assert _same(out, ref)
    s = fq.stats()
    assert (s["fused_qkv"], s["fused_out"], s["fused_fc1"], s["fused_fc2"]) == (1, 1, 1, 1)
    assert s["consumed"] == 4 and s["materialize"] == 0 and s["rerun"] == 0
    assert s["selfcheck_pass"] == 4 and s["selfcheck_fail"] == 0
    # the four linears' own quant never ran on the full tensors (only on the self-check rows)
    for lin in (blk.attn.qkv_proj, blk.attn.out_proj, blk.mlp.fc1, blk.mlp.fc2):
        assert lin.quant_method.fp8_linear.quant_fp8.calls == 1          # the self-check's stock pair
    assert fq._tls.ctx is None
    out2 = blk(**inp)                                                     # second call: no self-check any more
    assert _same(out2, ref) and fq.stats()["consumed"] == 8
    for lin in (blk.attn.qkv_proj, blk.attn.out_proj, blk.mlp.fc1, blk.mlp.fc2):
        assert lin.quant_method.fp8_linear.quant_fp8.calls == 1


@pytest.mark.parametrize("sites", [("qkv",), ("out",), ("fc1",), ("fc2",), ("qkv", "fc2"), ("out", "fc1")])
def test_site_subsets(sites):
    ref = _reference()
    mod = _load(_make_quant_cls())
    assert _install(mod, sites) == list(sites)
    blk, inp = _block(mod)
    assert _same(blk(**inp), ref)
    s = fq.stats()
    assert s["consumed"] == len(sites) and all(s["fused_" + x] == (x in sites) for x in fq.SITES)


def test_selfcheck_mismatch_disables_site(monkeypatch):
    ref = _reference()
    mod = _load(_make_quant_cls())
    _install(mod)
    real = fq.rms_quant

    def off_by_one(*a, **k):  # noqa: ANN002,ANN003,ANN202
        q, s = real(*a, **k)
        q = q.clone()
        q.view(torch.uint8)[0, 3] ^= 1
        return q, s
    monkeypatch.setattr(fq, "rms_quant", off_by_one)
    blk, inp = _block(mod)
    assert _same(blk(**inp), ref)
    s = fq.stats()
    assert s["selfcheck_fail"] == 1 and s["fused_qkv"] == 0 and "qkv" in fq._state["disabled"]
    assert (s["fused_out"], s["fused_fc1"], s["fused_fc2"]) == (1, 1, 1)
    assert _same(blk(**inp), ref) and fq.stats()["selfcheck_fail"] == 1   # disabled for the process: not re-checked


def test_layer_that_never_quantises_is_rerun():
    """A planned layer whose forward does not call its QuantFP8 (e.g. a dequant path) must not keep the garbage."""
    ref = _reference()
    mod = _load(_make_quant_cls())
    _install(mod)
    blk, inp = _block(mod)

    def dequant_forward(x):  # noqa: ANN001,ANN202
        lin = blk.mlp.fc1
        a, s = fq.quant_torch(x.view(-1, x.shape[-1]))                 # same numbers, but not through the hook
        out = fake.dispatch(a, lin.weight, s, lin.weight_scale, BF16, lin.bias)
        return out.view(*x.shape[:-1], lin.output_size_per_partition), None
    blk.mlp.fc1.forward = dequant_forward
    assert _same(blk(**inp), ref)
    s = fq.stats()
    assert s["rerun"] == 1 and s["materialize"] == 1 and s["consumed"] == 3


def test_quant_of_another_tensor_materialises():
    """A QuantFP8 call on something else while a stash is pending: the placeholder is made real first."""
    ref = _reference()
    mod = _load(_make_quant_cls())
    _install(mod)
    blk, inp = _block(mod)
    other = mod.quant_cls()
    orig_attn = mod.MiniMaxH3Attention.forward

    def attn_forward(self, x, **kw):  # noqa: ANN001,ANN202
        other(torch.ones(3, 8, dtype=BF16), None, None)             # an unrelated quant before qkv's
        return orig_attn(self, x, **kw)
    mod.MiniMaxH3Attention.forward = attn_forward
    assert _same(blk(**inp), ref)
    s = fq.stats()
    assert s["materialize"] == 1 and s["consumed"] == 3 and s["fused_qkv"] == 1


def test_quant_with_a_static_scale_is_not_taken():
    ctx = fq._Ctx(None, {})
    ph = torch.zeros(4, 16, dtype=BF16)
    qobj = _make_quant_cls()()
    fq._state["orig_quant"] = type(qobj).forward
    fwd = fq.make_quant_forward(type(qobj).forward)
    filled = []
    ctx.pending = fq.Pending("qkv", ph, 16, torch.zeros(4, 16, dtype=torch.float8_e4m3fn), torch.ones(4, 1), qobj,
                             None, lambda t: filled.append(1))
    fq._tls.ctx = ctx
    with pytest.raises(AssertionError):            # the fake QuantFP8 asserts no static scale: it was called stock
        fwd(qobj, ph, torch.ones(1), None)
    assert filled == [1] and ctx.pending is None   # materialised before the stock quant ran
    ctx.pending = fq.Pending("qkv", ph, 16, torch.zeros(4, 16, dtype=torch.float8_e4m3fn), torch.ones(4, 1), qobj,
                             None, lambda t: filled.append(2))
    other = _make_quant_cls()()
    fwd(other, ph, None, None)                     # another instance on the same placeholder: stock + materialise
    assert filled == [1, 2]
    fq._tls.ctx = None


def test_placeholder_match_is_exact():
    ph = torch.empty(1, 6, 4, 8, dtype=BF16)     # the out site's [1, M, H, d]
    q = object()
    p = fq.Pending("out", ph, 32, None, None, q, None, None)
    x = ph.squeeze(0).reshape(6, 32)
    assert p.matches(q, x)
    assert not p.matches(object(), x)
    assert not p.matches(q, x[:5])
    assert not p.matches(q, ph.view(6, 4, 8)[:, :2].reshape(6, 16))
    assert not p.matches(q, x.clone())
    assert not p.matches(q, x.float())


def test_vsa_gate_skips_qkv_site():
    ref = _reference()
    mod = _load(_make_quant_cls())
    _install(mod)
    blk, inp = _block(mod)
    blk.attn.to_gate_compress = None
    plan, _ = fq.block_plan(blk, fq.SITES)
    assert "qkv" in plan
    blk.attn.to_gate_compress = object()           # (not run: only the plan is checked)
    plan, _ = fq.block_plan(blk, fq.SITES)
    assert "qkv" not in plan and {"out", "fc1", "fc2"} <= set(plan)
    blk.attn.to_gate_compress = None
    assert _same(blk(**inp), ref)


@pytest.mark.parametrize("quant_kw,reason", [({"static": True}, "per-token"), ({"per_token": False}, "per-token"),
                                             ({"padding": 16}, "per-token")])
def test_unsuitable_quant_is_not_planned(quant_kw, reason):
    qcls = _make_quant_cls()
    mod = _load(qcls)
    _install(mod)
    blk, inp = _block(mod, quant_kw=quant_kw)
    plan, why = fq.block_plan(blk, fq.SITES)
    assert plan is None and reason in why


def test_lora_and_tp_are_not_planned():
    mod = _load(_make_quant_cls())
    _install(mod)
    blk, _ = _block(mod)
    plan, why = fq.block_plan(blk, fq.SITES, lora_types=(mod.FakeFP8Linear,))
    assert plan is None and "LoRA" in why
    blk.mlp.fc2.tp_size = 2
    plan, why = fq.block_plan(blk, fq.SITES)
    assert plan is None and "tp_size" in why


def test_producers_outside_a_block_stay_stock():
    mod = _load(_make_quant_cls())
    _install(mod)
    blk, inp = _block(mod)
    x = inp["x"]
    w = blk.norm1.weight
    sh = torch.zeros(2, x.shape[1], dtype=BF16)
    out = mod.rms_norm_indexed_scale_shift(x, w, sh, sh, inp["combined_indices"], 1e-6)
    assert out.dtype == BF16 and fq.stats()["fused_qkv"] == 0          # no block context: the stock function
    act = blk.mlp.act_fn(torch.randn(5, 192).to(BF16))
    assert act.shape == (5, 96) and fq.stats()["fused_fc2"] == 0


def test_verify_mode_counts_and_logs(caplog):
    ref = _reference()
    mod = _load(_make_quant_cls())
    _install(mod)
    fq._state["verify"] = True
    blk, inp = _block(mod)
    assert _same(blk(**inp), ref)
    assert fq.stats()["verify_calls"] == 4 and fq.stats()["verify_mismatch"] == 0


def test_install_skips():
    mod = _load(_make_quant_cls())
    assert _install(mod, ("out", "qkv"), ulysses_mode="fwd") == ["qkv"]   # out needs the exact reverse unpack
    mod = _load(_make_quant_cls())
    fc._state["sites"] = ("out",)                                         # tensor32 + consumer-fuse out site
    try:
        assert _install(mod, ("fc1", "fc2")) == ["fc2"]
    finally:
        fc._state["sites"] = ()
    assert fq.parse_sites("1") == fq.SITES and fq.parse_sites("0") == () and fq.parse_sites("") == ()
    assert fq.parse_sites("fc2, qkv") == ("qkv", "fc2") and fq.parse_sites("bogus") == ()


def test_block_wrappers_keep_source_fingerprints(monkeypatch):
    monkeypatch.setitem(fc._state, "cpu_mirror", True)
    monkeypatch.setattr(fp8_gemm, "_stash_hooks", list(fp8_gemm._stash_hooks))
    mod = _load(_make_quant_cls(), ulysses=False)
    before = fc._fingerprints(mod)["block_forward"]
    assert before == fc.EXPECTED["block_forward"]
    fp8_gemm._state["level"] = "tensor32"
    assert fc.install(("fc2",), tr=mod, check_source=False, lora_types=()) == ["fc2"]
    assert fc._fingerprints(mod)["block_forward"] == before                 # consumer fuse wrapper: functools.wraps
    _install(mod)
    assert fc._fingerprints(mod)["block_forward"] == before                 # and ours
    assert getattr(mod.MiniMaxH3DiTBlock.forward, "_h3_tp2_quant_fuse", False)
    assert _install(mod) == list(fq._state["sites"])                       # idempotent


def test_quant_torch_matches_vllm_formula():
    """The eager mirror is QuantFP8.forward_native's arithmetic (CPU: IEEE division; the GPU tests use the real op)."""
    g = torch.Generator().manual_seed(3)
    h = (torch.randn(9, 300, generator=g) * torch.exp(torch.randn(9, 1, generator=g) * 3)).to(BF16)
    h[2] = 0                                              # an all-zero row: the minimum scale
    q, s = fq.quant_torch(h)
    amax = h.abs().amax(-1, keepdim=True).float()
    ref_s = torch.clamp(amax / 448.0, min=1.0 / (448.0 * 512.0))
    assert torch.equal(s, ref_s) and s[2, 0].item() == pytest.approx(1.0 / (448.0 * 512.0))
    ref_q = torch.clamp(h.float() * (1.0 / ref_s), -448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
