"""CPU checks for H3_TP2_FP8_CONSUMER_FUSE (speed plan item 6b): the scale pass fused into the consumers of the qkv, out
and fc2 GEMMs.

Every comparison is ``torch.equal`` on bf16 bit patterns between
  fused:  fp32 product -> fp8_consumers' fused consumer (eager mirror of the Triton copy, with its pointer arithmetic)
  stock:  fp32 product -> scale pass to bf16 -> the upstream consumer kernel's arithmetic (eager mirror,
          tests/fake_h3_consumers.py)
on small random and directed inputs, odd row counts, a hidden size that is not a power of two, GQA-style q/k head
counts, biases, and the strided v view of the qkv output.  The block-level tests run upstream's verbatim DiT block,
attention and MLP forwards (fake layers) with and without the fusion.  The Triton kernels themselves (fma contraction,
reduction tree) are checked on the GPU: the in-process self-check and scripts/fp8_consumer_probe.py.
"""
import pytest

torch = pytest.importorskip("torch")

from h3_tp2 import fp8_consumers as fc  # noqa: E402
from h3_tp2 import fp8_gemm  # noqa: E402
from tests import fake_h3_consumers as fake  # noqa: E402

BF16 = torch.bfloat16


@pytest.fixture(autouse=True)
def _cpu_mirror():
    fc._state.update(cpu_mirror=True, checked={}, disabled=set())
    for k in fc._stats:
        fc._stats[k] = 0
    fc._tls.ctx = None
    fake.CALLS.clear()
    yield
    fc._tls.ctx = None
    fc._state.update(cpu_mirror=False, checked={}, disabled=set())


def _bits(t):
    return t.contiguous().view(torch.int16)


def _same(a, b):
    """bf16 bit identity; NaNs only need the same positions (payloads differ between kernels)."""
    assert a.shape == b.shape and a.dtype == b.dtype
    na, nb = torch.isnan(a), torch.isnan(b)
    assert torch.equal(na, nb)
    return torch.equal(_bits(a)[~na], _bits(b)[~nb])


def _operands(T, N, seed, *, bias=False, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    x32 = torch.randn(T, N, generator=g) * 3000 * scale        # fp8 x fp8 accumulators are in the thousands
    sa = torch.exp(torch.empty(T).uniform_(-9.0, -2.0, generator=g))
    sb = torch.exp(torch.empty(N).uniform_(-9.5, -7.5, generator=g))
    b = (torch.randn(N, generator=g) * 0.1).to(BF16) if bias else None
    return x32, sa, sb, b


def _stash(x32, sa, sb, b):
    return fc.Stash(x32, sa, sb, b, torch.empty(x32.shape, dtype=BF16))


def _ref_pass(x32, sa, sb, b):
    return fc.scale_pass_into(x32, sa, sb, b, torch.empty(x32.shape, dtype=BF16))


def _rope(T, dtype, seed=3):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(T, 1, generator=g) * 500 * torch.exp(-torch.arange(48) / 8.0)[None, :]
    return torch.cat((torch.cos(pos), torch.sin(pos)), dim=-1).to(dtype)


# ---- the fused consumers against scale pass -> bf16 -> stock consumer --------------------------------------------------


@pytest.mark.parametrize("T", [1, 37])
@pytest.mark.parametrize("H,Hk", [(2, 2), (3, 1)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rope_dtype", [torch.bfloat16, torch.float32])
def test_qk_norm_rope_fused_bit_identical(T, H, Hk, bias, rope_dtype):
    D = 128
    N = (H + 2 * Hk) * D
    x32, sa, sb, b = _operands(T, N, seed=T * 7 + H, bias=bias)
    g = torch.Generator().manual_seed(5)
    wq = (1 + 0.1 * torch.randn(D, generator=g)).to(BF16)
    wk = (1 + 0.1 * torch.randn(D, generator=g)).to(BF16)
    rope = _rope(T, rope_dtype)
    p = _ref_pass(x32, sa, sb, b)                                # stock: the bf16 qkv tensor
    q, k, v_ref = p[:, :H * D].view(T, H, D), p[:, H * D:(H + Hk) * D].view(T, Hk, D), p[:, (H + Hk) * D:].view(T, Hk, D)
    rq, rk = fake.fused_qk_norm_rope(q, k, wq, wk, rope, 1e-6)
    st = _stash(x32, sa, sb, b)
    oq, ok, v = fc.qk_norm_rope_fused(st, H, Hk, D, wq, wk, rope, 1e-6)
    assert _same(oq, rq) and _same(ok, rk)
    # v: the strided view of the qkv tensor the attention reads, same values, same layout, inside the placeholder
    assert _same(v, v_ref) and v.stride() == v_ref.stride() and (T == 1 or v.stride() == (N, D, 1))
    assert v.data_ptr() == st.ph.data_ptr() + (H + Hk) * D * 2


@pytest.mark.parametrize("T,hidden", [(1, 128), (37, 200), (64, 256)])
@pytest.mark.parametrize("bias", [False, True])
def test_gate_rms_fused_bit_identical(T, hidden, bias):
    x32, sa, sb, b = _operands(T, hidden, seed=T + hidden, bias=bias)
    g = torch.Generator().manual_seed(9)
    residual = torch.randn(T, hidden, generator=g).to(BF16)
    mods = torch.randn(3, 6 * hidden, generator=g).to(BF16).chunk(6, dim=-1)   # strided [U, H] views, like AdaLN
    gate, shift, scale = mods[2], mods[3], mods[4]
    weight = (1 + 0.1 * torch.randn(hidden, generator=g)).to(BF16)
    idx = torch.randint(0, 3, (T,), generator=g)
    r_res, r_mod = fake.indexed_gate_rms_norm_scale_shift(residual, gate, _ref_pass(x32, sa, sb, b), weight, shift,
                                                          scale, idx, 1e-6)
    st = _stash(x32, sa, sb, b)
    f_res, f_mod = fc.gate_rms_fused(st, residual, gate, weight, shift, scale, idx, 1e-6)
    assert _same(f_res, r_res) and _same(f_mod, r_mod)
    assert f_res.data_ptr() == st.ph.data_ptr()                 # residual_out reuses the placeholder
    assert f_res.stride() == torch.empty_like(residual).stride()


@pytest.mark.parametrize("T,hidden", [(1, 128), (37, 200)])
@pytest.mark.parametrize("bias", [False, True])
def test_gate_fused_bit_identical(T, hidden, bias):
    x32, sa, sb, b = _operands(T, hidden, seed=3 * T + hidden, bias=bias)
    g = torch.Generator().manual_seed(11)
    x = torch.randn(T, hidden, generator=g).to(BF16)
    gate = torch.randn(2, 6 * hidden, generator=g).to(BF16).chunk(6, dim=-1)[5]
    idx = torch.randint(0, 2, (T,), generator=g)
    ref = fake.indexed_gate(x, gate, _ref_pass(x32, sa, sb, b), idx)
    out = fc.gate_fused(_stash(x32, sa, sb, b), x, gate, idx)
    assert _same(out, ref)


def _directed(T, N):
    vals = torch.tensor([0.0, -0.0, 1e-40, -1e-40, 1.0, -1.0, 3e38, -3e38, 1e4, -1e4, 0.5000001, 1.9999999,
                         float("inf"), -float("inf"), float("nan")])
    x32 = vals.repeat(T, N // vals.numel() + 1)[:, :N].clone()
    return torch.roll(x32, shifts=1, dims=1) if T > 1 else x32


def test_directed_values_all_consumers():
    """Midpoints, subnormals, +-0, overflow to inf after the scales, NaN: still the stock bits."""
    T, H, D = 5, 2, 128
    N = 3 * H * D
    x32 = _directed(T, N)
    x32[1:] = torch.randn(T - 1, N) * 1e3
    x32[2, 7] = float("nan")
    sa = torch.tensor([1.0, 2.0 ** -7, 3.0, 1e-3, 1.0])
    sb = torch.ones(N)
    sb[::3] = 2.0 ** 100                                          # pushes large products to inf
    w = torch.ones(D, dtype=BF16)
    rope = _rope(T, BF16)
    p = _ref_pass(x32, sa, sb, None)
    rq, rk = fake.fused_qk_norm_rope(p[:, :H * D].view(T, H, D), p[:, H * D:2 * H * D].view(T, H, D), w, w, rope, 1e-6)
    oq, ok, v = fc.qk_norm_rope_fused(_stash(x32, sa, sb, None), H, H, D, w, w, rope, 1e-6)
    assert _same(oq, rq) and _same(ok, rk) and _same(v, p[:, 2 * H * D:].view(T, H, D))

    hid = 200
    x32 = _directed(T, hid)
    sb = torch.ones(hid)
    sa = torch.ones(T)
    res = torch.randn(T, hid).to(BF16)
    g6 = torch.randn(1, 6 * hid).to(BF16).chunk(6, dim=-1)
    idx = torch.zeros(T, dtype=torch.long)
    p = _ref_pass(x32, sa, sb, None)
    r = fake.indexed_gate_rms_norm_scale_shift(res, g6[2], p, torch.ones(hid, dtype=BF16), g6[3], g6[4], idx, 1e-6)
    f = fc.gate_rms_fused(_stash(x32, sa, sb, None), res, g6[2], torch.ones(hid, dtype=BF16), g6[3], g6[4], idx, 1e-6)
    assert _same(f[0], r[0]) and _same(f[1], r[1])
    assert _same(fc.gate_fused(_stash(x32, sa, sb, None), res, g6[5], idx), fake.indexed_gate(res, g6[5], p, idx))


def test_epilogue_mirror_equals_scale_pass():
    """The in-kernel epilogue (scale_b * acc, then scale_a, one fma for a bias, one RNE rounding) = the scale pass."""
    x32, sa, sb, b = _operands(33, 300, seed=1, bias=True)
    for bias in (None, b):
        e = fc.epilogue_bf16_torch(x32, sa[:, None], sb[None, :], None if bias is None else bias.float()[None, :])
        assert torch.equal(_bits(e.to(BF16)), _bits(_ref_pass(x32, sa, sb, bias)))
        assert torch.equal(e, e.to(BF16).float())                 # already a bf16 value, as the consumer's load gave


# ---- the DiT block: upstream's verbatim forwards, fused vs stock -------------------------------------------------------


def _install(mod, sites=fc.SITES, lora_types=()):
    return fc.install(sites, tr=mod, check_source=False, lora_types=lora_types)


def _run(mod, *, rows=37, hidden=200, heads=2, ffn=96, bias=False, with_rope=True, seed=0):
    blk = mod.MiniMaxH3DiTBlock(hidden, heads, ffn, seed=seed, bias=bias)
    out = blk(**mod.block_inputs(blk, rows, with_rope=with_rope))
    return blk, out


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rows", [1, 37])
def test_block_fused_equals_stock(bias, rows):
    stock_mod, fused_mod = fake.load(), fake.load()
    stock_blk, ref = _run(stock_mod, bias=bias, rows=rows)
    assert _install(fused_mod) == list(fc.SITES)
    fused_blk, out = _run(fused_mod, bias=bias, rows=rows)
    assert _same(out, ref)
    s = fc.stats()
    assert (s["fused_qkv"], s["fused_out"], s["fused_fc2"], s["materialize"]) == (1, 1, 1, 0)
    assert s["selfcheck_pass"] == 3 and s["selfcheck_fail"] == 0
    # the attention read v as the strided view of the qkv placeholder, with stock's values and layout
    v_f, v_s = fused_blk.attn.seen_v[0], stock_blk.attn.seen_v[0]
    assert _same(v_f, v_s) and v_f.stride() == v_s.stride()
    assert fc._tls.ctx is None


@pytest.mark.parametrize("sites", [("qkv",), ("out",), ("fc2",), ("out", "fc2")])
def test_block_site_subsets(sites):
    stock_mod, fused_mod = fake.load(), fake.load()
    _, ref = _run(stock_mod)
    assert _install(fused_mod, sites) == list(sites)
    _, out = _run(fused_mod)
    assert _same(out, ref)
    s = fc.stats()
    assert [s[f"fused_{x}"] for x in fc.SITES] == [int(x in sites) for x in fc.SITES] and s["materialize"] == 0


def test_block_without_rope_table_is_stock():
    stock_mod, fused_mod = fake.load(), fake.load()
    _, ref = _run(stock_mod, with_rope=False)
    _install(fused_mod)
    _, out = _run(fused_mod, with_rope=False)
    assert _same(out, ref) and fc.stats()["stash"] == 0


def test_block_plan_rejections():
    mod = fake.load()
    blk = mod.MiniMaxH3DiTBlock(200, 2, 96)
    rope = torch.zeros(4, 96)
    want, _ = fc.block_plan(blk, rope, fc.SITES)
    assert sorted(want.values()) == sorted(fc.SITES)
    assert fc.block_plan(blk, None, fc.SITES)[0] is None
    assert fc.block_plan(blk, rope, fc.SITES, lora_types=(mod.FakeFP8Linear,))[0] is None   # LoRA-wrapped layers
    blk.mlp.fc2.tp_size = 2
    assert fc.block_plan(blk, rope, fc.SITES)[0] is None                                    # all-reduce would read it
    blk.mlp.fc2.tp_size = 1
    blk.attn.out_proj = mod.FakeFP8Linear(256, 200, torch.Generator().manual_seed(0), pad_n=16)
    assert fc.block_plan(blk, rope, fc.SITES)[0] is None                                    # padded N is sliced+copied


def test_lora_wrapped_block_runs_stock():
    stock_mod, fused_mod = fake.load(), fake.load()
    _, ref = _run(stock_mod)
    _install(fused_mod, lora_types=(fused_mod.FakeFP8Linear,))
    _, out = _run(fused_mod)
    assert _same(out, ref) and fc.stats()["stash"] == 0


def test_selfcheck_failure_disables_site_and_serves_stock(monkeypatch):
    stock_mod, fused_mod = fake.load(), fake.load()
    _, ref = _run(stock_mod)
    _install(fused_mod)
    real = fc.gate_rms_fused

    def broken(st, residual, *a, **kw):
        r, m = real(st, residual, *a, **kw)
        m.view(torch.int16)[..., 0] ^= 1                         # one ulp off in one column: must be caught
        return r, m

    monkeypatch.setattr(fc, "gate_rms_fused", broken)
    _, out = _run(fused_mod)
    assert _same(out, ref)
    s = fc.stats()
    assert s["selfcheck_fail"] == 1 and s["fused_out"] == 0 and s["materialize"] == 1 and "out" in fc._state["disabled"]
    assert s["fused_qkv"] == 1 and s["fused_fc2"] == 1
    _, out2 = _run(fused_mod)                                     # disabled: no stash for out any more
    assert _same(out2, ref) and fc.stats()["materialize"] == 1


def test_consumer_mismatch_materialises_then_stock():
    """A pending stash met by a call that does not match it (here: q copied out of the placeholder) is made real by the
    ordinary scale pass before the stock consumer runs."""
    T, H, D = 9, 2, 128
    N = 3 * H * D
    x32, sa, sb, _ = _operands(T, N, seed=2)
    w = torch.ones(D, dtype=BF16)
    rope = _rope(T, BF16)
    ctx = fc._Ctx({})
    st = _stash(x32, sa, sb, None)
    ctx.pending["qkv"] = st
    fc._tls.ctx = ctx
    wrapped = fc.make_qk_wrapper(fake.fused_qk_norm_rope)
    q = st.ph[:, :H * D].view(T, H, D).clone()                   # not the placeholder's view
    k = st.ph[:, H * D:2 * H * D].view(T, H, D)
    wrapped(q, k, w, w, rope, 1e-6)
    assert not ctx.pending and fc.stats()["materialize"] == 1 and fake.CALLS == ["fused_qk_norm_rope"]
    assert torch.equal(_bits(st.ph), _bits(_ref_pass(x32, sa, sb, None)))   # the placeholder now holds stock's tensor


def test_other_site_stash_is_materialised():
    hid, T = 200, 6
    x32, sa, sb, _ = _operands(T, hid, seed=4)
    ctx = fc._Ctx({})
    st = _stash(x32, sa, sb, None)
    ctx.pending["out"] = st
    fc._tls.ctx = ctx
    x = torch.randn(T, hid).to(BF16)
    gate = torch.randn(1, hid).to(BF16)
    idx = torch.zeros(T, dtype=torch.long)
    out = fc.make_gate_wrapper(fake.indexed_gate)(x, gate, st.ph, idx)      # fc2's consumer sees out's stash
    assert _same(out, fake.indexed_gate(x, gate, _ref_pass(x32, sa, sb, None), idx))
    assert fc.stats()["materialize"] == 1 and not ctx.pending


def test_no_context_means_plain_stock_and_no_stash():
    mod = fake.load()
    _install(mod)
    lin = mod.FakeFP8Linear(64, 32, torch.Generator().manual_seed(0))
    y, _ = lin(torch.randn(5, 64).to(BF16))
    assert fc.stats()["stash"] == 0 and y.shape == (5, 32)


def test_stash_hook_only_for_the_blocks_weights():
    gen = torch.Generator().manual_seed(0)
    mod = fake.load()
    mine, other = mod.FakeFP8Linear(64, 32, gen), mod.FakeFP8Linear(64, 32, gen)
    a = torch.randn(4, 64).to(torch.float8_e4m3fn)
    s = torch.ones(4, 1)
    ctx = fc._Ctx({(mine.weight.data_ptr(), tuple(mine.weight.shape)): "fc2"})
    fc._tls.ctx = ctx
    assert fc.stash_hook(a, other.weight, s, other.weight_scale, BF16, None) is None
    assert fc.stash_hook(a, mine.weight, s, mine.weight_scale, torch.float16, None) is None   # bf16 only
    ph = fc.stash_hook(a, mine.weight, s, mine.weight_scale, BF16, None)
    assert ph is not None and ph.shape == (4, 32) and ph.dtype == BF16 and set(ctx.pending) == {"fc2"}
    assert ctx.pending["fc2"].x32.dtype == torch.float32 and ctx.pending["fc2"].sa.shape == (4,)
    assert fc.stash_hook(a, mine.weight, s, mine.weight_scale, BF16, None) is None  # once per block forward


def test_parse_sites():
    assert fc.parse_sites("1") == fc.SITES and fc.parse_sites("all") == fc.SITES
    assert fc.parse_sites("0") == () and fc.parse_sites(None) == () and fc.parse_sites("") == ()
    assert fc.parse_sites("fc2,qkv") == ("qkv", "fc2") and fc.parse_sites("bogus") == ()


def test_install_needs_two_pass_level(monkeypatch):
    monkeypatch.setitem(fp8_gemm._state, "fused", True)
    assert fc.install(fc.SITES, tr=fake.load(), check_source=False, lora_types=()) == []


def test_install_with_source_check_skips_without_upstream_kernels():
    """Without vLLM-Omni importable the kernel fingerprints cannot be verified: every site is skipped."""
    assert fc.install(fc.SITES, tr=fake.load(), check_source=True, lora_types=()) == []


def test_fingerprints_were_taken_over_the_verbatim_upstream_source():
    from h3_tp2.step_cache import fingerprint_source

    fps = fc._fingerprints(fake.load())
    for name in ("attn_forward", "block_forward", "mlp_forward"):
        assert fps[name] == fc.EXPECTED[name], name
    for name, src in fake.UPSTREAM_SOURCES.items():
        assert fingerprint_source(src) == fc.EXPECTED[name], name
    assert set(fake.UPSTREAM_SOURCES) | {"attn_forward", "block_forward", "mlp_forward"} == set(fc.EXPECTED)
