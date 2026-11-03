"""CPU checks for ``H3_TP2_FP8_GEMM=cutlass`` (h3_tp2/fp8_cutlass.py): the dispatcher routing, not the kernel.

The DSL kernel computes the two-pass arithmetic (y = bf16(sa * (sb * acc))) on its own accumulator; off the GPU the
router runs that arithmetic on the same fp32 product as the fake two-pass path (``fp8_cutlass._state["cpu_mirror"]``),
so these tests check what the GPU cannot show in isolation: which GEMMs the router takes (the running DiT block's
qkv / out / fc1, nothing else), that it runs before the stash hooks, that fp8_consumers keeps only fc2 at this level,
that the stock consumers and the MLP's own activation then see bf16 tensors with the two-pass values (block output
bit-identical to the plain two-pass block), and the fallbacks.  The kernel is checked bit for bit on the GPU:
``tests/test_fp8_cutlass_gpu.py`` and the install check (``fp8_cutlass.probe``).
"""
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from h3_tp2 import fp8_consumers as fc  # noqa: E402
from h3_tp2 import fp8_cutlass as fcut  # noqa: E402
from h3_tp2 import fp8_gemm, fp8_swiglu  # noqa: E402
from tests import fake_h3_consumers as fake  # noqa: E402

BF16 = torch.bfloat16


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    hooks, routers = list(fp8_gemm._stash_hooks), list(fp8_gemm._routers)
    gstate = dict(fp8_gemm._state)
    fc._state.update(cpu_mirror=True, checked={}, disabled=set())
    for d in (fc._stats, fcut._stats):
        for k in d:
            d[k] = 0
    fc._tls.ctx = None
    fcut._tls.want = None
    fcut._state.update(cpu_mirror=True)
    fake.CALLS.clear()
    yield
    fp8_gemm._stash_hooks[:] = hooks
    fp8_gemm._routers[:] = routers
    fp8_gemm._state.clear()
    fp8_gemm._state.update(gstate)
    fc._tls.ctx = None
    fcut._tls.want = None
    fc._state.update(cpu_mirror=False, checked={}, disabled=set())
    fcut._state.update(cpu_mirror=False)


def _same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _cutlass_level(mod, *, consumer_sites=fc.SITES, swiglu=True):
    """What the plugin does at H3_TP2_FP8_GEMM=cutlass once the kernel passed its install check."""
    fp8_gemm._state["level"] = "cutlass"
    fp8_gemm.set_router(fcut.route)
    assert fcut.install_sites(tr=mod)
    seen = []
    if swiglu:                                   # level 1 stays installed; its hook must never see fc1 here
        assert fp8_swiglu._install_on(mod.MiniMaxH3MLP, level=1, lora_layer_type=())

        def spy(a, b, *rest):  # noqa: ANN001,ANN202
            if getattr(fp8_swiglu._tls, "stash_next", False):
                seen.append(tuple(b.shape))
            return None                          # (the real hook would stash: it needs Triton)
        fp8_gemm.set_stash_hook(spy)
    done = fc.install(consumer_sites, tr=mod, check_source=False, lora_types=())
    return done, seen


def _run(mod, *, rows=37, hidden=200, heads=2, ffn=96, bias=False, seed=0):
    blk = mod.MiniMaxH3DiTBlock(hidden, heads, ffn, seed=seed, bias=bias)
    return blk, blk(**mod.block_inputs(blk, rows))


@pytest.mark.parametrize("rows", [1, 37])
def test_block_cutlass_equals_two_pass_block(rows):
    stock_mod, cut_mod = fake.load(), fake.load()
    stock_blk, ref = _run(stock_mod, rows=rows)
    done, seen = _cutlass_level(cut_mod)
    assert done == ["fc2"]                                       # qkv / out consumers stay stock at this level
    cut_blk, out = _run(cut_mod, rows=rows)
    assert _same(out, ref)
    assert fcut.stats() == {"dsl": 3, "fallthrough": 0}          # qkv, out, fc1
    s = fc.stats()
    assert (s["stash"], s["fused_fc2"], s["fused_qkv"], s["fused_out"], s["materialize"]) == (1, 1, 0, 0, 0)
    assert seen == []                                            # fc1 never reached the SwiGLU stash hook
    assert "fused_qk_norm_rope" in cut_mod.CALLS and "indexed_gate_rms_norm_scale_shift" in cut_mod.CALLS
    v_c, v_s = cut_blk.attn.seen_v[0], stock_blk.attn.seen_v[0]    # v: the strided view of the bf16 qkv output
    assert _same(v_c, v_s) and v_c.stride() == v_s.stride()
    assert fcut._tls.want is None and fc._tls.ctx is None


def test_bias_falls_through_to_two_pass():
    """The kernel has no bias epilogue: every routed GEMM falls through to the hooks / two-pass, same bits."""
    stock_mod, cut_mod = fake.load(), fake.load()
    _, ref = _run(stock_mod, bias=True)
    _cutlass_level(cut_mod, swiglu=False)
    _, out = _run(cut_mod, bias=True)
    assert _same(out, ref)
    assert fcut.stats() == {"dsl": 0, "fallthrough": 3}
    assert fc.stats()["fused_fc2"] == 1


def test_without_consumer_fuse_and_swiglu():
    stock_mod, cut_mod = fake.load(), fake.load()
    _, ref = _run(stock_mod)
    done, _ = _cutlass_level(cut_mod, consumer_sites=(), swiglu=False)
    assert done == []
    _, out = _run(cut_mod)
    assert _same(out, ref) and fcut.stats()["dsl"] == 3 and fc.stats()["stash"] == 0


def test_router_takes_only_the_running_blocks_weights():
    gen = torch.Generator().manual_seed(0)
    mod = fake.load()
    blk = mod.MiniMaxH3DiTBlock(200, 2, 96)
    other = mod.FakeFP8Linear(200, 768, gen)
    a = torch.randn(4, 200).to(torch.float8_e4m3fn)
    s = torch.ones(4, 1)
    qkv = blk.attn.qkv_proj
    assert fcut.route(a, qkv.weight, s, qkv.weight_scale, BF16, None) is None      # no block forward running
    fcut._tls.want = fcut.block_want(blk)
    assert sorted(fcut._tls.want.values()) == sorted(fcut.SITES)
    assert fcut.route(a, other.weight, s, other.weight_scale, BF16, None) is None   # not this block's
    y = fcut.route(a, qkv.weight, s, qkv.weight_scale, BF16, None)
    assert y is not None and y.shape == (4, 768) and y.dtype == BF16
    assert _same(y, fake._two_pass_cpu(a, qkv.weight, s, qkv.weight_scale, BF16))
    fc2 = blk.mlp.fc2
    a2 = torch.randn(4, 96).to(torch.float8_e4m3fn)
    assert fcut.route(a2, fc2.weight, s, fc2.weight_scale, BF16, None) is None      # fc2 stays on cuBLASLt
    assert fcut.route(a, qkv.weight, s, qkv.weight_scale, torch.float16, None) is None   # bf16 output only
    assert fcut.stats() == {"dsl": 1, "fallthrough": 1}


def test_router_runs_before_the_stash_hooks():
    calls = []
    fp8_gemm.set_stash_hook(lambda *a: calls.append("hook"))
    fp8_gemm.set_router(lambda *a: torch.zeros(2, 2, dtype=BF16))
    a = torch.zeros(2, 16).to(torch.float8_e4m3fn)
    y = fake.dispatch(a, torch.zeros(16, 2).to(torch.float8_e4m3fn), torch.ones(2, 1), torch.ones(2), BF16, None)
    assert calls == [] and y.shape == (2, 2)


def test_launch_failure_disables_the_level(monkeypatch):
    mod = fake.load()
    blk = mod.MiniMaxH3DiTBlock(200, 2, 96)
    fcut._tls.want = fcut.block_want(blk)

    def boom(*a, **k):  # noqa: ANN002,ANN003,ANN202
        raise RuntimeError("launch")
    monkeypatch.setattr(fcut, "scaled_mm", boom)
    monkeypatch.setitem(fcut._state, "ok", True)
    q = blk.attn.qkv_proj
    a = torch.zeros(3, 200).to(torch.float8_e4m3fn)
    assert fcut.route(a, q.weight, torch.ones(3, 1), q.weight_scale, BF16, None) is None
    assert fcut._state["ok"] is False and fcut.stats()["fallthrough"] == 1


def test_block_wrapper_keeps_source_fingerprint():
    mod = fake.load()
    before = fc._fingerprints(mod)["block_forward"]
    fcut.install_sites(tr=mod)
    assert getattr(mod.MiniMaxH3DiTBlock.forward, "_h3_tp2_cutlass", False)
    assert fc._fingerprints(mod)["block_forward"] == before == fc.EXPECTED["block_forward"]
    fcut.install_sites(tr=mod)                                    # idempotent: wrapped once
    assert mod.MiniMaxH3DiTBlock.forward.__wrapped__.__name__ == "forward"
    assert not getattr(mod.MiniMaxH3DiTBlock.forward.__wrapped__, "_h3_tp2_cutlass", False)


def test_consumers_keep_all_sites_at_tensor32():
    fp8_gemm._state["level"] = "tensor32"
    assert fc.install(fc.SITES, tr=fake.load(), check_source=False, lora_types=()) == list(fc.SITES)


def test_probe_failures_keep_two_pass(monkeypatch):
    monkeypatch.setitem(fp8_gemm._state, "triton", False)
    assert fcut.probe("cpu") is False and "Triton" in fcut._state["reason"]
    monkeypatch.setitem(fp8_gemm._state, "triton", True)

    def no_dsl():  # noqa: ANN202
        raise ImportError("No module named 'cutlass'")
    monkeypatch.setattr(fcut, "DslGemm", no_dsl)
    assert fcut.probe("cpu") is False and "ImportError" in fcut._state["reason"] and fcut._state["gemm"] is None
    monkeypatch.setattr(fcut, "DslGemm", lambda: types.SimpleNamespace(compile_s=0.5))
    monkeypatch.setattr(fcut, "selfcheck", lambda g, d: ["rand M=200 N=5376 K=7168"])
    assert fcut.probe("cpu") is False and "not bit-identical" in fcut._state["reason"]
    monkeypatch.setattr(fcut, "selfcheck", lambda g, d: [])
    assert fcut.probe("cpu") is True and fcut._state["ok"]
    fcut._state.update(ok=False, gemm=None)


def test_dispatcher_install_falls_back_without_the_kernel(monkeypatch):
    """fp8_gemm.install("cutlass") with a fake vllm: no usable kernel (no GPU here) -> level tensor32, no router."""
    ops = types.SimpleNamespace(cutlass_scaled_mm=lambda *a: "stock")
    vllm = types.ModuleType("vllm")
    vllm._custom_ops = ops
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", ops)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fp8_gemm.install(level="cutlass")
    assert fp8_gemm._state["level"] == "tensor32" and fcut.route not in fp8_gemm._routers
    assert getattr(ops.cutlass_scaled_mm, "_h3_tp2_tensor32", False)
    with pytest.raises(ValueError):
        fp8_gemm.install(level="bogus")
