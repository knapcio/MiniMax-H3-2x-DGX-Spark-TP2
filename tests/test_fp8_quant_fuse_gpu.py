"""H3_TP2_FP8_QUANT_FUSE on the GPU: the fused producers (h3_tp2/fp8_quant_fuse.py) against the stock pair, bit for bit.

Stock pair = the upstream producer (vLLM-Omni's ``rms_norm_indexed_scale_shift`` / ``indexed_gate_rms_norm_scale_shift``
Triton kernels, ulysses_exact's reverse unpack, ``SiluAndMul`` = ``torch.ops._C.silu_and_mul``) followed by the per-token
``QuantFP8`` built under the diffusion vLLM config (``CompilationConfig()``: custom ops "none", so its compiled
``forward_native``, which is what serving runs).  Compared: fp8 bytes, scale bits, by-products (residual_out), on the
DiT widths at small M with serving-like magnitudes, on directed values (zero rows, +-0, subnormals, saturation), and on
an exhaustive sweep of the row maximum over every positive finite bf16 value.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("triton")
pytest.importorskip("vllm._custom_ops")
from h3_tp2 import fp8_quant_fuse as fq  # noqa: E402

BF16 = torch.bfloat16
HIDDEN, INNER, FFN = 5376, 7168, 14336


@pytest.fixture(scope="module")
def serving_quant():
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(),
                                            device_config=DeviceConfig(device="cuda"))):
        q = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
    # the diffusion config disables custom ops: the op is the compiled forward_native, not ops.scaled_fp8_quant
    assert getattr(q._forward_method, "__name__", "") != "forward_cuda"
    return lambda h: q(h, None, None)


def _same_quant(a, b):
    (qa, sa), (qb, sb) = a, b
    assert qa.shape == qb.shape and qa.dtype == qb.dtype == torch.float8_e4m3fn and sa.shape == sb.shape
    return torch.equal(qa.view(torch.uint8), qb.view(torch.uint8)) and torch.equal(sa.view(torch.int32), sb.view(torch.int32))


def _mod_inputs(M, H, seed, *, std_scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mag = torch.exp(torch.randn(M, 1, device="cuda", generator=g) * 1.5)
    x = (torch.randn(M, H, device="cuda", generator=g) * mag).to(BF16)
    weight = (1.0 + 0.1 * torch.randn(H, device="cuda", generator=g)).to(BF16)
    ada = (torch.randn(6, 6 * H, device="cuda", generator=g) * std_scale).to(BF16)     # chunk views, row stride 6H
    shift, scale, gate = ada[:, :H], ada[:, H:2 * H], ada[:, 2 * H:3 * H]
    idx = torch.randint(0, 6, (M,), device="cuda", generator=g)
    return x, weight, shift, scale, gate, idx


def _upstream():
    from vllm_omni.diffusion.layers import indexed_modulation as im
    return im


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("M", [1, 200, 1000, 4100])
def test_qkv_site_rms_quant(M, mode, serving_quant):
    im = _upstream()
    x, w, shift, scale, _, idx = _mod_inputs(M, HIDDEN, seed=M)
    h_ref = im.rms_norm_indexed_scale_shift(x, w, shift, scale, idx, 1e-6)
    ph = torch.full_like(x, float("nan")) if mode else None
    assert _same_quant(fq.rms_quant(x, w, shift, scale, idx, 1e-6, None, ph), serving_quant(h_ref))
    if mode:      # RELOAD: upstream's bf16 row in the placeholder too
        assert torch.equal(ph.view(torch.int16), h_ref.view(torch.int16))


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("M", [1, 200, 1000, 4100])
def test_fc1_site_gate_rms_quant(M, mode, serving_quant):
    im = _upstream()
    x, w, shift, scale, gate, idx = _mod_inputs(M, HIDDEN, seed=M + 1)
    branch = (torch.randn(M, HIDDEN, device="cuda") * 3).to(BF16)
    res_ref, mod_ref = im.indexed_gate_rms_norm_scale_shift(x, gate, branch, w, shift, scale, idx, 1e-6)
    ph = torch.full_like(x, float("nan")) if mode else None
    res, q, s = fq.gate_rms_quant(x, gate, branch, w, shift, scale, idx, 1e-6, None, ph)
    assert torch.equal(res.view(torch.int16), res_ref.view(torch.int16))
    assert _same_quant((q, s), serving_quant(mod_ref))
    if mode:
        assert torch.equal(ph.view(torch.int16), mod_ref.view(torch.int16))


@pytest.mark.parametrize("M", [1, 200, 4100])
def test_out_site_unpack_quant(M, serving_quant):
    g = torch.Generator(device="cuda").manual_seed(M + 2)
    recv = (torch.randn(2, M, 28, 128, device="cuda", generator=g)
            * torch.exp(torch.randn(1, M, 1, 1, device="cuda", generator=g))).to(BF16)
    ref = serving_quant(fq.unpack_stock(recv, 56)[0].reshape(M, INNER))
    assert _same_quant(fq.unpack_quant(recv), ref)


@pytest.mark.parametrize("M", [1, 200, 1000])
def test_fc2_site_silu_mul_quant(M, serving_quant):
    from vllm_omni.diffusion.layers.activation import SiluAndMul
    act = SiluAndMul()
    g = torch.Generator(device="cuda").manual_seed(M + 3)
    h = (torch.randn(M, 2 * FFN, device="cuda", generator=g) * 4).to(BF16)       # gate values up to ~ +-16
    h[:, :7] = torch.tensor([0.0, -0.0, 1e-38, -88.0, 88.0, 3.0, -3.0], device="cuda").to(BF16)
    h[:, FFN:FFN + 7] = torch.tensor([-0.0, 0.0, -0.0, 1.0, -1.0, -0.0, 0.0], device="cuda").to(BF16)
    ref = serving_quant(act(h))
    assert _same_quant(fq.silu_mul_quant(h), ref)


def test_quant_alone_directed_rows(serving_quant):
    rows = []
    K = HIDDEN
    rows.append(torch.zeros(K))                                         # all zero: the minimum scale
    rows.append(torch.full((K,), -0.0))
    r = torch.zeros(K); r[17] = 1e-38; rows.append(r)                   # subnormal-range maximum
    r = torch.randn(K) * 1e-30; rows.append(r)
    r = torch.randn(K); r[5] = 3.0e38; rows.append(r)                   # near the bf16 maximum
    r = torch.linspace(-448, 448, K); rows.append(r)
    r = torch.randn(K) * 1e4; rows.append(r)
    h = torch.stack(rows).to(BF16).cuda()
    assert _same_quant(fq.quant(h), serving_quant(h))


def test_quant_alone_exhaustive_row_maximum(serving_quant):
    """Every positive finite bf16 value as a row maximum (and its negation in the row), random entries below it."""
    bits = torch.arange(1, 0x7F80, dtype=torch.int32)                  # +min subnormal .. +max finite
    amax = bits.to(torch.int16).view(BF16).cuda()
    n = amax.numel()
    K = 1024
    g = torch.Generator(device="cuda").manual_seed(7)
    h = (torch.rand(n, K, device="cuda", generator=g) * 2 - 1) * amax.float()[:, None]
    h = h.to(BF16)
    h[:, 0] = amax
    h[:, 1] = -amax
    a, b = fq.quant(h), serving_quant(h)
    assert torch.equal(a[1].view(torch.int32), b[1].view(torch.int32)), "scale differs"
    assert torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8)), "fp8 values differ"


# ---- the whole block on the GPU: upstream's verbatim forwards with the real producers, QuantFP8 and GEMM ----------
def _gpu_module(quant_factory):
    """tests/fake_h3_consumers' verbatim DiT block / attention / MLP forwards with vLLM-Omni's real Triton producers
    and consumers, vLLM's SiluAndMul op, FP8 linears that quantise through a serving QuantFP8 and multiply with the
    two-pass GEMM, and the Ulysses reverse exchange simulated through ulysses_exact.exchange_back (2 ranks)."""
    import types

    from vllm_omni.diffusion.layers import fused_qk_norm_rope as fqr
    from vllm_omni.diffusion.layers.activation import SiluAndMul

    from h3_tp2 import fp8_gemm, ulysses_exact
    from tests import fake_h3_consumers as fake

    im = _upstream()
    mod = fake.load()
    mod.rms_norm_indexed_scale_shift = im.rms_norm_indexed_scale_shift
    mod.indexed_gate_rms_norm_scale_shift = im.indexed_gate_rms_norm_scale_shift
    mod.indexed_gate = im.indexed_gate
    mod.fused_qk_norm_rope = fqr.fused_qk_norm_rope
    orig_run = fake.MiniMaxH3Attention._run_packed_attention

    def run_packed(self, q, k, v, **kw):  # noqa: ANN001,ANN202
        o = orig_run(self, q, k, v, **kw)
        m, h, d = o.shape
        o3 = o.view(m, 2, h // 2, d).transpose(0, 1).reshape(2 * m, h // 2, d)
        return ulysses_exact.exchange_back(o3.unsqueeze(0), h, 2,
                                           lambda recv, inp, group=None: recv.copy_(inp.view_as(recv))).squeeze(0)
    mod.MiniMaxH3Attention._run_packed_attention = run_packed

    def lin_forward(self, x):  # noqa: ANN001,ANN202
        x2 = x.view(-1, x.shape[-1])
        a, s = self.quant_method.fp8_linear.quant_fp8(x2, None, None)
        out = fp8_gemm.scaled_mm_tensor32(a, self.weight, s, self.weight_scale, BF16)
        return out.view(*x.shape[:-1], out.shape[-1]), None
    mod.FakeFP8Linear.forward = lin_forward

    def build(rows, seed):  # noqa: ANN202
        blk = mod.MiniMaxH3DiTBlock(HIDDEN, 4, FFN, seed=seed)
        blk.mlp.act_fn = SiluAndMul()
        for m in blk.modules():
            for name, val in list(vars(m).items()):
                if isinstance(val, torch.Tensor):
                    setattr(m, name, val.cuda())
            if isinstance(m, mod.FakeFP8Linear):
                m.quant_method = types.SimpleNamespace(fp8_linear=types.SimpleNamespace(quant_fp8=quant_factory()))
        inp = mod.block_inputs(blk, rows, seed=seed + 1)
        for k in ("x", "t_emb", "combined_indices", "rope_table"):
            inp[k] = inp[k].cuda()
        inp["x"] = (inp["x"].float() * 4).to(BF16)
        return blk, inp
    return mod, build


@pytest.mark.parametrize("rows", [64, 333])
def test_block_on_gpu_bit_identical(rows):
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm_omni.diffusion.layers.activation import SiluAndMul

    from h3_tp2 import fp8_gemm, ulysses_exact

    assert fp8_gemm.probe_triton("cuda")
    cfg = VllmConfig(compilation_config=CompilationConfig(), device_config=DeviceConfig(device="cuda"))

    def qf():  # noqa: ANN202
        with set_current_vllm_config(cfg):
            return QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    stock_mod, build_stock = _gpu_module(qf)
    blk, inp = build_stock(rows, seed=5)
    ref = blk(**inp)
    saved = (QuantFP8.forward, SiluAndMul.forward)
    fq._state.update(checked={}, disabled=set())
    for k in fq._stats:
        fq._stats[k] = 0
    try:
        mod, build = _gpu_module(qf)
        done = fq.install(fq.SITES, tr=mod, quant_cls=QuantFP8, act_cls=SiluAndMul, check_source=False, lora_types=(),
                          ulysses_mode="1")
        assert done == list(fq.SITES)
        blk2, inp2 = build(rows, seed=5)
        out = blk2(**inp2)
        torch.cuda.synchronize()
        st = fq.stats()
        assert (st["fused_qkv"], st["fused_out"], st["fused_fc1"], st["fused_fc2"]) == (1, 1, 1, 1), st
        assert st["consumed"] == 4 and st["materialize"] == 0 and st["selfcheck_fail"] == 0, st
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    finally:
        QuantFP8.forward, SiluAndMul.forward = saved
        ulysses_exact.set_unpack_hook(None)
        fq._state.update(sites=(), checked={}, disabled=set())
