"""H3_TP2_ULYSSES_EXACT=2 on the GPU: the receive-buffer quant (``ulysses_outq._quant_rows_kernel``) against the stock
path -- copy 5 (``unpack_rows``) + vLLM's ``QuantFP8`` exactly as the serving process builds and calls it (custom ops
"none" under the diffusion ``VllmConfig(compilation_config=CompilationConfig())``, so ``forward_native`` compiled by
Inductor with ``dynamic=True``) -- bit for bit: FP8 values, scales, shapes, strides, dtypes.

Inputs: random rows, attention-output-like rows (SDPA outputs of random q/k/v, the reverse exchange's real content),
directed rows (zeros, -0.0, tiny, huge, inf, NaN, maxima at the segment boundary), and every bf16 value as a row
maximum (every per-token scale that can occur: amax is a bf16 value, so the scale and its reciprocal take at most
2^15 + 2 distinct inputs; each is checked with its own row of random elements).

Run on Spark_06 from a copy of the tree (< 2 GB): ``python -m pytest tests/test_ulysses_outq_gpu.py -q``.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("triton")
pytest.importorskip("vllm")

from h3_tp2 import ulysses_outq as oq  # noqa: E402

WS, H, D = 2, 56, 128
K = H * D
SEG = K // WS
DEV = "cuda"


@pytest.fixture(scope="module")
def stock_quant():
    """The out-projection's QuantFP8 as the diffusion worker constructs it (FP8ScaledMMLinearKernel.__init__)."""
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    cfg = VllmConfig(compilation_config=CompilationConfig(), device_config=DeviceConfig(device=torch.device(DEV)))
    with set_current_vllm_config(cfg):
        q = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN, num_token_padding=None)
    assert getattr(q._forward_method, "__name__", "") == "forward_native", q._forward_method
    assert hasattr(q._forward_method, "_torchdynamo_orig_callable"), "expected the torch.compile'd forward_native"
    return q


def _bits(t):
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int32)


def assert_same(stock, got):
    for name, a, b in zip(("q", "scale"), stock, got):
        assert a.shape == b.shape and a.stride() == b.stride() and a.dtype == b.dtype, name
        diff = _bits(a) != _bits(b)
        assert not bool(diff.any()), f"{name}: {int(diff.sum())} elements differ, first rows " \
            f"{diff.reshape(a.shape[0], -1).any(1).nonzero().flatten()[:8].tolist()}"


def recv_from_rows(x):
    m = x.shape[0]
    return x.view(m, WS, H // WS, D).transpose(0, 1).contiguous()


def stock_path(stock_quant, recv):
    return stock_quant(oq.unpack_rows(recv), None, None)


def attention_like(m, seed):
    """Two ranks' worth of attention output rows: SDPA over random q/k/v with 28 heads, rows of this rank's shard."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    hl = H // WS
    s = 2 * m
    parts = []
    for _ in range(WS):
        q, k, v = (torch.randn(1, hl, s, D, device=DEV, generator=g, dtype=torch.bfloat16) * sc for sc in (1.0, 1.0, 2.0))
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)   # [1, hl, S, d]
        parts.append(o[0, :, :m].transpose(0, 1).contiguous())           # [m, hl, d]: rows of the shard
    return torch.stack(parts)                                             # [P, M, hl, d]


@pytest.mark.parametrize("m", [4096, 4100, 18880])
def test_random_rows_bitwise(stock_quant, m):
    g = torch.Generator(device=DEV).manual_seed(m)
    x = (torch.randn(m, K, device=DEV, generator=g) * torch.rand(m, 1, device=DEV, generator=g) * 8).to(torch.bfloat16)
    recv = recv_from_rows(x)
    assert_same(stock_path(stock_quant, recv), oq.quant_from_recv(recv))


@pytest.mark.parametrize("m", [4096, 10240])
def test_attention_output_rows_bitwise(stock_quant, m):
    recv = attention_like(m, 7)
    assert_same(stock_path(stock_quant, recv), oq.quant_from_recv(recv))


def test_directed_rows_bitwise(stock_quant):
    g = torch.Generator(device=DEV).manual_seed(3)
    m = 64
    x = (torch.randn(m, K, device=DEV, generator=g) * 2).to(torch.bfloat16)
    x[0] = 0
    x[1] = -0.0
    x[2] = (torch.randn(K, device=DEV, generator=g) * 1e-38).to(torch.bfloat16)   # bf16 subnormals
    x[3] = (torch.randn(K, device=DEV, generator=g) * 1e36).to(torch.bfloat16)
    x[4, 17] = float("inf")
    x[5, 4000] = float("nan")
    x[6, 9] = float("-inf")
    x[7, SEG - 1] = 900.0
    x[8, SEG] = -900.0
    x[9] = torch.finfo(torch.bfloat16).max
    x[10, :SEG] = 0
    x[11, SEG:] = 0
    x[12] = 448.0
    x[13] = (torch.arange(K, device=DEV) % 448 - 224).to(torch.bfloat16) * 0.01
    recv = recv_from_rows(x)
    assert_same(stock_path(stock_quant, recv), oq.quant_from_recv(recv))


def test_every_row_maximum_bitwise(stock_quant):
    """Every non-negative bf16 bit pattern 0x0000..0x7F80 plus NaN as the row's abs-max, with random elements below
    it (and its negative somewhere): covers every scale and reciprocal the quant can compute."""
    vals = torch.arange(0, 0x7F81, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    vals = torch.cat([vals, torch.tensor([float("nan")], dtype=torch.bfloat16)]).to(DEV)
    g = torch.Generator(device=DEV).manual_seed(11)
    for c in range(0, vals.numel(), 4096):
        v = vals[c:c + 4096].float()
        m = v.numel()
        r = torch.rand(m, K, device=DEV, generator=g) * 2 - 1
        x = (r * v[:, None]).to(torch.bfloat16)
        x = torch.where(torch.isfinite(x) | torch.isnan(v[:, None]), x, torch.zeros_like(x))   # inf * 0 -> keep finite
        pos = torch.randint(0, K, (m,), device=DEV, generator=g)
        sign = torch.where(torch.rand(m, device=DEV, generator=g) < 0.5, -1.0, 1.0)
        x[torch.arange(m, device=DEV), pos] = (v * sign).to(torch.bfloat16)
        recv = recv_from_rows(x)
        assert_same(stock_path(stock_quant, recv), oq.quant_from_recv(recv))


def test_contiguous_and_recv_layouts_agree(stock_quant):
    recv = attention_like(4096, 21)
    x = oq.unpack_rows(recv)
    a = oq.quant_from_recv(recv)
    b = oq.quant_rows(x, x.shape[0], WS, SEG, SEG, K)
    assert_same(a, b)
    assert_same(stock_quant(x, None, None), b)


def test_selfcheck_on_gpu(stock_quant):
    oq._state.update(disabled=None, checked=False, sources=None)
    recv = attention_like(4096, 5)
    ph = oq._placeholder((1, recv.shape[1], H, D), recv)
    pend = oq._Deferred(recv, ph)
    res = pend.quantize(stock_quant)
    assert oq._state["checked"] and oq._state["disabled"] is None, oq._state
    assert_same(stock_path(stock_quant, recv), res)
    assert_same(stock_path(stock_quant, recv), oq._Deferred(recv, ph).quantize(stock_quant))   # the recv path now
    oq._state.update(disabled=None, checked=False, sources=None)
