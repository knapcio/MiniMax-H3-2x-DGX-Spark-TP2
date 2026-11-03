"""H3_TP2_ULYSSES_EXACT on the GPU: the real producers, the real exchange code and real cuDNN SDPA, bitwise.

* the two producers writing per destination into the send buffers (upstream ``_rms_norm_rope_kernel`` via
  ``ulysses_nocopy._triton_launch``; the consumer-fused ``fp8_consumers`` kernels) against the stock producer + stock's
  pack copy;
* two ranks simulated in one process (one thread per rank, the collective emulated by a byte copy of each rank's
  dim-0 chunk, i.e. what ``all_to_all_single`` moves): the exact forward against vLLM-Omni's own ``all_to_all_4D``
  (``comm.py``) with ``CuDNNAttentionImpl.forward_cuda`` (the serving backend, with the ``valid_kv_length`` K/V slice)
  on the received tensors, for every producer and both reverse modes, and ``verify_forward``;
* the two effects the 2026-11-02 nocopy failure was attributed to, which turn out NOT to differ: cuDNN on the strided
  packed-QKV views vs contiguous copies, and the whole nocopy exchange against the same stock path.

Run on a Spark next to the idle head (< 3 GB):
``python -m pytest tests/test_ulysses_exact_gpu.py -q`` with the serving env sourced.
"""
import threading

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("triton")
fq = pytest.importorskip("vllm_omni.diffusion.layers.fused_qk_norm_rope")
import torch.distributed as dist  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata  # noqa: E402
from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl  # noqa: E402
from vllm_omni.diffusion.distributed import comm  # noqa: E402

from h3_tp2 import fp8_consumers as fc  # noqa: E402
from h3_tp2 import ulysses_exact as ux  # noqa: E402
from h3_tp2 import ulysses_nocopy as nc  # noqa: E402

WS, H, D, EPS = 2, 56, 128, 1e-5
DEV = "cuda"


def _bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


class GpuA2A:
    """all_to_all_single for WS thread-ranks on one GPU (equal splits, contiguous buffers, like NCCL)."""

    def __init__(self):
        self.barrier = threading.Barrier(WS, timeout=120)
        self.slots = [None] * WS
        self.tl = threading.local()
        self.calls = [0] * WS

    def __call__(self, out, inp, group=None, **kw):  # noqa: ARG002
        assert not kw.get("output_split_sizes") and not kw.get("input_split_sizes")
        assert out.is_contiguous() and inp.is_contiguous() and out.numel() == inp.numel()
        r = self.tl.rank
        self.calls[r] += 1
        self.slots[r] = inp
        self.barrier.wait()
        torch.cuda.synchronize()
        out.view(-1).copy_(torch.cat([self.slots[s].reshape(-1).chunk(WS)[r] for s in range(WS)]))
        torch.cuda.synchronize()
        self.barrier.wait()

    def run(self, fn):
        res, err = [None] * WS, []

        def body(r):
            self.tl.rank = r
            try:
                res[r] = fn(r)
            except BaseException as e:  # noqa: BLE001
                err.append(e)
                self.barrier.abort()

        ts = [threading.Thread(target=body, args=(r,)) for r in range(WS)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if err:
            raise err[0]
        torch.cuda.synchronize()
        return res


@pytest.fixture
def a2a(monkeypatch):
    fake = GpuA2A()
    monkeypatch.setattr(dist, "all_to_all_single", fake)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: WS)
    return fake


def _rope_inputs(m, seed, table_dtype=torch.float32):
    g = torch.Generator(device=DEV).manual_seed(seed)
    qkv = (torch.randn(m, 3 * H * D, generator=g, device=DEV) * 2).to(torch.bfloat16)
    q, k, v = (t.view(m, H, D) for t in qkv.split([H * D] * 3, dim=-1))
    wq = (1 + 0.1 * torch.randn(D, generator=g, device=DEV)).to(torch.bfloat16)
    wk = (1 + 0.1 * torch.randn(D, generator=g, device=DEV)).to(torch.bfloat16)
    th = torch.rand(m, 48, generator=g, device=DEV) * 6.28
    return q, k, v, wq, wk, torch.cat([th.cos(), th.sin()], -1).to(table_dtype)


def _stash(m, seed, with_bias=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = 3 * H * D
    x32 = torch.randn(m, n, generator=g, device=DEV) * 40
    sa = torch.rand(m, generator=g, device=DEV) * 0.01 + 1e-3
    sb = torch.rand(n, generator=g, device=DEV) * 0.01 + 1e-3
    bias = (torch.randn(n, generator=g, device=DEV) * 0.1) if with_bias else None
    st = fc.Stash(x32, sa, sb, bias, torch.empty(m, n, dtype=torch.bfloat16, device=DEV))
    wq = (1 + 0.1 * torch.randn(D, generator=g, device=DEV)).to(torch.bfloat16)
    wk = (1 + 0.1 * torch.randn(D, generator=g, device=DEV)).to(torch.bfloat16)
    th = torch.rand(m, 48, generator=g, device=DEV) * 6.28
    return st, wq, wk, torch.cat([th.cos(), th.sin()], -1)


def _attn(pad):
    impl = CuDNNAttentionImpl(num_heads=H // WS, head_size=D, softmax_scale=D**-0.5)

    def run(q, k, v):
        md = AttentionMetadata(extra={"valid_kv_length": q.shape[1] - pad})
        return impl.forward_cuda(q, k, v, md)

    return run


# --- producers -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", [4096, 4100])
@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
def test_upstream_producer_per_destination_bitwise(m, table_dtype):
    q, k, _, wq, wk, table = _rope_inputs(m, m, table_dtype)
    rq, rk = fq.fused_qk_norm_rope(q, k, wq, wk, table, EPS)  # the stock producer (Triton path on CUDA bf16)
    sq, sk = (torch.full(ux.send_shape(m, H, D, WS), float("nan"), dtype=q.dtype, device=DEV) for _ in range(2))
    ux.rope_into_send(sq, sk, q, k, wq, wk, table, EPS, nc._triton_launch)
    for ref, got in ((rq, sq), (rk, sk)):
        stock_send = torch.empty_like(got)
        ux.pack_into(stock_send, ref, WS)  # comm.py:44's copy
        assert _bits_equal(got, stock_send)


@pytest.mark.parametrize("m", [4096, 4100])
@pytest.mark.parametrize("with_bias", [False, True])
def test_consumer_fused_producer_per_destination_bitwise(m, with_bias):
    if not fc._use_triton(torch.empty(1, device=DEV)):
        pytest.skip("fp8_consumers Triton kernels unavailable")
    st, wq, wk, table = _stash(m, 3 * m + with_bias, with_bias)
    out_q, out_k, v = fc.qk_norm_rope_fused(st, H, H, D, wq, wk, table, EPS)  # the serving producer
    sends = [torch.full(ux.send_shape(m, H, D, WS), float("nan"), dtype=torch.bfloat16, device=DEV) for _ in range(3)]
    st2 = fc.Stash(st.x32, st.sa, st.sb, st.bias, torch.empty_like(st.ph))
    ux.fused_into_send(fc, st2, H, D, wq, wk, table, EPS, sends)
    for name, ref, got in zip("qkv", (out_q, out_k, v), sends):
        stock_send = torch.empty_like(got)
        ux.pack_into(stock_send, ref, WS)
        assert _bits_equal(got, stock_send), name


# --- the exchange with cuDNN, two thread-ranks -----------------------------------------------------------------------


def _stock_layer(r, ins, attn):
    q, k, v = ins[r]
    qh, kh, vh = (comm.all_to_all_4D(t.unsqueeze(0), 2, 1, group=None) for t in (q, k, v))  # pre_attention, strict
    return comm.all_to_all_4D(attn(qh, kh, vh), 1, 2, group=None)  # post_attention


@pytest.mark.parametrize("producer", ["pack", "upstream", "fused"])
@pytest.mark.parametrize("back", ["exact", "stock"])
def test_two_ranks_with_cudnn_equal_upstream_stock(a2a, producer, back):
    m, pad = 2048, 37
    attn = _attn(pad)
    if producer == "fused":
        if not fc._use_triton(torch.empty(1, device=DEV)):
            pytest.skip("fp8_consumers Triton kernels unavailable")
        stashes = [_stash(m, 90 + r) for r in range(WS)]
        ins = []
        for st, wq, wk, table in stashes:
            oq, ok_, v = fc.qk_norm_rope_fused(fc.Stash(st.x32, st.sa, st.sb, st.bias, torch.empty_like(st.ph)), H, H, D,
                                               wq, wk, table, EPS)
            ins.append((oq, ok_, v))
    else:
        raw = [_rope_inputs(m, 80 + r) for r in range(WS)]
        ins = [(*fq.fused_qk_norm_rope(q, k, wq, wk, table, EPS), v) for q, k, v, wq, wk, table in raw]
    ref = a2a.run(lambda r: _stock_layer(r, ins, attn))

    def ours(r):
        q, k, v = ins[r]
        sends = None
        if producer == "upstream":
            rq, rk, rv, wq, wk, table = raw[r]
            sends = [torch.empty(ux.send_shape(m, H, D, WS), dtype=q.dtype, device=DEV) for _ in range(2)]
            ux.rope_into_send(sends[0], sends[1], rq, rk, wq, wk, table, EPS, nc._triton_launch)
            q, k = ux._placeholder(q), ux._placeholder(k)
        elif producer == "fused":
            st, wq, wk, table = stashes[r]
            sends = [torch.empty(ux.send_shape(m, H, D, WS), dtype=q.dtype, device=DEV) for _ in range(3)]
            ux.fused_into_send(fc, st, H, D, wq, wk, table, EPS, sends)
            q, k, v = ux._placeholder(q), ux._placeholder(k), torch.full_like(v, float("nan"))
        return ux.exact_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn, ws=WS, sends=sends, back=back)

    a2a.calls = [0] * WS
    got = a2a.run(ours)
    assert a2a.calls == [4, 4]
    for r in range(WS):
        assert got[r].is_contiguous()
        assert _bits_equal(got[r], ref[r]), f"rank {r}"


def test_verify_forward_on_gpu(a2a):
    m, pad = 2048, 5
    attn = _attn(pad)
    raw = [_rope_inputs(m, 60 + r) for r in range(WS)]
    ins = [(*fq.fused_qk_norm_rope(q, k, wq, wk, table, EPS), v) for q, k, v, wq, wk, table in raw]
    ref = a2a.run(lambda r: _stock_layer(r, ins, attn))
    got = a2a.run(lambda r: ux.verify_forward(*(t.unsqueeze(0) for t in ins[r]), attn, ws=WS))
    for r in range(WS):
        res, eq = got[r]
        assert eq == {"q": True, "k": True, "v": True, "attn": True, "result": True}
        assert _bits_equal(res, ref[r])


# --- the nocopy failure analysis (2026-11-03): neither suspected GPU effect differs ---------------------------------


@pytest.mark.parametrize("s", [4096, 8192])
def test_cudnn_on_strided_packed_views_is_bitwise_contiguous(s):
    hl = H // WS
    packed = torch.randn(s, 3, hl, D, device=DEV, dtype=torch.bfloat16)
    qs, ks, vs = (packed[:, i].unsqueeze(0) for i in range(3))  # row stride 3*hl*d: nocopy's views
    qc, kc, vc = (t.contiguous() for t in (qs, ks, vs))
    for used in (s, s - 37):
        outs = []
        for q, k, v in ((qc, kc, vc), (qs, ks, vs)):
            q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k[:, :used], v[:, :used]))
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                outs.append(torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=D**-0.5))
        assert _bits_equal(outs[0], outs[1])


def test_nocopy_exchange_is_bitwise_too(a2a):
    m, pad = 2048, 37
    attn = _attn(pad)
    raw = [_rope_inputs(m, 70 + r) for r in range(WS)]
    ins = [(*fq.fused_qk_norm_rope(q, k, wq, wk, table, EPS), v) for q, k, v, wq, wk, table in raw]
    ref = a2a.run(lambda r: _stock_layer(r, ins, attn))

    def ours(r):
        q, k, v, wq, wk, table = raw[r]
        send = torch.empty(nc.send_shape(m, H, D, WS, 1), dtype=q.dtype, device=DEV)
        nc.rope_into_send(send, q, k, wq, wk, table, EPS, WS, 1, nc._triton_launch)
        return nc.nocopy_forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn, ws=WS, groups=1, a2a=a2a,
                                 send_box=[send])

    got = a2a.run(ours)
    for r in range(WS):
        assert _bits_equal(got[r], ref[r]), f"rank {r}"
