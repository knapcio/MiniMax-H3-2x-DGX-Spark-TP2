"""H3_TP2_FP8_GEMM=cutlass on the GPU: the CuTe-DSL sm120 FP8 GEMM (h3_tp2/fp8_cutlass.py) against the two-pass path.

Bit for bit (``torch.equal`` on the bf16 bit patterns) against ``fp8_gemm.scaled_mm_tensor32`` on the four DiT shapes at
small M with partial row tiles (per-token / per-channel quantised inputs like the serving path's), on integer inputs
with partial row and column tiles, on a side stream, and with guard elements around the output (no write outside
[M, N]).  Plus fc1's bf16 output through vLLM's ``silu_and_mul`` against fused-SwiGLU level 1, and a negative control
for the install check.  Needs CUTLASS's CuTe DSL (nvidia-cutlass-dsl) in the environment; skipped otherwise.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("triton")
pytest.importorskip("cutlass")
from h3_tp2 import fp8_cutlass, fp8_gemm, fp8_swiglu  # noqa: E402

E4M3 = torch.float8_e4m3fn
SHAPES = {"qkv": (21504, 5376), "out": (5376, 7168), "fc1": (28672, 5376), "fc2": (5376, 14336)}


@pytest.fixture(scope="module", autouse=True)
def _kernel():
    assert fp8_gemm.probe_triton("cuda"), "Triton scale pass must work for the reference"
    assert fp8_cutlass.probe("cuda"), fp8_cutlass._state["reason"]
    yield
    fp8_cutlass._state.update(ok=False, gemm=None)


def _quant_rows(rows, cols, std, g):
    x = (torch.randn(rows, cols, device="cuda", generator=g) * std).to(torch.bfloat16).float()
    s = torch.clamp(x.abs().amax(dim=1) / 448.0, min=1.0 / (448.0 * 512.0))
    return torch.clamp(x / s[:, None], -448.0, 448.0).to(E4M3), s


def _serving_like(M, N, K, seed):
    """Per-token activations with log-normal row magnitudes, per-channel weights (scripts/fp8_epilogue_probe.py)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    a, sa = _quant_rows(M, K, torch.exp(torch.randn(M, 1, device="cuda", generator=g) * 1.5), g)
    w, sb = _quant_rows(N, K, 0.02, g)
    return a, w, sa.view(M, 1), sb


def _bits_equal(x, y):
    return x.shape == y.shape and torch.equal(x.view(torch.int16), y.view(torch.int16))


@pytest.mark.parametrize("M", [200, 1000, 4100])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_dit_shapes_bit_identical(shape, M):
    N, K = SHAPES[shape]
    a, w, sa, sb = _serving_like(M, N, K, seed=M * 7 + N)
    ref = fp8_gemm.scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16)
    y = fp8_cutlass.scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
    assert _bits_equal(y, ref)


@pytest.mark.parametrize("M,N,K", [(1, 128, 64), (200, 400, 512), (333, 272, 1024), (129, 5376, 7168), (8200, 144, 320)])
def test_integer_inputs_partial_tiles_and_guards(M, N, K):
    g = torch.Generator(device="cuda").manual_seed(M + N + K)
    a = torch.randint(-4, 5, (M, K), device="cuda", generator=g).to(E4M3)
    w = torch.randint(-2, 3, (N, K), device="cuda", generator=g).to(E4M3)
    sa = torch.exp(torch.empty(M, device="cuda").uniform_(-9.0, -2.0, generator=g))
    sb = torch.exp(torch.empty(N, device="cuda").uniform_(-9.5, -7.5, generator=g))
    ref = fp8_gemm.scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16)
    guard = 4096                                        # sentinel elements on both sides of the [M, N] output
    buf = torch.full((M * N + 2 * guard,), float("nan"), device="cuda", dtype=torch.bfloat16)
    buf[:guard] = 7.0
    buf[-guard:] = 7.0
    y = buf[guard:guard + M * N].view(M, N)
    fp8_cutlass._state["gemm"].run(a, w, sa, sb, y)
    torch.cuda.synchronize()
    assert _bits_equal(y, ref)
    assert bool((buf[:guard] == 7.0).all()) and bool((buf[-guard:] == 7.0).all())


def test_side_stream():
    a, w, sa, sb = _serving_like(1000, 5376, 7168, seed=5)
    ref = fp8_gemm.scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y = fp8_cutlass.scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
    torch.cuda.current_stream().wait_stream(s)
    assert _bits_equal(y, ref)


def test_not_taken_calls_return_none():
    a, w, sa, sb = _serving_like(64, 256, 128, seed=1)
    bias = torch.zeros(256, device="cuda", dtype=torch.bfloat16)
    assert fp8_cutlass.scaled_mm(a, w.t(), sa, sb, torch.float16) is None
    assert fp8_cutlass.scaled_mm(a, w.t(), sa, sb, torch.bfloat16, bias) is None
    assert fp8_cutlass.scaled_mm(a[:0], w.t(), sa[:0], sb, torch.bfloat16) is None
    big = torch.empty(64 * 128 + 1, device="cuda", dtype=torch.uint8)[1:].view(E4M3).view(64, 128)   # misaligned A
    big.copy_(a)
    assert fp8_cutlass.scaled_mm(big, w.t(), sa, sb, torch.bfloat16) is None


@pytest.mark.parametrize("M", [200, 4100])
def test_fc1_bf16_plus_silu_and_mul_equals_fused_swiglu(M):
    """fc1 at this level: DSL bf16 output + the MLP's own silu_and_mul = fused-SwiGLU level 1 on the fp32 product."""
    pytest.importorskip("vllm._custom_ops")
    N, K = SHAPES["fc1"]
    a, w, sa, sb = _serving_like(M, N, K, seed=11 + M)
    s_a, s_b = fp8_gemm._vec_scales(sa, sb, M, N)
    one = fp8_gemm.unit_scale(a.device)        # a freshly filled scale can be read stale by cuBLASLt (fp8_gemm.unit_scale)
    x32 = torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
    ref = fp8_swiglu.scale_swiglu(x32, s_a, s_b, None)
    h = fp8_cutlass.scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
    out = torch.empty((M, N // 2), device="cuda", dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(out, h)
    assert _bits_equal(out, ref)


def test_install_check_catches_a_one_ulp_error():
    """Negative control: the self-check's comparison fails on one flipped low bit."""
    gemm = fp8_cutlass._state["gemm"]
    a, w, sa, sb = _serving_like(200, 400, 512, seed=3)
    assert fp8_cutlass._check_case(gemm, a, w, sa, sb)

    class OffByOne:
        def run(self, a, w, sa, sb, out):  # noqa: ANN001,ANN201
            gemm.run(a, w, sa, sb, out)
            out.view(torch.int16)[17, 3] ^= 1
            return out
    assert not fp8_cutlass._check_case(OffByOne(), a, w, sa, sb)
    assert fp8_cutlass.selfcheck(gemm, "cuda") == []
