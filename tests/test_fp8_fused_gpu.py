"""H3_TP2_FP8_GEMM=fused on the GPU: the one-kernel Triton FP8 GEMM (+ SwiGLU for fc1) against the two-pass path.

Integer-valued fp8 inputs make every fp32 accumulator exact in any summation order, so there the fused kernels must
equal the two-pass path bit for bit (the epilogue, the masks and the tile map are then the only things that can
differ).  On random inputs they agree to reduction-order noise; whether they are bit-identical there depends on
cuBLASLt's K order and MMA instruction and is measured by scripts/fp8_epilogue_probe.py, not asserted here."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("triton")
from h3_tp2 import fp8_gemm, fp8_swiglu  # noqa: E402

E4M3 = torch.float8_e4m3fn
ONE = None


def _one():
    global ONE
    if ONE is None:
        ONE = torch.ones((), device="cuda")
    return ONE


def _int_ops(M, N, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randint(-4, 5, (M, K), device="cuda", generator=g).to(E4M3)
    w = torch.randint(-2, 3, (N, K), device="cuda", generator=g).to(E4M3)
    return a, w


def _scales(M, N, seed):
    g = torch.Generator(device="cuda").manual_seed(seed + 100)
    sa = torch.exp(torch.empty(M, 1, device="cuda").uniform_(-9.0, -2.0, generator=g))
    sb = torch.exp(torch.empty(N, device="cuda").uniform_(-9.5, -7.5, generator=g))
    return sa, sb


@pytest.fixture(autouse=True)
def _enable():
    assert fp8_gemm.probe_triton("cuda"), "Triton scale pass must work for the reference"
    saved = dict(fp8_gemm._state)
    fp8_gemm._state["fused"] = True
    yield
    fp8_gemm._state.update(saved)


@pytest.mark.parametrize("cfg", [None, (64, 64, 128, 4, 3), (128, 256, 64, 8, 3)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M,N,K", [(200, 400, 512), (333, 272, 464), (1024, 5376, 7168)])
def test_fused_bit_identical_on_exact_accumulators(M, N, K, out_dtype, bias, cfg):
    a, w = _int_ops(M, N, K, 0)
    sa, sb = _scales(M, N, 0)
    bb = (torch.randn(N, device="cuda") * 0.1).to(out_dtype) if bias else None
    ref = fp8_gemm.scaled_mm_tensor32(a, w.t(), sa, sb, out_dtype, bb)
    y = torch.empty_like(ref)
    try:
        fp8_gemm.launch_fused(a, w.t(), sa.view(-1).contiguous(), sb, bb, y, cfg)
    except Exception as exc:  # noqa: BLE001  (a tile that does not fit this GPU's shared memory)
        if cfg is not None and "OutOfResources" in type(exc).__name__:
            pytest.skip(f"tile {cfg}: {exc}")
        raise
    assert torch.equal(y.view(torch.int16), ref.view(torch.int16))


def test_fused_raw_accumulator_exact_on_integer_inputs():
    a, w = _int_ops(300, 640, 1024, 1)
    x32 = torch._scaled_mm(a, w.t(), scale_a=_one(), scale_b=_one(), out_dtype=torch.float32)
    y = torch.empty_like(x32)
    fp8_gemm.launch_fused(a, w.t(), torch.ones(300, device="cuda"), torch.ones(640, device="cuda"), None, y)
    assert torch.equal(y, x32)


@pytest.mark.parametrize("M,N,K", [(272, 5376, 7168), (512, 21504, 5376)])
def test_fused_random_within_reduction_noise(M, N, K):
    torch.manual_seed(0)
    a = (torch.randn(M, K, device="cuda") * 100).clamp(-448, 448).to(E4M3)
    w = (torch.randn(N, K, device="cuda") * 100).clamp(-448, 448).to(E4M3)
    sa, sb = _scales(M, N, 2)
    ref = fp8_gemm.scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16)
    out = fp8_gemm.scaled_mm_fused(a, w.t(), sa, sb, torch.bfloat16)
    d = (out.float() - ref.float()).abs().max().item()
    assert d <= 2 ** -7 * ref.float().abs().max().item(), d     # reduction-order noise: at most ~1 bf16 ulp at the top
    same = (out.view(torch.int16) == ref.view(torch.int16)).float().mean().item()
    print(f"\nM={M} N={N} K={K}: {same * 100:.4f} % of outputs bit-identical to two-pass, max abs diff {d:.3e}")


@pytest.mark.parametrize("M,half,K", [(200, 320, 512), (333, 1500, 464), (512, 14336, 5376)])
def test_gemm_swiglu_bit_identical_on_exact_accumulators(M, half, K):
    a, w = _int_ops(M, 2 * half, K, 3)
    g = torch.Generator(device="cuda").manual_seed(4)
    sa = torch.rand(M, device="cuda", generator=g) * 0.02 + 1e-3      # gate values over the whole silu range
    sb = torch.rand(2 * half, device="cuda", generator=g) * 0.02 + 1e-3
    x32 = torch._scaled_mm(a, w.t(), scale_a=_one(), scale_b=_one(), out_dtype=torch.float32)
    ref = fp8_swiglu.scale_swiglu(x32, sa, sb, None)
    out = fp8_swiglu.gemm_swiglu(a, w.t(), sa, sb, None)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def test_install_checks_pass():
    assert fp8_gemm.probe_fused("cuda")
    assert fp8_swiglu.probe_gemm_swiglu("cuda")
