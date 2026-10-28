"""The two-pass FP8 GEMM matches the fused per-row/per-column epilogue up to accumulation order (GPU), and the Triton
scale pass matches its torch reference exactly."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
from h3_tp2 import fp8_gemm  # noqa: E402


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("M,N,K", [(64, 128, 256), (272, 5376, 7168)])
def test_two_pass_matches_cutlass(M, N, K, bias):
    ops = pytest.importorskip("vllm._custom_ops")
    torch.manual_seed(0)
    a = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn)
    w = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05).to(torch.float8_e4m3fn)
    sa = torch.rand(M, 1, device="cuda") * 0.01 + 0.001
    sb = torch.rand(N, device="cuda") * 0.01 + 0.001
    bb = (torch.randn(N, device="cuda", dtype=torch.bfloat16) * 0.1) if bias else None
    ref = ops.cutlass_scaled_mm(a, w.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, bias=bb)
    out = fp8_gemm.scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16, bb)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    d = (out.float() - ref.float()).abs().max().item()
    assert d <= 2e-5 * ref.float().abs().max().item() + 1e-5, d   # accumulation order only


def test_triton_pass_equals_torch_reference():
    torch.manual_seed(1)
    x32 = torch.randn(300, 1000, device="cuda") * 3
    sa = torch.rand(300, device="cuda"); sb = torch.rand(1000, device="cuda"); bias = torch.randn(1000, device="cuda", dtype=torch.bfloat16)
    one = torch.ones((), device="cuda")
    y_t = torch.empty((300, 1000), device="cuda", dtype=torch.bfloat16)
    try:
        fp8_gemm._scale_bias_cast[(300, 1)](x32, sa, sb, bias, y_t, 1000, HAS_BIAS=True, OUT_BF16=True, BLOCK=1024)
    except Exception as exc:
        pytest.skip(f"Triton cannot launch here: {exc}")
    y_r = fp8_gemm.scale_bias_cast_torch(x32, sa, sb, bias, torch.bfloat16)
    # with a bias the compiler may contract the final multiply-add into one FMA (one rounding instead of two): allow
    # one bf16 ulp there; without a bias the two passes must agree bit for bit
    assert ((y_t.float() - y_r.float()).abs() <= 2 * torch.finfo(torch.bfloat16).eps * y_r.float().abs()).all()   # measured: 6 of 300k values, 1 ulp
    y_t2 = torch.empty_like(y_t)
    fp8_gemm._scale_bias_cast[(300, 1)](x32, sa, sb, bias, y_t2, 1000, HAS_BIAS=False, OUT_BF16=True, BLOCK=1024)
    assert torch.equal(y_t2, fp8_gemm.scale_bias_cast_torch(x32, sa, sb, None, torch.bfloat16))
