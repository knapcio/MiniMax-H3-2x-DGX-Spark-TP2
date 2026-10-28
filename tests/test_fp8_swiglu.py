"""The fused epilogue + SwiGLU kernel reproduces two-pass + vLLM's silu_and_mul bit for bit."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
from h3_tp2 import fp8_swiglu  # noqa: E402


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("M,width", [(64, 256), (300, 28672)])
def test_fused_swiglu_bit_identical(M, width, bias):
    torch.manual_seed(0)
    x32 = torch.randn(M, width, device="cuda") * 30          # wide range: silu saturation both ways
    sa = torch.rand(M, device="cuda") * 0.01 + 0.001
    sb = torch.rand(width, device="cuda") * 0.01 + 0.001
    bb = (torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1) if bias else None
    ref = fp8_swiglu.scale_swiglu_reference(x32, sa, sb, bb)
    out = fp8_swiglu.scale_swiglu(x32, sa, sb, bb)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    d = (out.float() - ref.float()).abs()
    n = int((d > 0).sum())
    assert n == 0, f"{n} of {d.numel()} values differ, max {d.max().item():.3e}"


def test_stash_hook_takes_fc1_only_when_asked():
    from h3_tp2 import fp8_gemm
    a = torch.zeros(32, 64, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros(128, 64, device="cuda", dtype=torch.float8_e4m3fn).t()
    sa = torch.ones(32, 1, device="cuda"); sb = torch.ones(128, device="cuda")
    fp8_swiglu._tls.stash_next = False
    assert fp8_swiglu._stash_hook(a, b, sa, sb, torch.bfloat16, None) is None
    fp8_swiglu._tls.stash_next = True
    ph = fp8_swiglu._stash_hook(a, b, sa, sb, torch.bfloat16, None)
    assert ph is not None and ph.shape == (32, 128) and ph.dtype == torch.bfloat16
    x32, s_a, s_b, bias = fp8_swiglu._tls.stash
    assert x32.shape == (32, 128) and x32.dtype == torch.float32 and s_a.shape == (32,) and s_b.shape == (128,)
    assert fp8_swiglu._tls.stash_next is False
    fp8_swiglu._tls.stash = None
