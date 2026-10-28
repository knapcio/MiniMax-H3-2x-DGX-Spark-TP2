"""The fused epilogue + SwiGLU kernel reproduces two-pass + vLLM's silu_and_mul bit for bit (bf16 bit patterns)."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("vllm._custom_ops")          # registers torch.ops._C.silu_and_mul, the reference
from h3_tp2 import fp8_gemm, fp8_swiglu  # noqa: E402

assert hasattr(torch.ops._C, "silu_and_mul"), "vLLM's silu_and_mul must be the reference"


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("M,width", [(64, 256), (300, 28672)])
def test_fused_swiglu_bit_identical_random(M, width, bias):
    torch.manual_seed(0)
    x32 = torch.randn(M, width, device="cuda") * 3000       # scales ~0.005 -> gate values up to ~ +-15: saturation both ways
    sa = torch.rand(M, device="cuda") * 0.01 + 0.001
    sb = torch.rand(width, device="cuda") * 0.01 + 0.001
    bb = (torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1) if bias else None
    ref = fp8_swiglu.scale_swiglu_reference(x32, sa, sb, bb)
    out = fp8_swiglu.scale_swiglu(x32, sa, sb, bb)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert torch.equal(_bits(out), _bits(ref))


def test_fused_swiglu_directed_values():
    """Midpoints, subnormals, +-0, saturation, a masked tail (width not a multiple of the block) and non-finite inputs."""
    width = 2 * 1500                                        # half = 1500: two column tiles, the second masked
    vals = torch.tensor([0.0, -0.0, 1e-40, -1e-40, 1.0, -1.0, 88.0, -88.0, 1e4, -1e4, 0.5000001, 1.9999999,
                         float("inf"), -float("inf"), float("nan")], device="cuda")
    x32 = vals.repeat(1, width // vals.numel() + 1)[:, :width].clone()
    x32 = torch.cat([x32, torch.randn(7, width, device="cuda") * 100])
    sa = torch.ones(x32.shape[0], device="cuda")
    sb = torch.ones(width, device="cuda")
    ref = fp8_swiglu.scale_swiglu_reference(x32, sa, sb, None)
    out = fp8_swiglu.scale_swiglu(x32, sa, sb, None)
    r, o = _bits(ref), _bits(out)
    nan_ref = torch.isnan(ref)
    assert torch.equal(torch.isnan(out), nan_ref)            # same NaN positions
    assert torch.equal(o[~nan_ref], r[~nan_ref])             # everything else bit for bit


def test_stash_hook_takes_fc1_only_when_asked_and_matching_width():
    a = torch.zeros(32, 64, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros(128, 64, device="cuda", dtype=torch.float8_e4m3fn).t()
    sa = torch.ones(32, 1, device="cuda"); sb = torch.ones(128, device="cuda")
    fp8_swiglu._tls.stash_next = False; fp8_swiglu._tls.expect_n = 0
    assert fp8_swiglu._stash_hook(a, b, sa, sb, torch.bfloat16, None) is None
    fp8_swiglu._tls.stash_next = True; fp8_swiglu._tls.expect_n = 256   # a padded fc1 would show a different N
    assert fp8_swiglu._stash_hook(a, b, sa, sb, torch.bfloat16, None) is None
    assert fp8_swiglu._tls.stash_next is False
    fp8_swiglu._tls.stash_next = True; fp8_swiglu._tls.expect_n = 128
    ph = fp8_swiglu._stash_hook(a, b, sa, sb, torch.bfloat16, None)
    assert ph is not None and ph.shape == (32, 128) and ph.dtype == torch.bfloat16
    x32, s_a, s_b, bias = fp8_swiglu._tls.stash
    assert x32.shape == (32, 128) and x32.dtype == torch.float32 and s_a.shape == (32,) and s_b.shape == (128,)
    assert fp8_swiglu._tls.stash_next is False
    fp8_swiglu._tls.stash = None


def test_mlp_forward_clears_state_on_exception():
    """A failing fc1 must not leave a stash or the flag behind."""
    class Boom(Exception):
        pass

    class FC1:
        output_size_per_partition = 128
        def __call__(self, x):
            fp8_swiglu._tls.stash = ("x", "sa", "sb", None)      # as if the hook had fired
            raise Boom()

    class MLP:
        fc1 = FC1(); act_fn = None; fc2 = None
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as tr
    fwd = tr.MiniMaxH3MLP.forward if getattr(tr.MiniMaxH3MLP, "_h3_tp2_fused_swiglu", False) else None
    if fwd is None:
        fp8_gemm._state["triton"] = True
        fp8_swiglu.install()
        fwd = tr.MiniMaxH3MLP.forward
    with pytest.raises(Boom):
        fwd(MLP(), torch.zeros(1))
    assert getattr(fp8_swiglu._tls, "stash", None) is None
    assert fp8_swiglu._tls.stash_next is False and fp8_swiglu._tls.expect_n == 0


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("M,width", [(64, 256), (120, 28672)])
def test_fused_swiglu_quant_matches_vllm_quant(M, width, bias):
    """fused SwiGLU + per-token FP8 quant == (two-pass + silu_and_mul) followed by vLLM's compiled quant, bit for bit."""
    torch.manual_seed(1)
    x32 = torch.randn(M, width, device="cuda") * 3000
    sa = torch.rand(M, device="cuda") * 0.01 + 0.001
    sb = torch.rand(width, device="cuda") * 0.01 + 0.001
    bb = (torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1) if bias else None
    h = fp8_swiglu.scale_swiglu_reference(x32, sa, sb, bb)
    q_ref, s_ref = fp8_swiglu.quant_reference(h)
    q, s = fp8_swiglu.scale_swiglu_quant(x32, sa, sb, bb)
    assert q.shape == q_ref.shape and q.dtype == q_ref.dtype and s.shape == s_ref.shape
    assert torch.equal(q.view(torch.int8), q_ref.view(torch.int8))
    assert torch.equal(s.view(-1), s_ref.view(-1))
