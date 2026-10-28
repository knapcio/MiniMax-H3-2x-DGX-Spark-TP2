"""Pieces of the tensor32 variant on all four DiT shapes (idle GPU): scalar-scale FP8 GEMM with bf16 and fp32 output,
the fp32 -> scaled bf16 pass alone (torch and a Triton one-pass kernel), and the totals against cutlass_scaled_mm."""
import sys, torch, triton, triton.language as tl
from vllm import _custom_ops as ops

M = int(sys.argv[1]) if len(sys.argv) > 1 else 36864
shapes = [("qkv", 21504, 5376), ("out", 5376, 7168), ("fc1", 28672, 5376), ("fc2", 5376, 14336)]
dev = "cuda"

@triton.jit
def _scale_cast(x_ptr, sa_ptr, sb_ptr, y_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0); col0 = tl.program_id(1) * BLOCK
    cols = col0 + tl.arange(0, BLOCK); m = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=m, other=0.0)
    y = x * tl.load(sa_ptr + row) * tl.load(sb_ptr + cols, mask=m, other=0.0)
    tl.store(y_ptr + row * N + cols, y.to(tl.bfloat16), mask=m)

def scale_cast(x, sa, sb):
    y = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16); N = x.shape[1]; B = 2048
    _scale_cast[(x.shape[0], triton.cdiv(N, B))](x, sa, sb, y, N, BLOCK=B)
    return y

def t(fn, it=7):
    fn(); torch.cuda.synchronize(); best = 1e9
    for _ in range(it):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e))
    return best

one = torch.ones((), device=dev, dtype=torch.float32)
print(f"M={M}")
for name, N, K in shapes:
    torch.manual_seed(0)
    a = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05).to(torch.float8_e4m3fn); bt = b.t()
    sa = torch.rand(M, 1, device=dev) * 0.01 + 0.001; sb = torch.rand(N, device=dev) * 0.01 + 0.001
    ref = ops.cutlass_scaled_mm(a, bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    x32 = torch._scaled_mm(a, bt, scale_a=one, scale_b=one, out_dtype=torch.float32)
    y_t = scale_cast(x32, sa.view(-1), sb); y_p = (x32 * sa * sb.view(1, N)).to(torch.bfloat16)
    r = {
        "cutlass": t(lambda: ops.cutlass_scaled_mm(a, bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)),
        "gemm_bf16out": t(lambda: torch._scaled_mm(a, bt, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)),
        "gemm_fp32out": t(lambda: torch._scaled_mm(a, bt, scale_a=one, scale_b=one, out_dtype=torch.float32)),
        "scale_torch": t(lambda: (x32 * sa * sb.view(1, N)).to(torch.bfloat16)),
        "scale_triton": t(lambda: scale_cast(x32, sa.view(-1), sb)),
        "total_triton": t(lambda: scale_cast(torch._scaled_mm(a, bt, scale_a=one, scale_b=one, out_dtype=torch.float32), sa.view(-1), sb)),
    }
    fl = 2 * M * N * K / 1e9
    print(f"  {name:4s} N={N:5d} K={K:5d}  " + "  ".join(f"{k} {v:7.2f}ms" for k, v in r.items()) +
          f"  | cutlass {fl/r['cutlass']:.0f} TF, total_triton {fl/r['total_triton']:.0f} TF ({r['total_triton']/r['cutlass']*100:.0f} % of cutlass time)"
          f"  maxabs triton {(y_t.float()-ref.float()).abs().max().item():.2e} torch {(y_p.float()-ref.float()).abs().max().item():.2e} triton==torch {torch.equal(y_t, y_p)}")
