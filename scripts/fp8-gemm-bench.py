"""FP8 GEMM microbench on the H3 DiT shapes: vLLM's cutlass_scaled_mm (what the recipe runs) vs cuBLASLt rowwise
torch._scaled_mm.  Per-token activation scales (M,1) and per-channel weight scales (N,) as in the fp8 linear path.
Reports min time of 5 runs, TFLOPS, and whether the two outputs are bit-identical / max abs diff."""
import sys, torch
from vllm import _custom_ops as ops

M = int(sys.argv[1]) if len(sys.argv) > 1 else 36864
shapes = [("qkv", 21504, 5376), ("out", 5376, 7168), ("fc1", 28672, 5376), ("fc2", 5376, 14336)]
dev = "cuda"
def t(fn, iters=5):
    fn(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); out = fn(); e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e))
    return best, out
print(f"M={M}")
for name, N, K in shapes:
    torch.manual_seed(0)
    a = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn)
    w = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05).to(torch.float8_e4m3fn)   # (N,K) row-major as stored
    sa = torch.rand(M, 1, device=dev, dtype=torch.float32) * 0.01 + 0.001
    sw = torch.rand(N, device=dev, dtype=torch.float32) * 0.01 + 0.001
    b_t = w.t()  # (K,N) column-major view, what cutlass_scaled_mm wants
    def cutlass():
        return ops.cutlass_scaled_mm(a, b_t, sa, sw, torch.bfloat16, None)
    def cublaslt():
        return torch._scaled_mm(a, b_t, scale_a=sa, scale_b=sw.view(1, N), out_dtype=torch.bfloat16)
    flop = 2.0 * M * N * K
    try:
        tc, oc = t(cutlass); print(f"  {name:4s} N={N:5d} K={K:5d}  cutlass  {tc:8.2f} ms  {flop/tc/1e9:6.1f} TFLOPS", flush=True)
    except Exception as ex:
        print(f"  {name} cutlass failed: {type(ex).__name__}: {str(ex)[:120]}"); oc = None
    try:
        tb, ob = t(cublaslt); print(f"  {name:4s} N={N:5d} K={K:5d}  cublasLt {tb:8.2f} ms  {flop/tb/1e9:6.1f} TFLOPS", flush=True)
        if oc is not None:
            d = (oc.float() - ob.float()).abs().max().item()
            print(f"       identical={torch.equal(oc, ob)} maxabs={d:.3e} ref_absmax={oc.float().abs().max().item():.3e}")
    except Exception as ex:
        print(f"  {name} cublasLt failed: {type(ex).__name__}: {str(ex)[:160]}")
    del a, w, sa, sw; torch.cuda.empty_cache()
