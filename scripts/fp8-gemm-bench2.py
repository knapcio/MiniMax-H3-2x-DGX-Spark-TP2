"""FP8 GEMM scaling variants on one DiT shape (out projection, M x 7168 -> 5376), interleaved min-of-N timing:
  cutlass   = vLLM cutlass_scaled_mm with per-token (M,1) and per-channel (N,) scales, fused epilogue (what the recipe runs)
  rowwise   = cuBLASLt torch._scaled_mm with the same per-row/per-column scales
  tensor    = torch._scaled_mm with scalar scales (a different numeric contract: NOT what the model needs, speed reference)
  tensor32  = scalar-scale GEMM with fp32 output, then the per-row/per-column scales applied in fp32 and cast to bf16:
              the same math as cutlass's fused epilogue, so the max abs difference against cutlass is reported.
Run on an idle GPU."""
import sys, torch
from vllm import _custom_ops as ops

M = int(sys.argv[1]) if len(sys.argv) > 1 else 36864
N, K = 5376, 7168
dev = "cuda"
torch.manual_seed(0)
a = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn)
b = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05).to(torch.float8_e4m3fn)
sa = (torch.rand(M, 1, device=dev, dtype=torch.float32) * 0.01 + 0.001)
sb = (torch.rand(N, device=dev, dtype=torch.float32) * 0.01 + 0.001)
one = torch.ones((), device=dev, dtype=torch.float32)
bt = b.t()
fns = {
    "cutlass": lambda: ops.cutlass_scaled_mm(a, bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16),
    "rowwise": lambda: torch._scaled_mm(a, bt, scale_a=sa, scale_b=sb.view(1, N), out_dtype=torch.bfloat16),
    "tensor": lambda: torch._scaled_mm(a, bt, scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
    "tensor32": lambda: (torch._scaled_mm(a, bt, scale_a=one, scale_b=one, out_dtype=torch.float32) * sa * sb.view(1, N)).to(torch.bfloat16),
}
outs, best = {}, {k: 1e9 for k in fns}
for k, f in fns.items():
    outs[k] = f()
torch.cuda.synchronize()
for _ in range(7):
    for k, f in fns.items():
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.cuda.synchronize(); best[k] = min(best[k], s.elapsed_time(e))
ref = outs["cutlass"].float()
print(f"M={M} N={N} K={K}  sm {torch.cuda.get_device_properties(0).multi_processor_count}")
for k in fns:
    d = (outs[k].float() - ref).abs().max().item() if k != "tensor" else float("nan")
    print(f"  {k:9s} {best[k]:8.2f} ms {2*M*N*K/best[k]/1e9:6.1f} TFLOPS   maxabs vs cutlass {d:.3e}  (ref absmax {ref.abs().max().item():.3e})")
