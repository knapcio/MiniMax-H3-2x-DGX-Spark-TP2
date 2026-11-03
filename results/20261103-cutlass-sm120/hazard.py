import sys, torch
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else ".")
from h3_tp2 import fp8_gemm, fp8_cutlass
mode, pre = sys.argv[1], sys.argv[2]
fp8_gemm.probe_triton("cuda")
if pre == "dsl":
    gm = fp8_cutlass.DslGemm(); assert not fp8_cutlass.selfcheck(gm, "cuda"); fp8_cutlass._state.update(ok=True, gemm=gm)
else:   # same memory traffic through two-pass only
    for (M, N, K) in [(200, 5376, 7168), (4100, 5376, 7168), (4100, 21504, 5376), (200, 28672, 5376)]:
        a = torch.randn(M, K, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn); w = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
        fp8_gemm.scaled_mm_tensor32(a, w.t(), torch.exp(torch.empty(M, 1, device="cuda").uniform_(-9, -2)), torch.exp(torch.empty(N, device="cuda").uniform_(-9.5, -7.5)), torch.bfloat16)
torch.cuda.synchronize(); torch.cuda.empty_cache()
E4M3 = torch.float8_e4m3fn
g = torch.Generator(device="cuda").manual_seed(0)
M, N, K = 200, 21504, 5376
a = (torch.randn(M, K, device="cuda", generator=g) * 64).clamp(-448, 448).to(E4M3)
w = (torch.randn(N, K, device="cuda", generator=g) * 64).clamp(-448, 448).to(E4M3)
if mode == "lt":       # fresh scalar -> cuBLASLt
    one = torch.ones((), device="cuda"); x = torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
    torch.cuda.synchronize(); ok = torch.equal(x, torch._scaled_mm(a, w.t(), scale_a=fp8_gemm.unit_scale(a.device), scale_b=fp8_gemm.unit_scale(a.device), out_dtype=torch.float32))
elif mode == "torch":  # fresh scalar -> plain torch elementwise consumer
    big = torch.randn(M, N, device="cuda"); one = torch.ones((), device="cuda"); y = big * one
    torch.cuda.synchronize(); ok = torch.equal(y, big)
elif mode == "dsl":    # fresh per-row scales -> DSL kernel
    sa = torch.exp(torch.empty(M, 1, device="cuda").uniform_(-9, -2, generator=g)); sb = torch.exp(torch.empty(N, device="cuda").uniform_(-9.5, -7.5, generator=g))
    y = fp8_cutlass.scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
    torch.cuda.synchronize(); ok = torch.equal(y.view(torch.int16), fp8_gemm.scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16).view(torch.int16))
print(f"pre={pre} consumer={mode}: {'ok' if ok else 'WRONG'}", flush=True)
