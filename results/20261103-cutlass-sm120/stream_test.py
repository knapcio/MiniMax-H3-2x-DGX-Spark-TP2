"""Is the CuTe-DSL launch ordered on a NON-BLOCKING current torch stream? (producer + _sleep, then DSL, then a consumer)"""
import sys, torch
sys.path.insert(0, sys.argv[1])
from h3_tp2 import fp8_gemm, fp8_cutlass
fp8_gemm.probe_triton("cuda"); assert fp8_cutlass.probe("cuda"), fp8_cutlass._state["reason"]
E4M3 = torch.float8_e4m3fn
g = torch.Generator(device="cuda").manual_seed(0)
M, N, K = 4096, 5376, 7168                       # ~0.6 GB in all
a0 = (torch.randn(M, K, device="cuda", generator=g) * 64).clamp(-448, 448).to(E4M3)
w = (torch.randn(N, K, device="cuda", generator=g) * 64).clamp(-448, 448).to(E4M3)
sa0 = torch.rand(M, 1, device="cuda") * 1e-3 + 1e-4; sb = torch.rand(N, device="cuda") * 1e-3 + 1e-4
ref = fp8_gemm.scaled_mm_tensor32(a0, w.t(), sa0, sb, torch.bfloat16); torch.cuda.synchronize()
s = torch.cuda.Stream()                          # torch side streams are non-blocking
print("side stream", s.cuda_stream, "default", torch.cuda.current_stream().cuda_stream, flush=True)
for mode in ("default", "side"):
    bad = 0
    for trial in range(8):
        ctx = torch.cuda.stream(s) if mode == "side" else torch.cuda.stream(torch.cuda.current_stream())
        with ctx:
            a = torch.zeros_like(a0); sa = torch.zeros_like(sa0)
            torch.cuda._sleep(300_000_000)       # producer still busy when the host launches the GEMM
            a.copy_(a0); sa.copy_(sa0)
            y = fp8_cutlass.scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
            y2 = y.clone()                       # consumer on the same stream right after the GEMM
        torch.cuda.synchronize()
        bad += int(not torch.equal(y2.view(torch.int16), ref.view(torch.int16)))
    print(f"{mode} stream: {bad}/8 wrong", flush=True)
