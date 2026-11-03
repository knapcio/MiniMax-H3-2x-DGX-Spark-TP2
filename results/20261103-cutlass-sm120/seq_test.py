import sys, torch
sys.path.insert(0, sys.argv[1])
from h3_tp2 import fp8_gemm, fp8_cutlass
fp8_gemm.probe_triton("cuda"); assert fp8_cutlass.probe("cuda"), fp8_cutlass._state["reason"]
E4M3 = torch.float8_e4m3fn
def q(x):
    s = torch.clamp(x.abs().amax(dim=1, keepdim=True) / 448.0, min=1.0 / (448.0 * 512.0))
    return torch.clamp(x / s, -448, 448).to(E4M3), s
def check(a, w, sa, sb):
    y = fp8_cutlass.scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
    ref = fp8_gemm.scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16)
    torch.cuda.synchronize()
    return torch.equal(y.view(torch.int16), ref.view(torch.int16))

# (A) one buffer reused at the same address with a different shape / stride
g = torch.Generator(device="cuda").manual_seed(1)
buf = torch.empty(4096 * 7168, device="cuda", dtype=torch.uint8)
res = []
for (M, K, N) in [(4096, 5376, 1024), (4096, 7168, 1024), (2048, 7168, 512), (4096, 5376, 1024), (3000, 7168, 2048)]:
    a = buf[: M * K].view(E4M3).view(M, K)
    a.copy_((torch.randn(M, K, device="cuda", generator=g) * 64).clamp(-448, 448).to(E4M3))
    w = (torch.randn(N, K, device="cuda", generator=g) * 64).clamp(-448, 448).to(E4M3)
    sa = torch.rand(M, 1, device="cuda") * 1e-3 + 1e-4; sb = torch.rand(N, device="cuda") * 1e-3 + 1e-4
    res.append(check(a, w, sa, sb))
print("(A) same address, different shapes:", res, flush=True)

# (B) request-like: 3 requests x (2 shapes) x 4 layers; activations fresh per layer; empty_cache between requests
H, M = 5376, int(sys.argv[2]) if len(sys.argv) > 2 else 2048
W = {}
for name, (N, K) in {"qkv": (21504, 5376), "out": (5376, 7168), "fc1": (28672, 5376)}.items():
    w0 = torch.randn(N, K, device="cuda", generator=g) * 0.02
    W[name] = q(w0)
    del w0
seq_bad = []
side = torch.cuda.Stream()
for req in range(6):
    Mr = M if req % 2 == 0 else M + 512
    use_side = req >= 4
    keep = []
    ctx = torch.cuda.stream(side) if use_side else torch.cuda.stream(torch.cuda.current_stream())
    with ctx:
        torch.cuda._sleep(500_000_000)              # the GPU is far behind: the host enqueues every layer ahead
        for layer in range(16):
            x = torch.randn(Mr, H, device="cuda", generator=g).to(torch.bfloat16)
            a, sa = q(x.float())
            wq, sq = W["qkv"]; y = fp8_cutlass.scaled_mm(a, wq.t(), sa, sq.view(-1), torch.bfloat16)
            h = (y[:, :7168].float() * 0.5).to(torch.bfloat16)
            a2, sa2 = q(h.float())
            wo, so = W["out"]; z = fp8_cutlass.scaled_mm(a2, wo.t(), sa2, so.view(-1), torch.bfloat16)
            a3, sa3 = q(z.float())
            wf, sf = W["fc1"]; f = fp8_cutlass.scaled_mm(a3, wf.t(), sa3, sf.view(-1), torch.bfloat16)
            keep.append(((a, wq, sa, sq.view(-1), y.clone()), (a2, wo, sa2, so.view(-1), z.clone()), (a3, wf, sa3, sf.view(-1), f[:, :4096].clone())))
            del x, y, h, z, f
    torch.cuda.synchronize()
    bad = 0
    for trip in keep:
        for (aa, ww, s1, s2, out) in trip:
            ref = fp8_gemm.scaled_mm_tensor32(aa, ww.t(), s1, s2, torch.bfloat16)[:, : out.shape[1]]
            torch.cuda.synchronize()
            bad += int(not torch.equal(out.view(torch.int16), ref.view(torch.int16)))
    seq_bad.append((req, Mr, "side" if use_side else "cur", bad))
    del keep
    torch.cuda.synchronize(); torch.cuda.empty_cache()
print("(B2) async requests (req, M, stream, wrong GEMMs of 48):", seq_bad, "peak GB %.1f" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)
