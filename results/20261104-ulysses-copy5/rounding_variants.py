"""Sensitivity of the bitwise check: kernel variants differing only in the scale/reciprocal rounding vs the serving QuantFP8."""
import torch, triton, triton.language as tl
from torch._inductor.runtime import triton_helpers as _th
from torch._inductor.runtime.triton_helpers import math as tl_math
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from h3_tp2 import ulysses_outq as oq
from scripts.ulysses_outq_bench import serving_quant

@triton.jit
def _k(x_ptr, q_ptr, s_ptr, fp8_max, stride_p, stride_m, P: tl.constexpr, SEG: tl.constexpr, P_BLOCK: tl.constexpr,
       BLOCK: tl.constexpr, VARIANT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ps = tl.arange(0, P_BLOCK); cols = tl.arange(0, BLOCK)
    mask = (ps[:, None] < P) & (cols[None, :] < SEG)
    x = tl.load(x_ptr + row * stride_m + ps[:, None].to(tl.int64) * stride_p + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    amax = tl.broadcast_to(_th.max2(_th.max2(tl_math.abs(x), 1), 0), [1])
    if VARIANT == 1:      # x/448 folded into a multiply by fp32(1/448)
        t = amax * 0.002232142857142857
    elif VARIANT == 2:    # IEEE division (div_rn) for the scale
        t = tl.math.div_rn(amax, tl.broadcast_to(fp8_max, [1]))
    else:
        t = amax / fp8_max
    scale = _th.maximum(t, tl.full([1], 4.359654017857143e-06, tl.float32))
    if VARIANT == 3:      # IEEE reciprocal
        inv = tl.math.div_rn(tl.full([1], 1.0, tl.float32), scale)
    else:
        inv = tl.full([1], 1.0, tl.float32) / scale
    y = _th.minimum(_th.maximum(x * inv[:, None], tl.full([1, 1], -448.0, tl.float32)), tl.full([1, 1], 448.0, tl.float32))
    tl.store(q_ptr + row * (P * SEG) + ps[:, None] * SEG + cols[None, :], y.to(tl.float8e4nv), mask=mask)
    tl.store(s_ptr + row + tl.arange(0, 1), scale)

quant = serving_quant()
vals = torch.arange(0, 0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
g = torch.Generator(device="cuda").manual_seed(1)
K, SEG = 7168, 3584
bad = {v: [0, 0] for v in range(4)}
n = 0
for c in range(0, vals.numel(), 4096):
    v = vals[c:c + 4096].float(); m = v.numel(); n += m
    x = ((torch.rand(m, K, device="cuda", generator=g) * 2 - 1) * v[:, None]).to(torch.bfloat16)
    x[torch.arange(m, device="cuda"), torch.randint(0, K, (m,), device="cuda", generator=g)] = v.to(torch.bfloat16)
    recv = x.view(m, 2, 28, 128).transpose(0, 1).contiguous()
    sq, ss = quant(oq.unpack_rows(recv), None, None)
    for var in range(4):
        q = torch.empty((m, K), dtype=torch.float8_e4m3fn, device="cuda"); s = torch.empty((m, 1), device="cuda")
        _k[(m,)](recv, q, s, 448.0, m * SEG, SEG, P=2, SEG=SEG, P_BLOCK=2, BLOCK=4096, VARIANT=var, num_warps=8)
        bad[var][0] += int((s.view(torch.int32) != ss.view(torch.int32)).sum())
        bad[var][1] += int((q.view(torch.uint8) != sq.view(torch.uint8)).any(1).sum())
names = {0: "as shipped (Triton '/', runtime 448)", 1: "amax * fp32(1/448)", 2: "div_rn scale", 3: "div_rn reciprocal"}
for var in range(4):
    print(f"{names[var]}: scales differing {bad[var][0]} of {n}, rows with differing FP8 {bad[var][1]} of {n}")
