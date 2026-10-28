# FP8 GEMM kernel choice on the DiT shapes: CUTLASS (vLLM) and cuBLASLt rowwise are the same speed

`scripts/fp8-gemm-bench.py 36864` on spark-06 (GB10, clocks capped at 2200 MHz, vLLM 0.30 / torch 2.13 venv), min of
5 runs, per-token activation scales and per-channel weight scales as in vLLM's FP8 linear path. M = 36864 rows is one
DiT step's token count for a 5 s 960x576 clip on one rank.

| GEMM | N x K | vLLM `cutlass_scaled_mm` | cuBLASLt `torch._scaled_mm` | outputs |
|---|---|---|---|---|
| qkv | 21504 x 5376 | 159.1 ms, 53.6 TFLOPS | 157.5 ms, 54.1 TFLOPS | not bit-identical, max abs diff 3.8e-6 |
| out | 5376 x 7168 | 47.7 ms, 59.6 TFLOPS | 47.5 ms, 59.8 TFLOPS | same |
| fc1 | 28672 x 5376 | 210.4 ms, 54.0 TFLOPS | 209.8 ms, 54.2 TFLOPS | same |
| fc2 | 5376 x 14336 | 93.2 ms, 61.0 TFLOPS | 93.0 ms, 61.1 TFLOPS | same |

The idea from the Fable DiT round (the sm120 CUTLASS kernel untuned against cuBLASLt's rowwise path, which vLLM gates
to ROCm) is closed: 0.3-1 % apart, within noise, and the outputs differ in rounding order so a switch would not be
output-preserving either. Both run at 54-61 TFLOPS on these shapes; whether that is the GB10's FP8 ceiling at 2200 MHz
or a tiling limit is a separate question (the one-step profile in `20261028-dit-step-profile/` says how much of a step
the GEMMs are).

## Scaling variants on the out-projection shape (`scripts/fp8-gemm-bench2.py`, idle GPU, 15:19)

| Variant | Time | TFLOPS | max abs diff vs CUTLASS |
|---|---|---|---|
| CUTLASS per-row / per-column scales, fused epilogue (the recipe) | 48.15 ms | 59.0 | 0 |
| cuBLASLt rowwise, same scales | 47.10 ms | 60.3 | 3.8e-6 |
| cuBLASLt with scalar scales, bf16 out (a different contract, speed reference only) | 14.99 ms | 189.5 | n/a |
| cuBLASLt scalar scales, fp32 out, then the per-row / per-column scales applied in fp32 and cast to bf16 | 34.35 ms | 82.7 | 3.8e-6 |

The per-row-scale epilogue is what makes the FP8 GEMMs slow on the GB10: the same GEMM with scalar scales runs 3.2x
faster (189 TFLOPS). Applying the per-token and per-channel scales after an fp32-output GEMM is the same arithmetic
as the fused epilogue (accumulate in fp32, scale in fp32, round once to bf16) and differs from CUTLASS only in the
accumulation order (3.8e-6, the same as cuBLASLt rowwise); with torch's separate multiply-and-cast pass it is already
29 % faster than the recipe's kernel. A one-pass Triton epilogue (fp32 in, scaled bf16 out) and the other three shapes
are measured in `fp8-gemm-bench3.py` (next idle gap). GEMMs are 47 % of a DiT step, so this is the largest lossless
speed lever found since the pair itself; the plugin switch and the frame-level check follow.
