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
