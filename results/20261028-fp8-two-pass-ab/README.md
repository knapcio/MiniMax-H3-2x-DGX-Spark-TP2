# Two-pass FP8 GEMM in the DiT (`H3_TP2_FP8_GEMM=tensor32`): render A/B

Ref2VA, 5 s 960x576, 20 steps, seed 1101, bench prompt, reference at 2048 px, two renders per arm on one boot each
(the first request after a boot is the warm-up). The switch replaces vLLM's per-row-scaled CUTLASS FP8 GEMM with a
scalar-scale cuBLASLt GEMM (fp32 output) plus one Triton pass that applies the per-token and per-channel scales, the
bias and the rounding (`h3_tp2/fp8_gemm.py`; micro-bench in `20261028-fp8-gemm-bench/`).

| | stock (CUTLASS fused epilogue) | two-pass |
|---|---|---|
| Wall clock, first request after boot | 771.1 s | 437.8 s (-43 %) |
| Denoising, first request | 723.3 s | 392.8 s |
| Wall clock, warm repeat | 592.3 s | 451.6 s (-24 %) |
| Denoising, warm repeat | 550.2 s | 413.2 s (-25 %) |
| Peak (X-Peak-Memory-MB) | 62.9 GB | 64.7 GB (+1.8 GB: the fp32 GEMM output) |
| framemd5 | 85437ffa (both stock runs identical) | 8e58f759 (both two-pass runs identical) |
| SSIM / PSNR vs stock | | 0.809 / 24.8 dB |

Frames 30 and 90 side by side: the same scene, light, grass detail and sharpness; the composition differs (sun
position, tree placement) as it does between the pair and one Spark (SSIM 0.906) or between two seeds (0.776). The two
paths use the same operations (fp32 accumulation, fp32 scales and bias, one rounding) in different kernels, so one GEMM
differs by its reduction order (3.8e-6 on the DiT shapes) and twenty diffusion steps amplify that into a visibly
different clip of the same scene. Measured: SSIM 0.81 against stock, two inspected frames of the same look and detail;
nothing more general than that is established. Not bit-identical to the stock output; deterministic (the two two-pass renders match bit for bit).

The honest headline is the warm pair: -24 % wall clock, -25 % denoising. The first-request pair shows -43 % because the
stock kernel path pays a 30 % warm-up on its first request after a boot (723 vs 550 s) that the two-pass path does not
(393 vs 413 s): the slow part of a cold first request appears to be in the CUTLASS FP8 path itself. GEMMs are 47 % of a
step in the profile, so a 25 % step saving means the GEMM phase got about twice as fast, consistent with the micro-bench.
