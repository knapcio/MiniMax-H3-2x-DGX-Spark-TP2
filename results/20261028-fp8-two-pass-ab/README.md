# Two-pass FP8 GEMM in the DiT (`H3_TP2_FP8_GEMM=tensor32`): render A/B

Ref2VA, 5 s 960x576, 20 steps, seed 1101, bench prompt, reference at 2048 px, two renders per arm on one boot each
(the first request after a boot is the warm-up). The switch replaces vLLM's per-row-scaled CUTLASS FP8 GEMM with a
scalar-scale cuBLASLt GEMM (fp32 output) plus one Triton pass that applies the per-token and per-channel scales, the
bias and the rounding (`h3_tp2/fp8_gemm.py`; micro-bench in `20261028-fp8-gemm-bench/`).

| | stock (CUTLASS fused epilogue) | two-pass |
|---|---|---|
| Wall clock, first request after boot | 771.1 s | 437.8 s (-43 %) |
| Denoising, first request | 723.3 s | 392.8 s |
| Wall clock, warm repeat | 592.3 s | see below |
| Peak (X-Peak-Memory-MB) | 62.9 GB | 64.7 GB (+1.8 GB: the fp32 GEMM output) |
| framemd5 | 85437ffa (both stock runs identical) | 8e58f759 |
| SSIM / PSNR vs stock | | 0.809 / 24.8 dB |

Frames 30 and 90 side by side: the same scene, light, grass detail and sharpness; the composition differs (sun
position, tree placement) as it does between the pair and one Spark (SSIM 0.906) or between two seeds (0.776). The
arithmetic is the fused epilogue's (fp32 accumulation, fp32 scales and bias, one rounding); only the accumulation
order inside the GEMM differs (3.8e-6 on one GEMM, the same as cuBLASLt rowwise against CUTLASS), and twenty diffusion
steps amplify that into a different sample of the same distribution. Not bit-identical to the stock output, same
quality by eye.
