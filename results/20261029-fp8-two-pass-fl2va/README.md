# Two-pass FP8 GEMM at the serving cell: FL2VA 1344x768 on vLLM-Omni v0.30.0, first+last-frame task

Both Sparks switched to the FL2VA partition of the v0.30 stack (`select-stack.sh long FL2VA` on both, agent then head),
bench prompt, the 3840x2160 source image as the first frame, 20 steps, seed 1101. Stock arm = vLLM's CUTLASS FP8 GEMM;
two-pass arm = `H3_TP2_FP8_GEMM=tensor32`. The worker's dispatch counter: "two-pass 2000, stock 0" (every FP8 GEMM
went through the two-pass path).

| Cell | Stock: wall (diffusion / decode) | Two-pass: wall (diffusion / decode) | Wall gain | Output |
|---|---|---|---|---|
| 5 s, first request after boot | 853.3 s (798.6 / 45.1) | 537.4 s (485.9 / 43.4) | -37 % | **byte-identical MP4** (framemd5 3f06def6, SSIM 1.0) |
| 5 s, warm | 687.6 s (640.4 / 44.2) | 558.4 s (512.9 / 42.6) | -19 % | **byte-identical MP4** |
| 10 s, warm | 1741.2 s | 1487.9 s (1399.2 / 84.9) | -15 % | frames differ (8fd1b8b8 vs b825141f) |
| 20 s, warm | (no v0.30 stock run; 6740 s wall on e1aa6eae, peak 92.7 GB) | 5630.0 s wall (denoising 5454.3 s), peak 81.5 GB | about -16 % against the e1aa6eae figure | frames not compared (no same-stack baseline) |
| Peak | 62.5-67.5 GB | 63.5-69.5 GB | +1-2 GB | |

The 5 s result is the striking one: a different GEMM kernel (different reduction order, 3.8e-6 on a single GEMM) and
the final MP4 is identical byte for byte, in two separate renders. At this cell the first-frame conditioning pins the
sample firmly enough that the reduction-order noise never reaches the uint8 pixels; at 10 s it does (as it did at
960x576 Ref2VA, where the frames differed with SSIM 0.81). So for the 4-20 s single-window clips this recipe serves,
the two-pass GEMM is lossless in the strongest sense at 5 s and "same sample class" above that, for 15-20 % less time
(37 % on the first request after a boot, which also no longer pays the stock path's warm-up).
