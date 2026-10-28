# Reference short edge 768 (`H3_TP2_REF_SHORT_EDGE=768`): -9 % against 1024, diminishing

Ref2VA, 5 s 1344x768, 20 steps, seed 1101, bench prompt, the same 3840x2160 reference; first request after a boot
(like the 1024 run it is compared with, so the warm-up affects both the same way).

| | short edge 2048 (upstream, warm) | 1024 | 768 |
|---|---|---|---|
| Wall clock | 1026.7 s | 752.9 s | 687.3 s |
| Peak | 65.6 GB | 63.2 GB | 62.9 GB |
| SSIM / PSNR vs 2048 | | 0.776 / 24.3 dB | 0.773 / 23.6 dB |
| SSIM / PSNR vs 1024 | | | 0.849 / 27.9 dB |
| framemd5 | 80e963c8 | 7b9e4c16 | d442753b |

Frame 60 of the three clips stacked: the same scene, light, grass detail and sharpness; compositions differ as
between samples (768 and 1024 are closer to each other than either is to 2048). Halving the reference rows again
(1024 -> 768 is 2.3x fewer rows) saves only 9 % more, so the reference rows are no longer the main cost at 1024;
the recommendation stays at 1024 (closer to upstream's 2048, 1366x768 is still at the output size for a 16:9
reference), 768 is a legitimate further step for speed-first runs. One sample each.
