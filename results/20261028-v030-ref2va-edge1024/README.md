# Reference image resized to a 1024 short edge instead of upstream's 2048 (`H3_TP2_REF_SHORT_EDGE=1024`)

Ref2VA, 5 s 1344x768, 20 steps, seed 1101, bench prompt, the same 3840x2160 reference as `20261027-v030-ref2va`.
Upstream resizes every reference image UP or down to a 2048 short edge before the Qwen vision block and the VAE
(about 7300 VAE reference rows plus the vision tokens join the DiT sequence at every step). With 1024 the reference
rows drop to a quarter.

| | short edge 2048 (upstream) | short edge 1024 |
|---|---|---|
| Wall clock | 1248.6 s | 752.9 s (-40 %) |
| Denoising / decode | 1182.1 / 43.5 s | ~690 / 47 s |
| Peak / min free head | 65.6 / 49.5 GB | 63.2 / 51.7 GB |
| framemd5 | 80e963c8 | 7b9e4c16 |
| SSIM / PSNR between the two clips | | 0.776 / 24.3 dB (a different sample, as between two seeds) |

Frames at 2.5 s and 4.9 s side by side: the same scene type, lighting, sharpness and grass detail; the composition
differs (hedge lines, tree placement) as it does between any two samples. The reference is subject/style
conditioning in Ref2VA, and 1824x1024 is still above the 1344x768 output. This changes the conditioning input, not
the model's arithmetic; it is a promising candidate for the continuation mode (every window pays the reference
rows); validation on continuation and on reference fidelity is pending. For a reference whose fine detail must carry
into the clip, keep 2048.
