# 30 s continuation with `continuation_overlap_frames: 39` (vs the default 22), 960x576, 20 steps, seed 1101

Same request as `20261027-v030-continuation-30s` with a 1.6 s latent tail instead of 0.9 s. Windows [0,277) [238,515)
[476,736); output seams at frames 277 and 515.

| | overlap 22 | overlap 39 |
|---|---|---|
| Wall clock | 3615.0 s | 3938.2 s (+9 %) |
| Denoising per window | 1151 / 1272 / 1020 s | 1136 / 1378 / 1253 s |
| Peak / min free head | 71.1 GB / 42.3 GB | 67.0 GB / 37.5 GB |
| Seam 1 inter-frame diff (before-mean) | 4.44 (3.96) | 4.13 (3.96) |
| Seam 2 inter-frame diff (before-mean) | 6.33 (3.62) | 6.03 (3.69) |
| Scene score max | 0.06 (plus a generated dissolve at 23 s) | 0.06, no dissolve in this run |
| Audio mean per 10 s | -27.2 / -30.5 / -34.0 dB | -27.2 / -29.6 / -29.5 dB |

Window 1 is identical to the overlap-22 run (same seed, same window), so the comparison is clean from the second
window on. The longer tail does not remove the motion hand-over at the second seam (the camera still changes speed),
smooths the first seam slightly and happened not to produce a dissolve this time. Not worth 9 % more time by itself;
the default stays 22. The prompt wording test follows.
