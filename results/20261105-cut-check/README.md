# Interior cut check of a continued window (2026-11-05)

**What happened.** The 118 s Polish doll scene (queue81, 11 windows of 277 frames, `results/20261105-polish-pacing/`)
passed every seam check (boundary ratio 1.0 in all windows; window 11 was re-sampled once for a cut right after its
prefix, ratio 2.72) and still cut away at 88.7 s, two seconds into window 9, to a close-up of the hands and the letter,
and stayed there to the end. The seam check measures the one transition between the overlap and the first free frame;
a cut later in the window is invisible to it.

**The check.** `cont_match.cut_ratio`: for every free transition of the sampled window the mean |v[t+1]-v[t]| of the
video latents, divided by the median of the free transitions of the same phase (t mod 5). The VAE's 17-frame blocks
give the transitions into and out of a block-start token (5k) their own, larger level (phase means in a clean window:
0.36 / 0.27 / 0.23 / 0.20 / 0.50), so a pooled median either hides a cut that lands on a block transition (this one did:
token 25) or flags the block transitions themselves. The largest ratio is the window's score. Above
`H3_TP2_CONT_CUT_RATIO` (default 1.7, `0` = off) the window is re-sampled with another seed, the same retries and the
same best-attempt rule as the seam check (`H3_TP2_CONT_RETRY`); the journal line is `h3_tp2 cut check window k attempt n
(seed s): peak p at token t median m ratio r`.

**Calibration** on the 18 saved window latents of 2026-11-05 (960x544, Ref2VA 1024, `H3_TP2_SAVE_LATENTS` files on
Spark_06, CPU):

| window | peak | at token | phase median | ratio | what it is |
|---|---|---|---|---|---|
| queue81 w1-w8 | 0.211-0.260 | | 0.195-0.216 | 1.07-1.21 | clean |
| queue81 w9 (14:02) | **0.699** | 25 | 0.356 | **1.96** | the cut-away to the close-up, 2 s after the hand-over; seam ratio 1.0 |
| queue81 w10 | 0.279 | 38 | 0.214 | 1.30 | the close-up continues, no new cut |
| queue81 w11 attempt 1 (14:24) | **0.748** | 6 | 0.276 | **2.72** | the cut right after the prefix the seam check caught (its ratio 2.72) |
| queue81 w11 attempt 2 (14:35) | 0.436 | 38 | 0.234 | 1.87 | the kept re-sample; something moves hard at 111 s inside the close-up |
| queue82 w9 (17:27, seed 3609) | 0.227 | 8 | 0.199 | 1.14 | the repair's first new window: clean |
| queue82 w10 (17:37) | 0.210 | 53 | 0.192 | 1.09 | clean |
| queue82 w11 (17:46, seed 3611) | **0.713** | 53 | 0.214 | **3.33** | **an unseen cut**: the clip cuts to a close-up of the woman at 114 s (scene detector 114 and 116 s, frames checked), seam ratio 1.0; the window was sampled before the check was in serving |
| 5 s probes and calibration (T2VA) | | | | 1.07-1.11 | clean |

Clean windows sit at 1.07-1.30, the three certain cuts at 1.96, 2.72 and 3.33 (the third one a cut the check had not
been tuned on: it was sampled an hour after the calibration and found by the frames first); 1.7 leaves a margin to all
and also flags the ambiguous 1.87. A false positive costs one re-sample of that window (about 10 min at 960x544, 28 min at 1344x768), a
missed cut costs the clip. One scene, one resolution: the first 1344x768 windows through this check will say whether
the limit holds there (every continued window logs its ratio whether or not it retries).

**Repair of the clip** (`H3_TP2_REPLAY_PARTIAL=1`): queue82 replayed the saved latents of windows 1-8 in seconds and
re-sampled windows 9-11 with request seed 3601 (36 min instead of 2.3 h): faces to the end (man 0.80 / woman 0.69 mean),
but window 11 cut away at 114 s (the 3.33 above). queue85 replays windows 1-10 and re-samples window 11 with seed 3701
under the check; see `results/20261105-polish-pacing/`.

**Serving gate (queue83, 2026-11-05 18:09-19:00).** First boot with the check: probe `bf6e85687a6a`, the 16 s 1344x768 FL2VA
continuation gate (seed 1101) gave the digests of every default boot since 2026-11-03 (`200e38721c9e 4138eba5a964`,
framemd5 `e46129c05cc6`), diffuse 1583.7 / 511.0 s, decode 87.1 s. The check only measured: `cut check window 2 attempt 1
(seed 1102): peak 0.1872 at token 23 median 0.1858 ratio 1.01`, the first number at 1344x768 (the 960x544 clean windows
1.07-1.30). Output-preserving by construction and in the gate.
