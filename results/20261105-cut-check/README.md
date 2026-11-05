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
| 5 s probes and calibration (T2VA) | | | | 1.07-1.11 | clean |

Clean windows sit at 1.07-1.30, the two certain cuts at 1.96 and 2.72; 1.7 leaves a margin to both and also flags the
ambiguous 1.87. A false positive costs one re-sample of that window (about 10 min at 960x544, 28 min at 1344x768), a
missed cut costs the clip. One scene, one resolution: the first 1344x768 windows through this check will say whether
the limit holds there (every continued window logs its ratio whether or not it retries).

**Repair of the clip** (`H3_TP2_REPLAY_PARTIAL=1`, queue82): the saved latents of windows 1-8 replayed in seconds, windows
9-11 re-sampled with request seed 3601; see `results/20261105-polish-pacing/`.
