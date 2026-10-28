# Repeats after 16 h of continuous rendering, and a 345-frame window

Two requests repeated on the same boot (Ref2VA partition, loaded 12:13 the day before) to check timing drift and
determinism, then the continuation with `continuation_window_frames: 345` (the upstream maximum) instead of 277.

| Run | First | Repeat (04:24 / 04:42) |
|---|---|---|
| Ref2VA 5 s 1344x768 (first request after the Ref2VA boot) | 1248.6 s, framemd5 80e963c8 | 1026.7 s, framemd5 80e963c8 (identical) |
| Plain continuation 30 s 960x576 | 3615.0 s, framemd5 61a879f4 | 3591.3 s, framemd5 61a879f4 (identical) |

Both repeats are bit-identical to the first runs, so the continuation on the pair is deterministic and there is no
sustained-load slowdown (16 h in, -0.7 %). The first 5 s Ref2VA request was 22 % slower than its repeat: the first
request after a boot pays warm-up, so the reference-size comparison is 753 s (1024, warm) against 1027 s (2048, warm):
-27 %, not -40 %. The two slow outliers of the afternoon (4520 s and 4858 s for 30 s requests) remain unexplained.

| 30 s continuation | window 277, overlap 22 | window 345, overlap 22 |
|---|---|---|
| Windows | [0,277) [255,532) [510,736) | [0,345) [323,668) [646,736) |
| Wall clock | 3591.3 s | 3884.1 s (+8 %; denoising 25 / 29 / 8 min) |
| Peak / min free head | 71.3 / 37.3 GB | 72.0 / 33.5 GB |
| Seams (mean before) | 4.44 (3.96) / 6.33 (3.62) | 2.06 (2.09) / 2.69 (2.51): smooth |
| Audio mean per 10 s | -27.2 / -30.5 / -34.0 dB | -24.8 / -25.7 / -26.5 dB: 1.7 dB fade instead of 7 dB |
| Apparent motion | 6.2 -> 3.6 -> 2.0 | 1.7-2.3 throughout (a calm sample from the first frame; the first window already differs because its length changed) |

One sample: the longer window gave smooth seams and much less audio fade at +8 % time. Worth a 60 s repeat with
generated audio before drawing a conclusion.
