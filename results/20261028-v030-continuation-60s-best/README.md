# 60 s continuation with the two remedies: single-take prompt + driving soundtrack (960x576, 20 steps, seed 1101)

Same request as `20261027-v030-continuation-60s` plus the "one single continuous take, no cuts, no transitions"
wording and `audio_mode: lock_source` with the 60 s WAV. Six windows, overlap 22, reference short edge 2048.

| | plain (one prompt, generated audio) | single-take prompt + driving soundtrack |
|---|---|---|
| Wall clock | 7985.6 s | 7313.0 s |
| Peak / min free head | 69.4 / 40.4 GB | 69.2 / 45.4 GB |
| Scene cuts or dissolves | 0.39 at 23.2 s, 0.16 at 28 s, one more at 56 s | none (max scene score 0.056) |
| Seams 277 / 532 / 787 / 1042 / 1297 (mean before) | 4.44 (3.96) / 5.57 (3.62) / 3.45 (3.03) / 2.34 (1.90) / 0.94 (0.87) | 4.29 (4.39) / 1.50 (1.32) / 4.32 (3.91) / 4.56 (4.50) / 2.52 (2.64): all smooth |
| Apparent motion per window | 6.2 -> 3.9 -> 3.4 -> 2.5 -> 1.4 -> 1.0 (monotone decay) | 5.6 / 4.4 -> 1.3 / 1.3 -> 3.3 / 4.1 -> 3.2 / 4.5 -> 2.5 / 2.6: a calm second window, then lively again, no decay |
| Audio mean per 10 s | -27 / -31 / -33 / -37 / -43 / -45 dB | -42 / -41 / -42 / -41 / -42 / -41 dB (the track's level, flat) |

The first 60 s clip without any of the three plain-mode defects. The picture stays one continuous flight over the
misty meadow (frames at 0 / 11 / 22 / 33 / 43 / 54 / 60 s). The cost of the driving track is that the audio is the
given file, not generated to the picture; the per-window prompt mode is the alternative when generated audio is
wanted (motion kept, audio fades slowly).
