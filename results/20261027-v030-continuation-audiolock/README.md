# 30 s continuation with a driving soundtrack (`audio_mode: lock_source`), 960x576, 20 steps, seed 1101, bench prompt

`audio_reference` = a 60 s WAV made from the audio of the 20 s T2VA clip (looped), served from a local HTTP server;
everything else as in `20261027-v030-continuation-30s`.

| | plain | lock_source |
|---|---|---|
| Wall clock | 3615.0 s | 4858.3 s (+34 %; windows 1535 / 1650 / 1432 s vs 1151 / 1272 / 1020; part of it may be sustained-load drift, re-measured below) |
| Peak / min free head | 71.1 / 42.3 GB | 66.0 / 38.8 GB |
| Scene cuts or dissolves | one dissolve at 23 s | none (max scene score 0.045) |
| Seam 1 / seam 2 inter-frame diff (mean before) | 4.44 (3.96) / 6.33 (3.62) | 4.20 (3.90) / 7.12 (6.38) |
| Apparent motion per window (luma change per frame) | 6.2 -> 3.6 -> 2.0 | 4.2-5.2 / 3.9-4.9 / 5.0-7.1: no decay |
| Audio mean per 10 s | -27.2 / -30.5 / -34.0 dB | -42.3 / -40.8 / -42.2 dB (the source's level, flat by construction) |

The driving soundtrack removes the audio fade by construction and, with it, the motion decay and the shot changes:
the clip is one continuous misty-meadow flight with lively grass to the last frame (frames at 0 / 8 / 17 / 21 / 25 /
30 s inspected). This supports the hypothesis that the quiet generated audio tail was dragging the next window's
motion down through the shared latent-tail conditioning. The price is that the audio is no longer generated to the
picture; it is the given track.
