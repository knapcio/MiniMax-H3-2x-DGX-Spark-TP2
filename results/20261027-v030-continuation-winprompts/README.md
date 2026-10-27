# 30 s continuation with one prompt per window (`continuation_prompts`), 960x576, 20 steps, seed 1101, overlap 22

Three prompts, one per window, each restating the camera motion ("the camera keeps gliding slowly forward ... the
same shot continues without any cut"); generated audio (no driving track).

| | plain (one prompt) | per-window prompts |
|---|---|---|
| Wall clock | 3615.0 s | 3656.0 s (same; the +25 % seen with the long single prompt was not prompt tokens) |
| Peak / min free head | 71.1 / 42.3 GB | 69.5 / 36.3 GB |
| Scene cuts or dissolves | one dissolve at 23 s | none (max scene score 0.045) |
| Seam 1 / seam 2 inter-frame diff (mean before) | 4.44 (3.96) / 6.33 (3.62) | 4.49 (4.60) / 6.63 (7.11): smooth |
| Apparent motion per window | 6.2 -> 3.6 -> 2.0 | 2.3-5.9 / 1.9-4.8 / 4.5-7.0: no decay |
| Audio mean per 10 s | -27.2 / -30.5 / -34.0 dB | -22.4 / -26.4 / -30.0 dB: still fading |

Per-window prompts keep the motion alive and the shot continuous (one flight over the misty meadow toward the sunrise,
frames at 0 / 8 / 17 / 21 / 25 / 30 s), at no extra cost. The generated audio still loses about 8 dB over 30 s, so
the audio fade is not driven by the text conditioning; the driving-soundtrack mode or the window-level experiments
(seed step, local-time windows) have to address it.
