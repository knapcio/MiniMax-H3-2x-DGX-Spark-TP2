# The long-clip recipe, 30 s 960x576: reference short edge 1024 + one prompt per window + driving soundtrack

`H3_TP2_REF_SHORT_EDGE=1024`, `CONT_PROMPTS=auto` (bench prompt expanded to one prompt per window), `audio_mode:
lock_source` with the 60 s WAV, Ref2VA, 20 steps, seed 1101, windows of 277 with overlap 22.

| | plain continuation (repeat, same boot age) | recipe |
|---|---|---|
| Wall clock | 3591.3 s | 2725.1 s (-24 %) |
| Denoising per window | ~1140 / 1260 / 1022 s | 855 / 950 / 743 s |
| Peak / min free head | 71.3 / 37.3 GB | 64.1 / 50.1 GB |
| Scene cuts or dissolves | one dissolve at 23 s | none (max scene score 0.036) |
| Seam 1 / seam 2 (mean before) | 4.44 (3.96) / 6.33 (3.62) | 4.66 (4.23) / 8.78 (7.93): smooth |
| Apparent motion per window | 6.2 -> 3.6 -> 2.0 | 3.8-5.2 / 4.2-5.4 / 5.4-8.3: lively to the end |
| Audio mean per 10 s | -27.2 / -30.5 / -34.0 dB | -42.3 / -40.8 / -42.2 dB (the track) |

Frames at 0 / 8 / 17 / 21 / 25 / 30 s: one continuous flight over the misty meadow at sunrise, sharp and consistent.
In this sample the three settings together give the cleanest 30 s clip of the series and the fastest one: the
reference rows are a quarter of upstream's, the per-window prompts keep the shot and the motion, the soundtrack keeps
the audio flat. The audio is the given track; for generated audio see the window-345 and per-window-prompt results.
