# Per-second metrics of every continuation clip so far (`scripts/clip-metrics.sh`)

All clips: Ref2VA, 20 steps, seed 1101, bench prompt, windows of 277 with overlap 22 unless noted. "detail" is the
variance of the Laplacian of the lower half (the grass) of one frame per second: it falls when the texture blurs and
rises with streaky or noisy texture, and it also falls with motion blur, so read it next to "motion" (mean absolute
luma change between consecutive frames). Windows: 1 = 0-11.5 s, 2 = 11.5-22.2 s, 3 = 22.2-30.7 s (277/22).

| Clip | Size | Motion w1 / w2 / w3 (and on) | Detail w1 / w2 / w3 (and on) | Cuts (scene > 0.1) |
|---|---|---|---|---|
| plain 30 s | 960x576 | 3-7 / 3-4 / 1.3-4 | 46-65 / 32-63 / 19-37 | none |
| plain 60 s | 960x576 | 3-7 / 3-4 / 2.4-4.8 / 2-3.5 / ~2 / 0.8-1.3 | 46-65 / 32-63 / 23-78 / 20-28 / 23-31 / 9-33 | 23.1 s (0.39), 28.0 s (0.16): the dissolves |
| plain 30 s | 1344x768 | 2.4-3 / 3-4.4 / 2.3-4 | 52-90 / 60-107 / 63-124 | 11.5 s (0.28): seam 1 |
| single-take prompt 30 s | 960x576 | 4.5-6.8 / 2.1-5.3 / 1.3-3.7 | 40-70 / 36-67 / 12-34 | none |
| driving soundtrack 30 s | 960x576 | 4.2-5.5 / 4.2-5.3 / 4.6-9.7 | 40-58 / 23-55 / 35-72 | none |
| per-window prompts 30 s | 960x576 | 2.2-6.6 / 2-5.2 / 1.3-8 | 11-44 / 5-39 / 12-79 | none |
| window 345, 30 s | 960x576 | 1.2-1.6 / 1.1-2.0 (two windows) | 69-143 / 86-133 | none |
| prompt + soundtrack 60 s | 960x576 | 4.6-6 / 2.1-4.2 / 1.1-1.6 / 2.4-4.5 / 3.6-4.9 / 2.9-3.4 | 49-83 / 35-80 / 22-31 / 8-34 / 23-39 / 6-11 | none |
| recipe 30 s (ref 1024 + prompts + soundtrack) | 960x576 | 4.1-5.7 / 4.6-6 / 6.6-9.4 | 26-45 / 22-49 / 44-79 | none |
| window 345 + per-window prompts 60 s | 960x576 | 1.3-1.7 / 1.0-2.3 / 1.8-3.8 / 2.6-3.3 / 3.1-4.3 (five windows) | 78-155 / 60-151 / 115-189 / 157-205 / 162-242 (over-sharpened at the end) | 32.0 s (0.22), 44.3 s (0.23); dissolve at 21 s, fog at 7 s |
| ref 1024 only 30 s | 960x576 | 3-8 / 3-5.5 / 2.1-3.4 | 18-50 / 19-49 / 23-33 | none |
| recipe 30 s | 1344x768 | 2.3-2.9 / 2.6-4.8 / 4-5.5 | 46-93 / 58-120 / 89-161 (streaky) | 14.0 s (0.26) |

What this says, with one clip per row:

- Detail falls across windows in the plain 960x576, single-take and per-window-prompt clips and in the 60 s prompt + soundtrack
  clip, whose grass is a blurred surface in the last 12 s (6-11 against 49-83 in the first window, confirmed by eye).
  Motion falls with it in the plain and prompt-only clips; the soundtrack keeps the motion but not the detail. The
  plain 1344x768 clip does not fall within 30 s (52-124).
- The two clips with the reference at 1024 px (the recipe rows) and the window-345 clip do not lose detail within 30 s;
  the 1344x768 recipe clip overshoots instead (streaky grass in window 3, visible by eye). Whether the 1024 reference
  is the reason or the sample is stays open: the queued 60 s runs test whether the recipe keeps its texture for 60 s
  (window 345 + prompts; ref 1024 + prompts + H3 audio), and a plain 30 s run with only the reference at 1024 is queued
  to isolate the reference size.
- The 1344x768 size cut once in both of its clips (seam 1 without the recipe, 14.0 s inside window 2 with it); the
  960x576 clips never cut at a seam.

Raw output: `metrics-all.txt` on spark-06 (`~/h3-tp2-results/`); the script needs ffmpeg and the H3 venv's numpy + PIL.
