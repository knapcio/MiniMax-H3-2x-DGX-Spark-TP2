# 30 s continuation at 1344x768 (Ref2VA, 20 steps, seed 1101, bench prompt, overlap 22)

| Metric | Value |
|---|---|
| Wall clock | 10113.9 s (2.8 h; 5.6 min per second of video, 2.8x the 960x576 cost) |
| Denoising per window | 3144 / 3694 / 2938 s; VAE decode 295 s |
| Peak (X-Peak-Memory-MB) | 78.8 GB |
| Min free, head / worker | 26.1 / 35.6 GB |
| framemd5 | b83ac4b2 |

Seam 1 (frame 277 -> 278) is a hard cut: inter-frame luma difference 30.4 against 2.8 before it, ffmpeg scene score
0.28 (the only one in the clip). Seam 2 is smooth (4.88 vs 3.99). Audio mean level -29.0 / -32.9 / -41.1 dB per 10 s.
The scene stays consistent (sunset over the meadow, frames at 0 / 8 / 17 / 21 / 25 / 30 s), so the cut is a
hand-over failure at the first window boundary, not a shot change by the model: frames 276 and 277 inspected, window
2 starts from a different camera position (low in the grass, sun behind the stalks) while the guide tail showed the
wide view with the forest on the right. The journal shows the guide rows were passed (no warning); the model did not
follow them. One sample at this resolution; the
960x576 runs with the same request never produced a cut at a seam. 26 GB free on the head at peak: 1344x768 fits with
room for one more window-size step.
