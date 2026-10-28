# The long-clip recipe at 30 s 1344x768: one cut moved inside a window, time about the same

Same request as the 960x576 recipe run (`H3_TP2_REF_SHORT_EDGE=1024`, `CONT_PROMPTS=auto`, `audio_mode: lock_source`
with the 60 s WAV, Ref2VA, 20 steps, seed 1101, windows of 277 with overlap 22), at 1344x768. The 1344 clip and its
plain counterpart (`20261027-v030-continuation-30s-1344/`) are one sample each.

| | plain continuation | recipe |
|---|---|---|
| Wall clock | 10113.9 s | 9805.2 s (-3 %) |
| Denoising per window | 3144 / 3694 / 2938 s | 2690 / 3983 / 2859 s; VAE decode 257 s |
| Peak (X-Peak-Memory-MB) / min free head, worker | 78.8 / 26.1, 35.6 GB | 73.0 / 39.0, 36.8 GB |
| Cuts (ffmpeg scene score > 0.1) | one at seam 1 (frame 277, score 0.28) | one at frame 335 (13.9 s, score 0.26), inside window 2 |
| Seam 1 / seam 2 (mean of 27 frames before) | 30.4 (2.8) hard cut / 4.88 (3.99) | 3.67 (2.92) / 5.96 (4.81): both smooth |
| Apparent motion per window | not measured | 2.6-3.3 / 3.0-5.1 / 4.3-5.9: no decay |
| Audio mean per 10 s | -29.0 / -32.9 / -41.1 dB | -42.2 / -40.8 / -42.2 dB (the track) |
| framemd5 | b83ac4b2 | a3e2c59d |

Frames 334 -> 335: the view jumps from "sun on the left, forest on the right" to "sun in the centre", an unrequested
hard cut 58 frames after the first hand-over. Both hand-overs themselves are smooth, and the last five seconds are a
continuous pan to the right towards the forest and a path (no cut, scene score <= 0.034 elsewhere). So at this
resolution the recipe removed the seam cut of the plain run but the model still cut once on its own; at 960x576 the same
recipe produced no cut (one sample each; the 960x576 plain run also stayed cut-free at the seams while the 1344x768
plain run did not). Motion did not decay and the audio is the given track.

Time: at 1344x768 the reference rows are a smaller share of the sequence (video tokens grow with the frame area, the
reference stays ~1024 px), so the reference-1024 saving is -3 % here against -24 % at 960x576. Window 2 is the
slowest window in both 1344x768 runs (+48 % and +17 % against window 1 with the same frame count); not explained.
Peak 73 GB with 39 GB free on the head: 1344x768 continuation fits with headroom.

Texture: the owner noticed the grass getting worse from about 24 s. Measured as the variance of the Laplacian of the
lower half of one frame per second (grass; higher = more fine detail or more noise): 46-93 in window 1, 58-131 in
window 2, 129-161 at 23-27 s then 89-98 by the end. The crops at 25 s show streaky, repetitive blade texture instead
of the individual blades of window 1, so the third window drifts in texture even though the motion and the audio hold.
The same metric on the 60 s 960x576 clip with the prompt + soundtrack recipe (`20261028-v030-continuation-60s-best/`)
falls from 49-83 in the first 15 s to 6-9 in the last 12 s: the grass there is a blurred green surface by the end. The
continuation therefore has a third decay axis, detail, that neither per-window prompts nor the driving soundtrack
fix; at 30 s 960x576 it stays within 26-79 (no visible loss), at 30 s with window 345 within 69-143.

The music cut at 20 s is the test soundtrack itself: `music60.wav` is the 20 s clip's audio looped three times, and
`lock_source` plays the given file back as it is. A real track has no such seam.

Files kept on spark-06: `~/h3-tp2-results/v030-best30-1344/` (clip, headers, timing.json, memory samples, verify).
