# First long clip: 30 s by latent-tail continuation (Ref2VA, vLLM-Omni v0.30.0, two Sparks), 960x576, 20 steps, seed 1101

`extra_params`: `{"task": "ref2va", "duration": 30, "long_video": true, "long_video_mode": "continuation"}`, one
reference image (`source-4k.jpg`), bench prompt. Upstream defaults: windows of 277 frames, 22-frame latent-tail
overlap, so 736 frames (30.68 s) came out of three windows [0,277), [255,532), [510,736).

| Metric | Value |
|---|---|
| Wall clock | 3615.0 s (60 min; about 2 min per second of video at this resolution) |
| Denoising per window | 1150.7 / 1272.0 / 1020.4 s; prompt 13.5 s; VAE decode 151.8 s |
| Peak (X-Peak-Memory-MB) | 71.1 GB |
| Min free, head / worker | 42.3 / 44.4 GB |
| Output | 736 frames, 30.675 s, audio present; framemd5 61a879f4, frames sha256 2d3afc72… |

Continuity: no scene cut anywhere (max ffmpeg scene score 0.06 over the whole clip). The output seams are where the
discarded overlap ends, frames 277 (11.5 s) and 532 (22.2 s), not at the window starts. Inter-frame luma difference
around them: 3.71 4.76 [4.44] 3.94 3.72 at 277 (smooth) and 3.23 5.79 [6.33] 4.94 4.33 at 532: the picture stays
continuous (frames 530..534 inspected, no cut) but the camera changes speed and direction at the second seam, which
reads as a transition when watched. What the owner actually spotted at 23 s is a second thing: between frames 552 and
588 (23.0 to 24.5 s, inside window 3, 20 frames after the seam) the model generated a dissolve from the low flight over
the grass to a wide shot with fog and sun. Scene detection does not flag it (max score 0.013 there) because it is a
crossfade, not a cut. The prompt said nothing about a single take; upstream warns that continuation "does not
guarantee seamless cuts". Follow-ups queued: 39-frame overlap, a prompt that asks for one continuous take, 1344x768. The clip is one continuous camera flight over the
landscape; motion slows from about 6 to 2 luma units per frame between the first and the last window. Follow-up:
the same clip with `continuation_overlap_frames: 39` (1.6 s of tail instead of 0.9 s) to see whether the motion
hand-over improves.
Audio level drifts down over the windows: -29.3 / -25.8 / -30.4 / -30.5 / -33.8 / -34.2 dB mean per 5 s (the 5 s
Ref2VA clip: -31.0 dB). One sample; the 60 s render that follows checks whether motion and audio keep decaying.
