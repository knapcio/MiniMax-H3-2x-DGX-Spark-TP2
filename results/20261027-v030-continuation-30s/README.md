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

Continuity: no scene cut anywhere (max ffmpeg scene score 0.06 over the whole clip); at the two window boundaries the
inter-frame luma difference is 4.07 (neighbours 4.32 / 3.90) and 3.46 (3.77 / 3.54), i.e. the seams are invisible
in the motion signal and in the frames (frames 253..256 and 508..511 inspected). The clip is one continuous camera
flight over the landscape; motion slows from about 6 to 2 luma units per frame between the first and the last window.
Audio level drifts down over the windows: -29.3 / -25.8 / -30.4 / -30.5 / -33.8 / -34.2 dB mean per 5 s (the 5 s
Ref2VA clip: -31.0 dB). One sample; the 60 s render that follows checks whether motion and audio keep decaying.
