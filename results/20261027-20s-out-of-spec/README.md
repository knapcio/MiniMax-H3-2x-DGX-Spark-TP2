# 20 s at 1344x768: outside the model's 4-15 s specification (experiment, 2026-10-27 08:21-10:14)

`H3_TP2_MAX_OUTPUT_SECONDS=20` on the final two-Spark configuration, bench prompt, seed 1101, 20 steps, T2VA.

| Metric | Value |
|---|---|
| Wall clock | 6740.1 s (112 min); phases: prompt 2.0 s, denoising 6553.9 s (328 s per step), VAE decode 173.8 s |
| Peak (X-Peak-Memory-MB) | 92.7 GB |
| Min free, head / worker | 20.6 / 22.4 GB |
| Output | 481 frames, 20.08 s, audio present (mean -41 dB; per 5 s window -43.2 / -40.9 / -40.6 / -40.6 dB) |
| framemd5 / frames sha256 | 644303be… / ee8e255d… |
| Motion (mean luma change per frame, 5 s windows) | 0.37 / 0.36 / 0.34 / 0.38: no die-out in the extrapolated part |
| Ulysses all-to-all share of denoising | 3800 calls, 187.7 s incl. permute copies (2.9 %), 140.2 s inside the collectives (2.1 %), 3.9 TB moved per rank |

Frames at 0 / 5 / 10 / 15 / 17.5 / 20 s were inspected: one coherent scene, no collapse, repetition or sharpness loss in
the last five seconds for this calm landscape prompt. One sample; the DiT extrapolates its temporal RoPE beyond
training here and other prompts (people, fast motion, speech) were not tried. The cap stays at the model's 15 s in the
serving configuration.
