# Long clips with the plain agent request (Ref2VA, `long_video` continuation, reference 1024, no per-window prompts, no soundtrack)

The request shape an agent sends (README, "Long clips from a client or an agent"): `task=ref2va`, `long_video=true`,
`long_video_mode=continuation`, one prompt, the forest still as the reference, 960x576, 20 steps, seed 1101. Serving
config of the evening of 2026-10-29: long stack / Ref2VA, `H3_TP2_REF_SHORT_EDGE=1024`, two-pass GEMM + fused SwiGLU
level 1, batched decode off.

| Clip | Wall | Windows (denoising) | Decode | Peak | Cuts | Seams (frame 277 / 532) | Motion per s | Audio mean per 10 s |
|---|---|---|---|---|---|---|---|---|
| 30 s (`long30-forest-s1101`) | 2097.4 s | 640.2 / 735.6 / 561.1 s | 148.5 s | 68.5 GB | none | 9.2 / 8.2 against neighbours 5-11: invisible | 7-12 for 22 s, then 4-6 (calmer, not dead) | -52.8 / -53.0 / -52.8 dB: flat |

Prompt: "An old forest with gnarled oaks and a small stream in the foreground. Leaves and branches sway gently in a
light breeze, the stream ripples and glistens, sunlight flickers through the canopy, and a woman in a white dress
walks slowly along the stream. Steady camera, one continuous take, natural forest ambience." The picture: a slow
travel through the oaks, the stream and the woman appear at 10 s and stay; the water's glints get strong in the last
10 s (the detail metric rises from ~70 to ~550), which is the "glistens" of the prompt taken literally. One sample;
the 60 s clip with the same request follows.
