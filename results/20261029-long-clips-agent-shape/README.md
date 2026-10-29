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

## Owner's review of the first clip, and the server-side per-window prompts

The owner watched the 30 s clip: at 12 s the light brightened oddly (that is the first seam, frame 277), the woman
appeared, then the camera moved right to a second copy of the same woman. Both are the plain-request pattern: one
prompt per window re-introduces the subject in every window and hands over with a bump. The per-window wording that
`scripts/render.sh` applied client-side is now applied by the server when a request sends no `continuation_prompts`
(`h3_tp2/cont_prompts.py`, 9a65815, `H3_TP2_CONT_PROMPTS=off` disables). Same request and seed again:

| Clip | Wall | Windows | Decode | Peak | Cuts | Seam 277 / 532 | Motion per s | Audio per 10 s | By eye |
|---|---|---|---|---|---|---|---|---|---|
| 30 s plain (`long30-forest-s1101`) | 2097 s | 640 / 736 / 561 s | 149 s | 68.5 GB | none | 9.2 / 8.2 | 7-12 then 4-6 | -52.8 / -53.0 / -52.8 dB | light and grain jump at frame 277; a second woman after the seam |
| 30 s per-window prompts (`long30-forest-cp-s1101`) | 2163 s | 663 / 763 / 575 s | 148 s | 68.7 GB | none | 8.8 / 6.7 | 6-11 throughout | -51.6 / -52.1 / -51.8 dB | seam invisible (frames 265-289 uniform); one woman the whole clip |

The seam metric (inter-frame change at the seam against its neighbours) did not see the first clip's bump: it is a
brightness and texture step, not motion. Frames 276/278 of each clip are in `seam1-*.png` here. 60 s with the same
request follows.
