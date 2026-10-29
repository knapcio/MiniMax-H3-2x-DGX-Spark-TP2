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

## 60 s with the same request (`long60-forest-cp-s1101`, 16:01-17:16)

| Wall | Windows | Decode | Peak | Cuts | Seams 277 / 532 / 787 / 1042 / 1297 | Motion per s | Detail per s | Audio per 10 s |
|---|---|---|---|---|---|---|---|---|
| 4469.7 s (74.5 min) | 653 / 769 / 766 / 764 / 767 / 421 s | 290 s | 67.9 GB | none | 9.0 / 8.5 / 10.5 / 12.3 / 11.0 against neighbours 4-18: no step at any seam (frame strips checked) | 6-10 for 30 s, 12-17 at 35-45 s, 13-15 after | 70 at the start, 400-700 in the second half | -51.6 / -52.3 / -51.9 / -51.6 / -51.2 / -50.4 dB: flat |

No cut, no seam step, one woman, flat audio over six windows. The second half is the known texture drift of long
continuations: from about 30 s the foliage and the water take on a grainy, sparkly texture (the detail metric is 5-10x
the opening's) while the composition settles into a near-static shot of the woman by the stream; the high "motion"
numbers there are that shimmer, not camera or subject motion (`long60-window-prompts-sheet.png`, one frame per 5 s).
The window-2 and later denoising times (764-769 s) are 17 % above window 1 (653 s), the second-request slowdown seen
at 1344x768 too. So: 30 s is the clean offer; 60 s renders reliably in this sample but its second half looks worse
than its first, and a 60 s request costs 75 min.

**Owner's review of the 60 s clip (17:40): "a disaster: colour changes and a looping figure."** The metrics used here
(scene-cut score, seam deltas, motion, detail, audio level) saw none of it: colour drift between windows and the
subject repeating her walk in every window are not cuts, not seam steps and not loudness. Two consequences: the
clip-metrics script gets per-second colour means (and the sheets stay the primary check, by eye), and 60 s is off the
offer; the README now says up to 30 s. The per-window prompt cannot stop the action loop (each window renders "a
woman walks along the stream" again); a lever against that would have to describe the *state* reached, not the
action, in the later windows, or carry a longer tail. Open for the next fleet window.

Re-measured with chroma means (clip-metrics `colour U/V per 5 s`): the chroma barely moves in either clip (U 116-120,
V 133-135), so the "colour changes" are not a hue shift; it is the **luma** that climbs steadily, 47 at the start to
83 at 30 s and 88 at 60 s (the `light per second` line, 0-255), window after window: the clip washes out. The 30 s
clip has the same climb (47 -> 83). Light drift is therefore the third decay axis to fix for anything beyond one
window, next to texture and the action loop; a per-window "the light stays as in the opening seconds" sentence
(`H3_TP2_CONT_PROMPTS_EXTRA`) was tested on 10-28 for cuts, not for this, and is in the staged `queue19.sh` (arm B).

**Correction (17:45):** the "seam invisible" reading of the 30 s window-prompt clip was wrong. The owner saw a colour
and texture change at 12 s and the woman doubled; per-frame signalstats around frame 277 give luma 50.9 / 51.4 / 52.3
/ 53.1 / 53.9 -> 64.1 / 65.5 / 65.4 (a +10 step, +19 %, inside six frames) with chroma flat, so the hand-over is a
brightness and texture step that a 300 px frame strip hides and a viewer sees at once. Per-window prompts do not fix
the hand-over. Owner's verdict: the continuation is not usable; serving restored to single-window FL2VA (4-15 s) at
17:35. The chain's metrics need a per-seam luma step (frames -3..+3) and the owner's eye stays the gate.
