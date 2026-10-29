# The first hand-over, fast loop (16 s = two 277-frame windows, 960x576, seed 1101, ~20 min per render)

Owner, 2026-10-29 17:40: the continuation is not usable: a light and texture step at 12 s and the subject doubled.
Diagnosis with a per-window hook (`h3_tp2/cont_match.py`): upstream renders every window fresh under a latent-tail
guide and switches the output to the new window at the hand-over frame; the regenerated overlap does **not** match the
guide (video latents: std +14 %, one channel's mean shifted by ~1 sigma) and the free frames continue the regenerated
statistics, so the picture steps in exposure and in fine texture. Three levers, measured one at a time and combined:

| Arm | Hand-over | Latent mismatch (regenerated overlap vs guide tail) | Luma step at the seam (frames -3..+3) | Inter-frame change at the seam vs neighbours | Light per second around the seam |
|---|---|---|---|---|---|
| T0 default (window prompts) | hard switch | std 0.66 vs 0.58, max channel shift 0.58 | **+4.2** | 8.09 vs 4.3 / 9.0 (spike) | 49 51 55 / 62 61 61 |
| T3 overlap 56 | hard switch | std 0.67 vs 0.60, max shift 0.28 | +1.4 | 6.47 vs 4.1 / 9.0 | 49 51 55 / 61 60 60 |
| T2 latent match (`H3_TP2_CONT_MATCH=1`) | hard switch, new window affine-matched per channel to the tail | matched: 0.58 / 0.58 | -0.6 | 7.87 vs 4.2 / 7.0 | 49 49 53 / 53 53 52 |
| X1 match + crossfade (`H3_TP2_CONT_XFADE=1`) + later-window prompt without the subject (`H3_TP2_CONT_PROMPTS_REST`) | linear latent crossfade over the 22-frame overlap | matched | **-1.3** | **5.35 vs 5.1 / 5.7 (no spike)** | 52 53 56 / 53 53 53 |

Owner on T0: textures jump at 12 s. Owner on T2: a slight cut remains, textures change a little (exposure fixed, the
hard switch between two renderings of the same frames still shows). X1 removes the spike in the inter-frame change
too; the frames 254/266/278/290 (`x1-frames-254-266-278-290.png`) show one woman walking on through the seam with
the same background and water. Owner's verdict on X1 pending; X2 (overlap 56, a 2.3 s ramp) and a 30 s clip with it
follow. The T1 arm (prompt only) did not run: an unquoted value in `~/.h3-tp2.env` kept the head from starting (fixed:
quoted in later queues).

What the levers are: the match is a per-channel gain/offset of the whole new window's video latents computed on the
hidden overlap (deterministic, no model change, not bit-identical to upstream by design); the crossfade replaces
upstream's hard switch by a ramp over the overlap in latent space (`h3_tp2/plugin.py` 9e keeps a copy of upstream's
loop for it); the later-window prompt describes continuation of what is in the frame instead of repeating the subject.

## Owner on X1 / X3 (20:10, 20:45): a blink at 11 s. Per-frame luma confirms it and the crossfade is withdrawn

Luma per frame (every 2nd, frames 240-300): T2 (no crossfade) 48 48 48 49 ... 53 53, smooth; X1 and X3 (latent
crossfade, plain and variance-preserving) ... 54 55 54 55 **52 55 53 57** 55 55 ...: an oscillation of +-3 at the
latent-frame period inside the ramp. Blending two latent trajectories is not something the VAE decodes smoothly; the
crossfade is withdrawn (code kept behind `H3_TP2_CONT_XFADE`, off).

## X4: the previous tail pinned as target rows (`H3_TP2_CONT_EDIT=1`, upstream's latent-edit path)

Instead of conditioning the new window on a latent-guide block only, the overlap frames of the new window are made
pinned target rows: mask 0 (the model sees the noise-augmented clean rows at the condition timestep, as FL2VA keeps
its keyframe) and restore 0 (the clean rows are restored at the end); audio rows the same way; upstream's guide block
kept (`=2` drops it). 16 s, seed 1101, the later-window prompt without the subject, no match, no crossfade:

| | hand-over diagnostics | luma step (-3..+3) | inter-frame change at the seam vs neighbours | luma per frame 240-300 (every 2nd) |
|---|---|---|---|---|
| X4 | regenerated overlap == guide tail (max channel shift 0.0000) | **+0.3** | **3.99 vs 4.07 / 4.20** | 49 50 50 50 50 51 51 52 52 52 53 53 54 54 54 55 55 55 55 56 56 56 56 57 57 56 56: monotone, no step, no oscillation |

The cleanest hand-over of the day by every number; frames 254/266/278/290 in `x4-frames-254-266-278-290.png`.
**Owner (21:20): no jump noticed, "it was nice"**: the first seam that passed the eye. X5 (guide block dropped), a 30 s and
a 60 s clip follow for the morning.
