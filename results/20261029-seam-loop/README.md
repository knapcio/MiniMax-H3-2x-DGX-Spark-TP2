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
