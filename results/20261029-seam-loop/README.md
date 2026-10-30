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

## X5: the same without upstream's guide block (`H3_TP2_CONT_EDIT=2`)

| | luma step (-3..+3) | inter-frame change at the seam vs neighbours | light per second around the seam | window 2 denoising |
|---|---|---|---|---|
| X4 edit=1 (guide kept) | +0.3 | 3.99 vs 4.07 / 4.20 | 51 55 / 56 55 55 | 312 s |
| X5 edit=2 (guide dropped) | +0.2 | 4.13 vs 3.93 / 4.27 | 51 54 / 56 55 55 | 255 s (-18 %: 3780 fewer rows in the sequence) |

With the tail pinned as target rows the guide block adds nothing measurable; dropping it makes every window after the
first cheaper. The 30 s and 60 s clips of the night run with edit=1 first (queue25/26) and edit=2 after the
calibration (queue27) for the morning comparison.

## 30 s with the pinned prefix (edit=1): the model cut right after the prefix at the first seam

`edit30-guide`, 3 windows, seed 1101, 2740 s (window 1 967 s: a slow first request; windows 2-3 770 / 560 s), peak
68.5 GB, audio flat (-50..-51 dB), motion 6-9 throughout (no decay), detail 59-232 (no runaway grain), luma 46 -> 69
(the wash-out remains).

| Seam | hand-over diagnostics | inter-frame change at the seam vs neighbours | luma step | scene score |
|---|---|---|---|---|
| 277 (11.5 s) | regenerated overlap == tail (pinned) | **24.80** vs 4.68 / 6.60 | +4.6 | **0.164: a hard cut** |
| 532 (22.2 s) | pinned | 7.65 vs 7.30 / 8.88 | +1.0 | none |

Frames 272/276 (window 1) against 277/281 (window 2) in `edit30-frames-272-276-277-281.png`: the woman by the stream,
then a different shot of an oak trunk. With the overlap pinned as target rows, the model still started a new shot
in the first free frame of window 2; in the 16 s render with the same seed and prefix (X4) the same window continued
the shot. So the pinned prefix makes the hand-over exact when the model continues, but whether it continues is a
sampling event per window. Consequence: a per-window seam check in latent space (mean |v[t+1]-v[t]| at the boundary
against the window's median; `H3_TP2_CONT_RETRY`, `H3_TP2_CONT_SEAM_RATIO`) that re-samples the window with another
seed. The 60 s render of the night runs with retries 2 at ratio 3.0 and logs the ratio of every window; the ratios of
the cut and the clean seams calibrate the threshold.

## 60 s with the pinned prefix and the seam check (`edit60-guide`, 22:38-23:50, memory compacted before the boot)

4319 s (windows 658 / 735 / 745 / 735 / 735 / 407 s: the first request is back at 650 s after the compaction), peak
68.8 GB. Seam check per window (latent boundary delta / window median): 0.90, 0.92, 0.94, 0.96, 1.06, no retry
triggered. Pixel side, all five seams:

| Seam | inter-frame change at the seam vs neighbours | luma step |
|---|---|---|
| 277 | 5.26 vs 4.04 / 5.53 | +0.7 |
| 532 | 8.07 vs 5.58 / 8.60 | +1.6 |
| 787 | 9.92 vs 7.05 / 11.27 | +1.6 |
| 1042 | 11.58 vs 8.37 / 13.37 | +1.4 |
| 1297 | 11.06 vs 9.35 / 11.23 | +0.8 |

No cut, no seam spike, one woman through six windows (`edit60-sheet.png`), audio -51.5 / -52.0 / -46.5 dB. So the
pinned prefix plus the seam check gives clean hand-overs over a minute; the latent ratio of a continued window sits at
0.9-1.1 and a cut should stand far above (the 30 s cut window was not measured; the threshold 3.0 stays provisional).

What remains is the **drift inside the free frames, window after window**: luma 47 -> 88 over 60 s (+41), the
detail metric 70 -> 1000+ (grain / sparkle in the second half, visible in the sheet), motion 7 -> 15 (shimmer, not
camera). Each window is a fresh sample that starts exactly from the previous tail and then drifts toward the model's
brighter, sharper prior; the drift compounds. Next lever: a per-channel latent look-lock of every window's free frames
to the first window's statistics (`H3_TP2_CONT_LOCK=first`), tested at 30 s.

## Look-lock at 30 s (`lock30-edit1`, 02:43-03:27): the wash-out is gone, a whole-window correction steps at seam 2

`H3_TP2_CONT_LOCK=first` (per-channel affine of every later window's free frames to the first window's latent
mean/std, strength 1.0), pinned prefix, seam retries; 2660 s, no cut, seam ratios 0.88 / 0.93.

| | without lock (`edit30-guide`) | with lock |
|---|---|---|
| light per second | 46 .. 69 (+23 over 30 s) | 44 .. 58, ends at 52: **flat** |
| detail per second | 59 .. 232 | 50 .. 300 (same range) |
| motion per second | 6-9 | 4-10 |
| seam 277 luma step | +4.6 (cut) | -0.2 |
| seam 532 luma step | +1.0 | **-8.5** |

The lock holds the exposure over the clip (`lock30-sheet.png`), but a correction applied uniformly to a whole window
steps the brightness at the next hand-over: the pinned tail of window 2 carries window 2's end-of-window brightness
while window 3's free frames start at the window-1 level. Fix (a9cfe12): the correction ramps from zero at the first
free frame to full at the window's end, so the hand-over stays continuous and the drift is still pulled back within
the window. Rendered again at 30 s (`lockr30-edit1`) and 60 s with edit=2 (`lockr60-edit2`).

## Ramped look-lock at 30 s (`lockr30-edit1`, 03:36-04:18)

| | no lock | uniform lock | ramped lock (full by the window's end) |
|---|---|---|---|
| light per second | 46 .. 69 | 44 .. 58 | 47 .. 65, back to 51 at the end |
| detail max | 232 | 300 | 567 (window 3) |
| motion in the last 8 s | 6-9 | 4-8 | **1.8-3.5 (the shot calms down)** |
| seam 277 / 532 luma step | +4.6 (cut) / +1.0 | -0.2 / **-8.5** | **+0.3 / +0.3** |
| seam ratios | - | 0.88 / 0.93 | 0.81 / 1.16 |
| cuts | 1 | none | none |

The ramp keeps both hand-overs continuous (the owner's test), and still pulls the exposure back by the end of each
window, but a correction that reaches full strength only at the window's end lets the drift run for most of the
window (light 65 at 22 s). A ramp fraction (`H3_TP2_CONT_LOCK_RAMP`, full strength after that fraction of the free
frames, then held; 0.25 = about 2 s) is the next setting. The calmed motion in window 3 is the subject-free
later-window prompt letting the scene settle; with the pinned prefix the original wording (the prompt repeated with
"the same continuous shot goes on") may no longer duplicate the subject, to be tested at 60 s.

## 60 s, pinned prefix without the guide block, seam check, ramped look-lock (`lockr60-edit2`, 04:26-05:30, compacted boot)

3857 s (edit=2 saves a window's worth over edit=1's 4319 s), peak 65.7 GB, seam ratios 0.84 / 0.87 / 0.88 / 0.89 / 1.12
(no retry), no cut.

| | 60 s without lock (`edit60-guide`) | 60 s ramped lock (`lockr60-edit2`) |
|---|---|---|
| light per second | 47 .. 88, rising all the way | 46 .. 66 as a sawtooth per window, ends at 54 |
| detail per second | 70 .. 1084 | 70 .. 462 |
| motion per second | 7 .. 17 (shimmer) | 5 .. 10, the last 5 s calm (2-3) |
| seam luma steps | +0.7 .. +1.6 | +0.8 .. +1.2 |
| audio per 10 s | -51.5 / -52.0 / -46.5 | -51.3 / -50.7 / -40.9 / **-28.6** (the last 20 s loud) |

The look holds over the minute (`lockr60-sheet.png`), the seams are continuous, the grain stays in the first clip's
range. Two open points: the sawtooth (each window drifts up for ~6 s before the ramp pulls it back: a faster ramp,
`H3_TP2_CONT_LOCK_RAMP=0.25`, is in the next 60 s), and the audio level climbing 23 dB over the last two windows,
which the video lock does not touch (the audio rows are pinned at the hand-over only); per-5 s levels below.

Audio mean level per 5 s (dB): without lock -51.6 -51.4 -52.1 -52.7 -51.8 -52.3 -50.3 -50.0 -46.7 -46.4 -39.6 -36.8;
with the video lock -51.4 -51.2 -51.8 -52.5 -50.7 -50.7 -47.3 -47.5 -39.8 -42.4 -29.2 -28.1. The generated ambience
drifts louder from the fourth window on in both; the audio latents get the same ramped lock to the first window
(`H3_TP2_CONT_LOCK_AUDIO=1`, 2026-10-30 05:40) in the next 60 s render.

## The night's result: 60 s with everything on (`lockq60-edit2-defaultprompt`, 06:08-07:15)

Pinned prefix without the guide block (edit=2), seam check with retries (none needed: ratios 0.91 / 0.93 / 0.94 / 0.95 /
1.07), video look-lock to window 1 with a 0.25 ramp, audio look-lock, the DEFAULT later-window wording (the prompt
repeated after "the same continuous shot goes on"); 4023 s, peak 65.8 GB.

| | 60 s plain (yesterday evening) | 60 s, this |
|---|---|---|
| cuts | none (but a duplicated subject and a looping walk by eye) | none |
| seam luma steps | +0.7 .. +1.6 | **+0.3 .. +0.8**, no inter-frame spike at any seam |
| light per second | 47 .. 88 | **43 .. 62**, ends at 49 |
| detail per second | 70 .. 1084 | **55 .. 442** |
| motion per second | 7 .. 17 (shimmer) | **5 .. 9 throughout**, no dying, no shimmer |
| audio per 5 s | -52 .. -37 dB (rising) | **-48 .. -52 dB flat** |

Every axis the owner rejected yesterday (light step and texture change at the seams, duplicated subject, wash-out,
looping, grain, shimmer) is now held by a measurement over a full minute; `lockq60-final-sheet.png` shows one woman by
the stream through six windows. Owner's eye pending (morning). These settings become the continuation defaults.

**Owner, 2026-10-30 07:35, on the 60 s clip: "all the transitions are smooth" (the hand-over problem is closed by
eye), but "it showed the woman 5 or 6 times".** The frame grid every 2 s (`lockq60-final-every2s.png`) shows what
that is: one woman standing by the stream the whole minute, and the camera repeating the same travelling move once
per window, passing a trunk and revealing her again. The default later-window wording ("the same continuous shot
goes on ..., the camera keeps moving the same way: <prompt>") makes every window re-stage the move. The subject-free
wording of the X renders did not re-stage but let the motion die late; a drift wording ("the camera drifts very
slowly in the same direction and never comes back to what it already showed, the woman simply continues what she was
doing, nothing new enters the frame") is in `drift30-rest`.

## Another prompt and seed with the defaults (`gen30-deer-s2202`, 07:17-07:57): seams continuous, the scene freezes

The same still, "a deer walks to the water and drinks, birds flit ...", seed 2202, 30 s, 2407 s. Seam ratios 1.03 /
1.06, seam luma steps +0.2 / -0.0 with no inter-frame spike, no cut, light flat at 82, detail 88-212, audio -42..-58 dB.
But motion per second 3.6 .. 0.5 from 10 s on: with this seed the first window itself goes nearly static (3.2 -> 1.7
by 9 s) and the continuation keeps it so (0.3-0.7 for the last 20 s). The hand-over machinery holds on a different
prompt; what the windows contain is the model's sample, and a window that ends still continues still.
