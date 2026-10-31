# A painting comes alive: FL2VA continuation (the still as the first frame, locked camera)

Owner, 2026-10-30 08:25: the real use is a painting (here Philips Koninck's landscape, cropped from a screenshot,
2316x1320) that comes alive with a locked camera and subtle motion (clouds, water, birds), the soundscape added by
hand. That is FL2VA (frame 0 = the picture), where upstream has no continuation; with the pinned prefix no reference
block is needed, so the continuation now runs on the FL2VA partition: window 1 is the FL2VA request as sent, windows
2+ are t2va requests with the previous tail pinned (plugin 9f), plus the night's defaults (seam check, video and audio
look-lock, a seed per window). 1344x768, 20 steps, seed 1101, prompt "the painting comes alive ... the camera is
locked on a tripod and does not move at all".

| Clip | Wall | Seam (277) | Motion per s | Light per s | Detail per s | Cuts |
|---|---|---|---|---|---|---|
| 16 s (`koninck16-fl2va`, 10:14-10:54) | 2410 s (window 1 1712 s: 1344x768 x 277 frames) | luma step -0.0, inter-frame 0.31 vs 0.23 / 0.21 (nothing), seam ratio 1.00, look-lock a no-op (0.0060 vs 0.0060) | 0.2-0.3 (a locked camera on a painting: the motion is the clouds and the figures) | 115 -> 124 in the first 3 s, then flat 122-124 | 137 -> 218 -> 173 | none |
| 30 s (`koninck30-fl2va`, 10:54-12:18) | 5016 s (three 1344x768 windows of ~25 min) | 277: -0.3, 0.40 vs 0.24 / 0.29; 532: -0.3, 0.40 vs 0.31 / 0.20; ratios 0.98 / 0.98 | 0.2-0.6 | 115 .. 129, a slow sway the lock pulls back (121-129 in windows 2-3) | 137 -> 219 -> 112 | none |

`koninck16-first-last.png`: frame 0 is the painting; frame 380 is the same composition with the clouds moved and the
light a touch warmer (the model brightens the painting's tone as it animates it, +9 luma in the first 3 s, then holds).
`koninck16-grid.png`: one frame every 2 s. The hand-over is invisible by every number; the scene barely drifts, so
the locks have little to correct. The owner later judged the 16 s and default 30 s clips good; a 60 s render follows.
For other exact 24 fps clips, `python3 scripts/h3_contact_sheet.py INPUT.mp4 OUTPUT.png --interval 10` produces a
timestamped, non-overwriting still sheet after checking frame timestamps and source hashes. It helps inspect visual
drift, but individual stills cannot establish smooth motion or cloud-shadow correspondence.

30 s (`koninck30-grid.png`, one frame every 3 s): the picture stays the picture across three windows, the clouds keep
drifting, both hand-overs are invisible in the numbers, audio -39..-40 dB flat. At 1344x768 a window costs ~25 min
(84 min for 30 s); 960x576 would be about half.

## 60 s painting render (18:22)

`koninck60-fl2va` completed HTTP 200 / curl 0 in 13,444.6 s at 1344x768, seed 1101, 20 steps; full decode and the
audio check passed. Native video is 1,450 frames, 60.4167 s. All five latent seam ratios were 0.98–0.99; decoded
luma steps at frames 277 / 532 / 787 / 1042 / 1297 were -0.3 / -0.5 / -0.3 / -0.1 / +0.3, with no cut at scene-score
0.1. Per-second luma ranges 115–131; generated audio averaged -41.6 to -45.8 dB across the six ten-second bands.
These metrics do not settle whether the clouds, shadows and figures look natural, and the owner has not reviewed this
60 s clip. It is at `~/Downloads/h3-prod-smoke/d60_koninck_painting_fl2va_static.mp4`; the 60.000 s copy is
`d60_koninck_painting_EXACT60s.mp4` in the same directory.

An inspection of six stills at 0, 10, 20, 30, 40 and 50 s found a separate long-term drift: the clouds progressively
stretch into pronounced diagonal wisps and the road/foreground becomes much brighter than at the start. The locked
composition persists, but the cloud shapes and lighting do not stay faithful to the original painting. These stills
cannot establish whether ground shadows follow the clouds or how the motion looks in playback; the owner has not
reviewed this clip. Reproduce the still sheet with `ffmpeg -i INPUT -vf
'select=not(mod(n\,240)),scale=672:384,tile=3x2' -frames:v 1 SHEET.png` (24 fps).

Unlike the accepted 16/30 s examples at the same global threshold, this clip has repeated 17-frame global luma drops
at 23.38–24.79 s (three, median -2.85 YAVG) and 44.62–48.17 s (six, median -2.10). The more sensitive regional screen
also flags longer stretches; it flags accepted controls too, so neither screen alone is a visual verdict. A separate
`d60_koninck_painting_regional_deflicker_EXPERIMENT.mp4` applies the same spatial correction the owner preferred on
the 30 s history sample to 38 of 1,450 frames: all other decoded frames, chroma and AAC packets are identical to
the original, and the global >=2 YAVG repeated-drop screen no longer finds a run. Regional candidates remain and this
new copy has not been reviewed by eye. Its exact-60 s variant is
`d60_koninck_painting_regional_EXACT60s_EXPERIMENT.mp4`. The regional screen's raw outputs are
`koninck60-regional.txt` and `koninck60-regional-experiment.txt`.

### Aspect-matched 60 s control (2026-10-31)

The 960x544 combined soft-light/slow-figure prompt completed in 3,198.5 s with HTTP 200, full video/audio decode,
five seam ratios 0.98–0.99, no scene-score >0.1 cuts, and minimum available head/worker memory 49,058/50,067 MiB.
The native result has 1,450 frames / 60.4167 s; the separately verified exact copy has 1,440 frames / 60.000 s,
unchanged retained decoded frames and AAC packets ending with the video. It is
`~/Downloads/h3-prod-smoke/d60_koninck_softlight_slowfig_960x544_EXPERIMENT_EXACT60s.mp4`. The first 255 decoded
frames are byte-identical to the matched 16 s control; its first-window diffusion times were 536.2 and 535.9 s.
This does not establish visual quality or explain the prior 1344x768 slowdown.

The exact 60 s file has seven global >=2 YAVG drops every 17 frames at 55.25–59.50 s (native: eight through 60.21 s)
and regional candidates at other points. Its lower-frame mean ranges 47.24–73.73 gray; the still sheet at
`d60_koninck_960x544_timeline_20261031.png` shows strong changes in ground brightness and a gradual change in cloud
shape. Stills cannot establish how the shadows move relative to the clouds. The prompt alone did not keep the ground
light nearly constant, and playback has not been reviewed.

The separate `d60_koninck_960x544_regional_deflicker_EXPERIMENT_EXACT60s.mp4` changes luma in 140 of 1,440 frames
by at most four levels; chroma, video timing and AAC packets match the exact source. The global two-drop screen no
longer flags a run, but regional candidates remain. Across 38 selected repeated source frames, the median absolute
regional mean change fell from 1.892 to 0.542 grayscale levels; these measurements include possible real motion and
do not replace playback review. The signed report is `h3-regional-transitions-current60-20261031.json`. A second separate copy,
`d60_koninck_960x544_regional_softground_EXPERIMENT_EXACT60s.mp4`, applies the bounded soft-ground experiment after
that correction: 1,437 frames have luma changes up to ten levels, the top 337 luma rows, all chroma and AAC packets
remain identical to the regional source, and video timing matches. At one sample per second the lower-frame gray range
changes from 26.49 to 19.85, and mean absolute deviation from the first frame from 15.01 to 11.24. This attenuates
lighting variations; it does not track cloud shadows and may suppress real light or details. The labelled
`porownanie_cieni_60s.mp4` compares the regional and soft-ground copies with the left audio. Both corrections remain
experiments, not visual approval or proof that flicker and shadows are fixed.

A more invasive, separate `--local-periodic` experiment on this 60 s source follows the dominant regional 17-frame
phase across the clip, applying the same bounded spatial adjustment to five frames at each detected boundary. It
changes 340/1,450 luma frames (289..1432), preserves all chroma and copied AAC, and decodes fully. Over the 69 phase-0
boundaries from frames 289..1445, median frame change in the lower-left/middle/right regions falls from 1.99 / 1.87 /
2.01 to 0.67 / 0.63 / 0.64 at the 336x192 grayscale screen. Some shorter regional spike runs remain; real motion can
also contribute. This may smooth natural motion or details and has **not** been reviewed visually, so it is not a
default or a claim that flicker is fixed. The copy is `d60_koninck_allregional_deflicker_EXPERIMENT.mp4`; its exact
60 s version is `d60_koninck_allregional_EXACT60s_EXPERIMENT.mp4`. The full screen is
`koninck60-allregional-experiment.txt`. The refactored streaming implementation produced the same decoded video and
AAC hashes as the previously reviewed 30 s regional experiment when run in the default mode.

## A 5 s pinned history (`koninck30-hist5s`, window 345 / overlap 124, 12:18-14:30)

The owner asked whether a sliding window with more history would help. With 124 frames (5.2 s) of the previous
window pinned instead of 22 (0.9 s): seams -0.5 / +0.2 luma, inter-frame 0.57 / 0.50 against neighbours 0.3-0.5,
ratios 0.98 / 0.98, no cut, motion 0.3-1.4, light 115 -> 135 in the first (longer) window then 125-131, 7865 s
(1.57x the 22-frame run, 5016 s: fewer new frames per window and 345-frame windows at 1344x768). The earlier 2.6x
claim was an arithmetic error. On this locked-camera example the seams were already inconspicuous with 0.9 s of
history, and the owner reported flicker in the 5 s variant. The default stays 22 frames. Longer history cannot
recover content outside the retained frames.

Owner's eye (16:20): the default 16 s and 30 s clips are good ("all good"); in this 5 s history clip the picture
flickered a little between 15 and 22 s. **Frame-level check:** frames 357..561 (14.88..23.38 s) have 13 consecutive
global luma drops every 17 frames (median -6.10, range -6.21..-2.22 YAVG at 320 px width), each recovering within a
few frames. The 16 s and default 30 s clips have no such sequence at the same >=2 YAVG threshold. The previous
once-per-second brightness values showed slow drift but missed this repeated flicker. Reproduce with
`python3 scripts/luma_flicker.py <clip>`; a global luma test is a diagnostic, not a substitute for visual review.

Owner follow-up (17:40): the second experimental repair, which uses a blurred spatial luminance adjustment instead of
one global offset, **looks better**; the people moved a little too fast and strangely, although the clip was generally
very good. This is an observation of the reviewed clip, not proof that the generative cause has been fixed or that no
motion artifacts remain. The local file is `d30_koninck_history5s_regional_deflicker_EXPERIMENT.mp4` in
`~/Downloads/h3-prod-smoke/`; source preserved. `experimental-regional-deflicker.py` reproduces the experiment with
Pillow and FFmpeg; it uses neighboring frames to estimate a bounded blurred Y-plane correction only on the 42 frames
identified by the existing detector. It can alter real movement or light and is **not a default**.

For the owner's 15–22 s interval, a later signed-regional screen selected 28 source frames with repeated changes in
at least three regions. The median absolute regional mean change at those frames fell from 3.424 to 0.105 grayscale
levels in the preferred copy. This supports the reported improvement but can also include suppressed real motion or
light; it is not a complete visual-quality measure. Full report:
`~/Downloads/h3-prod-smoke/h3-regional-transitions-history30-20261031.json`.

At 336x192 grayscale, median mean absolute frame change at the 13 flicker boundaries in the three lower image regions
is 7.07 / 7.59 / 6.88 in the original, 2.15 / 2.02 / 2.09 after the first global repair, and 1.16 / 1.16 / 1.12
after this spatial experiment; other frame-to-frame changes there are about 0.13–0.19. `regional-flicker.py` reproduces
this clip-specific diagnostic. The new file fully decodes, has 736 video frames and the same audio SHA-256; direct
YUV frame comparison finds 42 changed luma frames, unchanged chroma throughout and unchanged other decoded frames.
The remaining signal and the user's motion feedback call for further visual and model-side testing. A separate
16 s comparison changing only the wording for the figures to very slow small steps completed in 2,332.1 s; it passed
media validation and has no detected repeated global luma drops at the two-drop threshold. The regional screen flags
12.04–16.29 s, which may also include real motion. On sampled stills the figure's path looks similar to the original;
the prompt has not established a clear walking-speed improvement. The local enlarged comparison is
`~/Downloads/h3-prod-smoke/porownanie_ruchu_postaci_16s.mp4`; playback review remains pending.

The broader `scripts/regional_flicker.py` screen compares six regions' frame-to-frame changes against their nearby
baseline, looking for three or more same-phase spikes every 17 frames. In the history5s source, all three lower
regions show a candidate run at frames 357..731 (14.88–30.46 s), extending beyond the 23.38 s global-luma run;
the spatially repaired copy still has a weaker run in those regions. The owner-approved default 16 s and 30 s
painting clips also trigger this sensitive screen in some regions. Thus a flag is **not evidence of an objectionable
flicker**: real scene motion and harmless periodic changes can qualify. It is a way to choose segments for inspection,
not an automatic acceptance threshold, and it cannot establish the source of the periodicity.

For a separate corrected copy with AAC audio, `python3 scripts/verify_luma_repair.py SOURCE REPAIRED --limit 8`
checks the full decoded YUV frames, bounds the largest per-pixel luma change, compares every chroma byte, video-frame
timestamp and AAC packet, and records SHA-256 hashes. On the 960x544 exact-16 s local-periodic experiment it finds
30 changed luma frames of 384, with a maximum change of 5, unchanged chroma and timestamps, and identical AAC packets.
`--unchanged-above 337` additionally verifies that the top 337 luma rows remain identical in every frame. Those
invariants do not measure visual quality: a bounded adjustment can still alter real movement or detail.

The 17-frame period matches the checkpoint's `vae_clip_length=17` and the decoder's 17-frame chunk geometry; that
alignment suggests a temporal decode boundary artifact, but the cause is not established. Crossfade was OFF. The
look-lock is one possible contributor: it adjusts latent mean/std, not decoded luma, so the earlier claim that it
caused an ~10-luma pull-back was unsupported. Candidate `H3_TP2_CONT_LOCK_REF=tail` uses the last quarter of window 1
for both video and audio, but is unmeasured. A matched A/B against the 345-frame/124-overlap control must compare the
17-frame drops, audio, and the owner's eye; separate decode-only tests would help distinguish the decoder from the
latents. Default overlap and lock reference remain unchanged.

To distinguish the two inputs to the decoder's five-frame overlap, `H3_TP2_VAE_BLEND_TRACE=1` enables a diagnostic
on the next head start after this code is installed on both nodes. It logs the old, new and blended approximate luma
for each temporal overlap, from spatially sampled decoded tensors, without intentionally changing the result. Its
frame index is approximate; compare the trace with the rendered frame-level metrics. The hook is off by default and
was CPU-tested on stand-ins. It completed the two-Spark smoke and 16 s runs below, but a same-latent comparison is
still needed to establish whether tracing changes the decoded result.
`python3 scripts/analyze_vae_trace.py VIDEO TRACE` compares the same output's encoded global frame changes with the
overlap samples and lists repeated >=2 YAVG drops. It rejects ambiguous duplicate overlap indices and missing traces.
The indices and sampled luma are approximate; the analyzer also reports exploratory offsets of ±8 frames when
checking a repeated drop against a boundary. Selecting an offset after observing periodic changes can produce a
chance match. It also reports the first-to-second-frame changes in the old, incoming and blended samples beside the
encoded-frame rebound; global averages can miss regional flicker, so the comparison does not prove a cause. Six
synthetic/unit tests pass; the 16 s painting-trace preview is described below.
The report also compares encoded frame-to-frame luma changes across all 17 temporal phases; this is descriptive,
not a significance test or a visual-quality verdict.

The 2026-10-31 5 s T2VA trace smoke emitted six temporal-blend records and completed; its decoded video differed
from the trace-off smoke on a previous boot. Their step-0 tensor digests match (`c2f21d37…`, 14 tensors), but their
final latent digests differ (`2c63a5fc…` vs `f01d2c13…`, two tensors) before the decode-only trace runs. Thus the
different decoded frames cannot isolate any effect of tracing; the tensor digests do not cover all request or runtime
state. On the trace-on 124-frame smoke itself, the analyzer read all six records, found no qualifying global periodic
drop and measured a 0.078-level median absolute difference between its sampled blend luma and encoded luma at the
approximate boundary frames. That is a sanity check on this smoke, not a flicker diagnosis.

The 960x544 painting trace completed at 16:52 on 2026-10-31. A checksum-verified preview compared all 22 recorded
overlaps with its 396-frame encoded result. Two global drops at frames 289 and 306 (-2.23 and -3.63 YAVG) align with
the approximate 17-frame boundaries without an offset. At frame 306, the sampled first-to-second-frame luma rises
by 1.90 in the old overlap, 1.32 in the incoming overlap, and 1.88 after blending; the encoded rebound is 1.11.
The checked `blend` implementation constructs a new tensor and does not mutate its inputs, so these samples do
describe its two inputs. Similar smaller rebounds appear at later sampled boundaries. This is evidence that part of
the boundary-shaped change is present in both decoder chunks before the blend, not proof that the blend or the VAE
causes all visible flicker: sampled and encoded luma differ, indices are approximate, only two drops meet this global
threshold, and real scene changes can contribute. The independent preview report is
`~/Downloads/h3-prod-smoke/h3-vae-trace-painting-preview-20261031.json` (SHA-256
`c7338d2d6a62fb4e30e3b92596ea2848fd100d52782bb3417d9130b0a4133293`). Its checked video SHA-256 is
`fa4153f484132a49dac7149f8b486074d1f2bf34d7d87409befdabb4b0dd1382` and trace SHA-256 is
`cbe8874889c85f9c714f1bcfd5c6414e6d9df252bd9acfd86ad124c7ca741f80`. At the time of the preview the queue
was restoring the normal head. Restoration and trace-off health were verified at 17:00:19 CET; the automatic
collector then verified and saved the original video and trace, and its analysis reproduced the same 22 overlaps
and two global drops. The collected report is `~/Downloads/h3-prod-smoke/h3-vae-trace-comparison-960x544-20261031.json`
(SHA-256 `a92288a10b3650b3f8d2053045c9969416eafef0e28640b8abebd8f853eb0264`).

Across this encoded video, all 23 transitions at frame numbers divisible by 17 are negative (median -0.280 YAVG);
the median changes for the other 16 phases are positive (+0.038 to +0.1555), with negative changes in 0–30.4% of
their transitions. The final such transition has no corresponding trace overlap among the 22 recorded, so the phase
statistics and trace sample counts differ. This strengthens the evidence for a phase-related bias in this clip, but
neither proves a decoder cause nor implies that every change is visible. The separate phase report is
`~/Downloads/h3-prod-smoke/h3-vae-trace-painting-phase-20261031.json` (SHA-256
`9657fad03e837f09da273ac4eb888462fdc30ab224cdc6e09f1e3ce8e2b06efc`).

`scripts/compare_temporal_phases.py SOURCE CANDIDATE --start FRAME --end FRAME` compares all phases between two
fully decoded clips over a half-open frame interval. It checks matching dimensions, average frame rate and decoded
frame count, and verifies both SHA-256 hashes before and after; the frame-index comparison alone does not establish
identical timestamps. Two comparator tests and six analyzer tests pass. The following medians are signed global YAVG
changes at frames divisible by 17, not scores of visible quality:

| Clip and interval | Transitions | Source | Regional correction |
|---|---:|---:|---:|
| Owner's history5s 30 s, 15–22 s | 10 | -6.117 | -0.5225 |
| New 60 s, 55–60 s | 7 | -2.054 | -0.454 |
| Exact 16 s, full clip | 22 | -0.287 | -0.287 |

The owner preferred the regional 30 s copy; the 60 s correction has not been visually reviewed. The 16 s median
is unchanged because the bounded correction targets particular stronger local events, not every phase-0 frame.
Both useful reductions can include suppression of real scene lighting; they are not proof that the decoder alone
caused those changes. The respective local reports are `h3-phase-compare-history30-15to22-20261031.json`,
`h3-phase-compare-current60-55to60-20261031.json` and `h3-phase-compare-original16-20261031.json` in
`~/Downloads/h3-prod-smoke/`.

The first 277-frame window of that 960x544 painting trace took 590.5 s, versus 535.9 s for the earlier matched
16 s combined-prompt control (10.2% longer); the 14-tensor step-0 digests match. The runs used different boots,
and that digest does not cover all request or runtime state. The trace hook operates during decode, so the timing
observation does not establish why denoising differed or how repeatable the difference is.
The first-window output-latent digests also differ before decode (`2c5fb185…` for the earlier control versus
`54aee5ef…` for the trace run, two tensors each). Consequently a later trace-on/off pixel comparison would mix
different diffusion outputs and cannot isolate the trace hook.

A sweep of all 25 locally saved `h3-prod-smoke` MP4s found a second sequence meeting this specific threshold in
`d60_seedstep_per_window.mp4`: frames 1326..1445 (55.25..60.21 s), eight drops every 17 frames, median -4.38 YAVG.
The other 23 clips had no >=2 YAVG run lasting at least three periods. This is a global-luma screen, not proof that
those clips are flicker-free: smaller, local or shorter events can be visible to the owner and miss this threshold.

## Separate audio-level experiment (2026-10-31)

The 960x544 60 s clip's generated ambience is very quiet: -46.68 LUFS integrated and -32.25 dBTP measured with
FFmpeg loudnorm. A separate copy of the regional-deflicker + soft-ground candidate applies a **linear** level change
to target -20 LUFS, with no dynamic-range compression or limiter; the new AAC measures -20.01 LUFS and -5.58 dBTP.
The 1440 video packets, timestamps and codec extradata were verified identical to the source, and both video and
stereo 32 kHz audio start at zero and end at 60.000 s. The source is unchanged. Raising the level may reveal noise;
the sound has not been reviewed by ear and the brightness corrections are still experimental. File:
`~/Downloads/h3-prod-smoke/d60_koninck_960x544_regional_softground_LOUDNESS_EXPERIMENT_EXACT60s.mp4`.

The earlier 30 s regional + soft-ground candidate also has a separate levelled copy:
`d30_koninck_history5s_regional_softground_LOUDNESS_EXPERIMENT_EXACT30s.mp4`. Its audio measured -44.00→-20.00 LUFS,
with -8.51 dBTP after AAC encoding. All 720 video packets, timestamps and codec extradata match the source; the
30.000 s video and 29.984 s audio retain the source timing. Full decode passed. The owner preferred the earlier
regional-deflicker clip, but the subsequent soft-ground and audio changes still need playback review; they do not
correct the reported fast figures.

[`scripts/normalize_ambient_audio.py`](../../scripts/normalize_ambient_audio.py) can create a new opt-in copy and
JSON report for other durations. It refuses silent input, more than 30 dB of gain, peak limiting, unexpected streams,
overwrites and failed media verification; it does not change the generation defaults.

## Playback-speed experiment (2026-10-31)

The owner reported that the people walked a little too quickly. A separate local experiment takes the first eight
seconds of the 960x544 16 s combined-prompt control, slows **all** picture motion by 2x and uses FFmpeg motion
interpolation to return to 24 fps. The resulting `d16_koninck_960x544_polowa_predkosci_EXPERIMENT.mp4` and a labelled
side-by-side `porownanie_predkosci_postaci_16s.mp4` are in `~/Downloads/h3-prod-smoke/`. Both are exactly 16.000 s
and fully decode. In the slowed copy, every even output frame (192/192 checked) equals the corresponding decoded
source frame; the in-between frames are synthesized and may introduce warping or other artifacts. All 501 AAC packets
and their timestamps are identical to the original 16 s soundtrack, so sound events may no longer track the slowed
picture. The source is unchanged. The 17-frame global and regional screens report no repeated spikes here, but this
experiment excludes the source's later flickering section; that is not evidence that interpolation fixed flicker.
No one has reviewed the new playback. This changes playback speed, not model generation time; whether a shorter
direct render plus interpolation is faster at equivalent quality remains untested.

On 2026-10-31 at 21:31, an independent 15 s 960x544 FL2VA render (951.5 s wall) was converted into a separate
30.000 s / 720-frame 24 fps candidate with motion-interpolated odd frames. This slows the entire scene, including
clouds and people, and stretches the source's 15 s of events; it does not generate another 15 s of new action.
The candidate retains every one of the 360 decoded source frames exactly at the even indices; its final frame
repeats the last source frame. Odd frames can warp or smear, and still-frame checks do not establish natural motion.
The 32 kHz stereo audio was stretched with `atempo=0.5`, re-encoded to AAC and padded at the end to 30 s. Its
measured level was -41.49 to -41.41 LUFS; it needs listening review and event synchronization is not guaranteed.
The Mac processing and verification took 21.76 s in this single run; there is no matched native 30 s baseline on the
same boot, so this does not establish an equal-quality generation speedup. The global and regional screens found no
qualifying periodic runs at either 17 or 34 frames, but those screens can miss motion or interpolation defects.

`scripts/slow_video_2x.py` reproduces the experiment as an opt-in separate file, refuses overwrites, verifies a full
decode and source stability, checks every even frame and the final frame, and verifies output rate, length and audio
format. Six CPU/FFmpeg tests cover audio/no-audio inputs, frame and timestamp preservation, wrong format, source change
and existing or dangling outputs. For the actual 15-to-30 s experiment an additional independent check verified 720 continuous
frame timestamps, all even frames and stereo audio. Files in `~/Downloads/h3-prod-smoke/`:

* `d30_koninck_z_direct15_spowolniony_2x_EXPERIMENT_EXACT30s.mp4` (SHA-256
  `6d0c47401ae49b34d91594e8776aff1b4041ef5b813b6739884d9d2e9f6cd8a6`);
* `porownanie_predkosci_direct15_vs_slow30_20261031.mp4` (15 s side-by-side with the original 15 s on the left,
  the first 15 s of the slowed clip on the right, and the original audio); and
* `d30_koninck_z_direct15_spowolniony_2x_LOUDNESS_EXPERIMENT_EXACT30s.mp4`, an additional audio-only level change
  from -41.41 to -20.00 LUFS, preserving every video packet and timestamp. It can amplify noise and has not been
  reviewed by ear. The quiet retimed source and original generation remain intact.

The same opt-in pipeline was additionally exercised on a temporary, synthetic 96x64, 24 fps 150 s clip with AAC:
it produced an exact 300.000 s / 7,200-frame output, with all 3,600 input frames preserved at the even indices,
continuous timestamps, retimed audio and a successful full decode. The synthetic media was deleted after verification;
`diagnostics/h3-300-20261030/synthetic-retime-300-20261031.json` in the workspace retains the report (SHA-256
`ac39373f70346735a4c70c8d41da0bd46d1b86cc57501220b2f0941dfeee3515`). This is a format and timing test, not an H3
GPU render and not evidence that a real five-minute video looks or sounds good.

`scripts/interpolation_diagnostics.py` provides a supplementary review aid for the interpolated frames: in each
even/odd/even triplet it compares the two grayscale changes with the direct change between retained frames, both
globally and across image regions. On the actual 30 s candidate it screened 359 triplets at 336x192; median excess
was 0.071 gray levels over the whole frame and the largest regional excess was 0.232 near 20.792 s. The three frames
around that point are saved as `d30_slow30_klatki_interpolowane_20s_20261031.png`; a separate labelled road crop video
is `porownanie_chodu_direct15_slow30_zblizenie_20261031.mp4`. This screen does not prove that the intermediate frame
is correct: real motion can raise its score, and small figure warps can be missed. Three CPU/FFmpeg tests cover a
linear triplet, an intermediate-only flash, and an actual encoded retiming. The analysis report is
`d30_koninck_z_direct15_spowolniony_2x_interpolation_screen.json` (SHA-256
`18111ae42d5853563744449109070ae958137a205d758a2b18f7ac2c727f81bb`).

### Experimental repair of the existing 5 s-history clip (not a model fix)

`experimental-deflicker.py` makes a separate, explicitly altered copy: it detects the repeated drops and lifts luma
over the affected frames toward a short interpolation between neighboring frames. On this sample it changes 42 of 736
decoded frames, all inside the affected interval; their chroma bytes are identical, no luma pixels newly clip at 255,
and all other decoded video frames and the copied AAC packets are byte-identical. The video was encoded losslessly
after the brightness edits; the edits themselves change the picture. At the 13 original dip frames, the luma step
changes from -2.22..-6.21 YAVG to -0.02..-0.22, and the 17-frame detector finds no remaining run above 2 YAVG.
At frame 408, full-resolution Y changed by -6.14 on average and 98.8% of pixels darkened; in the repaired copy the
same frame transition averages -0.14 and 41.6% darken (10th/50th/90th percentiles -3/0/+3). These measurements
include real scene motion and do not establish visual acceptability.
Full decode succeeds. This is a one-clip experiment, not a default or proof of the decoder cause; owner playback review
at 17:07: "Jest lepiej, ale miganie zostało" (better, but flicker remains). Local copy:
`~/Downloads/h3-prod-smoke/d30_koninck_history5s_deflicker_EXPERIMENT.mp4`. Global luma correction is therefore not
an adequate fix; the residual may be spatial or textural, and requires further diagnosis. At the 13 original dip
frames, mean absolute pixel change at 336x192 falls from a median 7.11 to 1.62 gray levels, still above the 0.35
median away from the boundary. This difference also includes actual motion, so it is not a separate visual verdict.

### Owner feedback: cloud shadows

The owner reports that the cloud shadows on the ground in the reviewed clips look wrong: possibly too strong or not
moving consistently with the clouds. This is a subjective observation; the exact cause is unknown. The current prompt
explicitly asks the shadows to glide over the plain. Next controlled test: keep the painting, seed, 16 s duration,
1344x768 resolution, 20 steps, and all other prompt wording, but replace that clause with soft, nearly constant
landscape lighting without strong moving cloud shadows. Compare against the original 16 s clip by eye; prompt wording
alone cannot guarantee physically aligned shadows. In the original 16 s clip, mean Y in the bottom 35% of the image
rises from 56.2 to 67.6 over the first 5 s. This confirms a lighting change in that region but does not establish
whether the ground shadows follow the clouds. The soft-light 16 s A/B completed in 2,371.4 s and passed media
validation; the global two-drop screen found no repeated drops. The clip is
`~/Downloads/h3-prod-smoke/d16_koninck_softlight_EXPERIMENT_EXACT16s.mp4`. Whether its ground light looks more natural
still needs playback review; the prompt does not measure or enforce cloud-shadow alignment.

For the newer 960x544 16 s clip, the global two-drop detector found frames 289 and 306 (12.04–12.75 s), and regional
changes continued at the same 17-frame period through 15.58 s. Separate, **unreviewed** spatial repairs use the
previously reviewed luma-adjustment method with `--local-periodic`: they alter 30 of the 384 exact-duration frames
(289–378), preserve all decoded chroma and every AAC packet/timestamp, and change luma by at most 5 levels for radius
16 or 6 levels for radius 8/4. Global repeated drops are no longer detected, while regional flags remain. Across the
six affected boundary frames, lower-left/middle/right median changes fell from 2.11/2.12/1.78 gray levels to
0.83/0.86/0.78 (radius 16) or 0.71/0.72/0.68 (radius 4). These metrics include real scene movement; the narrower
blur can alter details or motion and neither candidate is a default. The separate copies and
`porownanie_migania_960x544_16s.mp4` are in `~/Downloads/h3-prod-smoke/`.

An additional unreviewed locked-camera soft-ground copy applies the earlier bounded correction after the radius-16
repair. It leaves the upper 337 rows, chroma and AAC packets unchanged, changes lower luma by at most 10 levels and
reduces the sampled lower-frame grayscale range from 26.34 to 19.77. It does not track shadows or prove physical
alignment, and can suppress real light changes. File: `d16_koninck_960x544_regional_softground_EXPERIMENT.mp4`.

On 2026-10-31 a one-sample-per-second grayscale screen over the bottom 37.5% of each exact 16 s frame found the
original at 47.06 -> 60.90 (range 13.84), a same-prompt repeat at 47.03 -> 60.94 (range 13.91), and the soft-light
sample at 47.10 -> 66.21 (range 19.11). Mean absolute change from the first sample was 10.54 / 10.57 / 13.58 gray
levels respectively. Thus the soft-light wording did not keep this coarse brightness measure more constant in this
one sample. These means include content and real motion and cannot tell whether shadows match the clouds or whether
the result looks good. Reproduce with `scripts/ground_light_stats.py`; JSON reports and the playback comparison
`porownanie_swiatla_16s.mp4` are in `~/Downloads/h3-prod-smoke/`. The original and soft-light sample were rendered
at different times, so the difference is not proof of prompt causality.

### Experimental soft-ground copies (19:04)

While that prompt A/B is queued, [`experimental-soft-ground.py`](experimental-soft-ground.py) makes a separate
postprocessed copy for a locked camera: it estimates spatially smooth luma change from frame 0, damps 35% of it,
and blends the adjustment below a configurable vertical boundary (62–77% of this image's height), capped at 10
luma levels. It does not track clouds, infer shadow geometry, fix cloud-shape drift or slow figures. It can suppress
real lighting changes; it is not a production default or proof that the shadows align with the clouds.

The 30 s input is the regional deflicker copy the owner preferred; the 60 s input is the targeted regional copy,
which has not been reviewed. At one ground-luma sample per second (bottom 28% of the image), mean absolute change
from the first sample fell from 19.69 to 14.28 Y in the 30 s pair, and 18.24 to 13.80 Y in the 60 s pair. This
is a coarse brightness measure, not a shadow-quality or playback verdict. Exhaustive decoded-frame comparisons
confirm the upper 476 rows and all chroma are identical, luma changes at most 10 levels, and AAC packet content
and timing are identical to the respective inputs: 734/736 and 1447/1450 luma frames changed. The more sensitive
two-drop global screen still finds an event at 34.71–35.42 s in the 60 s candidate; regional screening also flags
changes and can include real motion. Both candidates need visual review, including figures, natural lighting and
residual flicker.

The native and exact-length copies are in `~/Downloads/h3-prod-smoke/`:

* `d30_koninck_history5s_regional_softground_EXPERIMENT.mp4` and
  `d30_koninck_history5s_regional_softground_EXACT30s_EXPERIMENT.mp4`;
* `d60_koninck_regional_softground_v2_EXPERIMENT.mp4` and
  `d60_koninck_regional_softground_EXACT60s_EXPERIMENT.mp4`.

The exact copies retain 720 and 1,440 verified frames respectively; at 30 s copied AAC ends 16 ms early, at 60 s
it ends with the video. Raw comparisons are in `soft-ground-30-verification.txt` / `soft-ground-verification.txt`;
regional screens are in `koninck30-softground-regional.txt` / `koninck60-softground-regional.txt`.

### Direct 10/15 s lighting and a stronger 10 s comparison (2026-10-31 21:17)

The direct 960x544 10 s and 15 s renders completed with the soft-light/slow-figure prompt; they are separate
generations and do not isolate a duration effect. Across one sample per second in the bottom 37.5% of the frame,
the 10 s source ranges from 46.57 to 90.84 gray levels (range 44.27); the 15 s source spans 19.03 levels. Sampled
still frames show a visibly brightening foreground in the 10 s sample, but cannot establish whether shadows follow
the clouds or whether movement is natural. The existing 35% / 10-luma-cap 10 s postprocess reduces that range to
36.50. A separate stronger experiment uses 60% / a 20-luma cap and reduces it to 29.47, without claiming a
physically correct result. All 240 frames and AAC packets/timestamps were verified; 239 luma frames changed by at
most 20 levels, with the top 337 luma rows and all chroma unchanged. Neither the global nor regional periodic
screen flags a qualifying run, which does not rule out visible defects. As it changes more of the picture and can
suppress real movement or lighting, it remains unreviewed and is not a default.

The source, stronger candidate and labelled side-by-side comparison are in `~/Downloads/h3-prod-smoke/`:
`d10_koninck_direct_960x544_EXPERIMENT_EXACT10s.mp4`,
`d10_koninck_960x544_softground_s06_l20_EXPERIMENT_EXACT10s.mp4`, and
`porownanie_cieni_10s_silniejsze_20261031.mp4`. The comparison is 240 frames / 10 s and retains the source AAC
packets and timing. `d10_koninck_960x544_softground_s06_l20_EXPERIMENT.verification.json` records the checks and
ground-light samples.

## The 60 s render runs at 120 s/step instead of 86-88 (`koninck60-fl2va`, 14:38-, investigation in progress)

The first post-render 5 s T2VA calibration on the same boot finished at 18:38: 1344x768 ran at 21.10–21.12 s/step,
versus the earlier short-request reference of 23.2 s/step; two 960x576 warmups were 8.51–8.56 s/step versus 8.65.
These short shapes were not slow. They do not prove why the 277-frame FL2VA windows slowed: the tasks, frame counts
and conditioning differ. A matched 16 s continuation control (`calib-safe-l16`) began at 18:38 to investigate.
At 19:06 its first 277-frame window completed in 1,670.3 s, or 83.5 s/step (journal 83.27–83.51), close to the
earlier 85.4 s/step 16 s reference and much faster than the 60 s render's first 277-frame window. The head had not
restarted: systemd still showed the 14:31 boot and the same rank-0 process. Thus a persistent slow head boot alone
does not explain the difference. The request lengths and intervening work differ, including the 5 s warmups, so this
observation does not establish whether warmup, total request length or another runtime condition caused it. The
whole control finished at 19:18: HTTP 200, curl 0, full verification and audio check passed; wall 2,357.8 s, second
141-frame window 542.8 s, VAE decode 137.2 s, seam ratio 1.00, 396 frames / 16.5 s. The previous 16 s painting
control took 2,410 s; these are single observations, not a demonstrated optimization.

The first-window `step0` digest matches between the 60 s and 16 s requests (`9e7843a1`, 14 tensors), as do the first
prompt (`583119c1`) and image-conditioning (`2a5ba12d`) tensor digests. The first-window output-latent digests differ
(`56fb4303` vs `1aadfdbe`). The digest hashes tensor bytes only, with limited recursion; it does not cover scalar
arguments, all object state, tensor shape metadata or the model/backend state. The match narrows the input comparison
but does not prove the complete requests or runtime states identical, and the different output does not by itself
identify the cause of the slowdown.

The earlier 10:14 16 s render also has the same first-step tensor digest and another first-window latent digest
(`89163e97`). Compared with the newly verified 16 s control, all 396 decoded frames have mean SSIM 0.966 (Y 0.954),
and a still at 15 s has a similar locked composition. That is neither bitwise reproducibility nor a playback-quality
verdict; details and hash limitations are in `sameboot-l16-comparison.txt`.

Window 1 (277 frames, the same shape as the 30 s windows of 11:23 and 11:52 at 88.4 / 87.0 s/step) took 122.7 s/step,
window 2 120.0 s/step, every step within 0.4 s of the next: a steady 36-40 % slowdown on these samples. Stable
step times alone do not rule out a constant source of interference.
The first 16 s painting run at 10:14 and the 60 s run at 14:38 also logged the same four large FP8 GEMM shapes
(`M=42880`, with `N/K=21504/5376`, `28672/5376`, `5376/7168`, `5376/14336`); this rules out a change in those
logged matrix dimensions alone, not a change in kernel, memory, communication or other runtime state.
Checks on the live pair did not reveal a cause: GPU clocks (2164 MHz on both, cap 2200, no throttle flag, 67 / 79 C),
the fabric (both RoCE ports ACTIVE 200 Gb/s, the render's 2 x 258 MB/s on them, nothing on the management links),
CPU (performance governor, 3.35 GHz, worker at 230 % on both ranks, no cgroup limits), disk (0 MB/s), the plugin
configuration (the init lines of the 10:07 boot and the 14:31 boot are identical except ports), the host's page cache
(the head had 880 MB free, 2 GB swapped and 46 GB of cache; `drop_caches` at 15:24 during step 2 of window 2 freed
45 GB and changed nothing: 120.8 s/step before, 120.0 after), and fragmentation at boot (the compaction before the
14:31 boot left thousands of order-9..13 blocks on both hosts).

What the power draw says: spark-05 (rank 1) pulses 28-90 W, spark-06 (rank 0, the head) holds 53-58 W, whereas the
head drew 72-78 W in the evening renders. This pattern suggests an imbalance but does not by itself identify the slow
rank or cause. The UVM bottom-half
thread (host-side GPU page-fault service) runs at 0.6 core on the head and 1.8 cores on the agent during the render.
The slowdown is also progressive on the 10:07 boot: the 283-frame window at 14:25 was 104 s/step (18 % over the
277-frame rate), the 345-frame windows 136-138. Open: whether it is the head's boot state, the agent's (up since
10:07, compacted while running at 14:30) or the 3.8-day host uptime. After a verified 60 s result, the guarded
calibration uses two 960x576 warm-ups and a 5 s 1344x768 T2VA request (reference `cmp-t2va-1344`: 23.2 s/step). If
the current boot is slow, repeat after a head restart and an agent + head restart; if it is fast, first run a matched
16 s continuation (window 1 reference 85.4 s/step), then restart the idle head to activate the longer timeouts. A
failed benchmark or missing rate is a hold, not a completed calibration.

At 16:39 the third window finished in 2398.1 s; at 17:18:59 the fourth finished in 2397.4 s (seam ratio 0.98) and
window 5 began. The already running client and API have 14,400 s
limits (client deadline about 18:37:51). The earlier ~18:39 projection incorrectly applied a full-window rate to the
shorter final window. At 18:11 the journal showed step 11/20 at 42.48 s/step, but the progress lines were buffered:
window 6 actually finished at 18:13:07 in 850.0 s, seam ratio 0.99. The earlier ~18:18 projection from the stale
progress was also too late. Decode and delivery remain pending; there are about 24.7 minutes until the client
deadline. Longer limits are staged for the next head restart, not effective in the live processes.
The same request provides a useful, **uncontrolled** size indication: denoising window 5 (277 frames, 255 new after
overlap) took 2397.4 s, while window 6 (175 frames, 153 new) took 850.0 s. That is 9.40 versus 5.56 seconds per new frame,
about 41% less for this pair of windows. They have different seeds, temporal positions and run order; this is not a
whole-video speedup measurement and excludes decode, retries and quality effects from additional seams. The queued
same-boot 16 s comparison holds the request and seed fixed while changing the window size.
The timeout recovery queue checks the actual failure, waits for both nodes to be idle, and verifies the effective
limits before retrying. The 23.2 s/step T2VA reference followed two 960x576 warm-up requests; the recovery queue
repeats that sequence before each 1344 measurement. Individual runs still cannot establish a root cause on their own.

### A gentler 20% slowdown of the owner-preferred clip (2026-10-31)

The first 24 s of `d30_koninck_history5s_regional_EXACT30s_EXPERIMENT.mp4` were trimmed without changing any
retained decoded frame or AAC packet, then stretched to exactly 30 s at 24 fps. The whole picture plays at 0.8x
speed, not only the figures; audio is time-stretched and re-encoded, and no new narrative action is generated.
The separate motion-compensated result is
`~/Downloads/h3-prod-smoke/d30_koninck_history5s_regional_spowolniony20_v2_EXPERIMENT_EXACT30s.mp4`
(SHA-256 `59ba3b854171646c752580257eea57c11ecb611ab6d986e9e4427ecb65b58899`). All 144 output frames whose times
coincide with source frames were verified identical, all 720 timestamps are continuous 24 fps, and both streams
fully decode. The other output frames are interpolated and can warp or smear; audio and playback need review.

A first ordering of the interpolation and a second ordering gave identical 720 decoded frames and identical AAC
packets, despite different MP4 file hashes. Both show uneven median absolute frame changes by five-frame output
phase: in the lower-middle region, 0.092..0.162 grayscale levels versus 0.172..0.184 across four source phases.
Removing the exact-anchor overlay did not change the 720 decoded frames of a blended control; a motion-compensated
and a blended output matched on 543/720 frames and showed nearly the same phase medians. A synthetic moving-stripe
control had lower-middle min/max phase ratios 0.874 at the source and 0.861 after retiming; it does not model the
painting. These measurements describe a content- and phase-dependent cadence, not proof of visible judder or a root
cause. The available phase diagnostics and their limitations are in `scripts/retime_phase_diagnostics.py`.

A separate copy additionally attenuates smooth ground-light differences from frame 0 by 35%, capped at 10 luma
levels. It changed 717/720 luma frames while preserving the top 476 rows, all chroma, frame timing and AAC packets;
the one-per-second lower-frame brightness range decreased from 33.96 to 25.48 grayscale levels. This can suppress
real lighting and does not establish whether clouds and shadows align. It is
`d30_koninck_history5s_slow20_softground_EXPERIMENT_EXACT30s.mp4` (SHA-256
`73e021035910f11c7e60e3a50e66f1fe1a12bc36b186accc1daccade3e431571`). Full-frame and road-zoom comparisons against
the unretimed clip are `porownanie_tempa_30s_20proc_20261031.mp4` and
`porownanie_chodu_30s_20proc_zblizenie_20261031.mp4`; both retain the original left clip's AAC and need playback
review. The 1.25x-duration and optional blend modes are opt-in; nine retiming and three phase-screen tests pass.

The ground diagnostic also measures spatial spread of the smoothed, signed grayscale difference from frame 0 in the
lower region, once per second. In this pair its average standard deviation fell from 12.66 to 10.17 levels; no sampled
pixel reached the diagnostic's signed-difference bounds. This describes less uneven change in that region, not the
motion or geometry of shadows. Uniform changes, local changes, boundary saturation and sampling are covered by four
tests for `scripts/ground_light_stats.py`.

### Wider regional flicker repair on the 30 s history clip (2026-10-31)

The owner preferred the earlier regional repair but reported residual flicker. Two separate `--local-periodic`
copies correct the dominant 17-frame phase over a wider interval, with a spatial blur radius of 16 or 4 pixels.
Each changes 110 of the original 736 luma frames by at most 8 levels; chroma, video timestamps and AAC packets
are identical to the original. Compared directly with the owner-preferred repair, the radius-4 copy differs in 68
frames (18 in 15–22 s, 50 in 22–30 s), again by at most 8 luma levels.

Across source-detected transitions in at least three regions, the median absolute regional **mean** change for
28 selected frames in 15–22 s is 0.105 gray levels in both the preferred and radius-4 copies. For 17 selected
frames in 22–30 s it is 1.452 versus 0.441; the radius-16 copy measured 0.454 there. These statistics may include
real motion and can hide local change. The separate absolute-pixel screen still reports repeated lower-region spikes
through 24.79 s in the radius-4 candidate; it does not prove that the remainder is artifact-free. The narrower
correction can alter motion or fine detail, so the candidate is not a default and needs playback review.

`~/Downloads/h3-prod-smoke/d30_koninck_history5s_allregional_radius4_EXPERIMENT_EXACT30s_20261031.mp4` is a
720-frame exact-length copy (SHA-256 `3105365039e43dca2fd7459706c05101a62b3753ca5ee76691fcc3db0a6f141b`), with
retained decoded frames and AAC prefix checked and the whole file decoded. The side-by-side
`porownanie_migania_30s_regional_radius4_20261031.mp4` retains the preferred left clip's AAC. The signed-transition
and exhaustive luma/chroma verification reports are `d30_koninck_allregional_radius4_transitions_20261031.json` and
`d30_koninck_allregional_radius4_20261031.verification.json` in the same folder.

A more conservative hybrid retains the **exact owner-preferred frames through frame 533** (22.25 s), then uses the
radius-4 broader repair. Frames 532–536 match in both input clips, so the switch introduces no new source difference
at the boundary. Every one of the 720 decoded output frames matches the selected input, video timestamps are
continuous 24 fps, AAC packets/timestamps are identical and the file fully decodes. Compared with the preferred copy,
49 frames after the switch change by at most 6 luma levels, with all chroma unchanged. The clip is
`d30_koninck_history5s_regional_hybrid_EXPERIMENT_EXACT30s_20261031.mp4` (SHA-256
`4dedf52d3a40a3f4ba189491c047a156578b52033de1446f0e33f81e1eb4c7ce`), and the labelled comparison is
`porownanie_migania_30s_hybrid_20261031.mp4`.

An independent exact-30 s comparison against the original source selected 30 repeated-change frames in 15–22.25 s;
median absolute regional mean change was 0.095 gray levels for both preferred and hybrid clips. Across 15 selected
frames in 22.25–30 s it was 1.680 for preferred and 0.463 for hybrid. The screen can include real motion and does
not establish perceived flicker or detail quality; regional spikes still remain. Report:
`d30_koninck_hybrid_transitions_20261031.json` (SHA-256
`6583b91f2c37c40a257877bce898906ea2d5b053f1da21dc4974b675ab1512ee`). Playback review is still required.

### A 60 s copy correcting only the final global flicker run (2026-10-31)

The 960x544 exact-60 s original and its regional repair were combined at frame 1323 (55.125 s), just before the
first global drop at frame 1326. Both inputs have identical frames 1321–1325. The separate result retains the first
1,323 source frames exactly and changes 35 later luma frames (1326–1432) by at most four levels; all chroma, AAC
packets/timestamps and continuous 24 fps video timestamps match. Every output frame was checked against its selected
input and both streams fully decoded. File:
`d60_koninck_960x544_tail_deflicker_EXPERIMENT_EXACT60s_20261031.mp4` (SHA-256
`371e308fe19b7471335ae5a044b63c1951114fea514a1f94c3f7b622ee402f04`).

At 13 source-selected repeated-change frames after the cut, median absolute regional mean change was 1.951 gray
levels in the original and 0.467 in the candidate; 25 selected frames before the cut are unchanged. The global
17-frame, two-drop screen no longer finds the earlier >=2 YAVG run, while regional signals remain elsewhere and in
the ending. These measurements can include real motion, and do not establish visual or sound quality. The full
side-by-side `porownanie_migania_koncowki_60s_20261031.mp4` uses the original left AAC. The reusable
`scripts/combine_repairs.py` refuses mismatched boundary frames, timing, color geometry or AAC, verifies every
selected frame and never overwrites an output; three FFmpeg tests cover success and refusal cases.

### Bounded regional experiment for a verified long clip (2026-10-31)

`scripts/experimental_regional_repair.py` packages the radius-4 experiment for a separate verified clip. It checks
the source hash, geometry, frame count, AAC and 24 fps, selects the dominant repeated regional phase and skips a
clip with no qualifying events. It refuses to process more than 30% of frames, never overwrites a file and checks
every decoded output frame, chroma, timestamps, AAC packets, full decode and source stability before publication.
The 30% bound is a guard against a broad correction, not a visual-quality threshold. The output report compares
regional changes at the source-selected transitions and explicitly flags that real lighting or motion can be changed.

Five unit tests pass. A real 384-frame 960x544 source (`d16_koninck_duration_precedence_960x544_VALIDATION_EXACT16s`,
SHA-256 `320bb6dbf9e2b517aff1a8f70c31b77bee8391e47eed01c1687388c7fec90a3a`) produced a separate, fully decoded
test copy: 30 frames changed, at most 8 luma levels, chroma/timestamps/AAC unchanged. Across six source-selected
transition frames the median absolute regional mean change was 1.477 before and 0.629 after. This numerical test
does not establish perceived quality or prove the decoder cause. The 300 s application remains pending the render.

### Ground-light comparison for the 0.8x 30 s clip (2026-10-31)

The 0.8x clip and its separate soft-ground copy now have a labelled side-by-side:
`porownanie_cieni_30s_spowolniony20_20261031.mp4` (SHA-256
`4418f85f2473e1b3777805d66759b7f3117a7f37f321de7cba20299ddedb5ffd`). It retains the left clip's AAC and has
720 frames / 30 s; both streams fully decoded and both source hashes were unchanged.

The ground-light report additionally measures the spatial spread of the smoothed gray difference between consecutive
one-second samples. Its mean / maximum were 2.20 / 9.04 gray levels in the slowed source and 1.81 / 7.36 in the
soft-ground copy; no sample reached the signed-difference boundary. These values show smaller measured lower-frame
changes in this pair, including any real scene changes. They do not say whether shadows align with clouds or look
natural. Full report: `h3-ground-adjacent-slow20-20261031.json` (SHA-256
`a3ef6393a4ac6ee54c85afd169eece67b68d846d201daf753839f83546b9e545`). The added adjacent-sample metric has five
unit/media tests, including a localized light change introduced after the first second.

### Symmetric temporal ground smoothing: mixed result, not promoted (2026-10-31)

`experimental-temporal-ground.py` tests an optional 25-frame centered Gaussian filter of low-spatial-frequency
ground luma (sigma six frames, maximum six levels). It fades its correction to zero at the first and last frame.
The symmetric kernel has no constant time shift in the interior, but can still smear real changes and does not
establish that shadows follow clouds. Three unit/media tests cover symmetry, a linear time ramp, bounded masking,
first/last frames and exact chroma/audio/timing preservation.

On the 0.8x soft-ground 30 s clip, the separate v2 output changed 707 of 720 luma frames by at most five levels;
the first and last frames, top 476 rows, all chroma, video timestamps and AAC packets remained identical, and both
output streams fully decoded. The mean / maximum adjacent-second ground spatial change moved from 1.81 / 7.36 to
1.78 / 6.90 gray levels, while the median / 95th-percentile absolute frame change in the lower-middle region rose
from 0.149 / 0.588 to 0.175 / 0.597. These metrics include actual motion and do not establish perceived quality,
but give no clear basis to promote the filter. Output:
`d30_koninck_slow20_softground_temporal_v2_EXPERIMENT_EXACT30s_20261031.mp4` (SHA-256
`dd4e7bcc7a7ff438200c04e88b78ca83bf7b080a90e2d8de27eedf5ecfa3536b`); comparison:
`porownanie_czasowej_stabilizacji_cieni_30s_v2_20261031.mp4` (SHA-256
`4d64d3edfc0c2daee94e25264f15a5499a8b385ba34eb70876fec059b24ff8df`). The side-by-side uses the left AAC. Playback
review remains pending; the unfiltered soft-ground source and production defaults remain unchanged.

### Direct 15 s ground-light comparison (2026-10-31)

The direct 15 s 960x544 clip had a 19.03-level range in the mean of the lower image, but its mean spatial spread of
smoothed lower-frame changes from the first frame was 10.88 and its mean adjacent-second spread was 5.20 gray levels.
That illustrates why a whole-region mean alone can miss localized changes; none of these statistics identifies actual
cloud shadows.

A separate 35%-attenuation experiment reduced those numbers to 13.60, 8.49 and 4.08 respectively. It changed luma in
356 of 360 frames by at most 10 levels; the top 337 rows, chroma, timestamps and AAC packets are unchanged, and both
streams fully decoded. Output: `d15_koninck_direct_softground_EXPERIMENT_EXACT15s_20261031.mp4` (SHA-256
`35bb3b309280c57796419c2a550f3cbd7ba67dec266c906820a822ff8e21689b`). The labelled side-by-side is
`porownanie_cieni_15s_direct_20261031.mp4` (SHA-256
`259dc7c4212e587f618632b9fce694172bb2ca2b96ad8275d2ee93bc7c285d7e`). This can suppress real lighting and detail;
cloud alignment and playback quality are unverified.

`scripts/compare_videos.py` now makes labelled comparisons for any specified exact 24 fps frame count. It preserves
the left AAC packets, keeps each image's aspect ratio, validates continuous video timing and full audio/video decode,
checks both source hashes and refuses overwrite. Two synthetic media/guard tests passed; the 15 s comparison above
was also checked on the real pair. The side-by-side itself is scaled and recompressed, so the original clips remain
the quality reference.

### Timeline and handover stills for long-clip review (2026-10-31)

The exact 60 s source has a timeline including its final frame at 59.958 s:
`d60_koninck_timeline_z_koncem_20261031.png` (SHA-256
`e61779c48d1f5544c17b25fff496cc3e8b3f57c279eb5d9de4ee6f36620cfa9b`). A separate sheet shows three stills around
each of its five planned window handovers at frames 277, 532, 787, 1042 and 1297:
`d60_koninck_sklejenia_20261031.png` (SHA-256
`df3d91912eed8bfeb9846fc555bdb3a38fcb3b20615ba8c1244a6c1cdca72522`). Each sampled index, 24 fps timestamp and source
hash was checked. These are downscaled stills, so they cannot establish smoothness, flicker, audio quality, precise
decoded seam timing or physical cloud-shadow alignment. The timeline does show noticeable shifts in ground brightness
between several samples, consistent with the owner's concern and motivating the separate lighting comparisons.

`scripts/h3_contact_sheet.py --include-last` adds the final frame to a timed overview;
`scripts/h3_seam_sheet.py` derives approximate handover indices from a declared duration and window plan. Four tests
cover timestamped extraction, final-frame inclusion, handover indices and invalid inputs; both tools were also run
against the real 60 s source. Any later automated 300 s handover sheet must check its plan against the actual run log.
