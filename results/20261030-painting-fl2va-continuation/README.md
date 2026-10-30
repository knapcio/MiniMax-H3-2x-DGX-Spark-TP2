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
the locks have nothing to correct. Owner's eye pending; 30 s, a 5 s pinned history variant and 60 s follow.

30 s (`koninck30-grid.png`, one frame every 3 s): the picture stays the picture across three windows, the clouds keep
drifting, both hand-overs are invisible in the numbers, audio -39..-40 dB flat. At 1344x768 a window costs ~25 min
(84 min for 30 s); 960x576 would be about half.
