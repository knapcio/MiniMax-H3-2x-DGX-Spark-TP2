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

## A 5 s pinned history (`koninck30-hist5s`, window 345 / overlap 124, 12:18-14:30)

The owner asked whether a sliding window with more history would help. With 124 frames (5.2 s) of the previous
window pinned instead of 22 (0.9 s): seams -0.5 / +0.2 luma, inter-frame 0.57 / 0.50 against neighbours 0.3-0.5,
ratios 0.98 / 0.98, no cut, motion 0.3-1.4, light 115 -> 135 in the first (longer) window then 125-131, 7865 s
(2.6x the 22-frame run: fewer new frames per window and 345-frame windows at 1344x768). On a locked camera the
hand-over was already invisible with 0.9 s of history, so the extra history buys nothing visible here and costs
2.6x; the default stays 22 frames. (On a moving camera the longer history can only help what was in the frame within
those 5 s.)

## The 60 s render runs at 120 s/step instead of 86-88 (`koninck60-fl2va`, 14:38-, investigation in progress)

Window 1 (277 frames, the same shape as the 30 s windows of 11:23 and 11:52 at 88.4 / 87.0 s/step) took 122.7 s/step,
window 2 120.0 s/step, every step within 0.4 s of the next: a deterministic slowdown of 36-40 %, not interference.
Ruled out on the live pair (read-only checks): GPU clocks (2164 MHz on both, cap 2200, no throttle flag, 67 / 79 C),
the fabric (both RoCE ports ACTIVE 200 Gb/s, the render's 2 x 258 MB/s on them, nothing on the management links),
CPU (performance governor, 3.35 GHz, worker at 230 % on both ranks, no cgroup limits), disk (0 MB/s), the plugin
configuration (the init lines of the 10:07 boot and the 14:31 boot are identical except ports), the host's page cache
(the head had 880 MB free, 2 GB swapped and 46 GB of cache; `drop_caches` at 15:24 during step 2 of window 2 freed
45 GB and changed nothing: 120.8 s/step before, 120.0 after), and fragmentation at boot (the compaction before the
14:31 boot left thousands of order-9..13 blocks on both hosts).

What the power draw says: spark-05 (rank 1) pulses 28-90 W, spark-06 (rank 0, the head) holds 53-58 W, whereas the
head drew 72-78 W in the evening renders; the head is the slow rank and the agent waits for it. The UVM bottom-half
thread (host-side GPU page-fault service) runs at 0.6 core on the head and 1.8 cores on the agent during the render.
The slowdown is also progressive on the 10:07 boot: the 283-frame window at 14:25 was 104 s/step (18 % over the
277-frame rate), the 345-frame windows 136-138. Open: whether it is the head's boot state, the agent's (up since
10:07, compacted while running at 14:30) or the 3.8-day host uptime. `queue41.sh` runs after the 60 s render:
a 5 s 1344x768 T2VA calibration (reference `cmp-t2va-1344`: 23.2 s/step) on the current boot, then after a head
restart, then after an agent + head restart; if the current boot is already fast, a 16 s continuation instead
(window 1 reference 85.4 s/step).
