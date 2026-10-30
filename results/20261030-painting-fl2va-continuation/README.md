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

30 s (`koninck30-grid.png`, one frame every 3 s): the picture stays the picture across three windows, the clouds keep
drifting, both hand-overs are invisible in the numbers, audio -39..-40 dB flat. At 1344x768 a window costs ~25 min
(84 min for 30 s); 960x576 would be about half.

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
frame index is approximate; compare the trace with the rendered frame-level metrics. The hook is off by default,
CPU-tested on stand-ins, and not yet qualified in a distributed GPU decode. Run only when the pair is available.

A sweep of all 25 locally saved `h3-prod-smoke` MP4s found a second sequence meeting this specific threshold in
`d60_seedstep_per_window.mp4`: frames 1326..1445 (55.25..60.21 s), eight drops every 17 frames, median -4.38 YAVG.
The other 23 clips had no >=2 YAVG run lasting at least three periods. This is a global-luma screen, not proof that
those clips are flicker-free: smaller, local or shorter events can be visible to the owner and miss this threshold.

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
is pending. Local copy: `~/Downloads/h3-prod-smoke/d30_koninck_history5s_deflicker_EXPERIMENT.mp4`.

## The 60 s render runs at 120 s/step instead of 86-88 (`koninck60-fl2va`, 14:38-, investigation in progress)

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

At 16:39 the third window finished in 2398.1 s and window 4 began. The already running client and API have 14,400 s
limits (client deadline about 18:37:51); the current rate puts the end of denoising around 18:39, before decode, so
the clip may time out. Longer limits are staged for the next head restart, not effective in the live processes.
The timeout recovery queue checks the actual failure, waits for both nodes to be idle, and verifies the effective
limits before retrying. The 23.2 s/step T2VA reference followed two 960x576 warm-up requests; the recovery queue
repeats that sequence before each 1344 measurement. Individual runs still cannot establish a root cause on their own.
