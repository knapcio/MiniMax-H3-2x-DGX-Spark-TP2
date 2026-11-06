# Identity guard calibrated on the published 120 s clip (2026-11-06, CPU only)

The 120 s English shop scene published on X (`results/20261103-xclip/`, `xclip120r`, seed 2404, 1344x768) shows the
two defects the guard is for: a background figure walks behind something and a different one comes out, and a lead
turns away and comes back looking different. Before the guard's first live run, the clip was scored offline exactly
as the guard scores a window: 241 frames (2 fps, resized to 960 px) against the bank built from the clip's own
references (`ref_2208.png` and its two 512 px crops, 4 bank faces), `scripts/face_bank.py score` with the
2026-11-06 defaults (`min_sim` 0.4, `min_face` 0.05). Per window (277 frames, 22 overlap) the fraction of frames with
an unknown principal face is what the guard compares with its limit; the guard samples 8 frames per window, the second
table uses every 2 fps frame.

## What the scorer flagged, and what it was

310 principal faces, 25 unknown (similarity below 0.4). Three kinds (`sheet-flagged-frames.png`, times in the corner):

- **The lead in profile.** The man at x = 0.26-0.32 turned to the woman scores 0.30-0.38 against the reference frame
  and the crops (10-21 s, 40-50 s, 66-67 s, 95-96 s). Same person, wrong angle for a bank that holds one frontal view.
- **Not a face.** A 0.15-0.17 wide "face" at x = 0.35, y = 0.75 with similarity -0.10 to 0.07 (11, 18, 76-83, 87, 95 s):
  the brass scale on the counter, detected by MTCNN at probability 0.90-0.976. The real faces in the same frames sit at
  0.995-1.000.
- **A stranger.** At 87.0-87.5 s a man in a flat cap enters at the left edge (x = 0.03-0.11, probability 0.999,
  similarity 0.30): the figure that "came out as somebody else". The only genuine unknown in the clip.

## Two rules, simulated window by window

`unknown needs prob >= 0.98`: a face counts as unknown only when the detector is sure of it (default since 2026-11-06,
`--unknown-min-prob`). `grow 0.6`: bank growth (`H3_TP2_IDENTITY_GROW=0.6`), the accepted window's faces with similarity
>= 0.6 join the bank (cap 24), simulated in window order with the guard's own acceptance (limit 0.2).

8 sampled frames per window (what the guard sees):

| rule | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w9 | w10 | w11 | w12 | re-samples at 0.2 | at 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as deployed in queue93/94 (any probability, no growth) | 0 | .25 | 0 | .25 | .12 | 0 | .12 | 0 | .25 | 0 | 0 | 0 | 3 | 5 |
| unknown needs prob >= 0.98 | 0 | .12 | 0 | .25 | .12 | 0 | .12 | 0 | .12 | 0 | 0 | 0 | 1 | 5 |
| growth 0.6 | 0 | .12 | 0 | 0 | 0 | 0 | 0 | 0 | .25 | 0 | 0 | 0 | 1 | 2 |
| **both** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **.12** | 0 | 0 | 0 | **0** | **1** |

Every 2 fps frame of each window:

| rule | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w9 | w10 | w11 | w12 | at 0.2 | at 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as deployed | .09 | .31 | 0 | .18 | .12 | 0 | .17 | .19 | .23 | 0 | 0 | 0 | 2 | 6 |
| unknown needs prob >= 0.98 | .09 | .25 | 0 | .18 | .12 | 0 | .17 | 0 | .18 | 0 | 0 | 0 | 1 | 5 |
| growth 0.6 | .09 | .06 | 0 | 0 | 0 | 0 | 0 | .19 | .14 | 0 | 0 | 0 | 0 | 2 |
| **both** | .09 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **.09** | 0 | 0 | 0 | **0** | **0** |

With both rules the guard would have left the published clip alone at a limit of 0.2 and re-sampled exactly one
window at 0.1: window 9, where the stranger walks in (one of 8 sampled frames; 2 of 22 at 2 fps). The man's profile
no longer counts once the bank has grown through windows 1-2 (his three-quarter views at 0.6-0.9 get in, the profile
then matches them above 0.4). The scale never counts. The margins of the probability rule are thin on this one clip
(object up to 0.976, faces from 0.995), so it is a default to watch, not a law.

What this does not show: whether a re-sample of window 9 would have kept the stranger out (that needs the live guard,
queue94 and the 60 s event clip queue92), and the cost: 67-68 s per window at 1344x768 for the decode and the scorer
(`results/20261105-multiref/`).

Also tried and dropped: a relative size rule (principal = at least 0.6 of the widest face in the frame) changed
nothing, because the false detection was the widest "face" in its frames; `min_face` 0.03 instead of 0.05 only added
background faces (5 re-samples at 0.2).

**The leads by eye** (`sheet-lead-crops.png`: the woman at 5-52 s, the man at 5-110 s). Both are the same person
throughout; the woman's own-face similarity falls to 0.43-0.48 (12.5 s, 35 s) and the man's to 0.17-0.40 (10-20 s,
65-70 s, 95-100 s) when they stand in profile, and the woman is not detected at all through most of 10-50 s. So on this
clip the embedding separates "turned away" (0.17-0.5) from "somebody else" (0.30 for the stranger) poorly on a single
frame, and only the bank that has grown through the earlier windows keeps the profiles known; `min_sim` 0.4 stays.

Files: `sheet-flagged-frames.png` (eleven of the flagged frames with their times). The frame scores live in
`/tmp/h3-calib120/` on spark-06 (`faces-chunk.*.npz`: embedding, best similarity, width, probability per principal face).

## The first live run with the fixes (queue94, 2026-11-06 08:10-08:48)

The 22 s occlusion probe of queue93 (960x544, seed 2301, scene + two crops, she walks left behind him in window 1 and
back to the right in window 2) with `H3_TP2_IDENTITY_GUARD=0.1` and `H3_TP2_IDENTITY_GROW=0.6` on the fixed code
(`c17ba9f`). From the head's journal:

- `identity decode: shape (1, 3, 277, 544, 960) dtype torch.uint8 min 0 max 255 mean 68.7`: the decoder returns uint8,
  which is what the first run scaled to white; the scored frames now have a mean of 63-70 and the scorer finds
  11-13 faces per window (`sheet-live-scored-frames.png`: the 8 frames of the two accepted windows, left to right).
- Window 1, attempt 1 (seed 2301): 13 faces, 2 unknown in **1 of 8 frames** (12 %, min similarity 0.08: the woman
  passing behind him, blurred) -> score 1.25 at the 0.1 limit -> **re-sampled** with seed 10220: 11 faces, 0 unknown
  frames, accepted; the bank grew by 10 faces. Window 2 (seed 2302): 12 faces, 0 unknown frames, bank +11 (25 faces,
  4 of them references). In both accepted windows one doubtful detection with similarity -0.10 / 0.01 was ignored by
  the probability rule, as calibrated above.
- Cost: decode 34 s + scorer 7 s = **41 s per check at 960x544** (67-68 s at 1344x768), and one re-sampled window
  (679 s): 2261 s against 1534 s for the same request without the guard (queue93), peak 63.9 GB (unchanged).
- The clip (`sheet-live-clip.png`): one take, 0 cuts; the man 0.89 mean throughout, the woman 0.89 at the start and
  0.88 at the end (embedding-matched), 0.36-0.47 while she passes behind him. Against queue93 (0.63 start, 0.81 end)
  her return is closer to the reference; the re-sampled window 1 also keeps her behind him instead of crossing the
  foreground, which is a different seed, not the guard's merit.

What the limit means with 8 scored frames: one frame is 12.5 %, so **0.1 re-samples a window on a single unknown
frame** (a stranger who stays, or, as here, one blurred pass-by) and 0.2 tolerates one. On the published 120 s clip
the stranger's window would trip at 0.1 only. For a dialogue scene in which nobody new should appear, 0.1 and one
re-sampled window now and then is the better trade; 0.2 when extras may walk through.
