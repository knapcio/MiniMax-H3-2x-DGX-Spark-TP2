# 2026-11-03: character identity across continuation windows ("Lalka" shop scene, 60 s, 960x544)

Owner's verdict on the 11-01 doll films (300 s, six FL2VA blocks chained through a PNG of each block's last frame, Polish
dialogue from `<d>` tags): "marne"; on the first arm below: "calkiem niezle", with three complaints: one line came out in
Russian, 2-3 s of silence between lines (one line per 11.5 s window), and the faces change "troche". This directory is
about the faces. Measure: `scripts/face_drift.py` (MTCNN + InceptionResnetV1/VGGFace2, cosine similarity of each face to
the reference frame, one sample per second; same-person pairs are typically > 0.6-0.7, different people < 0.4; a profile
view lowers the number too, so compare arms, not absolute values).

Inputs for every arm: the same reference frame (1672x941), seed 3101, 20 steps, six 277-frame windows with a 22-frame
overlap, one dialogue line per window (`~/h3-tp2-results/lalka-ab/prompt.txt`, `extra.json` on Spark_06), the serving
defaults of 2026-11-03 (look-lock classes/tail/ramp 1.0, pinned prefix, seam retries, seed per window).

| arm | request | wall | face similarity man: first 10 % -> last 10 % (mean, min) | woman | cuts / seams / light |
|---|---|---|---|---|---|
| baseline 11-01 (first 60 s of the 300 s film, PNG-chained blocks) | FL2VA per block | - | 0.78 -> ~0.70 (w6) | 0.75 -> ~0.53 (w6); 0.21 at 300 s | - |
| A: one FL2VA continuation request | `task=fl2va`, `long_video`, `continuation_prompts` | 3194 s | 0.764 -> 0.554 (0.659, 0.495) | 0.737 -> 0.476 (0.601, 0.380) | none / 0.25-0.54 / 74-78 |
| **B: one Ref2VA continuation request, reference 2048 px** | `task=ref2va`, Ref2VA partition, `H3_TP2_REF_SHORT_EDGE=2048` | 6169 s (one seam retry, ~5270 s without) | **0.797 -> 0.905 (0.856, 0.682)** | **0.781 -> 0.762 (0.790, 0.661)** | none / 0.34-0.58 (1.05 at 22 s) / 73 flat |
| C: FL2VA + hidden end anchor (the reference as a hidden last keyframe of every window) | `task=fl2va`, `H3_TP2_ID_LOCK=end`, `H3_TP2_ID_END_EXTEND=68` (canvas 277 -> 345 frames, the extra frames dropped) | 4472 s (1.40x A, 0.72x B) | 0.770 -> 0.875 (0.798, 0.674) | 0.770 -> 0.702 (0.730, 0.549) | 2 in window 4 (an insert close-up of the letter, 33.6-38 s, faces missing on 4 samples) / 0.27-0.76 / 75 flat except the insert |
| **D: one Ref2VA continuation request, reference 1024 px** (queue64, 11-04 06:16-07:28, Ulysses exact default) | `task=ref2va`, Ref2VA partition, `H3_TP2_REF_SHORT_EDGE=1024` | **3384 s** (no retry; 1.06x A, 0.55x B) | **0.785 -> 0.877 (0.806, 0.574)** | **0.751 -> 0.781 (0.746, 0.365)** | none / 0.28-0.81 (2.24 at 43.4 s) / 67-75 (window 4: a passer-by crosses the foreground 35-40 s, both faces hidden on 4 samples) |
| C2: arm C with seed 3201 (queue65, 11-04 08:57-10:34, Ulysses exact default) | as C, seed 3201 | 4165 s (no retry) | 0.722 -> 0.869 (0.777, 0.671) | 0.731 -> 0.717 (0.739, 0.569) | none / 0.21-0.42 / 69-71 flat, no face missing |

Per window (mean similarity), A vs B vs C vs C2 vs D:

| window | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| A man | 0.748 | 0.673 | 0.636 | 0.684 | 0.609 | 0.554 |
| B man | 0.784 | 0.887 | 0.847 | 0.870 | 0.864 | 0.905 |
| C man | 0.750 | 0.802 | 0.821 | 0.781 | 0.795 | 0.865 |
| C2 man | 0.719 | 0.827 | 0.745 | 0.768 | 0.772 | 0.869 |
| D man | 0.764 | 0.864 | 0.836 | 0.700 | 0.791 | 0.876 |
| A woman | 0.741 | 0.682 | 0.568 | 0.524 | 0.554 | 0.476 |
| B woman | 0.813 | 0.781 | 0.799 | 0.763 | 0.807 | 0.762 |
| C woman | 0.773 | 0.724 | 0.749 | 0.690 | 0.709 | 0.711 |
| C2 woman | 0.699 | 0.699 | 0.748 | 0.789 | 0.774 | 0.717 |
| D woman | 0.792 | 0.735 | 0.760 | 0.563 | 0.802 | 0.771 |

## Reading

- Arm A drifts exactly like the PNG-chained film's first minute: the identity loss is in the model's latent-tail
  continuation, not in the stitching. Code reason (`results/20261103-identity-drift/`): windows 2+ run as t2va and lose
  the FL2VA keyframe; the 22-frame tail plus the reference's image tokens in the prompt encoding carry too little at face
  scale (a face is ~3x3.5 model tokens at 960x544).
- Arm B holds both faces for the whole minute (the man even converges toward the reference), because the Ref2VA
  reference image conditions every window and every step. Seams, cuts, light and flicker are as good as in A. The price
  is 1.65x the render time (1.93x with the seam retry) and a partition switch (the boot probe sends a ref2va request
  there since ca923e0).
- Arm C (16:57) rescues most of the identity on the cheaper partition: the hidden end anchor (the reference pasted as
  the last keyframe of a 345-frame canvas whose last 68 frames are dropped) keeps the man at B's level (0.80 mean,
  rising to 0.87) and the woman at 0.73 (B 0.79, A 0.59), at 1.40x A's time against B's 1.93x. Its failure is a
  different one: in window 4 the model cut to an insert close-up of the handwritten letter for 4.4 s (two scene cuts,
  luma 75 -> 132 -> 65) and came back to the same two faces; the prompt says one continuous take. A and B had no cuts
  in 6 windows each, so one insert in 18 windows may be seed luck (the window-4 seed is 3104 in every arm) or the
  anchor pulling the window toward the reference composition. Ranking for character clips: B (Ref2VA) when time
  allows, C as the cheaper FL2VA fallback once the insert-cut rate is known (a second seed, or `H3_TP2_ID_END_EXTEND`
  shorter than 68); arm D (Ref2VA with the 1024 px reference, queue64) tests whether B's price can be cut.
- Arm D (11-04 07:28) cuts B's price to nothing: the 1024 px reference renders in 3384 s, 6 % over the FL2VA arm A
  (the 2048 px reference costs 1.65x because its image tokens sit in every attention step; Ulysses exact, default since
  02:25, takes a few per cent off as well), and the faces stay at B's level: man 0.806 mean (B 0.856, C 0.798, A 0.659),
  rising to 0.88 in window 6; woman 0.746 (B 0.790, C 0.730, A 0.601). No cuts, no seam retry, light flat. Window 4
  (seed 3104, the one that produced C's letter insert) again adds an event: a man in a brown coat walks through the
  foreground 35-40 s and hides both faces (4 missing samples, the 0.563 woman mean of that window is the half-hidden
  face), then the same two faces continue (window 5: 0.79/0.80). Ranking for character clips now: D (Ref2VA, 1024 px
  reference) as the default recipe, B when the last few points of identity matter more than 1.8x the time, C as the
  FL2VA-only fallback.
- Arm C2 (10:34, seed 3201) answers the insert-cut question: the hidden end anchor with another seed gives a
  continuous take (no cut, no missing face, seams 0.21-0.42, light flat) with the same identity as C (man 0.777,
  woman 0.739 mean; window 4 is the best window for the woman this time, 0.789). The 16:57 insert was the window-4
  seed, not the anchor: one insert in 12 anchored windows, the same rate as the plain arms (D's window 4 added a
  passer-by under the same seed). C stays the FL2VA-only fallback at 1.3x A's time; D remains the default.
- The X clip (English, 1344x768) goes on Ref2VA (queue60).

## 120 s on Ref2VA (queue66, 2026-11-04 10:55-13:52; reference 2048 px, seed 3301, 12 windows)

The same scene for two minutes in one Ref2VA continuation request (12 windows of 277 frames, 22-frame overlap, one
dialogue line per window, `lalka-ab/extra12.json`): 9807.9 s wall (817 s per window, 1.86x the 1024 px rate of arm D),
peak 71.4 GB, no seam retry (seam-check ratios 0.86-1.0), no cut, light flat (luma 49-59, steps <= 0.8 at the seams).

| face | first 10 % | last 10 % | mean over 120 s | min | windows 1-9 mean | windows 10-12 mean |
|---|---|---|---|---|---|---|
| man | 0.830 | 0.744 | 0.819 | 0.597 | 0.831 | 0.784 (3 samples with the face turned away) |
| woman | 0.804 | 0.721 | 0.779 | 0.391 | 0.799 | 0.673 |

Per window (mean), man / woman: 0.83/0.80, 0.87/0.79, 0.83/0.81, 0.85/0.82, 0.82/0.82, 0.85/0.78, 0.80/0.82,
0.82/0.78, 0.80/0.77, 0.84/0.67, 0.70/0.75, 0.81/0.60.

Reading: the identity holds for the first 100 s at the 60 s arm B level (both faces 0.77-0.87 per window), then
windows 10-11 (104-112 s) stage a move: the man turns his back and walks a step away, the woman is in profile (the 0.39
minimum is a profile, the 3 missing samples a back view), and at 114 s both are back in the reference composition
(window 12: man 0.81, woman 0.60). The dip is blocking, not a slow drift: the man's number returns to the window-1
level once he faces the camera again, the woman's final 0.60 is the one worsening to watch in a longer render. The
seam at 107 s (motion 2.6 against 3.4 before it) is the walk itself. So a 2-minute character clip on Ref2VA is a
single request with the faces held throughout, at 2.7 h per minute at 960x544 with the 2048 px reference (about
1.5 h per minute expected with the 1024 px reference of arm D). Clip
`~/Downloads/h3-prod-smoke/d120_lalka_960x544_ref2va_20261104.mp4`, numbers `lalka-ab/drift-ref2va120.txt`.

Clips: `~/Downloads/h3-prod-smoke/d60_lalka_960x544_fl2va_jeden_request_20261103.mp4` (A),
`~/Downloads/h3-prod-smoke/d60_lalka_960x544_ref2va_20261103.mp4` (B),
`~/Downloads/h3-prod-smoke/d60_lalka_960x544_fl2va_endhidden_20261103.mp4` (C),
`~/Downloads/h3-prod-smoke/d60_lalka_960x544_ref2va1024_20261104.mp4` (D),
`~/Downloads/h3-prod-smoke/d60_lalka_960x544_fl2va_endhidden_seed3201_20261104.mp4` (C2). Numbers: Spark_06 `~/h3-tp2-results/lalka-ab/`
(`drift-baseline300.txt`, `drift-fl2va60.txt`, `drift-ref2va60.txt`, `drift-fl2va60-endhidden.txt`, `drift-ref2va1024-60.txt`, `drift-fl2va60-endhidden-s3201.txt`, queue56/58/61/64/65 scripts and logs).
