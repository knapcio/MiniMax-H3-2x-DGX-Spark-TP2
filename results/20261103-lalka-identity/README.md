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

Per window (mean similarity), A vs B vs C:

| window | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| A man | 0.748 | 0.673 | 0.636 | 0.684 | 0.609 | 0.554 |
| B man | 0.784 | 0.887 | 0.847 | 0.870 | 0.864 | 0.905 |
| C man | 0.750 | 0.802 | 0.821 | 0.781 | 0.795 | 0.865 |
| A woman | 0.741 | 0.682 | 0.568 | 0.524 | 0.554 | 0.476 |
| B woman | 0.813 | 0.781 | 0.799 | 0.763 | 0.807 | 0.762 |
| C woman | 0.773 | 0.724 | 0.749 | 0.690 | 0.709 | 0.711 |

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
- The X clip (English, 1344x768) goes on Ref2VA (queue60).

Clips: `~/Downloads/h3-prod-smoke/d60_lalka_960x544_fl2va_jeden_request_20261103.mp4` (A),
`~/Downloads/h3-prod-smoke/d60_lalka_960x544_ref2va_20261103.mp4` (B),
`~/Downloads/h3-prod-smoke/d60_lalka_960x544_fl2va_endhidden_20261103.mp4` (C). Numbers: Spark_06 `~/h3-tp2-results/lalka-ab/`
(`drift-baseline300.txt`, `drift-fl2va60.txt`, `drift-ref2va60.txt`, `drift-fl2va60-endhidden.txt`, queue56/58/61 scripts and logs).
