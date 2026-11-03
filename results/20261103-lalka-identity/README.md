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

Per window (mean similarity), A vs B:

| window | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| A man | 0.748 | 0.673 | 0.636 | 0.684 | 0.609 | 0.554 |
| B man | 0.784 | 0.887 | 0.847 | 0.870 | 0.864 | 0.905 |
| A woman | 0.741 | 0.682 | 0.568 | 0.524 | 0.554 | 0.476 |
| B woman | 0.813 | 0.781 | 0.799 | 0.763 | 0.807 | 0.762 |

## Reading

- Arm A drifts exactly like the PNG-chained film's first minute: the identity loss is in the model's latent-tail
  continuation, not in the stitching. Code reason (`results/20261103-identity-drift/`): windows 2+ run as t2va and lose
  the FL2VA keyframe; the 22-frame tail plus the reference's image tokens in the prompt encoding carry too little at face
  scale (a face is ~3x3.5 model tokens at 960x544).
- Arm B holds both faces for the whole minute (the man even converges toward the reference), because the Ref2VA
  reference image conditions every window and every step. Seams, cuts, light and flicker are as good as in A. The price
  is 1.65x the render time (1.93x with the seam retry) and a partition switch (the boot probe sends a ref2va request
  there since ca923e0).
- Next: arm C = FL2VA with the hidden end anchor (`H3_TP2_ID_LOCK=end`, `H3_TP2_ID_END_EXTEND=68`, ~1.35x), to see
  whether the cheaper partition can be rescued; the X clip (English, 1344x768) goes on Ref2VA.

Clips: `~/Downloads/h3-prod-smoke/d60_lalka_960x544_fl2va_jeden_request_20261103.mp4` (A),
`~/Downloads/h3-prod-smoke/d60_lalka_960x544_ref2va_20261103.mp4` (B). Numbers: Spark_06 `~/h3-tp2-results/lalka-ab/`
(`drift-baseline300.txt`, `drift-fl2va60.txt`, `drift-ref2va60.txt`, queue56/58 scripts and logs).
