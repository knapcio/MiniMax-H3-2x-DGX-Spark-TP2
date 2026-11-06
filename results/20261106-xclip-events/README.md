# 2026-11-06: the X shop scene with the events written in (60 s, 1344x768, scene + crops, identity guard)

The identity work of 2026-11-05/06 applied end to end to the public scene: the 60 s English shop scene of
`../20261103-xclip/` (v2 lines and composition, seed 2304, Ref2VA at 1024 px) rendered with **the scene frame plus a
512 px head-and-shoulders crop per character** as references (`xclip/ref_2208.png`, `ref_2208_face0_512.png`,
`ref_2208_face1_512.png`; `../20261105-multiref/`) and with the three things the published 120 s clip got wrong
**deliberately written into the prompts**: in window 2 the woman turns to the shelves behind her for about three seconds
and turns back, in window 4 the man turns away to take a ledger from the shelf and comes back with it, in window 5 the
woman bends below the counter for about two seconds (`xclip/extra_en3.json`, one sentence each added to the v2
continuation prompts). Queue92, 03:57-07:55 on the two Sparks; the identity guard was set (`H3_TP2_IDENTITY_GUARD=0.2`)
but ran with the pre-fix code that scored no faces (`../20261105-multiref/`), so it was a no-op here: nothing below
is the guard's doing.

## Result

One continuous take, every event rendered as written, **each character comes back as the same person**
(`sheet-events.png`: the woman's turn at 12-15 s, the man's at 35-37 s with the red ledger, the woman below the counter
at 48-52 s; `sheet-xclip60v3.png`: the whole clip at 2 s steps).

| check | value |
|---|---|
| render | 13 391 s (3.7 h) for 6 windows **including one re-sample**: window 3 cut inside (cut ratio 4.23 > 1.70), attempt 2 with seed 10225 clean (seam 1.05, cut 1.14); the other windows seam 1.00-1.16, cut 1.05-1.21 |
| diffuse per 277-frame window | 1866-1963 s (92.7-100.4 s/step); the 120 s clip with the scene reference alone ran at 80-84 s/step, so the two crops cost about 15 % here; last partial window 917 s; decode 318 s |
| peak memory | 70.1 GB (head) |
| cuts (`scene > 0.1`) | 0 |
| pauses >= 1.5 s | 0 |
| speech (whisper small vs the v2 script) | word error rate **0.005** |
| faces vs the reference, embedding-matched (`face_drift.py --every 0.5 --match embed`) | man 0.704 mean, first 10 % 0.91, last 10 % 0.75, min 0.35; woman 0.775 mean, first 10 % 0.92, last 10 % 0.84, min 0.35 |
| per window (mean / min, position-matched) | man 0.79/0.57, 0.75/0.50, 0.69/0.52, 0.66/0.28, 0.64/0.36, 0.73/0.55; woman 0.79/0.45, 0.87/0.71 (6 of 11 samples undetected: turned away), 0.68/0.44, 0.87/0.71 (5 undetected: below the counter), 0.68/0.46, 0.86/0.79 |

The minima are the events themselves (the man's 0.28 in window 4 is his back; the woman's missing samples are her turn
and her bend), and the means after each event are back where they were before it. For comparison the 60 s v2 clip of
2026-11-04 (one reference at 2048 px, no events) held 0.54-0.64 means; this one, with the crops and three
disappearances, 0.70 / 0.78.

## Delivered

`~/Downloads/h3-prod-smoke/xclip/xclip60v3_raw_1344x768_20261106.mp4` (-17.7 LUFS), `xclip60v3_ambience_...`
(shop bed at -37 LUFS) and `xclip60v3_piano_...` (piano bed at -40 LUFS), mixed with `scripts/mix_bed.sh` from the
same two 16 s beds as the earlier X clips.

## What it does not show

Whether the guard catches a replacement: it scored nothing in this run. Its first real run with the uint8 fix, bank
growth and the confident-detection rule is queue94 (`../20261106-identity-guard-calibration/` for the offline
calibration on the published 120 s clip).

Files: `sheet-xclip60v3.png`, `sheet-events.png`. The prompts (`xclip/extra_en3.json`), the drift files
(`xclip/drift-v3occl*.txt/json`) and the speech report (`xclip/speech_qa_v3occl.txt`) are in `~/h3-tp2-results` on spark-06.
