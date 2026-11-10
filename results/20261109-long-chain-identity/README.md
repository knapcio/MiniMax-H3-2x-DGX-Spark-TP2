# Faces over two minutes: the 1024 px reference alone drifts after 90 s, a crop per character holds (2026-11-09/10)

The owner looked at the 118 s speech-guard clips of the evening and saw the characters turn into other people. This
directory holds what was measured about it on the two-Spark TP2 stack (960x544, 20 steps, the Polish doll scene with
the 120 s script `lalka-ab/prompt_pl120v2.txt` + `extra_pl120v2.json`, seed 3701, serving defaults of 2026-11-09,
rev 779102a), what the older clips say, and the first countermeasure that worked.

Tools: `scripts/face_drift.py` (facenet VGGFace2 cosine against the reference frame, one sample per second, means per
window), `scripts/window_detail.py` (variance of the luma Laplacian per window, whole frame and fixed face boxes:
woman x 520-700 / y 80-240, man x 220-400 / y 60-220; grain counts as detail, so read trends, not absolutes).

## 1. The three 118 s chains of seed 3701

| arm (queue) | references | samplings | render | woman, similarity per window w1 .. w11 | man | woman face sharpness w1 / w6 / w11 | frame sharpness w1 / w11 |
|---|---|---|---|---|---|---|---|
| plain (queue124) | scene 1024 px | 11 | 6566 s | 0.71 0.72 0.66 0.66 0.66 0.63 0.69 0.69 0.68 0.65 **0.60** | 0.65 0.85 0.84 0.84 0.79 0.79 0.74 0.81 0.82 0.79 0.75 | 312 / 214 / 196 | 186 / 158 |
| speech guard 0.01 (queue124) | scene 1024 px | 19 | 11 519 s | 0.71 0.72 0.66 0.64 0.63 0.60 0.63 0.60 0.57 0.58 **0.61** | 0.65 0.85 0.84 0.83 0.78 0.82 0.71 0.77 0.76 0.80 0.69 | 311 / 200 / 197 | 186 / 151 |
| **cards (queue129)** | scene + 2 head-and-shoulders crops, all 1024 px | 13 (window 3 re-sampled twice by the seam check) | 9802 s (712 s per window, +25 % per step; decode 361 s) | **0.76 0.80 0.76 0.76 0.75 0.77 0.77 0.77 0.78 0.75 0.66** | **0.82 0.81 0.83 0.83 0.83 0.85 0.82 0.84 0.81 0.83 0.80** | 325 / 262 / 255 | 156 / 165 |

- With the scene reference alone the woman is another person by the second minute (0.71 -> 0.60, the contact sheet
  shows a rounder, older face) and the picture softens (frame 186 -> 158, her face 312 -> 196). Plain and guarded
  renders agree, so this is the chain, not the guard (they share windows 1-3; the guard's re-sampled windows differ).
- The raw window latents saved by the plain chain keep their per-channel spread (std 0.70 in every window) and their
  high-pass energy rises (0.35 -> 0.38): the drift is in structure, not in latent statistics, so a lock on
  statistics has nothing to hold on to.
- **With a crop per character both faces stay the same person for the whole 118 s**: the woman 0.75-0.80 in windows
  1-10 (0.66 in window 11, a three-quarter turn at the end), the man 0.80-0.85 flat, and the face sharpness no longer
  falls (her face 255 at the end against 196). Cost +25 % per step (35.3-36.0 s against 27.8-29.3) and one more
  seam retry on this seed. Speech is unchanged in kind: window 8 babbles on this seed with or without the crops (CER
  0.70 / 0.39 / 0.25 in the three arms' first attempts).

## 2. What the older clips say (scored on 2026-11-09 23:00 with the same tools)

| clip | setting | woman per window | man per window | frame sharpness per window |
|---|---|---|---|---|
| voicepl-def-s3501 (64.6 s) | 1024 px, defaults | 0.69 0.74 0.76 0.72 0.71 0.72 | 0.79 0.82 0.82 0.82 0.73 0.81 | 272 297 287 288 268 264 |
| voicepl-def-s3601 | same | 0.60 0.63 0.60 0.61 0.65 0.69 | 0.69 0.82 0.81 0.81 0.77 0.81 | 277 293 308 294 282 258 |
| voicepl-default-64 (s3401) | same | 0.64 0.73 0.63 0.72 0.74 0.74 | 0.73 0.86 0.83 0.84 0.80 0.85 | 238 257 249 249 242 233 |
| voicepl-toff-s3401/3501/3601 | local RoPE time (`toff0`) | 0.64 0.73 0.67 0.68 0.73 0.75 / 0.69 0.74 0.69 0.71 0.73 0.72 / 0.60 0.64 0.60 0.59 0.62 0.69 | as the defaults within 0.03 | as the defaults |
| voicepl-cap-s3401/3501/3601 | RoPE shift capped (`toffcap` 425) | 0.64 0.73 0.65 0.72 0.73 0.74 / 0.69 0.74 0.73 0.74 0.72 0.73 / 0.60 0.63 0.61 0.58 0.66 0.67 | as the defaults within 0.04 | as the defaults |
| fresh-pl6-w2..w6 (single windows) | each window sampled fresh | 0.61 0.69 0.65 0.59 0.64 | 0.72 0.75 0.79 0.63 0.80 | 246 238 241 222 241 |
| lalka118-pl120v2r (118 s, seed 3601, 2026-11-05 code) | 1024 px | 0.66 0.73 0.66 0.72 0.75 0.69 0.75 0.65 0.65 0.60 0.68 | 0.74 0.85 0.84 0.84 0.78 0.84 0.81 0.83 0.81 0.83 0.64 | 237 257 253 255 247 245 260 246 250 253 219 |
| lalka120-ref2va (120 s, 2048 px, 2026-11-04) | 2048 px | 0.82 0.76 0.78 0.83 0.79 0.82 0.75 0.81 0.76 0.79 0.69 0.70 0.63 | 0.83 0.89 0.82 0.85 0.83 0.85 0.80 0.83 0.81 0.82 0.63 0.82 | 163 165 171 173 176 173 178 146 159 138 193 192 |

- Up to 60 s nothing drifts: the three default chains are flat, and moving the window's RoPE time (`toff0`, `toffcap`)
  changes the faces by less than the window-to-window noise. So the RoPE distance between window and reference is
  not what limits the first minute; whether it limits the second is what the window-11 screen below is for.
- A fresh window (no chain) scores where the chain's windows score: at 60 s the chain has not lost anything against a
  fresh sample.
- Earlier 118 s chains at 1024 px lose the woman late as well (0.75 -> 0.60-0.68 after 90 s); the 2048 px clip held
  both faces through window 9 and dipped in a staged turn. The softening of seed 3701 (frame 186 -> 158) is not in
  the other chains: it is a property of that seed's pictures, not of the chain.

## 3. A host finding on the way: CPU load starves the GPU

While queue128b rendered, a scoring job (facenet + numpy over 20 clips, `nice -n 19`) ran on the head for three
minutes. The host monitor shows the GPU falling from 82-84 W / 2125-2177 MHz to 47-52 W / 1800 MHz at 96 %
utilisation, the CPU at 95 C, and the window that overlapped taking 13:37 instead of 12:54. Killing the job restored
83 W / 2119 MHz within a minute (`slowcatch-1109-2152.log`, hostmon 21:50-21:53). The GB10 shares one power and
thermal budget between CPU and GPU, and `nice` does not change that. Nothing may run on a rendering Spark; the scoring
of this directory was done on the Mac. It is also the first concrete mechanism for the unexplained +25 % renders.

## 4. Running (2026-11-10 02:27, queue131, rev 50035f8)

The window-11 screen: windows 1-10 of the plain chain replayed from their saved latents, window 11 sampled again with
its own seed under one lever at a time, so every arm is paired with the plain window 11 (woman 0.60, man 0.75, face
sharpness 196 / 118). Arms: control (must reproduce the plain clip byte for byte), hand-over noise 0.15 and 0.35
(`H3_TP2_CONT_EDIT_NOISE`), reference images following the window in RoPE time (`H3_TP2_REF_FOLLOW_WINDOW=1`), the
low-noise ladder tail (`H3_TP2_LADDER_TAIL=0.025,0.0125`), the RoPE shift capped at window 6 (`toffcap` 2125), the cap
plus noise 0.35, a 2048 px reference, and a replay of all eleven windows with the look-lock off. Results follow here.

Clips: `lalka118-plain-s3701`, `lalka118-guard-s3701`, `lalka118-cards-s3701` on Spark_06 (copies in the owner's
Downloads); scores `voice/drift_*.json`, this directory's `cpu1/` tables came from `scratchpad` scoring on the Mac.
Queues: `queue124-long.sh`, `queue129-cards118.sh`, `queue131-screen11.sh`.
