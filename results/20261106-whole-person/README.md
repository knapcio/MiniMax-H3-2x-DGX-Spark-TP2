# Whole-person persistence: the park test bed (2026-11-06)

The owner's defect was never only the face: after a character walked behind something or turned away, the
hairstyle, sometimes the hair colour and even the clothes came back different. This directory is the test bed
for a generic fix: a scene with scripted disappearances, measured on the whole figure, not the face alone.

## Scene

A 22 s 960x544 Ref2VA clip (two 277-frame windows, seed 2601, 20 steps, reference short edge 1024), rendered from
the model's own 5 s frame (`reference-640.png`): a thick oak in the middle, a woman in a red coat and a white hat
on the left, a man with grey hair, a moustache, a green corduroy jacket and a yellow scarf on the right.

- window 1: the woman walks behind the trunk, is hidden for about two seconds and comes out next to the man;
- window 2: the man turns his back to the camera for about three seconds and turns back; the woman returns behind
  the trunk to her place.

Three reference sets on one boot, everything else equal (queue97 on Spark_06):

| variant | request images | render | peak |
|---|---|---|---|
| V0 | the scene frame | 1191 s | 62.3 GB |
| V1 | scene + a head-and-shoulders card per character (`make_character_cards.py`) | 1501 s (+26 %) | 63.5 GB |
| V2 | scene + head cards + full-figure cards (`--body`) | 1881 s (+58 %) | 64.9 GB |

## Measurements

Whole person: `scripts/person_bank.py` (DINOv2 figure + upper-body embeddings of every detected person at 2 fps
against the character cards; a person is judged only when a confident face shows in the upper part, else
`unjudged`). Faces: `scripts/face_bank.py` against the scene frame + head cards and `scripts/face_drift.py --match
embed` against the scene frame. Cuts: `select=gt(scene,0.1)`.

| | V0 scene only | V1 + head cards | V2 + head + body cards |
|---|---|---|---|
| persons w1: mean best / min / unknown / unjudged | 0.865 / 0.42 / 0 / 2 | 0.874 / 0.56 / 0 / 2 | 0.883 / 0.56 / 0 / 2 |
| persons w2: mean best / min / unknown / unjudged | 0.827 / 0.57 / 0 / 4 | 0.836 / 0.56 / 0 / 1 | 0.840 / 0.57 / 0 / 4 |
| faces w1: mean / min / unknown frames | 0.868 / 0.48 / 0 | 0.870 / 0.25 / 1 | 0.908 / 0.59 / 0 |
| faces w2: mean / min / unknown frames | 0.869 / 0.72 / 0 | 0.859 / 0.53 / 0 | 0.878 / 0.56 / 0 |
| drift, woman: first 10 % -> last 10 % | 0.78 -> 0.85 | 0.78 -> 0.88 | 0.84 -> 0.89 |
| drift, man: first 10 % -> last 10 % | 0.91 -> 0.90 | 0.92 -> 0.92 | 0.94 -> 0.91 |
| cuts | 0 | 0 | 0 |

(V0's person rows re-scored offline with the face-gated rule; its live run used the first rule and reported 2 + 4
unknown frames, all back views.)

`sheet-woman-V012.jpg` and `sheet-man-V012.jpg`: the same frames of the three variants (rows V0, V1, V2) around the
two events, cropped to each character.

## What it shows

- **The short probe does not reproduce the defect.** With the scene frame alone the woman comes back from behind the
  oak in the same hat, hair and coat, and the man turns back with the same hair, moustache and scarf, in one take.
  Both events happen inside a window, where the diffusion sees the character before and after the occlusion.
- The cards lift the numbers a little and consistently: V2 has the best face means in both windows (0.908 / 0.878
  against 0.868 / 0.869), the best figure means (0.883 / 0.840 against 0.865 / 0.827) and the smallest drift of
  the woman's face; the head cards alone are in between and produced the one low frame (0.25, a half-turned face
  at the trunk). The lift costs 26 % (head cards) and 58 % (head + body cards) of the render time at 960x544.
- The whole-person scorer behaves: 0 unknown persons in all six windows, the back views and the half-hidden
  figures `unjudged` (0.42-0.57 against the frontal cards, the same range as a stranger in similar clothes on the
  120 s shop clip, which is why a figure without a face is not judged).

## The seam test: hidden at the window hand-over (queue98)

The same scene with the events moved onto the hand-over: window 1 ends with the woman behind the trunk and the man
with his back to the camera, window 2 brings both back (`tree/prompt_seam.txt`, `extra_seam.json`, seed 2601).

| | S0 scene only | S2 scene + head + body cards, guard (faces, figure embeddings) |
|---|---|---|
| the man (turned away, never hidden) | the same man | the same man |
| the woman (hidden) after the hand-over | **another person**: an open orange-brown coat over a white blouse and a blue skirt instead of the closed red coat, blonder hair, an older face | the same change (the cards do not prevent it) |
| figure embedding vs her card, after | 0.86-0.92 ("known") | 0.90 |
| face vs her head card, after | 0.76-0.78 ("known") | 0.83 |
| colour distance to her cards, after (hair + torso) | 15-27 (10 of 11 instances above 15) | 15-28 (10 of 12) |
| colour distance before the seam | 5-9 | 3-7 |
| guard verdict (bank without colour) | - | window 2 accepted: 10 faces, 0 % unknown |
| render | 1239 s | 1956 s |

`sheet-seam-S0-vs-reference.jpg`, `sheet-seam-S2-vs-reference.jpg`: the reference frame and five frames of each clip.

**This is the owner's defect**, and it is a hand-over problem: with the character absent from the overlap frames the
next window invents her from the prompt and the references, and the references lose (the same with five reference
images as with one). The embedding checks pass the invented woman (same type of person, 0.86-0.92), which is why
`scripts/person_bank.py` got the colour signature: hair, torso and legs regions placed relative to the face box,
hue and saturation each, compared with the character's cards (hue weighted by chroma, saturation by brightness, so the
shop clip's near-black clothes carry no noise); above 15 the person is `changed`. Over the four clips every window of
the same person scores 0-12 (the 120 s shop clip 0-5 in all ten windows), the invented woman 15-28.

Prevention did not come from the cards; the guard's re-sample with the colour check is the mechanism under test
next (queue100: the same request with the colour-aware bank, so window 2 is re-sampled until she comes back right or
the retries run out).
