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

Prevention did not come from the cards. The guard's re-sample with the colour check was tried next (queue100, S2c).

## What the guard saw and what the clip shows (queue100)

S2c is S2 once more with the colour check live in the guard (`H3_TP2_IDENTITY_GUARD=0.1 GROW=0.6 PERSON_SIM=0.6`). Two
things came out of it:

- **Renders are deterministic at a fixed seed.** S2 and S2c (same request, same seed) have the same `framemd5`, so the
  invented woman of S2 is not a lottery of the run but of the seed: only another seed is a new draw.
- **The guard passed window 2 (0 % unknown, colour distance 3-5 to the cards) and the clip still shows the orange coat
  and another face.** The frame below is the guard's decoded sample of window 2 on the left and the same frame of the
  finished clip on the right: the identity check runs on the raw sample inside the attempt loop, and the look-lock
  (`H3_TP2_CONT_LOCK=first`, the per-class latent statistics of every later window pulled toward window 1's tail) is applied
  to the chosen window afterwards. In this scene window 1's tail is the moment the woman is hidden and the man shows his
  back, so the lock moves the returning character toward a window without her: the coat goes red -> orange (torso hue
  3 -> 11-18, over the colour limit) and the face changes with it. The audio look-lock sits at the same place.

![window 2, the guard's frame against the clip's](frame-seam-S2c-guard-vs-clip.jpg)

Two guard defects found on the way are fixed (7bf65a5): the person bank lost its colour signatures at the first growth,
and a head crop and a body crop of one character were two characters, so a figure matched to the head crop was never
colour-checked against the body crop's torso.

## The look-lock on the saved latents (queue107)

The two saved S2c windows were replayed through the hand-over with no diffusion (`H3_TP2_REPLAY_LATENTS`, one glob per
window so that every rank finds its own copy), once per lock variant. Window 1 is the same in all five clips; the
numbers are window 2, where the woman is back.

| arm | lock | persons `changed` | colour distance max / mean | faces mean | woman's face, first 10 % -> last 10 % | UTMOS w2 |
|---|---|---|---|---|---|---|
| R0 | default (`first`, strength 1, per-class mean + std) | **10 of 35** | **27.7** / 11.7 | 0.829 | 0.84 -> 0.72 | 3.14 |
| R1 | off | 0 | 10.5 / 4.4 | 0.879 | 0.84 -> 0.84 | 3.47 |
| R2 | strength 0.5 | 0 | 14.4 / 6.3 | 0.854 | 0.84 -> 0.78 | 3.45 |
| R3 | mean only (`H3_TP2_CONT_LOCK_STD=0`) | 0 | 10.7 / 5.9 | 0.864 | 0.84 -> 0.81 | 3.14 |
| R4 | robust (`H3_TP2_CONT_LOCK_ROBUST=1`) | 0 | 11.6 / 5.7 | 0.858 | 0.84 -> 0.79 | 3.14 |

R0 reproduces S2c bit for bit (framemd5 `fa977861...`), so the replay stands for the render. `sheet-lock-replay.jpg`:
four frames of window 2 (13, 16, 19, 21.5 s), rows R0, R1, R3, R4.

![window 2 under four lock variants](sheet-lock-replay.jpg)

- **The default lock is what turns the coat orange**, and it does it along the window (the correction is ramped): red
  at 16 s, orange at 19 s and 21.5 s. With the lock off the same latents decode to the red coat and the same face in
  every frame.
- **Most of the damage is the gain**, not the shift: the mean-only lock (R3) keeps the coat (colour distance 10.7
  against 10.5 with no lock) and the face (0.864).
- **The robust lock keeps the character** (0 changed, 11.6, the coat red in every frame of the sheet): its statistics
  come from the positions that did not change between window 1's tail and the new window, so the returning figures do
  not enter them. Half strength stays just under the colour limit (14.4 of 15).
- The voice column is the audio look-lock, which rides on the same switch: window 2 scores 3.47 without it and 3.14
  with it (R2 runs it at half strength: 3.45). Window 1's tail in this scene is the quiet moment before the woman
  returns, and the lock pulls the speech of window 2 toward it (audio latent std 0.349 -> 0.255, reference 0.210).
  Followed up in `results/20261106-voice/`.

What the replay cannot show is the reason the lock exists: the slow whole-frame drift over many windows. Queue104
renders the 60 s painting clip with no lock, the default, the robust and the mean-only lock to see which of them still
holds the brightness.
