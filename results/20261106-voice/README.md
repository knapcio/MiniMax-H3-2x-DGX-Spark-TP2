# Voice naturalness: where the "robotic" voices come from (2026-11-06)

The owner hears the generated voices as robotic, "like AI", in the Polish doll clips and in the park clips. This directory
holds the measurements that located the loss. All renders on the two-Spark TP2 stack, 960x544, 20 steps unless noted.

## Two predictors

- **DNSMOS** (`speechmos`, OVRL / SIG / BAK / P.808): signal quality against noise and distortion. It barely moves when a
  voice turns flat or synthetic.
- **UTMOS22** (`tarepan/SpeechMOS`, trained on human naturalness ratings of synthetic speech): the one that moves with the
  complaint. Natural recordings score about 4.0-4.4, good TTS 3.8-4.2, poor TTS below 3.

`scripts/voice_naturalness.py` prints both per continuation window, plus the pitch spread (pyin F0, semitones).

## 1. The model's voice in a single window is fine

A 11 s T2VA bookshop monologue (one 26-word English line, seed 2701; queue99) and the same line in Polish. Every lever
that could plausibly change the voice was tried on the same seed:

| arm | change | DNSMOS OVRL / SIG | UTMOS | WER | render |
|---|---|---|---|---|---|
| A-en | baseline, 20 steps, audio_flow_shift 3.0 | 3.44 / 3.71 | 4.34 | 0.00 | 547 s |
| B-en | 40 steps | 3.56 / 3.78 | 4.36 | 0.04 | 988 s |
| C15-en | audio_flow_shift 1.5 | 3.46 / 3.69 | 4.35 | 0.04 | 511 s |
| C50-en | audio_flow_shift 5.0 | 3.53 / 3.75 | 4.35 | 0.04 | 508 s |
| D-en | a voice description in the prompt ("warm, natural and unhurried, with the small breaths, pauses and changing intonation of a real conversation, recorded close with a clean studio microphone") | 3.54 / 3.76 | 4.36 | 0.04 | 546 s |
| A-pl | baseline, Polish | 3.39 / 3.65 | 4.14 | 0.05 | 515 s |
| D-pl | voice description, Polish | 3.38 / 3.61 | 3.82 | 0.09 | 512 s |

Nothing here is outside the noise (the WER differences are "ten" vs "10" and one Polish ending): the five English arms
land at UTMOS 4.34-4.36, the level of natural recordings. The pitch spread is 2.0-4.6 semitones, a normal conversational
range. Steps, the audio shift and a voice description are not the lever; the single-window voice is not the problem.

## 2. The loss is in the continuation windows

The same predictor per 277-frame window on four stitched clips (window k starts at k x 10.6 s):

| clip | windows | UTMOS window 1 | UTMOS windows 2+ (mean / min) | DNSMOS SIG window 1 / rest |
|---|---|---|---|---|
| tree-seam S0 (22 s, park, two speakers) | 2 | 3.40 | 2.77 / 2.77 | 3.22 / 3.23 |
| tree-seam S2 (22 s, park, cards + guard) | 2 | 3.33 | 3.13 / 3.13 | 3.43 / 3.26 |
| doll 60 s PL5 (Polish dialogue) | 6 | 3.65 | 3.02 / 2.73 | 3.56 / 3.53 |
| doll 118 s pl120v2 (Polish dialogue) | 11 | 3.64 | 2.89 / 2.61 | 3.67 / 3.40 |

Every continuation window scores 0.4-1.0 below window 1 while the signal-quality score stays nearly flat: the later
windows lose naturalness, not fidelity. In the 118 s clip windows 4, 6, 9 and 10 sit at 2.6-2.7. Listening cuts of
window 1 and windows 9-10 of that clip were given to the owner.

Two things happen to the audio at every hand-over in the default recipe: the previous window's tail rows are pinned as
target rows (`H3_TP2_CONT_EDIT=2`, audio rows included) and the audio look-lock pulls every later window's audio latents
toward the first window's tail statistics (`H3_TP2_CONT_LOCK_AUDIO=1`, a per-dimension affine with the gain clamped to
0.5-2.0). The lock's log lines show how much it moved the audio: x1.37 on the latent std in the tree windows 2, only
x1.00-1.04 in the doll 118 s clip, so the lock alone cannot explain the doll clip.

## 3. The A/B (queue101)

A 22 s two-window version of the bookshop monologue (two lines per window, seed 2701, FL2VA) rendered with the default
hand-over and with the default without the audio lock (`H3_TP2_CONT_LOCK_AUDIO=0`), scored per window; plus two more
seeds of the single window for the seed spread of the predictor. Upstream's hand-over (`H3_TP2_CONT_EDIT=0 CONT_RETRY=0
CONT_LOCK=off`) did not render on the FL2VA head ("continuation requires Ref2VA request execution"): it is in queue103
on Ref2VA.

| clip | UTMOS window 1 | UTMOS window 2 | voiced fraction w1 / w2 | DNSMOS SIG w1 / w2 | WER | render |
|---|---|---|---|---|---|---|
| default hand-over (audio lock on) | 4.34 | 4.05 | 0.48 / 0.24 | 3.72 / 3.61 | 0.00 | 1158 s |
| audio lock off | 4.34 | 4.10 | 0.48 / 0.24 | 3.72 / 3.60 | 0.00 | 1034 s |
| single window, seed 2702 | 4.40 | - | 0.48 | 3.71 | 0.08 | 511 s |
| single window, seed 2703 | 4.45 | - | 0.51 | 3.82 | 0.04 | 512 s |

- **The audio lock is not the cause.** It moved the audio latent std of window 2 by 1 % (0.404 -> 0.400, reference
  0.390) and the UTMOS by 0.05. Window 1 is identical in both clips (same seed, deterministic).
- **The hand-over costs 0.25-0.3 here, not 0.4-1.0**: window 2 of the monologue scores 4.05-4.10, well above the 2.6-3.1
  of the park and doll clips in section 2. The seed spread of the single window is 0.11 (4.34-4.45).
- Window 2 is half as voiced as window 1 (0.24 against 0.48: the second pair of lines is shorter and the model leaves
  pauses). UTMOS rates speech, and a window with more silence scores lower on its own, so part of the 0.25-0.3 may be
  the text, not the hand-over. Queue103 renders the window-2 text alone as a single window (two seeds) to separate the
  two, and the 22 s monologue on Ref2VA with our hand-over against upstream's.
- The 118 s doll clip (Polish, 11 windows, 2.6-2.7 in four windows) is the clip the owner heard. Its drop is larger
  than anything in this English probe; the Polish single window already starts lower (4.14), and the drop may grow with
  the window count or with the pinned tail of a Polish line. Measured next on a 60 s Polish clip once the controls are in.

## 4. Controls (queue103, running)

1. The window-2 text alone (`voice/prompt_cont2_en.txt`, 11 s, seeds 2702 and 2703, FL2VA): the single-window score of
   the shorter lines. If it lands at 4.05-4.10 the hand-over loses nothing on this clip.
2. The 22 s monologue on Ref2VA (reference = the first frame of the default clip): default hand-over against upstream's
   `H3_TP2_CONT_EDIT=0 CONT_RETRY=0 CONT_LOCK=off`, per window.
