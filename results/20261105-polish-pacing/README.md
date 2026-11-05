# Polish dialogue pacing: three short lines per window (2026-11-05)

**Question.** The 120 s Polish doll scene of 2026-11-04 (queue66, `results/20261103-lalka-identity/`, one dialogue line
per 11.5 s window) was judged good, with two complaints: long pauses between the lines and, now and then, Russian or
wrong Polish words. The English clip for X fixed its pauses by putting three short lines into every window
(`results/20261103-xclip/`, v2). Does the same fix the Polish pacing, and does it also remove the foreign words?

**Answer: yes to both.** With three short lines per window the 60 s Polish scene has no pause of 1.5 s or longer
(the longest silence at any threshold is 1.4 s), the speech is Polish throughout, the word error rate halves, and the
render costs the same time as the one-line arm. The remaining errors are single mispronounced Polish words (nasal
vowels and `sz`/`rz` clusters), not foreign speech; one half-line was dropped in the short final window, which is a
planning error of this arm, not of the model.

## Arm PL3 (queue71, 2026-11-05 00:10-01:39, Spark_06 `~/h3-tp2-results/lalka60-pl3/`)

60 s at 960x544, one Ref2VA continuation request with the 1024 px reference (the identity default, arm D settings),
the doll reference frame, seed 3401 with `H3_TP2_WINDOW_SEED_STEP=1`, 20 steps, 6 windows (5 full windows of 277
frames plus a 165-frame tail window, 143 new frames), 18 Polish lines from `inputs/dialogue_pl3.txt` (3 per window) in an explicit
`continuation_prompts` list (`inputs/extra_pl3.json`); every window prompt repeats the composition (camera at the
counter end, man left, woman right, both faces in three-quarter view) as the X clip does. Wall 3433.9 s (arm D with
one line per window: 3384 s), peak 64.4 GB, seam-check ratios 1.0 in every window (no retry), diffuse 575-587 s per
full window and 295 s for the tail, decode 183 s.

## Comparison with the one-line-per-window arm

The 120 s clip (queue66, 12 windows, 2048 px reference, seed 3301) was measured again today with the same tools
(`scripts/speech_qa.py`, faster-whisper small, language pl, reference `inputs/dialogue_pl120.txt`; silencedetect on the
raw dialogue track; both clips carry no bed).

| | 120 s, one line per window (queue66) | 60 s, three lines per window (PL3) |
|---|---|---|
| pauses (raw, < -40 dB for >= 1.5 s) | **13**: 2.6, 2.1, 3.8, 2.3, 1.5, 1.5, 2.0, 2.0, 1.6, 2.1, 2.9, 1.7, 1.6 s | **0** |
| pauses (< -25 dB for >= 1.0 s) | 23, up to 5.7 s; one 6.5 s gap at 107-113 s | 9, all 1.0-1.3 s |
| longest silence at -20 dB | > 5 s | 1.4 s |
| language | pl, **but two windows of non-Polish babble**: 80-90 s ("Jebra śluciewcu. Diemere Gruzdra Stiewsmu Terefo.") and 97-104 s ("Adzieno mięsło się Moriowie siantyskotro.") | pl throughout, no foreign words |
| word error rate against the script | 0.172 (12 lines) | **0.082** (18 lines) |
| errors | the two babble inserts, "pieniądze" mangled, "pracownikach" -> "protunikach", "stał" -> "wstał", "nikogo" -> "nikiego" | mispronounced single words: "może" -> "meże" (twice), "pieniądze" -> "pienięże", "porządnie" -> "pożądnie", "czynsz" -> "czynż", "lepszym" -> "lepsym"; the half-line "Czynsz może poczekać." dropped |
| cuts (scene detector) | 0 | 0 |
| faces, mean similarity man / woman | 0.819 / 0.779 (2048 px reference) | 0.799 / 0.689 (1024 px reference; arm D, same settings, one line per window: 0.805 / 0.737) |

PL3 per window, man / woman: 0.72/0.63, 0.85/0.71, 0.83/0.69, 0.81/0.74, 0.80/0.73, 0.78/0.63; no face missing in
any sample; window 1 is the weakest, as in every doll arm. Line timing from the transcript: 18 lines at 0.3-60.2 s,
the gaps between consecutive lines 0.0-0.5 s, speech until the last frame.

## Reading

- **The pauses and the foreign words are one symptom.** A window of 11.5 s with one 3-4 s line leaves 7-8 s that the
  model has to fill, and it fills them either with silence (the 2-4 s pauses) or with invented speech that is not
  Polish (two of the twelve windows of the 120 s clip). With three lines per window the model has no spare time: no
  pause reaches 1.5 s and no window babbles. The X clip v2 showed the same for English; this arm shows it holds for
  Polish, where the babble was the owner's main complaint.
- **The remaining errors are pronunciation, not language.** Five of the six PL3 errors are one word each, all with a
  nasal vowel (ę, ą) or an `sz`/`rz` cluster rendered wrong ("meże", "pienięże", "pożądnie", "czynż", "lepsym"). These
  are the "wrong Polish words" the owner hears; pacing does not change them. They are worth a separate experiment
  (for example the same 18 lines with the hard words avoided or respelled, same seed) before concluding anything.
- **Plan the lines against the real window length.** 60 s is 1440 frames = 5 full windows (1297 frames, 54.0 s) plus a
  165-frame tail window (6.9 s, 6.0 s of new content), and the tail got the same three lines as a full window: the model
  spoke two of them and dropped the half-line "Czynsz może poczekać." The lines for the last window must fit its length
  (one line for a 6 s tail), or the clip length should be a whole number of windows (54.0 s for 5, 64.7 s for 6).
- **Cost: none.** 3434 s against 3384 s for arm D with the same geometry and reference; three lines per window is free.
- **Faces.** The woman's mean is lower than in arm D (0.69 against 0.74) and the man's equal; the per-window numbers sit
  inside the spread of the 60 s arms B-D (woman 0.56-0.81 per window there), and the speakers turn towards each other
  more when they answer each other promptly, which lowers a frontal-face metric. No face is missing in any sample and
  no window shows a step, so nothing here contradicts the identity verdict (Ref2VA 1024 px holds the faces).

## What this means for the recipe

Three short lines per 11.5 s window (and the composition restated in every window prompt) is now the dialogue
pacing for both languages; one line per window produces pauses and, in Polish, foreign babble. The last, partial
window gets as many lines as fit its length. The pronunciation slips on nasal vowels and `sz`/`rz` are the open Polish
quality item.

Clip (private, the doll reference frame was prompted as a named actor, so not for X):
`~/Downloads/h3-prod-smoke/lalka-pl3/lalka60_pl3_3linie_na_okno_960x544_ref2va1024_20261105.mp4`. Inputs and raw
tool outputs in `inputs/` (`speech_qa_pl3.txt`, `speech_qa_pl120.txt`, `drift-pl3.txt`).
