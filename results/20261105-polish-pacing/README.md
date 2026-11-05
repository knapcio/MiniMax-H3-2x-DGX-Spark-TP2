# Polish dialogue pacing: three short lines per window (2026-11-05)

**Question.** The 120 s Polish doll scene of 2026-11-04 (queue66, `results/20261103-lalka-identity/`, one dialogue line
per 11.5 s window) was judged good, with two complaints: long pauses between the lines and, now and then, Russian or
wrong Polish words. The English clip for X fixed its pauses by putting three short lines into every window
(`results/20261103-xclip/`, v2). Does the same fix the Polish pacing, and does it also remove the foreign words?

**Answer: yes to both**, and the remaining single-word slips on nasal vowels are fixed by respelling them in the prompt (PL5, below). With three short lines per window the 60 s Polish scene has no pause of 1.5 s or longer
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
  `scripts/plan_dialogue_windows.py` now reports each window's words per second of new content and warns above
  2.75 words/s (this tail: 17 words in 6.0 s = 2.85; the full windows 2.26-2.64), see `docs/DIALOGUE_PLAN_MEASUREMENT.md`.
- **Cost: none.** 3434 s against 3384 s for arm D with the same geometry and reference; three lines per window is free.
- **Faces.** The woman's mean is lower than in arm D (0.69 against 0.74) and the man's equal; the per-window numbers sit
  inside the spread of the 60 s arms B-D (woman 0.56-0.81 per window there), and the speakers turn towards each other
  more when they answer each other promptly, which lowers a frontal-face metric. No face is missing in any sample and
  no window shows a step, so nothing here contradicts the identity verdict (Ref2VA 1024 px holds the faces).

## Second seed (PL4, queue76, 2026-11-05 07:20-08:45): the mispronunciations are systematic

Same prompt, extra, reference and settings, seed 3501 (the defaults boot of 07:20; `H3_TP2_FP8_QUANT_FUSE=1` was still
in the env from the lever gate, a bit-exact lever). Wall 3255.9 s (PL3 3433.9), peak 64.4 GB, no seam retry.

| | PL3 (seed 3401) | PL4 (seed 3501) |
|---|---|---|
| speech (faster-whisper small, pl) | WER 0.0816, 18/18 lines | WER 0.0816, 18/18 lines |
| pauses (raw, < -40 dB for >= 1.5 s) | 0 | 1 (1.5 s) |
| cuts | 0 | 0 |
| faces, mean man / woman (min) | 0.799 / 0.689 (0.54 / 0.43) | 0.801 / 0.710 (0.64 / 0.48) |
| transcript errors | `dziewczyna pisze` -> `dziewczy napisze`; `może` -> `meże` (twice); `pieniądze` -> `pienięże`; `porządnie` -> `pożądnie`; `wyjścia` -> `wyjście`; `czynsz` -> `czynż`; `lepszym` -> `lepsym`; `Czynsz może poczekać` dropped in the tail | `nikogo` -> `nikiego`; `może` -> `meże` (twice); `pieniądze` -> `pieniejże`; `porządnie` -> `pożądnie`; `wyjścia` -> `wyjście`; `odpowiedź` -> `obsować`; `czynsz` -> `czynż`; `lepszym` -> `lepsym`; the same half-line dropped |

Seven of the nine errors repeat word for word across the two seeds (`może`, `pieniądze`, `porządnie`, `wyjścia`,
`czynsz`, `lepszym`, the dropped tail half-line); two are seed noise. So the Polish slips are driven by the text, not by
sampling: the letters `ż`, `ą`/`ę` and `sz` in those words are read the same wrong way every time, and the 6 s tail
cannot hold three lines whichever seed. Next arm (PL5): the same lines with the four nasal / `ż` words respelled the way
they sound (`morze`, `pieniondze`, `porzondnie`) and the tail reduced to two lines; the word-error reference keeps the
intended spellings, so a WER drop means the words were heard right. Clip: `~/Downloads/h3-prod-smoke/lalka-pl4/`.

## Respelling (PL5, queue79, 2026-11-05 10:42-11:50): the nasal-vowel slips are fixed by the prompt text

Same scene, reference, settings and seed as PL3 (3401, Ref2VA 1024, 960x544, 20 steps, the serving defaults of
2026-11-05) with two changes in the prompts only: `może` -> `morze`, `pieniądze` -> `pieniondze`, `porządnie` ->
`porzondnie` (`inputs/prompt_pl5.txt`, `inputs/extra_pl5.json`), and the tail window cut from three lines to two
("Czynsz może poczekać. Dziewczyna nie." removed, 17 lines). The speech-QA reference (`inputs/dialogue_pl5.txt`) keeps
the intended spellings, so a lower word error rate means the respelled words were heard as the real ones. Wall 3271.5 s,
no seam retry.

| | PL3 (seed 3401) | PL4 (seed 3501) | PL5 (seed 3401, respelled, two-line tail) |
|---|---|---|---|
| speech (faster-whisper small, pl) | WER 0.0816, 18/18 lines | WER 0.0816 | **WER 0.0352**, 17/17 lines |
| `może` / `pieniądze` / `porządnie` | wrong / wrong / wrong | wrong / wrong / wrong | **right / right / right** |
| tail window | half-line dropped | half-line dropped | both lines spoken, speech to 60.2 s |
| pauses (raw, < -40 dB for >= 1.5 s) | 0 | 1 (1.5 s) | **0** |
| cuts | 0 | 0 | 0 |
| faces, mean man / woman (min) | 0.799 / 0.689 (0.54 / 0.43) | 0.801 / 0.710 (0.64 / 0.48) | 0.810 / 0.701 (0.54 / 0.44) |
| remaining errors | 8 | 9 | 4: `dziewczyna pisze` -> `dziewczy napisze` (PL3 had it too), `winien` -> `winiem`, `czynsz` -> `czynż`, `lepszym` -> `lepsym` |

PL5 per window, man / woman: 0.73/0.65, 0.85/0.74, 0.83/0.67, 0.85/0.72, 0.78/0.72, 0.82/0.71. Whisper wrote the
respelled words with their real spellings (`Może byli zajęci`, `dać komuś pieniądze`, `porządnie i osobiście`): the
model read `morze` as the sound of `może`, which is what a Polish ear needs. The remaining slips are `sz` clusters read
as `ż` or `s` and one `n`/`m`; `czynsz` and `lepszym` are already phonetic, so a respelling for them is not obvious and
has not been tried.

## The 118 s version (PL120v2, queue81/82/85, 2026-11-05 12:12-20:29)

The owner's 120 s scene rebuilt with everything above: 117.8 s = 11 full 277-frame windows and no tail, 34 lines (3 per
window, 1.9-2.6 words/s of new content, the planner silent), the composition in every window prompt, the three respelled
words, Ref2VA 1024 reference, seed 3401 + `H3_TP2_WINDOW_SEED_STEP=1`, 960x544, 20 steps (`inputs/*pl120v2*`).

| | 120 s, one line per window (2026-11-04) | 118 s v2, first render (queue81) | + windows 9-11 re-sampled (queue82) | + window 11 re-sampled again (queue85) |
|---|---|---|---|---|
| wall | - | 8108 s | 2168 s (8 windows replayed) | 1038 s (10 replayed) |
| pauses >= 1.5 s | 13 | 0 | 1 (1.8 s) | **0** |
| non-Polish babble | 2 windows | one 5 s line at 83.8 s (window 8) | same (replayed) | same (replayed) |
| WER (34 lines) | 0.172 (12 lines) | 0.081 | 0.089 | 0.085 |
| cuts | 0 | **1 at 88.7 s** (window 9: a close-up of the hands and the letter to the end) | **2 at 114-116 s** (window 11: a close-up of the woman) | **0** |
| faces, mean man / woman | 0.82 / 0.78 (2048 px ref) | 0.82 / 0.70 while visible, none after 88.7 s | 0.80 / 0.69 | **0.81 / 0.68**, last 10 % 0.78 / 0.62 |
| cut check ratio (`results/20261105-cut-check/`) | | window 9 1.96 | window 11 3.33 | window 11 **1.10** (windows 1-10 1.07-1.21) |

Two of three samples of a late window cut away to a close-up, with the seam check at 1.0 each time: the cut-away is a
mode of this scene (a letter on the counter, a face to look at), not of one seed. The interior cut check now catches it
in serving and re-samples the window; the two repairs cost 36 and 17 minutes through the partial latents replay instead
of 2.3 h each. What remains in the final clip: one invented half-line in window 8 (the planner's lower bound was added
after this plan), `sz` slips, a stutter in window 11 ("dlaczego ... dlaczego nie pana nazwiskiem") and one dropped
half-line. Clips (private): `~/Downloads/h3-prod-smoke/lalka-pl120v2-final/` (raw and with the shop bed), the cut
versions in `lalka-pl120v2/` and `lalka-pl120v2r/`.

## What this means for the recipe

Three short lines per 11.5 s window (and the composition restated in every window prompt) is now the dialogue
pacing for both languages; one line per window produces pauses and, in Polish, foreign babble. The last, partial
window gets as many lines as fit its length (two seeds dropped the same half-line in the 6 s tail; two lines were spoken
in full). Polish words with nasal vowels or `rz`/`ż` that the model mispronounces on every seed are written phonetically
in the prompt and kept in their real spelling in the script and the speech-QA reference (`morze`, `pieniondze`,
`porzondnie`: WER 0.082 -> 0.035). The `sz` clusters are the open Polish quality item.

Clips (private, the doll reference frame was prompted as a named actor, so not for X):
`~/Downloads/h3-prod-smoke/lalka-pl3/lalka60_pl3_3linie_na_okno_960x544_ref2va1024_20261105.mp4`,
`~/Downloads/h3-prod-smoke/lalka-pl4/`, `~/Downloads/h3-prod-smoke/lalka-pl5/`. Inputs and raw tool outputs in
`inputs/` (`speech_qa_pl3.txt`, `speech_qa_pl120.txt`, `speech_qa_pl5.txt`, `drift-pl3.txt`, `drift-pl5.txt`,
`prompt_pl5.txt`, `extra_pl5.json`, `dialogue_pl5.txt`).
