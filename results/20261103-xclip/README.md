# 2026-11-03: the English shop scene for X (60 s, 1344x768, Ref2VA)

Goal: a 60 s English dialogue clip with the two characters of the "Lalka" shop scene (an invented 1880s shop owner and a
young customer, no likeness of a real actor), stable faces, no language slips, short pauses between lines, and a shop
ambience or light music under the voices. Everything below ran on the two-Spark TP2 stack with the serving defaults of
2026-11-03 plus the Ref2VA partition and the 2048 px reference (`H3_TP2_REF_SHORT_EDGE=2048`), the identity setting proven
in `../20261103-lalka-identity/`.

## Inputs

- Reference frame: a 5 s 1344x768 T2VA render (seed 2208, `xclip/reference_prompt3.txt`), frame 0 as PNG. Three prompt
  rounds were needed: the first reference (seed 2201) had the woman in profile and a 1930s telephone on the counter;
  "no telephone" (seeds 2205/2206) still produced one; stating the composition instead ("the camera stands at the end of
  the counter, the man on the left, the woman on the right, both faces fully visible in three-quarter view", a brass
  inkwell and a scale as the props) gave usable frames (2207/2208). Lesson: a negation summons the object; name the
  composition and other props instead.
- Main render: `xclip/prompt_en.txt` + `xclip/extra_en.json` (six 277-frame windows, 22-frame overlap, two English lines
  per window in `<d>[English] ...</d>` tags, `audio_flow_shift` 3.0), seed 2204, 20 steps, Ref2VA, reference 2048 px.
- Beds (speech-free 16 s 960x544 T2VA renders, looped): shop ambience seed 2202 (`ambience_prompt.txt`), solo piano seed
  2203 (`music_prompt.txt`).

## Result (queue60b, 19:57-23:44)

| item | value |
|---|---|
| render | 13 613.7 s for 60 s (six windows: diffuse 2932 / 2336 / 2157 / ... s; the first window includes the 2048 px reference encode), peak head 74.9 GB |
| speech (faster-whisper small, `scripts/speech_qa.py`) | language en (p = 1.0), **word error rate 0.007** against the 12-line script; all 12 lines present and in order; no Russian or other slip |
| pauses between lines (raw, < -40 dB for >= 1.5 s) | 7: 1.8, 1.7, 2.5, 2.1, 2.1, 1.9, 2.2 s |
| scene cuts (score > 0.1) | 2: 17.0 s (0.36) and 57.0 s (0.32) |
| seams (277-frame windows) | ratios 0.35-1.20 of the local motion, luma steps <= 2.9; no retry needed (ratios 0.81-1.11) |
| faces vs the reference (`scripts/face_drift.py`) | man: windows 1-6 mean 0.47 / 0.41 / 0.46 / 0.47 / 0.46 / 0.53; woman: 0.15 / 0.31 / 0.59 / 0.52 / 0.50 / 0.64 |

What the frames show: window 1 ignored the reference composition. The woman stands on the LEFT in profile, the man on the
right, a few extra customers behind them. At 17.0 s (inside window 2) the model cuts to the reference composition (man
left, woman right, both three-quarter to the camera) and holds it to 57 s, where it cuts once more to a two-shot close-up
for the last lines. The face numbers follow that: the woman's similarity is 0.15 while she is in profile on the wrong
side, 0.5-0.64 once the composition matches; the man sits at 0.41-0.53 throughout (he is small in the frame; the metric
is lower on wide shots than on the 960x544 doll clips, compare arms, not absolute values). Faces do not drift within the
60 s: first-to-last 10 % is +0.02 (man) and +0.50 (woman, the composition change).

Audio: the dialogue is at -23.9 LUFS; the ambience bed came out at -44.6 LUFS and the piano at -24.8 LUFS, so the first
mix (`mix_ambience.sh`, a fixed -16 / -20 dB offset) left the ambience inaudible (-64 dB in the pauses).
`scripts/mix_bed.sh` (`xclip/mix_v2.sh` on Spark_06) normalises the bed to a target loudness first (loudnorm to -23 LUFS,
then an offset; `loudnorm` refuses `TP` below -9) and ducks it under speech (sidechain, ratio 4): ambience at -37 LUFS,
piano at -40 LUFS. With the beds audible no pause in either mix is below -45 dB for 1 s; the mixes measure -24.4 / -24.0
LUFS, peaks -7 dB.

Delivered (`~/Downloads/h3-prod-smoke/xclip/`): `xclip60_raw_1344x768_ref2va_20261103.mp4`,
`xclip60_ambience_1344x768_20261103.mp4`, `xclip60_piano_1344x768_20261103.mp4`.

## v2 (queue68, queued 00:10 behind the Ulysses gate)

Two fixes, both in the prompt: the composition sentence of the reference prompt goes into the main and every continuation
prompt ("the camera stands at the end of a polished wooden counter, on the left the man ..., on the right the woman ...,
both faces fully visible in three-quarter view"), and three short lines per window instead of two (18 lines,
`xclip/dialogue_en2.txt`, "answering each other promptly") against the 1.7-2.5 s pauses. Same reference (2208), seed
2304. The queue runs face drift, speech QA, both normalised mixes and the pause check itself.

## Files on Spark_06 (`~/h3-tp2-results/`)

`xclip60/` (render, timing, memory CSVs), `xclip/` (prompts, references 2201-2208, `dialogue_en*.txt`, `drift*.txt`,
`speech_qa*.txt`, `mix_v2.sh`, mixes), `xclip-ambience/`, `xclip-music/`, `queue60b-xclip.sh`, `queue68-xclip2.sh`.
