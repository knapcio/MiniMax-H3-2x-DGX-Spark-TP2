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
60 s: first-to-last 10 % is +0.02 (man) and +0.50 (woman, the composition change). The table matches faces by
horizontal position, so in window 1 (sides swapped) it compared the wrong pairs; `--match embed` (greedy best similarity,
added for this case) gives the man 0.59 / 0.53 in windows 1-2 and the woman 0.15 / 0.30 (in profile she is not the
reference face for the metric either), 0.46-0.64 for both once the composition matches.

Audio: the dialogue is at -23.9 LUFS; the ambience bed came out at -44.6 LUFS and the piano at -24.8 LUFS, so the first
mix (`mix_ambience.sh`, a fixed -16 / -20 dB offset) left the ambience inaudible (-64 dB in the pauses).
`scripts/mix_bed.sh` (`xclip/mix_v2.sh` on Spark_06) normalises the bed to a target loudness first (loudnorm to -23 LUFS,
then an offset; `loudnorm` refuses `TP` below -9) and ducks it under speech (sidechain, ratio 4): ambience at -37 LUFS,
piano at -40 LUFS. With the beds audible no pause in either mix is below -45 dB for 1 s; the mixes measure -24.4 / -24.0
LUFS, peaks -7 dB.

Delivered (`~/Downloads/h3-prod-smoke/xclip/`): `xclip60_raw_1344x768_ref2va_20261103.mp4`,
`xclip60_ambience_1344x768_20261103.mp4`, `xclip60_piano_1344x768_20261103.mp4`.

## v2: PASS (queue68, 2026-11-04 02:41-06:01)

Two fixes, both in the prompt: the composition sentence of the reference prompt goes into the main and every continuation
prompt ("the camera stands at the end of a polished wooden counter, on the left the man ..., on the right the woman ...,
both faces fully visible in three-quarter view"), and three short lines per window instead of two (18 lines,
`xclip/dialogue_en2.txt`, "answering each other promptly") against the 1.7-2.5 s pauses. Same reference (2208), seed 2304.

| item | v1 (seed 2204) | v2 (seed 2304) |
|---|---|---|
| render | 13 613.7 s | 12 030.6 s (six diffuse phases 2086-2109 s), peak head 75.3 GB |
| speech (faster-whisper small) | en, WER 0.007, 12/12 lines | en, **WER 0.005**, 18/18 lines in order |
| pauses (raw, < -40 dB for >= 1.5 s) | 7 (1.7-2.5 s) | **2 (1.5, 1.7 s)** |
| scene cuts | 2 (17.0 s, 57.0 s) | **none**: one continuous take, the reference composition from frame 0 |
| seams (ratio to local motion, luma step) | 0.35-1.20, <= 2.9 | 0.19-0.57, <= 0.2 (no retry) |
| faces vs the reference, per window (position match) | man 0.47 / 0.41 / 0.46 / 0.47 / 0.46 / 0.53; woman 0.15 / 0.31 / 0.59 / 0.52 / 0.50 / 0.64 | man 0.77 / 0.57 / 0.65 / 0.54 / 0.63 / 0.68; woman 0.76 / 0.47 / 0.59 / 0.54 / 0.58 / 0.76 |
| overall mean / min | 0.46 / 0.30 (man), 0.43 / 0.05 (woman) | 0.64 / 0.44 (man), 0.62 / 0.37 (woman) |

The frames: man left, woman right, three-quarter to the camera from the first frame to the last, the same counter, lamp,
scale and inkwell, customers only far behind. The woman turns to the man (profile) for a few seconds in windows 2-5
(3-5 "missing" samples per window), which is where her minima come from, not a change of face. First-to-last 10 %:
man 0.90 -> 0.68, woman 0.89 -> 0.77 (the first second is the reference pose itself).

Delivered (`~/Downloads/h3-prod-smoke/xclip/`): `xclip60v2_raw_1344x768_ref2va_20261104.mp4`,
`xclip60v2_ambience_1344x768_20261104.mp4` (-21.2 LUFS), `xclip60v2_piano_1344x768_20261104.mp4`. v2 is the clip for X.

Lessons: the reference image fixes the faces, the prompt fixes the blocking (state the camera position and who stands
where in every window's prompt); three short lines per 11.5 s window keep the pauses under 2 s; name props instead of
negating them; normalise beds by loudness.

## 120 s (queue70 + queue72, 2026-11-04 16:01 - 2026-11-05 02:22)

The same recipe as v2 for two minutes: 1344x768, Ref2VA with the 2048 px reference 2208, seed 2404 with a seed step per
window, 12 windows of 277 frames, 34 English lines (`xclip/dialogue_en120.txt`, three per window, two in the last) with
the composition restated in every window prompt (`xclip/extra_en120.json`).

The render itself (queue70) diffused all 12 windows in 7.9 h (seam retries on windows 11 and 12, ratios 2.91 -> 1.05 and
3.20 -> 1.06) and then **died in the decode**: the API server's per-step wait `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` (upstream
600 s, never set by this recipe) fired during the 642 s VAE decode of 2880 frames at 1344x768, HTTP 500, worker torn
down. The saved window latents (`H3_TP2_SAVE_LATENTS`, 14 files per rank including the two retries) made the re-render
unnecessary: `scripts/env.sh` now defaults the wait to 7200 s and `H3_TP2_REPLAY_LATENTS` (commit e1a4971) replays the
files through the unchanged hand-over, stitch, decode and mux path. queue72 on a fresh Ref2VA 2048 boot: `REPLAY 14/14`,
every seam-check ratio identical to the original journal, the window-12 latents digest identical, decode 632.9 s,
HTTP 200 after 1263 s (prompt encodes 100 s, replay 0 s, decode 633 s, mux), peak 79.3 GB.

| check | 120 s (replayed render) | v2 60 s for comparison |
|---|---|---|
| speech (faster-whisper small, 34-line script) | en (p = 1.0), **WER 0.054**, 34/34 lines in order; errors: "Nowak" -> "Novak", "shop" -> "sharp" once, and one invented English line at 84-88 s ("You're still mad over history, it's a waste for you to ask this") | en, WER 0.005, 18/18 |
| pauses (raw, < -40 dB for >= 1.5 s) | **0** (longest 1.2 s) | 2 (1.5, 1.7 s) |
| cuts (scene score > 0.1) | 0, one continuous take | 0 |
| faces vs the reference, mean over the clip (position / embedding match) | man 0.54 / 0.55, woman 0.63 / 0.60; woman missing in 2-9 of 11 samples per window | man 0.64, woman 0.62 |
| composition on a 12-frame grid (every 10 s) and the 104-119 s tail | camera at the counter end, man left, woman right, lamp and scales on the counter, customers moving in the background, same light for 120 s | same |
| audio level | dialogue -13.2 LUFS, peak -0.1 dBFS (louder than v2's -21.1 LUFS) | -21.1 LUFS |

The face numbers are lower than v2's because the woman spends much of the clip turned towards the man (profile =
"missing" or a low similarity), not because the faces change: on the grid both faces are the reference faces in every
window, and the man's number returns to 0.6 whenever he faces the camera. The invented line at 84-88 s is the English
form of the window-filling seen in Polish (`results/20261105-polish-pacing/`): window 9 (85-96 s) had the shortest
three lines of the script.

Delivered (`~/Downloads/h3-prod-smoke/xclip/`): `xclip120_raw_1344x768_ref2va_20261105.mp4`,
`xclip120_ambience_1344x768_20261105.mp4` (bed at -37 LUFS), `xclip120_piano_1344x768_20261105.mp4` (bed at -40 LUFS);
with the hot dialogue the beds sit about 24 dB under the speech and are audible only in the pauses (no pause in either
mix is below -40 dB for 1 s). Lower the dialogue (or raise the beds to about -27 / -30 LUFS) if a fuller bed is wanted.

## Files on Spark_06 (`~/h3-tp2-results/`)

`xclip60/` (render, timing, memory CSVs), `xclip/` (prompts, references 2201-2208, `dialogue_en*.txt`, `drift*.txt`,
`speech_qa*.txt`, `mix_v2.sh`, mixes), `xclip120/` (the failed render), `xclip120r/` (the replayed clip), `latents/`
(the saved window latents), `xclip-ambience/`, `xclip-music/`, `queue60b-xclip.sh`, `queue68-xclip2.sh`,
`queue70-xclip120.sh`, `queue72-xclip120-replay.sh`.
