# Continuation window length: 192 frames instead of 277 (2026-11-05)

**Question.** The denoising step time of a continuation window grows faster than its length (a 277-frame window at
1344x768 runs 80-84 s per step, the 141-frame second window of the 16 s gate 26 s, the 124-frame 5 s calibration 20 s), so shorter windows should cut the time per generated
frame. What does a 60 s clip cost and look like with 192-frame windows (9 windows, 8 hand-overs) against the default
277 (6 windows, 5 hand-overs), same scene, same seed?

**Answer.** At 960x544 the 192-frame windows render the clip 11 % faster (3054 s against 3434 s) and the per-frame
denoising cost falls 13 %. The clip is one continuous take with no cut and no seam retry, the man's face holds as well
as before, but the woman's face is weaker (mean 0.61 against 0.69), the word error rate rises (0.14 against 0.08, new
single-word slips), and the 2.4 s tail window, which the plan left without a line, was filled with an invented
Polish-sounding line. The default stays 277. The arm adds one rule to the planner: a window with too few words warns
(`--min-words-per-second`, default 1.0), because spare time in a window becomes pauses or invented speech.

## Arm W192 (queue78, 2026-11-05 09:07-10:27, Spark_06 `~/h3-tp2-results/lalka60-w192/`)

The PL3 scene (`results/20261105-polish-pacing/`): 60 s at 960x544, one Ref2VA continuation request, 1024 px reference,
the doll reference frame, seed 3401 with `H3_TP2_WINDOW_SEED_STEP=1`, 20 steps, `audio_flow_shift` 3.0, serving defaults
of 2026-11-05 (plugin eda87aa: cutlass FP8 GEMM, exact Ulysses exchange mode 2, fused quant). Changed:
`continuation_window_frames` 192 (overlap 22 as before), so 9 windows: 8.0 s + 7 x 7.08 s + a 2.4 s tail (58 new
frames). The 18 PL3 lines were re-cut into 16 lines of two sentences each so that no line crosses a hand-over
(`inputs/dialogue_w192.json`, planned with `scripts/plan_dialogue_windows.py --window 192`): 21/16/16/19/19/15/15/12
words in windows 1-8 (2.6/2.3/2.3/2.7/2.7/2.1/2.1/1.7 words/s of new content), no line in the tail window. Every
window prompt repeats the composition as in PL3 (`inputs/prompt_w192.txt`, `inputs/extra_w192.json`).

| | PL3: 277-frame windows (queue71) | PL4: 277, seed 3501 (queue76) | W192: 192-frame windows (queue78) |
|---|---|---|---|
| windows / hand-overs | 6 / 5 | 6 / 5 | 9 / 8 |
| wall (one request) | 3433.9 s | 3255.9 s | **3054.3 s** |
| diffuse per full window | 575-587 s (255 new frames) | similar | 334-341 s (170 new frames) |
| step time, full window | 28.8-29.4 s/it | | 16.6-17.1 s/it |
| diffuse per new frame | 2.28 s | | **1.98 s** (-13 %) |
| tail window diffuse | 295 s (143 new frames) | | 133.5 s (58 new frames) |
| decode | 183 s | | 189 s |
| seam retries | 0 | 0 | 0 (boundary 0.275-0.288 vs median 0.267-0.272) |
| peak memory head / agent | 64.4 GB | 64.4 GB | 53.3 / 57.2 GB |
| cuts | 0 | 0 | 0 |
| pauses (raw, < -40 dB for >= 1.5 s) | 0 | 1 (1.5 s) | 1 (1.6 s) |
| speech (faster-whisper small, pl) | WER 0.0816, 18/18 lines | WER 0.0816 | **WER 0.1429**, 16/16 lines + one invented line |
| faces, mean man / woman (min) | 0.799 / 0.689 (0.54 / 0.43) | 0.801 / 0.710 (0.64 / 0.48) | 0.809 / **0.614** (0.63 / **0.32**) |
| woman, first 10 % -> last 10 % | | | 0.689 -> 0.554 |

W192 transcript errors against the script: `nikogo` -> `nikiego`, `przyszła wczoraj` -> `przysławczoraj`, `może` ->
`meże` (twice), `pieniądze` -> `pienięże`, `dlatego` -> `dla tego`, `odpowiedź` -> `obspoweć`, `winien` -> `winiem`,
`miesiące` -> `miesięcy`, and the inserted tail "Więc jesteś tu mnie wrozy? Ja, a ja." at 57.3-60.1 s (the tail window
starts at 57.6 s). Of the systematic PL3/PL4 slips `może`, `pieniądze` and `nikogo` repeat; `porządnie`, `czynsz`,
`lepszym` and `wyjścia` were spoken right this time, and four words that were right before were not. Speech until
57.3 s is continuous (gaps between lines 0.0 s in the transcript), one 1.6 s silence inside the clip.

Checks: `speech_qa_w192.txt`, `drift-w192.txt` (sampled every second; the window column there counts 277-frame windows,
use the time column), the render log and timing JSON on Spark_06.

## Reading

- **The saving at 960x544 is 11 %, not a half.** The step time has a term linear in the window's token count and a
  quadratic one; at 960x544 the quadratic part of a 277-frame window is small (29 s per step against 17 s for 192
  frames, 1.70x the time for 1.44x the frames), so the per-frame cost falls 13 % and the extra windows' fixed costs eat
  part of that. The S² share is larger at 1344x768: the three measured window lengths there (124 frames 20 s, 141 frames 26 s,
  277 frames 82 s per step) fit t = 0.058·F + 0.00086·F², which predicts 43 s per step for 192 frames, 5.0 s of
  denoising per new frame against 6.4 s for 277, a saving of about a fifth, not a half (queue80 measures the 192-frame
  step directly). A 120 s clip is 17 windows of 192 frames (16 hand-overs) against 12 of 277.
- **Every hand-over costs identity and pacing.** The woman's face lost 0.13 of similarity over the clip (PL3 and PL4
  held her at 0.69-0.71 mean) and the word error rate rose with new single-word slips; eight hand-overs instead of five
  give the model eight occasions to re-settle the second face and the voices. One seed, so the size of the cost is
  rough, but its sign matches the identity arms (`results/20261103-lalka-identity/`): fewer, longer windows drift less.
- **A window without a line babbles.** The 2.4 s tail had no line planned and got an invented line (the owner's
  "Russian"), the same symptom as the one-line-per-window arms with their 7-8 s of spare time. The planner now warns
  below 1.0 words per second of new content (`--min-words-per-second`); this plan would have got "window 9: 0 words in
  2.4 s of new content". For a clip whose length is not a whole number of windows, give the tail a short line or round
  the duration (57.6 s for 8 windows of 192 frames).
- **Verdict.** 277 stays the default. 192-frame windows are a documented option for a draft render where time matters
  more than the second face: `"continuation_window_frames": 192` in `extra_params`, plan the lines with `--window 192`.
  The lower peak memory (53-57 GB against 64 GB) is a side effect worth knowing for the 1344x768 renders.

## Inputs

`inputs/dialogue_w192.json` and `.txt` (the 16 lines; the `.txt` is the speech-QA reference with the intended spellings),
`inputs/plan_w192.json` (planner report), `inputs/extra_w192.json` (the request's `extra_params`),
`inputs/prompt_w192.txt` (the main prompt = first window prompt).
