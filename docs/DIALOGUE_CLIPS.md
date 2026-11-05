# Dialogue clips of 60-120 s: the recipe that worked

The path that produced the 60 s and 120 s English shop scenes (`results/20261103-xclip/`) and the 60 s Polish ones
(`results/20261105-polish-pacing/`): two invented characters, one continuous take, the dialogue spoken by the model with
lip sync, a bed under the voices. Every number below is from those renders on the two-Spark TP2 stack; your scene will
differ, the order of the steps should not.

## 1. A reference frame from the model itself

Render a 5 s T2VA clip of the scene (1344x768, 20 steps) and take frame 0 as a PNG. Write the composition into that
prompt: where the camera stands, who is on the left and who on the right, faces in three-quarter view, the props by
name. Three prompt rounds were needed for the shop scene: a negation ("no telephone") produced the telephone twice;
naming the counter, a brass inkwell and a scale instead gave a usable frame (`xclip/reference_prompt3.txt`). An
invented face is a must for anything public: do not prompt a real actor.

## 2. The lines

- About **three short lines per 11.5 s window** (277 frames), fewer in a partial last window. One line per window left
  the model 7-8 s to fill, and it filled them with 2-4 s pauses or, in Polish, with invented non-Polish speech; three
  lines removed both at the same render cost (`results/20261105-polish-pacing/`).
- Keep each window under about **2.75 words per second of new content**: in the Polish arm 2.26-2.64 words/s were all
  spoken and the 6 s tail at 2.85 dropped half a line, and above **1.0 words per second** in every window, including a
  short tail: a window with no line or one short line gets pauses or an invented line. `scripts/plan_dialogue_windows.py`
  warns outside that range.
- Tag the language in every line: `<d>[English] ...</d>`, `<d>[Polish] ...</d>`; put the speaker in front
  (`Izabela, the woman: <d>[Polish] ...</d>`). Known Polish weakness: single words with nasal vowels or `sz`/`rz`
  come out mispronounced (`może` -> `meże`, `pieniądze` -> `pienięże`, `porządnie` -> `pożądnie`), the same words
  on every seed. Write those words phonetically in the prompt only (`morze`, `pieniondze`, `porzondnie`) and keep the
  real spelling in the speech-QA reference: the 60 s Polish scene went from WER 0.082 to 0.035 with no other change
  (`results/20261105-polish-pacing/`, PL5). `sz` clusters (`czynsz`, `lepszym`) still slip; the language itself holds.

## 3. One prompt per window, the composition in every one of them

The reference image fixes the faces, the prompt fixes the blocking. The first 60 s render ignored the reference
composition in window 1 (the woman on the wrong side, in profile) and cut to it at 17 s; repeating the composition
sentence ("the camera stands at the end of a polished wooden counter, on the left the man ..., on the right the woman
..., both faces fully visible in three-quarter view") in the main prompt **and every continuation prompt** gave one
continuous take from the first frame (`results/20261103-xclip/`, v1 vs v2). Add "answering each other promptly", the
ambience you want under the voices ("the low murmur of customers further back, footsteps, a clock ticking"), and
"No subtitles, no text, no music" if a bed is to be mixed in later.

Build the explicit list with the planner, which also carries the window geometry so the server cannot plan different
windows than the prompts assume:

```bash
python3 scripts/plan_dialogue_windows.py --seconds 60 --scene "$(cat scene.txt)" \
  --dialogue lines.json --report plan.json --extra-output extra.json
```

`lines.json` is a list of `{"start", "end", "speaker", "text", "language"}`; a line must not cross a window hand-over.
Add `"audio_flow_shift": 3.0` to `extra.json` (the shop scenes used it) and use the first continuation prompt as the
main prompt. The report lists each window's lines, words and words per second; read the warnings.

## 4. The request

Serve the v0.30.0 stack on the **Ref2VA** partition (`scripts/select-stack.sh long Ref2VA` on both Sparks, agent before
head); the reference short edge is 1024 px by default since 2026-11-04 (faces held over 60 s in the doll arms at 1.06x
the FL2VA time; the 2048 px setting the 1344x768 X clips used costs 1.8x the time for the last few points of identity,
`results/20261103-lalka-identity/`).
Keep `H3_TP2_SAVE_LATENTS=<dir>` set.

```bash
H3_PROMPT_FILE=prompt.txt H3_EXTRA_JSON_FILE=extra.json TASK=ref2va \
  bash scripts/render.sh clip.mp4 60 1344 768 20 2304 '' reference.png
```

What it costs on the pair (20 steps, cutlass FP8 GEMM, the exact Ulysses exchange, the fused quant):

| clip | windows | render | notes |
|---|---|---|---|
| 60 s 960x544 | 6 | 3434 s (57 min) | the Polish doll scenes: diffuse 575-587 s per full window, decode 183 s |
| 60 s 1344x768 | 6 | 12 031 s (3.3 h) | `xclip60v2`, six diffuse phases of 2086-2109 s |
| 120 s 1344x768 | 12 | 7.9 h denoising + 633 s decode | `xclip120`; a 277-frame window at 1344x768 runs at 80-84 s per step |

The decode of 2880 frames at 1344x768 takes over 600 s: the API server's per-step wait (`VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT`,
upstream 600 s) is 7200 s in `scripts/env.sh` since 2026-11-05, after one 7.9 h render died there. If a request still
dies after the diffusion, `H3_TP2_REPLAY_LATENTS` replays the saved window files through the unchanged stitch, decode and
mux path (the 120 s clip was recovered that way in 21 min, bit-identical windows; `results/20261103-xclip/`).

## 5. Checks before anyone sees it

```bash
.speech-venv/bin/python scripts/speech_qa.py clip.mp4 --language en --model small --reference-file dialogue.txt
ffmpeg -i clip.mp4 -af silencedetect=n=-40dB:d=1.5 -f null -                      # pauses >= 1.5 s
ffmpeg -nostats -i clip.mp4 -vf "select=gt(scene\,0.1),showinfo" -f null - 2>&1 | grep -c pts_time   # cuts
~/facenv/bin/python scripts/face_drift.py clip.mp4 reference.png --json drift.json  # faces per window vs the reference
```

Targets met by the delivered clips: word error rate 0.005-0.054 (every line present and in order, the language never
slipping), 0-2 pauses >= 1.5 s, 0 cuts, face similarity means 0.54-0.64 with the minima explained by profile turns, not
by a different face. A clip with a cut of its own is best redone with another seed, not patched.

## 6. A bed under the voices

Render speech-free beds with the model (16 s 960x544 T2VA: "delicate solo piano music only, no singing, no speech" /
shop murmur), then `scripts/mix_bed.sh clip.mp4 bed.mp4 out.mp4 -37` (ambience) or `-40` (piano). The script
normalises the bed by loudness first because H3 beds come out anywhere between -25 and -45 LUFS, loops it, ducks it under
speech (sidechain, ratio 4) and copies the video stream. Deliver the raw clip too: the dialogue clips sit at -21 to -24
LUFS, the 120 s one at -13 LUFS (lower it by 8 dB for the same balance as the 60 s ones).

Window length: the default 277-frame windows are the right choice for a clip people will watch. 192-frame windows
(`"continuation_window_frames": 192`, lines planned with `--window 192`) rendered the 60 s 960x544 Polish scene 11 %
faster with 53-57 GB instead of 64 GB peak, but eight hand-overs instead of five cost the second face (0.61 against 0.69
mean), the speech (WER 0.14 against 0.08) and, with no line in the 2.4 s tail, an invented line
(`results/20261105-window-length/`). A draft option; at 1344x768 the saving should be larger (not measured).

## What is still open

- Polish `sz` clusters (`czynsz` -> `czynż`, `lepszym` -> `lepsym`) survive the respelling that fixed the nasal
  vowels; no respelling for them has been tried (`czynsz` is already phonetic).
