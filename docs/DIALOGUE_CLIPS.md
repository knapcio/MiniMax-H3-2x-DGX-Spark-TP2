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
  More steps (30) or a lower `audio_flow_shift` (2.0) do not change the spoken words at all (the same slips at every
  seed, `results/20261106-voice/` section 11); the sampler is not the lever, the text is.

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

Prompts written by hand (or edited after the planner) get the same check from
`python3 scripts/dialogue_lint.py --extra extra.json`: words per window and per second of the window's own time, windows
without lines, mixed or missing language tags, an unclosed `<d>`, a line repeated in another window; exit status 1 on a
window over the budget (`--budget`, default 30 words, the delivered clips carried 18-28) or a malformed tag.

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
| 60 s 1344x768, scene + 2 crops | 6 (+1 re-sample) | 13 391 s (3.7 h) | `xclip60v3`: diffuse 1866-1963 s per window (92.7-100.4 s/step, about 15 % over the scene reference alone), one window re-sampled by the cut check, peak 70.1 GB (`results/20261106-xclip-events/`) |
| 120 s 1344x768 | 12 | 7.9 h denoising + 633 s decode | `xclip120`; a 277-frame window at 1344x768 runs at 80-84 s per step |

The decode of 2880 frames at 1344x768 takes over 600 s: the API server's per-step wait (`VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT`,
upstream 600 s) is 7200 s in `scripts/env.sh` since 2026-11-05, after one 7.9 h render died there. If a request still
dies after the diffusion, `H3_TP2_REPLAY_LATENTS` replays the saved window files through the unchanged stitch, decode and
mux path (the 120 s clip was recovered that way in 21 min, bit-identical windows; `results/20261103-xclip/`).

A long clip can cut away in a late window (two of three samples of the 118 s Polish scene did, with the seam check at
1.0): the interior cut check (`H3_TP2_CONT_CUT_RATIO`, default 1.7) re-samples such a window during the render, and
with `H3_TP2_SAVE_LATENTS` set a window that still goes wrong is re-sampled afterwards in minutes with
`H3_TP2_REPLAY_LATENTS` + `H3_TP2_REPLAY_PARTIAL=1` and another request seed, instead of a full re-render.

Identity over two minutes: the scene reference alone holds the faces for about 60 s and loses them in the second
minute (118 s chains, 2026-11-09: the woman 0.71 -> 0.60 similarity, another person on screen); with a head-and-shoulders
crop per character (next paragraph) both faces stayed the same person for the whole 118 s (0.75-0.80 and 0.80-0.85 per
window) for +25 % per step (`results/20261109-long-chain-identity/`). So give any clip beyond about 90 s the crops, or let the server cut them: `H3_TP2_AUTO_CARDS=2` adds a crop per
detected face to a one-image Ref2VA request (verified byte for byte against the hand-cut crops on 2026-11-10). The lines of the second minute slip for another reason, the window's
RoPE time running away from the text: `H3_TP2_ID_LOCK=toffcap` with `H3_TP2_ID_LOCK_TOFF_CAP=2125` (the clock stops
after window 6) took the same chain's wrong characters from 0.065 to 0.025 with windows 8-9 clean, for a few points
of the woman's similarity at the very end (section 5 of the same results).

Identity through an occlusion or a turn-away: first, the look-lock must not undo it. A character hidden at the moment of
a hand-over came back as another person (open orange coat instead of the closed red one, blonder, older) although the
guard had passed the window: the plain look-lock pulled the window's statistics toward window 1's tail, a tail without
her. Since 2026-11-07 the lock uses only the latent positions that did not change between the tail and the window
(`H3_TP2_CONT_LOCK_ROBUST=1`, default); on the saved latents of that scene it keeps the coat and the face (0 of 35
figures changed against 10 of 35) and on a 60 s painting it holds the brightness like the plain lock
(`results/20261106-whole-person/`). Then give the request the scene frame **and a head-and-shoulders crop per
character** (`scripts/make_character_cards.py scene.png --out-dir refs` prints the `H3_REF_IMAGES=` line;
`TASK=ref2va H3_REF_IMAGES=scene.png,refs/face0.png,refs/face1.png bash scripts/render.sh ...`). On the 11 s occlusion
probes the crops raised the returning character's face similarity by 0.1-0.2 on two seeds, with no cut, and on the
60 s Polish doll scene (no occlusion, same seed and lines as PL5) they lifted both faces in every window (means 0.81 /
0.70 -> 0.84 / 0.76, minima 0.54 / 0.44 -> 0.66 / 0.56, the first window most) with the speech, pauses and cuts
unchanged (`results/20261105-multiref/`). The model's own rendered close-ups do not help, so cut the crops from the
scene frame itself. The crops cost 19-25 % of the render time at the default 1024 px reference size (three reference
blocks instead of one). Keeping them at their native 512 px (`H3_TP2_REF_NO_UPSCALE=600`) removes the cost and the
gain with it (a quarter of the rows, no identity effect on two seeds), so pay for the upscale. On the public 60 s shop
scene at 1344x768 with a turn-away, a turn to a shelf and a bend below the counter written into the prompts, every
character came back as the same person with the crops alone (`results/20261106-xclip-events/`).

A character replaced by someone else is what the **identity guard** catches: `H3_TP2_IDENTITY_GUARD=0.1` with
`H3_TP2_IDENTITY_GROW=0.6` (opt-in; needs facenet-pytorch in `~/facenv`, see `scripts/face_bank.py`). After every window
the guard decodes it, scores 8 frames against a face bank built from the request's reference images, and re-samples
the window with another seed when more than the limit of the frames show a confident face that matches nobody; the
faces of an accepted window join the bank, so a lead in profile stays known. With 8 frames, 0.1 re-samples on a single
unknown frame (a stranger who stays, but also a blurred pass-by), 0.2 tolerates one. Cost: 41 s per window at 960x544,
67-68 s at 1344x768, plus any re-sampled window. Calibrated on the published 120 s clip, where it would have re-sampled
exactly the window in which a stranger walks in, and run live on the 22 s occlusion probe
(`results/20261106-identity-guard-calibration/`).

A slip of the tongue is what the **speech guard** catches (opt-in, 2026-11-08): `H3_TP2_SPEECH_GUARD=0.03` (0.01 is the strict setting, see the cost below) with
`H3_TP2_SPEECH_RESPELL=respell.txt` when the prompt uses phonetic spellings (needs faster-whisper in `~/speech-venv`,
the interpreter of `scripts/speech_qa.py`; `H3_TP2_SPEECH_PY` names another). After every window the guard decodes
its audio, transcribes it on the CPU (the `small` model, 3-4 s per window; `H3_TP2_SPEECH_MODEL` picks another) and
compares the transcript with the `<d>` lines of that window's prompt as a character error rate with the word
boundaries removed and free speech before and after the lines. It listens to the part of the window the finished
clip keeps: the first 0.92 s of a continuation window repeat the previous window's tail and are dropped at assembly,
so a line the window starts there would be cut in the clip (`H3_TP2_SPEECH_WHOLE_WINDOW=1` for the whole window). A window above the limit is re-sampled with another seed, up to `H3_TP2_CONT_RETRY` times, and the best
attempt is kept, so a slip the model makes at every seed costs time, not the clip. One slipped word is 1-3 characters
of 110-130 in an 11.5 s Polish window (0.008-0.026), so 0.01 re-samples on most single slips and 0.02 on two; Whisper's
own mishearings count against the limit as well, which is why the limit is not 0. The language comes from the
`[Polish]` tag of the lines (`H3_TP2_SPEECH_LANGUAGE` overrides). Measured on three 64.6 s Polish chains at 0.01
(`results/20261106-voice/` section 15): 21 wrong characters of 2292 without the guard, 7 with it, for one to three
extra window samplings per chain (+17 to +48 % render time); on a 118 s chain (11 windows) 50 of 1302 without it and
13 with it for eight extra samplings (+75 %), where the plain chain lost whole words in windows 8 and 11; a second 118 s
chain at 0.01 (seed 3801, 2026-11-12) took ten extra samplings (+100 %, 21 attempts for 11 windows), four of them on words
that came out the same way at every seed. **The recommended limit for long chains is therefore 0.03 since 2026-11-13**
(one slipped word in a window passes, a mangled line does not), with the words the dry run names rewritten or respelled;
0.01 is for a clip whose every word must land and whose render time can double. The guard also stops after two attempts
that slip on the same words (nothing a seed can fix). The voice
score moved by -0.13 to +0.08 paired UTMOS over the four chains, which is chain-to-chain spread; a control with a
re-sampled window and no selection kept the voice. An English chain needed one re-sample and lost no voice. About one Polish window in five and one English window in
nine is above 0.01 on finished clips.
The guards judge a plain one-window request (4-14 s, no `long_video`) as well since 2026-11-10: with a guard configured
such a request runs as a one-window continuation, the same sampling plus the checks and retries (`H3_TP2_GUARD_SINGLE=0`
turns that off). Shots rendered one by one and cut together get the same speech guard as a long take.
Two judges are steadier than one: `H3_TP2_SPEECH_MODEL=small,medium` re-samples a window only when both models hear
the lines wrong (the lowest rate counts; `medium` adds about 8 s per window and is asked only when `small` objects). On
finished clips the two sizes disagree by 1-4 characters on about a third of the windows, so a single judge's rate has
that much noise in it.
`scripts/speech_guard_dryrun.py clip.mp4 --extra extra.json [--respell respell.txt]` runs the same judgement on a
finished clip, window by window, and lists which windows each limit would have re-sampled; use it on clips you already
have before choosing a limit. A name Whisper spells its own way (`Nowak` heard as `Novak`) costs a character in every
window that says it: put it in the respelling table too (`nowak novak`), or it eats the limit's margin.
Both name the words: the dry run prints `misheard: zaprośmy > zapróczmi` next to a window's rate, and the live guard's
summary line ends with the windows that stayed above the limit and the words that came out wrong in every attempt
(`still above 0.01 after every attempt: w4 (zaprośmy ją)`). A word the model slips on at every seed is not fixed by
re-sampling: rewrite the line around it or respell it phonetically in the prompt (section 2), then render again.
Respell inside the word and map it back in the table (`chrupnjęcie chrupnięcie` took `chrupnięcie` from 0 of 4 seeds
to 2 of 4, and the guard's retries do the rest); do not split a word into two, which broke the word next to it in three
of four takes (`results/20261106-voice/` section 18).

## 5. Checks before anyone sees it

```bash
.speech-venv/bin/python scripts/speech_qa.py clip.mp4 --language en --model small --reference-file dialogue.txt
ffmpeg -i clip.mp4 -af silencedetect=n=-40dB:d=1.5 -f null -                      # pauses >= 1.5 s
ffmpeg -nostats -i clip.mp4 -vf "select=gt(scene\,0.1),showinfo" -f null - 2>&1 | grep -c pts_time   # cuts
~/facenv/bin/python scripts/face_drift.py clip.mp4 reference.png --json drift.json  # faces per window vs the reference
~/facenv/bin/python scripts/voice_naturalness.py clip.mp4           # UTMOS + DNSMOS + pitch spread per window
python scripts/shot_cuts.py shot2.mp4 shot4.mp4                   # a shot's own cuts and silences, and the head trim they imply
python scripts/cut_shots.py shot1.mp4 shot2.mp4 ... --out film.mp4  # the shots cut together (head trims, one encoding, audio fades)
.speech-venv/bin/python scripts/speech_qa.py clip.mp4 --language pl --output speech.json
~/facenv/bin/python scripts/voice_naturalness.py clip.mp4 --segments speech.json   # the same per spoken line
python3 scripts/script_wer.py --segments speech.json --extra extra.json [--respell respell.txt]  # word error rate per window
bash scripts/voice-metrics.sh clip.mp4 --language pl --extra extra.json [--respell respell.txt]  # all of the speech checks at once
```

`voice_pairs.py a.voice/naturalness_segments.json b.voice/naturalness_segments.json` compares two renders of the same
script line by line: every transcribed line of A is paired with the line of B that says the same thing and the UTMOS
difference is printed per line and per window, so a hand-over setting that keeps the words but loses the voice shows
as a sign that holds across lines (two seeds of the same setting differ by about 0.05 per window, a setting that lets
the voice drift by -0.6 to -0.7 in the last window).

`voice-metrics.sh` runs the speech checks of this list in one go (transcript, word and character error rate per window,
UTMOS by spoken line and by window, speaker continuity, audio seams) in the interpreters they need (`SPEECH_PY`,
`MOS_PY`, `PY`, found under the usual paths) and prints one block; a tool whose interpreter is missing is skipped with a
note. CPU only, 1.5-4 minutes for a 60 s clip; the JSON of every step lands in `<clip>.voice/`.

`script_wer.py` scores every window's transcript against the `<d>` lines of that window's prompt (the
`continuation_prompts` of the extra JSON; `--prompt file.txt --windows N` when every window shares one prompt) and lists
the words that differ, so a slip shows up in the window and the word it happened in. The whole transcript is aligned to
the whole script and each error is charged to the window of the script word it touches, so segment timings do not matter
and the per-window errors add up to the clip's word error rate (`clip` in the output). Compare arms rendered from the same
script, not absolute values. A phonetic respelling in the prompt would count as an error even when the word is spoken
right, so list the respellings in a file of `prompt_spelling real_word` lines and pass it as `--respell`. Every window
also gets a character error rate with the word boundaries removed (`CER` in the output): `dziewczyna pisze` heard as
`dziewczy napisze` is a word error rate of 1.0 and a character error rate of 0.0, so a real slip of the tongue moves
both numbers and a word boundary Whisper placed elsewhere moves only the word rate. Count slips by the character rate.

`voice_naturalness.py` prints one line per continuation window: UTMOS (a predictor of how natural speech sounds; the
model's single windows score 4.3-4.5 in English and 3.7-4.1 in Polish, the same range as recordings), DNSMOS signal
quality and the pitch spread. A window far below the first one is where a voice turned flat or synthetic. Per-window
scores mix silence and whoever speaks, so judge a long dialogue with `--segments`: one score per transcribed line, with
the window it starts in and the speaker's median pitch. Before 2026-11-07 every line from the fourth window of a long
Polish clip on scored 0.4-0.7 below the same line rendered as a fresh single window; the audio pin released after 30 % of
the steps (the default since then) closed that gap, and the chain's windows now score at or above fresh windows
(`results/20261106-voice/`, sections 9 and 12).

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
  vowels; no respelling for them has been tried (`czynsz` is already phonetic). Steps and `audio_flow_shift` were ruled out as levers (2026-11-08).
