# A film from shots: the recipe that worked

A one-take comedy clip of a minute looks like one take: the model's cuts land where its seed puts them, the
camera stays where the reference put it, and a close-up asked for in the prompt is a shot the model visits and
leaves. A film cut from shots rendered one by one gives the cuts to the editor. This page is the procedure that
rendered a 48 s six-shot film on a scene the model had never seen in 2.8 h of renders, nine of nine renders at the
first attempt, with the close-ups holding their framing for the whole line
(`results/20261110-shots-and-close-ups/`, sections 4 and 10; the measurements behind every step are there).

Everything below runs on the head (`~/h3-tp2`) except the card cut and the final edit, which need no GPU. Two
partitions are involved: Ref2VA for the shots with references, FL2VA for the close-ups with two keyframes
(`scripts/select-stack.sh long Ref2VA|FL2VA` on both nodes, then a head restart).

## 1. The scene frame

Render the scene once as plain text-to-video, short, at the film's size, and keep a frame: that frame is the room,
the light and the cast for every later shot. Two seeds, three frames each, pick by eye (faces large enough to cut
cards from, everyone in it):

```sh
TASK=t2va H3_PROMPT_FILE=film/scene_prompt.txt bash scripts/bench-matrix.sh "5:1344:768:20:7101"
ffmpeg -nostdin -i out/d5_1344x768_s20_seed7101.mp4 -vf "select='eq(n\,80)'" -fps_mode passthrough -frames:v 1 film/reference.png
```

417 s per 5 s render at 1344x768. The scene prompt describes the place, the props and each person's look in one
sentence each; those sentences are reused word for word in every shot prompt.

## 2. The cards

One head-and-shoulders crop per character, cut from the scene frame, upscaled to 512 px; the server takes the
scene frame plus the cards as the Ref2VA references:

```sh
python scripts/make_character_cards.py film/reference.png --out-dir film/cards      # MTCNN, or --box x0,y0,x1,y1 per face
# prints: H3_REF_IMAGES=film/reference.png,film/cards/reference_face0.png,film/cards/reference_face1.png (left to right)
```

The cards are what keeps the faces the same person from shot to shot and what keeps invented people out of the
wide shots (`docs/DIALOGUE_CLIPS.md` section 1 for the long-clip measurements).

## 3. The lines and the shot list

One line per shot, one or two clauses, 8 s or less of picture. The model edits on the sentence: when a shot
leaves its framing it does so on a pause of the line, so a shot with one clause has nowhere to cut. Words the
model cannot say (invented names, rare spellings) are slips at every seed: write around them or respell them
(`docs/DIALOGUE_CLIPS.md` section 2, `scripts/speech_guard_dryrun.py` names them on a finished clip).

A shot prompt is the scene sentences, then the picture index ("Picture 1 shows the whole scene with the two
people behind the counter; Picture 2 is the woman on the left (yellow raincoat, short dark bob, freckles);
Picture 3 is the man on the right (...). Keep exactly these two faces, costumes and the same counter, pier and
lighting."), then the shot ("A medium close-up of the woman (Picture 2): only her face and shoulders in the frame,
nobody else visible, the counter edge at the bottom. She lifts the fried fish to her nose, sniffs and makes a
doubtful face. The woman first looks into the camera in silence for a moment, then speaks."), then the camera and
voice sentences ("The camera does not move, zoom or cut: one static continuous take. Natural, clearly pronounced
Polish in a relaxed, conversational vlog tone ... No subtitles, no text, no music.") and the line
(`The woman: <d>[Polish] Ryba pachnie morzem. Albo molo. Trudno powiedzieć.</d>`). The close-ups get a second
prompt file for pass B with the picture index replaced by "The first frame is given: keep this person, costume,
framing, pier, props and lighting exactly as in it." and no "(Picture 2)".

## 4. Pass A: every shot with the scene frame and the cards (Ref2VA)

```sh
# head: ~/.h3-tp2.env carries these, then restart the head
export H3_TP2_SPEECH_GUARD=0.03 H3_TP2_SPEECH_MODEL=small,medium H3_TP2_CONT_RETRY=2
# per shot
export TASK=ref2va H3_PROMPT_FILE=film/shot3.txt H3_REF_IMAGES=film/reference.png,film/cards/reference_face0.png,film/cards/reference_face1.png
bash scripts/bench-matrix.sh "7.5:1344:768:20:7003" film/reference.png
```

7.5 s requests 192 frames (8.0 s on the 17n+5 grid); 1086-1091 s each at 1344x768 with the two cards. A plain
one-window request with a guard configured runs through the continuation loop's first-window path, so the speech
guard judges it like a window of a long clip and samples it again with another seed when the line comes out wrong
(`h3_tp2/guard_single.py`; the guard path is bit-identical to the direct call). 0.03 lets one slipped word pass and
re-samples a mangled line.

Wide shots and two-shots are finished here. Close-ups come back as way (a): the shot opens on the scene frame's
composition and cuts to the close-up on its own after 1-2.6 s, or pulls back out of it later. Keep them as the
source of the stills for pass B.

## 5. Pass B: the close-ups with two keyframes (FL2VA)

Cut two stills from each pass-A close-up, both from after its cut into the close-up: a first frame (a calm face,
before the line, the props in hand) and a later frame of the same shot (another pose: the fry held up, the grin).
Then render the shot as FL2VA with both keyframes:

```sh
ffmpeg -nostdin -i out/film2a-s4.mp4 -vf "select='eq(n\,84)'" -fps_mode passthrough -frames:v 1 film/stills/s4.png
ffmpeg -nostdin -i out/film2a-s4.mp4 -vf "select='eq(n\,168)'" -fps_mode passthrough -frames:v 1 film/stills/s4_late.png
# FL2VA partition, the same guard variables
export TASK=fl2va H3_PROMPT_FILE=film/shot4_fl.txt H3_END_IMAGE_FILE=film/stills/s4_late.png
bash scripts/bench-matrix.sh "7.5:1344:768:20:7004" film/stills/s4.png     # bench-matrix passes keyframe position "both"
```

The shot opens on the first still and closes on the second (39-43 dB at both ends), 945-953 s each. In the film of
section 10 all three held the framing for the whole 8 s; in the inn's shots of section 8, whose stills came from
another seed's render, the shot left the framing for a 2-4 s beat on a pause of the line and came back (a tilt to
the props, or a cutaway to a wide shot of invented people). Use the seed of the still's source render for pass B
(as above: shot 4's still came from seed 7004, pass B runs at 7004); whether that is the cause is being measured.
Keep the shot at 8 s or less: at a full 11.5 s window the beat away grows to 4-6 s and the render costs +81 %.
If a beat shows invented people and must not, cut it out in the edit (the close-up on both sides of it is the same
framing and the cut hides under the pause), or render the shot on the Ref2VA path with both keyframes pinned as
latents and the cards as references (`scripts/first_frame_pin.py --last`, section 9: no invented person in eleven
of twelve beats, at the price of a hard cut right after the pinned frames in some shots).

## 6. The edit

```sh
python scripts/cut_shots.py out/film2a-s1.mp4 out/film2b-s2.mp4 out/film2a-s3.mp4 out/film2b-s4.mp4 out/film2a-s5.mp4 out/film2b-s6.mp4 --out film.mp4 --no-head-trim
```

Jump cuts, one encoding for all shots, a 40 ms audio fade-in and fade-out on every shot (the room tone would click
otherwise). Without `--no-head-trim` the tool trims each shot's head by `scripts/shot_cuts.py`'s rule (a scene cut
within the first 2 s with silence before it), which is what a way-(a) close-up kept as rendered needs; pass-B
close-ups have no head, and `--trim N=seconds` sets a trim by hand.

## 7. Checks

- `scripts/framing_trace.py shot.mp4`: the face scale and the camera's zoom per quarter second, hard cuts, fast
  camera moves, where the opening settles and the shot size the model ended on; this is what says whether a
  close-up held.
- `scripts/shot_cuts.py`: scene cuts and the leading silence per shot, for the trim.
- The speech guard's own lines in the head journal (`speech check window 1 attempt N ... CER`) and
  `scripts/speech_guard_dryrun.py` on a finished shot for the words that failed.
- `scripts/face_drift.py` / `scripts/window_detail.py` when the cast must be checked across many shots.

## 8. What it costs

| step | renders for a six-shot film | each | total |
|---|---|---|---|
| scene frame (5 s T2VA, two seeds) | 2 | 417 s | 14 min |
| pass A (6 shots, 1344x768, 8 s, cards) | 6 | 1086-1091 s | 1 h 49 min |
| pass B (3 close-ups, two keyframes) | 3 | 948-951 s | 47 min |
| boots and partition switches | 3 + 2 | 5-7 min | about 45 min |

2.8 h of renders for 48 s of film at the first attempt; a re-sampled line adds one render.

## What is still open

- A beat the model leaves for shows invented people on the FL2VA path (no references there); the pinned variant
  with cards keeps them out but cuts harder. The co-host's card does not put him in the beat.
- Whether pass B with the still's source seed is what kept the second film's close-ups whole (queue161).
- The silent beat before a line lands where the model wants it (section 5): plan the trim, not the beat.
