# A film from shots: what a Ref2VA shot opens on, and how to get a close-up (2026-11-10/11)

A vlog-style clip was built three times from the same invented characters (a hobbit, a host in a hoodie, an elf in a
hobbit inn) to learn how the two-Spark TP2 stack behaves when a film is cut from separately rendered shots instead of
one continuation take. All renders Ref2VA, 20 steps, the serving defaults of 2026-11-10 (rev 5288f50-8163b99).

## 1. One take versus shots

| version | how | what the owner saw |
|---|---|---|
| v1 (queue128b) | one 54 s continuation take, 960x544, scene frame + three crops, speech guard 0.08 | static three-shot, slips, props not matching the words, soft hand-overs |
| v2 (queue132) | nine shots of 4.5-10 s, 960x544, one window each, cut together; the guard set but never ran | cuts and close-ups, but slips in five shots and "pixelation" in the close-ups |
| v3 (queue134) | the same nine shots at 1344x768, the guard on every shot (`guard_single`), conversational lines, a static camera in every prompt | seven shots clean at the first attempt, two re-sampled; close-ups still open wide and push in |
| v4 (queue139) | the six close-up shots of v3 again with the character's crop as the only reference (wide shots and the two-shot kept from v3) | every close-up opens in its framing and holds it, no push-in; the set behind a close-up is the prompt's inn, not the scene frame's; five of six shots clean at the first attempt, one re-sampled twice |

Shots are the right unit: one window each, so no hand-over drift, window-1 quality everywhere, and the speech guard
re-samples a bad shot in minutes. The faces hold across framings from one reference set (scene frame + a crop per
character): wide shots, two-shots and close-ups all show the same three people.

Cost at 1344x768: 83 s per step for a 10 s shot (1658 s), 45 s per step for 6 s; 62 s of finished clip from nine shots
took 3.5 h including two re-samples and the boot. The guard's verdicts on the nine shots: CER 0.009, 0.000,
0.040 -> 0.013 (re-sampled), 0.088 -> 0.053 -> 0.070 (kept the best), 0.029, 0.000, 0.000, 0.015, 0.000.

## 2. The one-window guard path is exact

Plain 4-14 s requests never entered the continuation loop, so no guard judged them (v2). `h3_tp2/guard_single.py`
routes such a request through the loop as a one-window continuation. The control of queue137 (the same shot with the
guard off, direct path) is byte for byte the v3 shot rendered through the loop: the path adds the checks and nothing
else.

## 3. What a shot opens on (queue137, queue138: shot 2, the elf in medium close-up, 6 s, 1344x768, seed 5002)

| references | opening frame (0.25 s) | at 3 s | frame sharpness per 0.25 s |
|---|---|---|---|
| scene frame + three crops (v3) | the wide three-shot of the scene frame | the elf's close-up, after a 1.5 s push-in | 142 141 136 119 64 52 33 31 ... 43 |
| the elf's crop first, then the scene and the other crops | the wide three-shot | the close-up, after the push-in | 137 144 135 125 51 54 50 45 ... 48 |
| scene frame alone | the wide shot, and the host instead of the elf | wide | 124 ... |
| crops at their native 512 px (`H3_TP2_REF_NO_UPSCALE=600`) | the wide three-shot | the close-up | 130 124 ... 36 |
| **the elf's crop alone** | **the elf in a medium shot at a table, in his framing from the first frame** | the same framing, static | **148 154 151 139 147 147 144 149 ... 175** |
| a crop of the scene frame in the shot's framing (the elf, his part of the table, the wall) + his crop (queue141) | the elf in a medium shot, in framing from the first frame | static | 168 174 172 167 168 165 ... 206 |
| the same + a crop of the lembas (queue141) | as above | static | 145 145 148 133 ... 161 |
| shot 6 (the host with the kebab): framing crop + his crop; + a kebab crop; his crop + the kebab crop (queue141) | the host in a medium shot in every arm | static | 85-99 / 81-90 / 102-118 |

- With the scene frame among the references, in any position, the shot opens on the scene frame's composition and the
  camera rushes into the prompt's framing in about 1.5 s. That push-in is the "quality drop" and the odd motion of
  the close-ups, and the prompt's "the camera does not move" does not prevent it.
- Without the scene frame the model ignores the close-up and casts whoever it likes.
- With the character's crop alone the shot opens in the prompt's framing, holds it, and the frame-level sharpness
  stays at the level of the wide shots. The set is re-imagined from the prompt (a similar inn, not the same one), so
  a wide shot and a close-up of this kind do not share a background.
- The frame-level sharpness numbers of a close-up (30-50 against 120-170 for a wide shot) are mostly composition
  (large smooth skin areas), not blur: the v3 close-ups are detailed on inspection.
- **What a reference anchors (queue141, 2026-11-11).** A crop of the scene frame in the shot's framing opens the shot
  in that framing and keeps it, like the character's crop alone, and it brings the scene's style (the beams, the round
  green door, the lamps) but not the room itself: the door and the fireplace sit elsewhere in every arm. Object crops
  (the lembas, the kebab) do not make the prop look like the crop either: the lembas came out as a pale toast slice
  with and without its crop, the kebab wrap looks alike in all three arms because the prompt names it. So Ref2VA
  references anchor subjects (faces, costumes) and style; backgrounds and props are regenerated per shot from the
  prompt unless the whole scene frame is a reference, and then the shot opens on that frame. The next thing to try is
  the model's own frame: a late frame of a v3 close-up (after its push-in) shows the real room and props at close-up
  framing, so as the scene reference of that character's close-ups it should give the framing and the room at once
  (queue143).
- **The model's own frame (queue143).** A late frame of a v3 close-up as Picture 1 (the real room and props at
  close-up framing) plus the face card: the elf shot opens on a wide view of a room and pushes into the close-up
  within 1.5 s again; the host shot opens on a table insert (hands, the scene's pale square lembas, the kebab) and
  then cuts to the host at the door. The room and the props now match the scene (the lembas is the scene's loaf, not
  toast), but the opening is an establishing move whenever the reference shows a room. So: a reference with a room
  gives the room and an establishing head; a reference without one (the face card alone) gives a static shot in an
  invented room. There is no reference set that opens static in the real room.

## 4. The recipe for a film from shots, as of now

1. One reference set: the scene frame plus a head-and-shoulders crop per character (`scripts/make_character_cards.py`,
   or `H3_TP2_AUTO_CARDS=2`).
2. Wide shots and two-shots: scene frame + crops as references, one window each (4-14 s), the speech guard on
   (`H3_TP2_SPEECH_GUARD=0.03`, `H3_TP2_SPEECH_MODEL=small,medium`, `H3_TP2_CONT_RETRY=2`).
3. Close-ups, two ways. (a) Scene frame (or a late frame of an earlier close-up) + the character's crop: the real
   room and props, but the shot opens on the reference's composition and cuts to the close-up on its own after
   1-2.6 s; render 1.5 s longer, ask for a beat before the line, and trim the head in the edit when the audio before
   the cut is silent (`scripts/shot_cuts.py`; section 5 has the measured shape of that head). (b) The character's crop alone: static from the first frame, faces
   held, but the room and props are the prompt's and differ from the wide shot's. (v3/v4, queue137-143.) (c) FL2VA from
   the model's own close-up still: the opening is exact, but five of six shots cut or pull back to a wide shot of invented
   people within 0.75-3.5 s, on a clause boundary of the line; usable as a close-up insert of the first clause
   (section 6). **(d) FL2VA with two keyframes: the model's own close-up still as the first frame and a later frame of
   the same shot as the last** (`H3_END_IMAGE_FILE`): the shot opens and closes in the framing, and the beat it leaves
   for (a tilt to the props, a two-second cutaway on a pause of the line) comes back; twelve of twelve on two seeds
   (section 8). Keep such a shot at 8 s or less: at 11.5 s the beat away grows to 4-6 s and the render costs +81 %.
   This is the way to render close-ups now. When the beat must not show invented people, the same two keyframes pinned
   as latents on the Ref2VA path with the character's card (`scripts/first_frame_pin.py --last`) keep the cast right
   in seven of eight beats, at the price of a hard cut right after the pinned frames in the food shot (section 9; two
   latents at each end make the ends exact, the departures stay cuts). The whole recipe on a new scene: section 10
   (a 48 s film in 2.8 h of renders, nine of nine shots at the first attempt, the close-ups held throughout).
4. Cut with jump cuts; a 40 ms audio fade at each cut keeps the room tone from clicking (`ffmpeg -nostdin` in a loop).
5. Words the model cannot say (invented names) are slips at every seed: write around them.

Clips: `fun-lembas-s4201` (v1), `fun2-s1..9` (v2), `fun3-s1..9` + `fun3/out/ksiazulo-v3.mp4` (v3), `fun4-s{2,4,5,6,7,9}`
(v4, cut together on the Mac), `cu2-*-s5002` (queue137/138) on Spark_06; queues `queue128b-fun.sh`, `queue132-fun2.sh`, `queue134-fun3.sh`, `queue137-cu-sharp.sh`,
`queue138-reforder.sh`, `queue139-fun4.sh`. The clips show invented characters only.

## 5. Way (a) rendered as a set: the real room follows, the head is a cut, the beat lands where the model wants (queue144)

Six close-ups of the comedy clip rendered the way (a) prescribes: scene frame + the three head crops as references,
1344x768, one window each, 1.5 s longer than the line needs, the prompt asking for a silent beat before the first word,
the guard at 0.03 with two retries. What came back, shot by shot (first scene cut by ffmpeg `scene>0.3`, leading
silence by `silencedetect -32 dB`, speech by faster-whisper small on the Mac):

| shot | asked | what the model did | first cut | leading silence | guard CER | elapsed |
|---|---|---|---|---|---|---|
| 2 elf | close-up, beat | wide table 1.0 s, then a hard cut to the close-up; same door, mug, leaf, lembas | 1.04 s | 1.69 s | 0.019 | 1211 s |
| 4 host | close-up, beat | steady medium shot of the host alone at the table, scale and kebab in place, no head | none | 0 s | 0.035 (best of 3) | 3550 s |
| 5 hobbit | close-up, beat (chews) | 3.5 s close-up chewing in silence, then a cut back to the wide table where he says the line | 3.54 s | 0 s (chewing) | 0.029 | 935 s |
| 6 host | close-up, beat | wide table 2.6 s while he unwraps the kebab and starts the line, then a cut to the close-up with the kebab | 2.63 s | 0.78 s | 0.000 | 1213 s |
| 7 elf | close-up, beat | the whole shot stays wide; the beat is there (2 s of silence) | none | 1.97 s | 0.000 | 933 s |
| 9 host | close-up, beat | wide table 1.3 s with the thumb up, then a cut to an extreme close-up | 1.29 s | 1.35 s | 0.000 | 1212 s |

What this says:

- The establishing head of way (a) is a **cut**, not a push-in, in 3 of 6 shots (2, 6, 9): the model opens on the
  reference's composition and cuts to the close-up on its own. The head is 1.0-2.6 s. When the audio before the cut
  is silence (2, 9) the head trims off cleanly; when the line starts in the head (6) the wide opening is a usable
  shot of its own (he unwraps the kebab and starts talking, then the close-up).
- The silent beat is honoured in 4 of 6 shots, but where the model puts it is its own: in shot 5 the beat became a
  3.5 s chewing close-up and the line moved to the wide shot after the cut. A beat the prompt asks for should be
  checked against the cut list, not assumed to sit before the first word.
- Two shots ignored the close-up and stayed wide or medium (4, 7). The room and props were right in all six; the
  faces held in all six (the crops carry them).
- The guard's slip was the catchphrase "chrupnięcie" (heard "chrupiącie" in two of three attempts at seed 5004);
  the best attempt kept CER 0.035. A respell entry is the next thing to try for words like it.
- Cost: 1211 s for an 8.0 s shot with four references (v3's 6 s shot took 935 s with the same references; the 6 s
  card-only shots of v4 took 692 s). The guard's retries dominate when they fire (shot 4: 3550 s).

The edit (`fun2/v5/assemble.sh` on the Mac): head trim = the first scene cut under 2 s when the audio before it is
silence (shots 2 and 9; `scripts/shot_cuts.py shot.mp4` measures the cuts and silences of a shot and prints that trim,
`--apply out.mp4` writes it), two hand overrides (shot 5 keeps 1.2 s of the chewing close-up, shot 7
keeps 1 s of its 2 s beat), wide shots 1/3/8 from v3, 40 ms audio fade at every cut, `ffmpeg -nostdin`. Result
`ksiazulo-v5-realroom-closeups.mp4`: 65 s, 9 shots, the same inn in every shot. Clips `fun5-s{2,4,5,6,7,9}` on
Spark_06, queue `queue144-fun5.sh`. Invented characters only.

## 6. Way (c): the close-up as FL2VA from the model's own still opens exactly and then leaves (queue147)

The third way to a close-up in the real room: no references at all, the shot rendered as FL2VA from a still of the
model's own close-up frame (the elf, the host with the kebab, the host's extreme close-up, cut from the v5 shots), the
prompt saying "the first frame is given: keep this person, framing, room and props exactly as in it" and "the camera
does not move, zoom or cut". Three stills, two seeds, 1344x768, 8.0 s (192 frames), the lines of shots 2/6/9, the
guard at 0.03 with two retries (the one-window request goes through the guard path). Measured: PSNR of frame 0 against
the still, the face scale and the camera's zoom over time (`scripts/framing_trace.py`, its first use on rendered
shots), scene cuts (`scene>0.3`), word times (faster-whisper small), guard CER, elapsed:

| shot, seed | frame 0 vs still | what the model did | leaves the framing at | the line against the move | guard CER | elapsed |
|---|---|---|---|---|---|---|
| 2 elf, 6001 | 39.0 dB | holds the medium close-up for all 8 s; the scale and the kebab slide into the frame from 4.7 s, the face stays the same size | never (zoom x1.03) | 1.3 s beat, then the whole line in the close-up | 0.000 | 899 s |
| 2 elf, 6002 | 39.2 dB | a hard cut at frame 16 to a medium shot of the elf behind the scale, the kebab and three lembas, then static | 0.67 s cut | starts at 0.0 s, no beat | 0.000 | 893 s |
| 6 host, 6001 | 40.7 dB | close-up for 3 s, then a pull-back to a two-shot with an invented red-haired woman | 3.5 s, pull-back x0.56, no cut | "A to jest kebab" close, "z Mordoru, i niestety mielonka" wide; no beat | 0.000 | 895 s |
| 6 host, 6002 | 40.9 dB | close-up for 3.2 s, then a hard cut to a three-shot with two invented women | 3.17 s cut | 2.7 s beat; "A to" before the cut, the rest after | 0.000 | 892 s |
| 9 host, 6001 | 42.2 dB | extreme close-up with the thumb for 3.3 s, then a hard cut to a wide two-shot with an invented bearded man; sliced white bread where the lembas were | 3.25 s cut | "Lembas dostaje muala" close, "a kebab z Mordoru nie dostaje nic" wide; no beat | 0.000 | 893 s |
| 9 host, 6002 | 43.1 dB | extreme close-up for 2.3 s, then a hard cut to a two-shot with an invented red-haired woman | 2.25 s cut | the same split at the comma; no beat | 0.022 | 1062 s |

What this says:

- **The opening is solved by construction.** Frame 0 is the still (39-43 dB; the last frame is 12-16 dB from it),
  so a shot from a still has no invented room and no head to trim. The faces and costumes held in all six.
- **The model leaves the framing anyway.** Five of six shots pull back or cut to a wider shot within 0.67-3.5 s,
  and the wider shot invents the other people at the table, because the still carries nobody else. This is section 5
  mirrored: given the wide scene frame it opens wide and cuts in; given a close-up it opens close and cuts out.
  The close-up is a shot the model visits; the table is its home framing. "The camera does not move, zoom or cut"
  was ignored in five of six, and "keep the props exactly" lost the lembas in shot 9.
- **The cut lands on the phrase.** In all five shots that cut or pulled back during the line, the move sits on a
  clause boundary of the line (the comma of shot 9 at both seeds, "A to | jest kebab" and "kebab | z Mordoru" in
  shot 6, after "Patrzcie," in shot 2 seed 6002). The model edits on the sentence, which makes the half before the cut a clean close-up insert of the
  first clause (2.3-3.5 s) and the half after a wide shot of the wrong people.
- The silent beat held in two of six (four of six in section 5); four shots start the line at 0.0 s.
- Cost: 892-899 s per 8.0 s shot with one still and no retries, against 1211 s for the same shot by Ref2VA with the
  scene frame and three crops; one shot took 1062 s at the same settings.
- The trace tool earned its place once and misled once. One of the five moves is a pull-back with no scene cut
  (shot 6 seed 6001: `scene>0.3` saw nothing, the zoom column shows x0.57 over 2.5 s and the `head_end` rule dates it
  at 3.5 s). Shot 2 seed 6002 was first written up here as a 0.75 s pull-back x0.76: it is a hard cut at frame 16
  (`scene>0.3` had it at 0.667 s all along). The first trace sampled through ffmpeg's `fps` filter, which hands out
  the last frame of each quarter second and never frame 0, and its cut rule asked only for a failed match; 84
  inliers across the cut read as a zoom. Corrected 2026-11-12 15:45: the trace now samples exact frames and calls
  a step a cut when the samples barely correlate and the match's zoom is outside 0.8-1.25; rerun on all twelve clips
  of sections 6 and 7 it finds every cut the eye and `scene>0.3` find and keeps the pull-back a pull-back.

Where this leaves close-ups, as of now: a close-up that must stay a close-up for a whole line has no reliable path
yet. Way (a) gives it after a 1-2.6 s wide head in half the shots; way (c) gives it for the first clause in most shots
and for the whole shot in one of six. The pinned first frame of `scripts/first_frame_pin.py` (the still as latent
frame 0 plus the character's card as the only reference, so the wide shot would at least show the right people) is
the next test (queue148, section 7). Clips `flcu-s{2,6,9}-s{6001,6002}` on Spark_06, queue `queue147-fl2va-stills.sh`.
Invented characters only.

## 7. The pinned first frame: the still is frame 0 and nothing more, the card keeps the right people (queue148)

The last idea for the opening: instead of FL2VA's first-frame conditioning, the still becomes latent frame 0 itself.
`scripts/first_frame_pin.py` writes the still as a 192-frame source clip plus a token mask with the first latent
pinned (`H3_SOURCE_VIDEO`, `H3_VIDEO_NOISE_MASK`), the shot renders by Ref2VA with the character's card as the only
reference, same three stills, lines, seeds and settings as section 6 (1344x768, 7.5 s, 20 steps). Arm "pin" ran with
the one-window guard off, the lines were transcribed afterwards on the workstation (faster-whisper small); arm
"pinloop" re-rendered one shot through the guard path. PSNR against the still is frame 0 / frame 1 / frame 24.

| shot, seed | 0 / 1 / 24 vs still (dB) | what the model did | leaves the framing at | line | elapsed |
|---|---|---|---|---|---|
| 2 elf, 6001 | 33.8 / 28.2 / 16.8 | holds the medium close-up for all 8 s (the same story as `flcu-s2-s6001`) | never | clean | 1040 s |
| 6 host, 6001 | 34.3 / 11.2 / 9.5 | cut at frame 1 to a hands-and-food insert, back to the host's close-up at about 2.7 s | frame 1 | clean | 1038 s |
| 9 host, 6001 | 35.2 / 29.8 / 21.0 | holds the extreme close-up, cut at 2.6 s to a wide shot of the host alone at the table | 2.6 s cut | clean | 1034 s |
| 2 elf, 6002 | 34.2 / 11.8 / 10.5 | cut at frame 1 to a static wide shot of the elf at the table | frame 1 | one word split | 1038 s |
| 6 host, 6002 | 34.7 / 24.9 / 10.7 | cut at frame 2 to a static wide shot of the host at the table | frame 2 | clean | 1036 s |
| 9 host, 6002 | 35.5 / 27.3 / 20.1 | holds the extreme close-up, cut at 2.2 s to a wide two-shot: the host and an invented bearded man | 2.2 s cut | one word repeated | 1034 s |

- The pin pins exactly what it says: one latent, one frame. Frame 0 is the still through the VAE (34-35.5 dB; FL2VA's
  first-frame conditioning gave 39-43 dB in section 6), and from frame 1 on the model is free. In three of six shots
  it cuts away at frame 1 or 2: a one-frame flash of the still and then another shot, worse than way (c), which at
  least held the opening for the first clause.
- The other three shots behave like way (c): the framing holds for the first clause and a cut lands at 2.2-2.6 s, or
  it holds for the whole shot in the one shot and seed that held in section 6 too. The shot grammar is the model's,
  not the conditioning's: first-frame conditioning of either kind does not override it.
- The card did its job. Every wide shot the model cut to shows the right character (the elf in shot 2, the host in
  shots 6 and 9), where section 6's FL2VA shots, with no reference, invented people. One wide shot still added a
  second man (shot 9 seed 6002).
- Section 6's "the cut lands on the phrase" did not repeat here. The two mid-line cuts fall inside a clause
  ("Lembas | dostaje muala" at 2.6 s in shot 9 seed 6001; between a repeated "dostaje" and "muala" at 2.2 s in seed
  6002), and the three frame-1 cuts come on the first syllable of the line. Five of five in section 6, zero of two
  here: a tendency of the FL2VA shots, not a rule to build a recipe on.
- The corrected trace (see the last bullet of section 6) sees all five departures: cuts at 0.25 s (the first
  sample after frame 1 or 2) in three shots, at 2.75 s and 2.25 s in shots 9, none in shot 2 seed 6001.
- The guard path is exact here too: `pinloop-s2-s6001` (guard on, CER 0.0185 on the first attempt) has the same
  `framemd5` as `pin-s2-s6001`, so the source rows and the mask survive the continuation loop unchanged (section 2,
  now also with a source video). Without the guard two of six lines had a slip (a split word, a repeated word),
  which is the usual rate the guard exists for.
- Cost 1033-1048 s per 8.0 s shot with the card, against 892-899 s for FL2VA with the still and 1211 s for Ref2VA with
  the scene frame and three crops.

Where this leaves close-ups: still no reliable path for a close-up that must hold for a whole line. The pin is not
a default and not a recommendation; it stays in the repo as a tool because it works mechanically and the card
half of it (the right people in whatever wide shot follows) is real. Pinning more latents (`--pin-latents 3`, frames 0-8)
is untested; the model that cuts at frame 1 would most likely cut at frame 9. The two-keyframe shot (first and last frame both set, queue151) is section 8: it constrains where the shot
ends as well as where it starts, and that is the one that works. Clips `pin-s{2,6,9}-s{6001,6002}` and
`pinloop-s2-s6001` on Spark_06, queue `queue148-pin.sh`. Invented characters only.

## 8. Way (d): the close-up as FL2VA with both keyframes opens, leaves for a beat and comes back (queue151)

The same three stills, lines and seed as sections 6 and 7, rendered as FL2VA with the still as the first frame **and**
a last frame (`H3_END_IMAGE_FILE`, `frame_indices [0, -1]`; `scripts/bench-matrix.sh` passes `both` when it is set).
Arm "same": the last frame is the same still, so the shot must end where it began. Arm "late": the last frame is a
later frame of the same v5 close-up (the same framing, another pose: the elf holding the lembas up, the host mid-bite,
the host's thumb lower), so the shot has somewhere to go without changing the framing. 1344x768, 8.0 s, 20 steps,
guard 0.03, seed 6001. PSNR of frame 0 against the first still and of frame 191 against the last; what happens in
between by eye and by `scripts/framing_trace.py` (exact frames, the corrected cut rule).

| shot | arm | frame 0 vs first | frame 191 vs last | what the model did in between | away from the framing | guard CER | elapsed |
|---|---|---|---|---|---|---|---|
| 2 elf | same | 39.2 dB | 39.1 dB | holds the medium close-up for all 8 s, the elf talks and handles the lembas | never | 0.000 | 953 s |
| 2 elf | late | 39.2 dB | 38.4 dB | holds the framing for all 8 s and moves from the reach to holding the lembas up, the last frame's pose | never | 0.000 | 948 s |
| 6 host | same | 40.7 dB | 40.3 dB | a fast tilt down to the food on the table at 0.5-1.2 s, the insert to 3.0 s, a tilt back up to the host's close-up, held to the end | 0.75-3.5 s, a camera move, no cut | 0.000 | 947 s |
| 6 host | late | 40.4 dB | 40.0 dB | the same tilt down to the food at about 1 s, back up by 4 s, then the host eats and talks into the last frame's pose | about 1-4 s, a camera move, no cut | 0.000 | 945 s |
| 9 host | same | 42.7 dB | 42.0 dB | extreme close-up, a hard cut at 3.25 s to a wide two-shot with an invented red-haired woman, a hard cut back at 5.17 s, extreme close-up to the end | 3.25-5.17 s, two cuts | 0.000 | 948 s |
| 9 host | late | 41.9 dB | 41.1 dB | the same: out at 3.33 s to the same kind of two-shot, back at 5.33 s | 3.33-5.33 s, two cuts | 0.022 | 950 s |
| 2 elf | same, seed 6002 | 39.2 dB | 39.2 dB | holds the medium close-up for all 8 s | never | 0.000 | 947 s |
| 2 elf | late, seed 6002 | 39.3 dB | 38.7 dB | holds the framing, the reach becomes holding the lembas up | never | 0.000 | 944 s |
| 6 host | same, seed 6002 | 40.8 dB | 40.6 dB | the tilt down to the hands and the food at about 1 s, back up to the close-up by 5 s, held to the end | about 1-5 s, a camera move, no cut | 0.000 | 945 s |
| 6 host | late, seed 6002 | 40.5 dB | 40.2 dB | pulls back to a medium shot of the host biting, the table and the scale in frame, and comes back to the close-up for the last frame's pose | about 1-5 s, a camera move, no cut | 0.027 | 951 s |
| 9 host | same, seed 6002 | 42.9 dB | 42.4 dB | out at 2.21 s to the two-shot with the invented woman, back at 5.5 s | 2.21-5.5 s, two cuts | 0.000 | 946 s |
| 9 host | late, seed 6002 | 42.2 dB | 41.6 dB | the same, out at 2.21 s, back at 5.42 s | 2.21-5.42 s, two cuts | 0.000 | 946 s |

- **The last keyframe turns "leaves" into "leaves and comes back".** With one keyframe (section 6) five of six shots
  left the close-up and never returned; with two, twelve of twelve (two arms, two seeds) open and close on it
  (38.7-42.9 dB at both ends), four hold it throughout, and eight leave it for a 2-4 s beat in the middle and return.
  The departure is the shot's, not the seed's: shot 2 holds at both seeds and both arms, shot 6 moves the camera to
  the food and back at both, shot 9 cuts to the same two-shot with an invented woman and back at both (out at
  2.2-3.3 s, back at 5.2-5.5 s). The last frame decides that the shot comes back from it; what it leaves for is
  written in the still and the line (a food shot leaves for the food, a comparison of two dishes leaves for the
  other person).
- **The beat sits on a pause of the line.** In shot 9 the cut away covers "Muala, | a" at both arms and both seeds
  (the silence between the clauses, 1.2-4.2 s at seed 6002, 2.2-3.1 s at 6001); in shot 6 the line starts during
  the insert and "I niestety, mielonka" is said back in the close-up. An editor would call the shot-9 version a close-up with a two-second cutaway and keep it;
  the shot-6 tilt is a camera move no cut check sees (`scene>0.3` and the trace both say no cuts, correctly).
- **Arm "late" gives motion without a cut.** The elf's shot goes from the reach to holding the lembas up in one
  framing with no departure at all; a second still of the same shot is a cheap way to tell the model where the
  action ends. The late still costs nothing (a frame of the earlier render).
- Lines clean in ten of twelve at the first attempt (0.022 and 0.027 under the 0.03 guard in the other two); the
  two-shot's invented woman is section 6's problem again (no reference carries the other person): the cutaway shows
  the wrong people, the same wrong person at both seeds. Elapsed 944-953 s against 892-899 s for one keyframe
  (+6 %) and 1033-1048 s for the pin with the card.

Where this leaves close-ups: **way (d) is the recommendation.** A close-up that must hold for a whole line is
rendered as FL2VA from the model's own still as the first frame and a later frame of the same shot as the last;
the shot opens and closes in the framing, and when it leaves it comes back within about two seconds on a pause of
the line. If the cutaway must not show invented people, cut it out in the edit (the close-up on both sides of it is
the same framing, a jump cut hides under the pause); another seed does not help, the shot leaves for the same thing
at both.

**Render pass B with the seed of the still's source render.** The inn's stills came from the v5 shots (seeds 5006 and
5009 for the host's two close-ups) and the two-keyframe shots above ran at 6001 and 6002. Rendered again at their
source seeds (queue161, 2026-11-14, both arms): shot 9 holds the extreme close-up for all 8 s in both arms, no cut
out and no two-shot with an invented woman (frame 0 42.8-43.0 dB, the last frame 41.7-42.5 dB, `scene>0.3` sees
nothing, four of four lines clean at the first attempt); shot 6 still tilts down to the food at about 1 s and is back
by 2.5 s in both arms, as at every other seed (the trace: 32 of 32 samples with the face and no cut or move in
shot 9's two clips, 25-26 of 32 in shot 6's). The other direction (queue162): the second film's three close-ups
rendered at seeds other than their stills' sources (8002/8004/8006) hold the framing throughout as well, 32 of 32,
no cut, no move, lines clean. So a shot that holds at any seed exists (the film's three), and a shot that cuts away
at some seeds and not at its still's source exists (the inn's shot 9, whose line compares two dishes and invites a
two-shot). Four fresh seeds of that shot (queue163, 5101-5104, arm "late"): two hold the extreme close-up throughout
(5101, 5104), one cuts out at 3.25 s to the two-shot and back at 5.25 s (5102), one cuts out at 2.0 s and pushes back
in by 5.0 s (5103). Over seven seeds the shot cuts away at four and holds at three, its still's source seed among the
three. The still's source seed is the safe default for pass B: it costs nothing, and it is the one seed known to
hold for a shot that wants to cut away; a shot that cuts away at that seed too has about an even chance at the next.
The tilt insert of the food shot is the shot's at every seed.

**The cut-away fills silence, so size the shot to the line.** The same shot 9 at the same four seeds with the line
cut to one clause, "Lembas dostaje muala." (queue164): four of four cut away (out at 1.8-2.8 s, back at 4.6-5.3 s or
by a push-in), against two of four with the two-clause line. A 1.5 s line in an 8 s shot leaves six seconds of
silence and the model fills them with its home framing; the two-clause line's cut-aways sat in the pause between
the clauses, and the second film's close-ups, whose lines fill their 8 s, never left. So the earlier "one clause per
shot" reading was wrong: what a shot must not have is a pause of two seconds or more, which means a shot as long
as its line and a beat. The same line in a 4 s shot (107 frames, 4.46 s, the shortest FL2VA takes; queue165, the
same four seeds): three of four hold the extreme close-up throughout and one (5103, the seed that cut earliest at
8 s too) cuts out at 1.25 s and back at 3.25 s; 404-413 s per render against 945 s. From four of four cut-aways to
one of four by shot length alone. A 1.5 s line still leaves three seconds of silence in the shortest shot the model
renders, so lines that short are better merged into a neighbour's shot than given one of their own.

**Keep the shot short.** At a full window (11.5 s, 277 frames; queue153, arm "late", seed 6001) the return still
happens in three of three (frame 0 39.2 / 40.4 / 42.1 dB, the last frame 38.7 / 40.3 / 41.4 dB against the late
still, no scene cuts) but the beat away grows with the shot: shot 2 holds throughout again (the elf lifts the lembas
in the first second and talks holding it), shot 6 is down on the food for 4 s instead of 2.5, and shot 9 leaves the
extreme close-up for 6 s of 11.5 (a crash zoom out at 1-2 s to the wide two-shot with the invented woman, a push back
in at 7-7.5 s, both camera moves, no cut). The model fills the extra time with its home framing. A full window also
costs 1716-1723 s against 945-953 s for 8 s (+81 % for +44 % frames). So: close-ups at 8 s or less, one line each.
Clips `bk-{same,late}-s{2,6,9}-s{6001,6002}` and `bk11-late-s{2,6,9}-s6001` on Spark_06, queues
`queue151-bookends.sh`, `queue152-bookends-s6002.sh`, `queue153-bookends-11s.sh`. Invented characters only.

## 9. Way (d) on the Ref2VA path: both keyframes pinned as latents, the card as the reference (queue154)

FL2VA takes no references, so the beat the two-keyframe shot leaves for shows invented people (section 8, shot 9: the
same red-haired woman at both seeds). The Ref2VA path can carry the character's card, and the keyframes can be pinned
there as latents instead: `scripts/first_frame_pin.py --last` writes a full-length source clip (the still, then the
late still on the last four frames) and a token mask that keeps latent frame 0 and the last latent, the card is the
only reference (`H3_SOURCE_VIDEO`, `H3_VIDEO_NOISE_MASK`, `H3_REF_IMAGES=card`). Shots 6 and 9, seeds 6001 and 6002,
the guard at 0.03 with one retry, otherwise as section 8.

| shot, seed | frame 0 vs first | frame 191 vs last | what the model did | the people in the beat | guard | elapsed |
|---|---|---|---|---|---|---|
| 6, 6001 | 38.2 dB | 34.8 dB | the still for one frame, a cut at frame 1 to a hands-and-food insert, back to the host's close-up at 2.2 s, held to the last frame's pose | the host's hands | 0.000 | 1039 s |
| 9, 6001 | 39.4 dB | 36.2 dB | extreme close-up for 4 s, a cut at 5 s to a wide shot of the empty table and room, back at 6 s, the last frame's pose | nobody | 0.000 | 1036 s |
| 6, 6002 | 38.7 dB | 34.9 dB | close-up to 0.67 s, a cut to a wide shot of the host alone at the table, back at 4.3 s, the last frame's pose | the host alone | 0.000 | 1040 s |
| 9, 6002 | 39.2 dB | 34.4 dB | extreme close-up to 0.8 s, a cut to a wide shot of the host alone with his thumb up, which stays to the end; only the pinned last frames are the extreme close-up again | the host alone | 0.111 at the first seed, 0.000 at the second | 1930 s |

- **The card does what it is for.** No invented person in four of four beats (section 8's FL2VA shots had the invented
  woman in four of four cutaways of shot 9): the host alone, his hands, or an empty room.
- **The pinned keyframes are weaker than FL2VA's.** The last frame matches the late still at 34-36 dB against 38-42 dB,
  and a pin is one latent: in one shot the first frame is a one-frame flash before a cut (section 7's failure mode),
  in another the return is a four-frame flash at the end instead of a return. Two of four shots behave like section 8
  (leave and come back on the line's pause); the pin path also takes 1036-1040 s against 945-953 s.
- So for a close-up that must not invent people, the pinned two-keyframe shot with the card is the way, with one shot
  in four to render again; when the beat can be cut out in the edit, FL2VA with both keyframes (section 8) is cheaper
  and its ends are exact.
- **Two latents at each end (`--pin-latents 2 --last-latents 2`, frames 0-4 and the last eight; queue155, the same
  four shots):** the last frame is now as exact as FL2VA's (38.6-39.9 dB; the first 38.7-39.6) and the shot comes back
  before the pinned tail in four of four (at 2.2, 6.0, 4.2 and 7.2 s). But the food shot cuts away right after the
  pinned frames at both seeds (frame 5, a 0.2 s hold instead of a one-frame flash), where FL2VA's conditioning gave a
  0.75-1 s hold and then a camera move; the pin's departures are cuts, FL2VA's are moves. And the card lowers the
  invented-people rate without removing it: one of the four beats shows a dark-haired woman next to the host, the
  other three the host alone or his hands (one of eight over queues 154 and 155, against four of four in section 8's
  shot 9). Elapsed 1039-1046 s.
- So: FL2VA with both keyframes is the default (exact ends, departures as camera moves, cheapest); the pinned pair
  with the card when a beat with invented people is worse than a hard cut after a 0.2 s hold; and when the beat must
  show the right second person, the second card does not put him there (queue156, the same four shots with the host's
  and the elf's cards: the beats show the host's hands, the host alone twice and the empty room once, the elf never;
  no invented person in four of four, the ends 38.7-39.8 dB, the same cut right after the pinned frames in the food
  shot). Over the twelve pinned shots with cards (queues 154-156) one beat invented a person against four of four in
  FL2VA's shot 9; the model fills a wide beat with whoever the references allow, and with only the host allowed it
  shows him alone or the room. Clips `pin2-s{6,9}-s{6001,6002}`, `pin2x2-s{6,9}-s{6001,6002}`,
  `pin2x2c-s{6,9}-s{6001,6002}` on Spark_06, queues `queue154-pin2.sh`, `queue155-pin2x2.sh`,
  `queue156-pin2x2-cards.sh`. Invented characters only.

## 10. The recipe end to end on a new scene: a 48 s film in three passes (queues 157-159, 2026-11-13)

Is the recipe general or fitted to the inn? A second film from scratch, nothing reused: a fish-and-chips hut on a
windy Baltic pier at golden hour, two invented hosts (a woman in a yellow raincoat, a man with a red beard and a
grey beanie), six shots with one Polish line each (a test of fish against fries, a gull steals a fry). Every step as
section 4 says, measured:

| pass | what | renders | per render | result |
|---|---|---|---|---|
| 0 | scene frames: a 5 s T2VA render of the scene prompt at 1344x768, two seeds, frames 0/40/80 | 2 | 417 s | seed 7102 chosen (the closer two-shot, faces large enough for cards); `scripts/make_character_cards.py` found both faces (512 px cards) |
| A | Ref2VA, scene frame + two cards: shots 1/3/5 (two-shots) for the film, close-ups 2/4/6 way (a) as the source of stills; guard 0.03, two retries | 6 | 1086-1091 s | six of six at the first attempt, CER 0 / 0.020 / 0 / 0.024 / 0 / 0; the same two people, counter, railing and gulls in all six; shot 5's gull lands, takes a fry and flies off as written; the close-ups open wide and cut to the close-up at 1.4-1.5 s (section 5's head) |
| B | FL2VA with both keyframes: a still from after the cut of each pass-A close-up as frame 0, a later frame of the same shot as the last frame (`film2/stills/`) | 3 | 948-951 s | three of three hold the close-up for all 8 s (32 of 32 samples with the face, no cut, no camera move; frame 0 40.2-42.7 dB, the last frame 39.2-40.5 dB against the stills), CER 0.024 / 0 / 0 |

- GPU time for the 48 s film: 2 x 417 + 6 x 1089 + 3 x 950 = 10 218 s, 2.8 h of renders, plus three boots and two
  partition switches (about 45 min). The three close-ups cost two renders each; the wide shots one.
- No beat away at all in the three pass-B close-ups, where the inn's shots 6 and 9 left for 2-4 s: these stills
  come from the same shot's own way-(a) render (the camera position the model chose for that framing), the inn's came
  from another seed's close-up. One film; worth keeping in mind when the stills are picked.
- A cheaper still source (queue166, 2026-11-14): shot 4's pass A at 10 steps instead of 20, with the guard on, took
  1100 s, not 550: the 10-step render said the line wrong (CER 0.24) and the guard sampled it again. The pass-B shot
  from the 10-step stills held the close-up throughout (32 of 32 samples, 39.7 / 39.0 dB at the ends), so the still
  itself is good enough; a still source does not need its line right, so pass A for stills runs without the guard
  (queue167 measures that).
- Two cuts: `film2-recipe.mp4` (shots 1, 2B, 3, 4B, 5, 6B) and `film2-way-a.mp4` (the six pass-A shots, the
  close-ups with their wide heads), both 48.0 s, jump cuts with 40 ms audio fades. Clips `film2ref-s710{1,2}`,
  `film2a-s{1..6}`, `film2b-s{2,4,6}` on Spark_06, queues `queue157-film2ref.sh`, `queue158-film2a.sh`,
  `queue159-film2b.sh`, prompts and stills in `film2/`. Invented characters only.
