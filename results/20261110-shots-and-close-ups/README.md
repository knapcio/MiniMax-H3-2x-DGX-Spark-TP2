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
   (section 6).
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
| 2 elf, 6002 | 39.2 dB | pulls back in the first 0.75 s to a medium shot of the elf behind the scale, the kebab and three lembas, then static | 0.75 s, pull-back x0.76, no cut | starts at 0.0 s, no beat | 0.000 | 893 s |
| 6 host, 6001 | 40.7 dB | close-up for 3 s, then a pull-back to a two-shot with an invented red-haired woman | 3.5 s, pull-back x0.56, no cut | "A to jest kebab" close, "z Mordoru, i niestety mielonka" wide; no beat | 0.000 | 895 s |
| 6 host, 6002 | 40.9 dB | close-up for 3.2 s, then a hard cut to a three-shot with two invented women | 3.17 s cut | 2.7 s beat; "A to" before the cut, the rest after | 0.000 | 892 s |
| 9 host, 6001 | 42.2 dB | extreme close-up with the thumb for 3.3 s, then a hard cut to a wide two-shot with an invented bearded man; sliced white bread where the lembas were | 3.25 s cut | "Lembas dostaje muala" close, "a kebab z Mordoru nie dostaje nic" wide; no beat | 0.000 | 893 s |
| 9 host, 6002 | 43.1 dB | extreme close-up for 2.3 s, then a hard cut to a two-shot with an invented red-haired woman | 2.25 s cut | the same split at the comma; no beat | 0.022 | 1062 s |

What this says:

- **The opening is solved by construction.** Frame 0 is the still (39-43 dB; the last frame is 12-16 dB from it),
  so a shot from a still has no invented room and no head to trim. The faces and costumes held in all six.
- **The model leaves the framing anyway.** Five of six shots pull back or cut to a wider shot within 0.75-3.5 s,
  and the wider shot invents the other people at the table, because the still carries nobody else. This is section 5
  mirrored: given the wide scene frame it opens wide and cuts in; given a close-up it opens close and cuts out.
  The close-up is a shot the model visits; the table is its home framing. "The camera does not move, zoom or cut"
  was ignored in five of six, and "keep the props exactly" lost the lembas in shot 9.
- **The cut lands on the phrase.** In all four shots that cut or pulled back during the line, the move sits on a
  clause boundary of the line (the comma of shot 9 at both seeds, "A to | jest kebab" and "kebab | z Mordoru" in
  shot 6). The model edits on the sentence, which makes the half before the cut a clean close-up insert of the
  first clause (2.3-3.5 s) and the half after a wide shot of the wrong people.
- The silent beat held in two of six (four of six in section 5); four shots start the line at 0.0 s.
- Cost: 892-899 s per 8.0 s shot with one still and no retries, against 1211 s for the same shot by Ref2VA with the
  scene frame and three crops; one shot took 1062 s at the same settings.
- The trace tool earned its place: two of the five moves are pull-backs with no scene cut (`scene>0.3` saw nothing
  in shot 6 seed 6001 and shot 2 seed 6002), which only the zoom column shows (x0.56, x0.76) and the
  `head_end` rule dates (3.5 s, 0.75 s).

Where this leaves close-ups, as of now: a close-up that must stay a close-up for a whole line has no reliable path
yet. Way (a) gives it after a 1-2.6 s wide head in half the shots; way (c) gives it for the first clause in most shots
and for the whole shot in one of six. The pinned first frame of `scripts/first_frame_pin.py` (the still as latent
frame 0 plus the character's card as the only reference, so the wide shot would at least show the right people) is
the next test (queue148). Clips `flcu-s{2,6,9}-s{6001,6002}` on Spark_06, queue `queue147-fl2va-stills.sh`. Invented
characters only.
