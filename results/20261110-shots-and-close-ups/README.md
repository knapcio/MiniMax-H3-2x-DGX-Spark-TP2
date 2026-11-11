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

- With the scene frame among the references, in any position, the shot opens on the scene frame's composition and the
  camera rushes into the prompt's framing in about 1.5 s. That push-in is the "quality drop" and the odd motion of
  the close-ups, and the prompt's "the camera does not move" does not prevent it.
- Without the scene frame the model ignores the close-up and casts whoever it likes.
- With the character's crop alone the shot opens in the prompt's framing, holds it, and the frame-level sharpness
  stays at the level of the wide shots. The set is re-imagined from the prompt (a similar inn, not the same one), so
  a wide shot and a close-up of this kind do not share a background.
- The frame-level sharpness numbers of a close-up (30-50 against 120-170 for a wide shot) are mostly composition
  (large smooth skin areas), not blur: the v3 close-ups are detailed on inspection.

## 4. The recipe for a film from shots, as of now

1. One reference set: the scene frame plus a head-and-shoulders crop per character (`scripts/make_character_cards.py`,
   or `H3_TP2_AUTO_CARDS=2`).
2. Wide shots and two-shots: scene frame + crops as references, one window each (4-14 s), the speech guard on
   (`H3_TP2_SPEECH_GUARD=0.03`, `H3_TP2_SPEECH_MODEL=small,medium`, `H3_TP2_CONT_RETRY=2`).
3. Close-ups: the character's crop as the only reference, the prompt describing the set (v4, queue139: static,
   in-frame from the first frame, faces held; the background is the prompt's room, so write the set into every
   prompt the same way, or accept that a close-up's background differs from the wide shot's, as it often does in a
   real edit).
4. Cut with jump cuts; a 40 ms audio fade at each cut keeps the room tone from clicking (`ffmpeg -nostdin` in a loop).
5. Words the model cannot say (invented names) are slips at every seed: write around them.

Clips: `fun-lembas-s4201` (v1), `fun2-s1..9` (v2), `fun3-s1..9` + `fun3/out/ksiazulo-v3.mp4` (v3), `fun4-s{2,4,5,6,7,9}`
(v4, cut together on the Mac), `cu2-*-s5002` (queue137/138) on Spark_06; queues `queue128b-fun.sh`, `queue132-fun2.sh`, `queue134-fun3.sh`, `queue137-cu-sharp.sh`,
`queue138-reforder.sh`, `queue139-fun4.sh`. The clips show invented characters only.
