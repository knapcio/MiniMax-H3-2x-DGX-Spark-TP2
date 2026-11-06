# Several reference images per Ref2VA request: identity through occlusion (2026-11-05)

**Question.** In the published 120 s shop clip a background figure that walked behind something came back as someone
else, and the woman's face after a turn was not quite hers (owner, 2026-11-05). The model has no object permanence: a
hand-over carries 22 frames, the one reference image holds only what it shows. Upstream's Ref2VA packs every image of
the request's `input_references` list as its own reference block in the sequence (`pipeline_minimax_h3.py` encodes each
image, `packed_sequence.py` lists each shape), so a scene frame plus a head-and-shoulders crop per character puts every
identity into every window, visible or not. Does that hold a character through an occlusion or a turn-away?

**Probe.** The X-clip shop scene (generic reference `ref_2208.png`, 1344x768), 11 s at 960x544 = one 277-frame window,
Ref2VA, reference short edge 1024, 20 steps, seed 2301, no speech. Two scenes (`inputs/prompt_occl.txt`: the woman
walks behind the man, hidden for about two seconds, re-appears on his far side; `inputs/prompt_turn.txt`: she turns her
back to the camera for three seconds and turns back). Arms: **A** = the scene frame alone; **B** = the scene frame + two
crops of the characters cut from it (MTCNN boxes, 2.4x wide, upscaled to a 512 px short edge: the server refuses
references under 256 px, the raw 236 px crops failed). `scripts/render.sh` `H3_REF_IMAGES=a.png,b.png,c.png`. Measure:
`scripts/face_drift.py --match embed --every 0.5` against the scene reference (cosine similarity of each face to its
reference face; "missing" = no face found at that second).

| | occlusion A | occlusion B (+2 crops) | turn A | turn B (+2 crops) |
|---|---|---|---|---|
| wall | 585 s | 726 s (+24 %) | 603 s | 755 s (+25 %) |
| cuts | 0 | 0 | 0 | 0 |
| woman before (0-1 s) | 0.52-0.72 | **0.71-0.85** | 0.63-0.82 | **0.79-0.85** |
| woman hidden / turned | 1.5-7 s missing | 1.5-7 s missing | 1.5-8 s missing or < 0 | 1.5-8 s missing |
| woman after return | 0.34-0.59 (mean of the last 10 % 0.49) | **0.42-0.67 (0.57)** | 0.54-0.78 (0.70) | **0.46-0.87 (0.84)** |
| man, whole clip | 0.84-0.91 | **0.87-0.94** | 0.24-0.91 (he turns with her) | 0.29-0.94 |

**Second seed and model-rendered cards (queue87, 22:00-22:38).** Seed 2302 of the occlusion scene, and arm **C** = the
scene + two "cards" the model rendered itself (a 5 s Ref2VA close-up of each character from the scene frame + its crop,
`inputs/prompt_card_*.txt`, frame at 1 s; the cards resemble the scene faces at 0.75 / 0.72 and look right by eye).

| occlusion, woman | before (first 10 %) | after re-entry (last 10 %) | man, mean |
|---|---|---|---|
| A seed 2301 (scene only) | 0.685 | 0.494 | 0.881 |
| A2 seed 2302 | 0.738 | 0.366 | 0.851 |
| B seed 2301 (+2 crops) | 0.829 | 0.572 | 0.905 |
| B2 seed 2302 | 0.869 | 0.575 | 0.875 |
| C seed 2301 (+2 rendered cards) | 0.781 | 0.424 | 0.866 |

Two seeds agree: the crops add about +0.15 before and +0.1-0.2 after the occlusion (0.57 both seeds against 0.49 / 0.37),
the man gains too, no cut in any arm. The model's own close-up cards, although sharper and recognisably the same
people, do not help the re-entry (0.42): the identity signal the model follows is the scene's own pixels, not a
re-imagined face. Title the result "crops, not cards".

**Reading so far.** The extra references work mechanically: no error, no cut (the 10-28 experiment with an extra *video*
reference block made the model cut; image blocks do not), and every identity measure is 0.1-0.2 higher with the crops,
before and after the event. The loss at re-appearance is not removed: after the occlusion the woman comes back at
0.57 against 0.83 before it (A: 0.49 against 0.69). Sharper references do not change that (cards, above), and the second seed confirms the crops' gain. Cost: the two
crops add a quarter of the window time at this size, because `H3_TP2_REF_SHORT_EDGE` upscales them to 1024 px like the
scene; `H3_TP2_REF_NO_UPSCALE=600` (queue90) keeps a reference whose short edge is below 600 px at its own size.

**Reference key scaling: closed (queue88, 22:38-23:46).** `H3_TP2_REF_KEY_SCALE` multiplies the attention keys of the
reference-image rows (one factor per image block, applied to the local rows before the exact Ulysses exchange), the
idea being that a re-entering character attends more to its crop. Occlusion scene, seed 2301, scene + 2 crops as in B:

| arm | factors (scene, crop, crop) | wall | result |
|---|---|---|---|
| V1 | 1.0, 1.6, 1.6 | 736 s | the picture collapses into a smeared, drifting close-up of the crops; no face found in 23 samples; 5 scene cuts |
| V2 | 1.4, 1.4, 1.4 | 728 s | a frozen, washed-out wide shot (motion 0.6-1.1 against 3-8 in B); one face found once (0.43) |

`key_scale_V1_V2.png` (top V1, bottom V2, one frame per second). The layout was right (three image blocks of
1792/1632/1536 rows after 5191 text rows in a 52928-row sequence) and the probes ran at the normal speed, so this is
the mechanism itself: a multiplicative factor on the keys scales the attention logits of those rows, and at the logit
magnitudes of this model 1.4x already pins every video token to the references. Closed; a smaller factor would be a
weaker version of the same distortion. The lever that remains for re-entry is on the reference side (what the images
show), not in the attention.

**Identity guard, prototype (2026-11-06 00:06, `H3_TP2_IDENTITY_GUARD`).** The generic answer to "a figure went behind
something and someone else came out": after every window (the first one too) the plugin decodes the window's latents,
scores eight frames spread over it against a bank of face embeddings built from the request's own reference images
(`scripts/face_bank.py`, facenet-pytorch in `~/facenv`, rank 0 judges and broadcasts), and re-samples the window with
another seed, like the seam and cut checks, when more than the given fraction of the frames show a principal face
(at least 5 % of the frame wide) that matches nobody in the bank (cosine below 0.4). Calibration on the probes above,
eight frames each, bank = the scene frame + the two crops (4 faces):

| probe | principal faces | unknown frames | note |
|---|---|---|---|
| occlusion A, A2 (scene only) | leads 6-7 % wide, woman after re-entry matches her crop at 0.48-0.63 | 0 / 8 | the double detections next to her (3-4 % wide) are filtered |
| occlusion B, B2 (+ crops) | woman after re-entry 0.45-0.74 | 1 / 8 | the one unknown frame is her blurred pass-by in front of the camera (15 % wide, 0.01-0.11) |
| turn B | 0.42-0.95 | 0 / 8 | |

So at the default limit (`H3_TP2_IDENTITY_GUARD=0.2`, two of eight frames) none of these windows would be re-sampled:
the guard is for a replaced character, not for the 0.1-0.2 of similarity the crops recover. The first live run
(queue91, limit 0.1 so that B's pass-by frame trips it) measures the plumbing and the cost of a decode plus scoring
per attempt.

**The 60 s Polish doll scene with the crops (queue89, 2026-11-05 23:49 to 2026-11-06 01:12).** The same request as
the PL5 baseline of `results/20261105-polish-pacing/` (960x544, 6 windows, 17 respelled lines, seed 3401, Ref2VA 1024,
20 steps), with two head-and-shoulders crops of the doll scene frame added to the reference list
(`H3_REF_IMAGES=reference.png,refs/lalka_face0.png,refs/lalka_face1.png`, 512 px short edge, upscaled to 1024 by the
default). No occlusion in this scene: it measures what the crops do to an ordinary two-shot over a minute.

| | PL5, scene frame only | scene frame + 2 crops |
|---|---|---|
| render | 3434 s | 4083 s (+19 %) |
| diffuse per full window | 575-587 s (28.8 s/step) | 682-703 s (34.1 s/step) |
| peak head memory | 64.0 GB | 64.7 GB |
| cuts / pauses >= 1.5 s | 0 / 0 | 0 / 0 |
| seam and cut checks | all first attempts | all first attempts (ratios 1.00-1.11) |
| speech, Polish WER | 0.035 | 0.021 (`czynsz` -> `czynż` again; `lepszym` right this time) |
| faces, mean man / woman | 0.810 / 0.701 | **0.844 / 0.758** |
| faces, minimum | 0.543 / 0.444 | **0.664 / 0.557** |
| first 10 % of the clip | 0.616 / 0.570 | **0.780 / 0.659** |
| per window, man | 0.73, 0.85, 0.83, 0.85, 0.78, 0.82 | 0.81, 0.86, 0.84, 0.86, 0.83, 0.87 |
| per window, woman | 0.65, 0.74, 0.67, 0.72, 0.72, 0.71 | 0.70, 0.78, 0.78, 0.77, 0.79, 0.72 |

Both faces are closer to the reference in every one of the six windows, the minima rise by 0.12 / 0.11 and the first
window, the only one without a continuation anchor, gains the most (man 0.73 -> 0.81, woman 0.65 -> 0.70). The crops
cost 19 % of the render at 1024 px (three reference blocks instead of one; queue90 measures the native-size variant).
The speech, pauses and cuts are unchanged. `sheet-multiref.png` is the contact sheet (one frame every 2 s): one take,
the same two people throughout. The recipe in `docs/DIALOGUE_CLIPS.md` now lists the crops as a default step for any
clip with recognisable characters.

**Native-size crops: closed (queue90, 2026-11-06 01:27-02:18).** `H3_TP2_REF_NO_UPSCALE=600` on the occlusion scene,
seeds 2301 and 2302: the scene frame still goes to 1792x1024, the two crops stay at 512x768 / 512x832 (the journal
prints `reference image 512x773 -> 512x768`). The cost disappears: diffuse 587 / 581 s against 585 s for the scene
alone and 726 s with the upscaled crops, peak 62.2 GB. So does the gain: the woman re-enters at 0.03-0.14 (seed 2301)
and 0.30-0.42 (seed 2302) against 0.42-0.67 / 0.44-0.70 with the upscaled crops and 0.34-0.59 / 0.37 with no crops
(`xclip/drift-noupscale-occl-s*-embed.txt`, `sheet-noupscale-occl-s2301.png`, `sheet-noupscale-occl-s2302.png`:
she comes back small and far behind the counter on both seeds, as in arm A). A 512 px crop block has a quarter of the
rows of a 1024 px one, and the identity effect goes with the rows, not with the pixels: the crops have to be upscaled
to the scene's reference size to count. The option stays opt-in for anyone who wants cheaper extra references; the
recipe keeps the default size and pays the 19-25 %.

**Identity guard, first live run (queue91, 02:18-02:58): the bank, not the check.** The 11 s occlusion probe with
`H3_TP2_IDENTITY_GUARD=0.1` logged the bank (`identity bank: 4 face(s) from 3 reference image(s): ref0.png@x=0.27,
ref0.png@x=0.74, ref1.png@x=0.50, ref2.png@x=0.49`, the scene's two faces and one per crop) and nothing else: an 11 s
request is one window and never enters the continuation loop where the guard sits, so no window was scored. The run
also rendered with `H3_TP2_REF_NO_UPSCALE=600` still in the head's environment file from queue90 (the queue helper
strips only the variables it knows; the two new ones were not on its list, fixed in `scripts/queue-lib.sh`), which is
why its numbers equal queue90's seed 2301 to the third decimal. The guard's first real run is a two-window request
(queue93: 22 s, she walks behind him and back), then the 60 s shop scene at 1344x768 with the events written into its
windows (queue92).

**Second live run (queue93, 03:14-03:40): the guard scored both windows and saw no faces.** The 22 s occlusion scene
(960x544, seed 2301, `H3_TP2_IDENTITY_GUARD=0.1`) logged `identity check window 1/2 attempt 1: 0 face(s), 0 % unknown,
score 0.00` while the clip itself has both faces in every frame (`face_drift.py --match embed`: man 0.893 mean, woman
0.81 after her return). The cause (confirmed 05:50 from the code and from a scorer test, before the validation
render): with the streaming VAE patch installed, which is the default of this stack, `decode_latent` returns **uint8**
`(B, C, T, H, W)` frames, and the guard's conversion scaled them a second time to white. The scorer itself is fine:
`face_bank.py score` on five frames cut from the queue93 clip with the same three-image bank finds 7 faces, all known,
mean similarity 0.83 (min 0.64). `identity_guard.frames_to_uint8` now re-lays uint8 frames out without rescaling and
the journal prints the decoded tensor's dtype per check; queue94 (the same request, after queue92) validates it live.
The guard's price at 1344x768, read from the queue92 journal where it runs as a no-op: **67-68 s per window** for the
decode plus the scorer (8 frames), against a 1866-1927 s diffuse phase (3.5 %).
