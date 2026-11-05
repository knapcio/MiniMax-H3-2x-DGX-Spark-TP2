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
scene; `H3_TP2_REF_NO_UPSCALE=1` (queue90) keeps a reference below the setting at its own size.

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
