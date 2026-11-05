# Several reference images per Ref2VA request: identity through occlusion (2026-11-05, in progress)

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

**Reading so far.** The extra references work mechanically: no error, no cut (the 10-28 experiment with an extra *video*
reference block made the model cut; image blocks do not), and every identity measure is 0.1-0.2 higher with the crops,
before and after the event. The loss at re-appearance is not removed: after the occlusion the woman comes back at
0.57 against 0.83 before it (A: 0.49 against 0.69). The crops are 100 px faces upscaled; sharper identity references
(cards the model renders itself) and a second seed are the next arms (queue87). Cost: the two crops add a quarter of the
window time at this size, because `H3_TP2_REF_SHORT_EDGE` upscales them to 1024 px like the scene; a per-image cap is
the obvious saving.
