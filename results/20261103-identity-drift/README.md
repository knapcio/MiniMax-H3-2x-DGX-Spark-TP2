# Identity drift across continuation windows: what can carry the faces, ranked (2026-11-03)

Status: a code survey and a plan. Nothing here was rendered; the fleet was running a proof queue. The new module
`h3_tp2/identity_lock.py` is opt-in (`H3_TP2_ID_LOCK`, unset = untouched) and is tested on CPU only
(`tests/test_identity_lock.py`, 15 tests). The plugin change is 8 lines in section 9e, outside the continuation loop
body.

## The measurement that sets the problem

Request `lalka60-fl2va` (queue56 arm A, `~/h3-tp2-results/lalka-ab/` on Spark_06): FL2VA, 60 s, 960x544, 20 steps,
seed 3101, `long_video` continuation with 277-frame windows and a 22-frame overlap, six per-window dialogue prompts,
serving defaults, 3194 s. `scripts/face_drift.py` (facenet VGGFace2 cosine against the reference frame), mean per window:

| window | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| man (x 0.35) | 0.748 | 0.673 | 0.636 | 0.684 | 0.609 | 0.554 |
| woman (x 0.65) | 0.741 | 0.682 | 0.568 | 0.524 | 0.554 | 0.476 |

Window 1 does not drift internally: it holds about 0.74 for 11 s (0.89 at t=0, then 0.70-0.79). The decline starts
in window 2 and continues within and across later windows. So the FL2VA keyframe conditions the whole of window 1
through attention, not only the frames near it. What windows 2+ lose is the keyframe itself (see Q2). Contact sheet:
the change is in face structure (jaw, nose, eyes, apparent age), not tone, so a colour or exposure correction cannot
recover identity.

Face boxes over the clip (MTCNN once per second, `face-boxes-lalka60-fl2va.json`): man x 0.27-0.43 / y 0.15-0.44,
woman x 0.59-0.72 / y 0.23-0.50. A face is about 90x115 px, which is 5.5x7 latent cells (16 px each) or 3x3.5 DiT
tokens. Heads move by up to ±2 cells horizontally and ±1.5 vertically. Faces this small receive few tokens, and they
move far enough to defeat a fixed-position latent paste.

Latent proxy (CPU, on the six saved window latents `latents-spark-06-20261103-1010..1050`, centered cosine of the face
patch against window 1's opening block, best of ±2-cell shifts): woman 0.66 / 0.60-0.63 / 0.61-0.65 / 0.58-0.60 /
0.57-0.60 / 0.52-0.55 (block-start and other tokens), which follows facenet. The man's values are 0.54 / 0.52-0.54 /
0.55-0.57 / 0.52-0.54 / 0.51-0.55 / 0.71, dominated by head pose. A background control patch varies between 0.61 and
0.92 with lighting. The proxy is useful as a log line but should not gate retries.

## Answers from the code (vLLM-Omni v0.30.0 a8576ccb7 on Spark_06, our plugin)

**Q1. Does the Ref2VA reference condition every window, and how strongly?** Yes. `continuation.py:111`
(`args = {**kwargs, ...}`) carries the request's `visual_condition` (the reference image rows) and its `ref_blocks`
into every window. Our edited loop does the same for ref2va because it clears the condition only for fl2va/t2va
(`plugin.py:700`). The reference is re-pinned at every step (`denoise_loop.py:432`).

There is no strength knob:
- The checkpoints are CFG-distilled, with no negative branch (`pipeline_minimax_h3.py:837`, `do_true_cfg = False` at `:2790`), so CFG on the reference is impossible.
- Condition noise augmentation is a constant 0.999 timestep (`denoise_loop.py:33`, `condition_noise.py`).
- There is no dropout flag at inference.

One structural weakening does exist. The reference image sits at the media time origin and is not moved by
`temporal_offset`, while each window's target is shifted by `start*40/24` RoPE units (`packed_sequence.py:612`,
docstring "leaving text, static images ... unchanged"). Window 6 of a 60 s clip is therefore 2125 units from its
reference, against roughly 600 inside a 15 s training clip. `H3_TP2_WINDOW_TEMPORAL_OFFSET` was meant to test this, but
it wraps upstream's loop, and since 2026-10-30 our edited loop (`H3_TP2_CONT_EDIT=2`, `plugin.py:757`) replaces that
loop. The switch is therefore a no-op on the current stack. The same applies to `H3_TP2_ANCHOR_FRAMES` and
`ANCHOR_SOURCE`: the 10-28 anchor results predate edit=2. The `toff0` mode in the new module restores the test.

**Q2. Can FL2VA take the reference as the last frame of every window?** The model can. The current continuation
cannot.

- What the model supports: FL2VA keyframes are separate condition rows on the target canvas, at the RoPE time of the frame they define. Frame 0 is at `text_len`; the last frame is at `text_len + span - 5/3` (`packed_sequence.py:236-238`). Accepted signatures are `[0]`, `[-1]` and `[0,-1]` (`encoder_processing.py:110-123`). A window with a pinned prefix and an end keyframe is therefore the trained first-to-last interpolation, with the first condition supplied as edit rows instead of a keyframe.
- Why the current continuation cannot: our continuation turns windows 2+ into t2va and removes the keyframe (`plugin.py:700`). It also rejects `frame_indices` containing -1 for multi-window requests before the first diffuse (`plugin.py:685`, `tests/test_cont_end_frame.py`). So `render.sh ... both` + `H3_END_IMAGE_FILE` fails on a 60 s request.
- Per-window end frames: there is no API field for them. FL2VA accepts at most two images per request.
- What the new module does: `H3_TP2_ID_LOCK=end` reuses window 1's own keyframe rows as each later window's `[-1]` keyframe, so no new input is needed.
- The text side already matches: every window's prompt is re-encoded with the request's images (`pipeline_minimax_h3.py:2382`, `replace(prepared, prompt=...)`), so each window's text already begins `<Picture 1>: [Qwen3-VL tokens of the reference]` (`preprocessing.py:246`). This is a semantic identity channel present in every window today. It is visibly too weak at the face scale.

**Q3. What is the cheapest way to lengthen the identity context?**
- Overlap is a request field (`continuation_overlap_frames`, 17n+5, below the window, `continuation.py:60-64`), not a constant. 39 was tested on 10-27 (+9 % time). The pinned tail is already a drifted copy, so a longer tail slows the per-window random walk without re-anchoring it. At 56 frames a 60 s clip needs 7 windows instead of 6 (~+17 %).
- 345-frame windows (the maximum) mean 5 hand-overs instead of 6, each window carrying +24 % rows. That is fewer re-samplings, but within-window drift remains.
- An opening-frames video reference, masked or down-weighted, is closed. Any extra video-reference block made the model cut on 10-28, including the tail-content control. Ref blocks must be rectangular latent grids normalised by their own area, so a "face-masked" block is really a crop presented as a full frame. No per-block weight exists either.
- The trained way to add face-only context is Ref2VA image references (up to 9): face crops as extra pictures. See arm R2.

**Q4. A spatially local latent identity lock.** Designed and implemented, opt-in; see the last section. Verdict: the
structural blend will read as ghosting or a stuck mask before it reads as identity, unless it is very weak and
low-passed. At that strength it mostly carries tone and face shape. The statistics-only lock is safe but does not
address the structural drift we see.

**Q5. Upstream or community mechanisms.**
- The vLLM-Omni tree has nothing for identity in continuation. Grepping "identity", "subject", "anchor", "consistency", "ip_adapter" and "face" finds only adapter metadata and condition-anchor rows.
- The recipe documents FL2VA `[0,-1]` and Ref2VA with up to 9 images.
- The HF card describes Ref2VA subjects as `<Subject 1>` (from its Context-IR prompt tool; our presentation labels them `<Picture i>`) and lists faces as "fully_preserved" in its Ref2VA example. It says nothing about continuation; generations are 4-15 s.
- ttulttul's ComfyUI continuation (the origin of upstream's algorithm) only guides with the latent tail. Its I2V node has `first_frame`/`last_frame` inputs that the workflow leaves disconnected; connecting `last_frame` is our `end` mode.

Other code facts used below:
- FL2VA and Ref2VA share the transformer config and every tensor shape, but every DiT tensor differs: a full fine-tune, checked byte-wise on shards 1/7/13. The text encoder and VAE are shared.
- Serving loads one partition (`H3_MODEL`, currently FL2VA). Ref2VA needs a partition switch and a head restart, as in `queue58-lalka-b.sh`.
- Reference tokens in the FL2VA partition are therefore out of distribution, except as keyframes at target frame positions.
- The latent-edit path accepts fractional per-row masks (quantised to 1/256, `latent_mask.py:17,115-121`): a soft-inpainting hook used by the phase-2 design.

## Ranked mechanisms

| # | mechanism | how | render cost (60 s 960x544, base 3194 s) | lossless? | expected effect on identity | risk |
|---|---|---|---|---|---|---|
| 1 | **Hidden end anchor**: FL2VA, reference as last keyframe of an extended canvas, cropped | `H3_TP2_ID_LOCK=end`, `H3_TP2_ID_END_EXTEND=68` | windows 2-6 at 102 instead of 82 tokens: ~+35 % (~4300 s, estimate) | no (changes windows 2+) | high: every window is attended by a clean reference keyframe, the same thing that holds window 1 at ~0.74 | motion drifts toward the reference pose near the cropped end; possible lip-sync slowdown; new path, never rendered |
| 2 | **Visible end anchor** | `H3_TP2_ID_LOCK=end` | +1 condition frame, ~+1 % | no | highest at window ends (snap back to ~0.89), sawtooth inside | every window returns to the opening pose (a "rewind" every 11 s), and windows start alike (the 10-30 "shown 5-6 times" problem) |
| 3 | **Ref2VA continuation with the reference** (queued: queue58 arm B) | partition Ref2VA, `TASK=ref2va`, 2048 px | +17 % rows; Ref2VA vs FL2VA at 5 s 1344 measured +59 % (2048, warm) on 10-28, so expect ~+40-60 %, plus a restart | no | medium-high: 7296 reference tokens in every window; unmeasured for faces so far | Ref2VA recomposes rather than starting on the frame; the RoPE distance grows per window (arm R1b) |
| 4 | **Ref2VA + face crops as Pictures 2-3** | partition Ref2VA, three images, `H3_TP2_REF_SHORT_EDGE=1024` | +9 % rows; Ref2VA at 1024 measured +17 % vs FL2VA at 5 s 1344, so expect ~+15-25 %, plus a restart | no | high in principle: subject references are what the partition was trained for, and each face gets ~1024 tokens instead of ~12 | a cut to a close-up of a crop; the prompt must name the roles |
| 5 | Ref2VA with local time per window | `H3_TP2_ID_LOCK=toff0` on arm 3 or 4 | 0 | no | small-medium (keeps the reference at window 1's distance) | out of training range the other way (the target sits at the same time as window 1) |
| 6 | Latent face blend (post-hoc) | `H3_TP2_ID_LOCK=blend` + boxes | ~0 (CPU-cheap tensor work) | no | small: +0.03-0.04 latent face similarity in the CPU replay at 0.3 | ghosting, a face "stuck" while the head moves, damped mouth; 17-frame steps if strong |
| 7 | Longer windows (345) | `continuation_window_frames: 345` | ~+8 % (10-28: 30 s +8 %) | no | small (5 instead of 6 re-samplings) | none new |
| 8 | Larger overlap (39/56) | `continuation_overlap_frames` | +9..17 % | no | small (re-anchors to a drifted tail) | none new |
| 9 | Latent face stats lock | `H3_TP2_ID_LOCK=stats` + boxes | ~0 | no | ~none on structure (+0.003 latent) | per-block face brightness steps if strong |
| 10 | 1344x768 | resolution | ~x2.3 | no | unknown; twice the tokens per face | time |

Rejected without a render:
- Keyframe at frame 0 plus the pinned tail. The frame-0 position would hold two contradictory conditions, the
  reference and the tail, and the seam check exists because conflicts become cuts.
- Opening-frame video anchors (closed on 10-28).
- External face swap or restoration of decoded frames. It is outside the stack, the usual models are non-commercial,
  and it is per-frame rather than temporal.

Order of testing: arm B (already queued) and E2 hidden end anchor first, because they are the two trained channels.
Then R2 face crops if B helps, E1 visible end as the upper bound for identity, then the latent locks only as add-ons.

## Testable arms (all on the lalka-ab inputs)

Common to all arms:
- Inputs: `~/h3-tp2-results/lalka-ab/prompt.txt`, `extra.json` (six per-window prompts, `audio_flow_shift` 3.0), and `~/h3-tp2-results/lalka-coherent300-20261101/inputs/reference.png` (1672x941).
- Request: `run <name> 60 "60:960:544:20:3101" <reference.png>` via `queue-lib.sh`, followed by
  `face_drift.py <mp4> <reference.png> --json lalka-ab/drift-<name>.json`, as in `queue56-lalka-a.sh`.
- Prerequisites for the plugin arms: sync `h3_tp2/identity_lock.py` and `plugin.py` to both hosts, then restart. All
  `H3_TP2_*` variables are forwarded to rank 1 (`executor.py` `_FORWARD_PREFIXES`).

**A0 (done):** FL2VA baseline above (`lalka60-fl2va`, 3194 s).

**E1, visible end anchor (FL2VA partition):**
```bash
restart_head_gated "export H3_TP2_SAVE_LATENTS=/home/knapcio/h3-tp2-results/latents
export H3_TP2_WINDOW_SEED_STEP=1
export H3_TP2_ID_LOCK=end"
export TASK=fl2va H3_PROMPT_FILE=~/h3-tp2-results/lalka-ab/prompt.txt H3_EXTRA_JSON_FILE=~/h3-tp2-results/lalka-ab/extra.json
run lalka60-fl2va-end 60 "60:960:544:20:3101" ~/h3-tp2-results/lalka-coherent300-20261101/inputs/reference.png
```
Check the log for `identity end anchor (call k ...): reference as last keyframe, canvas 277 -> 277` on windows 2-6.
Window 1 must be byte-identical to A0 (same seed and args). Watch the seam ratios, which act as the cut detector, and
watch the last second of each window for the rewind.

**E2, hidden end anchor:** as E1, with `export H3_TP2_ID_END_EXTEND=68` added (canvas 277 -> 345, cropped back to 277). E2
is skipped automatically, with a warning, if `audio_mode=lock_source` or a pinned `pad_seq_len` is in use. Peak memory is
that of 345-frame windows (72.0 GB on 10-28 at 960x576).

**E3, E2 with the latent face lock as an add-on (only if E2 is clean):**
```bash
export H3_TP2_ID_LOCK=end,blend H3_TP2_ID_END_EXTEND=68 H3_TP2_ID_LOCK_STRENGTH=0.2 \
       H3_TP2_ID_LOCK_BOXES="0.306,0.151,0.400,0.376;0.612,0.232,0.703,0.440"
```
The boxes are the reference's MTCNN boxes from `face-boxes-lalka60-fl2va.json`.

**R1, Ref2VA reference in every window (= queue58 arm B, already queued):** partition switch to Ref2VA, `TASK=ref2va`,
`H3_TP2_REF_SHORT_EDGE=2048`, same prompt/extra/seed. **R1b** adds `export H3_TP2_ID_LOCK=toff0` on the same boot. If
the faces hold, repeat at `REF_SHORT_EDGE=1024` for the speed setting.

**R2, Ref2VA with face crops as extra pictures** (Ref2VA partition, `H3_TP2_REF_SHORT_EDGE=1024`). `render.sh`
sends at most two images, and only for FL2VA, so this arm posts directly. Crops are squares around the reference
boxes, about twice the face height:
```bash
cd ~/h3-tp2-results && mkdir -p lalka-ab/r2 && ~/h3-long/.venv/bin/python - <<'EOF'
from PIL import Image
im = Image.open("lalka-coherent300-20261101/inputs/reference.png").convert("RGB")      # 1672x941
im.crop((370, 28, 810, 468)).save("lalka-ab/r2/face-wokulski.png")                       # man, box x 512-669 y 142-354
im.crop((879, 96, 1319, 536)).save("lalka-ab/r2/face-izabela.png")                      # woman, box x 1023-1175 y 218-414
EOF
P='<Picture 1> is the full scene: the shop, the counter, both people, their clothes and the lighting. <Picture 2> is the face of Wokulski, the man. <Picture 3> is the face of Izabela, the woman. Keep exactly these two faces. '
jq --arg p "$P" '.continuation_prompts |= map($p + .) | . + {"task":"ref2va","duration":60}' lalka-ab/extra.json > lalka-ab/r2/extra.json
{ printf '%s' "$P"; cat lalka-ab/prompt.txt; } > lalka-ab/r2/prompt.txt
curl -sS --max-time 30000 -X POST http://127.0.0.1:18093/v1/videos/sync \
  -F "prompt=<lalka-ab/r2/prompt.txt" --form-string width=960 --form-string height=544 --form-string aspect_ratio=16:9 \
  --form-string fps=24 --form-string num_inference_steps=20 --form-string flow_shift=12.0 --form-string seed=3101 \
  -F "extra_params=<lalka-ab/r2/extra.json" \
  -F "input_references=@lalka-coherent300-20261101/inputs/reference.png;type=image/png" \
  -F "input_references=@lalka-ab/r2/face-wokulski.png;type=image/png" \
  -F "input_references=@lalka-ab/r2/face-izabela.png;type=image/png" \
  -o lalka-ab/r2/d60_960x544_s20_seed3101_ref2va_faces.mp4 -w '%{http_code} %{time_total}\n'
```
Reference rows at 1024 px: the scene is 1824 rows and each face crop is 1024 rows, about 3900 in total against 41820
target rows per window. Prompt length and image labels are re-encoded per window, as in Q2.

**C1, cheap structural control (FL2VA, no plugin change):** add `"continuation_window_frames": 345` to `extra.json`.
This makes 5 windows, so `continuation_prompts` must have 5 entries: drop the last line. Optionally add
`"continuation_overlap_frames": 39`; that is still 5 windows (shift 306). Expect small effects; this arm separates "fewer
re-samplings" from "a clean anchor".

**What to read in each arm:**
- `face_drift` per-window means and the last-10 % value: target is within 0.05 of window 1 (0.74) through window 6.
- `clip-metrics` cuts and seams.
- The look-lock and seam log lines.
- For E1/E2/E3: the `identity end anchor` and `identity lock` lines.
- The owner's eye on the last second of each window (E1) and on face edges (E3).

## The latent identity lock: design, implementation, honest expectation

**Reference.** No VAE encode is needed. In FL2VA, window 1's keyframe rows are the reference image VAE-encoded on the
target canvas: `visual_condition`, a block-start token (class A). Window 1's opening block is the model's own
rendering of that frame, with token 0 in class A and tokens 1-4 in class B. Both ranks already hold these identically.
The lock uses token 0 for class A and the mean of tokens 1-4 for class B. This respects the class structure behind the
17-frame flicker (`results/20261102-vae-chunk-flicker/`): block-start tokens are only ever pulled toward block-start
statistics.

**Alignment.** The camera is fixed, so faces live near their reference boxes but move by up to 2 cells. For each free
token and face, an integer shift within `H3_TP2_ID_LOCK_RADIUS` (default 2) maximises the centered cosine between the
token's shifted face patch and the reference patch. Shifts are then held constant per 17-frame block (median of the
block's tokens), so the anchor moves between blocks and does not jitter inside one. On the A0 latents 47-72 of 75
tokens per window chose a non-zero shift, consistent with the MTCNN box motion.

**Correction.**
- Feathered box mask: cosine falloff over `H3_TP2_ID_LOCK_FEATHER` cells, moved with the shift.
- Block-constant temporal ramp to full strength over `H3_TP2_ID_LOCK_RAMP` of the free tokens. The decoder renders a per-token latent ramp as per-block steps anyway, so the steps are kept small and placed at block starts.
- The pinned overlap is never touched, and only the free tokens of windows 2+ change.
- `blend` moves the face patch toward the shifted reference by `strength`, low-passed by default on the 2x2 DiT token grid (`H3_TP2_ID_LOCK_LOWPASS=1`).
- `stats` only matches the per-channel mean/std of the aligned face patches per class (gain clamped to 0.8-1.25).
- `measure` logs the latent similarity only.
- The lock runs outside the seam-retry/look-lock wrapper, after the look-lock, so seam scores are computed on the raw sample.

**CPU replay on the A0 latents** (window 1 as the reference, real boxes, no decode):

| mode | latent face similarity, man (w2..w6) | woman (w2..w6) | max / mean abs latent change (latent std 1.05) |
|---|---|---|---|
| measure | 0.532 0.563 0.538 0.540 0.713 | 0.622 0.644 0.596 0.592 0.542 | 0 |
| stats 0.5 | +0.003..0.004 | +0.002..0.003 | 0.55-0.68 / 0.008 |
| blend 0.3 lowpass | +0.030..0.044 | +0.031..0.046 | 0.58-0.70 / 0.007 |

**How it will render.** This is an expectation, not a measurement.

The post-hoc blend works on clean final latents. The decoder (a nonlinear ViT) renders a mix of two faces as a mix:
- If alignment is off by about one cell (16 px, a sixth of the face) or the expression differs, eyes, brows and mouth edges double faintly: ghosting.
- Because the reference is one static frame, the anchored part does not move with the head. Above roughly 0.3 it reads as a mask sliding over the face, and it damps mouth motion during speech.
- Low-passing to the 2x2 token grid suppresses the doubling but leaves mainly skin tone, hair mass and face outline: a small identity gain.
- Per-block strength changes can show as 17-frame face brightness steps if strong.

So it is not a "snap": it is applied gradually per block and never jumps. The failure mode is ghosting and a stuck face.
Recommended only at 0.15-0.3, low-passed, as an add-on to a trained anchor (E3), never alone.

**Phase 2 (design only, not implemented): the in-loop version that avoids ghosting.**
- Feed the same aligned reference face through the edit path that already pins the prefix: `video_edit_clean_rows` holds the reference rows in the face boxes, and `video_edit_mask_rows` is fractional there (e.g. 0.85).
- The model then sees the face rows as partly clean at a lower noise level and re-renders them in context, with motion.
- The obstacle: the stock edit uses one mask for all steps. At the last step it would blend the anchor into the output just like the post-hoc blend.
- The fix is a step-scheduled subclass of `MiniMaxH3LatentEdit` that uses a fractional mask while sigma > ~0.6 and mask 1 afterwards. `model_rows`, `target_timesteps` and `x0` receive the timestep; `to()` must be overridden to keep the subclass.
- It would be installed by rebinding `pipeline_minimax_h3.MiniMaxH3LatentEdit` and fed per window through the inner wrapper.
- Fractional masks are a supported encoder input (ComfyUI-style), but a fraction applied only early is untested in H3.
- Worth building only if E2/R1 hold identity but a residual drift remains.

## Side findings

- `H3_TP2_WINDOW_TEMPORAL_OFFSET` and `H3_TP2_ANCHOR_FRAMES` / `ANCHOR_SOURCE` (plugin section 9) wrap upstream's loop. Since edit=2 became the default (2026-10-30), section 9e replaces that loop, so these switches do nothing on the current stack. Document or remove them; `toff0` in the new module replaces the first.
- Before this change, `tests/test_cont_checkpoint_loop.py`, `test_cont_end_frame.py` and `test_cont_hook_binding.py` already ran 30 errors (`NameError: _lock_classes`). Their synthetic loop environment predates `H3_TP2_CONT_LOCK_CLASSES`. The count is identical with and without the plugin hunk here.
- Window 1 of an FL2VA continuation already reaches identity ~0.74 at 960x544 and holds it. This is the ceiling the anchors aim for, not 0.89 (only t=0 has that).

## Files

- `h3_tp2/identity_lock.py`: modes `end`, `toff0`, `stats`, `blend`, `measure`; env reference in the module docstring.
- `h3_tp2/plugin.py` section 9g: `identity_lock.install(cont_match)`, wrapped so a bad setting only logs an error.
- `tests/test_identity_lock.py`: 15 CPU tests on fake latents and a fake diffuse.
  - Configuration: env parsing and validation; install is a no-op when unset.
  - End anchor: keyframe reuse; padding of the extended edit rows (video and channel-major audio) and cropping of the result; skip with a locked soundtrack.
  - Modes: `toff0` scope.
  - Face-lock helpers: feather, shift, low-pass, alignment recovery of a moved face, block-constant ramp.
  - Face-lock behaviour: nothing outside the feathered box and pinned prefix changes; strength 0 is identity; blend reduces the error; stats reach the reference statistics.
  - Composition: through the real `cont_match.wrap_diffuse`, keeping its checkpoint hooks.
- `face-boxes-lalka60-fl2va.json`: reference and per-second MTCNN face boxes of A0.
