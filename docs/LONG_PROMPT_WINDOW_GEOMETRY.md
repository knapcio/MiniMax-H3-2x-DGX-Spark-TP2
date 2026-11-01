# Long prompts: keep the submitted window plan and text assignments identical

Verified 2026-11-01, Mac CPU; no model invocation, service change or queue change.

`scripts/plan_dialogue_windows.py` now includes `continuation_window_frames`
and `continuation_overlap_frames` in its generated `extra_params`. Previously
`--window` and `--overlap` changed the planning report and dialogue assignments,
but the request omitted both values. The server therefore used 277/22.

This is a request correctness fix, not a demonstrated improvement in model
quality. The defaults remain 277/22 and produce the same prompt strings.

## Reproduced failures before the fix

| Duration / planner geometry | Planner / server window counts | Failure |
|---|---|---|
| 30 s, 345/22 | 3 / 3 | A line at 12–13 s is assigned to prompt 1, but the default server's first window ends at 11.542 s. Count validation cannot detect this. |
| 30 s, 277/39 | 3 / 3 | A line at 21.7–22 s is assigned to prompt 3. The planned second window ends at 21.458 s; the default server's ends at 22.167 s. |
| 60 s, 345/22 | 5 / 6 | The server rejects the five explicit prompts because it plans six windows. |

The old behavior was reproduced by executing the planner from git HEAD before
the fix. The two geometry fields were absent in every request above. The fix
sends both integers with the explicit prompt list through `render.sh`.

## Actual text path and limits

The inspected Spark_06 source tree is `/home/knapcio/h3-long/vllm-omni`.
Read-only copies and full SHA-256 values are retained in the enclosing workspace
under `diagnostics/h3-long-prompt-audit-20261101/sources.json`. This is an on-disk
source inspection, not a claim that every byte was observed executing in the
already running process.

1. `render.sh` merges `EXTRA_JSON`, preserves explicit `continuation_prompts`,
   and sends literal text with curl `--form-string`. The earlier semicolon
   truncation bug is already fixed separately.
2. `h3_tp2/plugin.py` generates continuation wording only when an explicit
   prompt list is absent. Its default repeats the whole base prompt in every
   window, including any dialogue; it does not split a story into timed scenes.
3. `encoder_processing.py` selects duration using `target.duration_seconds`,
   then `duration_seconds`, then `duration`, and aligns the 24 fps frame count.
   `continuation.py:resolve_continuation` reads the two exact geometry fields
   above, with defaults 277/22. Windows must be 107..345 on the 17n+5 grid;
   overlap must be on that grid, at least 5 and smaller than the window.
4. `pipeline_minimax_h3.py:_prepare_local_conditioning` validates exactly one
   nonempty string per planned window. It encodes each prompt separately and
   stores all window embeddings before diffusion. `encode_prompt` calls
   `build_minimax_h3_presentation`; `preprocessing.py:_text_ids` calls
   `tokenizer(text, add_special_tokens=False)`. No truncation or maximum length
   is requested at that call. This does **not** establish unlimited supported
   context, memory, or instruction following. A verified safe token budget for
   the full text-plus-vision presentation is unknown.
5. The continuation loop selects embeddings by window index. It anchors media
   time after the longest text prefix and carries the preceding latent tail;
   it does not carry a textual narrative state or enforce exact word timestamps.

The planner sends scene text in every prompt and each dialogue line in one
window's new frames. It rejects dialogue crossing a handover. The input times
choose an owner window; they are not converted into intra-window timing tokens.
Exact delivery still requires separately timed audio and verification.

Existing results support caution: the 30 s per-window sample removed a dissolve
and retained motion but still lost about 8 dB of generated audio. The 60 s
345-frame sample had smooth joins yet two cuts *inside* windows and an
over-sharpened sky. Those single-sample observations do not establish a general
benefit from longer text or larger windows. See
`results/20261027-v030-continuation-winprompts/README.md` and
`results/20261028-v030-continuation-60s-win345-prompts/README.md`.

## Verification and next falsifiable experiment

`python3 -m unittest discover -s tests -p 'test_plan_dialogue*.py' -v`: 10/10
passed, including five new geometry tests. They exercise the actual render
shell script with captured curl, custom geometry with identical/different
window counts, default prompt compatibility, JSON integer preservation and
the 300 s plan. `python3 -m unittest discover -s tests -p 'test_render*.py' -v`:
14/14 passed, including the real local multipart punctuation test.

No GPU experiment is needed to demonstrate the fixed request mismatch. When
the render queue is idle, a separate quality experiment can compare a 30 s
345/22 request containing one line at 12–13 s using the old omitted geometry
versus the corrected explicit geometry, with the same source audio, reference,
seed and steps. Record actual server windows; measure per-line waveform delay,
ASR and speaker mouth motion around 11–15 s. A failure to improve these metrics
would falsify a quality claim, while leaving the request-geometry correction
valid. Changing the window geometry itself changes the denoising problem, so
this comparison cannot isolate a text-only quality effect.
