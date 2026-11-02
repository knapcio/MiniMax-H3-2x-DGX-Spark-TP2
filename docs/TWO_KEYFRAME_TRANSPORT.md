# First and last frame transport (experimental)

The opt-in `H3_END_IMAGE_FILE` lets `scripts/render.sh` send two distinct
images for FL2VA, in first/last order, using repeated multipart
`input_references`. The ninth positional argument must be `both`:

```sh
TASK=fl2va H3_END_IMAGE_FILE=/path/to/last.png \
  bash scripts/render.sh out.mp4 10.125 960 544 20 3101 \
  'A person sets the letter down on the counter.' /path/to/first.png both
```

The client snapshots both images to private files, records their byte counts
and SHA-256 values in `out.mp4.timing.json`, and rejects mismatched task or
frame indices and any request exceeding one continuation window. PNG, JPEG,
WEBP, HEIC and HEIF filenames are supported; each input is bounded to 30 MiB
based on the inspected server source. Content decoding and the server's actual
limits still need verification. Existing one-image calls retain their prior
multipart form.

H3 aligns its internal frame count to the 17n+5 grid. The timing record
therefore includes `requested_frames`, `aligned_frames`, and
`alignment_tail_frames`: 10.0 s requests 240 frames but aligns to 243.
The model's `-1` endpoint refers to the internal last frame. Trimming that
result to exactly 10.0 s can remove the endpoint; the example 10.125 s is
exactly 243 frames and needs no such alignment trim. Even then, decoding and
conditioning do not guarantee that the visible final frame equals the supplied
image or that the movement between images is realistic.

CPU tests exercise real curl multipart transport to a localhost receiver,
preserve reference order and hashes, and reject malformed or multi-window
requests before HTTP. The inspected H3 source accepts two image references
and indices `[0, -1]`; actual loaded server routing, model inference, motion,
identity and endpoint fidelity have not been tested with this client. Native
multi-window final-frame conditioning is deliberately excluded: an older loop
could use that image at the end of the first window and drop it afterward.
Do not use a single-frame target as evidence that the action was performed.

## Exact-length plans for separate segments

`plan_keyframe_segments.py` is a CPU-only geometry planner for an edited sequence of independent, single-window
two-keyframe requests. It finds the fewest source segments on the 17n+5 grid between 107 frames and the chosen
window limit; it does not submit requests or estimate render time. The 107-frame minimum follows this client's
four-second request minimum and alignment rule. The balanced result is not necessarily the fastest or best-looking
shot division.

```sh
python3 scripts/plan_keyframe_segments.py --seconds 300
python3 scripts/plan_keyframe_segments.py --seconds 300 --drop-boundary-frame
python3 scripts/plan_keyframe_segments.py --frames 260  # exact 260/24 s, no rounded decimal required
```

At the default 277-frame limit, 300 s at 24 fps is exactly 7,200 frames. Keeping every segment frame needs a
minimum of 29 segments: 20 of 243 frames and nine of 260. If the previous final image is used as the next
reference, both sides of a boundary remain in the edit; they might look like a repeated frame, but they are not
guaranteed to be identical. `--drop-boundary-frame` instead plans to discard frame 0 of every segment after the
first while retaining each generated last frame. That policy needs 36 segments: 17 of 192 and 19 of 209 frames,
for 7,235 source frames less 35 discarded frames. Neither policy guarantees a smooth transition or actual
movement between references.

The output includes source-request duration, retained timeline frames and cumulative sample boundaries. Audio
boundaries are calculated on the retained timeline without accumulating per-segment rounding: frame and sample
intervals are half-open, and cumulative sample positions use round-half-to-even. `possible_preserving_endpoints`
refers to each request's **last** generated frame; with the drop option, later requests' first frames are
deliberately removed. At 32 kHz a frame
is 4,000/3 samples, so `source_audio_cut_start_samples_exact` can be fractional: the edit must explicitly choose
how to align or resample the actual decoded audio. The planner does not cut audio. These values are a plan, not
proof that a generated AAC stream has the requested length or preserves the dialogue.

Some target lengths have no solution with this particular one-window, no-endpoint-truncation policy. For example,
10, 15 and 30 s have no solution under either boundary policy at the default window; 60 s has a seven-segment
solution only with `--drop-boundary-frame`. The planner returns JSON with `possible_preserving_endpoints=false`
and exit code 2 rather than silently trimming. This does not prevent normal H3 renders of those lengths; it means
another editing or conditioning strategy is needed if every segment's generated final frame must be retained.
`--seconds` accepts exact decimal values on the 24 fps grid; use `--frames` when the exact time is a repeating
decimal. Server-side behavior and all visual/audio quality remain to be tested.
