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
