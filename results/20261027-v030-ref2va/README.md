# Ref2VA partition on vLLM-Omni v0.30.0, two Sparks: 5 s 1344x768, 20 steps, seed 1101, one reference image

Reference: `source-4k.jpg` (a painted oak forest), bench prompt (hills at golden hour). Upstream documents Ref2VA at
1.92x the T2VA cost on one GB10.

| | FL2VA (same stack) | Ref2VA |
|---|---|---|
| Wall clock | 645.0 s | 1248.6 s (1.94x) |
| Prompt / denoise / decode | 3.9 / 593.4 / 44.7 s | 13.5 / 1182.1 / 43.5 s |
| Peak (X-Peak-Memory-MB) | 62.2 GB | 65.6 GB |
| Min free, head / worker | 53.2 / 54.1 GB | 49.5 / 49.7 GB |
| framemd5 | 9ded56d6 | 80e963c8 |

Observation: Ref2VA treats the image as an omni-reference (subject, style), not as the first frame: with a prompt that
describes a different scene the clip follows the prompt and keeps only the lighting mood of the reference. For a
reference-faithful clip the prompt has to describe the reference. FL2VA with `frame_indices [0]` remains the mode for
"start from this image".
