# `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: identical frames, 1 GB less peak, 3 % slower. Rejected.

5 s 1344x768, 20 steps, seed 1101, final two-Spark configuration, fresh boot with compacted memory, both ranks with
the setting (it is forwarded to rank 1).

| | default allocator | expandable segments |
|---|---|---|
| Wall clock | 643.0 s | 664.0 s (+3.3 %) |
| Denoising / VAE decode | 598.9 / 42.1 s | 609.6 / 49.8 s |
| Peak (X-Peak-Memory-MB) | 75.2 GB | 74.1 GB |
| Min free, head / worker | 38.8 / 39.3 GB | 38.9 / 40.7 GB |
| framemd5 | 8b3e06a8 | 8b3e06a8 (identical) |

The 20 s render already leaves 20 GB free on both hosts, so the memory is not needed and the time is. Off.
