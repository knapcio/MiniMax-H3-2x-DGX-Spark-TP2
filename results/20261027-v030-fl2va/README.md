# vLLM-Omni v0.30.0 (vLLM 0.30.0, torch 2.13) on the two Sparks: FL2VA partition, 5 s 1344x768, 20 steps, seed 1101

Same package, same flags, same hosts; new venv at `~/h3-long` built with uv. First request on this stack failed on the
engine's request-cancellation shared memory (fixed: the remote rank defers to rank 0), the second is this row.

| | e1aa6eae + vLLM 0.26.0 (recipe) | v0.30.0 + vLLM 0.30.0 |
|---|---|---|
| Wall clock | 643.0 s | 645.0 s |
| Prompt / denoise / decode | 0.2 / 598.9 / 42.1 s | 3.9 / 593.4 / 44.7 s |
| Peak (X-Peak-Memory-MB) | 75.2 GB | 62.2 GB |
| Min free, head / worker | 38.8 / 39.3 GB | 53.2 / 54.1 GB |
| framemd5 | 8b3e06a8 | 9ded56d6 |
| SSIM / PSNR against the e1aa6eae two-Spark clip | | 0.906 / 30.8 dB |
| SSIM / PSNR against the one-Spark clip | 0.959 / 37.2 dB | 0.913 / 31.0 dB |

Frames at 2.5 s and 4.9 s inspected side by side: same scene, same sharpness and lighting, detail differences of the
kind seen between the two-Spark and one-Spark renders (a tree on the horizon, cloud shapes). Same speed, 13 GB less
peak memory on both hosts.
