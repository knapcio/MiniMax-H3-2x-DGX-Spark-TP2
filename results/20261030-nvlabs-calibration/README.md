# NVlabs sol-engine GB10 port, lossless tier, calibrated on one of our Sparks (2026-10-30 01:39-02:30)

`run-calib.sh` (see `20261028-nvlabs-sol-engine-gb10/`): the single-GPU `gb10_kernel` configuration (pruned FP8 DiT,
FP8 Qwen3-VL conditioner, fused QKV / quant / AdaLN / RoPE / SwiGLU Triton kernels, batched VAE tiles; Sol-Attn and the
block cache OFF, so the lossless tier), T2VA, seed 0, 20 steps (19 DiT evaluations), one warm-up request, on spark-06
with the TP2 head stopped. `benchmark-k960-k1344.json` has both runs.

| Cell (single GB10) | Denoising (19 evals) | per DiT eval | VAE decode | Request | Peak |
|---|---|---|---|---|---|
| 960x576, 124 frames | 288.9 s | 15.2 s | 37.2 s | 333.2 s | 74.3 GB |
| 1344x768, 124 frames | 790.4 s | 41.6 s | 68.9 s | 862.8 s | 74.9 GB |

Our pair's numbers at the same cells (T2VA, 20 steps, two-pass GEMM + fused SwiGLU level 1, stock decode) are
rendered right after the night's look-lock chain (`queue29.sh`: `cmp-t2va-960`, `cmp-t2va-960-warm`, `cmp-t2va-1344`)
and go into this table as GPU-seconds per evaluation (pair seconds x 2) for the per-GPU comparison. Known so far from
FL2VA renders of the evening: 1344x768 5 s on the pair, 459 s denoising for 20 steps = 23 s per eval on two GPUs
(46 GPU-s), against 41.6 GPU-s here: the single-GPU port is about 10 % more efficient per GPU at 1344x768 before any
of its lossy tiers. Their decode (batched tiles) 37 / 69 s on one GPU against our batched decode 16 / 27 s on two
(32 / 54 GPU-s): the same order.
