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

## The pair at the same cells (`queue29.sh`, 05:37-05:53, long stack / FL2VA partition, T2VA, seed 0, 20 steps)

| Cell | Pair denoising (20 steps) | per step, pair | GPU-s per step (x2) | NVlabs single GB10 per eval | Pair decode | Pair wall | Pair peak |
|---|---|---|---|---|---|---|---|
| 960x576, first request | 180.0 s | 9.0 s | 18.0 | 15.2 | 24.7 s | 209.2 s | 60.4 GB |
| 960x576, warm | 173.0 s | 8.65 s | 17.3 | 15.2 | 24.8 s | 198.7 s | 60.5 GB |
| 1344x768 | 464.2 s | 23.2 s | 46.4 | 41.6 | 41.9 s | 507.8 s | 63.5 GB |

Per GPU the single-Spark lossless port is 12-14 % more efficient at 960x576 and 10 % at 1344x768 than our pair with the
two-pass GEMM and the fused SwiGLU (its 19 evaluations against our 20 steps make the gap about 5 % smaller still).
The night's earlier impression of a 2x gap came from Ref2VA numbers that carry the reference rows. What they fuse and
we do not: QKV projection (one GEMM), the per-token FP8 quantisation (we withdrew that level), AdaLN modulation and
the gate-add, RoPE; our batched decode (off in this run) brings the decoder to their level. The remaining per-GPU gap
is the size of those fusions; the pair's two-GPU scaling is nearly ideal at these cells (1344: 23.2 s/step on two
against 41.6 on one).
