# 2026-11-03: splitting the VAE tile decode across the two ranks (`H3_TP2_VAE_SPLIT_RANKS`): already done, not built

Question: the head journal of the 60 s 1344x768 render (2026-11-02) shows six denoise windows on both GPUs and then a
510 s VAE decode of 1450 frames. Does that decode run on the head GPU only, so that a lever splitting the tiles over
the two ranks would halve it?

**Answer: no. The decode is already split over both ranks**, by vLLM-Omni's native tile-parallel decode
(`--vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling`, `scripts/start-head.sh:9`, our default since
2026-10-27 05:40, measured bit-identical then: framemd5 unchanged). Rank 1 holds the video VAE decoder, enters the same
decode at the same moment, decodes half of the tiles of every temporal clip and all-gathers them. Two-rank serving
decodes are 1.84-2.0x faster than the single-GPU decode of the same shape, so a second split has at most ~5 % of the
decode left to win (~0.1-0.2 % of a render). `H3_TP2_VAE_SPLIT_RANKS` was therefore not implemented: it would duplicate
the existing path. No code, env, diagnostics line or `queue-lib.sh` STRIP entry changed.

## How rank 1 takes part today (code, vLLM-Omni `a8576ccb7` and the checkpoint's `video_vae/` remote code)

- Every rank runs the whole pipeline `forward` (`DiffusionWorker_SP0` on Spark_06, `DiffusionWorker_SP1` on Spark_05),
  including `decode()` -> `video_vae.decode_latent` (`pipeline_minimax_h3.py:2075-2101`). The VAE is built on every rank
  (rank 1 log: `vae_temporal.py:266 MiniMax-H3 VAE streaming uint8 decode patch installed`).
- `registry.py:489` calls `MiniMaxH3VideoVAE.set_parallel_size(2, mode="tile")` (`vae.py:476-510`): the checkpoint's
  parallel state gets `sp_size=2`, `sp_rank=rank`, the DiT world group, and `model.parallel_tiling=True`.
- The checkpoint's `tiled_decode` (`video_vae/klvae.py:365-430`) gives rank r the tiles `range(r, num_tiles, 2)`
  (`_local_tile_indices`), runs them through `_run_tile_tasks` (where our `vae_tile_scope` batches up to 4 per decoder
  call, rank-locally), then `dist.barrier` + `_all_gather_tiled_results` (`all_gather_var_shape`) and blends the full
  grid on every rank in the stock order. This is exactly the design proposed for `H3_TP2_VAE_SPLIT_RANKS` (half the tile
  list per rank, every tile computed in full, stock blend), using NCCL on the DiT group, per temporal clip.
- The only rank-local fallback is a shape with fewer tiles than ranks (`vae.py:947`, logs "decoding rank-locally");
  that line is absent from both journals for every request since 2026-11-01.
- Tile grid (`split_tiles`, tile 256 px, overlap >= 64 px, spatial ratio 16, all tiles the same size): 1344x768 = 4x7 =
  **28 tiles, 14 per rank**; 960x576 and 960x544 = 3x5 = 15 tiles, 8 / 7. (The top-level README said "3x6"; corrected.)

## Measured evidence that rank 1 decodes (existing logs, nothing re-run)

1. **Same phase, same window on both ranks.** 60 s render: head `phase decode 510.3 s` at 18:19:34, rank 1
   (`~/h3-tp2-logs/worker-rank1-1793629881.log` on Spark_05) `phase decode 510.3 s` at 18:19:34. 5 s 1344 calibration of
   11-03 02:05:18: 26.4 s on both.
2. **Same memory signature on both hosts** (memwatch, MemAvailable MiB, 5 s 1344 T2VA 11-03): head 55 758 (denoise
   freed) -> 53 187 during decode -> 55 473 after; rank 1 57 664 -> 55 097 -> 58 166. Both hold the same ~2.5 GB
   decode working set over the same 26 s. During the 60 s decode both are flat (head 49.3 GB, rank 1 50.8 GB available).
3. **Two-rank serving vs single-GPU offline decode** (offline: `item2c`/`item2d` of `results/20261102-speed-plan/`, one
   GPU, head stopped; scaled by frame count):

| request | serving decode (2 ranks) | single GPU, same arm, scaled | ratio |
|---|---|---|---|
| 5 s 1344x768, 124 frames, stock tiles | 41.8-42.0 s | 186.8 s x 124/277 = 83.6 s | 2.0x |
| 5 s 1344x768, 124 frames, cap 4 | 26.3 s | 113.9 s x 124/277 = 51.0 s | 1.94x |
| 30 s 1344x768, 736 frames, cap 4 | 161.7 s | 113.9 s x 736/277 = 302.6 s | 1.87x |
| 60 s 1344x768, 1450 frames, stock | 510.3 s | 186.8 s x 1450/277 = 978 s | 1.92x |
| 5 s 960x576, 124 frames, stock | 23.9 s | 81.2 s x 124/226 = 44.6 s | 1.87x |

With rank 1 idle the ratio would be ~1.0x. The scaling is linear in temporal clips only to first order and the offline
numbers come from a stopped head, so read the ratios as +-10 %; they are far from 1.0 either way.

## What is left (none of it is a rank-split lever)

- **Post-decode tail on the head, 48 s of the 60 s render** (0.5 %): decode returned 18:19:34, engine stats 18:20:12
  (38 s: uint8 frames 4.5 GB to the CPU and through the worker -> engine handoff), MP4 encode on the API server 9.9 s,
  `200 OK` 18:20:22. 30 s: 4 s + 5.5 s; 5 s: 1 s + 0.9 s. The 38 s at 60 s is nonlinear (the head's MemAvailable fell
  to 36 GB in that window) and is the larger candidate. The existing `preencode_mp4` request option
  (`decode_to_mp4`: chunked decode with the MP4 mux on the worker, the default of the 11-01 queues) returns ~56 MB of
  MP4 instead of 4.5 GB of frames; whether its encoder settings match the API server's is unverified.
- **Tile balance:** 14/14 at 1344 (balanced); 8/7 at 960 (rank 0 one tile more, with cap 4 two decoder calls each,
  4+4 vs 4+3). At most a few percent of a 960 decode.
- **Per-rank batch granularity at 1344:** 14 tiles with cap 4 = calls of 4, 4, 4, 2. A different cap changes the bits
  (batch size changes cuDNN algorithm choice, 1 LSB as in item 2) and the peak; not lossless against today's default.

## Optional confirmation for a future serving slot (no restart of its own; piggy-backs on any 1344 request)

Sample both GPUs during the decode phase of the next request, e.g. on each host
`nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw --format=csv -lms 500 > /tmp/gpu-$(hostname).csv` from the
`phase diffuse` line to `phase decode`; expected: both GPUs busy for the whole decode. A clean A/B of the tile split is
`VAE_PP=1 scripts/start-head.sh` (rank-local decode on both ranks) against the default on the same 5 s 1344 T2VA seed 0
(latents digest `4d6d69e4` unchanged, framemd5 equal, decode ~2x longer with `VAE_PP=1`); it costs a restart slot and
was already done on 2026-10-27 (CHANGELOG), so it is not queued.

No GPU job was run for this question; no CPU tests were added (no code changed).
