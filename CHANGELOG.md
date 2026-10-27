# Changelog

## Unreleased

- 2026-10-26: first two-Spark boot of MiniMax H3-Base FL2VA with text-encoder TP2 + Ulysses SP2 on vLLM-Omni `e1aa6eae`
  through `h3_tp2.executor.RemoteReaderExecutor` (rank 1 reached over the ConnectX-7 fabric with vLLM's remote-reader
  message queues; no Ray, no changes to the vLLM-Omni tree). Worker agent + systemd units, preflight, render/memory/verify
  scripts.
- 2026-10-27: hardening after review. Executor: one startup deadline over both ranks and both ZMQ handshakes, abort path
  that terminates what was started, serialized failure handling (callbacks once, never during an intentional stop),
  head-to-agent lease pings, request watchdog (`H3_TP2_REQUEST_TIMEOUT_S`), env allowlist for rank 1 (no credentials or
  worker-owned keys). Agent: single writer/reader threads, lease expiry, SIGTERM/SIGKILL on every session end. Wire:
  optional HMAC-SHA256 per frame (`H3_TP2_SECRET`), TCP keepalive + `TCP_USER_TIMEOUT`. Plugin fails closed. Memory
  guards: role-based actions that actually run (the previous unquoted `Environment=` lines did not), post-action check.
  Benchmark: detached memory sampler (the first matrix hung 3.5 h on an ssh channel held open by the sampler), cleanup
  traps, no overwrites, per-case STATUS, media acceptance that fails on missing streams, wrong size/duration or silent
  audio. Preflight validates netdev/IPv4 per HCA, compacts before checking the 110 GiB floor and fails on missing
  software. CPU test suite for the control channel (10 tests).
- 2026-10-27 05:17: uint8 output path (`h3_tp2.h3_output`, default on): frames leave the GPU as uint8 with the stock
  path's arithmetic; same decoded frames as the FP32 path (framemd5 `c36d17dc` on the 5 s 960x576 seed-1101 case from
  both), the head keeps ~10 GB more free during a 10 s 1344x768 render (19.9 vs 10.2 GB). Phase timers and a SHA-256 of
  the frames in the head journal. Measured against the one-Spark recipe, same seed: 5 s 1344x768 20 steps 681.8 s vs
  1399.0 s; 5 s 960x576 10 steps 177.4 s vs 290.9 s; 10 s 1344x768 FL2VA 1940.3 s (one Spark cannot). 15 s 1344x768 ran
  the head out of host memory with the FP32 output path (kernel OOM-kill at 7796 s, no livelock).
- 2026-10-27 05:40: defaults `--vae-patch-parallel-size 2` and `H3_TP2_EMPTY_CACHE_BEFORE_DECODE=1`, both measured
  bit-identical on the two 5 s seed-1101 cases (framemd5 unchanged) and faster: 960x576 177.4 -> 153.8 s, 1344x768
  681.8 -> 643.0 s, peak 77.5 -> 75.2 GB. Frame SHA-256 and phase timers now reach the journal (loggers under
  `vllm_omni`).
- 2026-10-27 06:31: 15 s 1344x768 20 steps renders on two Sparks in 3079.9 s (peak 85.6 GB, min free 25.8/27.2 GB)
  with the final configuration; the same clip had killed the head on the FP32 output path.
- 2026-10-27 07:55: final-configuration numbers for 10 s T2VA (1690.5 s, was 2119.0 s on the first boot; min free 26.1/29.6 GB)
  and 15 s FL2VA from a still (3166.3 s, min free 24.1/26.5 GB). `H3_TP2_MAX_OUTPUT_SECONDS` (off by default,
  experimental) lifts the pipeline's 15 s cap for out-of-spec tests.
- 2026-10-27 08:10: `H3_TP2_A2A_TIMING=1` (off by default) logs CUDA-event totals of the Ulysses all-to-alls per
  diffuse phase. `scripts/attn-bench.py`: cuDNN SDPA is the fastest exact attention kernel on the GB10 for the
  per-rank shapes (FlashAttention-2 +3 %, torch flash SDPA +4-7 %).
- 2026-10-27 08:40: review round 3 (overlap, timing, uint8 test): timing reports on its own diffuse hook and clears in
  `finally`; overlap records the inputs on the comm stream, joins outstanding communication on an exception, falls back
  for any mask or joint tensor and for non-default scatter/gather axes, and no longer claims identity "by
  construction"; phase timers synchronise the device at each boundary; uint8 test covers all batch elements, exact
  ties and infinities. `H3_TP2_A2A_OVERLAP=<groups>` (off by default) added; GPU measurement pending.
- 2026-10-27 10:14: 20 s 1344x768 (out of spec, `H3_TP2_MAX_OUTPUT_SECONDS=20`) renders in 6740.1 s, peak 92.7 GB,
  min free 20.6/22.4 GB, coherent output on the bench prompt. All-to-all timing: 2.9 % of the denoising phase
  (`results/20261027-20s-out-of-spec/`).
- 2026-10-27 10:47: all-to-all overlap measured: framemd5 identical to stock, 674.8 s vs 643.0 s (+4.9 %), rejected and
  left off; the record_stream version of the patch had tripped the head memory guard within 10 s (fixed before the
  measurement by allocating cross-stream buffers on the compute stream).
- 2026-10-27 11:06: expandable_segments measured: identical frames, 1 GB less peak, 664.0 s vs 643.0 s; rejected. The
  serving configuration is final: defaults (uint8 output, VAE on both GPUs, cache freed before decode) plus the
  experimental 20 s cap on the owner's decision.
- 2026-10-27 11:20: the package also loads on vLLM-Omni v0.30.0 (the first release with the MiniMax-H3 long-video
  continuation, PR #7838): executor placeholders for the newer shutdown path, worker thread budget via
  `set_multiprocessing_worker_envs`, keyword worker init, upstream's own GPU uint8 path and stage-cache release
  detected (our conversion and cache patches step aside), frame digest hooked on the shared encoder entry, duration
  cap routed through `long_video.max_output_seconds`. `render.sh` takes `TASK=ref2va` and `EXTRA_JSON`. Not yet run on
  the Sparks: the v0.30 venv and the Ref2VA partition are being installed.
- 2026-10-27 11:55: first boot on v0.30.0 across the two Sparks succeeded (6 min); the first request failed because the
  engine's cancellation signal is a head-local shared-memory byte. The remote rank now defers to rank 0.
