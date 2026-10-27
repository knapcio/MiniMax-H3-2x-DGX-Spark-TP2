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
