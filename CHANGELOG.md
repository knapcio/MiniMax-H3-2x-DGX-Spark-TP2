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
- 2026-10-27 12:13: v0.30.0 renders on the pair: 5 s 1344x768 645.0 s (643.0 on e1aa6eae), peak 62.2 GB (75.2), 53 GB free
  on both hosts, quality unchanged by eye (SSIM 0.906 against the e1aa6eae two-Spark clip, same class as two-Spark
  against one-Spark). Ref2VA partition and continuation next.
- 2026-10-27 12:55: review round 4 (port): broadcast-queue writes serialized against shutdown (bounded wait, then
  terminate), `render.sh` refuses an invalid `EXTRA_JSON`, `select-stack.sh` validates first and replaces the env file
  atomically, docs on the cancellation stand-in cost and the missing digest with `preencode_mp4`. Ref2VA on v0.30:
  5 s 1344x768 1248.6 s (1.94x FL2VA), peak 65.6 GB; the reference image is style/subject conditioning, not a first
  frame.
- 2026-10-27 13:42: first continuation clip on the pair: 30 s 960x576 (Ref2VA, 3 windows) in 3615 s, peak 71.1 GB,
  seams invisible, no scene cut; audio level drifts down across windows (one sample). 60 s render started.
- 2026-10-27 16:00: 60 s continuation clip: 7985.6 s, peak 69.4 GB, deterministic (windows 1-2 identical to the 30 s
  clip); plain mode drifts: motion -7x and audio -18 dB over six windows, unrequested shot changes. Remedies
  (per-window prompts, driving audio, longer overlap) under test.
- 2026-10-27 21:15: 30 s continuation at 1344x768: 10113.9 s, peak 78.8 GB, min free 26 GB; first seam a hard cut
  (one sample), audio -29 -> -41 dB. Experiment switches added for the night queue: per-window seed step, local-time
  windows (temporal offset), reference short edge.
- 2026-10-27 22:35: driving soundtrack (`lock_source`) on the 30 s clip: no audio fade, no motion decay, no shot changes,
  seams smooth; 4858 s (+34 %, partly sustained-load drift to be re-measured).
- 2026-10-27 23:40: per-window prompts: no cuts, smooth seams, motion kept, same time as the plain run; the generated
  audio still fades (8 dB over 30 s).
- 2026-10-28 00:50: local-time windows (temporal offset 0) change nothing: same motion decay and audio fade. The
  single-take prompt's +25 % was sustained-load drift (same prompt now 3664 s).
- 2026-10-28 01:55: a different seed per window changes nothing either (same decay). Mechanism narrowed to the
  latent-tail conditioning itself; remedies stay: driving soundtrack, per-window prompts.
- 2026-10-28 02:20: reference short edge 1024 instead of 2048: Ref2VA 5 s 1344x768 752.9 s vs 1248.6 s (-40 %), same
  quality by eye; recommended for continuation.
- 2026-10-28 04:25: 60 s with single-take prompt + driving soundtrack: no cuts, all five seams smooth, flat audio, no
  motion decay, 7313 s. First clean 60 s clip.
- 2026-10-28 06:50: repeats after 16 h: bit-identical, -0.7 % time (no load drift); the first Ref2VA request after a boot
  is 22 % slower than a warm one, so the reference-1024 gain is -27 %. Window 345: smooth seams, audio fade 1.7 dB
  instead of 7 dB over 30 s, +8 % time (one sample).
- 2026-10-28 07:45: recipe (ref 1024 + auto per-window prompts + driving soundtrack) at 30 s 960x576: 2725 s (-24 % vs
  plain), no cuts, smooth seams, flat audio, lively motion; peak 64 GB.
- 2026-10-28 10:30: the same recipe at 30 s 1344x768: 9805 s (-3 % vs plain, the reference rows are a smaller share at
  this size), seams smooth, motion kept, audio flat, but one unrequested cut inside window 2 (13.9 s); peak 73 GB, 39 GB
  free on the head (one sample). Detail decay found by the owner and measured (Laplacian variance of the grass per
  second): the 60 s clip with prompt + soundtrack loses most of its fine texture by the end (83 -> 6), the 1344x768
  recipe clip turns streaky in window 3; a third decay axis beside motion and audio, open.
- 2026-10-28 11:00: `scripts/clip-metrics.sh` (motion, detail, audio, seams, cuts per clip) and the metrics of every
  continuation clip so far in results/. Anchor experiment added (`H3_TP2_ANCHOR_FRAMES`, unit-tested): the first
  window's opening frames as a video reference for every later window; A/B against the plain 30 s clip queued.
- 2026-10-28 11:00: review round 7 (astra): anchor length validated against the first window before rendering,
  `H3_TP2_ANCHOR_SOURCE=tail` as the same-shaped control, test fixture rows derived from the shape, `clip-metrics.sh`
  fails on extraction errors and adds gradient-orientation coherence (the 60 s clip's coherence rises 0.10 -> 0.45 as its
  detail falls: streaking plus blur), results wording softened (recipe timing and the seam are combined-change,
  single-sample observations). Queued: plain 30 s with only the reference at 1024, and the tail-anchor control.
- 2026-10-28 11:50: `audio_flow_shift` 1 (review round 6 idea) with per-window prompts: the generated audio fades 11 dB
  over 30 s instead of 8 dB, rejected; the knob stays at 3.0. Light swings by a third across the windows, motion and
  detail hold (one sample).
- 2026-10-28 14:20: 60 s with window 345 + per-window prompts and generated audio: audio flat (-25.6 .. -22.4 dB), no blur,
  light steady, seams smooth; but fog at 7 s, a dissolve at 21 s, hard cuts at 32.0 s and 44.3 s (all inside windows)
  and an over-contrasty, streaky sky in the last 10 s (owner's viewing, confirmed frame by frame). `clip-metrics.sh`
  gained a one-second-change line that shows dissolves. Windows with a
  guide cost +41 % against the first window (no throttling; profile queued). 8776 s, peak 72.3 GB (one sample).
- 2026-10-28 14:45: reference short edge 768: 687 s against 753 s at 1024 and 1027 s at 2048 (5 s 1344x768), same look;
  diminishing, 1024 stays the recommendation.
- 2026-10-28 15:05: FP8 GEMM kernel choice closed: CUTLASS (vLLM) and cuBLASLt rowwise within 1 % on all four DiT shapes,
  54-61 TFLOPS, outputs not bit-identical. Queue reordered (speed profiles first, then anchor / light lock / ref-1024
  isolation; the 1344x768 window-192 run dropped).
- 2026-10-28 15:25: the per-row-scale epilogue is the FP8 GEMM bottleneck on the GB10: scalar-scale GEMM 189 TFLOPS vs
  59 for the fused per-row epilogue; scalar GEMM with fp32 output + separate exact scaling is 29 % faster already
  (3.8e-6 from CUTLASS, rounding order only). One-step profile: GEMM 47 %, attention 39 % (sm80 flash kernel), a2a 6 %.
- 2026-10-28 15:55: cuDNN 9.27 A/B (5 s 1344x768, ref 1024): 718.7 s vs 752.9 s (-4.5 %), frames differ (SSIM 0.848);
  candidate only, pinned 9.20 kept.
- 2026-10-28 16:50: two-pass FP8 GEMM render A/B (5 s 960x576): first request after boot 437.8 s vs 771.1 s stock
  (-43 %), warm repeat 451.6 s vs 592.3 s (-24 %, denoising -25 %); frames differ (SSIM 0.809, same quality by eye),
  deterministic, peak +1.8 GB. The owner's call on the non-identical output before it becomes the default.
- 2026-10-28 16:55: review round 8 (astra) on the two-pass GEMM: block-scaled and mismatched scale shapes now stay on
  the stock kernel (guards on scale counts, dtypes, 2-D operands), the Triton pass follows vLLM's epilogue order
  (scale_b first, bias as one fma), 64-bit row offsets, install-time probe instead of a per-call fallback, dispatch
  counters in the log; bench wording corrected (PyTorch's rowwise `_scaled_mm` is a CUTLASS kernel on sm121, the fc1
  fp32 temporary is 4.2 GB).
