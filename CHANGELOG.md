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
- 2026-10-28 17:30: NVlabs Sana `sol-engine` read (the owner's find): NVIDIA's single-GB10 H3 port confirms the FP8 GEMM
  ceiling (188 TFLOPS) and the attention ceiling (92.5 TFLOPS, which our profiled step already reaches); their
  lossless fusions are mostly in vLLM-Omni already; VAE tile batching (1.47x on decode, near-exact) is the one exact
  lever to port. Notes in results/20261028-nvlabs-sol-engine-gb10/.
- 2026-10-28 19:15: opening-frames anchor rejected: three hard cuts inside windows (the model cuts back to the
  reference), +25 % time; motion, audio and texture held. Window-1 step profile at 277 frames: attention 61 %, GEMM 30 %.
- 2026-10-28 21:30: `H3_TP2_FP8_FUSE_SWIGLU=1`: fused FP8 epilogue + SwiGLU for the MLP (fc1's fp32 product read once,
  no bf16 intermediate), bit-identical to two-pass + `silu_and_mul` in the unit test; render A/B queued.
- 2026-10-28 21:25: light-lock wording (60 s, window 345): the same two cuts on the same frames as without it; cuts
  are sampling events tied to the per-window seed, not prompt reactions. All 60 s clips with a driving soundtrack are
  cut-free, all with generated audio cut twice. Fused SwiGLU: review round 9 applied (memory lifetime, exception
  cleanup, NaN-safe rounding, width guard, signed-zero parity with vLLM's kernel).
- 2026-10-28 21:45: `H3_TP2_FP8_FUSE_SWIGLU=2`: the SwiGLU output is quantised per token inside the fused kernel and
  fed to fc2's scaled-mm kernel directly; a Triton per-token FP8 quant reproduces vLLM's compiled native quant bit for
  bit (fp8 values and scales, tested). Render A/B for both levels queued (queue7, queue8).
- 2026-10-28 22:15: reference 1024 alone on the plain 30 s clip: -24 % time, no cuts, smoother seams, the same audio
  fade and end-of-clip detail as with 2048: a speed lever only; the recipe's texture-holding came from the soundtrack
  and per-window prompts (or the sample), not the reference size.
- 2026-10-28 23:30: tail-anchor control: one cut on the same frame as the opening anchor (423), +10 % time: any extra
  video-reference block in a continuation window makes the model cut; both anchor variants closed.
- 2026-10-29 01:40: H3's own looped audio as the driving track (60 s): audio flat but two cuts and runaway motion
  (4 -> 21) with over-sharpened texture; the no-track fallback is not a recipe, and cut-freeness is not a property of
  `lock_source` itself (one music-track sample clean, this ambient-loop sample not).
- 2026-10-29 02:55: VAE `stack_tiling` as a plain flag: decode -34 % but peak +32 GB (all tiles in one batch) and
  a different clip (the flag also batches the reference-image encode, which changed the conditioning); 1344x768 run
  cancelled. To be redone as decode-only with a tile cap.
- 2026-10-29 04:50: two-pass FP8 GEMM at the serving cell (FL2VA 1344x768): 5 s warm 558 vs 688 s (-19 %), first
  request 537 vs 853 s (-37 %), 10 s 1488 vs 1741 s (-15 %); the 5 s MP4s are byte-identical to the stock render in
  both runs (SSIM 1.0), the 10 s frames differ. 20 s pending.
- 2026-10-29 05:45: the two-pass FP8 GEMM becomes the default (owner: a clean speedup with no quality loss);
  `H3_TP2_FP8_GEMM=stock` restores the CUTLASS kernel. Deployed to the Sparks after the night queue's A/B steps that
  assume a stock baseline.
- 2026-10-29 06:20: FL2VA 20 s 1344x768 with the two-pass GEMM: denoising 5454 s (e1aa6eae stock: ~6600 s), about
  -17 %; the GEMM share shrinks as attention grows with the clip length.
- 2026-10-29 06:40: the fused SwiGLU level-1 render A/B failed at once (`UnboundLocalError`: the patched forward
  referenced a deleted local on the level-1 path; the unit tests had not exercised the success path). Fixed, a
  success-path test for both levels added, redeployed; the level-1 rerun is queued after the VAE and profile steps.
- 2026-10-29 07:10: fused SwiGLU level 2 in the real model: two identical requests gave two different clips (one equal
  to the stock-kernel render) and only ~1 % less denoising; withdrawn as unsafe under the DiT's compiled regions.
  Level 1 rerun pending; the fusion stays opt-in and experimental.
- 2026-10-29 09:25: VAE batched tile decode, decode-only with a 4-tile cap, at 5 s 960x576: decode 24.5 -> 15.9 s
  (-35 %), peak +0.1 GB, the clip is the stock-decode clip (45.1 dB / SSIM 0.978, decoder arithmetic only). The
  first try's different scene came from the batched encode. Found at the same time: the two-pass digest 8e58f759
  predates round 8's epilogue change, so the night's "not identical to 8e58f759" verdicts (fused SwiGLU, batched
  decode) were against a stale baseline; a latents digest is now logged per render and a reproducibility chain
  (same boot x2, fresh boot, batched decode, fused SwiGLU level 1, stock) replaces the fused-SwiGLU rerun.
- 2026-10-29 09:35: batched tile decode at 1344x768: decode 42.8 -> 27.2 s (-36 %), peak -0.3 GB, but the sample
  differs from the baseline rendered in the previous boot (29.0 dB from frame 1; same code, request and seed), while
  at 960x576 the same wrapper reproduced the stock clip across boots. A 1344 cross-boot/within-boot chain with the
  latents digest (queue12c) decides whether the diffusion is boot-dependent at that cell.
- 2026-10-29 11:00: reproducibility chain at 5 s 960x576 (six renders, five boots): one latents digest and one
  framemd5 (85437ffa) for the two-pass default, the fused SwiGLU level 1 and the stock CUTLASS kernel, within and
  across boots; batched tile decode keeps the latents and changes only the decoder output (deterministic itself).
  Fused SwiGLU level 1 is bit-identical and denoises in 389 s against 442-447 s (first requests); a warm-pair chain
  measures the gain before it becomes a default. Stock 590 s vs two-pass 495-501 s wall at this cell.
- 2026-10-29 11:55: 1344x768 chain: two requests in one process gave different latents (F, G) and the next boot
  reproduced G's (H): the diffusion at this cell takes one of at least two values, not boot-bound; at 960x576 it is
  deterministic. Batched decode at 1344 is decode-only (H = G's latents, 45.2 dB, decode 42 -> 28 s). Attribution
  queued: prompt/media/latents digests per render, two renders per boot for ref 1024 / ref 2048 / stock CUTLASS, and a
  cuBLASLt determinism sweep on the DiT GEMM shapes (PyTorch sets no reduction-scheme mask for scaled_gemm).
- 2026-10-29 12:00: cuBLASLt scalar-scale `_scaled_mm` (the two-pass first pass) is bit-stable on the DiT shapes for
  M 36864-73728, same inputs and moved inputs; the 1344x768 variation is not this kernel on these shapes.
- 2026-10-29 13:30: fused SwiGLU level 1 warm pairs: fused 380-406 s (median 385) vs plain 399-447 s (median 416)
  denoising at 5 s 960x576, bit-identical in all six renders; boot-to-boot spread ~10 %. Becomes a default (code
  change after the evening's serving window). Evening plan (owner 12:50): both hosts on the FL2VA partition with the
  two-pass GEMM + fused level 1, batched decode off, a 10 s and a 5 s smoke render from a still; the 1344 attribution
  pairs and the NVlabs calibration are postponed.
- 2026-10-29 14:15: evening serving config live (FL2VA partition, two-pass GEMM + fused SwiGLU level 1, batched decode
  off): 10 s 1344x768 from a still in 1441 s (stock 1690 s, two-pass alone 1488 s), peak 69.6 GB; 5 s warm 505 s.
- 2026-10-29 15:05: owner: the wife needs 30-60 s clips tonight, the image as a reference is fine, her agent follows
  the README. Serving switched to long stack / Ref2VA with reference 1024 + two-pass + fused level 1; README section
  "Long clips from a client or an agent: the request" (curl, limits, times, prompt rules); `clients/h3-generate.py
  --long`. 30 s with that plain request: 2097 s, no cuts, flat audio, seams invisible (`results/20261029-long-clips-agent-shape/`).
- 2026-10-29 16:05: owner saw a light step at the first seam and a second woman after it in the plain 30 s clip ->
  the per-window prompt wording is now generated by the server for any continuation request without its own prompts
  (9a65815). Same request and seed: seam invisible, one woman, no cuts, flat audio, 2163 s.
- 2026-10-29 17:20: 60 s with the plain agent request and server-side window prompts: 4470 s, no cut, no seam step,
  one subject, flat audio; grainy texture and a near-static shot in the second half (the known long-continuation
  drift). README: 30 s by default, 60 s on request with the caveat. Serving stays on this config for the evening.
- 2026-10-29 17:30: fused SwiGLU level 1 becomes the code default (`H3_TP2_FP8_FUSE_SWIGLU=0` disables); the serving
  pair already runs it through its env since 13:40. Server-side window prompts documented as default.
- 2026-10-29 17:45: owner's review of the 60 s clip: colour changes and the subject's walk looping per window, invisible
  to the metrics. 60 s withdrawn from the README offer (up to 30 s); per-second colour means to be added to clip-metrics.
- 2026-10-29 17:55: clip-metrics gets chroma means per 5 s; on the 30/60 s clips the chroma is flat and the luma
  climbs 47 -> 83 (30 s) and 47 -> 88 (60 s): the "colour change" is a wash-out, a light drift per window.
- 2026-10-29 17:45: owner's verdict on the 30 s window-prompt clip: colour/texture step at 12 s and the subject doubled;
  per-frame luma confirms a +10 step across frame 277 (the strips had hidden it). Continuation declared not usable;
  serving restored to single-window FL2VA 4-15 s (two-pass + fused level 1). README leads with the verdict.
- 2026-10-29 19:30: first-seam loop at 16 s: the regenerated overlap does not match the guide tail in latent
  statistics (std +14 %, a channel shifted ~1 sigma); per-channel latent match removes the exposure step (+4.2 -> -0.6),
  a latent crossfade over the overlap plus a subject-free later-window prompt removes the inter-frame spike at the
  seam (8.1 -> 5.4 at neighbours 5.1-5.7). `H3_TP2_CONT_MATCH`, `H3_TP2_CONT_XFADE`, `H3_TP2_CONT_PROMPTS_REST`
  (`results/20261029-seam-loop/`). Owner's eye pending.
- 2026-10-29 21:15: the latent crossfade blinks (luma +-3 at the latent-frame period inside the ramp; owner saw it
  twice) and is withdrawn. New: the previous window's tail pinned as target rows through upstream's latent-edit path
  (`H3_TP2_CONT_EDIT=1|2`, video and audio): at 16 s the regenerated overlap equals the tail exactly, luma step +0.3,
  no spike in the inter-frame change, per-frame luma monotone across the seam.
- 2026-10-29 21:20: owner on X4 (pinned prefix): no jump noticed; the first hand-over that passed the eye.
- 2026-10-29 21:40: pinned prefix without the guide block (edit=2): seam as clean as with it (luma step +0.2, no
  spike) and the second window 18 % faster. 30 s / 60 s with both variants queued for the morning.
- 2026-10-29 22:35: 30 s with the pinned prefix: the first hand-over is a hard cut (scene 0.164, inter-frame 24.8 at
  neighbours 5-7), the second clean; the prefix makes a continuation exact but does not force one. Added a latent
  seam check per window with seed retries (`H3_TP2_CONT_RETRY=2`, ratio 3.0) and run the 60 s with it. Memory
  compacted on both Sparks before that boot (fragmented after a day of restarts).
- 2026-10-29 23:55: 60 s with the pinned prefix + seam check: no cut, five clean seams (luma steps +0.7..+1.6, no
  inter-frame spike), one subject, flat audio; seam ratios 0.90-1.06 (no retry). Remaining: the per-window drift
  (luma 47 -> 88, grain in the second half). Next: a latent look-lock of each window to the first window's statistics.
- 2026-10-30 01:00: 1344x768 stage attribution: prompt embeddings and reference encoding identical within a boot,
  latents sometimes not (K1 != K2 at ref 2048, I1 == I2 at ref 1024): the per-request variation is inside the
  denoising loop; stock pair pending.
- 2026-10-30 01:35: stock CUTLASS pair at 1344x768: identical latents and frames to the two-pass GEMM (and 22-40 %
  slower). The rare per-request variation at 1344 is in the shared denoising loop, not in our kernels.
- 2026-10-30 02:35: NVlabs gb10_kernel calibration on one Spark (lossless tier, T2VA): 15.2 s per DiT eval at 960x576
  and 41.6 s at 1344x768 (requests 333 / 863 s, decode 37 / 69 s, peak 74-75 GB). Pair comparison renders queued.
- 2026-10-30 03:35: look-lock at 30 s: light per second stays 44-58 (46 -> 69 without it), no cut, but the uniform
  per-window correction steps -8.5 luma at the second seam; the correction is now ramped across the window's free
  frames (a9cfe12), rendered again at 30 s and 60 s.
- 2026-10-30 04:25: ramped look-lock at 30 s: both seams +0.3 luma, no cut, exposure pulled back by each window's end
  (max 65 instead of 69/58); the shot calms in window 3. Ramp fraction parameter added for a faster hold.
- 2026-10-30 05:40: 60 s with the pinned prefix (no guide), seam check and ramped look-lock: five continuous seams
  (+0.8..+1.2), no cut, light 46-66 (was 47-88), detail max 462 (was 1084), 3857 s. The generated audio climbs 23 dB
  in the last two windows: audio look-lock added (`H3_TP2_CONT_LOCK_AUDIO=1`), in the next 60 s together with the
  0.25 ramp and the default later-window wording.
- 2026-10-30 06:00: the pair at the calibration cells (T2VA): 8.65-9.0 s/step at 960x576 and 23.2 s/step at 1344x768
  on two GPUs = 17.3-18 / 46.4 GPU-s per step against NVlabs' 15.2 / 41.6 on one: their lossless fusions are worth
  10-14 % per GPU over our current stack; TP2 scaling is near ideal.
- 2026-10-30 07:20: 60 s with the pinned prefix (no guide), seam check, video look-lock (ramp 0.25), audio look-lock
  and the default later-window wording: five seams +0.3..+0.8 with no spike, no cut, light 43-62, detail <= 442,
  motion 5-9 throughout, audio flat -48..-52 dB, 4023 s. The continuation defaults in code are now these
  (`H3_TP2_CONT_EDIT=2`, `RETRY=2`, `LOCK=first`, `LOCK_RAMP=0.25`, `LOCK_AUDIO=1`); `H3_TP2_CONT_EDIT=0` etc. restore
  upstream's behaviour.
- 2026-10-30 07:40: owner on the 60 s clip: all transitions smooth; the camera re-stages its travelling move every
  window with the default later-window wording (the woman revealed 5-6 times). A drift wording for the later windows
  is under test; the subject-free wording avoided it but let the motion die.
- 2026-10-30 08:40: seam-check calibration: a hard cut right after the prefix measured a latent ratio of 1.43
  (continued windows 0.81-1.16), threshold now 1.3. Continuation enabled on FL2VA requests (the still as the first
  frame, static camera: the wife's use) through the pinned prefix; Koninck painting tests queued.
- 2026-10-30 10:10: a different seed per window (`H3_TP2_WINDOW_SEED_STEP=1`) at 60 s: five clean seams, the seam
  check re-sampled one window (ratio 1.80 -> 0.85), light 41-65, audio flat. Owner's report on the drift clip: window 2
  replayed the first 12 s (same seed = same noise per window) and the subject re-entered; seed step is the answer
  under test by eye.
- 2026-10-30 10:20: owner on the seed-step 60 s: no replay, but whatever leaves the frame comes back different (no
  object permanence across windows), some pixelation, a look change in the last window. A moving-camera 60 s is
  closed; the locked-camera use is what the FL2VA painting tests check. Mean-only look-lock option added.
- 2026-10-30 11:00: the painting use (FL2VA, locked camera) with the continuation: 16 s Koninck, seam luma step 0.0,
  no cut, seam ratio 1.00, the clouds move and the picture stays the picture (`results/20261030-painting-fl2va-continuation/`).
- 2026-10-30 12:20: Koninck 30 s on FL2VA: two invisible hand-overs (-0.3 luma, no spike), no cut, light 115-129, the
  painting stays the painting; 5016 s at 1344x768.
- 2026-10-30 14:35: 5 s pinned history on the painting (window 345 / overlap 124): seams as clean as with 0.9 s,
  1.57x the time (7865 / 5016 s; correction: previously reported as 2.6x); default overlap stays 22. Memory compacted
  again before the 60 s painting render.
- 2026-10-30 16:25: owner's eye on the painting clips: 16 s and 30 s good; the 5 s history clip flickers a little at
  15-22 s. Whole-window look-lock after a first window that brightened 115 -> 135 is a possible contributor, not a
  verified cause. New option `H3_TP2_CONT_LOCK_REF=tail` (reference = the last quarter of window 1 for both video and
  audio; default `window` unchanged until measured).
- 2026-10-30 16:34: per-frame check found 13 periodic luma drops in the 5 s-history clip, frames 357..561, every
  17 frames, median -6.10 YAVG; none in the default 16/30 s clips at the same threshold. This matches the VAE's
  17-frame chunk geometry but does not establish causation. `scripts/luma_flicker.py` detects this pattern.
- 2026-10-30 16:48: experimental brightness-only repair of the saved history5s clip (separate file; no runtime
  change): 42/736 decoded frames altered, chroma and copied audio identical, no newly clipped luma at 255, original
  dip steps -2.22..-6.21 -> -0.02..-0.22 YAVG; no remaining 17-frame run >=2 YAVG. Playback review pending.
- 2026-10-30 16:55: opt-in `H3_TP2_VAE_BLEND_TRACE=1` logs sampled old/new/blended luma at temporal VAE overlaps to
  investigate the 17-frame flicker. CPU-only checks pass; distributed GPU decode and measurement overhead unverified.
- 2026-10-30 16:56: 25 saved H3 clips screened with the >=2 YAVG/three-period detector: the history5s clip and the
  last 5 s of `d60_seedstep_per_window` contain repeated 17-frame drops; no such runs in the other 23. Other visual
  defects, local flicker and shorter/lower-amplitude sequences are outside this screen.
- 2026-10-30 17:08: owner on the experimental deflickered history5s copy: improved, but flicker remains. Owner also
  reports the cloud shadows on the ground look too strong or misaligned with the clouds; exact cause unknown. Next
  painting A/B removes the explicit moving-shadow request while keeping the rest of the 16 s request matched.
- 2026-10-30 17:45: corrected continuation duration precedence to match the running encoder, including explicit nulls;
  CPU plans for 4–300 s match the upstream planner. 300 s remains unverified; guarded streaming, memory and speed tests
  are queued. A local spatial deflicker experiment looks better to the owner, who also reports figures move too fast;
  a matched slower-figures prompt is queued. See `results/20261030-300s-readiness/`.
- 2026-10-30 18:27: exact-duration copies on the 24 fps frame grid are now available via `scripts/exact_duration.py`.
  It verifies every retained decoded video frame and copied AAC packet/timestamp; audio can finish one packet early.
  Separate 10, 15, 16, 30 and 60 s sample copies verified, originals preserved (`results/20261030-exact-duration/`).
- 2026-11-01: fixed the continuation audio look-lock overriding `audio_mode=lock_source` in later windows: the ambient
  lock now skips requests with locked source rows. A CPU regression test verifies source audio is untouched by that
  wrapper while native audio and video locking retain their behavior. Not deployed or validated on a real GPU render.
- 2026-11-01: fixed `render.sh` truncating text and JSON fields at semicolons through curl's multipart `-F` parsing.
  Literal fields now use `--form-string`; a real local HTTP multipart receiver verifies punctuation, Unicode,
  continuation JSON, image bytes and a large file-backed audio field. Live H3 verification is pending.
- 2026-11-02: `render.sh` disables curl's automatic config files with `-q` as its first argument. A real two-server
  loopback regression demonstrates that a `.curlrc` could previously add a second multipart POST. Proxy environment
  variables and explicit endpoint settings still apply; existing frozen or deployed copies remain unchanged.
- 2026-11-02: added a CPU-only exact-length segment planner for separate single-window two-keyframe requests. It
  reports whether generated last frames can all be retained, an optional one-frame boundary drop, and fractional
  source-audio cuts. This is geometry, not a qualified assembler or a model-quality result.
- 2026-11-02 09:50: the periodic 17-frame luma flicker of long clips traced to the look-lock, not to the decoder's
  chunking: offline decodes of the same saved latents give a -0.66 Y step at every 17-frame block boundary raw and
  -2.52 with the serving lock (served file -2.0; the 5 s-history clip -6). The VAE's block-start token (every 5th) has
  its own statistics and one per-channel affine fitted on the mixture moves it off its class. Fix: token-class-aware
  lock (`H3_TP2_CONT_LOCK_CLASSES=1`, now the default; `0` = old behaviour): serving control at 960x544 keeps the
  brightness correction (light per second identical) and returns the boundaries to the raw decoder's (-0.53 / -0.74
  vs -1.94 / -2.09), no drop >= 1 YAVG. Closed negatives on the way: left-context chunk decode
  (`H3_TP2_VAE_LEFT_CONTEXT`, the decoder's RoPE is normalised by clip length: -7 / -52 Y) and a full-length decode
  (not a reference, the decoder is trained on 17-frame clips). New diagnostics: `H3_TP2_SAVE_LATENTS`,
  `scripts/vae_decode_offline.py`, `scripts/lock_decode_offline.py` (`results/20261102-vae-chunk-flicker/`).
- 2026-11-02 12:00: look-lock reference = the tail (last quarter) of window 1 by default (`H3_TP2_CONT_LOCK_REF=tail`;
  `window` = the old whole-window reference). The whole window included the clip's dark start, so every later window
  was pulled ~9 Y below its hand-over level in a staircase of 17-frame steps right after the seam (visible at 1344 as
  the remaining period-17 drops). 960x544 control: windows stay at the hand-over level, block boundaries -0.22 / -0.40
  (raw decoder -0.66), no drop >= 1 YAVG, seams +0.1 / -0.2.
- 2026-11-02 12:15: look-lock ramp default 1.0 (was 0.25): at 1344 the model brightens a window by up to +20 Y and the
  decoder renders a latent brightness correction as one level per 17-frame block, so a correction completed within a
  quarter of the window was a staircase of -1.5..-2.3 Y steps right after each seam; spread over the whole window the
  steps are ~0.9 Y (raw decoder -0.6). Replay on saved 1344 latents in `results/20261102-vae-chunk-flicker/`.
- 2026-11-02 13:30: speed plan item 1 measured: `H3_TP2_EMPTY_CACHE_BEFORE_DECODE` is inert on v0.30 (upstream releases
  the stage cache itself); the same 5 s 1344 calibration ran 22.9 s/step on one boot and 21.2-21.4 on the next
  (identical latents digest), no per-request degradation: boot-to-boot variance of 7 % is real and a boot speed gate
  is the lever (`results/20261102-speed-plan/`).
- 2026-11-02 15:10: 1344x768 control with the three look-lock defaults (classes, tail reference, ramp 1.0): no
  period-17 drop >= 1 YAVG, block boundaries -0.39 / -0.62 (min -0.88) against -0.69 / -1.85 with the quarter-window
  ramp; seams -0.0 / +0.1. The flicker work is closed; the remaining brightness movement is the prompt's.
- 2026-11-02 18:25: 60 s Koninck at 1344x768 with the full 11-02 defaults on a speed-gated, compacted boot: 9486 s
  (2 h 38 min; 13445 s on 10-30), five invisible hand-overs (-0.0..+0.2 luma), no cut, block boundaries median -0.50
  (one run at -1.0 Y, below the eye; 10-30: -1.62 / -3.38). `results/20261030-painting-fl2va-continuation/`.
- 2026-11-02 18:35: owner approved the 60 s 1344 clip ("wyglada spoko"); the README's painting request now carries the
  no-shadow human-pace prompt, the border check and the 11-02 lock defaults.
- 2026-11-02 19:20: speed plan items 3 and 5 measured on the GPU (queue50): the no-copy Ulysses exchange is numerically
  wrong (rel-l2 0.20 vs stock) and no faster -> closed (`H3_TP2_ULYSSES_NOCOPY=force` for debugging only); comm stream
  priority -1 + NCCL_MAX_CTAS=4 with overlap 2: bit-equal, 21.27 vs 21.32 s/step -> no effect. Two stock boots at 1344
  gave bit-equal latents.
- 2026-11-02 21:05: batched tile decode on by default (`H3_TP2_VAE_STACK_TILES=2`, cap 4): -38/-39 % decode time on
  identical latents at 960x544 and 1344x768 with max 1 gray level on 0.2-0.3 % of pixels (73.9 / 74.7 dB), two-rank
  serving renders at 960 and 1344 unchanged in every metric (`results/20261102-speed-plan/`).
- 2026-11-02 22:20: speed plan item 4 measured and closed: cuBLASLt outer-vector (per-row/col) scaling is unsupported on
  sm121, and a Triton FP8 GEMM with the epilogue in registers runs at 86-90 TFLOPS against cuBLASLt's 169-177, so the
  fused path is +3 s/step slower at both cells; the two-pass GEMM stays (`results/20261102-speed-plan/`). Items 5 and 7
  landed opt-in earlier tonight (nocopy exchange: wrong on the GPU, closed; step caches: 0.03-0.1 %, off).
- 2026-11-02 23:50: end state: 30 s Koninck 1344x768 with every default of the day (lock classes + tail + ramp 1.0,
  batched tile decode, speed-gated compacted boot) in 4648 s against 4964 s in the afternoon, same quality profile.
- 2026-11-03 01:35: consumer-fused FP8 scale pass on by default (`H3_TP2_FP8_CONSUMER_FUSE=1`): the qkv/out/fc2 scale
  pass is applied inside the consuming kernels; GPU probe bit-identical on every site with no spills and a lower peak,
  -1.04 s/step at 1344 (-0.52 at 960); serving: 5 s 1344 digest 4d6d69e4 unchanged at 21.5-22.1 s/step.
- 2026-11-13 06:45: results: queue152 repeats the two-keyframe close-ups at seed 6002: twelve of twelve over two seeds open
  and close in the framing, and what each shot leaves for in the middle is the shot's, not the seed's (shot 2 holds, shot 6
  moves to the food and back, shot 9 cuts to the same two-shot and back, at both seeds). Section 8 and the recipe say
  "twelve of twelve on two seeds"; another seed does not remove a cutaway.
- 2026-11-13 04:42: results: queue149b, the hand-over noise probe (22.2 s, three windows, with and without the lever, no
  retries): the lever acts (different `framemd5`, window-3 seam ratio 1.17 -> 1.21); rank 0 logs no per-window line for it,
  as it logs none for the default audio pin either. Section 8 of `results/20261109-long-chain-identity/README.md`. The first
  run (queue149) sent the 118 s extra JSON's prompt list to a three-window request and failed in 0.4 s, which is what the
  22:29 client-side count check now catches.
- 2026-11-13 03:15: results: queue151, the close-up as FL2VA with both keyframes (the still first, the same or a later
  frame of the shot last), section 8 of `results/20261110-shots-and-close-ups/README.md`: six of six open and close in the
  framing (38-43 dB at both ends), two hold it throughout, four leave it for a 2-2.5 s beat on a pause of the line (a tilt
  to the props in shot 6, a cutaway to a two-shot in shot 9) and come back; +6 % time. Way (d) is now the recipe's way to
  render close-ups (section 4); the README's long-clip table says so.
- 2026-11-13 01:16: results: queue150, the voice anchor on a second seed (3801), closes the lever in section 8 of
  `results/20261109-long-chain-identity/README.md` (`speakers_q150_voice.txt`): the anchor's worst speaker window is above
  the recipe's in all four comparisons over two seeds (0.943 / 0.865 vs 0.81 / 0.838; 0.947 / 0.935 vs 0.857 / 0.869), the
  recipe's dips are single short lines ("Czynsz może poczekać" at both seeds); naturalness, lines, faces and time unchanged.
  Opt-in `H3_TP2_VOICE_ANCHOR=5` for dialogue chains, not a default.
- 2026-11-12 22:51: `scripts/speaker_drift.py` leaves segments shorter than 1.5 s out of the references and the window means
  and counts them per window in the detail line (`short`): a one-word segment's x-vector matches nobody (a 1.0 s "Dobrze."
  at 0.30 / 0.54 pulled the man's window 5 to 0.81 in the seed-3801 recipe chain; without it 0.95). The queue145 numbers in
  `results/20261109-long-chain-identity/` section 8 and `speakers_q145_voice.txt` are rescored (the woman's worst windows
  0.838 / 0.865 / 0.916, the man's 0.81 / 0.943 / 0.917; the verdict stands). Test in `tests/test_speaker_drift_summary.py`.
- 2026-11-12 22:42: the speech guard stops re-sampling a window when the same script words come out wrong at two seeds
  running (`Guard.repeated`, a fourth broadcast value from the scorer, the retry loop keeps the best attempt and logs
  `the same words wrong as at the seed before`). Another seed does not fix a word the model reads its own way: in queue150's
  recipe chain (seed 3801) window 7 came out `klientów -> klienw` at three seeds and window 10 `długo -> dłongo` at two of
  three, 683 s per attempt; the chain spent 15013 s on 11 windows of 683 s (21 attempts, limit 0.01). Tests in
  `tests/test_speech_guard.py` and `tests/test_cont_speech_check.py`.
- 2026-11-12 22:29: `scripts/render_text_fields.py` rejects an explicit `continuation_prompts` list whose length is not the
  request's window count before anything is uploaded (`continuation_prompts has 6 strings but a 22.2 s request is 3 windows`),
  using the plugin's own window arithmetic (`h3_tp2/cont_prompts.py`, loaded by path). The server rejects the same request
  only after the boot and the upload, which is how queue149 lost an hour (two probes, 0.4 s each, HTTP 500). An explicit
  null duration is not counted. Test in `tests/test_render_text_files.py`.
- 2026-11-12 15:45: `scripts/framing_trace.py` samples exact frames (frame 0, then every `round(every*fps)` frames)
  instead of ffmpeg's `fps` filter, which hands out the last frame of each interval and never frame 0, so a cut on frame 1
  was invisible; and a step is a cut when the match fails or the samples barely correlate (under 0.5) with a match zoom
  outside 0.8-1.25, calibrated on the nine cuts and two moves of queues 147/148 (the old rule read a cut with 84 inliers
  as a pull-back x0.76). `corr` column in the JSON; a cut no longer counts as zoom. Section 6 of the shots README
  corrected: shot 2 seed 6002 is a 0.67 s cut, not a pull-back. Tests: `tests/test_framing_trace.py` (12).
- 2026-11-12 15:40: results: queue148, the pinned first frame (`scripts/first_frame_pin.py` + the character's card as the
  only reference) as section 7 of `results/20261110-shots-and-close-ups/README.md`. Frame 0 is the still (34-35.5 dB);
  three of six shots cut away at frame 1 or 2, two hold the first clause and cut at 2.2-2.6 s, one holds the whole shot:
  first-frame conditioning of either kind does not override the model's shot grammar. The card keeps the right character
  in every wide shot the model cuts to. The guard path is bit-identical to the direct one with a source video and mask
  (`pinloop` = `pin` framemd5). Not a default.
- 2026-11-12 13:24: `scripts/voice-metrics.sh` prints the speaker detail line (segments per window, the lowest one,
  relabelled segments, the window-1 same-voice floor) above the speaker summary.
- 2026-11-12 13:20: the pitch tracker of `scripts/speaker_drift.py` uses the unbiased autocorrelation and takes the shortest
  lag whose peak reaches 90 % of the highest (walked up to its peak). The biased estimate let the half-period peak win when
  the odd harmonics were weak: 3 of 105 lines of the 118 s clips came out an octave high against pyin (frame 2048, the
  long frame; the short frame doubles too), now 1. Tests: pure tones, a strong second harmonic, an even-only tone.
- 2026-11-12 13:12: `scripts/speaker_drift.py` assigns every segment to the nearer window-1 voice by its x-vector; the
  pitch split stays only to name the two window-1 references. On the 118 s clips the pitch tracker read 167-206 Hz on three
  of the man's lines and handed them to the woman, which made her similarity drop to 0.46-0.61 on those lines; by voice they
  score 0.87-0.95 against him. The tool also prints a detail line (segments per window, the lowest one, relabelled
  segments, the same-voice floor inside window 1, here 0.94). WITHDRAWN with it: the 08:22 reading that the voice anchor
  held the woman's voice at the end of the clip (0.865 against 0.71 and 0.735); assigned by voice the three arms are
  0.865 / 0.874 / 0.916, within one line of each other (`results/20261109-long-chain-identity/` section 8, arm 2). Tests added.
- 2026-11-12 13:01: `scripts/bench-matrix.sh` passes the keyframe position `both` to `render.sh` when `H3_END_IMAGE_FILE`
  is set (it hard-coded `first`, so a two-keyframe FL2VA request could not go through the queue tooling); test added.
  The shots recipe (`results/20261110-shots-and-close-ups/` section 4) lists way (c).
- 2026-11-12 12:56: way (c) to a close-up, the shot rendered as FL2VA from the model's own close-up still (queue147,
  `results/20261110-shots-and-close-ups/` section 6). The opening is solved by construction (frame 0 is the still, 39-43 dB,
  no invented room, no head to trim), but the model leaves the framing anyway: five of six shots pull back or cut to a
  wider shot within 0.75-3.5 s and invent the other people, the mirror image of section 5. The move sits on a clause
  boundary of the line in all four shots that cut during it. Two pull-backs have no scene cut and show only in
  `scripts/framing_trace.py`'s zoom column (its first use on rendered shots). 892-899 s per 8 s shot with one still
  against 1211 s by Ref2VA with four references. Next: the pinned first frame plus the card (queue148, running).
- 2026-11-12 10:50: a word the model slips at every seed (queue146, `results/20261106-voice/` section 18): `chrupnięcie`
  came out as `chrupiącie`-like at all four seeds with the plain spelling (0 of 4), and right in 2 of 4 with `chrupnjęcie`
  in the prompt and the real word in the respelling table; the split spelling `chrup nięcie` also hit 2 of 4 but broke
  `wjeżdżamy` next to it in 3 of 4 takes (clean in all four controls). The picture follows the seed, not the spelling.
  Rule: respell inside the word, map it back with `H3_TP2_SPEECH_RESPELL`, let the guard re-sample; never split a word.
- 2026-11-12 08:33: `scripts/face_drift.py` reports the head's turn with the similarity: `turn` per sample (the nose's offset
  from the midpoint of the eyes along the eye line over the eye distance, from MTCNN's five landmarks; 0 frontal, about
  0.5 three-quarter, 1 and beyond a profile; a rolled head is not a turn), the mean |turn| per window, and `matched`, the
  mean similarity over the samples whose |turn| is within 0.2 of the reference face's own. On the three queue145 arms the
  reference two-shot has both heads at about 0.6; the woman runs 0.7-0.9 (profile) in the two-shot windows and 0.4-0.6 in
  the noise arm's close-ups, which is most of her 0.69-0.88 spread. Pose-matched, the woman is 0.77 (recipe) / 0.76
  (anchor) / 0.85 (noise close-ups) and her window-10 dip to 0.69 stays at matched pose (two samples), so the end-of-clip
  sag is partly real and partly pose. The man's negative minima in the noise arm's close-ups are the position match
  landing on no face, not drift. Tests `tests/test_face_drift_windows.py` (turn, roll, matched).
- 2026-11-12 08:27: `scripts/queue-lib.sh` gained `lever_lines "<since>"`: how often the levers that run inside the
  denoising loop (hand-over noise, audio pin release, voice anchor) logged, per rank, from the head journal and the rank 1
  worker logs newer than the start time; the head journal alone misses the loop wrappers' per-attempt lines (queue145's
  `lever:` field came out empty for the noise arm). queue150 (the voice anchor on a second seed, armed after queue149)
  uses it. Also in section 8: the woman's crops from windows 4-10 of the three arms show the same face everywhere; the
  0.68-0.76 spread of the face score in the two-shot is head angle, so a 0.05 gap between arms there is pose, not likeness.
- 2026-11-12 08:22: the recipe plus the voice anchor (queue145 arm 2, `results/20261109-long-chain-identity/` section 8):
  `H3_TP2_VOICE_ANCHOR=5` on the 118 s recipe chain keeps the recipe's composition and moves one number, the speaker
  similarity to window 1 (WITHDRAWN 13:12: the dips were the man's lines mislabelled by pitch): the woman's worst window 0.865 against 0.71 and 0.735 in the two arms without it (window 11
  0.865 against 0.79 in both), the man's 0.943 against 0.81 and 0.933; voice quality, words and faces within noise, pace
  +1.5 %. One seed; a candidate for the end-of-clip speaker drift, to be confirmed on a second seed and scene. Correction
  to the 04:22 entry: over the three chains 12 windows reached a third attempt and one passed (arm 2's window 3), so
  `H3_TP2_CONT_RETRY=1` is a speed setting (15-20 %) that keeps about one window in 33 with a 1-3 % character slip, not a
  free one; the default stays. Wording: three entries below no longer name the tool that proposed them.
- 2026-11-12 04:22: the recipe plus the noised hand-over (queue145 arm 1, `results/20261109-long-chain-identity/` section 8):
  `H3_TP2_CONT_EDIT_NOISE=0.35` on the 118 s recipe chain re-frames the clip (window 6 a cutaway of hands and letter,
  windows 7-10 a close-up, ten cut-check warnings against two, five scene cuts against one); the close-up windows carry
  the best identity of any 118 s window (the woman 0.87-0.89) and the voice is a little better (UTMOS 3.34 against 3.15),
  the pace the same. Not a default, not for a fixed two-shot; a lever for shots meant to cut. Open: the head's journal
  has no per-window hand-over noise line while rank 1 logs one for every attempt (to be probed one window on both logs).
  Also: at guard 0.01 the third attempt of a window never passed and never became the kept one (9 of 9 over two
  chains), so `H3_TP2_CONT_RETRY=1` is lossless there and 15-20 % faster.
- 2026-11-12 00:07: `scripts/framing_trace.py`: the framing of a shot over time (largest face height over frame height per 0.25 s, MTCNN; the camera's zoom between samples by ORB + RANSAC, a failed match = a cut), where the final framing starts (first sample at or after the last cut within 20 % of the final face scale with under 10 % zoom left), and the shot size the model ended on (wide under 0.22, medium, close-up from 0.35). On the 21 shots of v3-v5 it agrees with the hand labels in 19 (cuts within one sample at 1.00/2.75/1.25 s against 1.04/2.63/1.29; v3 push-ins show as zoom x2.0-2.5; card-only and wide shots as static); off on the v3 shot 6 push-in (0.5 s) and the v5 shot 9 extreme close-up that keeps moving after its cut. Tests `tests/test_framing_trace.py`.
- 2026-11-12 00:01: a pinned first frame for one shot (EXPERIMENT, not yet run on the GPU): `scripts/first_frame_pin.py still.png --frames 192 --width 1344 --height 768 --out-dir pin` writes upstream's latent-mask editing inputs (a one-frame source clip the server pads to the request, and the token-grid mask with latent frame 0 at 0 = keep), and `render.sh` sends them as `source_video` and `video_noise_mask` when `H3_SOURCE_VIDEO` and `H3_VIDEO_NOISE_MASK` are set (recorded under `latent_edit` in the request metadata). The idea (from a read of the reference path): the still of the real room at the shot's framing becomes frame 0, the character's card alone the reference, so a close-up opens static in the real room. Tests `tests/test_first_frame_pin.py`.
- 2026-11-11 23:58: `guard_single` logs a one-window request that is in range but off the 17n+5 grid (with the nearest grid sizes) instead of leaving it unjudged in silence (the "guard set but never ran" shape of the v2 shots). Not yet on the nodes.
- 2026-11-11 23:53: `shot_cuts.py --apply` and `cut_shots.py` cut with the trim filters instead of an output `-ss`: ffmpeg applies an output `-ss` after the filters, so the 40 ms audio fade-in sat in the discarded head and trimmed shots joined unfaded (found by a read of the scripts). Every shot now fades in at its new start and out at its end; the test decodes the PCM and checks the first 10 ms (`tests/test_shot_cuts.py`).
- 2026-11-11 23:37: `scripts/cut_shots.py`: rendered shots cut into one film (the measured head trim per shot or `--trim N=seconds` by hand, one encoding, a 40 ms audio fade-in at every cut, stream-copy concat). Rebuilds the comedy clip's v5 from its nine shots to the same 64.96 s. Tests `tests/test_cut_shots.py`. Both scripts are listed in `docs/DIALOGUE_CLIPS.md` section 5.
- 2026-11-11 23:38: `scripts/shot_cuts.py`: the model's own scene cuts and silences in a one-shot clip and the head trim they imply (the first cut under 2 s whose leading audio is silence; `--apply` writes the trimmed clip with a 40 ms audio fade-in). On the six queue144 close-ups it reproduces the hand measurements (1.042 s, 0, 0, 0, 0, 1.292 s). Tests `tests/test_shot_cuts.py` (the rule on the CPU, the measurement on a synthetic ffmpeg clip).
- 2026-11-11 23:30: way (a) close-ups rendered as a set (queue144, `results/20261110-shots-and-close-ups/` section 5): the establishing head of a scene-frame close-up is a hard cut from the reference's composition (1.0-2.6 s) in three of six shots, not a push-in; it trims off when the audio before it is silent. The prompt's silent beat lands where the model wants (one shot turned it into a 3.5 s chewing close-up and moved the line to the wide shot after the cut); two of six shots stayed wide or medium. Room, props and faces were right in all six. The edit trims the head at the first scene cut under 2 s with leading silence; `ksiazulo-v5-realroom-closeups.mp4` is the result.
- 2026-11-11 16:42: the model's own close-up frame as a close-up's scene reference (queue143, `results/20261110-shots-and-close-ups/`): the room and the props now match the scene (the lembas is the scene's loaf), but the shot still opens with an establishing move (a wide view or a table insert) before the close-up. Rule of thumb recorded: a reference that shows a room gives the room and an establishing head; the face card alone gives a static shot in an invented room. The shots recipe offers both; for the real room, render 1.5 s longer and trim the head.
- 2026-11-11 15:34: the voice anchor's first run (queue142, `results/20261106-voice/` section 17): on the 64.6 s chain the speaker similarity moves within noise (the voices do not drift at 60 s), UTMOS is neutral, no line is echoed, two of three slipped windows came out clean (0.009 against 0.025 wrong characters, one seed), +2 % time. Harmless, unproven; the 118 s chain is where it would show.
- 2026-11-11 14:04: framing crops and object crops as references (queue141, `results/20261110-shots-and-close-ups/` section 3): a crop of the scene frame in the shot's framing opens the shot in that framing, static and sharp, with the scene's style but not its room (door and fireplace move), and an object crop does not make the prop look like it (the lembas came out as toast with and without its crop). References anchor subjects and style; rooms and props come from the prompt unless the whole scene frame is referenced, which costs the opening composition. Next (queue143): the model's own late frame of a close-up as that character's close-up scene reference.
- 2026-11-11 13:12: `H3_TP2_VOICE_ANCHOR=<seconds>` (experiment, off by default, not yet run on the GPU; `h3_tp2/voice_anchor.py`, 6 CPU tests): the opening seconds of window 1's audio latent as a Ref2VA audio reference for every later window, against the voice drift of long chains. `cont_match.overlap_audio_t` now reads the pin mask before the condition lengths, so an audio reference before the hand-over guide cannot be mistaken for it. GPU test queued (queue142, 64.6 s doll chain, seed 3501, against the plain render).
- 2026-11-11 12:26: the two-minute recipe rendered in one chain (queue140, `results/20261109-long-chain-identity/` section 7): crops + `toffcap` 2125 + speech guard 0.01 on the 118 s doll chain, seed 3701, under the compaction setting: 0.010 wrong characters of the script (plain 0.064), WER 0.050, the woman's face 0.69 at the end (plain 0.60, crops 0.66, cap 0.61), the man 0.84; 21 samplings at a flat 682-686 s each (no pace drift in 4.2 h), 15 042 s, 2.3x the plain render. The four words still wrong are prompt spellings. README's identity row carries the combined numbers.
- 2026-11-11 07:34: the night's outage of spark-05 and spark-06 was their Wi-Fi (both lost the same access point at 03:45:46-48 and never reconnected, `journalctl -b -1`), not a hang: the hosts rendered all night under `vm.compaction_proactiveness=0`, queue126 (two 64.6 s chains) at 547-551 s per window, the fastest chains recorded on the pair, and queue139 after it. The withdrawal of the setting is reverted: the setting stays at 0 on both Sparks (README, `results/20261111-slow-state-compaction/`). queue126's voice controls: a window re-rolled without selection changed the windows after it by -0.12 and +0.00 paired UTMOS on the two seeds, so the guarded clips' -0.1 is the chain's seed spread (`results/20261106-voice/` section 15). The comedy clip v4 (queue139): the six close-ups with the character's crop as the only reference open in their framing and hold it, five of six clean at the first attempt; cut together with v3's wide shots on the Mac (`results/20261110-shots-and-close-ups/`).
- 2026-11-11 03:53: WITHDRAWN until explained: `vm.compaction_proactiveness=0`. About 90 minutes after it was persisted on both Sparks, spark-05 and spark-06 stopped answering together (no LAN, no fabric, no ARP) during queue126, while spark-01..04 stayed up. Suspected: proactive compaction keeps the high-order free blocks the NVIDIA driver needs; without it a GB10 host can livelock on `NV_ERR_NO_MEMORY`. The setting returns to 20 as soon as the hosts are back (the sysctl.d file is removed), the previous boot's journal decides. README and the results directory say so; the slow-state diagnosis itself stands.
- 2026-11-11 03:05: how early a slip can be heard (queue127, `results/20261106-voice/` section 16): the early speech tap is lossless (both clips byte for byte the plain renders), the words of a window are intelligible in the loop's clean estimate from step 10-12 of 20 and the verdict settles around step 14-16 (still moving by 0.01-0.02 to the end); an early stop at step 14-16 with the limit at 0.03 would have caught the two clear slips and raised one false alarm. The saving (20-30 % of a failed attempt) does not pay for the false alarms, so `H3_TP2_SPEECH_EARLY_STOP` stays an unenabled experiment. The journal logged only some of the early checks; the table was scored offline from the saved transcripts.
- 2026-11-11 02:16: the slow render state is proactive memory compaction (`results/20261111-slow-state-compaction/`). Caught live in queue127: both chains in one head process, chain 1 at 587-620 s per window with kcompactd0 idle, a compaction storm from the CPU scoring between the chains (kcompactd0 at 1000-2100 ticks a minute, no wakeups, few migrations: scanning), chain 2's first window 710 s (+21 %) at normal GPU power and clock. `vm.compaction_proactiveness` 20 -> 0 at 01:58:49: kcompactd0 silent within the minute, the next windows 612 s (half under the storm) and 578 s, the fastest of the process. Persisted on both Sparks in `/etc/sysctl.d/99-h3-compaction.conf`. The explicit compaction before a boot stays. README's host-settings note updated.
- 2026-11-11 00:17: two results overnight. (1) The slowed RoPE clock (`toffslow` 0.5 beyond 2125, queue136, `results/20261109-long-chain-identity/` section 6) does not replace the cap: windows at offsets of 3187 and beyond had the worst lines of all chains (0.18 wrong characters), windows 7-9 were as clean as under the cap; the lever is the distance, the cap at about 2125 stays. (2) What a Ref2VA shot opens on (queue138, `results/20261110-shots-and-close-ups/`): with the scene frame among the references, in any position, a close-up opens on the scene frame's wide composition and pushes in within 1.5 s; with the character's crop as the only reference the shot opens in its framing, static and as sharp as a wide shot, on a set re-imagined from the prompt. The results directory holds the recipe for a film cut from shots (one window per shot, guards on, scene + crops for wide shots, the crop alone for close-ups) and the history of the three versions. queue139 renders the close-ups of the clip that way after the night's speech queues.
- 2026-11-10 19:41: the close-up test (queue137, shot 2 of the comedy clip four ways, 6 s, 1344x768, seed 5002): the control is byte for byte the v3 shot (the one-window loop path equals the direct call); without the character cards the model ignored the close-up and cast the wrong character (the prompt's pictures did not exist); the cards at native size (`H3_TP2_REF_NO_UPSCALE=600`) are not sharper; the ladder-tail arm was void (byte-identical to the control: with no guard configured the request took the direct path, which skips the loop's levers) and `guard_single` now routes one-window requests through the loop whenever `H3_TP2_LADDER_TAIL` is set as well. The finding that matters: every close-up opens on the scene reference's composition (the wide three-shot) and the camera rushes into the prompt's framing within about 1.5 s, which is the quality drop and the odd motion the owner saw; the frame-level sharpness numbers of close-ups (30-50 against 120-170 wide) are mostly composition, not blur. queue138 tests whether the order of the references sets the opening frame (the character's card first).
- 2026-11-10 17:10: `H3_TP2_AUTO_CARDS` verified on the GPU (queue135): a 10 s Ref2VA request of the doll scene with one reference image and the switch at 2 produced the same clip, byte for byte (framemd5 e1b5a1f7...), as the request given the scene plus the two crops `scripts/make_character_cards.py` cuts by hand; the server found the same two faces (512x798, 512x818). README and `docs/DIALOGUE_CLIPS.md` offer it next to the hand-cut crops. Also: the comedy clip v3 (queue134, nine shots at 1344x768 with the guard on every shot, `h3_tp2/guard_single.py` live): seven shots passed at the first attempt, two were re-sampled; the close-ups stay soft at 1344x768 as at 960x544 (frame sharpness 17-40 against 120-170 in the wide shots of the same clip), so queue137 renders one close-up four ways (crops as today, no crops, crops at native size, ladder tail).
- 2026-11-10 12:40: `H3_TP2_ID_LOCK=toffslow` (experiment, `h3_tp2/identity_lock.py`): the window's RoPE shift runs at full speed up to `H3_TP2_ID_LOCK_TOFF_CAP` and at `H3_TP2_ID_LOCK_TOFF_SCALE` (default 0.5) of it beyond, a slower clock instead of the stopped one of `toffcap`, meant to keep the cap's gain on the late lines without the few points of face similarity it cost (CPU test in `tests/test_identity_lock.py`; GPU run queued as queue136 on the 118 s chain with the crops).
- 2026-11-10 11:59: crops + capped RoPE time on the 118 s chain (queue133, `results/20261109-long-chain-identity/` section 5): with `H3_TP2_ID_LOCK=toffcap` at 2125 the second minute's lines go from 0.065 to 0.025 wrong characters of the script (windows 8-9 clean, which babbled in every chain without the cap), UTMOS 3.24, the man's face unchanged, the woman's a few points lower at the end (0.61 vs 0.66). README and `docs/DIALOGUE_CLIPS.md` carry the recipe: crops for the faces beyond 90 s, the cap when the late lines matter. Nodes synced to 5288f50 at 11:57; queue134 (the comedy clip re-shot at 1344x768 with the guard on every shot) running, then queue135 (auto-cards check), queue127, queue126.
- 2026-11-10 08:47: the guards judge one-window requests (`h3_tp2/guard_single.py`, 4 CPU tests; `H3_TP2_GUARD_SINGLE=0` turns it off). The speech and identity guards live in the continuation loop, which a plain 4-14 s request never enters: nine single-shot renders of 2026-11-10 ran with `H3_TP2_SPEECH_GUARD=0.03` set and nothing was judged (five of their lines came out with a slip). With a guard configured, a request of 107-345 frames without `long_video` now gets a continuation of exactly one window (its own frame count, 22-frame overlap) and takes the loop's first-window path: the same arguments as the direct call, plus the checks and seed retries. Requests longer than one window are left alone. Also: the nine-shot comedy clip (queue132) cut together on the Mac: 62 s from nine 4.5-10 s Ref2VA shots on one reference set (scene + three crops), close-ups and wide shots with the same faces, 19.9 s per step for a 10 s shot; the on-head assembly lost every second shot because `ffmpeg` inside a `while read` loop ate the list (fixed with `-nostdin`).
- 2026-11-10 06:59: the window-11 screen (queue131, `results/20261109-long-chain-identity/` section 4): windows 1-10 of the plain 118 s chain replayed, window 11 re-sampled with its seed under one lever per arm; the control reproduced the plain clip byte for byte. No lever restores a drifted face in one window (woman 0.55-0.65 in every arm, 2048 px reference included) and the look-lock is not the softening (replay without it: same sharpness). The RoPE shift capped at window 6 (`H3_TP2_ID_LOCK=toffcap`, cap 2125) took window 11's wrong characters from 0.17 to 0.007 (0.014 with hand-over noise) with UTMOS 3.20 against 2.86: the text's RoPE distance looks like the cause of the second minute's slips. Hand-over noise 0.35 sharpened the face by a quarter, the ladder tail the frame by 9 %, neither moved identity. Next: the 118 s chain with crops and the cap (queue133).
- 2026-11-10 02:38: `h3_tp2/pace.py` ignores replayed windows: a window taken from saved latents (`H3_TP2_REPLAY_LATENTS`) took 0.04 s and became the fastest of its size, so the first real sampling after it was logged as 285 % slower (queue131). Samplings under 5 s are logged as replayed and neither timed nor compared (`tests/test_pace.py`).
- 2026-11-10 02:35: `H3_TP2_AUTO_CARDS=<max faces>` (experiment, off by default, not yet run on the GPU; `h3_tp2/auto_cards.py`, 11 CPU tests). The server cuts the character cards itself: a Ref2VA request with one image and no video reference gets a crop per detected face appended on rank 0 before `prepare_encoder_inputs` (upstream prepares on rank 0 and broadcasts, so the ranks need not agree), using `scripts/make_character_cards.py` in the identity guard's interpreter. First GPU check planned: the 60 s doll scene at seed 3401 against `lalka60-multiref` (the same crops by hand) and the 118 s chain against `lalka118-cards-s3701`. `AUTO_CARDS` is in the queue library's strip list.
- 2026-11-10 02:30: faces over two minutes (`results/20261109-long-chain-identity/`). The 118 s doll chain with the scene reference alone loses the woman in the second minute (similarity to the reference per window 0.71 -> 0.60, the man 0.85 -> 0.75; plain and speech-guarded renders alike); with a head-and-shoulders crop per character (queue129, `scripts/make_character_cards.py`, same seed) both faces stay the same person for 118 s (woman 0.75-0.80, man 0.80-0.85 per window) for +25 % per step. Up to 60 s nothing drifts on three seeds, and moving the windows' RoPE time (`toff0`, `toffcap`) changes the faces by less than the noise there. README and `docs/DIALOGUE_CLIPS.md` now say: beyond about 90 s, add the crops. Also recorded: a `nice -n 19` CPU scoring job on the rendering head cut the GPU from 84 W / 2150 MHz to 50 W / 1800 MHz for as long as it ran (CPU at 95 C): CPU work on a rendering Spark is a GPU slowdown, and a candidate for the unexplained slow renders. The owner's three-speaker comedy clip (queue128b, 53.9 s, scene + three crops): 38.5 s per step (+40 % for four references), all three faces 0.77-0.81 mean, speech 0.04 CER with two invented words misheard.
- 2026-11-09 21:46: two more experiments from a read-only code review of the long-clip quality (none run on the GPU yet; `tests/test_ref_follow.py`, `tests/test_ladder.py`, 14 CPU tests). `H3_TP2_REF_FOLLOW_WINDOW=1` (`h3_tp2/ref_follow.py`): upstream's `packed_sequence.py` shifts target, guide and video-reference rows by the window's temporal offset and leaves image blocks at the media time origin, so window 11 of a 118 s clip sits about 4200 RoPE units from its reference; the switch moves the image rows with the window. `H3_TP2_LADDER_TAIL=<u,...>` (`h3_tp2/ladder.py`): the 20-step ladder ends with one x0 jump from sigma 0.39; the switch appends low-noise positions (0.025,0.0125: 0.39 -> 0.24 -> 0.13 -> 0, +2 forwards) with the earlier steps unchanged. The review also found the queued hand-over noise levels (0.05, 0.15) too low to matter and the 64.6 s chain the wrong regime for identity: the night's test is now a screen on window 11 of the saved 118 s chain (`H3_TP2_REPLAY_LATENTS` of windows 1-10 + `H3_TP2_REPLAY_PARTIAL=1`, same seed): control, noise 0.15 and 0.35, `toffcap` 2125, reference follow, ladder tail, a 2048 px reference, and a replay with the look-lock off.
- 2026-11-09 20:57: `H3_TP2_CONT_EDIT_NOISE=<level>` (experiment, off by default, not yet run on the GPU; `h3_tp2/handover_noise.py`, 13 CPU tests incl. upstream's loop). The owner looked at the 118 s clips of queue124: the characters become other people and the picture softens, in the plain and the guarded render alike. Measured on both: face similarity to the reference per window, woman 0.71 -> 0.60, man 0.85 -> 0.75; Laplacian sharpness of the frame 186 -> 155 and of the woman's face 311 -> 196, the largest step from window 1 to 2; the raw window latents keep their spread and their high-pass energy (0.35 -> 0.38), so a lock on latent statistics has nothing to hold on to. The speech test had been reported on its speech numbers without a look at the picture. A continuation window is shown the previous tail as a keyframe (timestep 0.999); the switch shows it at a chosen noise level and timestep `1 - level`, as the audio pin was loosened on 2026-11-07, and restores it clean in the output. Queued next to it: the same 118 s chain with a head-and-shoulders crop per character (queue129). `scripts/window_detail.py` prints the sharpness per window, whole frame and named boxes (`tests/test_window_detail.py`).
- 2026-11-09 20:47: the speech guard listens to what the finished clip keeps. The continuation loop appends a window's audio without its first 0.92 s (they repeat the previous window's tail), and the guard used to transcribe the whole window. Window 9 of the 118 s chain started its first line inside that part: the guard heard `Nie sucha, ale ludzie ...` (1 wrong character), the clip starts at `ale ludzie` (9 wrong), in the plain and the guarded render alike. `cont_match.heard_audio` cuts the repeated steps off before the check, for the guard and for the early check (`H3_TP2_SPEECH_WHOLE_WINDOW=1` restores the old behaviour); `tests/test_cont_speech_heard.py`. One such window in 370 scored, so the cost is a rare extra sampling. Not yet run on the GPU.
- 2026-11-09 20:41: `H3_TP2_KEEP_POOL` (measurement / experiment, off by default, not yet run on the GPU; `h3_tp2/keep_pool.py`, 17 CPU tests). A read-only code review named a candidate cause of the slow state: vLLM-Omni v0.30.0 hands the allocator's idle pages back at every stage boundary (`_release_stage_cache`, `torch.accelerator.empty_cache()` on component offload), so every stage takes its pool from the host's free memory as it is at that moment (after a model load that is mostly page cache). `observe` logs every release on both ranks with the reserved size, the device allocation counters and MemAvailable; `<GiB>` skips the release while MemAvailable is at or above that floor. `KEEP_POOL` is in the queue library's strip list. The review also checked the step for exact reuse: one forward per step (the checkpoint is CFG-distilled), attention at 87-89 TF of a 90-100 TF device, so the lossless headroom left is the slow states, the Ulysses exchange and items of 1-2 %.
- 2026-11-09 20:31: Speech guard on a 118 s chain (queue124, 11 windows, doll scene, seed 3701, limit 0.01, two judges; `results/20261106-voice/` section 15). Plain: 6566 s, 50 wrong characters of 1302 on the finished clip, whole words lost in windows 8 and 11, whole-clip WER 0.104. Guard: five windows re-sampled, eight extra samplings, 11 519 s (+75 %), 13 wrong characters, WER 0.042. Paired UTMOS guard - plain +0.08 (the three 64.6 s chains had -0.09 to -0.13), so the voice change of guarded clips is chain-to-chain spread; README and `docs/DIALOGUE_CLIPS.md` no longer state a 0.1 UTMOS cost. Two words wrong at every seed (`klientków`, `dłongo`) cost four samplings for nothing: prompt spelling, found by the dry run. Open: nine of the 13 characters left are the two words that open window 9, which the guard hears in the window alone and the finished clip lacks in both arms (the line starts inside the hand-over).
- 2026-11-09 15:04: `scripts/speech_early_table.py`: reads the head journal of renders made with `H3_TP2_SPEECH_EARLY` and prints the character error rate of every window sampling at each checked step next to the finished window's, then per step and candidate limit how many of the samplings that failed in the end a check there would have stopped and how many passing ones it would have stopped by mistake (the numbers behind a choice of `H3_TP2_SPEECH_EARLY_STOP=<step>:<limit>`). Pure Python, `tests/test_speech_early_table.py`.
- 2026-11-09 15:02: `H3_TP2_SPEECH_EARLY_STOP=<step>[:<max CER>][,...]` (EXPERIMENT, not yet run on the GPU; needs the speech guard): at the listed denoising steps the loop's clean estimate of the audio is judged like a finished window, and an attempt whose lines are already above the limit (the guard's own when none is given) is given up there and the window is sampled again with the next seed, so a failing attempt costs the steps up to the check instead of a whole sampling. Never the last attempt a window has, so a window always gets a complete sampling; an attempt that is not stopped is untouched. Rank 0 judges and broadcasts the verdict, both ranks leave the loop at the same step (`speech_early.Stop`, caught in `cont_match.wrap_diffuse`). The guard's summary line shows such an attempt as `0.052 at step 10`; a stopped sampling enters neither the pace record nor the latents digest and `H3_TP2_SAVE_LATENTS` (`h3_output`: a sampling that raises has no latents to hash or save, before it wrote a file holding `None`). CPU tests in `tests/test_speech_early.py`, `tests/test_cont_speech_early.py`, `tests/test_speech_guard.py`, `tests/test_output_diffuse_abort.py`. Which step and limit make sense is what the queue127 measurement decides.
- 2026-11-09 14:52: `H3_TP2_SPEECH_EARLY=<steps>` (measurement, observe only; `h3_tp2/speech_early.py`): with the speech guard on, the denoising loop's clean estimate of the audio at the listed steps of every sampling is decoded, transcribed and compared with the window's lines, and the character error rate is logged (`h3_tp2 speech early check window W attempt A step S of N: CER ...`). The tap wraps the loop's update call, reads its `x0` argument and returns the update untouched, so the render is unchanged; it is installed only when the loop is the reviewed v0.30.0 source, rank 0 listens and no verdict is taken (nothing to broadcast). The question behind it: the guard judges a window after its last step, so a slipping attempt costs a whole sampling; if the slip is already audible at step 8 or 10 of 20, a failing attempt can be stopped early. 15 CPU tests (`tests/test_speech_early.py`, `tests/test_cont_speech_early.py`, two in `tests/test_speech_guard.py`); the GPU measurement is queue127.
- 2026-11-09 14:39: speech guard control, `results/20261106-voice/` section 15 (queue125, rev 779102a). Seed 3601 with window 3 re-sampled and no selection (`H3_TP2_CONT_RETRY_FORCE=3`) against the same seed with window 3 sampled once at the retry seed in a new head process (`H3_TP2_CONT_RETRY_SEED_SHIFT=3:7919`): the two files are byte for byte the same, so the retry path leaves no state behind. Against the default render that clip loses no voice (paired UTMOS -0.01, windows 5-6 +0.03 and +0.04), so the 0.09-0.13 UTMOS the three guarded clips lost is not caused by re-sampling itself; whether it is the cleanly spoken attempt the guard keeps or chain-to-chain spread is open (two more controls queued).
- 2026-11-09 12:05: `scripts/queue-lib.sh`: `restart_head_gated` really restarts after a slow boot probe. `restart_head` skips the restart when the head is healthy with the requested env and code, which after a slow probe is exactly the boot the gate wants to replace, so the gate probed the same slow boot three times and kept it; the gate now sets `H3_FORCE_RESTART=1` for the next call. Test `tests/test_queue_lib_gated_restart.py` runs the real functions in bash with stub commands.
- 2026-11-09 12:01: sampling pace in the log (`h3_tp2/pace.py`, wired into `cont_match.wrap_diffuse`): every window sampling of a continuation request is timed and logged (`h3_tp2 pace window 4 attempt 1: 555 s (27.8 s/step)`), and a sampling more than 15 % slower than the fastest of the same size in the process (task, latent and audio sizes, steps, conditioning shapes; kept across requests) is a warning that says the host slowed down. Wall time only, no tensor is touched. Reason: the guarded seed 3601 chain of queue123 sampled its nine windows at 34.7 s/step in a head process that had just done the seed 3501 chain at 27.8, and nothing but the total render time showed it; the same happened 2026-11-08 17:10-17:45 and 2026-11-09 02:10-03:00. On the head `sar` shows every slow interval as about 0.3 core more kernel time; the cause is open (a once-a-minute host monitor is running on both Sparks). Tests `tests/test_pace.py` (pure python), `tests/test_cont_pace_loop.py`. Not yet on the nodes.
- 2026-11-09 11:48: speech guard calibrated, `results/20261106-voice/` section 15 (queue121-123). Three seeds of the 64.6 s Polish chain at limit 0.01 against the default render of the same seed: wrong characters by the guard's method with two judges 21 -> 7 of 2292, whole-clip character error rate 0.0175 -> 0.0057, 1-3 extra window samplings per chain (+17 .. +48 % render time), UTMOS -0.09 .. -0.13 per paired line with the loss in the windows after the first re-sampled one (a control that re-samples a window without selecting is running, queue125). One window failed all three attempts on the same two words. English showcase chain: one re-sample, 5 -> 1 wrong characters, no voice loss. Dry run over 108 finished windows: 22 % of Polish and 11 % of English windows are above 0.01. Guarded windows that are not re-sampled are bit-identical to the default render. The guard stays opt-in; `docs/DIALOGUE_CLIPS.md` carries the numbers.
- 2026-11-09 09:57: the speech guard names the words. `speech_guard.misheard` reads the character alignment back onto the script words (among the alignments of least distance the one that touches the fewest words, so a word that was not said is reported whole); the guard's per-window journal line gains `| misheard: zaprośmy > zapróczmi`, `speech_guard_dryrun.py` prints the same next to each window's rate (and writes it to the JSON), and the summary line lists, for the windows that stayed above the limit, the words that came out wrong in every attempt (`still above 0.01 after every attempt: w4 (zaprośmy ją)`): a word the model slips on at every seed is fixed in the prompt, not by re-sampling. From queue123: window 4 of the doll chain at seed 3501 failed all three attempts on the same two words (CER 0.046, 0.046, 0.030). Tests added.
- 2026-11-09 09:50: two measurement switches in the continuation retry loop (`h3_tp2/cont_match.py`): `H3_TP2_CONT_RETRY_FORCE=<k[,k]>` samples window k a second time whatever the checks say and drops the first attempt, `H3_TP2_CONT_RETRY_SEED_SHIFT=<k:n[,k:n]>` adds n to the seed of window k. FORCE=3 and SEED_SHIFT=3:7919 must give the same clip, which tests on the nodes whether a re-sampled window is a first attempt with another seed and nothing else (three guarded renders so far have a lower UTMOS in the windows after a re-sampled one). Both off by default and stripped by `queue-lib.sh`. Test fix: the plugin-loop harness (`tests/test_cont_checkpoint_loop.py::production_loop`) did not know the speech guard's names since 2026-11-08, so `test_cont_end_frame.py` and one test of `test_cont_seam_postcheck.py` failed with a NameError (the plugin itself was right); harness fixed, and the real loop body is now tested with a refusing speech guard and with the two switches (`tests/test_cont_retry_force.py`). Whole suite on a node copy: 601 passed, 8 skipped.
- 2026-11-09 08:00: speech guard with several judges: `H3_TP2_SPEECH_MODEL=small,medium` (and `speech_guard_dryrun.py --model small,medium`) fails a window only when every listed model hears the lines wrong, the lowest rate counts and a judge that hears them right settles it (`speech_guard.judge_wav`). On nine finished clips (54 windows) `small` and `medium` disagree by 1-4 characters on about a third of the windows and agree on every slip of 5 characters or more; at a limit of 0.01 the pair would re-sample 19 % of the Polish windows (small alone 25 %) and 6 % of the English ones (11 %). Tests added.
- 2026-11-09 07:55: `scripts/speech_guard_dryrun.py`: the speech guard's judgement on a finished clip, window by window, with the windows each limit would re-sample (reproduces the live guard's rates on `voicepl-guard01b-s3401` to the character); over the 15 Polish chain clips of sections 10-14 (90 windows) a limit of 0.005 / 0.01 / 0.02 would re-sample 40 % / 22 % / 11 % of the windows, over three English showcase renders (18 windows) 28 % / 11 % / 6 %, the 0.005 English hits being Whisper's `Novak` for `Nowak`. Test added.
- 2026-11-09 07:45: speech guard: one summary line at the end of a request (`h3_tp2 speech guard summary: 6 windows judged, 2 re-sampled (w3 0.037 > 0.000; ...), best attempts CER max 0.009`, plus the windows still above the limit after every attempt); the per-window prompt summary no longer reads the respelling file (the worker rank logged "prompts not captured" for it). Not yet on the nodes (queue123 runs 468b49c).
- 2026-11-09 00:25: speech guard fix: `config_from_env` reads only the environment and never opens files; the respelling table is read by rank 0 at scoring time (`Guard.respellings`). queue120 (2026-11-08 19:11-00:30) had the table on the head only, so rank 1 ran without the guard while rank 0 re-sampled the slipped windows (window 3 CER 0.037 -> 0.000, window 4 0.046 -> 0.053 -> 0.000, window 5 0.039 -> 0.000; the guard itself judged right) and the two ranks hung on mismatched collectives; its three renders are void. Test added; calibration re-run as queue122.
- 2026-11-08 19:15: queue119 (`results/20261106-voice/README.md` section 14): `toffcap` at three seeds lands between the default and `toff0` on every number (slips 2-3 per chain vs 5-6 and 1-4, CER 0.0065-0.0079, voice -0.07..-0.10 per paired line, last window -0.27..-0.48); the voice loss starts two windows after the first changed window at both settings, so the RoPE time is a dial between words and voice; both modes stay opt-in, the speech guard is the lever that leaves the sampling alone.
- 2026-11-08 18:45: `scripts/dialogue_lint.py`: words per window and per second, silent windows, mixed or missing language tags, unbalanced `<d>`, repeated lines, for hand-written continuation prompts (`docs/DIALOGUE_CLIPS.md` section 3); test added.
- 2026-11-08 18:40: speech guard (opt-in `H3_TP2_SPEECH_GUARD=<max CER>`, `h3_tp2/speech_guard.py`): every window's audio latents are decoded, transcribed on the CPU with `scripts/speech_qa.py` (3-4 s per window with the `small` model) and compared with the `<d>` lines of that window's prompt as a character error rate without word boundaries and with free speech around the lines; a window above the limit is re-sampled like a seam, cut or identity failure (`cont_match.wrap_diffuse` `speech_fn`/`max_cer`), rank 0 judges and broadcasts. `H3_TP2_SPEECH_RESPELL`, `H3_TP2_SPEECH_LANGUAGE` (default: the `[Language]` tag of the lines), `H3_TP2_SPEECH_MODEL`, `H3_TP2_SPEECH_PY`. STRIP widened; tests `tests/test_speech_guard.py`, `tests/test_cont_speech_check.py`. Calibration pending (queue120).
- 2026-11-08 15:20: queue118 (`results/20261106-voice/README.md` section 13): `H3_TP2_ID_LOCK=toff0` against the default on the 64.6 s Polish doll chain at three seeds; local RoPE time cuts the real slips of windows 2-6 from 5-6 to 1-4 (CER 0.005-0.009 vs 0.013-0.025) and costs 0.11-0.19 UTMOS per paired line, growing to -0.36..-0.70 in the last window (two default seeds differ by +0.06), so `toff0` stays opt-in; `toffcap` is rendered next (queue119).
- 2026-11-08 12:40: `scripts/voice_pairs.py`: the same spoken lines of two renders paired by text, UTMOS difference per line and per window (`docs/DIALOGUE_CLIPS.md` section 5); on the queue118 chains two default seeds differ by 0.06 per line while `toff0` loses 0.6-0.7 in the last window at both seeds. Test added.
- 2026-11-08 12:30: `h3_tp2/identity_lock.py` mode `toffcap` (opt-in, `H3_TP2_ID_LOCK=toffcap`, cap `H3_TP2_ID_LOCK_TOFF_CAP`, default 425 RoPE units = the shift of window 2): later Ref2VA continuation windows keep window 2's RoPE time instead of upstream's growing shift or `toff0`'s zero; the middle ground to test after queue118 (toff0 halves the real slips of Polish chains on two seeds but the voice drifts along the chain again). Test added; the STRIP pattern already covers the new variable.
- 2026-11-08 10:45: `scripts/voice-metrics.sh`: the speech checks of `docs/DIALOGUE_CLIPS.md` section 5 (transcript, word and character error rate per window, UTMOS by line and by window, speaker continuity, audio seams) in one command, each tool in its own interpreter, missing ones skipped with a note; the stale "under investigation" note on chain voice quality in that doc replaced by the 2026-11-07 result.
- 2026-11-08 10:00: `scripts/script_wer.py` prints a per-window and clip character error rate with the word boundaries removed (`CER`), so a Whisper word boundary placed elsewhere (`dziewczyna pisze` heard as `dziewczy napisze`) no longer counts as a slip; the 2026-11-08 chain-vs-fresh count is 5 real extra slips, not 7 (`results/20261106-voice/README.md` section 12, `docs/DIALOGUE_CLIPS.md`); test added.
- 2026-11-08 07:45: the chain adds a few slips of the tongue the fresh windows do not make (queue117,
  `results/20261106-voice/` section 12): windows 2-6 of the doll chain rendered fresh with the chain's seeds make 2 slips
  in 121 words against the chain's 7 (one seed, four of the five extra slips in the last window), while the chain's voice
  scores higher than the fresh windows' (3.60 vs 3.35 UTMOS) now that the pin is released. `scripts/script_wer.py
  --respell file`: phonetic respellings written into the prompt (`morze` for `może`) are scored as the real word; the
  Polish numbers of sections 10-11 were 0.02-0.03 too high without it. Defaults unchanged.
- 2026-11-08 05:55: sampling levers ruled out for Polish pronunciation (queue116 part 3, `results/20261106-voice/`
  section 11): 30 steps (1.47x the time) and `audio_flow_shift` 2.0 give the same words, slip for slip, as the default on
  three scripts x two seeds (mean WER 0.044 in all three, UTMOS within 0.1). The text is the lever (phonetic respelling),
  not the sampler. Defaults unchanged. Queue116 ended with FL2VA restored and the GPU free.
- 2026-11-08 02:21: `scripts/face_drift.py` counts the clip's windows the way the renderer did (`window_count`,
  `window_of`): the final overlap frames stay in the last window instead of printing a "window 7 (1 sample)" row for a
  six-window clip; the sample rows carry their window in the JSON. `tests/test_face_drift_windows.py`.
- 2026-11-08 02:20: the video side of the released audio pin measured on the three doll chains (hard pin / released /
  no pin): motion, detail, coherence within a unit, video seams 0.33-0.80 / 0.34-0.58 / 0.43-1.10 of the local motion,
  luma steps <= 0.2, no cut, no flicker, faces 0.813 / 0.818 / 0.823 and 0.707 / 0.700 / 0.701. The digest differs
  (joint sampling), the picture does not measurably. `results/20261106-voice/` section 10.
- 2026-11-08 02:15: `scripts/script_wer.py` scores by alignment, not by segment timing: the whole transcript is aligned
  word by word to the whole script and each error is charged to the window of the script word it touches (an insertion to
  the window heard before it). The midpoint rule put a Whisper segment that fused the last line of one window with the
  first of the next into one window and charged the other for the missing words (shop clip v4: 0.13 / 0.11 in windows 2-3,
  0.0 / 0.0 aligned). The per-window errors now add up to the clip's own `speech_qa.py` rate (`clip` in the output).
  Doll chains re-scored: 0.05-0.10 over the clip, the same slips in every hand-over arm. Tests updated (6).
- 2026-11-08 02:12: the 60 s shop showcase re-rendered on the 2026-11-07 defaults alone (1024 px reference, robust
  look-lock, audio pin released at 30 %): equal to the 2026-11-04 clip within one seed's noise (whole-clip WER 0.016 vs
  0.005 with the same two slips plus a clipped last word, UTMOS windows 2-6 3.82 vs 3.85, faces 0.68 / 0.64 vs 0.64 /
  0.62, no pause over 1.5 s, no cut, 12 096 s vs 12 031 s). The defaults that fixed the Polish chains cost the English
  clip nothing. `results/20261103-xclip/` section v4.
- 2026-11-07 22:55: `tests/test_audio_pin_steps_loop.py`: the step-gated audio pin inside upstream's verbatim denoise loop
  (20 steps: stock through step 5, released from step 6, logs once). queue116 confirmed the default alone reproduces
  queue115's `steps30r-64` bit for bit; the release line is logged by the rank holding the audio rows (rank 1).
- 2026-11-07 21:35: **the continuation audio pin is released after 30 % of the denoise steps by default**
  (`H3_TP2_CONT_EDIT_AUDIO_STEPS=0.3`; `1` restores the whole-run pin). Measured on the 64.6 s Polish doll chain
  (queue113-115, `results/20261106-voice/` section 10): windows 2-6 average 3.60 UTMOS against 3.26 with the whole-run
  pin, 3.64 with no pin, 3.51 with the spatial ramp, 3.40 with 12 pinned rows and 3.19 when released at 50 %; speakers
  0.84-0.98 to window 1; the quietest hand-overs of every arm in dialogue (9.9-25.2 against the clip's grid median 17.2)
  and in the 60 s painting's continuous ambience (2.4-5.0 against 4.5). The logger of `audio_pin_steps` moved under
  `vllm_omni.` so the per-window release line reaches the worker log. Deliberate change of the audio output.
- 2026-11-07 18:00: `scripts/script_wer.py` (+ `tests/test_script_wer.py`): the Whisper transcript of each
  continuation window against the dialogue written in that window's prompt (`<d>` tags of `continuation_prompts`, the one
  prompt otherwise), so slips of the tongue show up per window and per word. On the six 64.6 s Polish doll chains of
  queue111-114 the audio hand-over does not change pronunciation (means 0.07-0.11, the same words fail in every arm:
  `sz` clusters, one fused word boundary, the last window worst). The slips are the model's Polish, not the chain; the
  remedy stays the phonetic respelling in `docs/DIALOGUE_CLIPS.md`.
- 2026-11-07 16:30: the first build of the step release patched the step API's request state, which a render never
  uses (queue114 `steps30-64` came out identical to the control, framemd5 and all); it now wraps `minimax_h3_denoise_loop`
  and hands it a step-gated edit (`audio_pin_steps.StepGatedEdit`: the real edit before the release step, the free
  sampler's arithmetic from it on). queue115 re-renders the doll chain with it.
- 2026-11-07 11:40: `H3_TP2_CONT_EDIT_AUDIO_STEPS=<fraction>` (opt-in, `h3_tp2/audio_pin_steps.py`): later windows keep
  the pinned audio prefix only for the first fraction of the denoise steps and sample the rest free (the pinned rows
  then hold the clean tail re-noised at that sigma). The
  spatial mask ramp (`H3_TP2_CONT_EDIT_AUDIO=ramp`, queue113) recovered only half of the chain's voice loss
  (per-window UTMOS 3.88, 3.47, 3.48, 3.40, 3.65, 3.57 against the control's 3.87 .. 2.93 and 3.89 .. 3.68 without
  the pin). queue114 measures the step release on the doll chain and the painting.
- 2026-11-07 10:10: the seam scene from the scene frame alone (queue112: no character cards, no identity guard, seeds
  2601 and 2602) keeps the returning woman as herself on the new defaults: `changed` 0 of 32 and 0 of 30 figures in
  window 2, faces 0.85 mean, no cut, render 1228 s and 1534 s against 1908 s with cards + guard. The robust look-lock is
  the generic fix for a character who disappears behind something or turns away; cards and the guard stay opt-in
  insurance (`results/20261106-whole-person/`, last section).
- 2026-11-07 09:10: the voice loss along a continuation chain is the hard-pinned audio prefix (queue111: control
  3.87 -> 2.93 UTMOS over six windows, local RoPE time per window within 0.1 of the control, audio prefix not pinned
  3.89 -> 3.68 with the speakers as stable and the hand-overs not salient in dialogue; `results/20261106-voice/`
  section 9). `H3_TP2_CONT_EDIT_AUDIO` now takes soft values through `cont_match.audio_pin_mask`: `ramp` (0 at the
  seam rising to 1 over the prefix), `rows:N` (hard pin of the first N rows), a constant in (0, 1), `0` (none);
  default `1` unchanged. New `scripts/audio_seams.py` (log-mel distance and level jump at every hand-over against the
  clip's own grid) and `scripts/speaker_drift.py` (WavLM x-vector continuity per speaker across windows). queue113
  measures `ramp` and `rows:12` on the doll chain and on the 60 s painting.
- 2026-11-07 04:45: the new defaults verified on a real render of the seam scene (queue111 `tree-seam-S3`,
  `results/20261106-whole-person/`): video bit-identical to the robust replay (framemd5 `4908413a`), 0 of 35 figures
  changed (S2c: 10), faces 0.858, window 2 speech 3.47 UTMOS (S2c 3.14), no cut, 1908 s.
- 2026-11-07 03:20: continuation look-lock defaults: robust video lock ON (`H3_TP2_CONT_LOCK_ROBUST=1`), audio lock
  OFF (`H3_TP2_CONT_LOCK_AUDIO=0`). Queue104 (`results/20261106-whole-person/`): the 60 s painting rendered with no lock
  on today's code does not drift (light 115-140, seams 0.0-0.1, no flicker, colour within a unit), the robust lock holds
  it like the plain one (115-141 vs 115-142, inliers 88-93 %) and keeps a returning character where the plain lock
  recoloured her (queue107). The audio lock held 1-4 dB of a 10 dB ambience wobble and cost a window of speech 0.33
  UTMOS; the 23 dB climb of 2026-10-30 does not reproduce; a level-only latent lock was analysed on three chains of saved
  latents and dropped because the latent level does not track the decoded loudness (`results/20261106-voice/` section
  8). README, DIALOGUE_CLIPS updated. Verified next as a real render of the seam scene (queue111 `tree-seam-S3`).
- 2026-11-06 23:40: the voice loss of the long Polish clip is the chain's, not the lines' (queue110, `results/20261106-voice/`
  section 7): the lines of windows 4, 9 and 10 of the 118 s doll clip score 3.66-3.79 UTMOS as fresh single windows
  against 3.01 / 3.04 / 3.31 in the chain. Two candidate mechanisms go to a six-window test (queue111): the per-window
  RoPE time shift (window 4 starts at 1275 units, a training clip spans 600; `H3_TP2_ID_LOCK=toff0`) and the inherited
  pinned audio tail (new diagnostic `H3_TP2_CONT_EDIT_AUDIO=0`: later windows pin only the video prefix and sample
  their audio free, cutting the soundtrack at every hand-over; default 1, no change to renders). `scripts/queue-lib.sh`
  strips `H3_TP2_CONT_EDIT*`.
- 2026-11-06 20:50: look-lock replay on the saved seam-scene latents (queue107, `results/20261106-whole-person/`). The
  default lock is what changes the returning character (10 of 35 figures `changed`, colour distance 27.7; the coat goes
  red -> orange along the window); off, mean-only and the robust lock keep her (0 changed, 10.5 / 10.7 / 11.6). The
  audio lock costs the same window 0.33 UTMOS (3.47 -> 3.14). Defaults unchanged until the 60 s drift arms are in.
- 2026-11-06 20:00: `scripts/queue-lib.sh` strips `H3_TP2_CONT_LOCK*` and `H3_TP2_CONT_CUT_RATIO` between boots. The strip
  list named every lock variable one by one and missed the new `H3_TP2_CONT_LOCK_ROBUST`, so the robust arm of one queue
  stayed in the head's environment for the following boots (the control arm of the look-lock replay logged "robust").
  `tests/test_queue_lib_strip.py` fails when the plugin reads a continuation variable the list does not match.
- 2026-11-06 19:45: voice controls (queue103, `results/20261106-voice/` section 4). The second pair of lines of the
  monologue scores UTMOS 4.25-4.26 as a window of its own against 4.40-4.45 for the first pair, so about half of the
  FL2VA window-2 drop is the text and 0.15-0.2 the hand-over. On Ref2VA the hand-over costs 0.09 with our loop and 0.12
  with upstream's. The speaker's median pitch moves at the hand-over: -2.6 semitones with our loop, -7 with upstream's.
- 2026-11-06 17:00: robust look-lock, opt-in `H3_TP2_CONT_LOCK_ROBUST=1` (`h3_tp2/cont_match.py` `robust_params`): the lock's
  per-channel mean/std on both sides come from the positions that did not change between the reference tail and the
  window (median / MAD inliers of the per-position temporal-mean difference), so a character who returns from behind
  something no longer moves the correction. On the saved S2c window latents the plain lock shifted channels by up to
  0.40 std and squeezed one channel's std to 0.62 with the two figures as the outlier positions; the inlier estimate
  shifts 0.31 max and keeps the gains at 0.86-1.05. Checkpoint state carries the reference maps (`match.maps_*`). Under
  measurement in queue102 (replay) before any default changes.
- 2026-11-06 16:50: voice naturalness A/B (queue101, `results/20261106-voice/`): the audio look-lock is not what makes
  the later windows sound synthetic (window 2 of a 22 s monologue 4.05 UTMOS with the lock, 4.10 without; the lock moved
  the audio latent std by 1 %). The hand-over itself costs 0.25-0.3 on that clip against a seed spread of 0.11, with
  window 2 half as voiced as window 1, so the text is a confound; queue103 renders the window-2 lines alone and the
  monologue on Ref2VA with our hand-over against upstream's. New `scripts/voice_naturalness.py` (UTMOS22 + DNSMOS + pitch
  spread per continuation window), listed in `docs/DIALOGUE_CLIPS.md` checks.
- 2026-11-06 15:20: the seam-identity culprit located, two guard defects fixed. Renders are deterministic at a fixed
  seed (S2 and S2c share a `framemd5`). The guard judges the raw sample inside the attempt loop and the look-lock is
  applied to the chosen window afterwards: in the seam scene the frame the guard passed (red coat, the right face) and
  the same frame of the finished clip (orange coat, another face) differ because the lock's reference is window 1's
  tail, where the character is hidden (`results/20261106-whole-person/`, frame comparison). Fixed: the person bank
  lost its colour signatures at the first growth; head and body cards of one character were two characters (now merged
  by face, `person_bank.py build --merge-faces`). The lock is replayed on the saved window latents next (queue102).
- 2026-11-06 12:15: the owner's defect reproduced and measured: the occlusion ACROSS a window hand-over (queue98 S0, the
  woman hidden behind the oak and the man turned away at the end of window 1, scene reference only) brings the woman
  back as another person: an open orange-brown coat over a white blouse and a blue skirt instead of the closed red
  coat, blonder hair, an older face; the man, never fully hidden, is intact. The embedding checks pass her (figure
  0.86-0.92 against her card, face 0.76-0.78 "known"), so `scripts/person_bank.py` gets a **colour signature**: hair,
  torso and legs regions placed relative to the face box, (hue, saturation, value) each, from the cards at build time
  and from every judged person; the distance to the matched character (hue differences weighted by chroma,
  saturation differences weighted by brightness, summed over the regions) above `--max-colour` (default 15) makes
  the person `changed` = unknown for the guard, and it never joins the bank. Calibrated on the saved signatures: the
  same person 0-12 (the shop clip's dark clothes 0-5 over all 120 s, the park 3-9, the man after his turn up to 14),
  the changed woman 15-27 (10 of 11 instances; window unknown_frac 0.48 against 0 in every same-person window of
  four clips). Tests: signature regions, gating, wrap-around, card merging, changed-look scoring (18 pass).
- 2026-11-06 11:05: the park test bed finished (queue97, `results/20261106-whole-person/`): three reference sets on one
  boot, 22 s 960x544, the woman behind the oak and back in window 1, the man turned away and back in window 2. The
  short probe does not reproduce the owner's defect: with the scene frame alone both come back in the same hat, hair
  and clothes, one take, 0 cuts. Head cards + full-figure cards (V2) give the best faces (means 0.908 / 0.878 against
  0.868 / 0.869), the best figure similarity (0.883 / 0.840 against 0.865 / 0.827) and the smallest drift for 58 % more
  render time (1881 against 1191 s); head cards alone +26 % for half the lift. Whole-person scorer: 0 unknown persons in
  all six windows, back views unjudged. Next: the occlusion across the window hand-over (queue98: the woman hidden and
  the man turned away at the end of window 1; scene only against cards + the live guard with
  `H3_TP2_IDENTITY_PERSON_SIM=0.6`).
- 2026-11-06 09:55: the whole-person check judges only people with a confident face in their upper part. On the park
  probe (queue97 V0, scene reference only: the woman walks behind the oak and back, the man turns his back) a figure
  seen from behind scored 0.57-0.65 against the frontal cards, the same as a stranger in a similar coat on the 120 s
  clip (0.57), so without a face there is no evidence either way: such people are `unjudged`, not unknown. With the
  rule the V0 probe has 0 unknown frames in both windows (2 + 4 unjudged) and the 120 s clip exactly one unknown
  person, the stranger at 87 s (face 0.997, figure 0.57); the right-edge extras become unjudged. `person_bank.py
  --face-prob` (default 0.98, 0 judges everyone), `face_prob` saved with the persons. The guard's person check uses it
  through the script. Test bed V0: one take, 0 cuts, 1191 s, peak 62.3 GB, faces back at 0.85 / 0.90 after the events.
- 2026-11-06 09:10: the identity work widened from faces to the whole person (the owner's defect report: hairstyle,
  hair colour and clothes changed after a character went behind something or turned away). `scripts/make_character_cards.py
  --body` adds a full-figure card per character after the head cards (4 face widths wide, hat to about the knees).
  New `scripts/person_bank.py`: people found by a COCO person detector (torchvision Faster R-CNN MobileNet), each
  figure and its upper part embedded with DINOv2 ViT-S/14 (hair, hat, clothes and face together), scored against the
  characters' cards like `face_bank.py` (principal people only: at least 0.4 of the frame and 0.75 of the tallest
  person; `--save-persons` for bank growth). On the published 120 s clip at 1 fps: 246 principal people, the leads at
  0.82-0.95 to their cards (mean 0.85), 4 below 0.6, among them the stranger who walks in at 87 s (0.57). Both run on a
  CPU in the face venv on spark-06. Next: the park-and-oak test bed (a character walks behind a tree, another turns
  away) with scene-only, head-card and body-card references, 22 s 960x544.
- 2026-11-06 08:51: identity guard validated live (queue94, the 22 s occlusion probe with `GUARD=0.1`, `GROW=0.6`):
  the decoder's uint8 output confirmed in the journal, 11-13 faces per window scored, window 1 re-sampled on one
  unknown frame of 8 (the blurred pass-by, 0.08) and accepted on the next seed, the bank grown from 4 to 25 faces, two
  doubtful detections ignored by the probability rule; 41 s per check at 960x544, one re-sampled window, peak
  unchanged; the woman returns at 0.88 against 0.81 without the guard. Recipe in `docs/DIALOGUE_CLIPS.md`:
  `H3_TP2_IDENTITY_GUARD=0.1` + `H3_TP2_IDENTITY_GROW=0.6`, opt-in. Head restored to FL2VA with the serving defaults.
- 2026-11-06 07:58: the X shop scene rendered with the identity fix and its three failure cases written into the
  prompts (`results/20261106-xclip-events/`, queue92): 60 s 1344x768, scene + two 512 px crops, the woman turns to the
  shelves (window 2), the man turns away for a ledger (window 4), the woman bends below the counter (window 5). One
  take, 0 cuts, 0 pauses, WER 0.005, each character back as the same person after each event (faces 0.70 / 0.78 mean
  against 0.54-0.64 for the v2 clip without crops); 13 391 s including one window re-sampled on the interior cut check
  (ratio 4.23); the crops cost about 15 % per step at 1344x768; peak 70.1 GB. Delivered to `~/Downloads/h3-prod-smoke/xclip/`
  raw and with the two beds. The guard was a no-op in this run (pre-fix code).
- 2026-11-06 06:07: the guard calibrated offline on the published 120 s X clip (`results/20261106-identity-guard-calibration/`):
  of 25 unknown principal faces, 17 were the lead man in profile (0.30-0.38 to a frontal bank), 8 a brass scale on the
  counter detected as a face at probability 0.90-0.976 with similarity below 0.07, and 2 the stranger who walks in at
  87 s (the defect the clip was criticised for). `face_bank.py score` now counts a face as unknown only when the
  detector's probability reaches `--unknown-min-prob` (default 0.98; the real faces sit at 0.995+), and the saved faces
  carry width and probability. Simulated window by window with the guard's own acceptance: the probability rule plus
  bank growth at 0.6 leave the clip untouched at the 0.2 limit and re-sample exactly the stranger's window at 0.1
  (as deployed in queue93: 3 and 5 windows). A relative face-size rule was tried and dropped (no effect).
- 2026-11-06 05:57: identity bank growth (opt-in `H3_TP2_IDENTITY_GROW=<similarity>`, at most `H3_TP2_IDENTITY_GROW_MAX`
  grown faces, default 24). After a window the guard accepts, the principal faces that matched the bank at least that
  well join it (`face_bank.py score --save-faces`, `identity_guard.grow_bank`: oldest grown faces drop first, the
  reference faces never), so the bank follows a character's angle, light and expression through the clip instead of
  holding only the reference frame; a stranger stays below the threshold and never gets in. Exercised with the real
  scorer on frames of the queue93 clip (7 faces at 0.64-0.94 added to the 4 reference faces). The rank-0 scoring path
  now catches its own failures before the broadcast rank 1 waits in (a scorer crash accepted the window before too,
  but could have hung the pair). 3 more CPU tests (21 in the guard files). queue94 runs with `GROW=0.6`.
- 2026-11-06 05:50: the guard's empty scores explained before the validation render. The stack installs the VAE's
  streaming uint8 decode patch by default, so `decode_latent` returns uint8 `(B, C, T, H, W)` and the guard's float
  conversion scaled it to white (confirmed in `vae.py`); the scorer is healthy (`face_bank.py score` on five frames cut
  from the queue93 clip with the same bank: 7 faces, all known, mean 0.83). The fix in `dd00fd7` stands; queue94 shows
  it live. Guard cost at 1344x768 from the queue92 journal (no-op there): 67-68 s per window (decode + 8 scored frames)
  on a 1866-1927 s diffuse phase. queue92 itself re-sampled window 3 (cut ratio 4.23 > 1.70) and finishes later.
- 2026-11-06 03:46: identity guard diagnostics after its first scored run (queue93, two windows of the occlusion
  scene): the guard logged `0 face(s)` for both windows although the clip shows both faces in every frame, so the
  scored frames are wrong somewhere between the window's decode and the PNGs. The plugin now logs the decoded
  tensor's shape, dtype and value range per check (`identity decode: ...`), the guard keeps the sampled frames of the
  last two windows in its `h3-identity-*/wNNN` directory and logs their stats with the scorer's JSON, and a decoder
  that already returns uint8 is re-laid out instead of re-scaled (`identity_guard.frames_to_uint8`). 2 more CPU tests.
  The likely cause is exactly that: in plain requests the video VAE's `decode_latent` runs the streaming uint8
  write-back and returns uint8 frames, which the float conversion scaled to white (the stock output path checks
  the dtype first; the guard did not). The next probe (queue94) shows the decoded dtype in the journal.
- 2026-11-06 03:00: queue helper strips `H3_TP2_REF_NO_UPSCALE` and `H3_TP2_IDENTITY_*` from the head environment
  file between queues (`scripts/queue-lib.sh`): queue91 booted with queue90's `REF_NO_UPSCALE=600` still set and
  measured nothing new. The identity guard lives in the continuation loop, so a one-window request (up to 11 s at 277
  frames) is never scored; its first scored run is a two-window probe (queue93) before the 60 s shop scene (queue92).
- 2026-11-06 02:25: native-size crops closed (queue90, `results/20261105-multiref/`): `H3_TP2_REF_NO_UPSCALE=600`
  removes the crops' cost (diffuse 587 / 581 s against 726 s, same as the scene alone) and their effect with it (the
  woman after the occlusion 0.03-0.14 / 0.30-0.42 on the two seeds, against 0.42-0.70 with the crops upscaled to 1024
  px and 0.34-0.59 / 0.37 with none). The extra references work through their rows; the recipe keeps the default size.
- 2026-11-06 01:27: the character crops on a whole clip (queue89, `results/20261105-multiref/`): the 60 s Polish doll
  scene rendered with the scene frame + two 512 px head-and-shoulders crops against the PL5 baseline (same seed, lines,
  geometry): face similarity man / woman 0.844 / 0.758 against 0.810 / 0.701, minima 0.66 / 0.56 against 0.54 / 0.44,
  every window higher, the first window most; WER 0.021 against 0.035, 0 cuts, 0 pauses; +19 % render time (4083 s
  against 3434 s, 34.1 against 28.8 s/step) at the default 1024 px reference size. The crops are now a recipe step in
  `docs/DIALOGUE_CLIPS.md`.
- 2026-11-06 00:06: identity guard prototype, opt-in `H3_TP2_IDENTITY_GUARD=<fraction>` (`h3_tp2/identity_guard.py`,
  `scripts/face_bank.py`): every window of a continuation request (the first one too) is decoded after the denoising,
  eight frames are scored against a bank of face embeddings built from the request's reference images in a separate
  facenet-pytorch Python (`H3_TP2_IDENTITY_PY`, default `~/facenv/bin/python`), and a window where more than the
  fraction of the frames show a principal face (`H3_TP2_IDENTITY_MIN_FACE` 0.05 of the width) matching no reference
  face (`H3_TP2_IDENTITY_MIN_SIM` 0.4) is re-sampled with another seed like a seam or cut failure; rank 0 judges, the
  verdict is broadcast. `cont_match.wrap_diffuse(identity_fn=, max_unknown=)`; 9 CPU tests. Calibrated on the
  occlusion probes (`results/20261105-multiref/`): a stranger who stays trips it, a blurred pass-by frame or a double
  detection does not. First live run queued (queue91).
- 2026-11-05 23:55: reference key scaling closed (queue88, `results/20261105-multiref/`): `H3_TP2_REF_KEY_SCALE`
  1.6 on the two crops smears the picture into a drifting close-up (no face found, 5 cuts), 1.4 on all three images
  freezes and washes out the shot; the lever distorts the attention itself, not just its weighting. Stays opt-in,
  documented as a negative result. New opt-in `H3_TP2_REF_NO_UPSCALE=1` (`h3_tp2/ref_shape.py`): a reference image
  whose short edge is below `H3_TP2_REF_SHORT_EDGE` keeps its own size instead of being upscaled to it (a 512 px crop
  costs a quarter of the rows); 3 CPU tests; identity effect measured next (queue90). The 60 s Polish doll scene is
  rendering with scene + two crops (queue89) against the PL5 baseline.
- 2026-11-05 22:40: multi-reference, second seed and rendered cards (queue87): the crops' gain repeats (woman after the
  occlusion 0.57 on both seeds against 0.49 / 0.37 scene-only); model-rendered close-up cards do not help (0.42):
  crops, not cards. `scripts/make_character_cards.py` cuts the crops from a scene frame.
- 2026-11-05 21:30: several reference images per Ref2VA request measured (`results/20261105-multiref/`, queue84/86):
  the scene frame + a crop per character raises every face-similarity measure by 0.1-0.2 and causes no cut, but the
  woman still comes back weaker after an occlusion (0.57 against 0.83 before; scene-only 0.49 against 0.69). +25 %
  window time at 960x544 (the crops are upscaled to the reference short edge). Next: model-rendered identity cards,
  second seed (queue87).
- 2026-11-05 20:30: the 118 s Polish doll scene delivered clean (queue85): 0 cuts, 0 pauses >= 1.5 s, WER 0.085, faces
  0.81 / 0.68 to the end, window 11 cut check 1.10; second partial replay (10 windows) 1038 s. Story and numbers in
  `results/20261105-polish-pacing/`, "The 118 s version".
- 2026-11-05 19:00: first serving boot with the cut check PASS (queue83): the 16 s 1344x768 gate bit-identical to every
  default boot since 2026-11-03 (`200e38721c9e 4138eba5a964` / `e46129c05cc6`), cut check window 2 ratio 1.01, no retry.
  `scripts/queue-lib.sh` gained the `rate` and `digs` helpers the gate queues used to carry privately.
- 2026-11-05 18:15: the cut check scored an unseen cut: the repair render's window 11 (sampled before the check reached
  serving) cuts to a close-up at 114 s and measures 3.33 against 1.09-1.14 for the two clean windows next to it
  (`results/20261105-cut-check/`). The partial replay repaired windows 9-11 of the 118 s scene in 36 min (queue82).
- 2026-11-05 17:45: interior cut check of every continued window (`cont_match.cut_ratio`, `H3_TP2_CONT_CUT_RATIO`,
  default 1.7, `0` = off): the largest per-phase-normalised frame-to-frame latent difference of the window's free part;
  above the limit the window is re-sampled with another seed like a bad seam (same retries, same best-attempt rule).
  Calibrated on the 18 saved windows of 2026-11-05: clean 1.07-1.30, the cut-away the seam check had passed 1.96, a cut
  right after the prefix 2.72 (`results/20261105-cut-check/`). 6 CPU tests; the loop tests carry the new name.
- 2026-11-05 17:30: `H3_TP2_REPLAY_PARTIAL=1`: a latents replay may cover only the first windows of a request, the rest is
  denoised and saved as usual (`h3_tp2/h3_output.py`, 3 CPU tests). Repair of a long continuation whose late window went
  wrong without re-rendering the good windows; first use: the 118 s Polish doll scene of 12:12-14:57 (11 windows,
  WER 0.081, no pause >= 1.5 s) cut to a close-up of the letter at 88.7 s and stayed there, so windows 9-11 are
  re-sampled from the saved windows 1-8 with another request seed.
- 2026-11-05 12:15: the 192-frame window at 1344x768 measured directly (queue80, one-window 8 s FL2VA render): 43.0-43.3 s
  per step, as the step model from the 124/141/277-frame points predicted (42.8); 5.1 s of denoising per new frame against
  6.4 s for 277-frame windows, so shorter windows save about a fifth at the X-clip resolution, not a half
  (`results/20261105-window-length/`).
- 2026-11-05 12:00: Polish respelling proven (PL5, queue79, `results/20261105-polish-pacing/`): the same 60 s scene and
  seed with `może`, `pieniądze`, `porządnie` written phonetically in the prompt (`morze`, `pieniondze`, `porzondnie`;
  the speech-QA reference keeps the intended spellings) and a two-line tail: WER 0.035 against 0.082 (PL3/PL4), all
  three respelled words heard right, the tail fully spoken, 0 pauses >= 1.5 s, 0 cuts, faces 0.81 / 0.70. Left:
  `sz` clusters (`czynsz` -> `czynż`, `lepszym` -> `lepsym`) and `winien` -> `winiem`. Recipe in `docs/DIALOGUE_CLIPS.md`.
- 2026-11-05 10:45: continuation window length measured (`results/20261105-window-length/`, W192, queue78): 192-frame
  windows render the 60 s 960x544 Polish scene 11 % faster than 277 (3054 s vs 3434 s, 1.98 vs 2.28 s of denoising per
  new frame, peak 53-57 GB vs 64 GB) at a cost: eight hand-overs instead of five, the woman's face 0.61 vs 0.69 mean,
  WER 0.14 vs 0.08, and the 2.4 s tail window that had no line got an invented one. 277 stays the default; 192 is a
  documented draft option. `scripts/plan_dialogue_windows.py` now also warns below `--min-words-per-second` (default
  1.0 words per second of new content; a window with no line is the usual case) with a test.
- 2026-11-05 09:10: first serving boot on the flipped defaults (queue77, clean env): probe `bf6e85687a6a`, calibration
  `4d6d69e4aad0` at 19.60 s/it, both ranks install mode 2 and the fused quant (qkv, fc1, fc2) from the code alone, every
  self-check bit-identical. `results/20261105-levers-merge/`, "First serving boot".
- 2026-11-05 08:55: Polish pacing, second seed (PL4, `results/20261105-polish-pacing/`): WER 0.0816 both seeds, 0 cuts,
  0-1 pauses, faces 0.80 / 0.71; seven of nine transcript errors identical across seeds (`może`, `pieniądze`, `porządnie`,
  `czynsz`, `lepszym`, the dropped tail half-line): text-driven, not sampling noise. Respelling arm next.
- 2026-11-05 08:50: `docs/DIALOGUE_CLIPS.md`: the recipe for 60-120 s dialogue clips as delivered (reference frame from the
  model, three lines per window and the density rule, composition in every window prompt, the request and its costs,
  the checks, beds and mixing, replay after a late failure), with the open items.
- 2026-11-05 07:45: `scripts/plan_dialogue_windows.py` counts the words each window must speak in its new content
  and warns above `--max-words-per-second` (default 2.75, between the Polish pacing arm's observations: 2.26-2.64 words/s
  spoken, the 6 s tail at 2.85 dropped half a line); the report carries `words_approx` / `words_per_new_second_approx` per window and `warnings`.
  `docs/DIALOGUE_PLAN_MEASUREMENT.md`. `tests/test_plan_dialogue_geometry.py` imports its sibling tests as `tests.…`
  like the other files, so it runs by file name from the repository root (whole suite: 506 passed, 8 skipped).
- 2026-11-05 07:20: defaults flipped: `H3_TP2_ULYSSES_EXACT` `1` -> `2` (copy 5 of the exact Ulysses exchange
  not made, the out-projection quant reads the receive buffer) and `H3_TP2_FP8_QUANT_FUSE` `0` -> `1` (the per-token FP8
  quant fused into the qkv / fc1 / fc2 producers; the out site is skipped automatically under mode 2). Gate (queue75,
  boot with both): the 16 s 1344x768 FL2VA continuation twice, digests `200e38721c9e 4138eba5a964`, framemd5
  `e46129c05cc6`, the same as the cutlass gate (queue69) and every default boot since 2026-11-03. `=1` / `=0` restore
  the previous defaults. Expected saving ~0.55 s/step at 1344 (per-call timings); inside the boot-to-boot spread of
  the serving calibration. `results/20261105-levers-merge/`, "Default gate".
- 2026-11-05 06:10: the CPU test suite passes again (505 passed, 8 skipped; 57 failed since 2026-11-02). Two causes:
  the render tests' stub `curl` lives in `$H3_VENV/bin`, but `scripts/env.sh` sources `~/.h3-tp2.env`, which on a serving
  host sets `H3_VENV`, so the stub fell off PATH and the tests' requests reached the live API (one started a 60 s
  continuation on 2026-11-05 03:04); the render tests now run with a private `HOME`. And the continuation checkpoint
  prototype (`cont_checkpoint*`, the loop-extraction test helper, the inert `snapshot_state` / `restore_state`) did not
  know the class-aware look-lock reference (`lock_classes`, default since 2026-11-02): `lock_classes` is a request option
  and the A / B class statistics travel as `match.ref_A_*` / `match.ref_B_*`. Serving untouched. On a serving host run
  the suite with `H3_API_PORT=1 HOME=<empty dir>` so nothing can reach the head.
- 2026-11-05 05:35: serving proof of `H3_TP2_FP8_QUANT_FUSE=1` alone and together with `H3_TP2_ULYSSES_EXACT=2`
  (queue74, gated compacted boots): the 5 s 1344x768 calibration digest `4d6d69e4aad0` in all five lever runs, latents
  bit-equal to the defaults (max abs diff 0), all sites installed and self-checked bit-identical on both ranks (B: qkv,
  out, fc1, fc2, 20 000 fused calls; C: qkv, fc1, fc2 with the out site skipped automatically, copy 5 on the receive
  buffer), env clean after the restore. Speed inside the boot-to-boot spread (A 21.24, B 21.75 / 21.02 / 21.87, C 20.47
  / 19.82 s/it); the 16 s continuation gate (queue75) decides the default flip. `results/20261104-quant-fuse/` and
  `results/20261105-levers-merge/`, "Serving proof".
- 2026-11-05 03:50: `H3_TP2_ULYSSES_EXACT=2` (copy 5 of the exact Ulysses exchange dropped: the out-projection per-token
  FP8 quant reads the all-to-all receive buffer, `h3_tp2/ulysses_outq.py`, `results/20261104-ulysses-copy5/`) passes the
  serving proof: probe `bf6e85687a6a`, two 1344 calibrations `4d6d69e4aad0` bit-equal to the defaults boot, self-check
  bit-identical on both ranks. Opt-in until the combined gate with the fused quant (below) passes.
- 2026-11-05 02:30: the 120 s English clip for X is recovered by replay (`results/20261103-xclip/`, section 120 s):
  `H3_TP2_REPLAY_LATENTS` on a fresh Ref2VA boot reproduced the 14 saved windows (seam ratios and digests identical to the
  original journal), decode 632.9 s under the 7200 s wait, HTTP 200 after 21 minutes instead of an 8 h re-render; the
  clip: no pause >= 1.5 s, no cut, WER 0.054 over 34 lines, composition held for two minutes.
- 2026-11-05 01:50: Polish dialogue pacing (`results/20261105-polish-pacing/`): three short lines per 11.5 s window
  (queue71, 60 s 960x544 Ref2VA 1024 px, doll reference) against the 120 s one-line-per-window clip of 11-04: pauses
  >= 1.5 s 13 -> 0 (longest silence 1.4 s), non-Polish babble in 2 of 12 windows -> none, word error rate 0.172 -> 0.082,
  same render time (3434 vs 3384 s). Remaining errors are single mispronounced words with nasal vowels or `sz`/`rz`; the
  short final window must get fewer lines (one half-line was dropped there).
- 2026-11-05 00:45: `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` defaults to 7200 s in `scripts/env.sh` (upstream 600 s is a wait
  per engine step; the VAE decode of a clip is one step and took 642 s for 120 s at 1344x768, which killed the finished
  7.9 h render of the English X clip at 23:56). `H3_TP2_REPLAY_LATENTS=<globs>` (`h3_tp2/h3_output.py`) replays the
  `H3_TP2_SAVE_LATENTS` files of a request (one per window and seam attempt, saved on both ranks, geometry-checked) in
  place of the denoising, so the identical request is decoded and muxed in minutes; `scripts/queue-lib.sh` strips the
  variable at the next restart. The CPU loader test reproduces the render journal's latents digests bit-exact.
- 2026-11-04 15:55: `H3_TP2_FP8_GEMM=cutlass` is the default (`tensor32` = two-pass only). Gate queue69: two 16 s
  1344x768 FL2VA continuations on one cutlass boot bit-identical to both queue63 boots (digests 200e38721c9e
  4138eba5a964, framemd5 e46129c05cc6), window 1 82-87 vs 85-89 s/step; with queue67 five bit-identical requests
  in a row. Any DSL import/compile/check failure falls back to two-pass with a warning.
- 2026-11-04 14:05: 120 s doll scene on Ref2VA in one continuation request (queue66, 12 windows, 2048 px reference,
  `results/20261103-lalka-identity/`): 9808 s, no cut, no seam retry, faces 0.82 / 0.78 mean over two minutes (held
  at the 60 s level for 100 s, a staged turn-away in windows 10-11, back in composition by 114 s).
- 2026-11-04 10:45: doll identity arm C2 (`results/20261103-lalka-identity/`): the hidden end anchor with a second seed
  renders 60 s without a cut (faces man 0.777 / woman 0.739 mean, as C), so the 11-03 insert close-up was the window-4
  seed, not the anchor. Ranking unchanged: Ref2VA 1024 px (D) default, C the FL2VA-only fallback.
- 2026-11-04 09:10: cutlass level serving proof PASS (queue67, `results/20261103-cutlass-sm120/`): on one
  `H3_TP2_FP8_GEMM=cutlass` boot the 960 probe and three consecutive 5 s 1344x768 calibrations are bit-identical to the
  defaults boot (digests bf6e85687a6a / 4d6d69e4aad0, latents max abs diff 0), 20.6-21.4 vs 22.1 s/it. The second
  request no longer diverges, so the proxy fence closed the race in serving. Still opt-in; default gate = queue69
  (two 16 s 1344 FL2VA continuations against the queue63 digests and framemd5).
- 2026-11-04 07:50: doll identity arm D (`results/20261103-lalka-identity/`): Ref2VA with the 1024 px reference
  (`H3_TP2_REF_SHORT_EDGE=1024`) holds the faces at the 2048 px arm's level (man 0.806 / woman 0.746 mean vs 0.856 /
  0.790; FL2VA 0.659 / 0.601) in 3384 s, 6 % over plain FL2VA and 0.55x the 2048 px arm. No cuts, no seam retry. The
  default recipe for character clips is now Ref2VA with a 1024 px reference.
- 2026-11-04 06:30: X clip v2 (`results/20261103-xclip/`): composition stated in every window's prompt + three short
  lines per window: one continuous take, no cuts, pauses 1.5/1.7 s only (v1: seven of 1.7-2.5 s), WER 0.005, faces
  0.64/0.62 mean vs 0.46/0.43; 12 031 s. This is the clip for X.
- 2026-11-04 02:25: Ulysses exact exchange on by default (`H3_TP2_ULYSSES_EXACT=1`; `0` = stock). Gate queue63: on a
  defaults boot and on an exact boot the 960 probe, the 5 s 1344 calibration and a 16 s 1344 FL2VA continuation gave the
  same latents digests and the same decoded frames (`4d6d69e4aad0`/`c6f54e83dac8`, `200e38721c9e` `4138eba5a964`/
  `e46129c05cc6`); continuation window 1 85.1 vs 89.0 s/step. `results/20261103-ulysses-exact/` "Default gate".
- 2026-11-04 00:15: the English shop scene for X (`results/20261103-xclip/`): 60 s 1344x768 on Ref2VA with the 2048 px
  reference, 13 614 s; speech WER 0.007 (faster-whisper small), no language slip; faces hold (no drift within the clip)
  but window 1 ignored the reference composition and the model cut to it at 17 s; pauses 1.7-2.5 s. The first mixes left
  the ambience bed inaudible (fixed -16 dB on a -44.6 LUFS bed): `mix_v2.sh` normalises the bed to a target loudness
  (-37 LUFS ambience, -40 LUFS piano) before ducking. v2 queued: composition stated in every prompt, three lines per
  window.
- 2026-11-03 23:30: cutlass sm120 kernel race fixed (`h3_tp2/fp8_cutlass_kernel.py`, `results/20261103-cutlass-sm120/race.md`):
  a `fence.proxy.async` before the MMA-warp barrier at both stage releases orders the ldmatrix reads before the next
  TMA load; the stress reproducer `scripts/fp8_cutlass_stress.py` goes from 81 wrong GEMMs in 13216 to 0 in 12176.
  The level stays opt-in until the serving proof (three 1344 calibrations bit-identical on one cutlass boot).
- 2026-11-03 20:40: `scripts/boot-probe.sh` clears the caller's request variables (`H3_PROMPT_FILE`, `EXTRA_JSON`, ...)
  before its own request: a queue that exported a prompt file before its gated boot made the probe fail with
  "choose H3_PROMPT_FILE or PROMPT, not both" (boot kept, speed gate skipped).
- 2026-11-03 18:30: cutlass level VERIFY run (queue62): 4 of 48000 routed GEMMs differ from two-pass, each by 63-378
  elements in the first row(s) of a 128-row tile over one MMA warp's 64 columns: an A/B smem stage overwritten by the
  next TMA load while a lagging warp still read it (stage-release race); the per-call sync of VERIFY made the second
  1344 request bit-identical again. Opt-in stays off by default until the kernel fix (`results/20261103-cutlass-sm120/`).
- 2026-11-03 17:05: identity arm C measured (`results/20261103-lalka-identity/`): FL2VA with the hidden end anchor
  (`H3_TP2_ID_LOCK=end`, `H3_TP2_ID_END_EXTEND=68`) keeps the man at the Ref2VA level (0.77 -> 0.88) and the woman at
  0.73 mean (Ref2VA 0.79, plain FL2VA 0.59) for 1.40x the plain time (Ref2VA 1.93x); one window cut to a 4 s insert
  close-up of the letter. Ref2VA stays the recommendation for character clips; the end anchor is the cheaper fallback.
- 2026-11-03 15:10: speed plan item 5 proven in serving (`results/20261103-ulysses-exact/`, queue59): with
  `H3_TP2_ULYSSES_EXACT=1` the 5 s 1344x768 calibration gives the default digest `4d6d69e4aad0` twice, latents bit-equal
  to the defaults boot, 20.16 / 20.45 s/step against 21.21 (-3.6..-5 %, cross-boot). Still opt-in until the default gate
  (16 s 1344 continuation + 960 probe on an exact boot) passes.
- 2026-11-03 14:10: character identity across windows measured (`scripts/face_drift.py`, `results/20261103-lalka-identity/`):
  a one-request FL2VA continuation drifts like the 11-01 PNG-chained film (woman 0.74 -> 0.48 in 60 s) because windows
  2+ lose the keyframe; the same request on the Ref2VA partition (reference 2048 px conditions every window) holds both
  faces (man 0.80 -> 0.91, woman 0.78 -> 0.76) at 1.65x the time. Opt-in `H3_TP2_ID_LOCK` (`h3_tp2/identity_lock.py`)
  for the FL2VA partition is next to test; the boot probe sends a ref2va request on the Ref2VA partition.
- 2026-11-03 09:15: speed plan item 4 reopened and answered on the GPU (`results/20261103-cutlass-sm120/`): a CuTe-DSL
  sm120 FP8 GEMM with the per-row/per-column epilogue runs at 157-168 TF on GB10 (cuBLASLt 175-190) and is bit-identical
  to the two-pass output on all four shapes at M=21952 and 42848 (its fp32 accumulator equals cuBLASLt's). The 61 TF of
  the stock CUTLASS row-wise kernels is the tile raster without an L2 swizzle, not the hardware. Projection against the
  6b serving path: ~-1.5 s/step at 1344 (qkv, out, fc1), ~-0.7 at 960, double with SwiGLU in the epilogue; fc2 stays on
  cuBLASLt. Scripts `scripts/cutlass_sm120_*.py`; wiring as an opt-in `H3_TP2_FP8_GEMM=cutlass` level is next.
- 2026-11-03 12:30: opt-in `H3_TP2_FP8_GEMM=cutlass` wired (default unchanged: two-pass): `h3_tp2/fp8_cutlass.py` +
  `h3_tp2/fp8_cutlass_kernel.py` (the CuTe-DSL sm120 FP8 GEMM, NVIDIA BSD-3 header kept; `sa`/`sb` loads bounded by M/N,
  padded swizzle tiles skipped), compiled once at install (0.9 s) and checked bit for bit against two-pass (integer and
  random inputs, partial tiles); routes the DiT block's qkv / out / fc1 ahead of the stash hooks, fc2 keeps the
  consumer-fused path, any failure falls back to two-pass. Kernel level: bit-identical on all four shapes at M = 21952
  / 42848, fc1 bf16 + `silu_and_mul` bit-identical to fused SwiGLU level 1, host cost 41 us per call (no caching).
  Fleet proof (queue57): the first request of each resolution in a cutlass boot is bit-identical (1344 calibration
  4d6d69e4, 21.03 vs 21.64 s/step, -2.8 %), but the repeat of a resolution diverges (warm 960 probe a8713b89, second
  1344 calibration 1305859b): **not shippable**, root cause open (stream ordering, launcher caching and request-like
  offline sequences ruled out); `H3_TP2_FP8_CUTLASS_VERIFY=1` added to locate it in the next serving run. Found on the
  way and fixed in the default path: `torch._scaled_mm` (cuBLASLt, sm121) can read a scalar scale filled just before
  it as stale memory (some tiles off by a constant factor; reproduced without the DSL); the two-pass GEMM, the fc2 /
  fc1 stashes now share one synchronised `fp8_gemm.unit_scale` (same value; serving digest 4d6d69e4 unchanged).
  Test fixes: `tests/test_fp8_fused_gpu.py` did not parse (a comment swallowed a parametrize list) and leaked
  `fp8_swiglu._state` into later modules. `results/20261103-cutlass-sm120/` "Serving proof".
- 2026-11-03 09:00: boot probe tolerance 6 -> 8 % (limit 9.34 s/step on the 8.65 reference): the warm probes of the
  gated boots ran 8.96-9.16 against the old 9.17 limit, so a normal boot could be rejected and re-booted for nothing.
