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
