# MiniMax H3 on 2x NVIDIA DGX Spark (text-encoder TP2 + Ulysses SP2)

MiniMax H3-Base (FL2VA: first/last frame + prompt -> video with audio) served by vLLM-Omni across two DGX Sparks
connected back to back over ConnectX-7. The text encoder is tensor-parallel over the two GPUs, the DiT runs Ulysses
sequence parallel, so each Spark holds about two thirds of what a single Spark holds today and the diffusion steps are
shared. Same maths as the single-Spark recipe (online FP8 DiT, BF16 encoder and VAEs, cuDNN attention): the point is
longer clips (10 s and the model's maximum 15 s at 1344x768, which one Spark cannot hold) and twice the speed, not a
different model.

vLLM-Omni cannot launch diffusion workers on another host (`vllm serve` starts every rank locally and pins
`MASTER_ADDR=localhost`). This recipe adds `h3_tp2`, a small package that keeps the stock executor and only moves rank 1
to the second Spark: the executor's broadcast message queue gets a ZMQ remote reader, the worker's result queue is read
back the same way, and a tiny agent on the worker Spark starts the rank on request. No Ray, no fork of vLLM-Omni.

## Speed and memory

Same prompt, seed 1101, flow shift 12, 24 fps, T2VA (video + audio), online FP8 DiT, cuDNN attention. Times are wall
clock of `POST /v1/videos/sync` after a warm-up request; "peak" is the engine's `X-Peak-Memory-MB` (reserved pool, so an
upper bound); "min free" is the lowest MemAvailable sampled every 2 s on that host during the request. Raw files in
`results/`.

| Clip | One Spark (stock recipe) | Two Sparks (this recipe) |
|---|---|---|
| 5 s 960x576, 10 steps | 290.9 s, peak 97.2 GB, min free 15.7 GB | 153.8 s, peak 75.2 GB, min free 38.8 GB (head) / 39.6 GB (worker) |
| 5 s 1344x768, 20 steps | 1399.0 s, peak 103.3 GB, min free 6.0 GB | 643.0 s, peak 75.2 GB, min free 38.8 GB / 39.3 GB |
| 10 s 1344x768, 20 steps | does not fit: the host runs out of memory and livelocks | 1690.5 s, peak 84.2 GB, min free 26.1 GB / 29.6 GB (first boot with the stock FP32 output path and the VAE on one GPU: 2119.0 s, min free 10.2 GB) |
| 10 s 1344x768 from a still (FL2VA), 20 steps | does not fit | 1940.3 s, peak 91.2 GB, min free 19.9 GB / 29.3 GB (uint8 output path, VAE on one GPU) |
| 15 s 1344x768 from a still (FL2VA), 20 steps | does not fit | 3166.3 s, peak 84.4 GB, min free 24.1 GB / 26.5 GB |
| 15 s 1344x768, 20 steps | does not fit | 3079.9 s, peak 85.6 GB, min free 25.8 GB / 27.2 GB (final configuration). With the stock FP32 output path and the VAE on one GPU the same render ran the head out of host memory after 7796 s |

On one Spark the 5 s 960x576 render took 362.8 s when run right after the 1344x768 one (same frames, bit for bit): the
allocator keeps the bigger pool and the host is under pressure, which is also why the single-Spark recipe is limited to
5 s. The two-Spark numbers come from fresh boots with memory compacted first; all rows except the 10 s FL2VA one are
the final configuration (uint8 output path, VAE split over both GPUs, cache freed before decode). Phases: 10 s T2VA
prompt encoding 0.2 s, denoising 1602.6 s (80 s per step), VAE decode 84.2 s; 15 s T2VA 0.2 / 2946.7 (147 s per step)
/ 125.8 s; 15 s FL2VA 0.9 / 3030.9 / 125.9 s.

**State on 2026-11-02 (long stack, vLLM-Omni v0.30.0, FL2VA partition, all defaults):** 5 s 1344x768 T2VA runs at
21.1-22.0 s per step on a good boot (about 440 s of denoising; 23.2 was the 10-30 reference) and its VAE decode takes
26 s instead of 42 with the batched tile decode; 5 s 960x576 at 8.65-9.1 s per step warm. Long clips from a still at
1344x768: 16 s in 39 min, 30 s in 83 min, 60 s in 2 h 38 min (6 windows, `results/20261030-painting-fl2va-continuation/`).
Two things decide whether a boot is fast: the host's free memory layout (`sudo -n /usr/local/sbin/h3-compact` before
the start) and luck with kernel selection; `scripts/boot-probe.sh` renders two 5 s probes after every start and
restarts a slow boot (boots differed by 7-40 % with identical latents; `results/20261102-speed-plan/`). The first
request of a boot is 4-7 % slower and numerically different from every later one; warm up before comparing digests. A third thing decides whether a render stays fast: the kernel's proactive compaction (`vm.compaction_proactiveness`, default 20) scans the unified memory after any memory churn (the CPU scoring between two renders was enough) and a render under that scanning ran 21 % slower per step at normal GPU power and clock; with the setting at 0 the thread went silent within a minute and the next windows were the fastest of the process. Both Sparks run with the setting at 0 since 2026-11-11 (`/etc/sysctl.d/99-h3-compaction.conf`); under it two 64.6 s chains ran at 547-551 s per window, the fastest recorded on the pair. A LAN outage of both hosts the same night turned out to be their Wi-Fi access point, not the setting (`results/20261111-slow-state-compaction/`). Keep CPU scoring off a rendering host regardless: it halves the GPU's power budget.

## Quality checks

Tensor parallelism changes the order of floating-point reductions (all-reduce over two GPUs, sequence-parallel
attention), so the two-Spark render of a given seed is not bit-identical to the one-Spark render of the same seed; it
is the same model, the same steps and the same precision. What we measure, same seed and prompt on both layouts:

| Case | Decoded frames | SSIM | PSNR | Audio level (one / two Sparks) |
|---|---|---|---|---|
| 5 s 960x576, 10 steps | differ | 0.927 | 29.5 dB | -27.7 / -31.2 dB mean |
| 5 s 1344x768, 20 steps | differ | 0.959 | 37.2 dB | -46.4 / -46.9 dB mean |

Read this as "the same clip with a different noise realisation of the fine detail", which is what a change in
floating-point reduction order does to a 10 to 20 step diffusion; fewer steps diverge more. There is no step, precision
or model change between the two columns.

Run-to-run on one layout is deterministic (two one-Spark renders of the 5 s 960x576 case gave the same decoded frames,
`framemd5 b58a4612`). Every lever in the table below that claims "bit-identical" was checked by rendering the same seed
with and without it and comparing the SHA-256 of the uint8 frames that the plugin logs (`h3_tp2 frames sha256 ...` in
the head journal) and `scripts/verify-output.sh`'s framemd5 of the decoded video.

## Levers

| Lever | Setting | What it does | Lossless | Credit |
|---|---|---|---|---|
| Text-encoder TP2 | `--text-encoder-tp-size 2` | the 50-layer Qwen3-VL encoder (51.5 GB BF16) is sharded over both Sparks: ~26 GB per rank instead of 52 GB on one | same model and precision; reduction order changes (Quality checks) | vLLM-Omni |
| Ulysses SP2 | `--usp 2` | the DiT splits the video sequence across the two GPUs; all-to-alls around every attention layer over RoCE | same model and precision; reduction order changes | vLLM-Omni |
| Cross-host executor | `--distributed-executor-backend h3_tp2.executor.RemoteReaderExecutor` | rank 1 on the second Spark through vLLM's remote-reader message queues | yes | ours; vLLM shm_broadcast |
| Both ConnectX-7 twins | `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0` | all-to-all over two rails: 19.5 GB/s bus bandwidth instead of 12.8 | yes | NVIDIA playbook, our LLM recipes |
| Online FP8 DiT | `--quantization fp8` | the only way the DiT fits; encoder and VAEs stay BF16 | numerics of the single-Spark recipe | vLLM-Omni recipe |
| cuDNN attention | `--diffusion-attention-backend CUDNN_ATTN` | the GB10-validated backend | numerics only | vLLM-Omni recipe |
| Tiled VAE, split over both GPUs | `--vae-use-tiling --vae-parallel-mode tile --vae-patch-parallel-size 2` | the pixel tiles of each temporal chunk (4x7 = 28 at 1344x768, 3x5 = 15 at 960x576) are decoded round-robin by the two ranks and gathered; each tile runs the unchanged decoder, blend order unchanged. Measured: same frames (framemd5 `c36d17dc` / `8b3e06a8` with 1 and 2), 5 s 1344x768 681.8 -> 643.0 s, peak 77.5 -> 75.2 GB | bit-identical (measured) | vLLM-Omni |
| Batched tile decode (default since 2026-11-02; `H3_TP2_VAE_STACK_TILES=0` restores per-tile) | `H3_TP2_VAE_STACK_TILES=2`, `H3_TP2_VAE_TILE_BATCH=4` | each rank decodes its tiles of a temporal chunk four at a time (decode-scoped, no shared flag mutation): decode 81 -> 50 s per 226 frames at 960x544 and 186 -> 114 s per 277 frames at 1344x768 on identical latents, serving decode phase 23.9 -> 18.0 s at 960x576 | reduction order inside the tile batch only: max 1 gray level on 0.2-0.3 % of pixels, PSNR 73.9 / 74.7 dB (`results/20261102-speed-plan/`) | ours |
| Free cached pages before decode (superseded: vLLM-Omni v0.30 releases the stage cache itself, the plugin logs `empty_cache patch skipped`; measured 2026-11-02 as inert) | `H3_TP2_EMPTY_CACHE_BEFORE_DECODE` | `gc.collect()` + `torch.cuda.empty_cache()` right before the VAE decode so its peak does not stack on the allocator's denoise-time cache; no tensor arithmetic touched | bit-identical (measured) | brandneway, vllm-omni #7241 |
| uint8 output path | `H3_TP2_U8_OUTPUT=1` (default) | frames leave the GPU as uint8 (clamp, x255, round-half-even on the GPU: the same FP32 operations the stock path runs later in numpy) instead of FP32; every copy on the head shrinks 4x, and a SHA-256 of the frames is logged as the identity check | bit-identical (`tests/test_u8_identity.py`, same-seed frame digest) | ours |
| Two-pass FP8 GEMM (default 2026-10-29 to 2026-11-04, since then the path under the CuTe-DSL level below and its fallback; `H3_TP2_FP8_GEMM=tensor32` = two-pass only, `stock` restores the CUTLASS kernel) | `H3_TP2_FP8_GEMM=tensor32` | the per-row-scaled CUTLASS FP8 kernel runs at ~59 TFLOPS on the GB10 while the same GEMM with scalar scales runs at ~189; the plugin runs the scalar GEMM with fp32 output and applies the per-token and per-channel scales, bias and rounding in one Triton pass in vLLM's epilogue order. Warm 5 s 960x576 render: 451.6 s against 592.3 s (-24 %), first request after boot 437.8 against 771.1 s; at the serving cell (FL2VA 5 s 1344x768) 558 against 688 s (-19 %) with a byte-identical MP4, 10 s -15 % | same operations, different reduction order inside the GEMM; deterministic; at FL2VA 5 s 1344x768 the output is byte-identical to stock, at 960x576 Ref2VA and at 10 s the frames differ (same look) | ours; GB10 FP8 ceiling confirmed by NVlabs' sol-engine GB10 port |
| Fused in-GEMM FP8 epilogue (CLOSED 2026-11-02: measured slower, `H3_TP2_FP8_GEMM=fused` is a negative kept for reference) | off | one Triton FP8 GEMM with the row/column scales in registers: exact epilogue math but 86-90 TFLOPS against cuBLASLt's 169-177 on the GB10, +3 s/step; cuBLASLt's own outer-vector scaling is not supported on sm121 (`results/20261102-speed-plan/`) | no | ours |
| Fused epilogue + SwiGLU level 1 (default since 2026-10-29; `H3_TP2_FP8_FUSE_SWIGLU=0` disables, `2` experimental) | `H3_TP2_FP8_FUSE_SWIGLU=1` | fc1's fp32 product is read once by a kernel that applies the epilogue and `silu_and_mul`; the bf16 [M, 2*ffn] intermediate is neither written nor read | bit-identical to the two-pass path in nine 5 s renders (same latents and frames), denoising -5..-7 % at the median with a ~10 % boot-to-boot spread (`results/20261029-reproducibility/`); level 2 (per-token quant in the same kernel) stays experimental | ours; the fusion sites follow NVlabs' GB10 lossless tier |
| Consumer-fused FP8 scale pass (default since 2026-11-03; `H3_TP2_FP8_CONSUMER_FUSE=0` restores the separate pass) | `H3_TP2_FP8_CONSUMER_FUSE=1` | the kernels that consume the qkv, out and fc2 GEMM outputs (qk-norm/RoPE, gate-RMS, gate-add) read the fp32 product and apply the per-row/per-column scales themselves, so the bf16 intermediate is never written or read: 4.3 GB less traffic per layer at 1344x768, measured -1.04 s/step (1344) / -0.52 (960) in the microbenchmark, serving latents digest unchanged | bit-identical (same operations, same rounding point; probe bit-identical on every site, serving digest 4d6d69e4 unchanged) | ours |
| Continuation look-lock (defaults since 2026-11-02, robust since 2026-11-07; `H3_TP2_CONT_LOCK=off` disables) | `H3_TP2_CONT_LOCK_CLASSES=1`, `H3_TP2_CONT_LOCK_REF=tail`, `H3_TP2_CONT_LOCK_RAMP=1.0`, `H3_TP2_CONT_LOCK_ROBUST=1`, `H3_TP2_CONT_LOCK_AUDIO=0` | every continuation window's free latents are matched to the look of the first window's tail, the VAE's block-start tokens (every 5th) and the other tokens separately, with the correction growing linearly over the whole window; since 2026-11-07 the statistics come only from the latent positions that did not change between the tail and the window, so a character who comes back from behind something is not recoloured toward a tail without her (the plain lock turned a returning woman's red coat orange and changed her face, `results/20261106-whole-person/`); the audio lock is off since the same day (it cost a window of speech 0.33 UTMOS after a quiet tail and the ambience it was added for no longer drifts on today's code, `results/20261106-voice/`) | a deliberate look correction in latent space, same as before, now without the 17-frame steps (`results/20261102-vae-chunk-flicker/`) and without touching returning characters | ours |
| Ulysses exact exchange (default since 2026-11-04; `H3_TP2_ULYSSES_EXACT=0` restores the stock exchange, `=1` the 2026-11-04 default with copy 5) | default (mode 2 since 2026-11-05, next row) | keeps the stock all-to-all layouts and cuDNN inputs but drops the removable q/k/v packs and the reverse unpack: the consumer-fused qkv kernel writes q, k and v per destination rank, so each layer moves four 615 MB copies less | bit-identical (serving digest `4d6d69e4` on two consecutive 1344 requests, latents max|d| 0); 20.2-20.5 vs 21.2 s/step at 1344 (-3.6..-5 %, different boots) | ours |
| CuTe-DSL sm120 FP8 GEMM with the scale epilogue (default since 2026-11-04; `H3_TP2_FP8_GEMM=tensor32` restores two-pass only) | `H3_TP2_FP8_GEMM=cutlass` (default; `H3_TP2_FP8_CUTLASS_VERIFY=1` logs per-call mismatches) | one FP8 GEMM with the per-row/per-column scales in the epilogue (157-168 TF against the 61 TF of the stock CUTLASS raster) replacing the two-pass fp32 round trip | a stage-release race (fixed 11-03 with a `fence.proxy.async`, `results/20261103-cutlass-sm120/race.md`) made the second request of a boot diverge; after the fix three 1344 calibrations in a row on one boot are bit-identical to the defaults (digest 4d6d69e4aad0, latents max abs diff 0) at -3..-6.6 % per step; default gate (queue69) two 16 s 1344x768 FL2VA continuations on a cutlass boot bit-identical to the defaults boots (digests, framemd5), window 1 82-87 vs 85-89 s/step | ours; NVIDIA CUTLASS CuTe DSL |
| Ulysses exact without copy 5 (default since 2026-11-05; `H3_TP2_ULYSSES_EXACT=1` restores copy 5; `results/20261104-ulysses-copy5/`) | default (`H3_TP2_ULYSSES_EXACT=2`) | the reverse unpack, the last copy of the exact exchange, is not made: the attention gets a stride-0 placeholder and the out-projection's per-token FP8 quant reads the two 3584-wide row segments straight from the all-to-all receive buffer (one Triton kernel with the serving quant's Inductor arithmetic). -2.4 ms per layer and rank at 1344x768 (M = 18880), ~0.12 s per step (~0.6 %), 258 MB less transient memory | bit-identical on the GPU offline (every bf16 value as the row maximum; 9 GPU tests); the first call of every process compares it with copy 5 + the layer's own `QuantFP8` on the real tensor and keeps copy 5 for the process on a mismatch | ours |
| Per-token FP8 quant fused into the producers (default since 2026-11-05; `H3_TP2_FP8_QUANT_FUSE=0` restores the separate quant kernel; speed plan item 6, `results/20261104-quant-fuse/`) | default (`H3_TP2_FP8_QUANT_FUSE=1`, or a subset of `qkv,out,fc1,fc2`; `H3_TP2_FP8_QUANT_FUSE_VERIFY=1` compares every call, `H3_TP2_FP8_QUANT_FUSE_MODE=0\|1` forces a mode) | the kernels that produce the four GEMM inputs of a DiT block (qkv: the RMS-norm modulation, out: the Ulysses reverse unpack, fc1: the gate-RMS modulation, fc2: SwiGLU) write the fp8 row and its scale instead of the bf16 row, so the separate quant kernel and its two reads disappear: -0.30 / -0.55 / -1.26 s/step at M = 10240 / 18880 / 42848 from per-call timings | bit-identical on the GPU offline (fp8 bytes and scale bits at every site, the exhaustive row-maximum sweep, the whole block); self-check per site on the first call: a mismatch at the two RMS sites falls back to mode 1 (upstream's bf16 store kept, row read back), any other mismatch disables the site for the process | ours |
| The two together (the default pair since 2026-11-05) | `H3_TP2_ULYSSES_EXACT=2` + `H3_TP2_FP8_QUANT_FUSE=1` | mode 2 has no unpack copy to fuse into, so the fuse's out site is skipped automatically (head journal: `site out skipped automatically: H3_TP2_ULYSSES_EXACT=2 ...`; named in a list it logs `site out requested but ignored`) and the out-projection quant is ulysses_outq's; qkv, fc1 and fc2 stay fused, so `=1` and `=qkv,fc1,fc2` install the same three sites. With mode 1 the fuse's out site is the fused unpack + quant. A placeholder of one lever reaching the other's quant raises instead of computing | each lever's own checks hold unchanged; the pair has CPU tests (`tests/test_levers_combo.py`, whole block bit-identical to stock) and the serving proof (queue74: calibration digest `4d6d69e4aad0` twice, latents bit-equal to the defaults, the out line and `installed sites: qkv, fc1, fc2` on both ranks; queue75: the 16 s 1344x768 FL2VA continuation twice, digests `200e38721c9e 4138eba5a964`, framemd5 `e46129c05cc6`, the same as every default boot since 2026-11-03; first serving boot on the flipped defaults 2026-11-05 09:00: probe `bf6e85687a6a`, calibration `4d6d69e4aad0` at 19.60 s/it) | ours |
| Ulysses no-copy exchange (CLOSED 2026-11-02: numerically wrong on the GPU, rel-l2 0.20 against stock, and no faster; `H3_TP2_ULYSSES_NOCOPY=force` only for debugging) | off | one all-to-all for q/k/v written by the rope producer; 14 CPU tests pass, the GPU disagrees (cuDNN strided views or the split rope launches) | no | ours |
| Per-request step caches (EXPERIMENTAL, off by default, not yet measured on the GPU; speed plan item 7) | `H3_TP2_STEP_CACHE=1` (`verify` also recomputes and compares every hit) | within one `diffuse()` the RoPE table and the text token refiner output (2 blocks on the ~600 prompt rows) are computed on the first step and reused, and the loop's constant update masks become index tensors computed once (no `nonzero` + sync per mask use, no `torch.unique` sort per step); cleared at every `diffuse`; each patch is skipped with a warning when the upstream function is not the reviewed v0.30.0 source, and the refiner cache refuses step-gated attention backends, quantised KV and hooked refiner modules. Expected gain < 0.1 % per step (refiner ~5-8 ms of 21 s at 1344x768) | bit-identical by construction (same tensors reused, same rows in the same order; `tests/test_step_cache.py`, `torch.equal` against upstream's verbatim loop); GPU proof pending (same-boot latents digest) | ours |
| Watch or keep the allocator's pool at stage boundaries (MEASUREMENT / EXPERIMENT, off by default, not yet measured on the GPU) | `H3_TP2_KEEP_POOL=observe` or `=<MemAvailable floor in GiB>` | vLLM-Omni v0.30 returns the CUDA allocator's idle pages at every stage boundary (after the reference encode, before and after the decode, whenever a component goes back to the CPU), and on the GB10 the next stage takes them back from the host's free memory as it is then. `observe` logs every release on both ranks (reserved before and after, device allocation counters, MemAvailable); a number skips the release while MemAvailable is at or above it, so the pool keeps its pages for the life of the process. Kept as a measurement: the slow state it was meant to test turned out to be proactive memory compaction (`results/20261111-slow-state-compaction/`), and the pool watch showed each request returning only 0.5-7.5 GB of a 56-69 GB pool | no tensor arithmetic is touched (`tests/test_keep_pool.py`); GPU proof pending (latents digest) | idea from a code review, ours |
| Hand-over frames shown slightly noised (EXPERIMENT, off by default, not yet run on the GPU) | `H3_TP2_CONT_EDIT_NOISE=<level>`, e.g. `0.05` | a continuation window pins the previous window's last frames and the model sees them as a keyframe (0.1 % noise, timestep 0.999), so what a chain has already softened or changed in a face is copied on: over 118 s face sharpness falls by a third and the faces drift (0.71 -> 0.60 similarity to the reference) while the latent statistics hold. With a level the pinned rows are shown as `(1 - level) * clean + level * noise` at timestep `1 - level`, the model's own condition-noise convention, and restored clean in the output; the tail still fixes layout and motion, fine detail has to come from the reference image and the prompt (`h3_tp2/handover_noise.py`) | no: changes every window after the first | ours |
| Voice anchor (EXPERIMENT, off by default, not yet run on the GPU) | `H3_TP2_VOICE_ANCHOR=5` | every window after the first gets the opening seconds of window 1's audio latent as a Ref2VA audio reference (packed latent rows, no decode), a voice sample that cannot drift; the speaker similarity to window 1 falls to 0.75-0.94 over two minutes without it. The trained meaning of an audio reference is not documented, so the first run measures speaker similarity, UTMOS and whether lines are echoed (`h3_tp2/voice_anchor.py`) | no: changes every window after the first | idea from a code review, ours |
| Automatic character cards (opt-in; verified 2026-11-10: a 10 s Ref2VA request with the switch is byte for byte the request with the same two crops cut by hand) | `H3_TP2_AUTO_CARDS=2` | a Ref2VA request that carries one image and no video gets a head-and-shoulders crop per detected face (MTCNN, `scripts/make_character_cards.py` in the identity guard's interpreter) appended to its images on the head before the encoder inputs are prepared: the crops that kept both faces the same people over 118 s, without the user cutting them. Up to that many faces, left to right; a frame without a face gets none; a failure logs and the request goes on with the scene frame (`h3_tp2/auto_cards.py`) | no: more reference rows, about +12 % per card | ours |
| Reference images follow the window in RoPE time (EXPERIMENT, off by default, not yet run on the GPU) | `H3_TP2_REF_FOLLOW_WINDOW=1` | upstream leaves the image reference blocks at the media time origin while every continuation window moves 425 RoPE units further away (window 6: about 2100, window 11: about 4200, against a few hundred inside a training clip); the switch adds the window's offset to the image rows, so every window sees its references where window 1 saw them, with the text left in place (unlike `H3_TP2_ID_LOCK=toff0`, which moves the whole window and costs voice). The test of one hypothesis for the 60 s identity limit (`h3_tp2/ref_follow.py`) | no: changes every window after the first | idea from a code review, ours |
| A low-noise tail on the sigma ladder (EXPERIMENT, off by default, not yet run on the GPU) | `H3_TP2_LADDER_TAIL=0.025,0.0125` | the 20-step ladder with the video shift of 12 runs twelve forwards above sigma 0.92 and ends 0.68, 0.57, 0.39 -> 0: a window's fine detail is one x0 jump from 39 % noise, then the next window's clean truth. The switch appends the given positions before the final 0 (0.39 -> 0.24 -> 0.13 -> 0, two more forwards, about +10 % per window) with the earlier steps unchanged on the same seed; keep the audio pin release at the same step (`H3_TP2_CONT_EDIT_AUDIO_STEPS=0.27` for 20 + 2) when comparing (`h3_tp2/ladder.py`) | no: changes every window | idea from a code review, ours |

Tried, not adopted (lossy, so out of scope here): Cache-DiT / caching profiles, Turbo or distilled step LoRAs, NVFP4 or
INT8 DiT, CPU offload (unified memory: frees nothing).

## vLLM-Omni v0.30.0

The same package runs on vLLM-Omni v0.30.0 (vLLM 0.30.0, torch 2.13; the first release with the MiniMax-H3 long-video
continuation). Measured on the pair with the FL2VA partition, 5 s 1344x768 20 steps: 645.0 s against 643.0 s on the
pinned commit, peak 62.2 GB against 75.2 GB, 53 GB free on both hosts during the request, same quality by eye and by
SSIM (`results/20261027-v030-fl2va/`). A second runtime directory selected with `scripts/select-stack.sh` keeps both
stacks installed side by side.

Ref2VA on the same stack: 5 s 1344x768 in 1248.6 s (1.94x FL2VA, as upstream documents), peak 65.6 GB. The reference
image is subject/style conditioning, not a first frame.

**Clips longer than 15 s with real continuity**: upstream's latent-tail continuation (Ref2VA only, `long_video=true`)
renders the clip in windows of 277 frames with a 22-frame overlap, each window conditioned on the previous window's
latent tail (video and audio). First measurement on the pair: 30 s 960x576 20 steps in 3615 s (3 windows, peak
71.1 GB, 42 GB free on both hosts), no scene cut in the whole clip; the picture is continuous across both seams but
the camera changed speed and direction at the second one, which is visible when watched
(`results/20261027-v030-continuation-30s/`). A 60 s clip (6 windows, 7986 s, peak 69.4 GB) shows the limit of
the plain mode: with one fixed prompt the apparent motion dies down window by window (inter-frame change 7x smaller
by 53 s), the audio mean level fades by 18 dB and the model inserts shot changes it was not asked for; the picture itself stays sharp and consistent
(`results/20261027-v030-continuation-60s/`). At 1344x768 the same 30 s request takes 10114 s (2.8 h, peak 78.8 GB, 26 GB free) and its first seam came out as a
hard cut in the one sample so far (`results/20261027-v030-continuation-30s-1344/`). Asking for "one single continuous
take, no cuts, no transitions" in the
prompt removes the shot changes and the hand-over bumps at the seams (`results/20261027-v030-continuation-prompt/`,
+25 % time for the longer prompt); the motion and audio decay remain. A driving soundtrack (`audio_mode=lock_source`, `audio_reference`) removes the audio fade by construction, and its
samples also showed no motion decay and no shot changes over 30 s with the plain prompt
(`results/20261027-v030-continuation-audiolock/`), at the price of not generating the audio; the same two settings carried a 60 s clip through six windows with no cut,
smooth seams, flat audio and no motion decay in 7313 s (`results/20261028-v030-continuation-60s-best/`). One prompt
per window (`continuation_prompts`) keeps the motion and the
shot continuous at no extra cost but the generated audio still fades about 8 dB over 30 s
(`results/20261027-v030-continuation-winprompts/`). Local-time windows and a different seed per window were tested and
change nothing (`results/20261028-v030-continuation-toff0/`, `results/20261028-v030-continuation-seedstep/`). Repeats
after 16 h of continuous rendering are bit-identical and within 1 % in time, so the continuation is deterministic on the
pair and there is no sustained-load slowdown; a 345-frame window (the upstream maximum) gave smooth seams and a 1.7 dB
instead of 7 dB audio fade over 30 s at +8 % time in one sample (`results/20261028-v030-repeats-and-window345/`). Request shape in `scripts/render.sh` (`TASK=ref2va`, `EXTRA_JSON`).

**Ref2VA cost**: the 1.94x comes from the reference rows (upstream resizes the reference image to a 2048 short edge:
~7300 VAE rows plus the vision tokens in every DiT step). `H3_TP2_REF_SHORT_EDGE=1024` cut the 5 s Ref2VA render
from 1026.7 s (warm repeat; the first request after the boot took 1248.6 s) to 752.9 s (-27 %) with the same quality
by eye in one sample (`results/20261028-v030-ref2va-edge1024/`);
it changes the conditioning input, not the model; continuation runs with it are being measured.

**Long clips, where this stands after the 2026-10-27/29 campaign** (one sample per cell unless noted; all 960x576
unless noted; `results/20261028-clip-metrics/` has the per-second numbers):

| Goal | Lever | Result |
|---|---|---|
| Speed of the continuation | `H3_TP2_REF_SHORT_EDGE=1024` | -24 % at 960x576, -3 % at 1344x768, no other effect (same fades and detail): recommended |
| Motion decay across windows | per-window prompts (`CONT_PROMPTS=auto`) | fixed, free |
| Character identity across windows (faces) | Ref2VA with the reference at 1024 px (`H3_TP2_REF_SHORT_EDGE=1024`); **for clips beyond about 90 s add a head-and-shoulders crop per character** (`scripts/make_character_cards.py`, `H3_REF_IMAGES=scene.png,face0.png,face1.png`) | the scene reference alone holds the faces for 60 s (0.81 / 0.75 mean vs 0.66 / 0.60 on FL2VA) at 1.06x the FL2VA time and loses them in the second minute (118 s chains: the woman 0.71 -> 0.60, another person); with a crop per character both faces stay the same person for 118 s (0.75-0.80 and 0.80-0.85 per window) for +25 % per step; the 2048 px reference held a 120 s clip through window 9 for 1.8x. The lines of the second minute are a separate problem: the window's RoPE time runs away from the text, and `H3_TP2_ID_LOCK=toffcap` with `H3_TP2_ID_LOCK_TOFF_CAP=2125` (the clock stops after window 6) took the 118 s chain's wrong characters from 0.065 to 0.025 of the script with windows 8-9 clean; crops + cap + the speech guard at 0.01 together: 0.010 wrong characters, both faces held to the end (woman 0.69, man 0.84), 2.3x the plain render's time (`results/20261109-long-chain-identity/` section 7) (`results/20261109-long-chain-identity/`, `results/20261103-lalka-identity/`) |
| Identity after an occlusion or a turn-away | **the robust continuation look-lock, on by default** (2026-11-07: on the seam scene the returning woman keeps her coat and face from the scene frame alone on two seeds, 0 changed figures, 25-35 % less render time than with cards + guard; the plain lock had recoloured her, `results/20261106-whole-person/`). Extra insurance: the scene frame **plus a head-and-shoulders crop per character** as extra Ref2VA references (`scripts/make_character_cards.py`, `H3_REF_IMAGES=scene.png,face0.png,face1.png`) | +0.1-0.2 face similarity before and after the event on two seeds, no cut; on the 60 s Polish doll scene both faces higher in every window (means 0.81 / 0.70 -> 0.84 / 0.76, minima 0.54 / 0.44 -> 0.66 / 0.56) for +19 % render time (the crops must be upscaled like the scene: at their native size they cost nothing and do nothing); the model's own rendered close-ups do not help, scaling the reference attention keys breaks the picture (`results/20261105-multiref/`). On the public 60 s shop scene at 1344x768 with a turn-away, a turn to a shelf and a bend below the counter written into the prompts, every character came back as the same person: faces 0.70 / 0.78 mean against 0.54-0.64 without the crops, one take, WER 0.005 (`results/20261106-xclip-events/`). A replaced character is what the opt-in identity guard re-samples on (`H3_TP2_IDENTITY_GUARD=0.1` + `H3_TP2_IDENTITY_GROW=0.6`: every window decoded and 8 frames scored against a bank of the reference faces that grows with the accepted windows; 41 s per window at 960x544, 67-68 s at 1344x768; calibrated on the published 120 s clip and run live on the occlusion probe, `results/20261106-identity-guard-calibration/`) |
| Slips of the tongue in spoken lines | the speech guard, opt-in (`H3_TP2_SPEECH_GUARD=0.01`, `H3_TP2_SPEECH_MODEL=small,medium`): every window's speech is transcribed on the CPU and a window that says the lines wrong is sampled again; since 2026-11-10 a plain 4-14 s request is judged as well (it takes the loop's first-window path as a one-window continuation, `h3_tp2/guard_single.py`; `H3_TP2_GUARD_SINGLE=0` restores the direct call); since 2026-11-12 a window whose same words come out wrong at two seeds running is not sampled a third time, the best attempt is kept and the words are named in the journal (the prompt has to change, not the seed) | three 64.6 s Polish chains: 21 -> 7 wrong characters of 2292, for 1-3 extra window samplings per chain (+17 to +48 % render time); a 118 s chain (11 windows): 50 -> 13 of 1302 for 8 extra samplings (+75 %); paired UTMOS over the four chains -0.13 to +0.08, chain-to-chain spread and not a cost of the guard; an English chain: 5 -> 1, no voice loss. A word that is wrong at every seed needs the prompt, not the guard: `scripts/speech_guard_dryrun.py` names such words on a finished clip (`results/20261106-voice/` section 15, `docs/DIALOGUE_CLIPS.md`) |
| Generated-audio fade (-8..-18 dB over 30-60 s) | a driving soundtrack (`audio_mode: lock_source`) or the largest window (345 frames) | both keep the level flat; `audio_flow_shift` 1 makes it worse |
| Texture blur over 60 s | the largest window (345) | no blur, but over-sharpening instead at the end |
| Unrequested cuts and dissolves inside windows | prompt wording, light-lock wording, an opening-frames anchor, a tail anchor, H3's own looped audio as the track | none works: the cuts land on the same frames whatever the prompt (sampling events of the per-window seed), the anchors add cuts, the loop runs the motion away. One 60 s clip with a music track had no cuts; one sample. |
| A shot that must hold one framing (a close-up for a whole spoken line) | a film cut into shots rendered one by one (`results/20261110-shots-and-close-ups/`): Ref2VA with the scene frame plus the character cards, FL2VA from a still of the model's own close-up, the still pinned as latent frame 0 (`scripts/first_frame_pin.py`, with the card as the only reference) | none holds it: the wide table shot is the model's home framing, it opens on what it is given and cuts or pulls back to the wide shot within 0.7-3.5 s in five of six shots whichever way (sections 5-7); the first clause before the cut is a clean close-up insert, and the cards keep the right people in the wide shot. Framing is measured with `scripts/framing_trace.py` (face scale, camera zoom, cuts, where the opening settles). Open: the two-keyframe shot (first and last frame both given) |

So a 30 s 960x576 clip with reference 1024 + per-window prompts + a soundtrack came out clean (one sample,
`results/20261028-v030-recipe-30s-960/`); at 1344x768 and at 60 s no combination tried is reliably free of the
model's own scene changes, and the final detail drifts (blur with 277-frame windows, over-sharpening with 345). The
30 s 960x576 recipe is the honest offer; longer or larger is possible (the mode allows 300 s, about 2.2 h per 60 s)
but comes with the model's own cuts. Since 2026-11-03 the dialogue scenes (`results/20261103-lalka-identity/`,
`results/20261103-xclip/`) render 60 s without a cut when every window prompt restates the composition (camera
position, who stands where, faces three-quarter): six 60 s renders, four clean, one insert cut (the hidden-anchor arm C) and one
cut in the X clip's first version whose first window ignored the reference composition.
The X clip re-rendered on the 2026-11-07 defaults alone (1024 px reference, robust look-lock, audio pin released after
30 % of the steps) came out equal to the 2026-11-04 version within one seed's noise: 18/18 lines, no pause over 1.5 s,
no cut, faces 0.68 / 0.64, UTMOS 3.82 over windows 2-6 (`results/20261103-xclip/` section v4).

### A painting that comes alive (locked camera, the picture as the first frame): the request

This is the use the pair is tuned for since 2026-10-30 (`results/20261030-painting-fl2va-continuation/`): the still is
the first frame (FL2VA), the camera stays locked, the clouds, water, leaves and figures move, the soundscape is added
by hand afterwards. The continuation runs on the **FL2VA** partition with the pinned prefix (no reference block), so
the same request shape as a 10 s clip works for 16-60 s: add the two `long_video` keys.

```bash
curl -sS --max-time 30000 -X POST http://127.0.0.1:18093/v1/videos/sync \
  --form-string "prompt=The painting comes alive: a wide landscape under a huge sky. The clouds drift slowly across the sky, keeping their broad soft painted shapes, the light on the landscape stays soft and nearly constant, without strong moving cloud shadows, the river and the pond in the distance glint and ripple, leaves on the trees tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road walk naturally at a relaxed, unhurried pace, taking about one small step each second, without running, speeding up, time-lapse or jerky movements. The camera is locked on a tripod and does not move at all; one continuous take; soft natural ambience." \
  --form-string "width=1344" --form-string "height=768" --form-string "aspect_ratio=16:9" --form-string "fps=24" \
  --form-string "num_inference_steps=20" --form-string "flow_shift=12" --form-string "seed=1101" \
  --form-string 'extra_params={"task": "fl2va", "duration": 30, "audio_flow_shift": 3.0, "frame_indices": [0], "long_video": true, "long_video_mode": "continuation"}' \
  -F "input_reference=@painting.png" -o clip.mp4 -w "HTTP %{http_code}\n"
```

| | |
|---|---|
| Measured (Koninck landscape, 1344x768, 20 steps) | 16 s in 40 min, 30 s in 84 min, 60 s in 2 h 38 min (2026-11-02, speed-gated boot; owner: "wyglada spoko"): hand-overs invisible in every number (luma step 0.0 / -0.3, no inter-frame spike, no cut), the picture stays the picture, the clouds keep drifting, audio flat. 960x576 is about half the time. |
| Prompt | The wording above is the owner-approved one (2026-11-02, 60 s: "wyglada spoko"): camera locked on a tripod, clouds keep their painted shapes, no strong moving cloud shadows (with "their shadows glide over the plain" the shadows raced), figures at one small step per second. The model brightens a painting's tone over the first 20-30 s and then settles. Crop any border off the input first (below). |
| Sound | The generated ambience is there but can be muted. Its level moves within about 10 dB over a minute on today's code with no lock (`H3_TP2_CONT_LOCK_AUDIO=1` re-enables the latent audio lock, which held 1-4 dB of that at a cost to speech). Speech in a chain of windows used to lose about 0.15 UTMOS per window because every window hard-pinned the previous window's last 0.9 s of audio for the whole sampling and copied its flaws (3.87 -> 2.93 over six windows); since 2026-11-07 the pin is released after 30 % of the denoise steps (`H3_TP2_CONT_EDIT_AUDIO_STEPS=0.3`; `1` restores the whole-run pin), which keeps the chain at the no-pin level (3.60 against 3.64 average over windows 2-6, 3.26 pinned) with the same speakers and the quietest hand-overs of every variant measured, in dialogue and in continuous ambience; `H3_TP2_CONT_EDIT_AUDIO=ramp`, `rows:N` or `0` are the other hand-overs (`results/20261106-voice/` sections 9-10). |
| Flicker | Since 2026-11-02 the video look-lock fits the VAE's block-start tokens and the other tokens separately (`H3_TP2_CONT_LOCK_CLASSES=1`, default): the faint brightness step every 17 frames that the earlier lock produced in windows 2+ is gone (`results/20261102-vae-chunk-flicker/`). |
| Server | The FL2VA partition (`scripts/select-stack.sh long FL2VA`); the continuation defaults in `h3_tp2/plugin.py` (pinned prefix, seam retries, robust video look-lock, a seed per window). |

Long jobs can exceed the original four-hour limits, especially after a slowdown or a seam retry. Before starting one,
choose compatible server and client timeouts for the workload. For example, add these exports to the head's
`~/.h3-tp2.env` and restart the head **when no render is in progress**; existing processes keep their old limits:

```bash
export H3_TP2_REQUEST_TIMEOUT_S=28800
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=28800
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=29400
export H3_RENDER_TIMEOUT_S=30000
```

The first two bounds allow eight hours for the executor/wave, the API and client have time to report a failure, and
the memory guards remain active. A fifth limit is per engine step, not per request: the API server waits at most
`VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` (upstream default 600 s) for one step's output, and the VAE decode of a whole clip is
one step. That wait killed a 120 s 1344x768 render after 7.9 h of finished diffusion, during its 642 s decode
(2026-11-04; a 120 s clip at 960x544 decodes in 360 s and fitted). `scripts/env.sh` defaults it to 7200 s since
2026-11-05. Keep `H3_TP2_SAVE_LATENTS=<dir>` set for long renders: the diffusion output of every window (and every
seam-retry attempt) is written on both ranks, and a request that died after its diffusion can be replayed instead of
re-rendered. `H3_TP2_REPLAY_LATENTS='<globs of that request's files, in order>'` on a boot of the same partition makes
`diffuse` return the saved latents (geometry checked per file; a mismatching request fails rather than consuming a
file, so do not run the boot probe first) and the identical request then goes through the same hand-over, stitch,
decode and mux in minutes (the 120 s clip above came back this way in 21 minutes, with every seam check and latents
digest identical to the original journal, `results/20261103-xclip/`). With `H3_TP2_REPLAY_PARTIAL=1` the request goes
on with real denoising once the given files are used up: replay the good windows of a long clip whose late window went
wrong (a cut-away, invented speech) and re-sample the rest with another request seed, at the cost of only those windows
(the window seed is request seed + window index, so the replayed windows' seeds no longer matter). The queue library
strips both variables at the next restart. Increasing the executor timeout also delays detection of a genuinely stuck request;
these values are an example, not a guarantee that an arbitrarily long render will finish. External clients must use
their own matching timeout. In the workspace, `clients/h3-generate.py --long --first-frame --image painting.png ...`
sends this FL2VA request; without `--first-frame` the long client retains Ref2VA behaviour.

FL2VA resizes the first image directly to the requested width and height. Choose a canvas close to the image's aspect
ratio to avoid stretching it: for the 2316x1320 painting above, 960x544 differs by about 0.6%, while 960x576 compresses
it horizontally by about 5%. The workspace client now uses `ffprobe` to choose 32-aligned dimensions near the image
ratio when either dimension is omitted; supplying both `--width` and `--height` keeps the chosen canvas. At 960x544
the 5, 16 and 60 s controls passed media checks on the pair; repeated luma drops were detected in the longer two, so
these results are not a general quality guarantee.
The workspace client also accepts `--experimental-long --long --first-frame --duration 300` and requests streamed
MP4 encoding. One 960x544 painting request completed and passed media checks in 16,207.9 s; its sky progressively
deformed and the result is not a general quality guarantee. Its longer client timeout does not change the server's
own limits, and a failed request has no continuation checkpoint.
The workspace client refuses to replace an existing output. It saves the response under a unique temporary name,
checks the requested dimensions, 24 fps, frame count and approximate duration of both audio and video, and fully decodes the result
before publishing the MP4. On a transfer or validation failure, a nonempty partial response is kept separately for
inspection. These checks establish media integrity, not visual quality; the updated client has not yet been exercised
with a new live render.
Add `--exact-duration` to trim the result to the nearest 24 fps frame after generation. The helper verifies retained
decoded frames and AAC packets before publication; the audio can end up to one AAC packet before the target. This
optional step tries a fast stream copy for H.264 without B-frames and falls back to lossless video re-encoding if
needed. Verification still decodes the retained frames. A failed trim preserves the raw response separately.

**Check the input image for borders before rendering.** FL2VA keeps the first frame as it is, so a white band in a
screenshot crop stays in every frame (the owner's "white strip at the bottom", 2026-11-02: our Koninck crop had two
white rows at the top and eleven at the bottom). `python scripts/input_border_check.py image.png` reports nearly solid
or transparent bands along each edge (Pillow); crop them off before the request, for example
`python -c "from PIL import Image; im=Image.open('in.png'); w,h=im.size; im.crop((0,2,w,h-11)).save('out.png')"` with
the row counts the check printed. A border can be intentional, so the screen itself never alters inputs.

### Optional speech screen

`scripts/speech_qa.py` transcribes an audio/video file and, when given the intended dialogue, reports a word error
rate. It can help identify missing or garbled speech across clip lengths; automatic transcription can itself mishear
or hallucinate, and it does not assess delivery, speaker identity or lip synchronization. Run it in a separate Python
3.12 environment so its dependencies do not change the serving stack:

```bash
python3.12 -m venv .speech-venv
.speech-venv/bin/pip install 'faster-whisper==1.2.1' 'av>=11,<17'
.speech-venv/bin/python scripts/speech_qa.py clip.mp4 --language pl \
  --reference-file dialogue.txt --allow-download --output speech-report.json
```

Model download is opt-in; omit `--allow-download` once the chosen faster-whisper model is cached. The PyAV cap
avoids an observed incompatibility between faster-whisper 1.2.1 and PyAV 19. A synthesized Polish sentence was
transcribed with one word error out of 14 in the local smoke; that is an integration check, not a quality threshold.
`scripts/voice-metrics.sh clip.mp4 --language pl --extra extra.json` runs this transcript together with the per-window
word and character error rate, UTMOS, speaker continuity and audio seam checks of `docs/DIALOGUE_CLIPS.md` in one
command.

The automatic continuation prompt normally repeats the original prompt in every window, including any `<d>`
dialogue. In one 15 s Polish scene, speech recognition found the requested line again after the window boundary;
this is one observation, not a general frequency estimate. If a line is intended to finish in the first window,
the client can send different later-window wording with `CONT_PROMPTS=auto` and `CONT_PROMPTS_REST='The same shot
continues; the speaker rests, with quiet room ambience.'`. The rest wording applies to every later window; `{base}`
in it explicitly inserts the entire original prompt, including dialogue. For dialogue that must continue or change
across windows, supply an explicit `continuation_prompts` list instead. The dialogue-free continuation is being
tested and has not yet been established as a fix.

Pacing: put about three short lines into every 11.5 s window (277 frames at 24 fps) and fewer into a partial last
window. One line per window leaves the model 7-8 s to fill, and it fills them with 2-4 s pauses or, in Polish, with
invented non-Polish speech; three lines per window removed both at no render cost (`results/20261105-polish-pacing/`,
`results/20261103-xclip/`). Polish words with nasal vowels that the model mispronounces on every seed (`może`,
`pieniądze`, `porządnie`) are spoken right when written phonetically in the prompt (`morze`, `pieniondze`,
`porzondnie`): WER 0.082 -> 0.035 on the same scene and seed. Keep the 277-frame windows: 192-frame windows render the same 60 s scene 11 % faster at
960x544 but the extra hand-overs cost the second face and the speech, and a window left without a line gets an invented
one (`results/20261105-window-length/`).

The whole path from the reference frame to the mixed clip, with the costs and checks of the delivered 60-120 s scenes, is
in `docs/DIALOGUE_CLIPS.md`. `scripts/plan_dialogue_windows.py` can prepare that explicit list from a JSON array such as
`[{"start": 0.3, "end": 3.3, "speaker": "S1", "text": "Dzień dobry."}, {"start": 12, "end": 14.8,
"speaker": "S1", "text": "Do zobaczenia."}]`. Run, for example:

```bash
python scripts/plan_dialogue_windows.py --seconds 16 --scene 'A steady shot of one speaker.' \
  --dialogue lines.json --lock-source --report plan.json --extra-output extra.json
EXTRA_JSON="$(cat extra.json)" H3_AUDIO_FILE=/path/to/timed-voice.mp3 bash scripts/render.sh \
  result.mp4 16 960 544 20 1101 'A steady shot of one speaker.' /path/to/image.jpg first
```

The planner assigns each line to one window's new frames, rejects a line crossing a handover and does not put that
line in every other prompt. It was checked against the 10 / 15 / 30 / 60 / 300 s window counts, not yet with a real
model. Text assignment does not guarantee speech timing, continuity or silence; the `--lock-source` example also
needs a separately timed audio track. The 16 s, two-line GPU test is queued rather than completed.

For a local driving soundtrack, `H3_AUDIO_FILE=/path/to/audio.wav` lets `scripts/render.sh` send the audio as a
multipart reference without putting its potentially large base64 data in a single shell argument. WAV, MP3, FLAC,
OGG and M4A extensions are accepted by the client; set `EXTRA_JSON` with `"audio_mode":"lock_source"` when that
mode is wanted. Choose either `H3_AUDIO_FILE` or `AUDIO_URL`. The timing record stores the file's SHA-256 and size,
not its path or audio bytes. The checked server's FastAPI/Starlette stack defaults to a 1 MiB limit per multipart
text field, including base64 expansion. The client therefore refuses larger inline fields by default; use a URL the
server can reach, or set `H3_AUDIO_MAX_FIELD_BYTES` only after configuring and verifying a higher server limit. A 10 s
FL2VA reference with a Polish MP3 data URL passed media checks and transcription; the file transport itself has been
tested against a local multipart receiver, not a live H3 request or a 300 s track. The continuation wrapper now skips
its ambient audio look-lock when locked source rows are present: the old code could otherwise change later source
latents. This correction has CPU tests but has not yet been deployed or verified with a real multi-window GPU render;
always compare the delivered audio and timing against the source.

To screen whether the supplied audio is present in the result, use NumPy and ffmpeg/ffprobe with
`python scripts/compare_driving_audio.py source.mp3 rendered.mp4 --report audio-check.json`. It compares sparse
mono waveform windows and reports correlation, gain and estimated offset, accounting for available stream start
timestamps, plus approximate per-second levels at the analysis sample rate. These levels are not a true-peak meter.
In the one 10 s Polish audio-lock sample, three overlapping speech windows had correlations of
0.975–0.977 and zero measured offset; this is a single integration check. Silence, mixing, encoding and timestamp
ambiguity can affect the measurement. It does not assess lip sync, speaker identity, voice quality or the whole clip.

To screen each time-stamped line rather than fixed sparse windows, supply either the planner's report (its
`dialogue` list) or a JSON array of `{ "start": 0.5, "end": 3.2, "speaker": "A" }` objects:

```bash
python scripts/check_timed_audio.py source.mp3 rendered.mp4 --intervals plan.json --report lines-audio.json
```

The default threshold requires correlation at least 0.65 and a presentation offset at most 250 ms for each
interval; the CLI exits 1 if any line fails and 2 for invalid inputs. A synthetic noise test passed the source and
rejected silence and a 500 ms delay. These are screening thresholds, not proof of intelligibility or lip sync.

If an experimental render changed its driving soundtrack, an explicit post-processing option is
`python scripts/replace_driving_audio.py rendered.mp4 source.mp3 repaired.mp4 --report repaired.json`. It creates a
new file, encodes the supplied audio to AAC, and verifies every decoded video frame and timestamp against the input;
the original is preserved. This is not model-generated audio and does not establish whether mouths or speakers match.

For an optional per-line level experiment, `scripts/balance_dialogue_audio.py` accepts the same timed-dialogue JSON,
applies bounded gains with short ramps, checks decoded mono sample headroom and verifies every video frame. It writes
a separate copy and a report; RMS and sample-peak measurements do not replace listening or a true-peak measurement.

For an optional, experimental audio-to-mouth screen, `scripts/syncnet_probe.py` runs the original
[SyncNet](https://github.com/joonson/syncnet_python) on a fixed face crop and repeats the measurement after delaying
the audio by 200 ms. It needs ffmpeg/ffprobe, PyTorch, NumPy, OpenCV, SciPy and `python_speech_features`, a checkout
of SyncNet commit `907c0b579c2e2d83f0eae1b2ac9e720cde4e5623`, and the separately downloaded
[`syncnet_v2.model`](https://huggingface.co/ByteDance/LatentSync-1.6/blob/c42c7e6c8e9c213626389fa7d9a3c444b8536353/auxiliary/syncnet_v2.model)
(SHA-256 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`). The helper checks its two imported
SyncNet source files and weights before loading; the upstream loader uses `weights_only=True`.

It accepts a JSON face report with `samples` containing `time_s_approx`, `detected`, optional `face_count`, and a
normalized `[x, y, width, height]` `face_box`, with `y` measured from the bottom as in Apple Vision. For example:

```bash
python scripts/syncnet_probe.py clip.mp4 faces.json --duration 6.5 \
  --upstream-dir /path/to/syncnet_python --weights /path/to/syncnet_v2.model --report sync.json
```

Add `--start 10.5` to examine an interval later in the clip; both picture and audio are trimmed to the same start.
The source example returned offset 3 frames at 25 fps and confidence 10.09; delaying its audio by 200 ms changed
the offset to -2 frames. On two first 10 s Polish audio-driven scenes, the screen returned -2 frames and the
artificial-delay control returned -7. The fixed crop, speaker selection, language and synthetic speech can affect
the result, so neither a passing control nor a confidence value proves accurate lip sync or natural movement.

For multi-person scenes, macOS users can compile `scripts/face_landmarks_all.swift` with `swiftc -O` and pass it
sampled image paths to obtain all detected boxes as JSON. `scripts/track_face_detections.py` associates those boxes
with up to four initial normalized Vision positions and writes one `samples` report per track, compatible with the
SyncNet screen. Its assignment avoids reusing a face and handles missing detections, but position is not identity:
crossings, pose changes, occlusions and cuts can swap or lose a track. Check the crops and playback by eye.

For an edited sequence assembled from clips, `scripts/verify_assembled_shots.py` checks declared frame
counts, rates, geometry and audio start times, then compares three RGB frames and sparse audio windows per ordered
source. For example:

```bash
python scripts/verify_assembled_shots.py film.mp4 --shot first.mp4 --shot second.mp4 \
  --shot-seconds 10 --report assembly.json
```

Supply one `--shot-seconds` for all shots, or repeat it once per shot in order when durations differ, such as
`--shot first.mp4 --shot second.mp4 --shot-seconds 30 --shot-seconds 60`. Offsets are accumulated in frames.

NumPy, ffmpeg and ffprobe are required; each `--shot` must be supplied in the intended order. A synthetic test
distinguishes swapped red and green shots even though their grayscale brightness is similar. On a separate 20 s
two-shot H3 assembly, RGB mean absolute differences at the six sampled frames were 0.48–0.68 on a 0–255 scale;
median sampled audio correlations were at least 0.999. Re-encoding creates small differences. This sparse check
helps find assembly mistakes, but does not establish natural motion, scene continuity or speech quality.

`scripts/screen_segment_join.py first.mp4 second.mp4 --report join.json --contact join.png` also compares decoded
RGB frames around a join and saves a six-frame contact sheet. Its suggested-attention flag is only a screening
heuristic: an intentional cut or real motion can be large, and a small jump does not guarantee continuity of
characters, action or story. A synthetic test distinguishes a matching join from a color change.

An opt-in server experiment, `H3_TP2_VIDEO_FORM_LIMIT_MIB`, raises the implicit per-field limit only for POST
`/v1/videos` and `/v1/videos/sync` (allowed range 1–64 MiB); explicit caller limits remain in force. The plugin is
unchanged when this variable is absent. A local FastAPI 0.136.3 / Starlette 1.10.0 test accepted a field above 1 MiB
on the video endpoint while an unrelated endpoint kept rejecting it. It has **not** been installed or tested on the
live service. A larger limit allows more memory per field and does not guarantee that encoding a long soundtrack
fits GPU or host memory; keep the API private and verify actual long-audio behavior before relying on it.

### Long clips from a client or an agent: the request

**Status 2026-10-30 morning.** On 2026-10-29 the owner rejected the continuation by eye (a light and texture step at
the first hand-over, the subject doubled, a wash-out and a looping walk at 60 s). The night's fixes, now the defaults
(`results/20261029-seam-loop/`): the previous window's tail is pinned as target rows of the next window through
upstream's latent-edit path instead of a guide block (`H3_TP2_CONT_EDIT=2`); a latent seam check re-samples a window
that cuts right after the prefix (`H3_TP2_CONT_RETRY=2`), and since 2026-11-05 an interior cut check re-samples a window
that cuts away anywhere later in it (`H3_TP2_CONT_CUT_RATIO=1.7`, `0` disables; the 118 s Polish scene cut to a close-up
2 s after a hand-over the seam check had passed, `results/20261105-cut-check/`); a per-channel latent look-lock pulls every window's free
frames back to the first window's exposure and contrast, ramped over the window
(`H3_TP2_CONT_LOCK=first`, `H3_TP2_CONT_LOCK_RAMP=1.0`; since 2026-11-07 from the unchanged positions only,
`H3_TP2_CONT_LOCK_ROBUST=1`, so a returning character is not pulled toward a tail without her). The same lock for the
generated audio latents is opt-in since 2026-11-07 (`H3_TP2_CONT_LOCK_AUDIO=1`; it is skipped for a locked source), and
since the same day the pinned audio prefix is released after 30 % of the denoise steps (`H3_TP2_CONT_EDIT_AUDIO_STEPS=0.3`),
which stops the voice from degrading along the chain (`results/20261106-voice/` section 10).
Measured on a 60 s clip: five hand-overs with +0.3..+0.8 luma and no inter-frame spike, no cut, light 43-62 over the
minute (47-88 before), detail at most 442 (1084 before), motion 5-9 throughout, audio level flat. The owner's eye on the
16 s version passed ("no jump noticed"); the 60 s review is pending. `H3_TP2_CONT_EDIT=0`, `CONT_RETRY=0`,
`CONT_LOCK=off` restore upstream's behaviour.

The server must run the v0.30.0 stack on the **Ref2VA** partition (`scripts/select-stack.sh long Ref2VA` on both
Sparks, agent restarted before head) with `H3_TP2_REF_SHORT_EDGE=1024` in `~/.h3-tp2.env` on the head. On Ref2VA the
image is a **reference for look and subject**, not the first
frame: the clip resembles the picture, it does not start from it. One request at a time; the call is synchronous and
returns the MP4 (with audio) in the response body.

```bash
curl -sS --max-time 30000 -X POST http://127.0.0.1:18093/v1/videos/sync \
  --form-string "prompt=An old forest with a small stream. Leaves sway in a light breeze, the stream ripples and glistens, a woman in a white dress walks slowly along it. Steady camera, one continuous take, natural forest ambience." \
  --form-string "width=960" --form-string "height=576" --form-string "aspect_ratio=16:9" --form-string "fps=24" \
  --form-string "num_inference_steps=20" --form-string "flow_shift=12" --form-string "seed=1101" \
  --form-string 'extra_params={"task": "ref2va", "duration": 30, "audio_flow_shift": 3.0, "long_video": true, "long_video_mode": "continuation"}' \
  -F "input_reference=@picture.jpg" -o clip.mp4 -w "HTTP %{http_code}\n"
```

What to expect and what to tell the user:

| | |
|---|---|
| Lengths | 4-15 s: one window (same fields without the two `long_video` keys also work on Ref2VA). 16-60 s: windows of 277 frames continued from each other with the fixes above; longer is possible (the mode allows 300 s), untested beyond 60 s. |
| Size | 960x576 for long clips (1344x768 takes 2.8x longer at 30 s and its first seam cut in the one sample). |
| Time on the pair (two-pass GEMM + fused SwiGLU, reference 1024, pinned prefix) | 16 s: 18 min; 30 s: 42 min; 60 s: 67 min (first request after a boot up to 10 % slower; a fragmented host up to 50 %: compact memory before a boot). |
| What 30 s and 60 s look like | With the defaults above: 30 s in 2526 s (3 windows), 60 s in 4023 s (6 windows), seams measured continuous, one subject, exposure, grain and audio level held (`results/20261029-seam-loop/`). A window the seam check rejects costs its render time again (none of the night's 14 windows needed it). |
| Prompt | Describe the motion of what is **in** the picture (this water, these clouds, this person); a prompt that asks for something the picture does not contain makes the model move the camera to find it. End with "Steady camera, one continuous take". |
| Per-window prompts | Generated by the server when the request sends none (since 9a65815, `H3_TP2_CONT_PROMPTS=off` disables): the first window gets the prompt plus "one single continuous take", the following windows "the same continuous shot goes on without any cut or transition" plus the prompt. Without them every window re-introduces the prompt's subject (a second woman after the first seam) and the hand-over shows as a light bump. A request may still send its own `continuation_prompts` (one string per window). |
| Known limits | The model may insert a cut of its own inside a window (a seed-dependent sampling event, not a prompt error); the generated audio may fade over 30-60 s; motion calms down window by window. A clip with a cut is best redone with another `seed`. A driving soundtrack (`audio_reference`, `audio_mode: lock_source`) keeps the audio level flat by construction, see the table above. |
| Reference size | The server resizes the reference to a 1024 px short edge; a larger picture is fine. |

`clients/h3-generate.py --long --duration 30 --image picture.jpg "prompt"` in the workspace sends the same request
through an SSH tunnel.

## The model

MiniMax H3-Base FL2VA: DiT 62 GB BF16 (31 GB after online FP8), Qwen3-VL text encoder with 50 retained layers
(51.5 GB BF16), video VAE 9.8 GB, audio VAE. 24 fps, 4 to 15 s, native 768p. Weights: `MiniMaxAI/MiniMax-H3`, folder
`FL2VA`, identical path on both Sparks.

## Hardware

Two DGX Spark (GB10, 128 GB unified memory, DGX OS kernel 6.17.0-1032-nvidia, driver 580.178.04), one QSFP cable
between the top ConnectX-7 ports (both twins active: `rocep1s0f0` and `roceP2p1s0f0`, 200 Gb/s each). Management over
Wi-Fi or RJ-45 is fine; only NCCL and the rank-1 control channel use the fabric. Power both Sparks off before cabling: a
cable plugged into a running Spark brings the NIC back, but RDMA runs at ~13 Gb/s until the next reboot.

## Running it

Prerequisites on **both** Sparks, at identical paths: the single-Spark H3 runtime (vLLM-Omni at commit `e1aa6eae`, vLLM
0.26.0, Python 3.12 venv) and the FL2VA weights. Defaults are `~/h3-base-20260916/{runtime/.venv,weights/FL2VA}`;
override `H3_ROOT`, `H3_VENV`, `H3_MODEL`, the two fabric addresses (`HEAD_IP`, `WORKER_IP`), the netdev of rail 1
(`FABRIC_IF`) and the two HCAs (`IB_HCA`) in `~/.h3-tp2.env` (sourced by `scripts/env.sh`). Optional but recommended:
the same random string in `~/.h3-tp2-secret` on both Sparks authenticates the rank-1 control channel (see
docs/ARCHITECTURE.md, Trust boundary).

```bash
# on each Spark
git clone https://github.com/knapcio/MiniMax-H3-2x-DGX-Spark-TP2 ~/h3-tp2
~/h3-base-20260916/runtime/.venv/bin/pip install -e ~/h3-tp2
bash ~/h3-tp2/scripts/preflight.sh            # links ACTIVE, RoCE v2 GID + IPv4 per HCA, >=110 GiB free, venv/package/weights present; same versions on both
sudo -n /usr/local/sbin/h3-compact   # before every boot, on both hosts (root script + sudoers.d/h3-compact, 2026-11-02): boots into a fragmented host ran 7-40 % slower; scripts/queue-lib.sh restart_head does stop -> compact -> start -> boot probe (scripts/boot-probe.sh)

# worker Spark (rank 1)
cp ~/h3-tp2/systemd/h3-tp2-{agent,memguard-worker}.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now h3-tp2-agent.service h3-tp2-memguard-worker.service   # agent listens on WORKER_IP:18095, fabric only

# head Spark (rank 0, API on 127.0.0.1:18093)
cp ~/h3-tp2/systemd/h3-tp2-{head,memguard-head}.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now h3-tp2-head.service h3-tp2-memguard-head.service
journalctl --user -u h3-tp2-head.service -f     # ~7 min to health 200 with a warm page cache; rank-1 log in ~/h3-tp2-logs on the worker
bash ~/h3-tp2/scripts/status.sh                  # both units, health, MemAvailable on both nodes
```

First request after a boot pays lazy initialisation; the recipe warm-up is one 5 s, 2-step render:

```bash
bash ~/h3-tp2/scripts/render.sh ~/warmup.mp4 5 1344 768 2 1101 "a quiet valley at dawn"
bash ~/h3-tp2/scripts/render.sh ~/clip.mp4 10 1344 768 20 1101 "<prompt>" ~/still.jpg first   # FL2VA from a still
bash ~/h3-tp2/scripts/verify-output.sh ~/clip.mp4 10 1344 768                                   # streams, size, duration, decode, non-silent audio, framemd5; exit 1 on failure
bash ~/h3-tp2/scripts/bench-matrix.sh "5:1344:768:50:1101 10:1344:768:50:1101"                  # the table above, with memory sampling on both nodes; STATUS per case
```

`render.sh` disables curl's automatic configuration files so that a `.curlrc` cannot silently add destinations
or alter a multipart request. Settings previously supplied only there no longer apply. Proxy environment variables,
the configured API host, executable paths and HTTP transport remain the caller's responsibility; this is not an
egress sandbox. Frozen older clients do not receive this change automatically.

For a compatible server and adapter, `render.sh` accepts `H3_VIDEO_FLOW_SHIFT`, `H3_AUDIO_FLOW_SHIFT` and
`H3_LORA_JSON='{"name":"adapter-name","path":"/absolute/path/on/both/nodes","scale":1.0}'`. Its defaults remain
video/audio shifts 12/3 without an adapter. `EXTRA_JSON` can override the shifts; the timing record reports the
form shift and its value after that merge, and omits the adapter path. Authenticate the actual loaded adapter
separately. In particular, the current fused-MLP/LoRA interaction requires the checks documented in
[`results/20261101-lora-fusion-guard/`](results/20261101-lora-fusion-guard/README.md) before using Turbo.

The API is the stock vLLM-Omni one (`POST /v1/videos/sync`, `extra_params` with `task`, `duration`, `frame_indices`),
so existing clients only need the host. The memory guards kill the rank when MemAvailable stays under 4 GiB for two
samples: on GB10 the NVIDIA driver cannot fail an allocation gracefully and the whole host livelocks instead (that is
how we lost an afternoon to a 10 s render on one Spark). The head then fails closed and systemd restarts it; a stuck
request is bounded by `H3_TP2_REQUEST_TIMEOUT_S` (4 h by default) and the API by `VLLM_OMNI_VIDEO_SYNC_TIMEOUT`;
long-running requests need compatible limits at each layer as described above.

`python -m unittest discover -s tests` (any machine, no GPU) exercises the control channel and the agent's failure paths.

### Second runtime: vLLM-Omni v0.30.0 (long clips)

The same units can run the newer stack from a second directory; nothing in the first one is touched, and
`scripts/select-stack.sh` switches per host.

```bash
# on each Spark (the weights of the Ref2VA partition are 134 GiB; the FL2VA ones can be a symlink to the existing copy)
mkdir -p ~/h3-long && cd ~/h3-long
python3.12 -m venv .venv && .venv/bin/pip install -q --upgrade pip uv
.venv/bin/uv pip install --python .venv/bin/python "vllm==0.30.0" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
git clone https://github.com/vllm-project/vllm-omni && (cd vllm-omni && git checkout v0.30.0) && .venv/bin/uv pip install --python .venv/bin/python -e vllm-omni
.venv/bin/uv pip install --python .venv/bin/python --no-deps -e ~/h3-tp2
mkdir -p runtime/cache/{triton,flashinfer} weights && ln -s ../.venv runtime/.venv && ln -s ../vllm-omni runtime/vllm-omni && ln -s ~/h3-base-20260916/runtime/debs runtime/debs
ln -s ~/h3-base-20260916/weights/FL2VA weights/FL2VA
HF_HUB_DISABLE_XET=1 .venv/bin/hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/*" --local-dir ~/h3-long/weights
bash ~/h3-tp2/scripts/select-stack.sh long Ref2VA      # or: long FL2VA, or: base   (then restart the agent, then the head)
```

A 30 s clip with continuity:

```bash
TASK=ref2va EXTRA_JSON='{"long_video": true, "long_video_mode": "continuation"}' \
  bash ~/h3-tp2/scripts/render.sh ~/long.mp4 30 960 576 20 1101 "<prompt>" ~/reference.jpg
```

## Next

- Longer than 15 s is a model limit (duration range 4 to 15 s), not a memory one: with the cap lifted
  (`H3_TP2_MAX_OUTPUT_SECONDS=20`, experiment only) a 20 s 1344x768 clip rendered in 6740 s with 20 GB still free on
  both hosts and looked coherent to the last frame on a calm landscape prompt (`results/20261027-20s-out-of-spec/`).
  One sample outside the model's training range; the serving configuration keeps the 15 s cap.
- Measured all-to-all share of a denoising step at 20 s: 2.9 % including the permute copies, 2.1 % inside the
  collectives. That is the ceiling for `H3_TP2_A2A_OVERLAP`.
- Attention is 57 % of a denoising step at 10 s and 66 % at 15 s (fit of step time against sequence length:
  t ≈ 4.9e-4·S + 8.8e-9·S² s); an exact, faster attention kernel for SM121 is the only lever above 10 %. Of the exact
  kernels installed here cuDNN is already the fastest: vLLM's FlashAttention-2 is 3 % slower and torch's flash SDPA
  4-7 % slower on the per-rank shapes (`scripts/attn-bench.py`, `results/20261027-attn-bench/`).
- Ulysses sequence parallel issues three all-to-alls per attention layer (q, k, v) and one back, and the GPU idles
  during them: 2.9 % of a denoising step at 20 s (`H3_TP2_A2A_TIMING=1`). Hiding them behind the attention kernel on a
  second stream (`H3_TP2_A2A_OVERLAP=<groups>`, off, kept for reference) gives identical frames but is 4.5 % slower:
  NCCL kernels queue behind the attention kernel on this GPU instead of running beside it
  (`results/20261027-a2a-overlap/`). Closed.
- Phase timers (`h3_tp2 phase encode_prompt / diffuse / decode` in the head journal) for the 5 s 1344x768 20-step
  render: prompt encoding 0.2 s, denoising 598.9 s (30 s per step), VAE decode 42.1 s with the decoder split over both
  GPUs; the rest (audio decode, uint8 conversion, muxing) is about 2 s. Denoising is the whole game above 5 s.
- Regional `torch.compile` of the eager modulation and elementwise work (drop `--enforce-eager`) is 5-10 % of the
  non-attention time but changes floating-point association, so it stays out of a lossless recipe.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: identical frames, peak 75.2 -> 74.1 GB, but 3.3 % slower
  (VAE decode 42 -> 50 s). Off (`results/20261027-expandable-segments/`). Closed.

## Licences

Code: MIT (see LICENSE, NOTICE). vLLM-Omni and vLLM: Apache-2.0. MiniMax H3 weights and outputs: MiniMax H3 Community
License (see MODEL-LICENSE.md). Credits in CREDITS.md.
