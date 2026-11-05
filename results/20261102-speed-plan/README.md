# Lossless speed plan for the H3 pair (2026-11-02, from a read-only code and profile review)

Scope: MiniMax H3 on two DGX Sparks (GB10), vLLM-Omni v0.30.0 at `a8576ccb7`, our plugin, `results/` profiles and the
head journal of 2026-11-02. Nothing here is implemented yet; each item names the first hours of work and the proof.

## Where the time goes (1344x768, 277 frames = 82 tokens, M = 42848 rows per rank, S = 85 696, 28 heads x 128, 50 layers)

| part | arithmetic | per step | share |
|---|---|---|---|
| attention | 4·S²·128·28·50 = 5.26e15 FLOP at ~88 TFLOPS (cuDNN measured) | 60 s | 70 % |
| GEMMs | 1.65e15 FLOP at 164 TFLOPS (fp32-output GEMM) 10.1 s + scale pass 814 GB at ~235 GB/s 3.5 s | 13.6 s | 16 % |
| all-to-all (SP2) | 61 GB at 19.5 GB/s | 3.2 s | 4 % |
| copies, quant, misc | | ~3 s | |
| total | | ~80 s on a good boot | 86 s observed |

At 960x544 the same model gives 15.7 + 7.0 + 1.6 + 1.5 ≈ 26 s/step, matching the good boots (514-535 s per 20-step
window); the 566-619 s windows of 2026-11-02 08:00 / 09:04 were slow boots, not a different workload.

Corrections to our earlier picture: the "NVlabs fusions" are mostly already in this stack (QKV is one GEMM
`QKVParallelLinear` N=21504, `minimax_h3_transformer.py:430`; AdaLN modulate + gate-add are fused,
`rms_norm_indexed_scale_shift` / `indexed_gate_rms_norm_scale_shift` :940-983; RoPE is fused with qk-norm
`fused_qk_norm_rope` :670; SwiGLU is our level 1). The missing one is per-token FP8 quant fused into the producing
kernel (~1 %). The real 10-14 % per-GPU gap to the single-GPU port: our two-pass fp32 round trip (5-9 %) and the SP2
all-to-alls with their permute copies (5-8 %), which a single GPU does not pay. Encoders are 0.4-4 s per request today.

## Ranked by gain per engineering day

1. **Host-memory hygiene and a boot speed gate** (0.5 day; bad boots cost +10-20 %: 619 vs 514-562 s today, 120 vs 86
   s/step on 10-30). `H3_TP2_EMPTY_CACHE_BEFORE_DECODE=1` (`plugin.py:203`) hands the denoise pool back to the host
   on every request, so the next request reallocates from fragmented memory: a hypothesis to test. Work: sudoers rule
   + compaction as a pre-start step on both hosts (agent too); an automatic warm-up request at boot (the first request
   runs 6-7 % slow); a probe gate (attention at 43k rows within 5 % of 311 ms, else restart). Proof: 3 boots x 2
   requests, s/step per window, `buddyinfo`, the latents digest. Lossless by construction.
2. **Batched VAE tile decode on by default (`H3_TP2_VAE_STACK_TILES=2`)** (0.5-1 day): decode is 148 s per 30 s at
   960x544, -35 % = -52 s (-2.8 % of the render), ~-90 s on a 30 s 1344 render. Today's equivalence is 45 dB (below
   the 60 dB bar), so: offline A/B on saved latents with `allow_fp16_reduced_precision_reduction=False` and cuDNN
   deterministic; if > 60 dB or bit-exact, no eye check needed. The `=2` path has never run on a GPU: CUDA test first.
3. **All-to-all overlap with a high-priority comm stream** (1-2 days; 3-4 % at 1344, 5-6 % at 960). The earlier
   overlap was bit-identical but slower because NCCL queued behind cuDNN; its stream has default priority
   (`ulysses_overlap.py:90`). Try priority -1 + `NCCL_MAX_CTAS=2-4`. Separately `NCCL_NET_GDR_LEVEL=0` (`env.sh:25`)
   forces bounce-buffer copies with SM kernels: run `alltoall_perf` at 150-310 MB with GDR/C2C on. Fallback: an RDMA
   all-to-all without SM kernels (RoCEnante proxy). Proof: latents digest.
4. **Fused per-row/per-column FP8 GEMM epilogue** (2-4 days; -4.6 s/step = 5-6 % at 1344, ~9 % at 960): removes the
   fp32 output write, the scale pass (`fp8_gemm.py:62`) and the 4.9 GB fc1 temporary. Probe cuBLASLt outer-vector
   scaling on sm121, then a CUTLASS sm120 tile sweep with the vLLM epilogue order (scale_b, scale_a, bias as fma),
   SwiGLU in the epilogue (gate/up columns interleaved). Stock and two-pass already give identical latents, so a
   kernel without split-K is likely bit-exact: check bitwise on the 4 shapes, then the digest.
5. **Remove the Ulysses permute copies, one combined qkv all-to-all** (2 days; 1.3-2 %): four 615 MB copies per layer
   (`ulysses.py:143`, `:206`, the split view of v). Write the scale pass / `fused_qk_norm_rope` straight into a
   [dest, M, 3, 28, 128] send buffer; cuDNN reads strided q/k/v; the out-projection quant reads the returned tensor.
   Pure data movement: bit-identical. (`--ulysses-a2a-permute` will not help: NCCL load-store, single machine only.)
6. **Fuse per-token quant into the producing kernels** (2 days; 1-1.5 %): the two modulation outputs, the
   post-all-to-all output, the SwiGLU output (level 2; find why level 2 diverged in 1 of 2 runs first).
6b. **Fuse the scale pass into the GEMM consumers** (implemented 2026-11-03 as `H3_TP2_FP8_CONSUMER_FUSE=1`, opt-in,
   GPU proof pending; ~0.9 s/step = 1.1 % at 1344, 1.8 % at 960): the lossless half of item 4. See the section below.
7. **Small per-step waste** (0.5 day; < 0.2 % at 1344, ~0.5 % at 960; all bit-identical): boolean-mask indexing forces
   host syncs several times per step (`denoise_loop.py:255`, `:413`, `transformer:1491`), `torch.unique` (`:208`), the
   cancellation sync (`:455`); the RoPE table is rebuilt every step on CUDA (only the NPU path caches it,
   `transformer:1611-1620`); the text token refiner reruns every step on ~602 constant rows (`:1515`); the AdaLN
   projections reread ~13 GB of FP8 weights per step (~57 ms). Fix: precomputed index tensors, cached refiner output
   and RoPE table per branch.

## Outside the lossless bar (owner's call)

- cuDNN 9.27: -3.5 % denoising measured (~9 % faster attention kernel, ~-6 % per step at 1344) but a different sample
  (SSIM 0.85): the same class of change as the two-pass GEMM at 960 Ref2VA that the owner accepted.
- FlashInfer FA2: -4.7 % on the kernel, ~7 GiB per rank of workspace.

## Not worth doing

CUDA graphs (the GPU is the bottleneck; ~1250 launches per step cost milliseconds; "Command Buffer Full" in the profile
means the launch queue is full), regional `torch.compile` (cannot fuse across Triton / cuDNN / NCCL kernels, ≤ 1 %,
changes fp association), lossless attention savings (cuDNN is within ~5 % of the GB10 flash ceiling). Rank 1's GEMMs
were 15 % slower in the 10-29 profile: recheck after item 1 before any uneven sequence split.

## Measured 2026-11-02 12:40-13:28: item 1, the empty-cache knob and boot-to-boot variance (queue47)

Five 5 s 1344x768 T2VA calibrations (seed 0, 20 steps; reference `cmp-t2va-1344` 23.2 s/step), all with the same
latents digest `4d6d69e4aad0`:

| boot | `H3_TP2_EMPTY_CACHE_BEFORE_DECODE` | request | s/step | diffuse | decode | peak | wall |
|---|---|---|---|---|---|---|---|
| 11:42 (end of queue45) | 1 | 1st after two 30 s renders | 22.92 | 458.4 s | 42.0 s | 63.0 GB | 502 s |
| 11:42 | 1 | 2nd | 22.78 | 455.7 s | 42.0 s | 63.1 GB | 499 s |
| 12:57 (fresh) | 0 | 1st | 21.44 | 428.8 s | 43.3 s | 63.2 GB | 477 s |
| 12:57 | 0 | 2nd | 21.24 | 424.9 s | 41.9 s | 63.2 GB | 468 s |
| 12:57 | 0 | 3rd | 21.21 | ~424 s | 42.0 s | 63.2 GB | 468 s |

The knob is inert: with `=1` the plugin logs "upstream releases the stage cache before decode itself; empty_cache
patch skipped" (vLLM-Omni v0.30 does the release itself), with `=0` the code path is not entered; the decode time and
the peak are the same either way. What the table does show is **boot-to-boot variance of 7 %** between two boots 75
minutes apart on the same hosts, same code, same request (the 10-30 anomaly was 40 %). The second and third requests
on a boot are not slower than the first (no per-request degradation). So item 1 stands as "boot hygiene + a boot
speed gate", not as an env change: every boot should be probed (a 5 s calibration or a 43k-row attention probe)
before a long render, and a slow boot restarted. The README's "free cached pages before decode" lever is
superseded by upstream and marked so.

## Boot gate in use (queue49, 15:05-15:42): the host degrades with restarts, compaction brings it back

`scripts/boot-probe.sh` (5 s 960x576 T2VA, seed 0, first request of the boot) after each restart of the same code:

| boot | host state before | probe | latents digest |
|---|---|---|---|
| 15:05 (8th restart of the day) | 06: 1.7 GB free, 77 GB page cache, orders 8+ `55 13 20 16 2 0 0`; 05 no order-10+ blocks | 9.17 s/step (SLOW at the 5 % limit of the first version) | edea0733 |
| 15:16 | same, one more restart | **9.80** | 5244e25b |
| 15:31, after `drop_caches` + `compact_memory` on both hosts with the head stopped | 06: 125 GB free, orders 9-15 `21106 15244 10431 5439 2083 687 302` | **9.14** PASS (limit 9.5) | d76aabd8 |

Warm reference 8.65 (second request of a good boot); the first request carries 4-7 % of warm-up, hence the 10 %
tolerance. Two findings: (1) the hosts degrade over a day of restarts (page cache full, no large free blocks) and each
boot into that state is slower than the last; a compaction with the services down restores the fast state, so it
belongs before every boot that precedes a long render (sudo from the workstation for now); (2) the first request of
a boot gives a different latents digest on every boot (three boots, three digests at 960x576), while requests within
one boot repeat the digest exactly (queue47: 2 + 3 requests). Lossless A/B therefore means same-boot digests; the
2026-10-29 "bit-stable across boots at 960" held for the boots it compared and does not generalise.

## Measured 2026-11-02 18:21-19:20: items 3 and 5 (queue50), 5 s 1344x768 T2VA seed 0, one boot per arm

| arm | s/step | latents vs stock |
|---|---|---|
| S1 stock (the 60 s render's boot, warm) | 21.06 | - |
| N `H3_TP2_ULYSSES_NOCOPY=1` | 21.22 | **rel-l2 0.199, every element differs** (video and audio) |
| S2 stock, fresh boot | 21.32 | **bit-equal to S1** |
| P `H3_TP2_A2A_OVERLAP=2` + `H3_TP2_A2A_STREAM_PRIORITY=-1` + `NCCL_MAX_CTAS=4` | 21.27 | **bit-equal to S2** |

- **Item 5 (no-copy exchange): closed.** The GPU result is wrong (a 20 % relative difference is a bug, not reordering:
  cuDNN's reading of the strided packed-QKV views or the per-destination rope launches), and even so there was no
  speedup; the copies it removes are overlapped or cheap at this cell. The code stays behind `=force` for a future
  debug session with per-layer q/k/v comparison.
- **Item 3 (comm stream priority): no effect** at 1344 5 s: bit-equal and 21.27 vs 21.32 s/step. The all-to-all is
  not the bottleneck here; the overlap's earlier "slower" result stands. Closed unless a profile shows the a2a waiting.
- **Cross-boot reproducibility at 1344:** two stock boots gave bit-equal latents (and queue47's five requests on two
  boots all `4d6d69e4`), so the per-boot digest variance seen at 960x576 (three boots, three digests) is specific to
  that cell; at 1344 same-request digests can be compared across boots.
- Boot-to-boot speed on compacted hosts: 21.06 / 21.32 / 21.27 (1.2 % spread) against 22.9 on the 11:42 boot and 120
  s/step on 10-30: the compaction + probe gate is the lever that matters.

Remaining: item 2 (batched VAE decode default-on, offline A/B next), item 4 (fused FP8 epilogue, 2-4 days), items 6-7.

## Measured 2026-11-02 19:36-19:50: item 2, the batched tile decode, offline with the head stopped (`item2c`)

Same saved latents (window 3 of the 30 s 960x544 render, 67 tokens -> 226 frames) decoded through the serving adapter
path (`MiniMaxH3VideoVAE.decode_latent`, FP16 autocast, streaming uint8 fork) on one GPU, twice per arm (second time):

| arm | decode | peak alloc | frames vs stock |
|---|---|---|---|
| stock | 81.2 s | 11.5 GB | - |
| `vae_tile_scope` cap 4 (= `H3_TP2_VAE_STACK_TILES=2`, `H3_TP2_VAE_TILE_BATCH=4`) | **50.0 s (-38 %)** | 11.6 GB | max 1 gray level, 0.27 % of pixels, **PSNR 73.9 dB** |
| stock + deterministic flags | 82.4 s | 11.5 GB | bit-equal to stock |
| cap 4 + deterministic flags | 49.9 s | 11.6 GB | bit-equal to cap 4 |

The batched tiles change nothing but the reduction order inside the tile batch (1-LSB rounding on 0.27 % of pixels,
73.9 dB; the 45 dB of 10-29 was a different measurement); the deterministic cuDNN/cuBLAS flags change nothing at all
here. Two earlier attempts of this A/B were invalid: the direct chunk-loop script bypasses the adapter hook (so
"stack4" was stock), and next to the idle head three of four arms hit CUDA out-of-memory with 44 GB free (host
fragmentation after a boot): offline GPU jobs on the head only with the head stopped and the hosts compacted.
Serving check next (queue51): 960x576 stock vs `=2` on two ranks, then a 16 s 1344 continuation; then default-on.

### Serving check (queue51, 19:57-): two ranks, tile-parallel gather

| request | decode phase | note |
|---|---|---|
| A: 5 s 960x576 T2VA, stock | 23.9 s | |
| B: same request, `H3_TP2_VAE_STACK_TILES=2` + `VAE_TILE_BATCH=4` (new boot) | **18.0 s (-25 %)** | no error on two ranks |

The MP4-to-MP4 PSNR of A and B (32.3 dB luma) says nothing about the decoder: the two requests ran on different boots
and at 960x576 the first request's LATENTS differ per boot (rel-l2 0.09 between A's and B's saved latents, every
element), so the frames differ for diffusion reasons. The decoder equivalence is the offline result above (73.9 dB on
identical latents); a 1344 offline check (`item2d`, stock vs cap 4 on the saved 1344 window-1 latents, head stopped)
follows the 16 s 1344 continuation (C) of this queue.
| C: 16 s 1344x768 FL2VA continuation (two windows, pinned prefix, lock defaults) with `=2` | decode_to_mp4 86.6 s; 2340 s total | seam +0.0 luma, no cut, no drop >= 1 YAVG, block boundaries -0.31 / -0.39: the same profile as the stock 1344 renders; two-rank tile gather + batched tiles work at the larger tile count |

### 1344x768 offline (`item2d`, head stopped): stock 186.8 / 185.9 s vs cap 4 **113.9 / 114.0 s (-39 %)** per 277
frames, peak 13.4 GB both, frames max 1 gray level on 0.22 % of pixels, **PSNR 74.7 dB**. **Default since 2026-11-02
21:05:** `H3_TP2_VAE_STACK_TILES=2` (cap 4). Item 2 closed.

### Serving on the defaults (queue53, 21:48-22:07, boot 7942a3c through the gate)

5 s 1344x768 T2VA seed 0 on the new boot: **22.03 s/step**, latents `4d6d69e4aad0` (unchanged), **decode 26.3 s**
against 41.8-42.0 s with the stock per-tile decode on the same request (-37 % in serving at 1344). Boot probes on this
boot: 9.35 (first request) then 8.96 / 8.99 (warm), digests e20e93bd then bf6e8568 twice, the latter being the digest
the previous boot's later probes had: at 960x576 the FIRST request of a boot is the odd one (speed and numerics), later
requests repeat across boots. The previous boot read 9.19 on its first probe and 10.5-10.7 afterwards (1344 calibration
23.74), so the boot gate now runs two probes and judges the warm one (`PROBES=2`, 6 % over 8.65).

## Measured 2026-11-02 22:08-22:13: item 4, the fused FP8 epilogue (`scripts/fp8_epilogue_probe.py`, head stopped)

Per-step projections over 50 layers from the four GEMM shapes (random FP8 inputs, best config per family):

| M (rows per rank) | a) serving: two-pass + fused SwiGLU level 1 | c) Triton in-GEMM epilogue (`H3_TP2_FP8_GEMM=fused`) | d) cuBLASLt outer-vector scaling | ceiling: scalar bf16 cuBLASLt, no scales |
|---|---|---|---|---|
| 21952 (960x544/576) | 6.75 s/step | 9.82 (**+3.07**) | **not supported on sm121** (`cublasLtMatmulAlgoGetHeuristic` status 15) | 4.46 (-2.29) |
| 42848 (1344x768) | 13.09 s/step | 16.07 (**+2.98**) | not supported | 8.62 (-4.47) |

Kernel rates at M=21952, fc2: two-pass 150 TF (GEMM 175 + scale pass), Triton fused 86-90 TF, stock CUTLASS row-wise
61 TF, torch row-wise `_scaled_mm` 61 TF (<= 1 ulp from two-pass), scalar bf16 ceiling 182 TF. The Triton kernel's
epilogue math is exact (bit-identical on integer inputs) but its accumulator and speed are not cuBLASLt's: on GB10
Triton reaches half of cuBLASLt. **Item 4 closed:** the two-pass path stays; the -4.5 s/step of the ceiling is only
reachable with a CUTLASS sm120 FP8 kernel carrying the per-row/per-column epilogue at > 120 TF, which does not exist
today (vLLM's CUTLASS row-wise kernel runs 61 TF here). `H3_TP2_FP8_GEMM=fused` stays in the tree as a measured
negative. Probe log and JSON: `~/h3-tp2-results/fp8-epilogue-probe/` on spark-06.

## End state (23:50): 30 s 1344x768 from a still with every default: 4648 s against 4964 s at 13:42 (-6.4 %), same quality
profile (seams -0.0, no drops, boundaries -0.40 / -0.62). The day's lossless gains in one number for the owner's cell:
a 30 s clip in 77 min instead of 83, with the boot gate protecting against the 7-40 % slow boots.

## Item 6b (2026-11-03): the scale pass fused into the consumers of qkv / out / fc2 (`h3_tp2/fp8_consumers.py`)

Item 4 closed the in-GEMM epilogue; what is left lossless is the bf16 round trip AFTER the scale pass. For three of the
four GEMMs the pass's bf16 output (`fp8_gemm.scaled_mm_tensor32`, `y = torch.empty`, written by `_scale_bias_cast`)
is read by exactly one upstream kernel with no copy in between: q and k (`minimax_h3_transformer.py:655-666`) by
`fused_qk_norm_rope`, the out projection (`:703` -> `:970`) by `indexed_gate_rms_norm_scale_shift`, fc2 (`:741` ->
`:982`) by `indexed_gate`; v (`:661`) is read by the attention as a strided view of the qkv tensor. With
`H3_TP2_FP8_CONSUMER_FUSE=1` the dispatcher keeps those fp32 products and copies of the three upstream Triton kernels
read them: each loaded element gets the pass's arithmetic (scale_b, then scale_a, bias as one fma) and its single RNE
rounding to bf16, then the consumer's own code runs unchanged on that value. v's columns are still scale-passed into
the qkv tensor. Bytes saved = 4 B (bf16 write + read) per element of q, k, out and fc2:

| GEMM | columns saved | per layer at M = 42848 | per step (50 layers) at 235 GB/s | at M = 21952 |
|---|---|---|---|---|
| qkv (N = 21504; q + k, not v) | 14336 | 2.46 GB = 10.5 ms | 0.52 s | 0.27 s |
| out (N = 5376) | 5376 | 0.92 GB = 3.9 ms | 0.20 s | 0.10 s |
| fc2 (N = 5376) | 5376 | 0.92 GB = 3.9 ms | 0.20 s | 0.10 s |
| total | | 4.30 GB = 18.3 ms | **0.92 s/step (215 GB)**, ~1.1 % of 80 s | 0.47 s/step, ~1.8 % of 26 s |

The fp32 read stays (only an in-GEMM epilogue removes it). Memory: the fp32 product now lives while the consumer
writes its outputs: +1.2 GB transient on qkv at M = 42848 (6.8 GB against 5.5), below fc1's peak; out and fc2 reuse
the placeholder for their output. Bit-identity argument: same operands, same operation order, the bf16 rounding at the
same point; on the GPU additionally the same compiled consumer arithmetic (fma contraction, `tl.sum` tree), which the
copies keep by keeping upstream's launch shapes and a bf16 access per kernel; every site self-checks bit for bit on its
first call (first 64 rows, stock pair = scale pass + upstream kernel) and disables itself on a mismatch. CPU: 48 tests,
`torch.equal` (`tests/test_fp8_consumer_fuse.py`, upstream's DiT block/attention/MLP forwards verbatim). Next: on
Spark_06 with the head stopped `~/h3-long/.venv/bin/python scripts/fp8_consumer_probe.py --json
~/h3-tp2-results/fp8-consumer-probe/probe.json` (timing per site, bits on random and directed data, registers, peak),
then a same-boot latents digest A/B at 5 s 1344x768 T2VA.

### Item 6b measured 2026-11-03 00:36-01:35 (`item6b`): GPU probe with the head stopped, then serving

| site | M | old: scale pass + consumer | fused | saving per step (50 layers) | bit-identical | regs / spills |
|---|---|---|---|---|---|---|
| qkv (q, k) | 21952 | 19.29 ms (13.15 + 6.13) | 13.29 ms | +0.30 s | yes | 38 / 0 |
| out | 42848 | 14.40 ms (6.41 + 8.11) | 10.22 ms | +0.21 s | yes | 64 / 0 |
| fc2 | 42848 | 12.58 ms (6.41 + 6.26) | 8.12 ms | +0.22 s | yes | 62 / 0 |
| all sites | 42848 | | | **+1.04 s/step** | | |
| all sites | 21952 | | | **+0.52 s/step** | | |

Extra peak over the product drops (e.g. out 1.38 -> 0.92 GB). Serving, 5 s 1344 T2VA seed 0 through the gate:
default env (same code) 22.43 s/step digest `4d6d69e4` (the stash-hook list refactor leaves the default path exact);
`H3_TP2_FP8_CONSUMER_FUSE=1` 21.53 and 22.10 s/step, digest `4d6d69e4` (bit-identical); the 960 warm probe digest is
also the stock one (bf6e8568). **Default since 2026-11-03 01:35.** The remaining scale-pass cost is fc1's (already
fused with SwiGLU) and v's column scaling.

## Serving state 2026-11-03 02:06 (queue55): all defaults of the two days, boot through the compacting two-probe gate
(warm probe 9.00 s/step, digest bf6e8568): 5 s 1344x768 T2VA **21.98 s/step, digest `4d6d69e4` unchanged, decode 26.4 s**
(10-30 reference 23.2 s/step and 42 s decode). Levers in production: two-pass FP8 GEMM + fused SwiGLU level 1 +
consumer-fused scale pass, batched tile decode, class-aware/tail/ramp-1.0 look-lock, boot gate with host compaction.

## 2026-11-03 09:15: item 4 reopened with a CuTe-DSL kernel, see `results/20261103-cutlass-sm120/`

A generated sm120 FP8 kernel carries the scale epilogue at 157-168 TF, bit-identical to the two-pass output; the stock
CUTLASS kernels' 61 TF was the tile raster (no L2 swizzle). Worth ~1.5 s/step at 1344 against today's path.

## 2026-11-05: item 6 built as `H3_TP2_FP8_QUANT_FUSE=1` (opt-in, branch quant-fuse), see `results/20261104-quant-fuse/`

The per-token quant of all four DiT GEMM inputs (qkv: modulation 1, out: the Ulysses reverse unpack, fc1:
modulation 2, fc2: SwiGLU) moves into the producing kernel, bit-identical on every GPU check (fp8 bytes and scales,
exhaustive over the row maximum, whole block): -0.30 / -0.55 / -1.26 s/step at M = 10240 / 18880 / 42848 by kernel
time. Level 2's divergent render was the first request of a boot (the cuBLASLt stale-scale hazard, fixed 11-03); its
quant matches the serving op. Serving proof pending.
