# 2026-11-03: CuTe-DSL sm120 FP8 GEMM with the fused per-row/per-column epilogue on GB10 (item 4 reopened)

Question: can a CUTLASS sm120 FP8 GEMM carrying the scale epilogue (y = bf16_rn(sa[m] * (sb[n] * acc)), the order of
`h3_tp2/fp8_gemm.py`; the four DiT GEMMs have no bias) run at > 120 TF on GB10, against cuBLASLt's ~175-190 TF?

**Answer: yes. 157-168 TF with the epilogue, bit-identical to the two-pass output on all four shapes at both M.** The raw
fp32 accumulator is bit-identical to cuBLASLt's too (same K order and MMA rounding, no split-K), so the fused output
is bit-identical by construction, not by luck. The 61 TF of the stock CUTLASS/torch row-wise kernels comes from the tile
raster (no L2 swizzle), not from sm120's MMA or the DSL. Against today's serving path (item 6b consumer fusion) the kernel
is worth an estimated **~1.5 s/step at 1344x768 (~1.9 %) and ~0.7 s/step at 960x544 (~2.7 %)** when used for qkv, out and
fc1, roughly twice that for fc1 once SwiGLU moves into the epilogue (see the projection). fc2 should stay on cuBLASLt.

Spark_06, GB10 sm_121, 48 SMs, 24 MB L2, 99 KB smem per block, torch 2.13.0+cu130, nvidia-cutlass-dsl 4.7.1, cuBLASLt
13.1. The H3 serving head stayed up and idle the whole time (58.7 GB held); peak probe memory ~10 GB. Clocks untouched.
Timing: min of 5 CUDA-event runs after 2 warm-ups (the style of `scripts/fp8_epilogue_probe.py`, same inputs: per-token
/ per-channel quantised bf16 data with log-normal row magnitudes, seed per shape). The cuBLASLt/two-pass rows match the
11-02 head-stopped probe within 1 % (qkv fp32 GEMM 58.05 vs 58.01 ms at M=42848), so the idle head did not skew them.

Files: `full.jsonl` / `full.log` (the measurement below), `sweep4.jsonl` (tile/swizzle sweep at M=8192, after all fixes).
Scripts: `scripts/cutlass_sm120_fp8_probe.py` (harness), `scripts/cutlass_sm120_make_kernel.py` (derives the kernel from
the CUTLASS example by exact-anchor substitutions), `scripts/cutlass_sm120_dsl_fp8_gemm.py` (the generated kernel, NVIDIA's
BSD-3-Clause header kept). Reproduce: `~/h3-long/.venv/bin/python scripts/cutlass_sm120_fp8_probe.py --m 21952 42848
--cfgs "128,128,64,2,2,1,8,0,1" --jsonl out.jsonl` (cfg = BM,BN,BK, MMA warps M,N,K, swizzle, raster_along_m, epilogue
smem stages).

## Measured (ms per GEMM; TF of the fused kernel; config 128x128x64, 2x2 MMA warps + 1 TMA warp, swizzle 8 along M)

| M | GEMM (N, K) | ceiling: cuBLASLt bf16, no scales | cuBLASLt fp32 out | two-pass (fp32 + scale pass) | DSL, no scales | **DSL + scaled epilogue** | TF | vs two-pass | DSL acc vs cuBLASLt acc |
|---|---|---|---|---|---|---|---|---|---|
| 42848 | qkv (21504, 5376) | 51.31 | 58.05 | 85.85 | 58.92 | **59.08** | 167.7 | bit-identical | bit-identical (at 21952) |
| 42848 | out (5376, 7168) | 17.48 | 18.74 | 24.95 | 19.71 | **19.61** | 168.4 | bit-identical | " |
| 42848 | fc1 (28672, 5376) | 68.51 | 77.88 | 114.63 | 77.25 | **78.92** | 167.4 | bit-identical | " |
| 42848 | fc2 (5376, 14336) | 35.29 | 35.77 | 42.04 | 41.46 | **41.31** | 159.9 | bit-identical | " |
| 21952 | qkv | 26.63 | 29.92 | 44.56 | 30.59 | **31.26** (4x2 warps: 30.87) | 162.4 (164.4) | bit-identical | bit-identical |
| 21952 | out | 8.72 | 9.42 | 12.84 | 10.09 | **10.27** | 164.8 | bit-identical | bit-identical |
| 21952 | fc1 | 35.10 | 40.02 | 58.91 | 40.35 | **40.82** | 165.8 | bit-identical | bit-identical |
| 21952 | fc2 | 17.84 | 18.53 | 21.80 | 21.37 | **21.56** | 157.0 | bit-identical | bit-identical |

"Bit-identical" = every bf16 element equal to `fp8_gemm.scaled_mm_tensor32` (the DSL output buffer is NaN-filled first,
so an unwritten element cannot pass as stale data). The DSL no-scales column is also bit-identical to the cuBLASLt bf16
ceiling. The epilogue itself costs 0-3 % (two global scale loads per accumulator, L1-resident). The 4x2-warp variant is
within noise of 2x2. Kernel launch compile: 0.3-0.7 s per variant (dynamic layouts: one compile serves every M, N, K).

Per step (50 layers, the four GEMMs): ceiling 8.63 s (1344) / 4.41 s (960); DSL fused, fc1 still to bf16 without
SwiGLU, 9.95 / 5.20 s; the 11-02 two-pass column 13.09 / 6.75 s.

### What cuBLASLt runs, and why the stock CUTLASS kernels reach 61 TF

`torch._scaled_mm` scalar: `nvjet_sm121_qqsss_mma_192x176x128_2_48x88x128_tmaAB_bz_TNNN` (fp32 out) / `..._qqtst_...`
(bf16 out). Row-wise `torch._scaled_mm` and vLLM's `cutlass_scaled_mm`: CUTLASS
`MainloopSm120TmaWarpSpecialized<2 stages>`, `KernelTmaWarpSpecializedCooperativeSm120`, tile 128x128x128,
`SM120_16x8x32_TN` e4m3, EVT epilogue (ColBroadcast x RowBroadcast). Both run 21.3-21.5 ms at 8192x5376x14336
(59 TF). That is exactly a DRAM-bound raster: without an L2 swizzle each of the 64 row tiles re-reads the whole of B
(77 MB, more than the 24 MB L2), 64 x 77 MB = 4.9 GB = 21 ms at 235 GB/s. The same happens to CUTLASS's own bf16 DSL
example: 22.95 ms with the default scheduler, 7.38 ms (85 TF, cuBLAS 90 TF) with `swizzle_size=8, raster_along_m=False`.
Swizzle 8 along M is also the best FP8 setting here (4: -10 to -20 %; 16: fine on qkv, -28 % on fc2). Untested
corollary: vLLM's C++ kernel would probably recover most of its gap with the scheduler argument `max_swizzle_size=8`.

## What it took (four defects in the CUTLASS 4.7.1 sm120 DSL example path, all fixed in the generator)

1. **FP8 MMA**: the example only does fp16/bf16 (`warp.MmaF16BF16Op`); FP8 uses `warp.MmaFP8Op(e4m3, f32, (16,8,32))`
   with the sm120 8-bit N permutation `(8, warpsN, 2):(1, 16, 8)` (as `utils.blackwell_helpers.get_permutation_mnk` and
   the C++ sm120 builder). The example's plain N tiling corrupts columns 16-31 of every 32.
2. **Epilogue**: the example's TMA-store epilogue slices the accumulator by flat index per 64x32 epilogue tile. With the
   fp8 C layout at 128x128 tiles that scrambles fp32 output and corrupts 2-9 % of bf16 values (exact only at 64x64). It was
   replaced by a direct register-to-global store through the MMA's own C partition (`thr_mma.partition_C`), with
   element predication on edge tiles. That also frees the 32 KB of epilogue smem: 5 mainloop stages instead of 2.
3. **Pipeline race**: the example releases each smem stage with lane 0 of each MMA warp and no barrier. CUTLASS's C++
   sm120 mainloop (`sm120_mma_tma.hpp`) does `NamedBarrier::sync(thr_size(tiled_mma))` before every `consumer_release`.
   Without it, wrong tiles (different on every run) appear as soon as more than 2 stages let the producer refill early
   (ab_stage 4/5/11). The example never saw this because its 8 epilogue stages left room for only 2. Added; 4/4 runs
   bit-identical at 5 and 11 stages since.
4. **Swizzled scheduler pads the grid**: `PersistentTileSchedulerParams` rounds the swizzled dimension up to a multiple
   of the swizzle (M=21952: 172 row tiles -> 176). The padded tiles are real work items, so the store predicates M and N
   and the scale vectors are padded to the rounded grid. Cost: 2.3 % wasted MMA work at M=21952, 0.3 % at 42848.

Not resolved (not needed): 8-MMA-warp tiles 128x256 / 256x128 run at 2 TF (correct output). This is probably `setmaxnreg`
in a 9-warp CTA, where the TMA warp sits alone in a partial warpgroup. fc2 (K=14336, N=5376) reaches 157-160 TF against
cuBLASLt's 184; the gap is in the mainloop (the no-scales DSL variant is equally slow), not in the epilogue.

## DSL coverage on sm120 / sm121 (answers to the side questions)

- Plain FP8 (no block scales): `cute.nvgpu.warp.MmaFP8Op` (mma.sync m16n8k32 / m16n8k16, e4m3 / e5m2, f16 / f32 acc).
  There is no FP8 GEMM *example* for GeForce Blackwell; the only dense sm120 example
  (`examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py`) is fp16/bf16 and needs the four
  fixes above for FP8. sm120 has no tcgen05/TMEM and no wgmma; it uses TMA plus warp-level mma.sync with ldmatrix.
- Block-scaled: `MmaMXF8Op` (kind::mxf8f6f4 block_scale, UE8M0 per 32 K-elements), `MmaMXF4Op`, `MmaMXF4NVF4Op`
  (E4M3 per 16), `MmaMXF8F6F4Op` (mixed) for `sm_120a/f, sm_121a/f`, with examples
  `blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_{cooperative,pingpong}.py` (clusters
  fixed at 1x1). MXF8's power-of-two per-32-K scales are not our per-token / per-channel fp32 scales, so that path is
  not lossless for H3 and was not pursued.
- C++: `examples/79_blackwell_geforce_gemm` (nvfp4, mxfp8xmxfp6) and `87_blackwell_geforce_gemm_blockwise` (fp8 blockwise /
  groupwise) in the CUTLASS 4.7.1 tree on Spark_06 (`~/spark-ds41f-r6x-correctness/.work/build/vllm-.../src/cutlass`).

## Projection against today's serving path and the wiring plan (`H3_TP2_FP8_GEMM=cutlass`, opt-in)

Since item 6b, qkv / out / fc2 keep cuBLASLt's fp32 product and their consumers apply the scales (1.04 / 0.52 s/step
saved); fc1 is fp32 GEMM + fused scale/SwiGLU (`a_swiglu`, 108.61 / 55.29 ms). Serving-equivalent per layer = two-pass
minus the 6b saving per site (qkv 12.2 / 6.0 ms, out 4.2 / 2.2, fc2 4.4 / 2.2). The silu_and_mul pass after a bf16 fc1 is
**estimated** from bandwidth (read 2 x M x 14336 bf16 + write half: 15.7 / 8.0 ms), not measured:

| GEMM | 1344 (M=42848): serving -> DSL | per step | 960 (M=21952): serving -> DSL | per step |
|---|---|---|---|---|
| qkv (bf16 out, stock consumers, v scale pass gone) | 73.6 -> 59.1 ms | **-0.73 s** | 38.6 -> 30.9 ms | **-0.37 s** |
| out | 20.8 -> 19.6 | -0.06 s | 10.6 -> 10.3 | -0.02 s |
| fc1, bf16 out + silu_and_mul (est.) | 108.6 -> ~94.6 | ~-0.70 s | 55.3 -> ~48.8 | ~-0.32 s |
| fc1, SwiGLU in the epilogue (est., not built) | 108.6 -> ~80 | ~-1.4 s | 55.3 -> ~41.5 | ~-0.68 s |
| fc2 | 37.6 -> 41.3 (slower) | keep serving | 19.6 -> 21.6 (slower) | keep serving |
| **total, qkv + out + fc1** | | **~-1.5 s (~1.9 %)**, ~-2.2 s with SwiGLU fused | | **~-0.7 s (~2.7 %)**, ~-1.1 s fused |

Memory: no fp32 temporaries for those GEMMs. The fc1 product (4.9 GB at M=42848) and qkv's (3.7 GB) go away, so the
per-layer peak drops by ~4.9 GB at 1344.

Plan (about one day plus the GPU proof):
1. Vendor the generated kernel as `h3_tp2/fp8_cutlass.py` (keep NVIDIA's BSD-3 header, add CUTLASS to CREDITS.md). Import
   `cutlass` lazily; compile the scaled bf16 variant once at install (~0.5 s; dynamic layouts cover every M/N/K). The
   install check copies `probe_fused`: integer inputs plus random inputs including a partial row tile (e.g. M=200,
   M=4100), bitwise against `scaled_mm_tensor32`. Fall back to two-pass on any mismatch or import failure.
2. Kernel touch-ups before serving: bound the `sa` load by M in the kernel instead of the host-side padded copy, and
   either clip padded swizzle tiles in a custom scheduler or skip empty tiles before the mainloop (+~2 % at 960). Add
   the SwiGLU epilogue for fc1, either with interleaved gate/up columns or with each CTA computing the gate and up tiles
   of the same columns (two accumulators: check registers at 128x64 per half).
3. Dispatcher: in `install()`, level "cutlass" routes qkv, out and fc1 to the DSL kernel ahead of the stash hooks.
   fc2 keeps the 6b consumer-fused cuBLASLt path. The qkv consumers then read bf16 through their stock kernels, which is
   bit-identical because 6b's consumers round at the same point. Before relying on that for fc1, check that bf16 fc1
   output + `silu_and_mul` equals fused-SwiGLU level 1 bit for bit.
4. Measure the host cost per call: `from_dlpack` for 5 tensors plus the compiled launcher. If it is above ~50 us,
   cache the cute tensors per (data_ptr, shape) or use the TVM-FFI entry. 150 GEMMs/step make this small but nonzero.
5. Proof: same-boot latents digest A/B, 5 s 1344x768 T2VA (expect `4d6d69e4` unchanged) and the 960 warm probe
   (`bf6e8568`), then the s/step gain.

## Serving proof (`H3_TP2_FP8_GEMM=cutlass`, opt-in; wiring plan steps 1-5 done)

### What was built

- `h3_tp2/fp8_cutlass_kernel.py`: the kernel, vendored from the generated `scripts/cutlass_sm120_dsl_fp8_gemm.py` with
  NVIDIA's BSD-3-Clause header (the example's command-line driver dropped, the diagnosis env knobs fixed to their
  correct values). Kernel touch-ups of plan step 2: the `sa` / `sb` loads are bounded by M / N in the kernel (an edge row
  or column reads the last valid scale; its store is predicated off), so no host-side padded scale copy is made; tiles
  past M or N (the swizzled scheduler rounds the grid up, 172 -> 176 row tiles at M = 21952) are skipped by the producer
  warp and the MMA warps alike before the mainloop. The SwiGLU epilogue for fc1 (optional item) was not built.
- `h3_tp2/fp8_cutlass.py`: lazy `import cutlass`; one compile per process at install (0.9-1.0 s, dynamic layouts, one
  compile for every M / N / K) and a bitwise install check against `fp8_gemm.scaled_mm_tensor32` (NaN-filled output):
  integer inputs (200x400x512, 333x272x1024) and random inputs with partial row tiles (M = 200 and 4100 on the out,
  qkv and fc1 shapes); 0.2 s. Any import / compile / check failure logs one line and the process stays on two-pass.
- Dispatcher (`fp8_gemm.install(level="cutlass")`): a router ahead of the stash hooks takes the GEMMs whose weight is
  the running DiT block's `attn.qkv_proj`, `attn.out_proj` or `mlp.fc1` weight (block context set by a wrapper of
  `MiniMaxH3DiTBlock.forward`, `functools.wraps` so the consumer fuse's source fingerprints are unchanged); a call it
  cannot take (fp16 output, a bias, a misaligned pointer, M = 0) falls through to the ordinary dispatch, a launch
  exception switches the level off for the process. At this level `fp8_consumers` installs only its fc2 site (qkv / out
  go to the stock `fused_qk_norm_rope` / `indexed_gate_rms_norm_scale_shift` on the bf16 output), and fc1's bf16 output
  goes to the MLP's own `silu_and_mul` (fp8_swiglu's stash hook is never reached for fc1). Every other FP8 GEMM keeps
  two-pass. The plugin's diagnostics line now shows `FP8 GEMM requested=... running=...`.

### Kernel-level checks on Spark_06 (serving head up and idle; `scripts/fp8_cutlass_check.py`, `wiring-check.jsonl`:
the first host line is from the build with the weight-side cache, the last from the final build)

Per-token / per-channel quantised inputs (`scripts/fp8_epilogue_probe.make_inputs`), min of 5 CUDA-event runs:

| M | GEMM | two-pass ms | DSL ms | DSL TF | DSL vs two-pass |
|---|---|---|---|---|---|
| 21952 | qkv | 44.96 | 31.15 | 162.9 | bit-identical |
| 21952 | out | 12.72 | 10.29 | 164.5 | bit-identical |
| 21952 | fc1 | 59.53 | 41.03 | 164.9 | bit-identical |
| 21952 | fc2 | 22.60 | 21.13 | 160.1 | bit-identical (not routed) |
| 42848 | qkv | 86.09 | 59.25 | 167.2 | bit-identical |
| 42848 | out | 24.91 | 19.78 | 167.0 | bit-identical |
| 42848 | fc1 | 114.31 | 78.18 | 169.0 | bit-identical |
| 42848 | fc2 | 41.78 | 41.44 | 159.4 | bit-identical (not routed) |

The touched-up kernel runs at the speed of the probe's (qkv 31.15 vs 31.26 ms, fc1 41.03 vs 40.82 at M = 21952; the
skipped padded tiles are within noise here). **fc1, bf16 + `silu_and_mul` vs fused-SwiGLU level 1** (vLLM-Omni's
`SiluAndMul`, which dispatches to `forward_cuda` = `torch.ops._C.silu_and_mul` in this build): **bit-identical** at
M = 21952 and 42848 (and at 200 / 4100 in the GPU test), measured 55.61 -> 49.38 ms (21952) and 107.41 -> 96.37 ms
(42848) for the whole fc1 + activation; the projection's bandwidth estimate (~94.6 at 42848) was 2 ms optimistic.

Host cost per call (`--parts host`, 128x128x64 problem, 2000 calls): `from_dlpack` for the five tensors 23.4 us, the
compiled launcher 5.0 us, `DslGemm.run` 33.8 us, the dispatcher's `fp8_cutlass.scaled_mm` 41.4 us (torch._scaled_mm:
8.2 us). Below the ~50 us threshold of plan step 4, so nothing is cached (a per-(data_ptr, shape) cache of the weight
side saved ~10 us but kept references to the weights; removed). 150 routed GEMMs per step: ~6 ms of host time, hidden
behind the GPU.

### A cuBLASLt scale-read hazard found on the way (fixed in the two-pass path)

The first GPU test run failed 3 of 22 cases: the *reference* was wrong, not the DSL output. After CuTe-DSL launches and
`torch.cuda.empty_cache()` in the same process, `torch._scaled_mm` with a scalar scale tensor allocated and filled just
before the call (`one = torch.ones(())`, as `scaled_mm_tensor32`, `fp8_consumers._product_fp32` and the fc1 stash did
on every call) computed some output tiles with a stale scale: 8-12 % of the elements, every wrong one off by the same
factor (0.0014, the earlier contents of that memory). Correct with a long-lived scale or a `synchronize()` after the
fill; the inputs themselves were always read correctly (bit-identical when only the scale tensor was replaced). Not
reproduced without the DSL in the process (60 trials with planted stale values, with and without `empty_cache`), so
the trigger is not fully explained. The three call sites now share `fp8_gemm.unit_scale(device)`, one 1.0 per device
made and synchronised once; the probe no longer calls `empty_cache()`. With today's defaults the DSL never loads, the
value passed is the same 1.0, so the default path's numerics are unchanged (checked by the baseline arm below).

### Tests

- CPU (Mac, `tests/test_fp8_cutlass_routing.py`, 11 tests, fake H3 modules with upstream's verbatim block / attention /
  MLP forwards; the fake dispatcher now goes through the real `fp8_gemm.route`): block output bit-identical to the
  two-pass block with the router taking qkv / out / fc1 (3 DSL calls) and fc2 consumer-fused (1 stash, 1 fused), SwiGLU
  stash hook never reached, stock consumers called; bias falls through; only the running block's weights are routed;
  router before the hooks; launch failure switches the level off; fingerprints unchanged by the wrapper; probe failures
  (no Triton, import error, check mismatch) and dispatcher fallback with a fake vLLM.
- GPU (Spark_06, `tests/test_fp8_cutlass_gpu.py`, 22 tests): the four DiT shapes at M = 200 / 1000 / 4100 bit for bit,
  integer inputs with partial row and column tiles and 4096-element guards on both sides of the output, a side stream,
  the not-taken cases, fc1 + `silu_and_mul` = fused SwiGLU, and a negative control (one flipped low bit fails the check).
  All FP8 test files together on Spark_06: 169 passed.

<<SERVING-RESULTS>>
