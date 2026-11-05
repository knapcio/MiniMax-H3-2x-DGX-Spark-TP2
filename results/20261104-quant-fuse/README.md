# 2026-11-04: per-token FP8 quant fused into the producing kernels (speed plan item 6, `H3_TP2_FP8_QUANT_FUSE`)

Question: every FP8 linear of a DiT block quantises its bf16 input per token in a separate kernel before the GEMM.
Can the kernel that produces that input write the fp8 row and its scale directly, bit-identical to today's path?

**Answer: yes, at all four sites, bit-identical in fp8 bytes and scale bits on every input tested (kernels, the
exhaustive row-maximum sweep, the whole DiT block on the GPU).** The producers' copies take over the quant: -0.30
s/step at M = 10240 (960 probe, ~3.7 % of ~8 s/step), -0.55 s/step at M = 18880 (5 s 1344 calibration, ~2.6 % of ~21
s/step), -1.26 s/step at M = 42848 (277-frame 1344 windows, ~1.6 % of ~80 s/step), from per-call timings x 50 layers.
Opt-in `H3_TP2_FP8_QUANT_FUSE=1`; **serving proof PASS 2026-11-05 (queue74, below): bit-equal latents, all four sites installed and self-checked on both ranks.** Fused SwiGLU level 2's divergent render
is explained by the cuBLASLt stale-scale hazard fixed on 11-03, not by its quant (which matches the serving op bit for
bit).

Branch `quant-fuse` (from main 8a2da48), opt-in, off by default. Code: `h3_tp2/fp8_quant_fuse.py`; wiring:
`h3_tp2/plugin.py` (9b4), `h3_tp2/ulysses_exact.py` (`set_unpack_hook`), `h3_tp2/fp8_consumers.py` (block wrapper
now `functools.wraps`, so later source fingerprints still see upstream's forward); tests:
`tests/test_fp8_quant_fuse.py` (CPU), `tests/test_fp8_quant_fuse_gpu.py` (GPU); probe:
`scripts/fp8_quant_fuse_probe.py`; `scripts/queue-lib.sh` strips the new env lines on restore.

## Where the per-token quant runs today (vLLM-Omni v0.30.0 `a8576ccb7`, vLLM 0.30.0, default level `cutlass`)

`FP8ScaledMMLinearKernel.apply_weights` -> `self.quant_fp8(x_2d, None, None)` (`QuantFP8`, dynamic per token). The
diffusion vLLM config is `VllmConfig(compilation_config=CompilationConfig())` (`vllm_omni/diffusion/vllm_config.py:200`):
mode VLLM_COMPILE, backend inductor, `custom_ops=["none"]`, so `QuantFP8` is disabled as a custom op and runs its
`forward_native` compiled by Inductor with `dynamic=True` (`CustomOp.maybe_compile`). The generated kernel (in the
Inductor cache on Spark_06, `triton_red_fused__to_copy_abs_clamp_max_mul_reciprocal_unsqueeze_0`, the 1.0-1.5 % family
of the 10-28 step profile) reads the bf16 row twice (max, then quant) and computes, per row of K elements:

```
amax  = max |x|                                   (NaN-propagating maximum, order independent)
scale = maximum(amax / 448.0, 1/(448*512))        (448 is a runtime fp32 operand: div.full.f32)
q     = fp8_e4m3_rn(minimum(maximum(float(x) * (1.0 / scale), -448), 448))   (1/scale: div.full.f32)
```

Compiled for sm_121 without a device (`triton.compile` with an explicit target): `div.full.f32` for both divisions
and `cvt.rn.satfinite.e4m3x2.f32`, the instructions the fused kernels emit too. (A `dynamic=False` compile, as
`fp8_swiglu.quant_reference` uses, folds `amax / 448` into `amax * 0.002232142857142857`: another rounding, see
level 2 below.)

| site (GEMM fed) | producer of the GEMM input today | does one program see the whole row? | K |
|---|---|---|---|
| qkv | `rms_norm_indexed_scale_shift` (block `:950`), Triton, one program per row | yes (the RMS is over the row) | 5376 |
| out | Ulysses reverse unpack copy `result[0].view(m, ws, hl, d).copy_(recv.transpose(0, 1))` (`ulysses_exact.exchange_back`) | yes, once the copy is a per-row kernel (each row = 2 x 3584 contiguous source chunks) | 7168 |
| fc1 | `indexed_gate_rms_norm_scale_shift` (block `:970`), Triton, one program per row | yes (RMS over the row) | 5376 |
| fc2 | `SiluAndMul` (`torch.ops._C.silu_and_mul`) on the DSL GEMM's bf16 fc1 output | yes, once SwiGLU is a per-row kernel (one program, 14336 outputs) | 14336 |

None of the GEMM epilogues produces a quantised tensor: the DSL kernel's qkv / out / fc1 outputs and fc2's
consumer-fused output go to norm + RoPE, attention, the modulation kernels and the gated residual first, and a GEMM
tile never sees a whole row (a two-phase amax would be needed there). So the fusion sits in the four producers.
All four are exact candidates; no two-phase variant is needed.

## What was built

Per site a copy of the producer that computes exactly the bf16 value the stock producer stores (same source up to
the store, same launch shape: one program per row, `block_n = next_power_of_2(K)`, 8 warps; 16 for SwiGLU), keeps it
in registers and applies the quant above to the whole row, writing the fp8 row and the fp32 scale [M, 1] instead of
the bf16 row. The layer receives an uninitialised bf16 placeholder of the right shape; a wrapper of `QuantFP8.forward`
returns the stashed (fp8, scale) when, and only when, it is called by the planned layer's own `QuantFP8` instance on
exactly that placeholder (data pointer, shape, strides, dtype) with no static or upper-bound scale. Everything else
calls the original `forward` with its arguments untouched.

- qkv: `_rms_norm_indexed_scale_shift_quant_kernel`, out: `_unpack_quant_kernel` (ws = 2 sources, 2 x 4096 tile), fc1:
  `_indexed_gate_rms_norm_scale_shift_quant_kernel` (residual_out stored as upstream), fc2: `_silu_mul_quant_kernel`
  (vLLM's silu_and_mul arithmetic as fused SwiGLU level 1 reproduces it, then the quant).
- Saved per element of the four activations: the producer's bf16 write (2 B), the quant kernel's bf16 read (2 B, its
  second pass mostly from L2) and its fp8 write, against one fp8 write in the producer: ~4 B per element, fc2 also
  the SwiGLU round trip. Per layer at M = 18880: 32256 x 18880 elements = 0.61 G, ~2.4 GB.
- **The layout trap (found offline, before the GPU window).** The first version's 128-bit fp8 store made Triton pick
  16 elements per thread for the whole modulation kernel, including the RMS `tl.sum` (upstream: 8), i.e. a different
  reduction order (TTGIR of both kernels compiled for sm_121, `scripts` below). With the fp8 store's provable
  contiguity limited to 8 (`(c // 8) * 8 + c % 8`, 64-bit stores) every memory op asks for upstream's layout again:
  the normalised TTGIR of the fused kernels equals upstream's line for line up to the bf16 rounding of the output
  (same ops, same `#blocked<sizePerThread = [8]>` for every op and both reduces); only the quant follows.
- Mode 1 (fallback, picked automatically when mode 0 fails its self-check; `H3_TP2_FP8_QUANT_FUSE_MODE=0|1` forces):
  the RMS kernels keep upstream's bf16 store into the placeholder, `tl.debug_barrier()`, read the row back and
  quantise it. Its PTX reduction sequence equals upstream's instruction for instruction. Saves ~2 B per element instead
  of 4.
- Safety: a site is planned only for a DiT block whose four layers are FP8 linears with a dynamic per-token
  `QuantFP8` (no static / ub scale, no token padding), tp 1, not LoRA-wrapped, no VSA gate (it would read the qkv input
  twice), `VLLM_BATCH_INVARIANT` off, and reviewed upstream sources (fingerprints: block / attention / MLP forwards,
  the modulation kernels and launchers, `_run_packed_attention`, vLLM-Omni `Attention.forward`, `SiluAndMul.forward_cuda`,
  vLLM `ColumnParallelLinear.forward`, `RowParallelLinear.forward`, `FP8ScaledMMLinearKernel.apply_weights`). A stash
  not consumed by its quant (another quant call while it is pending, the block ending) is materialised first (the
  stock producer run into the placeholder); a planned layer whose forward returns without consuming it (a forward
  hook) is run again on the materialised input. Self-check on the first call per site and width: fused vs (stock
  producer + the layer's own `QuantFP8`) on the first 64 rows, fp8 bytes / scale bits / by-products; a mismatch of the
  last mode disables the site for the process. `H3_TP2_FP8_QUANT_FUSE_VERIFY=1` (diagnosis, slow) compares every call.
- Preconditions per site: out needs `H3_TP2_ULYSSES_EXACT=1` (default; its reverse unpack is the producer); fc1 needs
  the stock `indexed_gate_rms_norm_scale_shift` (cutlass level; at tensor32 fp8_consumers' `out` site owns it); fc2
  needs fc1's bf16 output to reach `SiluAndMul` (cutlass level; at tensor32 fused SwiGLU level 1 owns it).

## Measured (Spark_06, GB10 sm_121, 2026-11-05 02:24-02:28, head up and idle on the defaults, clocks untouched)

Commands (scratch copy `/tmp/quant-fuse` of this branch, `CPATH` as in `scripts/env.sh` for Triton's runtime helper,
own `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` under `/tmp/quant-fuse-cache`):

```
~/h3-long/.venv/bin/python -m pytest -q tests/test_fp8_quant_fuse_gpu.py            # 26 passed
~/h3-long/.venv/bin/python -m pytest -q tests/test_fp8_swiglu.py tests/test_fp8_cutlass_gpu.py \
    tests/test_ulysses_exact_gpu.py tests/test_fp8_gemm.py                           # 64 passed (regression)
~/h3-long/.venv/bin/python scripts/fp8_quant_fuse_probe.py --m 10240 18880 --jsonl probe.jsonl
~/h3-long/.venv/bin/python scripts/fp8_quant_fuse_probe.py --m 42848 --no-level2 --jsonl probe.jsonl
CUDA_VISIBLE_DEVICES= ~/h3-long/.venv/bin/python scripts/fp8_quant_fuse_ttgir.py     # offline, no GPU
pytest -q tests/test_fp8_quant_fuse.py                                               # CPU, 26 passed
```

Probe (`probe.jsonl` here; also `~/h3-tp2-results/fp8-quant-fuse-probe/` on Spark_06): per site, ms per call (min of
5 CUDA-event runs after 2 warm-ups) for the stock pair (upstream producer + the serving `QuantFP8`, the latter alone
in the "quant" column) and the fused producer; bits compared on the full M rows (fp8 bytes, scales, residual_out for
fc1). Inputs: bf16 rows with log-normal magnitudes, AdaLN chunk views with row stride 6H, the a2a receive layout
[2, M, 28, 128], fc1 outputs with gate values up to +-16. "per step" = saving x 50 layers.

| M | site (K) | producer | quant | stock pair | fused (mode) | saving | per step | bit-identical |
|---|---|---|---|---|---|---|---|---|
| 10240 | qkv (5376) | 0.962 | 0.766 | 1.702 | **0.704** (0) / 1.212 (1) | 0.997 ms | 0.050 s | yes / yes |
| 10240 | out (7168) | 1.263 | 1.031 | 2.278 | **0.974** | 1.304 | 0.065 | yes |
| 10240 | fc1 (5376) | 1.954 | 0.753 | 2.662 | **1.712** (0) / 2.197 (1) | 0.951 | 0.048 | yes / yes |
| 10240 | fc2 (14336) | 3.711 | 2.093 | 5.800 | **3.013** | 2.787 | 0.139 | yes |
| 10240 | **all** | | | | | | **0.30 s/step** | |
| 18880 | qkv | 1.715 | 1.359 | 3.039 | **1.233** / 2.146 | 1.806 | 0.090 | yes / yes |
| 18880 | out | 2.248 | 1.830 | 4.049 | **1.749** | 2.300 | 0.115 | yes |
| 18880 | fc1 | 3.493 | 1.331 | 4.818 | **3.036** / 3.967 | 1.782 | 0.089 | yes / yes |
| 18880 | fc2 | 6.762 | 3.782 | 10.535 | **5.458** | 5.077 | 0.254 | yes |
| 18880 | **all** | | | | | | **0.55 s/step** | |
| 42848 | qkv | 3.965 | 3.087 | 7.038 | **2.903** / 4.994 | 4.135 | 0.207 | yes / yes |
| 42848 | out | 5.129 | 4.149 | 9.262 | **3.984** | 5.278 | 0.264 | yes |
| 42848 | fc1 | 7.928 | 3.003 | 10.930 | **6.918** / 8.900 | 4.011 | 0.201 | yes / yes |
| 42848 | fc2 | 15.487 | 8.787 | 24.307 | **12.502** | 11.805 | 0.590 | yes |
| 42848 | **all** | | | | | | **1.26 s/step** | |

The fused producers are faster than the stock producers alone (they write 1 B per element instead of 2), so the
saving exceeds the quant kernel's own time. Mode 1 (bf16 store kept) still saves about half. Per step at the owner's
cells: 5 s 1344 calibration (~21 s/step) ~-2.6 %, 960 probe ~-3.7 %, 277-frame 1344 windows (~80 s/step) ~-1.6 %;
these are kernel-time projections, the serving proof gives the real number. Memory: the placeholders take the place
of stock's bf16 activations (same size, never touched); transient extra: the out site keeps the a2a receive buffer
until out_proj's quant (M x 7168 x 2 B, 0.27 GB at 18880), fc2 keeps fc1's bf16 output until fc2's quant
(M x 28672 x 2 B, 1.08 GB at 18880, 2.46 GB at 42848; both are what the materialise fallback needs).

GPU tests (`tests/test_fp8_quant_fuse_gpu.py`, 26): each site at M = 1 / 200 / 1000 / 4100 (both modes for qkv and fc1,
mode 1 also checks the bf16 row against upstream's), directed rows (zero and -0 rows = minimum scale, subnormal and
near-max maxima, saturation), the row maximum over **every positive finite bf16 value** (32639 rows, scale bits and
fp8 bytes equal), and upstream's verbatim DiT block (fake layers from `tests/fake_h3_consumers.py`) with vLLM-Omni's
real Triton producers and consumers, `SiluAndMul`, serving `QuantFP8` instances, the two-pass GEMM and a simulated
2-rank Ulysses reverse unpack: all four sites fused, self-checks passed in mode 0, block output bit-identical at 64 and
333 rows. A CPU dry run of `install()` on the real vLLM-Omni modules after the cutlass and consumer-fuse wiring (source
checks on) installs all four sites.

Quant implementations against the serving op (exhaustive row-maximum sweep, `kind=level2` in `probe.jsonl`): the fused
quant 0 / 0 differing scales / fp8 bytes; `fp8_swiglu`'s level-2 kernel 0 / 0; `QuantFP8` compiled with
`dynamic=False` (level 2's test reference, `amax * fp32(1/448)`) 0 / 0; **vLLM's CUDA `scaled_fp8_quant`
(`forward_cuda`) 9727 / 7324 of 32639 rows differ**: any fusion has to match the compiled native form that serving
actually runs, not the CUDA op.

## Fused SwiGLU level 2: why one of two renders differed (diagnosis)

The two level-2 renders of 2026-10-29 (`queue8.log` on Spark_06, 5 s 960x576 Ref2VA, seed 1101): `v030-fp8-fuse2-1`
framemd5 `cd1bcf1d` was **the first request after a fresh boot** (head restarted 06:50:13, launched 06:50:28),
`v030-fp8-fuse2-2` (`85437ffa`, the plain two-pass digest) the second request of the same boot. That is the
first-request-of-a-boot pattern the speed plan recorded on 11-02 without level 2 (960x576: three boots, three
first-request digests, the warm requests repeat), and its cause was found on 11-03: cuBLASLt's sm121 kernel read a
scalar scale written just before the call stale when its 512-byte block came from fresh memory (`results/20261103-
cutlass-sm120/README.md`, "A cuBLASLt scale-read hazard"); fixed in 5f89494 (11-03 10:29) with `fp8_gemm.unit_scale`.
The boot-probe digests show it: the first 960 probe of a boot had its own digest up to queue55 (11-03 02:06:
first `870616ea227f`, warm `bf6e85687a6a`), and from queue57 (11-03 10:54, after the fix) the first probe equals the
warm digest `bf6e85687a6a` on every boot logged since (queue57-69, about 25 boots, the race-affected queue62 boots
aside). In 10-29 every scalar-scale GEMM allocated its `one` on each call and level 2 changes the allocation pattern
(no bf16 [M, ffn] activation), so a first request on fresh segments was exposed. The level-2 kernel itself is
exact: its in-kernel quant equals the serving op on all 32639 row maxima and on 2048 serving-like SwiGLU rows (probe,
0 differing bytes), so it cannot produce a different sample. Level 2 is in any case unreachable at the cutlass
default (fc1 is routed to the DSL kernel before fp8_swiglu's stash hook fires); the fc2 site here replaces it.

## Exact env for a serving proof

Both ranks (head and agent, like `H3_TP2_FP8_GEMM`), on top of the defaults (cutlass level, Ulysses exact 1, consumer
fuse 1, fused SwiGLU 1): `export H3_TP2_FP8_QUANT_FUSE=1`. Optional: `H3_TP2_FP8_QUANT_FUSE_VERIFY=1` (every call
re-run stock and compared, logged; slow, diagnosis only), `H3_TP2_FP8_QUANT_FUSE_MODE=0|1` (force a mode; default auto
= 0, then 1). The branch's `scripts/queue-lib.sh` strips these lines on a defaults restore; with an older queue-lib
in `~/h3-tp2` the restore must drop them explicitly.

Recipe (queue67 pattern, `calib` = 5 s 1344x768 T2VA, seed 0, 20 steps, forest-stream prompt, latents saved):
1. A: `rm -f ~/h3-tp2-results/.head-rev`; `restart_head_gated "$DEFAULT_ENV"`; `calib qf-A-default` -> digest
   `4d6d69e4aad0`, baseline s/it.
2. B: `restart_head_gated "$DEFAULT_ENV
   export H3_TP2_FP8_QUANT_FUSE=1"`: the warm 960 boot probe must give `bf6e85687a6a`; then `calib` three times, every
   digest `4d6d69e4aad0`, `latents_diff.py` A vs B1 max abs diff 0; s/it from B2 and B3 (expect ~-0.5 s/it).
   Journal on both ranks: `EXPERIMENT FP8 quant fuse requested=1, installed sites: qkv, out, fc1, fc2`; four
   `self-check (...) bit-identical` lines (`'qkv', 5376, ..., 0`, `'out', 2, 3584`, `'fc1', 5376, ..., 0`,
   `'fc2', 28672`); no `materialised`, no `NOT bit-identical`; `grep "fp8 quant fuse"` for the counters.
3. C: `restart_head_gated "$DEFAULT_ENV"` (defaults restored).
Default gate afterwards, as for cutlass (queue69): the 16 s 1344x768 FL2VA continuation twice on a quant-fuse boot,
digests `200e38721c9e 4138eba5a964`, framemd5 `e46129c05cc6`.

## Serving proof: PASS (queue74, 2026-11-05 03:51-05:34, `levers` 706f2e6 synced to both Sparks)

Gated compacted boots through `queue-lib.sh` (`~/h3-tp2-results/queue74-levers-proof.sh` on Spark_06), the 5 s
1344x768 T2VA calibration (seed 0, 20 steps, the forest stream prompt, expected latents digest `4d6d69e4aad0`):

| arm | boot probe (warm s/step, latents) | calibration digests | `latents_diff.py` vs A | s/it |
|---|---|---|---|---|
| A: defaults | PASS 7.75 `bf6e85687a6a` | `4d6d69e4aad0` | | 21.24 |
| B: `H3_TP2_FP8_QUANT_FUSE=1`, three runs | PASS 7.91 `bf6e85687a6a` | `4d6d69e4aad0` x3 | B1: video `(1, 24, 37, 48, 84)` and audio `(2, 32, 207)` bit-equal, max abs diff 0 | 21.75 / 21.02 / 21.87 |
| C: `H3_TP2_ULYSSES_EXACT=2` + `H3_TP2_FP8_QUANT_FUSE=1`, two runs | PASS 7.44 `bf6e85687a6a` | `4d6d69e4aad0` x2 | C1: bit-equal, max abs diff 0 | 20.47 / 19.82 |
| E: defaults restored | PASS 8.29 `bf6e85687a6a`, health 200 | | | |

Head journal and rank 1's worker log in B (both ranks): `installed sites: qkv, out, fc1, fc2`, `fp8 quant fuse:
self-check ('qkv', 5376, ...) bit-identical`, `self-check ('out', 2, 3584) bit-identical`, `self-check ('fc1', 5376,
...) bit-identical`, `self-check ('fc2', 28672) bit-identical`, `first fused_qkv / fused_out / fused_fc1 / fused_fc2`,
and at the end of the boot `fused_qkv=5000 fused_out=5000 fused_fc1=5000 fused_fc2=5000 consumed=20000 selfcheck_pass=4`
(the probe + three calibrations = 100 steps x 50 layers per rank, every call fused). In C: `site out skipped
automatically: H3_TP2_ULYSSES_EXACT=2 makes no reverse unpack copy to fuse into ...`, `installed sites: qkv, fc1, fc2`,
`Ulysses exact exchange installed (mode 2)`, three fuse self-checks + copy5's self-check bit-identical. Absent in both:
`NOT bit-identical`, `materialised`, `requested but ignored`. `Q74 lever lines left in env: 0` after E.

Speed: the calibration cannot resolve this lever. The boot-to-boot spread of the A arm alone is 20.44 (queue73) to
21.24 (queue74) s/it, and the three B runs in one boot span 21.02-21.87, so the projected -0.55 s/step (2.6 %) at M =
18880 sits inside the noise; B's mean 21.55 vs A 21.24 is not a regression that this measurement could show either.
The combined boot C gave the two fastest calibrations recorded (20.47 / 19.82 s/it), but with one boot per arm that
is a hint, not a measurement. The proof here is the digest and the bit-equal latents; the default gate (queue75: the
16 s 1344x768 FL2VA continuation twice on a boot with both levers, digests `200e38721c9e 4138eba5a964`, framemd5
`e46129c05cc6`) decides the flip.

## What is left

- Default gate PASS (queue75, 2026-11-05, boot with both levers): the 16 s 1344x768 FL2VA continuation twice, digests
  `200e38721c9e 4138eba5a964`, framemd5 `e46129c05cc6` (gen.log: Q75-START 05:35:07; Q75-BOOT rc 0 PROBE PASS: first 8.05 (bf6e85687a6a), warm 7.58 s/step (reference 8.65, limit 9.342) latents bf6e85687a6a 05:48:06 05:48:06; Q75-CONT-1: 20/20 [27:53<00:00, 80.68s/it] 20/20 [27:53<00:00, 83.70s/it] 20/20 [08:36<00:00, 25.74s/it] 20/20 [08:36<00:00, 25.80s/it]  digests 200e38721c9e 4138eba5a964  framemd5: e46129c05cc6 phase diffuse 1674.1 s phase diffuse 516.1 s phase decode 86.5 s ; Q75 PASS 1; Q75-CONT-2: 20/20 [28:05<00:00, 83.83s/it] 20/20 [28:05<00:00, 84.26s/it] 20/20 [08:56<00:00, 26.71s/it] 20/20 [08:56<00:00, 26.80s/it]  digests 200e38721c9e 4138eba5a964  framemd5: e46129c05cc6 phase diffuse 1685.4 s phase diffuse 536.1 s phase decode 86.3 s ; Q75 PASS 2; Q75-BOOT-E (defaults restored) rc 0 health 200 07:19:44; Q75-DONE 07:19:44). **Both defaults
  flipped on 2026-11-05: `H3_TP2_FP8_QUANT_FUSE` default `1`, `H3_TP2_ULYSSES_EXACT` default `2`.**
- Mode 0's PTX schedules the RMS reduction's instructions in another order than upstream (same TTGIR, same layouts,
  LLVM does not reassociate float ops), so its exactness rests on the GPU checks (unit tests, probe on full M, the
  serving self-check) rather than on identical machine code; mode 1 has upstream's PTX reduction sequence and is the
  automatic fallback.
- Not fused: the token refiner blocks, the final layer and the AdaLN projections (their inputs are small or not
  produced by these kernels); the DSL GEMM epilogue cannot produce a quantised row (no CTA sees the whole row; a
  two-phase amax would cost a second pass over the output and is not worth it here).
- fc2 at tensor32 (fused SwiGLU level 1 owns fc1's output there) and fc1 at tensor32 with the consumer-fuse out site
  are skipped, not fused; the cutlass level is the default anyway.
