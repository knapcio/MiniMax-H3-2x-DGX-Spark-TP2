# 2026-11-03: the stage-release race of the CuTe-DSL sm120 FP8 GEMM (`H3_TP2_FP8_GEMM=cutlass`)

Follow-up to the VERIFY run (queue62) in `README.md`: 4 of 48000 serving GEMMs differed from two-pass, always the
first rows of a 128-row M tile and one MMA warp's 64 columns.

**Root cause: a write-after-read hazard across proxies.** The MMA warps read each smem stage with `ldmatrix` (generic
proxy) and release it with an mbarrier arrive; the DMA warp then refills the stage with TMA (async proxy). ptxas issues
the release arrive (and the named barrier before it) **without waiting for the ldmatrix loads of the k block just
issued from that stage**. Under load such a load can still be queued when the 4th arrive lands, the producer issues the
next TMA load into the stage, and the TMA write gets there first: the warp computes with the A and / or B data of fill
`g + ab_stage` instead of fill `g`. The PTX memory model only orders generic-proxy accesses before later async-proxy
accesses to the same memory with `fence.proxy.async`. Neither the DSL nor our kernel emitted one. The named barrier
added earlier ("defect 3" in `README.md`) only made the race rare.

**Fix** (`h3_tp2/fp8_cutlass_kernel.py`, both release sites, the default two-pass path is not touched):
`cute.arch.fence_proxy("async.shared", space="cta")` executed by every MMA thread right before
`mainloop_sync_barrier.arrive_and_wait()` and `consumer_release`. In SASS this becomes `MEMBAR.ALL.CTA` +
`FENCE.VIEW.ASYNC.S`, and the barrier now waits on the fence's scoreboard. That forces each thread's ldmatrix loads
to complete before any warp can release the stage.

**Stress result** (`scripts/fp8_cutlass_stress.py`, Spark_06, M = 10240, qkv + fc1 alternating, 2 activation buffers,
pointer-chase busy kernel on a second stream, every call compared bit for bit):

| kernel | runs | mismatching calls / calls |
|---|---|---|
| before (git HEAD 23e8cc6) | 6 x 90-240 s | **81 / 13216** (5/2224, 8/2304, 9/2416, 44/3312, 13/2096, 2/864) |
| after (this fix) | 5 x 90-240 s | **0 / 12176** (0/2240, 0/2224, 0/4768, 0/2096, 0/848) |
| back-to-back pair, same conditions | 240 s each | before 13/2096, after 0/2096 |
| positive control: named barrier removed | 60 s | 648/648 (hundreds of wrong rows per call) |
| barrier removed, fence kept | 60 s | 0/552 |

If the fix did nothing, about 75 mismatching calls would be expected in the 12176 after-calls (P(0) ~ e^-75). For the
single pair run alone, P(0 | 13 expected) ~ 2e-6.

## Reproducer and what triggers it

`scripts/fp8_cutlass_stress.py` (~1.9 GB GPU memory, < 5 min per run). The reference
`fp8_gemm.scaled_mm_tensor32` is computed once per (shape, buffer) and kept only as 64-bit checksums: one per row, per
128-column N tile and per MMA warp column group. That is why the GEMM stream never drains per call (one sync every 16
calls). A Triton check kernel captures every mismatching (row, N tile) segment. The report covers rows, the row inside
the tile, columns, the MMA warp (warp_m = (row % 32) // 16, warp_n = (col % 32) // 16), the persistent CTA and its tile
position (the kernel's swizzled raster), and an attribution fit. `--save` / `--reanalyze` repeat the attribution offline.

Busy kernels on the second stream (240 s each, pre-fix kernel):

| busy | result |
|---|---|
| `chase`: dependent random loads from a 32 MB table, 80 kernels per GEMM | 5-44 mismatching calls per run |
| `add`: streaming elementwise, 72-80 per GEMM | 0/2352, 0/1872 |
| `none` | 0/3312 (150 s) |
| `smem` (Triton tl.dot next to the GEMM CTA), `hostpoll` (volatile loads of pinned host memory) | 0 (15-30 s only) |
| `chase` with M = 1024 (many short launches, 3 busy per GEMM) | 0/25952 |
| `chase` with automatic busy calibration (330 per GEMM, GEMM slowed 3x) | 0/784 (1/160 in an earlier 15 s run at 77 per GEMM) |

The race needs long-latency loads from co-resident kernels on the GEMM's SMs, which is what keeps an issued ldmatrix
waiting. Streaming loads are not enough. In serving the co-runners are NCCL's kernels (flag polling, copies) on the
comm stream. That this is the trigger in serving is plausible but **not tested**.

The rates vary by run: the 5-hour render was running for the first runs (time-sliced with this process) and had
finished by about 19:50 (base_chase5: 44/3312 on a nearly idle GPU).

## Attribution of the reproduced events (pre-fix kernel)

`tile_fit` models every A row and every B column of the tile, per 16-byte chunk and per reading warp, as read either
correctly or from another fill of the same stage. 67 events from the three saved runs:

- 58 explained to >= 98 % of the error energy (median 99.9 %) by **fill + ab_stage** (the *next* TMA load into the
  stage: a WAR). 6 more are best explained by the same at 93-97 % (partially overwritten 16-byte chunks), 3 at 66-89 %.
  None is best explained by the previous fill (a read before the TMA data landed, i.e. a RAW).
- **Always k block 1** of the stage (bytes 32..63 of the 64-byte k tile). That is the k block whose 8 ldmatrix are
  issued right before the release; k block 0 was read one k block earlier.
- Rows / columns: whole or partial 16-row (A) or 16-column (B) ldmatrix groups of one or more warps, for example
  A rows 0..4 of one warp. That is the serving signature: the TMA writes the box from row 0, so a late A ldmatrix sees
  its first rows overwritten.
- 66 of 67 events are in the CTA's first tile (k tiles 0..12) and one is at tile position 292. Likely cause: the busy
  kernels are enqueued just before each GEMM, so the contention falls at the GEMM's start. The serving errors were in
  M tiles 0, 1, 5 (first raster group) and 36 (mid kernel).

Positive control (barrier removed): all events are WAR at k tile 83 (the hoisted last k tile, where ptxas puts the
arrive one instruction after the last ldmatrix), explained 99.9 %.

## SASS evidence (cuobjdump of the compiled cubin, control bits decoded)

Pre-fix k-tile loop (`dump/orig`; `Wn` = scoreboard set, `wait[...]` = scoreboards waited on):

```
01bf0  W1  LDSM.16.M88.4 R164, [R196]          k block 1 of stage s, A rows 0-15 (first of 8 LDSM, all on SB1)
...
01cf0  W1  LDSM.16.M88.4 R180, [R3+0x1800]
01d00      BAR.SYNC.DEFER_BLOCKING 0x3, 0x80   wait[-]: does not wait for SB1
01d10..    20 x QMMA (k block 0)
01e50      @!P1 SYNCS.ARRIVE.TRANS64.A1T0 [UR7+0x28]   consumer_release, wait[-]
01e60      SYNCS.PHASECHK.TRANS64.TRYWAIT ...  next stage
01f50      QMMA ... wait[1]                    first use of the LDSM results, after the release
```

Hoisted last k tile: LDSM at 0x2660 / 0x2680, `BAR.SYNC` at 0x2690, `SYNCS.ARRIVE` at 0x2710, no waits. Without the
barrier, the arrive at 0x2780 directly follows the LDSM at 0x2770. With the fix: `MEMBAR.ALL.CTA` (0x1d00) +
`FENCE.VIEW.ASYNC.S` (0x1d10, sets SB5), then `BAR.SYNC ... wait[5]` (0x1da0) and the arrive (0x1e70). The hoisted tail
gets the same at 0x2680 / 0x2690 / 0x26d0 / 0x2780. The fixed kernel's SASS equals the diagnosis variant used for the
first after-run (`RACE_FENCE=pre`).

## Answers to the four questions

DSL sources: `~/h3-long/.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/` on Spark_06
(nvidia-cutlass-dsl 4.7.1). Example: CUTLASS v4.7.1 (git cb42473)
`examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py`, the persistent warp-specialized
sm120 GEMM our generator was applied to, from the source tree on Spark_06
(`~/spark-ds41f-r6x-correctness/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-6f44879b/src/cutlass`). Kernel line
numbers are the fixed file's.

**1. `consumer_release` arrive and arrive count; acquire on the wait.**
`PipelineTmaAsync.consumer_release` (`pipeline/sm90.py:730`) arrives only from `is_signaling_thread`, which
`init_empty_barrier_arrive_signal` (`sm90.py:478`, line 506: `tidx % 32 < cute.size(cluster)`) sets to **lane 0 of
every warp** for a 1x1x1 cluster. With a 1-CTA cluster `dst_rank` is None (`sm90.py:671-672`), so the arrive is a local
`mbarrier.arrive.shared.b64` (PTX line 641; `cute/arch/mbar.py:415`, scope CTA by default, line 442, i.e. release.cta).
Our `consumer_arrive_cnt = mcast_size (1) * num_mma_warps (4) = 4` passes unchanged
(`enable_multicast_signaling=False`, `sm90.py:639`) into the empty-barrier init count (`helpers.py:271`, `:329`).
**4 arrivals per phase, one per MMA warp: it matches.**

So in the DSL the named barrier is not needed for counting: each warp arrives after its own ldmatrix in program order.
CUTLASS C++ is different: it signals with one thread per *warpgroup*, which is why `sm120_mma_tma.hpp` needs
`NamedBarrier::sync` there.

`consumer_try_wait` / `consumer_wait` (`sm90.py:298`, `:280`; `mbar.py:318`, `:287`) lower to
`mbarrier.try_wait.parity.acquire.cta.shared::cta.b64` plus a retry loop of `mbarrier.try_wait.parity.shared.b64`
(default acquire.cta, PTX lines 650-660): **acquire semantics, yes**. The following ldmatrix are control-dependent on
the result in SASS (`@P0 BRA` with `wait[5]` on the TRYWAIT). No RAW was observed.

**2. Program order and memory ordering of ldmatrix -> barrier -> release.**
PTX keeps program order (ldmatrix of k block 1, then `bar.sync 3, 128` at PTX lines 636 / 1178, then the arrive at 641 /
1183) and emits **no `fence.proxy.async`** in the mainloop. The only proxy-related fence is `fence.mbarrier_init` at
init. In SASS, neither `BAR.SYNC` nor `SYNCS.ARRIVE` waits on the LDSM scoreboard (excerpt above). `bar.sync` is a
real `bar.sync` (`NamedBarrier.arrive_and_wait` -> `cute.arch.barrier`, `helpers.py:689`), not `bar.arrive`, but it
does not complete outstanding loads.

CUTLASS C++ sm90 / sm120 load pipelines also release without a proxy fence (`sm120_mma_tma.hpp`: `NamedBarrier::sync`
then `consumer_release`, no fence). The C++ sm120 mainloop has the same structure and may have the same latent window;
not tested. Dump knobs: `CUTE_DSL_KEEP=ptx,sass CUTE_DSL_DUMP_DIR=... CUTE_DSL_NO_CACHE=1`
(`base_dsl/env_manager.py`).

**3. Producer run-ahead across the persistent tile boundary.** No defect.
- Producer (kernel lines 739-798) and consumers (567-736) walk the same static persistent schedule with the same skip
  predicate for tiles past M / N (575 vs 744). Each issues or consumes exactly `k_tile_cnt` fills per valid tile.
- `reset_count()` (`helpers.py:872`) resets only the k counter; index and phase continue across tiles on both sides.
- The producer can therefore be up to `ab_stage` fills ahead, including fills of the CTA's next tile. That is the
  design, the consumer's state expects it, and the mbarrier parity cannot alias (the producer can never be more than
  one phase ahead on a stage).
- `producer_tail` (`sm90.py:338`) waits for every stage's empty barrier before the DMA warp exits, as in the example
  (example line 955). M = 10240 has no padded swizzle tiles (80 M tiles is a multiple of 8).

Every difference to the example, with a judgement:

| # | difference | judgement |
|---|---|---|
| 1 | constructor knobs (atom layout, swizzle, raster, epilogue stages, scaled, mma regs, `b_ldsm`) | benign |
| 2 | FP8 `MmaFP8Op` 16x8x32 + sm120 8-bit N permutation `(8,2,2):(1,16,8)` | required (README defect 1) |
| 3 | `ab_stage < 2` raises instead of `sys.exit(0)` | benign |
| 4 | epilogue stages 1 instead of 8, so 5 A/B stages instead of 2 | changes timing only; the WAR window exists for any stage count once the producer waits on the empty barrier |
| 5 | scheduler swizzle 8, raster along N | benign (order only) |
| 6 | tiles past M / N skipped by producer and consumers alike | consistent, benign |
| 7 | NamedBarrier 3 before each `consumer_release` | redundant for DSL arrival counting; made the WAR rare, did not remove it |
| 8 | direct register-to-global epilogue with the scales instead of the TMA-store pipeline (`sC`, `tma_atom_c` and its descriptor prefetch remain, unused) | benign (4 KB smem); README defect 2 |
| 9 | extra kernel arguments `sa`, `sb`, `c` | benign |
| 10 | **added now:** `fence.proxy.async` before every release | the fix |

The example itself has the hazard. It releases with lane 0 of each warp right after the ldmatrix of the last k block,
which is our "barrier removed" variant (648/648 calls wrong at 5 stages). With its 2 A/B stages it was probably just
rarer.

**4. Is the named barrier executed by all MMA threads before both releases?** Yes.
- Both sites (k-tile loop, line 622; hoisted last k tile, line 671) sit under a Python-constexpr
  `if k_block_idx == num_k_blocks - 1`, so the code is not divergent.
- They run `k_tile_cnt - 1` times plus once per valid tile in all 4 MMA warps. The work tile and the skip predicate
  are warp-uniform.
- Partial edge tiles use the same mainloop (only the store is predicated). Skipped tiles skip mainloop and barrier in
  all 4 warps.
- The DMA warp never uses barrier 3; barriers 0 / 1 are only used at init.
- SASS: `BAR.SYNC.DEFER_BLOCKING 0x3, 0x80` at both sites.

## Diff summary

- `h3_tp2/fp8_cutlass_kernel.py`: `cute.arch.fence_proxy("async.shared", space="cta")` before the named barrier at
  both release sites, a comment on the race, and a module docstring update (+12 / -4 lines).
- `scripts/fp8_cutlass_stress.py`: new, the reproducer and attribution described above.
- `h3_tp2/fp8_cutlass.py`: unchanged.
- `tests/test_fp8_cutlass_gpu.py` against the fixed kernel on Spark_06: **22 passed**.

Speed under render contention (min / median of 15 runs, alternating processes orig / fixed / orig / fixed; the GPU
reached ~75 TF here instead of ~165 idle):

| shape | orig median ms | fixed median ms |
|---|---|---|
| qkv, M = 10240 | 32.83 / 32.23 | 32.83 / 32.34 |
| fc1, M = 10240 | 45.32 / 42.11 | 45.41 / 42.40 |
| qkv, M = 21952 | 73.37 / 73.05 | 74.03 / 73.00 |

No difference within noise, but **an idle-GPU timing is still owed**. The cost per k tile is one MEMBAR per warp,
right after its loads were issued.

## What remains unknown

- **The hardware path.** How an issued LDSM can be overtaken by a later `SYNCS.ARRIVE` plus a TMA write (MIO / LSU
  queueing versus the SYNCS / TMA path) is inferred from the data, not from documentation. The fit says the warp read
  the next fill. The fence removes it.
- **Serving trigger.** That NCCL's kernels are what delays the ldmatrix in serving is not tested. `hostpoll` ran 15 s
  only.
- **First-tile concentration.** Why nearly all stress events sit in the CTA's first tile is not verified (likely
  harness timing).
- **Serving proof not run.** It is the three-calibration queue from `README.md`: the warm 960 probe `bf6e85687a6a`,
  three 1344 calibrations `4d6d69e4aad0`, ideally first with `H3_TP2_FP8_CUTLASS_VERIFY=1`. Serving units, `~/.h3-tp2.env`
  and the synced tree were not touched.
- **Idle-GPU speed** of the fixed kernel (see above).
- **README.** The defect-3 paragraph and the VERIFY reading in `README.md` still describe the named barrier as the
  fix. They should point here.
- **Partial events.** 9 of 67 events are explained only partially (66-97 %). The best fit is still the next fill, but
  only part of a 16-byte chunk or more than one k block was hit.
- **Runs alongside the render.** Most runs shared the GPU with the render via time-slicing. Correctness results do not
  depend on that; the rates do.

## Commands (Spark_06; logs and saved events in `~/h3-tp2-race/logs/`, every invocation in `commands.txt` there)

Environment for every run:

```
cd ~/h3-long && set -a; . ~/.h3-tp2.env; . ~/h3-tp2/scripts/env.sh; set +a; export PATH=/usr/local/cuda/bin:$PATH
```

Trees: `~/h3-tp2-race/orig` (HEAD h3_tp2 + scripts), `fixed` (this worktree), `var` (diagnosis copy with env knobs
`RACE_BAR=0|1`, `RACE_FENCE=pre|post`; not committed).

```
# stress (before: tree=orig, after: tree=fixed); runner adds CUTE_DSL_NO_CACHE=1, timeout 330, nice -n 10
CUTE_DSL_NO_CACHE=1 PYTHONPATH=~/h3-tp2-race/<tree> timeout 330 nice -n 10 ~/h3-long/.venv/bin/python \
  ~/h3-tp2-race/<tree>/scripts/fp8_cutlass_stress.py --seconds 240 --window 16 --busy chase --busy-per-call 80 \
  --nbuf 2 --detail 30 --detail-rows 2 --save ~/h3-tp2-race/logs/<tag>.pt --jsonl ~/h3-tp2-race/logs/<tag>.jsonl
# offline attribution of a saved run
... fp8_cutlass_stress.py --reanalyze ~/h3-tp2-race/logs/base_chase5.pt --detail 15 --detail-from 0 --detail-rows 4
# positive control / fence-only variant
RACE_BAR=0 ... --seconds 60 --window 8 --detail 3          (tree=var)
RACE_BAR=0 RACE_FENCE=pre ... --seconds 60 --window 8      (tree=var)
# PTX / SASS
CUTE_DSL_NO_CACHE=1 CUTE_DSL_KEEP=ptx,sass CUTE_DSL_DUMP_DIR=<dir> PYTHONPATH=<tree> python compile_dump.py
/usr/local/cuda/bin/cuobjdump -sass <dir>/*.cubin      # control bits decoded with a small script (scratch)
# GPU tests on the fixed tree
cd ~/h3-tp2-race/fixed && CUTE_DSL_NO_CACHE=1 PYTHONPATH=~/h3-tp2-race/fixed timeout 290 nice -n 10 \
  ~/h3-long/.venv/bin/python -m pytest -q -x tests/test_fp8_cutlass_gpu.py -p no:cacheprovider
```
