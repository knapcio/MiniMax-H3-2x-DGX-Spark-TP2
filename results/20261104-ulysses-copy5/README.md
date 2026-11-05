# 2026-11-04: Ulysses exact exchange without copy 5, the out-projection quant reads the receive buffer (`H3_TP2_ULYSSES_EXACT=2`)

Follow-up of `results/20261103-ulysses-exact` ("What is left": copy 5 stays unless the out-projection's activation quant
reads the receive buffer). **Status: implemented, opt-in (`H3_TP2_ULYSSES_EXACT=2`; the default stays `1`), bitwise on
the GPU offline (9 GPU tests incl. every bf16 row maximum, 27 with the exact-exchange suite; 2.36-2.42 ms saved per layer and rank at M = 18880, ~0.12 s per step at the 1344x768 calibration, ~0.6 %); serving proof not yet run (recipe below).**

Spark_06, GB10 sm_121, torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.30.0 (`~/h3-long/.venv`), vLLM-Omni v0.30.0 `a8576ccb7`.
All GPU work from a scratch copy (`/tmp/copy5`), only while `~/h3-tp2-results/gpu-free` existed; serving untouched,
clocks untouched.

## What copy 5 is and who reads it

With `H3_TP2_ULYSSES_EXACT=1` the reverse exchange ends with one copy per attention layer and rank
(`ulysses_exact.exchange_back`): receive [P, M, 28, 128] -> new [1, M, 56, 128], 14336*M bytes written and read
(270 MB each way at M = 18880, the 5 s 1344x768 calibration's rows per rank). The only reader of that tensor is the
out-projection's dynamic per-token FP8 quant, through views only:

`_run_packed_attention` (`minimax_h3_transformer.py:628`: `.squeeze(0)`) -> `MiniMaxH3Attention.forward` (`:702-703`: `out.reshape(total, 7168)`,
`self.out_proj(out)`) -> `RowParallelLinear.forward` (tp 1, input parallel) -> `Fp8PerTensorOnlineLinearMethod.apply`
(`quantization/online/fp8.py:228`, the online FP8 method the head journal reports with
`CutlassFP8ScaledMMLinearKernel`) -> `FP8ScaledMMLinearKernel.apply_weights` (`ScaledMMLinearKernel.py:135`:
`x.view(-1, K)`, `self.quant_fp8(x_2d, None, None)`) -> `ops.cutlass_scaled_mm` (our dispatcher, the CuTe-DSL `out`
site).

Row m of the quant's input is recv[0, m] followed by recv[1, m] (3584 contiguous bf16 each). Per-token quant only needs
the row's abs-max and an elementwise map, so it can read those two segments where they lie.

## Design (`h3_tp2/ulysses_outq.py`)

- `ulysses_exact.exchange_back(..., defer=True)` (mode 2): after the reverse all-to-all, if the running
  `MiniMaxH3Attention` was armed for it, return a stride-0 placeholder of the stock shape [1, M, 56, 128] and keep the
  receive buffer as the pending deferral; otherwise make copy 5 as mode 1 does. The model's `.squeeze(0)` /
  `.reshape(total, 7168)` / `.view(-1, 7168)` keep the placeholder's pointer (strides (0, 0)).
- The out-projection's `quant_fp8` (on that layer's kernel instance only) is wrapped by `_OutQuant`: the placeholder
  (same pointer and [M, 7168] shape) is quantised from the receive buffer by `_quant_rows_kernel`; any other input goes
  to the original `QuantFP8` unchanged. The result has stock's shapes, strides and dtypes (fp8 [M, 7168] contiguous,
  fp32 [M, 1]), so `apply_scaled_mm` and the CuTe-DSL routing see the same operands.
- `_quant_rows_kernel`: one program per row, a [2, 4096] tile (3584 valid per segment) loaded from
  `recv + p*M*3584 + m*3584`; the arithmetic is the Inductor kernel's (`quant_inductor.py` here): amax = NaN-propagating
  `max2` of `abs` (a max: exact and order-free), scale = `maximum(amax / 448, 4.359654e-06)` with 448 a runtime fp32
  operand, `1.0 / scale`, `x * inv`, `minimum(maximum(., -448), 448)`, `.to(float8e4nv)`; the same Triton expressions
  and helpers (`triton_helpers.maximum/minimum/max2`, `tl_math.abs`), so the same compiler emits the same instructions
  on the same values.
- Which quant serving runs: under the diffusion `VllmConfig(compilation_config=CompilationConfig())` (mode 3, backend
  inductor, custom ops `['none']`) `QuantFP8` is disabled as a custom op, so `CustomOp.maybe_compile` compiles
  `forward_native` with `dynamic=True`; checked on Spark_06 (`_forward_method` is the compiled `forward_native`,
  `VLLM_BATCH_INVARIANT` False). The GPU tests and the bench build the quant exactly that way.

Guards (`prepare`, once per attention): `out_proj` is exactly `RowParallelLinear` (no LoRA wrapper), tp 1, input
parallel, width 56*128; method exactly `Fp8PerTensorOnlineLinearMethod`; kernel an `FP8ScaledMMLinearKernel` with the
base `apply_weights`; `QuantFP8` dynamic per token, no padding, e4m3fn; no static input scale or bound; no
batch-invariant mode; and the reviewed source of the five functions on the path (`EXPECTED`: attention forward, row
linear forward, online apply, apply_weights, `QuantFP8.forward_native`; checked once at hook time, all match on
Spark_06). Anything else keeps copy 5 for that layer (logged once). A placeholder reaching a quant without its buffer,
or a deferral that the attention returns without consuming, raises (never computes on garbage).

Self-check: the first deferral of each process runs stock (copy 5 + the layer's own compiled `QuantFP8`) and the
receive-buffer kernel on the full real tensor and compares FP8 bytes, scale bits, shapes and strides; a mismatch, a
compile error or a launch error disables mode 2 for the process (copy 5 again, logged) and that call returns stock's
result.

## Measurements

Spark_06 GPU, 2026-11-05 02:25-02:40 (gpu-free marker present, serving head idle, nothing restarted), random bf16
rows (timing does not depend on values; the outputs were compared bitwise in the same run), CUDA events, median of 30
(`bench1.json`) and 50 (`bench2.json`) iterations, one stream. Times in ms per attention layer and rank:

| M (rows per rank) | copy 5 | stock quant (compiled QuantFP8, contiguous) | stock = copy 5 + quant | receive-buffer quant (8 warps) | same kernel on the contiguous tensor | saved per layer | saved per DiT forward (50 layers) | transient peak stock / mode 2 |
|---|---|---|---|---|---|---|---|---|
| 10240 | 1.22 / 1.25 | 1.02 / 1.04 | 2.24 / 2.27 | 0.937 / 0.953 | 0.948 / 0.961 | **1.30 / 1.32** | **65 / 66 ms** | 210 / 70 MB |
| 18880 | 2.22 / 2.26 | 1.86 / 1.87 | 4.12 / 4.13 | 1.75 / 1.71 | 1.93 / 1.70 | **2.36 / 2.42** | **118 / 121 ms** | 388 / 130 MB |

(two runs per cell: bench1 / bench2; 4 and 16 warps within 0.05 ms of 8, so 8 stays.) Copy 5 runs at 240 GB/s. The
quant reading the two 7 KB segments of the receive buffer is as fast as reading a contiguous row (within run-to-run
noise both ways) and 0.1 ms faster than the Inductor kernel (one load of the row instead of two passes), so the
saving is copy 5 plus a little.

Per step at the 5 s 1344x768 calibration (M = 18880 per rank, 50 DiT layers, one forward per step: the checkpoint is
cfg-distilled): **~0.12 s of ~20.2-21.2 s/step = ~0.6 %**, ~2.4 s per 20-step request. Both ranks save the same, in
parallel. At the 960x544 probe (M = 10240): ~66 ms of ~8.2 s/step (~0.8 %). The saving grows linearly with M. The
transient memory of the out-projection input drops by 14336*M bytes (258 MB at M = 18880).

Commands (Spark_06, scratch copy, serving env sourced for the Python headers Triton needs):

    rsync -a --exclude .git ./ Spark_06:/tmp/copy5/          # from the worktree
    test -e ~/h3-tp2-results/gpu-free && cd /tmp/copy5 && set -a && . ~/.h3-tp2.env && . /tmp/copy5/scripts/env.sh && set +a
    export PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=/tmp/copy5
    ~/h3-long/.venv/bin/python scripts/ulysses_outq_bench.py --m 10240 18880 --json bench1.json
    ~/h3-long/.venv/bin/python scripts/ulysses_outq_bench.py --m 10240 18880 --iters 50 --json bench2.json
    ~/h3-long/.venv/bin/python results/20261104-ulysses-copy5/rounding_variants.py

## What is bit-identical

Against the stock path, copy 5 + the serving `QuantFP8` (built under the diffusion config, i.e. Inductor-compiled
`forward_native`, `dynamic=True`; the test asserts that it is the compiled function), compared byte for byte (FP8
values, fp32 scale bits, shapes, strides, dtypes):

- random rows, M = 4096, 4100, 18880; attention-output rows (SDPA of random q/k/v, two ranks' shards), M = 4096, 10240;
  directed rows (zeros, -0.0, bf16 subnormals, 1e36, +-inf, NaN, maxima on both sides of the segment boundary, bf16 max,
  one empty segment, exact 448): **bit-identical**;
- **every** non-negative bf16 value 0x0000..0x7F80 and NaN as a row's abs-max (32642 rows, random elements below it):
  bit-identical. amax is a bf16 value, so this covers every scale and every reciprocal the quant can compute; the
  per-element part is one fp32 multiply, the clamp and the fp8 cast, with the same instructions;
- the bench's timed tensors at M = 10240 and 18880: bit-identical;
- the kernel on the receive buffer = the same kernel on the contiguous tensor = stock;
- the self-check (`_Deferred.quantize` on the GPU with the real quant): passes and enables the receive path.

Sensitivity (`rounding_variants.py`, the same exhaustive rows, kernel variants that differ only in rounding):

| variant | scales differing (of 32640) | rows with a differing FP8 value |
|---|---|---|
| as shipped (Triton `/` with a runtime fp32 448, as Inductor emits) | 0 | 0 |
| `amax * fp32(1/448)` | 0 | 0 |
| IEEE `div_rn` for the scale | 9727 | 1372 |
| IEEE `div_rn` for the reciprocal | 0 | 776 |

So the check distinguishes the rounding paths: Triton's `/` is not IEEE division on sm_121 (it agrees with a multiply
by the rounded reciprocal on every bf16 input here). An eager torch quant (IEEE division) would **not** be bit-equal
to serving. That is why the shipped kernel copies the Inductor expressions and why the runtime self-check compares
against the layer's own compiled `QuantFP8`, not against an eager mirror. (The CPU tests check the layout and the
wiring against the eager mirror only.)

## Forward side: copy 3 (the v pack)

Since `H3_TP2_FP8_GEMM=cutlass` became the default (2026-11-04), `fp8_consumers` drops its qkv site (the CuTe-DSL
kernel writes qkv in bf16), so `ulysses_exact` runs with the upstream producer (journal: `producer = upstream
norm+RoPE (q and k written per destination)`, `installed sites: fc2`). q and k are still written straight into their
send buffers, but **v is packed by stock's copy** (copy 3: the strided [M, 56, 128] view of the qkv output, row stride
21504, into send_v [P, M, 28, 128]). The bench times it at 1.32-1.34 ms (M = 10240) and **2.43-2.49 ms per layer**
(M = 18880), the same size as copy 5 (~0.12 s per step at 1344).

It does not have copy 5's structure. Its consumer is NCCL (`all_to_all_single` needs one contiguous send buffer) and its
producer is the qkv GEMM, so the only copy-free form is the GEMM writing v's columns per destination into send_v (an
epilogue with a second output, or the v columns as separate N = 3584 launches into `send_v[p]`, bit-identical per
element only if the kernel's K loop does not depend on N). That is `h3_tp2/fp8_cutlass*.py` work, owned by another
agent tonight, so it was not touched. Under `H3_TP2_FP8_GEMM=tensor32` with the consumer fuse, v is already written per
destination (mode 1 removes copy 3 there).

## Tests

- `tests/test_ulysses_outq.py` (CPU; Spark_06 torch 2.13 with vLLM importable): 14 tests. The kernel's addressing
  (eager mirror through `as_strided` with the kernel's strides) vs copy 5 + quant on random and directed rows; the
  mirror vs vLLM's own `QuantFP8.forward_native` on CPU; the five reviewed fingerprints vs the installed sources; two
  thread-ranks: `exchange_back(defer=True)` -> placeholder -> the model's views -> wrapped quant, vs stock all-to-all +
  copy + quant (one all-to-all per call, as stock); the whole mode-2 attention (redirected rope producer, exact exchange,
  deferral, out-projection quant) vs stock; the self-check returns stock and disables on a wrong result or an exception,
  after which copy 5 comes back and real tensors pass through; refusals (unarmed, row mismatch, non-bf16, head count,
  unprepared layer, one deferral per forward); guards (a placeholder without its buffer or in another shape raises, an
  unconsumed deferral raises at the end of the forward, not while another exception propagates); mode 1 unchanged.
- `tests/test_ulysses_outq_gpu.py` (GPU): 9 tests, the list above.
- With `tests/test_ulysses_exact.py`, `tests/test_ulysses_exact_gpu.py` and `tests/test_ulysses_nocopy.py` in the same
  session on Spark_06: **69 passed** (18 s). `test_ulysses_exact.py`'s bad-mode test now uses "3" ("2" is a mode).

    cd /tmp/copy5 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/tmp/copy5 ~/h3-long/.venv/bin/python -m pytest tests/test_ulysses_outq.py tests/test_ulysses_exact.py tests/test_ulysses_nocopy.py -q
    # GPU (marker present, env as above)
    ~/h3-long/.venv/bin/python -m pytest tests/test_ulysses_exact_gpu.py tests/test_ulysses_outq_gpu.py -q -p no:logging

## Serving proof: PASS (queue73, 2026-11-05 02:31-03:49)

Branch rebased on main (4a1962c) and synced to both Sparks; gated compacted boots through `queue-lib.sh`
(`~/h3-tp2-results/queue73-copy5-proof.sh`):

| arm | boot probe | 5 s 1344x768 T2VA calibration (seed 0, 20 steps) | s/it |
|---|---|---|---|
| A: defaults | PASS | latents `4d6d69e4aad0` | 20.44 |
| B: `H3_TP2_ULYSSES_EXACT=2`, run 1 | `bf6e85687a6a` (PASS) | `4d6d69e4aad0`, `latents_diff.py` A vs B1: video and audio bit-equal (max abs diff 0) | 20.81 |
| B, run 2 | | `4d6d69e4aad0` | 20.40 |
| E: defaults restored | PASS 8.22 / 7.78 s/step `bf6e85687a6a` | | |

Journal of both ranks in B: `Ulysses exact exchange installed (mode 2)`, `copy 5 left to the out-projection quant
(self-check on the first call)`, `copy5: first deferred`, `copy5: self-check M=10240 K=7168 bit-identical to copy 5 +
QuantFP8; the out-projection quant reads the receive buffer`, `copy5: first quant_recv`, and at the end of the boot
`calls so far deferred=2001 quant_recv=2000 selfcheck_pass=1` (the probe + two calibrations: 1000 attention calls
each on each rank; the one extra deferral is the self-check's own call). Absent: `NOT bit-identical`, `not_prepared`,
`upstream source changed`, `out-projection not eligible`. Speed: 20.81 / 20.40 against 20.44 s/it, i.e. inside the
boot-to-boot spread, as expected for a 0.6 % lever; the digest and the bit-equal latents are the proof. One stray 60 s
render from a test suite hit the head during the boot-B restart (03:04-03:08, the head was SIGKILLed by the stop
timeout and rebooted by the gate); it started after calibration A had finished and before boot B's probe, so no arm
was affected.

The default flip was gated with `H3_TP2_FP8_QUANT_FUSE` together (`results/20261105-levers-merge/`, queue74/75): PASS,
`H3_TP2_ULYSSES_EXACT` defaults to `2` since 2026-11-05 (`=1` restores copy 5).

### The recipe as planned

Like queue59: sync the reviewed tree to both Sparks while nothing renders (`H3_VENV=/home/knapcio/h3-long/.venv
scripts/sync-to-nodes.sh Spark_06 Spark_05`), then gated compacted boots via `queue-lib.sh` (`H3_TP2_ULYSSES_EXACT` is
already in its STRIP list, so no tooling change is needed):

- A: defaults (`H3_TP2_ULYSSES_EXACT` unset = 1), 5 s 1344x768 T2VA calibration, seed 0, 20 steps (the queue58 `calib`
  with `PROMPT` = the forest stream): digest `4d6d69e4aad0`, s/step.
- B: `restart_head_gated` with `DEFAULT_ENV` + `export H3_TP2_ULYSSES_EXACT=2`. Boot probe latents `bf6e85687a6a`, then
  the calibration twice: both digests `4d6d69e4aad0`, `latents_diff.py A B1` bit-equal, s/step from the second.
- E: defaults restored.

Pass criteria for B, from the head journal (rank 1 in `~/h3-tp2-logs` on Spark_05): `Ulysses exact exchange installed
(mode 2)`; `ulysses exact: copy 5 left to the out-projection quant`; `copy5: first deferred`; `copy5: self-check
M=... K=7168 bit-identical to copy 5 + QuantFP8`; `copy5: first quant_recv`; no `self-check ... NOT bit-identical`, no
`out-projection not eligible` (except possibly once for the token refiner's attention, which falls back to the stock
forward anyway), no `first not_prepared`, no `upstream source changed`. Speed: ~-0.6 % expected, below the 1.2 %
boot-to-boot spread, so the speed claim needs a same-boot-quality A/B pair (or the a2a/copy timing profile). The digest
is the proof that matters.

## What is left

- Serving proof (above). Then a default gate like queue63 (16 s continuation) if the owner wants it on by default.
- Copy 3 (v pack) under the cutlass default, ~2.4 ms per layer at M = 18880: needs the qkv CuTe-DSL GEMM to write v
  per destination into send_v (fp8_cutlass territory, see "Forward side").
- After both, no permute copy is left on the Ulysses path at batch 1. The remaining exchange cost is the four
  all-to-alls themselves.
