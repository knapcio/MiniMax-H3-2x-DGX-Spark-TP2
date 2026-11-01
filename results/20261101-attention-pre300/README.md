# Isolated attention probe on Spark_06, 2026-11-01

After both nodes rebooted, an idle-pair guard ran `scripts/attention_probe.py` on the head before any new render.
The probe used NVIDIA GB10, PyTorch 2.13.0+cu130, cuDNN 9.20 and FlashInfer 0.6.18.post1. It compared non-causal
BF16 attention with 28 heads and head dimension 128, using synthetic Q/K/V. The long length, 43,392, is inferred
from an earlier GEMM shape and the model's Ulysses layout, not captured from a live attention invocation. Each
timing is a median of three measured calls after one warmup. The cuDNN function also makes its output contiguous,
matching the output layout returned by the two FlashInfer calls.

| Synthetic case | cuDNN median | FlashInfer FA2 | FlashInfer FMHA-v2 |
| --- | ---: | ---: | ---: |
| Sequence 43,392, Q/K scale 1 | 311.057 ms | 296.455 ms (4.7% less) | 499.106 ms (60.5% more) |
| Sequence 512, Q/K scale 1 | 0.095 ms | 0.080 ms | 0.130 ms |

For the long case, FA2's relative L2 error against cuDNN was 0.000677; FMHA-v2's was 0.000204. Neither comparison
had nonfinite values. The small 512-token controls also compare against FP32 math at Q/K scales 1, 4 and 16; all
had finite values. These are numerical diagnostics, not acceptance thresholds. FMHA-v2 requires FP16 QK reduction,
and FA2 differs from cuDNN as well. The timings do not demonstrate speed or quality of a full video, and no backend
or serving default was changed. A real H3 input shape, representative activations, two-rank end-to-end timing and
visual/audio review are required before considering any change.

The [raw result](attention-result.json) contains all samples, warmup times and error measures; its SHA-256 is
`395cf20aa6430749d40650d5b72bb54ab7686f06125f06414368eac75b7090ce`. The generated
[report](attention-report.md) was reproduced from the checked JSON. The FlashInfer JIT workspace was isolated and
compilation limited to two jobs. Both node kernel journals were checked after the probe; no Xid was observed in
the current boots. This single-run microbenchmark does not establish long-run reliability.

## Backend selection and memory audit

The later on-disk H3 source selects FlashInfer `cute-dsl`, not FA2, when its backend setting is `auto` and GPU
capability 12.1 is successfully read. Both Sparks reported GB10 capability 12.1. The FA2 microbenchmark above
used `single_prefill_with_kv_cache`; H3's inspected FlashInfer implementation uses a ragged batch-prefill wrapper,
so even explicitly selecting FA2 would not reproduce the same measured call automatically.

A CPU/source experiment checked a per-role configuration that selects `FLASHINFER_ATTN`/`fa2` for the DiT `self`
role while keeping the exact token-refiner role and unmatched roles on CUDNN. An unlisted role whose category is
`self` also selects FlashInfer. The config AST passed five tests and an independent 20-check review; actual CLI
parsing, running role resolution and both TP2 ranks have not been tested. No serving setting was changed.

Matching FL2VA configuration files on both hosts specify 50 DiT layers and two token-refiner layers per rank. In
the inspected source, each selected FA2 instance has a 128 MiB device workspace plus an 8 MiB device and an 8 MiB
pinned-host wrapper buffer. Fifty independent instances would account for **7.03125 GiB per rank** of known
constructor buffers on the unified-memory Sparks. This is a source-level calculation, not measured peak memory
or net increase over CUDNN. Activations, masks, planning, compilation and allocator effects remain unmeasured.
The candidate needs guarded real-allocation, mask/numerical and full-video A/B tests before any speed or quality
claim. Workspace and audit evidence are in the enclosing workspace under `diagnostics/h3-attention-config-audit-20261101/`
and `diagnostics/h3-attention-fa2-memory-review-20261101/`.
