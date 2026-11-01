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
