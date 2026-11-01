# Checkpoint input fingerprint: avoid a full Python byte copy

The experimental continuation checkpoint fingerprints each input tensor. The previous implementation converted
the contiguous byte view to a Python `bytes`, temporarily duplicating its payload. The new implementation feeds
the same contiguous NumPy buffer to SHA-256 through a `memoryview`; it retains the existing framing and digest.
This checkpoint code is **not wired into serving** and does not change the currently running render.

On the Mac, Python 3.12.14 / PyTorch 2.9.1 CPU, three fresh-process samples per implementation used one 128 MiB
contiguous float32 tensor. All six fingerprints were identical. Median measured high-water RSS increase after
allocating the input fell from 128.81 MiB to 0.80 MiB; median hashing time was 0.0506 s before and 0.0405 s after.
See [`mac-cpu-result.json`](mac-cpu-result.json) for each sample and code hashes. This small CPU microbenchmark does
not measure checkpoint writes, GPU memory, full request preparation, or end-to-end video generation.

Reproduce in an environment containing PyTorch, NumPy and safetensors:

```bash
python results/20261101-checkpoint-hash-memory/bench.py --before-ref cd1d42c --mib 128 --repeats 3
```

The checkpoint CPU suite passed 26 tests, including a known pre-change digest for mixed/non-contiguous tensor types,
a Python-allocation bound, and fresh two-rank synthetic restart/fallback tests. Real GPU/NCCL recovery, actual model
inputs and production integration are still unverified.
