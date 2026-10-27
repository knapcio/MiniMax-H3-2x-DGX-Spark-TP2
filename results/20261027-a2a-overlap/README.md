# Ulysses all-to-all overlap (`H3_TP2_A2A_OVERLAP=2`): bit-identical, slower. Rejected.

5 s 1344x768, 20 steps, seed 1101, final two-Spark configuration, fresh boot with compacted memory.

| | stock Ulysses | overlap, 2 head groups |
|---|---|---|
| Wall clock | 643.0 s | 674.8 s (+4.9 %) |
| Denoising phase | 598.9 s | 625.8 s (+4.5 %) |
| Peak (X-Peak-Memory-MB) | 75.2 GB | 75.2 GB |
| framemd5 | 8b3e06a8 | 8b3e06a8 (identical) |
| Time inside the all-to-all collectives | ~2.9 % of denoising (measured at 20 s) | 264.6 s of 625.8 s on the comm stream |

The frames are identical, so the cuDNN kernel computes the same thing for 14 heads as for 28. The time is worse: NCCL
kernels issued on a second stream do not run alongside the attention kernel on the GB10, they queue behind it and slow it
down (the collectives took 264.6 s on the comm stream instead of the ~18 s they take inline). The first version of the
patch (buffers `record_stream`ed across streams) also ran the head out of memory within 10 s, because the CPU enqueues
tens of layers ahead and every cross-stream block stayed pinned until the GPU passed its event; the measured version
allocates all cross-stream buffers on the compute stream instead.
