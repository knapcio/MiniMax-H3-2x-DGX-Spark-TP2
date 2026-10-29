# Fused FP8 epilogue + SwiGLU in the real model: level 2 is not deterministic and buys ~1 %; level 1 rerun pending

5 s 960x576 Ref2VA, 20 steps, seed 1101, two-pass GEMM on in every arm. Reference: the two-pass-only renders
(`20261028-fp8-two-pass-ab/`, framemd5 8e58f759, warm 451.6 s, denoising 413.2 s).

| Arm | Wall | Denoising | framemd5 | Peak |
|---|---|---|---|---|
| two-pass only (warm) | 451.6 s | 413.2 s | 8e58f759 | 64.6 GB |
| + fused SwiGLU level 1 (first try) | failed at once | | | |
| + fused SwiGLU level 2, run 1 | 458.5 s | 408.5 s | cd1bcf1d | 64.4 GB |
| + fused SwiGLU level 2, run 2 | 449.3 s | 408.8 s | **85437ffa (= the stock CUTLASS render's digest)** | 64.4 GB |

The level-1 first try died on a bug in the patched forward (a deleted local referenced on that path; the unit tests
had only checked the kernels, not the forward). Level 2 ran, but its two identical requests produced two different
clips, and the second one is bit-identical to the stock-kernel render, which the two-pass path never produces. So in
the real model the level-2 path does not do what the unit tests verify in isolation; the likely reason is that the
DiT blocks run inside torch-compiled regions and the patched forward (thread-local flags, a stash handed from the
GEMM dispatcher to the MLP) is not safe under tracing. The denoising gain is 1 % at best (408 against 413 s), far below
the traffic estimate, so there is nothing to chase here: level 2 is withdrawn (the switch stays for experiments and is
documented as not usable), level 1 gets one rerun with the fixed forward and is withdrawn too unless it reproduces the
two-pass digest with a measurable gain.
