# Fused FP8 epilogue + SwiGLU in the real model: level 1 bit-identical and faster (chain of 2026-10-29); the level-2 verdict below was against a stale digest

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

## Superseded (2026-10-29 11:00): the reference digest above was stale

8e58f759 was rendered before round 8 changed the two-pass epilogue (ff9c01e, 16:56); since then the plain two-pass
render at this cell is 85437ffa, the stock kernel's clip. So the level-2 run 2 above (85437ffa) *was* the plain digest,
and only run 1 (cd1bcf1d) differed. Level 1 with the fixed forward, in the reproducibility chain
(`../20261029-reproducibility/`): latents and frames identical to the plain two-pass render, first-request denoising
389 s against 442-447 s plain (-12 %; -6 % against the plain warm request). Level 2 keeps its withdrawn status until it
gets the same pair test (one run of two differed from the plain digest, which is still unexplained).
