# Reproducibility chain at 5 s 960x576 (repo 82b6564 + aea1aac latents digest): every arm bit-identical

After finding that the two-pass digest 8e58f759 (renders of 2026-10-28 16:42/16:50) predates round 8's epilogue
change (ff9c01e, 16:56), the night's "not identical to 8e58f759" verdicts had to be redone against a baseline rendered
on the current revision. One chain on spark-06 (`queue12b.sh`), every render Ref2VA 5 s 960x576, 20 steps, seed 1101,
reference image 2048 px, the bench prompt; the head restarted between arms (the agent on spark-05 untouched); the
diffusion output hashed before the VAE decode (`h3_tp2 latents sha256`).

| Arm | Boot | Latents sha256 | framemd5 | Denoising | Wall | VAE decode | Peak |
|---|---|---|---|---|---|---|---|
| A1 two-pass (default) | 1, first request | 50132300 | 85437ffa | 441.7 s | 495.1 s | 25.1 s | 64.7 GB |
| A2 two-pass | 1, second | 50132300 | 85437ffa | 416.0 s | 460.6 s | 24.3 s | 64.6 GB |
| B two-pass | 2, first | 50132300 | 85437ffa | 447.1 s | 500.7 s | 25.2 s | 64.7 GB |
| D two-pass + batched tile decode (cap 4) | 3, first | 50132300 | b0981821 | 440.7 s | 475.9 s | **16.0 s** | 64.7 GB |
| E two-pass + fused SwiGLU level 1 | 4, first | 50132300 | 85437ffa | **389.3 s** | 437.0 s | 25.1 s | 64.7 GB |
| C stock CUTLASS (`H3_TP2_FP8_GEMM=stock`) | 5, first | 50132300 | 85437ffa | 539.1 s | 590.3 s | 25.0 s | 62.8 GB |

Findings:

- **Within a boot and across boots the diffusion is deterministic** at this cell: one latents digest for six renders
  in five boots, and the two-pass kernel (since round 8), the fused SwiGLU level 1 and the stock CUTLASS kernel all
  produce it. The 85437ffa clip is the clip the stock kernel rendered on 2026-10-28 (`v030-fp8-base-2`).
- **Fused SwiGLU level 1 is bit-identical** to the plain two-pass path in the real model (the night's "level 1 failed"
  was the `UnboundLocalError`, fixed; the "level 2 non-deterministic" verdict compared against the stale digest and is
  reopened). Its first-request denoising, 389 s, is 12 % under the plain first requests (442-447 s) and 6 % under the
  plain warm request (416 s); a warm-pair chain (`queue12d.sh`: fuse x2, plain x2, fuse x2 in separate boots) measures
  the gain without the first-request effect before it becomes a default.
- **Batched tile decode is a pure decoder change**: identical latents, frames b0981821 (the same frames as the earlier
  `v030-vaestack2-960` render in another boot, so the batched decoder is deterministic too), 45.1 dB / SSIM 0.978 to the
  stock decode (see `../20261029-vae-stack-tiles/`), decode 25 -> 16 s.
- First request after a boot denoises 6-7 % slower than the second (442-447 vs 416 s); the chain's timings are
  first-request timings except A2.
- At 1344x768 (reference 1024) two renders in different boots gave different samples (`../20261029-vae-stack-tiles/`,
  29 dB); both predate the latents digest. `queue12c.sh` (F/G in one fresh boot, H batched decode in a further boot)
  repeats that cell with the digest.

## 1344x768, reference 1024 (`queue12c.sh`): not deterministic, and not boot-bound either

| Arm | Boot | Latents sha256 | framemd5 | Denoising | Wall |
|---|---|---|---|---|---|
| `v030-t32-ref1024-1344` (08:38 boot) | a, first | (predates the digest) | c86f7279 | 503.0 s | 555.9 s |
| `v030-vaestack2-1344` (09:09 boot, batched decode) | b, first | (predates the digest) | 5b4a3f5a | 510.3 s | 546.5 s |
| F two-pass | c, first | e474c9ac | 8a129b58 | 521.2 s | 575.1 s |
| G two-pass | c, second | **591e4c0d** | 7b9e4c16 | 594.9 s | 641.5 s |
| H two-pass + batched decode | d, first | **591e4c0d** | 9eb3db3d (45.2 dB to G: decoder only) | 519.7 s | 547.5 s |

Two requests in one process (F, G) gave different latents; the next boot's first request (H) gave G's latents. So the
diffusion output at this cell takes one of (at least) two values independent of the boot, while at 960x576 (reference
2048) six renders in five boots agreed. The same prompt, image and seed; the prompt and media digests (logged since
11:30) are the next attribution step: `queue12e.sh` runs two renders per boot for reference 1024, reference 2048 and
the stock CUTLASS kernel (is it the two-pass GEMM: PyTorch's `scaled_gemm` sets no cuBLASLt reduction-scheme mask, so
a split-K algorithm with atomics or an alignment-dependent algorithm choice would do exactly this), and
`scripts/fp8-gemm-determinism.py` repeats the scalar-scale `_scaled_mm` on the DiT shapes over a range of M, also with
the inputs moved to other addresses. G's denoising was 14 % slower than F's in the same process (595 vs 521 s), the
second-request slowdown seen before at this cell; H, a first request, was back at 520 s.

### cuBLASLt scalar-scale GEMM: bit-stable on the DiT shapes (`fp8-gemm-determinism-sweep.txt`, spark-05, 11:57)

`scripts/fp8-gemm-determinism.py` on the four DiT shapes (qkv, out, fc1, fc2) for M from 36864 to 73728 rows: six
repeats on the same inputs plus six repeats with the inputs re-allocated behind random pads (other addresses) all
returned identical bits, and vLLM's CUTLASS per-row kernel too. Whatever flips the 1344x768 result, it is not the
two-pass GEMM's first pass on these shapes at these M (the exact M values of a 1344 request are logged by the next
boots; the stock-CUTLASS pair in `queue12e.sh` is the direct test).

## Fused SwiGLU level 1: warm pairs (`queue12d.sh`, 12:18-13:27, 5 s 960x576)

Three boots, two requests each, fused / plain / fused; every render framemd5 85437ffa (the fusion is bit-identical in
all six of its renders today, E, E2-E5, plus the earlier A/B).

| Arm | Boot, request | Denoising | Wall |
|---|---|---|---|
| E2 fused | 1, first | 385.4 s | 437.0 s |
| E3 fused | 1, second | 384.6 s | 418.7 s |
| A3 plain | 2, first | 405.7 s | 451.0 s |
| A4 plain | 2, second | 399.3 s | 437.7 s |
| E4 fused | 3, first | 405.8 s | 453.1 s |
| E5 fused | 3, second | 379.9 s | 418.7 s |

With the morning's renders: fused 380-406 s (median 385), plain 399-447 s (median 416): about -7 % denoising at the
median, -5 % on the warm pairs (384.6 / 379.9 vs 399.3), inside a boot-to-boot spread of about 10 % that is larger than
the gain itself (E4 at 405.8 s is a fused first request). The fusion removes one bf16 write+read of the 28672-wide fc1
output per MLP (2.1 GB at this M) and never changes a bit, so it becomes a default; the exact gain per request is
whatever the boot gives.

## 1344x768 stage attribution (`queue12e.sh`, 2026-10-29/30 night): the variation is inside the denoising loop

Every render logs the prompt embeddings (text encoder, a VLM that also sees the reference image), the encoded
reference media and the diffusion latents.

| Pair | Boot | prompt sha256 | media sha256 | latents sha256 | denoising |
|---|---|---|---|---|---|
| I1 ref 1024 | 1, first | 37b0266c | 45ad333f | 591e4c0d (= G, H) | |
| I2 ref 1024 | 1, second | 37b0266c | 45ad333f | 591e4c0d | |
| K1 ref 2048 | 2, first | f594cf56 | fa377e9b | d6a72400 | 844.7 s |
| K2 ref 2048 | 2, second | f594cf56 | fa377e9b | **749fa85d** | 786.9 s |

The prompt embeddings and the reference encoding are identical within a boot; the latents are not always: K1 and K2
differ with the same inputs, I1 and I2 agree, and F / G (11:11, same boot, ref 1024) differed. So the per-request
variation at 1344x768 sits in the denoising loop (DiT forward + sampler), not in the encoders; the stock-CUTLASS pair
(S1/S2) says whether the two-pass GEMM is involved. Four of the five ref-1024 renders with the digest gave 591e4c0d,
one gave e474c9ac: not a coin flip per request, a rare event. (The step-0 input digest hook never fired: the DiT
forward is not reached through the class attribute; to be fixed before the noise can be separated from the loop.)

| S1 ref 1024, stock CUTLASS | 3, first | 37b0266c | 45ad333f | **591e4c0d** (= two-pass I1/I2/G/H) | 734.6 s wall |
| S2 ref 1024, stock CUTLASS | 3, second | 37b0266c | 45ad333f | 591e4c0d | 922.4 s wall |

Stock and two-pass give the same latents and the same frames (framemd5 7b9e4c16) at 1344x768 with reference 1024,
as they do at 960x576: the two-pass GEMM is bit-identical to the CUTLASS kernel on these renders, and it is 22-40 %
faster here (two-pass 553-576 s, stock 735-922 s wall). The per-request variation seen in F/G and K1/K2 is a property
of the denoising loop that both kernels share; with reference 1024 it occurred once in seven renders, with reference
2048 in one of two. Nothing in the serving configuration causes it, and it does not degrade a clip; it only means
two identical requests may sometimes give two valid samples.
