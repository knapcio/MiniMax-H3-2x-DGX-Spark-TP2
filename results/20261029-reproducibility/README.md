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
