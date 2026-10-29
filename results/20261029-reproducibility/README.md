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
