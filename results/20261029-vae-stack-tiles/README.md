# VAE batched tile decode (`stack_tiling`), first try: -34 % decode, +32 GB peak, and it changed the clip

`H3_TP2_VAE_STACK_TILES=1` (the checkpoint's own `stack_tiling` flag, batched tiles per rank), 5 s 960x576 Ref2VA, 20
steps, seed 1101, stock GEMM, same request as `v030-fp8-base-2`.

| | baseline | stack_tiling (all tiles in one batch) |
|---|---|---|
| Wall clock | 592.3 s | 602.3 s |
| VAE decode | 24.5 s | 16.2 s (-34 %) |
| Peak (X-Peak-Memory-MB) | 63.0 GB | **95.2 GB** |
| framemd5 | 85437ffa | 8776c2bc |
| SSIM / PSNR vs baseline | | 0.79 / 24.5 dB: a different scene |

Two surprises. The peak: the checkpoint batches every local tile into one decoder call, so the decoder's activations
scale with the tile count (12+ tiles per rank at this size); a 1344x768 run would not have fit and was cancelled. The
clip: the flag also switches the VAE *encoder's* tile path (the reference image is encoded through the same
`_run_tile_tasks`), so the reference latents changed by the batched encoder's algorithm selection and the diffusion
produced a different sample. As implemented this is not a decode-only lever. The next version toggles `stack_tiling`
only around the decode and caps the tiles per decoder call; then the diffusion stays bit-identical and only the
decoder's own algorithm choice can differ (NVlabs measured 71 dB for that).

## Second version: decode-only, at most 4 tiles per decoder call (`H3_TP2_VAE_STACK_TILES=1`, `H3_TP2_VAE_TILE_BATCH=4`)

5 s 960x576 Ref2VA, 20 steps, seed 1101, two-pass GEMM default (repo 82b6564). The reference encode keeps the stock
per-tile path; the flag is raised only around `decode_latent` / `decode_with_chunks`.

| | stock decode (`v030-fp8-base-2`, framemd5 85437ffa) | batched decode, cap 4 (`v030-vaestack2-960`) |
|---|---|---|
| Wall clock | 451.6 s (two-pass, old digest run) | 446.1 s |
| VAE decode | 24.5 s | 15.9 s (-35 %) |
| Peak | 64.6 GB | 64.7 GB |
| framemd5 | 85437ffa | b0981821 |
| PSNR / SSIM vs stock decode | | **45.1 dB / 0.978: the same clip**, decoder-level differences only |

The first try's "different scene" (24.5 dB) came from the batched *encode* changing the conditioning; with the encode
untouched the diffusion sample is the one the stock decode produces (the two-pass default renders 85437ffa at this
cell since round 8, see the note below) and only the decoder's own arithmetic differs (batched tiles pick other
cuDNN/cuBLAS algorithms; NVlabs measured 71 dB for the same change in their decoder, ours is 45 dB on uint8 frames).
A latents digest per render (`h3_tp2 latents sha256`, since aea1aac) is the direct check and runs in the
reproducibility chain (queue12b, arm D). 1344x768 below.

Note on digests: the two-pass digest 8e58f759 used as "the" reference during the night was rendered before round 8
changed the kernel epilogue (ff9c01e, 2026-10-28 16:56); after that change the plain two-pass render at this cell is
byte-identical to the stock render (85437ffa). Every "must equal 8e58f759" comparison after 16:56 was against a stale
baseline; the queue12b chain re-establishes same-boot, cross-boot, batched-decode and fused-SwiGLU digests on one
revision.

## 1344x768 (reference 1024): decode -36 %, but the sample changed between the two boots

| | baseline (`v030-t32-ref1024-1344`, 08:38 boot) | batched decode, cap 4 (`v030-vaestack2-1344`, 09:09 boot) |
|---|---|---|
| Wall clock | 555.9 s | 546.5 s |
| Denoising / VAE decode | 503.0 / 42.8 s | 510.3 / 27.2 s (-36 %) |
| Peak | 64.3 GB | 64.0 GB |
| framemd5 | c86f7279 | 5b4a3f5a |
| PSNR / SSIM vs baseline | | 29.0 dB / 0.873 from the first frame on: **a different sample** (the tree stands elsewhere) |

The decode-only wrapper cannot touch the latents, and at 960x576 the same wrapper reproduced the stock clip (45 dB)
across boots a day apart. So at 1344x768 the diffusion itself differed between these two boots, with the same code,
request and seed. Both renders predate the latents digest; the queue12c chain (two plain 1344 renders in one fresh
boot, then the batched decode in a further boot, every render with its latents digest) attributes it: within-boot vs
cross-boot vs the decode flag. Until then the 1344 decode gain (-16 s per 5 s clip) is measured but the flag is not a
default.
