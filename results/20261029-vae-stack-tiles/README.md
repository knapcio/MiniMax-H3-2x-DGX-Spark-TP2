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
