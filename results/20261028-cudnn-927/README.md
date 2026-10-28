# cuDNN 9.27.0.42 instead of torch 2.13's pinned 9.20.0.48: -4.5 %, not bit-identical (candidate, not default)

Ref2VA, 5 s 1344x768, 20 steps, seed 1101, bench prompt, reference short edge 1024, first request after a boot in both
runs. `pip install nvidia-cudnn-cu13==9.27.0.42` in the v0.30 venv on both Sparks (pip warns that torch 2.13.0+cu130
pins 9.20.0.48), reverted to 9.20.0.48 afterwards.

| | cuDNN 9.20.0.48 (pinned) | cuDNN 9.27.0.42 |
|---|---|---|
| Wall clock | 752.9 s | 718.7 s (-4.5 %) |
| Denoising / decode | ~690 / 47 s | 665.8 / 42.9 s |
| Peak | 63.2 GB | 63.3 GB |
| framemd5 | 7b9e4c16 | 8c6d4efe |
| SSIM / PSNR between the clips | | 0.848 / 28.4 dB |

The frames differ (a different attention kernel or plan: same class of difference as the pair against one Spark,
SSIM 0.906, and far above two different samples, 0.776), and the gain is 4.5 % in one sample. Attention is 39 % of a
step, so the library change moved the attention kernel by roughly a tenth; which kernel 9.27 picks on sm121 was not
profiled. Kept out of the recipe: it breaks torch's version pin and is not output-preserving. Documented as an
opt-in for speed-first installs once a second sample confirms it.
