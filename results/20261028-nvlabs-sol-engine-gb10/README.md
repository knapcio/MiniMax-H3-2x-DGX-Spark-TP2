# NVlabs Sana `sol-engine`: NVIDIA's own MiniMax-H3 port to one DGX Spark (read 2026-10-28, found by the owner)

https://github.com/NVlabs/Sana/tree/sol-engine (`models/minimax_h3/GB10/`, `config/minimax_h3/gb10_*.toml`,
`models/minimax_h3.toml` section `[gb10]`). Different stack from this recipe: Diffusers (PR 14355), the ComfyUI
`minimax_h3_fl2va_pruned_fp8_scaled` DiT (23 GB, static per-tensor FP8 scales from the checkpoint, fc2 weight-only),
Qwen3-VL-32B-Instruct-FP8 as the text encoder, one GPU, no context parallelism. Their measurements on a GB10 (832x480,
124 frames, 50 steps unless noted), quoted from `minimax_h3.toml`:

| Their row | ms per DiT step | Note |
|---|---|---|
| released BF16 model | 13417 | measured against their FP8 port: 0.08 % apart |
| FP8 eager baseline | 13576 | |
| + fused QKV | 12875 | bit-identical |
| + Triton quantiser (div_rn, rtne) | 11524 | bit-identical |
| + fused AdaLN (modulate, gate-add) | 10709 | bit-identical |
| + fused RoPE, SwiGLU (all lossless) | 9790 | 1.387x, bit-identical |
| Sol-Attn tau=1 (sparse) | 8157 | lossy (cos 0.9865 video) |
| FirstBlockCache 0.08 | 2.7x on the DiT | lossy |
| full-opt end to end | 710.6 -> 181.3 s | 3.92x, lossy |

Their 1344x768 x 124-frame cell is 38,247 packed rows at 40.1 s/step on one GB10 (unaccelerated FP8). Their measured
GB10 roofline: BF16 GEMM 100 TFLOPS, FP8 GEMM 188.3, flash attention 92.5 TFLOPS ("no lossless headroom left in
attention"). VAE: tile batching 1.47x on the video decode at PSNR 71.2 dB (max 1/255), untiled decode rejected.

What it means for this recipe:

- **FP8 GEMM ceiling confirmed.** Their 188.3 TFLOPS is the scalar-scale `torch._scaled_mm` figure; we measured 189.5
  for the same call and 59 for the per-row-scaled kernels vLLM uses. The two-pass GEMM (`20261028-fp8-two-pass-ab/`)
  brings the per-token-scaled path to that ceiling without giving up per-token scales (their checkpoint uses static
  per-tensor scales instead).
- **Attention is at the ceiling here too.** Our profiled step (`20261028-dit-step-profile/`) spends 10.48 s per rank in
  cuDNN attention; at the 960x576 5 s cell that is about 980 TFLOP of attention per rank (50 layers, 28 heads per
  rank, ~37k rows), i.e. ~93 TFLOPS: their measured flash ceiling. Nothing lossless left in the attention kernel.
- **Their lossless fusions are mostly already in vLLM-Omni** (`fused_qk_norm_rope`, `_indexed_gate_rms_norm_scale_shift`,
  `silu_and_mul`, fused QKV): in our profile those sites are ~7 % of a step, where in eager Diffusers they were ~28 %.
  Their bit-exactness lessons (correctly rounded division, round-to-nearest-even into FP8, explicit BF16 rounding of
  every intermediate) are the standard any fusion in this plugin must meet.
- **VAE tile batching** is the one exact-class speedup to port: our decode is 25 s (5 s 960x576), 43 s (5 s 1344x768)
  and 257-297 s (30 s 1344x768 continuation), 3-5 % of a render.
- **Not for this recipe:** Sol-Attn, FirstBlockCache, HyperFlow, the 2K refine LoRA (`models/Open-H3-Regenerate-2K`,
  lossy by their own description) and the fast VAE decoder (30 dB against the released decoder).
- **Where we stand** (different checkpoints and steps, so a rough comparison only): their lossless single GB10 at
  1344x768 would be about 29 s/step (40.1 / 1.387); our pair's stock 5 s 1344x768 render is 645 s / 20 steps = 32 s per
  step with a 2048-px Ref2VA reference in the sequence, and the two-pass GEMM takes about a quarter off. A same-cell
  run of their `gb10_kernel` config on one Spark would settle it; queued after the current tests.

## The other Spark entries in the same branch (read later the same day)

- **`models/minimax_h3/Sol-H3-Spark/`**: a two-stage pipeline on one Spark: a 672x384, 124-frame H3 draft in 4 updates
  (FastH3 VSA LoRA, 90 % sparse attention, W8A8 FP8 DiT, NVFP4 Qwen), a learned x2 latent upscaler, an H3-to-LTX
  latent adapter, then 3 updates of the LTX-2.5 refiner (distilled LoRA, Sol-Attn) and LTX's decoder, with the H3
  audio kept. Output 1344x768, 121 frames. Their ComfyUI table: **64.3 s per clip on a DGX Spark** (Qwen encoding and
  MP4 included, model loading excluded) against 989 s for the original 50-step H3 on an RTX 5090. The pixels come from
  LTX-2.5, conditioned on the H3 draft: "not a guarantee of unchanged identity", first/last frames are "conditioning
  tokens, not hard pixel replacement". Three separate runtimes (FastVideo + FA4, LTX source, ComfyUI in a container),
  LTX-2.5 weights gated under the LTX Community License, "a clean-environment installation has not yet been validated".
- **`Sol-H3/`**: Ulysses sequence parallelism up to 8 GPUs, 5/10/15 s at 1344x768, SOL/BSA sparse attention (B300
  numbers). **`HyperFlow/`**: the 8-step distilled adapter. **`super_acceleration/`**: H3 stage 1 + LTX-2.5 stage 2 on two
  GB200. All lossy by construction relative to the released H3.

For this recipe (lossless H3 on the pair) none of it applies directly; for a speed-first product on one Spark the
two-stage entry is the reference point: ~1 min per 5 s clip at 1344x768 with a different final model.
