# Guard fused SwiGLU against wrapped LoRA layers

The optional FP8 epilogue/SwiGLU fusion stashes the FP32 result of `fc1`, returns an uninitialized placeholder,
then discards that placeholder and applies SwiGLU to the stashed base result. In the vLLM-Omni source on the
Sparks (checkout `a8576ccb725c4e21cd13c3eb5f9a546b21149d2b`), a diffusion LoRA wrapper adds its adapter delta
to the base linear result. If it wraps `fc1`, the fusion can therefore discard that delta; the experimental level-2
direct `fc2` path can also bypass a wrapped `fc2`. This is a code-path finding and a CPU reproduction, **not** a
measured real-model LoRA output. The current 300 s request does not request a LoRA.

`h3_tp2/fp8_swiglu.py` now detects vLLM `BaseLayerWithLoRA` on either MLP projection and uses the original MLP
forward for that block. Non-wrapped blocks keep the existing fusion. This is deliberately conservative even if
an adapter has been deactivated but its wrapper remains: that block may be slower until it is unwrapped or the
service restarts. The underlying two-pass FP8 GEMM remains available on the ordinary path.

Three CPU stand-ins cover an active `fc1` delta, a wrapped `fc2` under level 2 and the unwrapped fused path. A
separate reproduction using the old and new `install()` code with synthetic linear/LoRA stand-ins found maximum
absolute error 0.2736 with the old fusion and 0 with the guard. The deployed files on both Sparks were only read:
`fp8_swiglu.py` SHA-256 `0331aa2961ee849b7a7010c5112a80e49b744fb00ae4290ca748c8290bb27ecf`, upstream
`diffusion/lora/layers/base_linear.py` SHA-256 `9baa38fca241a0b250a9b6f79ab3467da7385ec941b020f34680c451430d64eb`.
The fix has not been installed or restarted into the live service.

Before deploying or benchmarking Turbo, run a controlled GPU comparison with the actual wrapped vLLM layers and
both ranks, check nonzero adapter influence and unwrapped baseline, and measure the fallback's cost. Also compare
picture, motion and audio: fewer distilled steps change the generated result. On the Spark's pinned vLLM-Omni
checkout, the documented Turbo request values are **5/9** for four/eight forwards; newer upstream documentation
uses 4/8, so use the contract of the version actually installed, not the newer numbers by assumption.
