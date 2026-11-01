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
picture, motion and audio: fewer distilled steps change the generated result. The Spark's checked-out recipe still
lists **5/9** for four/eight forwards, but its actual `pipeline_minimax_h3.py` validates
`num_inference_steps == spec.denoise_steps`, i.e. **4/8**. The checked code and checked-in recipe disagree; verify
the request contract on the actual running version rather than trusting either document alone.

For a future comparison, the different LightX2V `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors`
artifact was downloaded to the Mac from the [publisher's repository](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
at revision `30f485ed874aa44c8183348ad361c5eabad318da`: 1,383,677,808 bytes, SHA-256
`c3d4a2cf618efea71b9e21a4baaa12d412f1eb6c2b6f86efacaf0ebb6814b689`. Its safetensors header declares rank-128
BF16 and alpha 8; 104 tensors target `fc1` and 104 target `fc2`, making the fusion guard relevant. No weights were
loaded into either Spark and no Turbo test has run. The user rejected an earlier 8-step v1.0 forest render's
lighting and blur; a different 4-step artifact is an experiment, not evidence that this concern is resolved.
