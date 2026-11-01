# Experimental scoped VAE tile batching

`H3_TP2_VAE_STACK_TILES=2` selects a separate, opt-in experiment. In the
inspected checkpoint, tiled video decode calls `_run_tile_tasks` with the
model's bound `decode` method. The experiment batches up to
`H3_TP2_VAE_TILE_BATCH` tiles per call (1–4, default 4), only inside the
synchronous decode entry points and without changing `model.stack_tiling`.
Input samples per tile multiply the actual batch. Encode and model calls
outside that decode scope retain their existing flags and behavior.

The default is still `0` (off). The legacy `1` remains available for existing
experiments; it changes a shared flag and is not safe to assume equivalent
under overlapping encode/decode or overlapping decodes. The new path has CPU
tests against a fake model and separate CPU tests using a copied checkpoint
method; repeated installation, copied execution contexts, overlapping threads
and training with a decoder mask have been exercised. These checks are not a
CUDA test, and multiple imports/hot reload or asynchronous decode entry points
are not qualified.

Before using this for speed claims or normal serving, test both ranks on an
idle, isolated pair with the same pinned model, latent inputs, dimensions and
decode settings. Record actual loaded code, wall time, peak memory and image
differences against the off baseline. A different batching strategy may change
cuDNN/cuBLAS algorithms and decoded pixels; check seams and full playback as
well. The historical 5 s batching measurement in
`results/20261029-vae-stack-tiles/` is not a measurement of this new path or a
prediction for longer clips.

Explicitly export both `H3_TP2_VAE_STACK_TILES` and `H3_TP2_VAE_TILE_BATCH`
from the head for each test, including mode `0` for the baseline, and verify
the effective values on both ranks. A variable absent on the head does not
erase a value inherited from the remote agent's environment; absence alone is
not proof that both ranks use the same default.
