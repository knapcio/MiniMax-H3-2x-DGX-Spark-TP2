# Optional returned-window seam diagnostic

`H3_TP2_SEAM_POSTCHECK=1` adds a diagnostic after the continuation wrapper's
normal video match/look-lock and audio correction stage. It measures the video
window actually returned by that wrapper, **before any downstream crossfade or
cumulative assembly**. Both plugin continuation routes use this wrapper. The
environment switch is read once when a request's wrapper is created; every
value except stripped `1`, including an absent variable, leaves it disabled.
Disabled means no extra tensor operations, measurements, synchronizations or
log records; only the configuration read and boolean branch remain.

The existing `seam check ... attempt` log and retry score still describe the raw
sample before correction. The new `seam postcheck ... returned window after
match/look-lock stage, before downstream crossfade` log reports boundary mean
absolute latent difference, the median difference over free frames and their
ratio. It is observational: no threshold, retry, seed, tensor values, audio or
checkpoint state changes. Measurement and diagnostic logging exceptions are
contained separately, after audio correction, and cannot skip that correction.
The diagnostic does not claim that preceding corrections succeeded if their
existing error handler reported a failure.

CPU reproduction with smooth synthetic latents and a pinned overlap gives a
raw ratio of 1.0 but a returned ratio of about 3.04654 after look-lock. The new
log exposes that distinction without fixing it or changing candidate selection.
A low ratio does not prove visual quality, identity continuity, natural motion
or lip sync. The metric does not measure the final decoded or crossfaded seam.
GPU reductions and scalar extraction add work/synchronization when enabled;
their runtime cost has not been measured. This change was tested only on CPU.

Run `python -m unittest tests.test_cont_seam_postcheck` in an environment with
CPU PyTorch. The suite checks the reproducer, bitwise AV/state equality, retry
seeds and RNG, failure isolation, and actual plugin loop bodies including
crossfade. No model inference, deployment or service change is part of the test.
