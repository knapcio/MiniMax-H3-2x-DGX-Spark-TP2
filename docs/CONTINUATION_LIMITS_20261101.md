# Continuation correctness notes (2026-11-01)

The fixes below have CPU regressions and have not been deployed or validated
with a real GPU render. The already running film uses its original pinned code.

- The optional legacy seed/temporal experiment and the diagnostic continuation
  wrapper used to share a late-bound original-function variable. With the
  experiment enabled and edited continuation/crossfade disabled, that could
  recurse before diffusion. Each hook now captures its own predecessor.
- Multi-window FL2VA with a final keyframe (`frame_indices` containing `-1`)
  previously applied it to the end of the first window, not the entire output.
  The edited/crossfade extension now rejects that combination before its first
  diffusion call. Encoder/conditioning work may already have run. Start-only
  `[0]`, single-window final-frame requests and Ref2VA are unaffected by this
  guard. Supporting a global final keyframe requires a separate implementation.
- The look-lock formula is unchanged: its hidden overlap stays untouched, but
  the first *free* latent frame receives `1/n_ramp` of the correction. A former
  comment claiming zero correction on that frame was incorrect.

A separate source-level observation remains open. In the inspected upstream
snapshot `pipeline_minimax_h3.py` SHA-256
`fda01a7ced807952bf80c3f9370d9dc0b4d77f960485b00de883abd974c952f0`, the actual
`_build_denoise_inputs` forwards `temporal_offset` and `media_time_origin` to
the Ref2VA packed-layout call, but not to its T2VA/FL2VA packed-layout call.
An isolated CPU probe confirmed the argument routing; it did not execute the
real packed-layout helper or determine visual consequences. Separately, the
default edited loop replaces the older temporal-offset/anchor experiment
wrapper. A configuration toggle alone therefore does not prove an effective
clock or anchor A/B in the default path.

The default later-window text says the camera keeps moving even when a user's
base prompt asks for a locked camera. That wording is verified; whether it
causes drift is not. A neutral template, and any change to the look-lock ramp,
need matched picture and sound A/B before changing defaults.
