"""Release the continuation audio pin after a fraction of the denoise steps (``H3_TP2_CONT_EDIT_AUDIO_STEPS``).

Later windows of a long clip pin the previous window's audio tail as clean rows of the latent edit
(``minimax_h3_audio_edit`` in the request state); the pinned rows are re-noised to the current sigma at every step and
the window matches them.  queue111/113 (results/20261106-voice, sections 9-10) measured that the whole-run pin hands the
tail's flaws down the chain (3.87 -> 2.93 UTMOS over six windows) and that a spatial ramp of the mask recovers only half.

This module keeps the pin for the first ``fraction`` of the steps and drops the edit from the request state after
that: the structure of the hand-over (what sounds, how loud, which speaker) is set in the early, high-sigma steps and
stays continuous, while the late steps that render the detail of the voice run free.  At the release step the pinned
rows hold the clean tail plus noise at the current sigma, so from there they are denoised like an SDEdit of the tail.

Opt-in, NOT output-preserving.  ``fraction`` 1.0 (or unset) leaves the stock behaviour (pin for the whole run).
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

STATE_KEY = "minimax_h3_audio_edit"
SIGMAS_KEY = "minimax_h3_sigmas_video"


def parse_fraction(text: str | None) -> float | None:
    """``H3_TP2_CONT_EDIT_AUDIO_STEPS`` -> fraction in (0, 1), or None when the whole-run pin stays."""
    text = (text or "").strip()
    if not text:
        return None
    value = float(text)
    if not 0.0 < value < 1.0:
        if value == 1.0:
            return None
        raise ValueError(f"H3_TP2_CONT_EDIT_AUDIO_STEPS={text!r}: use a fraction in (0, 1)")
    return value


def release_step(fraction: float, n_steps: int) -> int:
    """First step index that runs without the audio pin: at least 1 step pinned, at least 1 step free."""
    return max(1, min(n_steps - 1, math.ceil(fraction * n_steps)))


def release_due(state, fraction: float) -> bool:  # noqa: ANN001
    """True when this request's audio edit should be dropped before the coming step."""
    if state.extra.get(STATE_KEY) is None:
        return False
    sigmas = state.extra.get(SIGMAS_KEY)
    n_steps = (len(sigmas) - 1) if sigmas is not None else 0
    return n_steps > 1 and int(state.step_index) >= release_step(fraction, n_steps)


def install(pp, fraction: float) -> None:  # noqa: ANN001
    """Wrap ``MiniMaxH3Pipeline.denoise_step`` so every state past the release step runs without its audio edit."""
    cls = pp.MiniMaxH3Pipeline
    orig = cls.denoise_step

    def denoise_step(self, input_batch, *, states=None, **kwargs):  # noqa: ANN001
        for state in list(states if states is not None else input_batch.states):
            if release_due(state, fraction):
                n_steps = len(state.extra[SIGMAS_KEY]) - 1
                state.extra[STATE_KEY] = None
                logger.info("h3_tp2 audio pin released at step %d of %d (H3_TP2_CONT_EDIT_AUDIO_STEPS=%.2f)",
                            int(state.step_index), n_steps, fraction)
        return orig(self, input_batch, states=states, **kwargs)

    denoise_step.__wrapped__ = orig  # type: ignore[attr-defined]
    cls.denoise_step = denoise_step
