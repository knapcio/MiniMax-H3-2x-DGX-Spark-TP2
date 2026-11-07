"""Release the continuation audio pin after a fraction of the denoise steps (``H3_TP2_CONT_EDIT_AUDIO_STEPS``).

Later windows of a long clip pin the previous window's audio tail as the clean rows of a latent edit
(``MiniMaxH3LatentEdit``) that ``minimax_h3_denoise_loop`` applies at every step: the pinned rows are presented clean
at the condition timestep and written back clean, and the window matches them.  queue111/113 (results/20261106-voice,
sections 9-10) measured that the whole-run pin hands the tail's flaws down the chain (3.87 -> 2.93 UTMOS over six
windows), that a spatial ramp of the mask recovers only half and that a shorter hard pin recovers nothing.

This module keeps the pin for the first ``fraction`` of the steps and lets the pinned rows go free after that.  The
loop holds the edit as a local, so the plugin wraps the loop and hands it a step-gated edit: the real edit's
``prepare`` / ``x0`` while the step is below the release step, and the free sampler's arithmetic (rows unchanged, the
global timestep per row, plain rectified-flow x0) from the release step on.  At the release step the pinned rows hold
the clean tail re-noised to that sigma, so from there they are denoised like an SDEdit of the tail: the structure of
the hand-over (what sounds, how loud, which speaker) is set in the early, high-sigma steps and stays continuous, the
late steps that render the detail of the voice run free.

Opt-in, NOT output-preserving.  ``fraction`` 1.0 (or unset) leaves the stock behaviour (pin for the whole run).
"""
from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


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


class StepGatedEdit:
    """A latent edit that is the real edit before ``release`` and the free sampler from ``release`` on.

    The loop calls ``prepare`` once per step (through ``minimax_h3_prepare_edit_rows``) and ``x0`` once per step after
    it; the step is counted on ``prepare``.  ``v_to_x0`` is upstream's ``minimax_h3_rf_v_to_x0``.
    """

    def __init__(self, inner, release: int, n_steps: int, v_to_x0, *, calls: int = 0) -> None:  # noqa: ANN001
        self.inner, self.release, self.n_steps, self.v_to_x0, self.calls = inner, release, n_steps, v_to_x0, calls

    def to(self, *, device, dtype=torch.float32) -> StepGatedEdit:  # noqa: ANN001
        return StepGatedEdit(self.inner.to(device=device, dtype=dtype), self.release, self.n_steps, self.v_to_x0,
                             calls=self.calls)

    @property
    def pinned(self) -> bool:
        """Whether the step whose ``prepare`` ran last is still pinned."""
        return self.calls - 1 < self.release

    def prepare(self, state_rows: torch.Tensor, timestep: float, condition_timestep: float, *, sigma: float):
        step = self.calls
        self.calls += 1
        if step < self.release:
            return self.inner.prepare(state_rows, timestep, condition_timestep, sigma=sigma)
        if step == self.release:
            logger.info("h3_tp2 audio pin released at step %d of %d", step, self.n_steps)
        # the free sampler: rows unchanged, every row at the global timestep (what a mask of ones would give)
        target = torch.clamp(torch.full((state_rows.shape[0],), 1.0 - float(sigma), dtype=torch.float32,
                                        device=state_rows.device), max=float(condition_timestep))
        return state_rows, target

    def x0(self, model_rows: torch.Tensor, velocity: torch.Tensor, timestep: float) -> torch.Tensor:
        if self.pinned:
            return self.inner.x0(model_rows, velocity, timestep)
        return self.v_to_x0(model_rows, velocity.to(model_rows.dtype),
                            torch.tensor(float(timestep), dtype=model_rows.dtype, device=model_rows.device))


def gate(audio_edit, sigmas_audio, fraction: float, v_to_x0):  # noqa: ANN001
    """The edit the loop should run with: step-gated when there is one and more than one step, else unchanged."""
    if audio_edit is None or sigmas_audio is None:
        return audio_edit
    n_steps = len(sigmas_audio) - 1
    if n_steps < 2:
        return audio_edit
    return StepGatedEdit(audio_edit, release_step(fraction, n_steps), n_steps, v_to_x0)


def install(dl, pp, fraction: float) -> None:  # noqa: ANN001
    """Wrap ``minimax_h3_denoise_loop`` (module ``dl``; the pipeline module ``pp`` imports it by name)."""
    orig = pp.minimax_h3_denoise_loop
    v_to_x0 = dl.minimax_h3_rf_v_to_x0

    def minimax_h3_denoise_loop(*args, audio_edit=None, sigmas_audio=None, **kwargs):  # noqa: ANN001
        return orig(*args, audio_edit=gate(audio_edit, sigmas_audio, fraction, v_to_x0), sigmas_audio=sigmas_audio,
                    **kwargs)

    minimax_h3_denoise_loop.__wrapped__ = orig  # type: ignore[attr-defined]
    pp.minimax_h3_denoise_loop = minimax_h3_denoise_loop
    if getattr(dl, "minimax_h3_denoise_loop", None) is orig:
        dl.minimax_h3_denoise_loop = minimax_h3_denoise_loop
