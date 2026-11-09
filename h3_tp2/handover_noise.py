"""Show the model the pinned hand-over frames of a continuation window slightly noised (``H3_TP2_CONT_EDIT_NOISE``).

Later windows of a long clip pin the previous window's last frames as the clean rows of a latent edit
(``MiniMaxH3LatentEdit``, ``H3_TP2_CONT_EDIT=2``): at every step the model sees them as upstream's condition anchor,
``0.999 * clean + 0.001 * noise`` at timestep 0.999, i.e. as trustworthy as a keyframe.  They are not: they are the
model's own output, and what is slightly wrong in them (softer faces, a face that is no longer quite the reference's)
is copied into the new window and handed on.  Over eleven windows the picture loses detail and the faces drift
(2026-11-09: face sharpness -37 %, similarity to the reference 0.71 -> 0.60 over 118 s) although the latent
statistics hold.  The audio chain had the same disease and lost it when its pin stopped being absolute
(``audio_pin_steps``).

This module presents the fully pinned video rows at a chosen noise level instead: ``(1 - level) * clean + level *
noise`` at timestep ``1 - level``, the model's own condition-noise convention with more noise than 0.001.  The rows
are still restored clean in the output (the loop's ``x0`` is the edit's own), so the frames the clip keeps are
untouched; only what the free rows are conditioned on changes: the layout and motion of the tail stay, its fine
detail is no longer taken at face value and has to come from the reference image and the prompt.

The noise is the window's own initial noise at those rows, recovered from upstream's anchor (deterministic, the same
on every rank).  Rows of a ramped mask (0 < mask < 1) are left as they are.  Opt-in EXPERIMENT, NOT output-preserving.
"""
from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger("vllm_omni.h3_tp2.handover_noise")  # under vllm_omni so INFO reaches the worker log

MAX_LEVEL = 0.5


def parse_level(text: str | None) -> float | None:
    """``H3_TP2_CONT_EDIT_NOISE`` -> noise level in (0, 0.5], or None when the stock anchor stays."""
    text = (text or "").strip()
    if not text or text in ("0", "0.0", "off"):
        return None
    value = float(text)
    if not math.isfinite(value) or not 0.0 < value <= MAX_LEVEL:
        raise ValueError(f"H3_TP2_CONT_EDIT_NOISE={text!r}: use a noise level in (0, {MAX_LEVEL}]")
    return value


class NoisedPin:
    """A latent edit whose fully pinned rows the model sees at noise level ``level``; everything else is ``inner``."""

    def __init__(self, inner, anchor_rows: torch.Tensor, level: float) -> None:  # noqa: ANN001
        self.inner, self.anchor_rows, self.level = inner, anchor_rows, float(level)

    # the loop and other wrappers read these off the edit
    @property
    def clean_rows(self) -> torch.Tensor:
        return self.inner.clean_rows

    @property
    def mask_rows(self) -> torch.Tensor:
        return self.inner.mask_rows

    @property
    def restore_mask_rows(self) -> torch.Tensor:
        return self.inner.restore_mask_rows

    def to(self, *, device, dtype=torch.float32) -> NoisedPin:  # noqa: ANN001
        return NoisedPin(self.inner.to(device=device, dtype=dtype), self.anchor_rows.to(device=device, dtype=dtype),
                         self.level)

    def model_rows(self, state_rows: torch.Tensor) -> torch.Tensor:
        return torch.lerp(self.anchor_rows.to(state_rows.dtype), state_rows,
                          self.inner.mask_rows.to(state_rows.dtype).unsqueeze(-1))

    def target_timesteps(self, timestep: float, condition_timestep: float, *, sigma: float | None = None) -> torch.Tensor:
        stock = self.inner.target_timesteps(timestep, condition_timestep, sigma=sigma)
        pinned = self.inner.mask_rows == 0
        return torch.where(pinned, torch.clamp(stock, max=1.0 - self.level), stock)

    def prepare(self, state_rows: torch.Tensor, timestep: float, condition_timestep: float, *, sigma: float):
        return self.model_rows(state_rows), self.target_timesteps(timestep, condition_timestep, sigma=sigma)

    def x0(self, model_rows: torch.Tensor, velocity: torch.Tensor, timestep: float) -> torch.Tensor:
        return self.inner.x0(model_rows, velocity, timestep)


def noised(video_edit, level: float, stock_timestep: float):  # noqa: ANN001
    """The edit the loop should run with, or ``video_edit`` itself when there is nothing to noise.

    ``stock_timestep`` is upstream's condition timestep (0.999): the stock anchor is ``t * clean + (1 - t) * noise``,
    which gives the noise back.  Dividing by ``1 - t`` amplifies the anchor's float32 rounding by 1000 (about 1e-4 of
    a unit-variance noise), the same on every rank.
    """
    if video_edit is None:
        return video_edit
    t = float(stock_timestep)
    if not 0.0 < t < 1.0:
        logger.warning("h3_tp2 hand-over noise skipped: upstream's condition timestep is %s, the noise cannot be recovered", t)
        return video_edit
    mask = video_edit.mask_rows
    pinned = mask == 0
    count = int(pinned.sum())
    if count == 0:
        return video_edit
    clean = video_edit.clean_rows.float()
    stock = video_edit.anchor_rows.float()
    noise = (stock - t * clean) / (1.0 - t)
    anchor = torch.where(pinned.unsqueeze(-1), (1.0 - level) * clean + level * noise, stock)
    logger.info("h3_tp2 hand-over noise: %d of %d pinned video rows shown at noise level %.3f (timestep %.3f, stock %.3f)",
                count, int(mask.numel()), level, 1.0 - level, t)
    return NoisedPin(video_edit, anchor.to(video_edit.anchor_rows.dtype), level)


def install(dl, pp, level: float) -> None:  # noqa: ANN001
    """Wrap ``minimax_h3_denoise_loop`` (module ``dl``; the pipeline module ``pp`` imports it by name)."""
    orig = pp.minimax_h3_denoise_loop
    stock_timestep = float(dl.MINIMAX_H3_IMGVID_COND_TIMESTEP)

    def minimax_h3_denoise_loop(*args, video_edit=None, **kwargs):  # noqa: ANN001
        return orig(*args, video_edit=noised(video_edit, level, stock_timestep), **kwargs)

    minimax_h3_denoise_loop.__wrapped__ = orig  # type: ignore[attr-defined]
    pp.minimax_h3_denoise_loop = minimax_h3_denoise_loop
    if getattr(dl, "minimax_h3_denoise_loop", None) is orig:
        dl.minimax_h3_denoise_loop = minimax_h3_denoise_loop
