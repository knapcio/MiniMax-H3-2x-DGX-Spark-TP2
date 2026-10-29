"""Continuation hand-over diagnostics and an optional per-channel latent match (H3_TP2_CONT_MATCH=1).

Upstream renders every continuation window fresh under a latent-tail guide and appends only the frames after the
hidden overlap.  Measured 2026-10-29 (owner's eye + per-frame luma): the first frames of window 2 are ~19 % brighter
than the last frames of window 1 and the texture changes, so the hand-over is visible.  This wraps the per-window
``diffuse`` call to log, for every window after the first, how the regenerated overlap compares with the previous
window's tail (per-channel mean/std of the video latents) and, when enabled, to apply the per-channel affine map that
makes the regenerated overlap's statistics equal to the tail's, to the whole new window (video latents only; audio
untouched).  The matched window then continues from the exposure and contrast it was supposed to continue from.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger("vllm_omni.h3_tp2.cont_match")


def _stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [B, C, T, H, W] -> per-channel mean/std over B, T, H, W
    xf = x.float()
    dims = (0, 2, 3, 4)
    return xf.mean(dim=dims), xf.std(dim=dims)


def match_window(prev_tail: torch.Tensor, new_window: torch.Tensor, overlap_v: int, *, apply: bool, index: int) -> torch.Tensor:
    """Log the hand-over mismatch and, if ``apply``, return the affine-matched new window (same dtype/shape)."""
    regen = new_window[:, :, :overlap_v]
    m0, s0 = _stats(prev_tail)
    m1, s1 = _stats(regen)
    mfree, _ = _stats(new_window[:, :, overlap_v : overlap_v + max(1, overlap_v)])
    logger.info(
        "h3_tp2 hand-over window %d: guide tail mean %.4f std %.4f | regenerated overlap mean %.4f std %.4f | next frames mean %.4f | "
        "per-channel |dmean| max %.4f, std ratio %.3f..%.3f",
        index, m0.mean().item(), s0.mean().item(), m1.mean().item(), s1.mean().item(), mfree.mean().item(),
        (m1 - m0).abs().max().item(), (s0 / s1.clamp_min(1e-6)).min().item(), (s0 / s1.clamp_min(1e-6)).max().item(),
    )
    if not apply:
        return new_window
    gain = (s0 / s1.clamp_min(1e-6)).clamp(0.5, 2.0).view(1, -1, 1, 1, 1)
    offset = (m0 - m1 * gain.view(-1)).view(1, -1, 1, 1, 1)
    out = (new_window.float() * gain + offset).to(new_window.dtype)
    m2, s2 = _stats(out[:, :, :overlap_v])
    logger.info("h3_tp2 hand-over window %d: matched overlap mean %.4f std %.4f (gain %.3f..%.3f)", index,
                m2.mean().item(), s2.mean().item(), gain.min().item(), gain.max().item())
    return out


def wrap_diffuse(diffuse, *, apply: bool):  # noqa: ANN001
    """Return a ``diffuse`` replacement for one continuation request (state: the previous window's video latents)."""
    state = {"prev": None, "index": 0}

    def _diffuse(**args):  # noqa: ANN003
        state["index"] += 1
        video, audio = diffuse(**args)
        blocks = args.get("ref_blocks") or []
        guide = next((b for b in reversed(blocks) if isinstance(b, dict) and b.get("kind") == "latent_guide"), None)
        if state["prev"] is not None and guide is not None and isinstance(video, torch.Tensor) and video.dim() == 5:
            ov = int(guide["latent_t"])
            try:
                video = match_window(state["prev"][:, :, -ov:], video, ov, apply=apply, index=state["index"])
            except Exception as e:  # noqa: BLE001  (diagnostics never break a render)
                logger.warning("h3_tp2 hand-over diagnostics skipped (%s)", e)
        state["prev"] = video
        return video, audio

    return _diffuse


def crossfade_tail(video: torch.Tensor, new_window: torch.Tensor, overlap_v: int) -> torch.Tensor:
    """Replace the cumulative video's last ``overlap_v`` latent frames by a linear crossfade from them to the new
    window's regenerated overlap, then append the new window's free frames.  Upstream keeps the previous window's
    frames up to the hand-over and switches to the new window there, so any difference in fine texture between the
    two renderings of the same frames shows as a jump; the ramp spreads it over the overlap (0.9 s at 22 frames)."""
    prev_tail = video[:, :, -overlap_v:].float()
    regen = new_window[:, :, :overlap_v].float()
    w = torch.linspace(0.0, 1.0, steps=overlap_v + 2, device=video.device, dtype=torch.float32)[1:-1].view(1, 1, -1, 1, 1)
    blended = prev_tail * (1 - w) + regen * w
    # variance-preserving: a mix of two renderings with different fine texture has a lower std mid-ramp (the decoded
    # frame dips darker/flatter for a few frames: the owner's "blink" at the seam, 2026-10-29 20:10). Re-match every
    # blended frame's per-channel mean/std to the interpolation of the two sources' statistics.
    dims = (0, 3, 4)   # per channel and per latent frame
    m_prev, s_prev = prev_tail.mean(dim=dims, keepdim=True), prev_tail.std(dim=dims, keepdim=True)
    m_new, s_new = regen.mean(dim=dims, keepdim=True), regen.std(dim=dims, keepdim=True)
    m_target = m_prev * (1 - w) + m_new * w
    s_target = s_prev * (1 - w) + s_new * w
    m_b, s_b = blended.mean(dim=dims, keepdim=True), blended.std(dim=dims, keepdim=True)
    blended = (blended - m_b) * (s_target / s_b.clamp_min(1e-6)) + m_target
    return torch.cat((video[:, :, :-overlap_v], blended.to(video.dtype), new_window[:, :, overlap_v:]), dim=2)
