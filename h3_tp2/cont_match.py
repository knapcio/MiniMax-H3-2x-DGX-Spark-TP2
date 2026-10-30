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


def seam_ratio(video: torch.Tensor, overlap_v: int) -> tuple[float, float, float]:
    """Latent-space seam check of one sampled window: mean |v[t+1]-v[t]| at the boundary between the last overlap
    frame and the first free frame, against the median of the same quantity over the window's free part.
    Returns (boundary, median, ratio).  A cut right after the prefix shows as a ratio far above 1."""
    v = video.float()
    d = (v[:, :, 1:] - v[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))   # per transition
    boundary = float(d[overlap_v - 1].item())
    free = d[overlap_v:]
    median = float(free.median().item()) if free.numel() else boundary
    return boundary, median, boundary / max(median, 1e-6)


def wrap_diffuse(diffuse, *, apply: bool, overlap_v_hint: int | None = None, retries: int = 0, max_ratio: float = 2.0,
                 lock: str = "", lock_strength: float = 1.0, lock_ramp: float = 1.0, lock_audio: bool = False,
                 lock_std: float | None = None, lock_ref: str = "window"):  # noqa: ANN001
    """Return a ``diffuse`` replacement for one continuation request (state: the previous window's video latents).

    With ``retries`` > 0 a window whose latent seam ratio exceeds ``max_ratio`` is sampled again with another seed
    (up to ``retries`` times); the attempt with the lowest ratio is kept.
    """
    state = {"prev": None, "index": 0, "ref": None, "ref_audio": None}

    def _overlap_audio(args):  # noqa: ANN001
        lengths = args.get("audio_condition_lengths") or []
        if lengths:
            return int(lengths[-1])
        mask = args.get("audio_edit_mask_rows")
        if mask is not None:
            return int((mask == 0).sum().item() // 2)
        return None

    def _overlap(args):  # noqa: ANN001
        blocks = args.get("ref_blocks") or []
        guide = next((b for b in reversed(blocks) if isinstance(b, dict) and b.get("kind") == "latent_guide"), None)
        if guide is not None:
            return int(guide["latent_t"])
        mask = args.get("video_edit_mask_rows")
        if mask is not None and overlap_v_hint:
            return overlap_v_hint
        return None

    def _diffuse(**args):  # noqa: ANN003
        state["index"] += 1
        ov = _overlap(args) if state["prev"] is not None else None
        best = None
        for attempt in range(1 + (retries if ov else 0)):
            a = dict(args)
            if attempt and "seed" in a and a["seed"] is not None:
                a["seed"] = int(a["seed"]) + 7919 * attempt
            video, audio = diffuse(**a)
            if not ov or not (isinstance(video, torch.Tensor) and video.dim() == 5):
                best = (video, audio, 0.0)
                break
            try:
                b, m, r = seam_ratio(video, ov)
                logger.info("h3_tp2 seam check window %d attempt %d (seed %s): boundary %.4f median %.4f ratio %.2f",
                            state["index"], attempt + 1, a.get("seed"), b, m, r)
            except Exception as e:  # noqa: BLE001
                logger.warning("h3_tp2 seam check skipped (%s)", e)
                best = (video, audio, 0.0)
                break
            if best is None or r < best[2]:
                best = (video, audio, r)
            if r <= max_ratio:
                break
            logger.warning("h3_tp2 seam check window %d: ratio %.2f > %.2f, re-sampling with another seed", state["index"], r, max_ratio)
        video, audio, _ = best
        if state["prev"] is not None and ov:
            try:
                video = match_window(state["prev"][:, :, -ov:], video, ov, apply=apply, index=state["index"])
                if lock == "first" and state["ref"] is not None:
                    video = lock_to_reference(state["ref"], video, ov, lock_strength, index=state["index"], ramp=lock_ramp, std_strength=lock_std)
                if lock == "first" and lock_audio and state["ref_audio"] is not None and isinstance(audio, torch.Tensor) and audio.dim() == 3:
                    ov_a = _overlap_audio(args)
                    if ov_a:
                        audio = lock_audio_to_reference(state["ref_audio"], audio, ov_a, lock_strength, index=state["index"], ramp=lock_ramp)
            except Exception as e:  # noqa: BLE001  (diagnostics never break a render)
                logger.warning("h3_tp2 hand-over diagnostics skipped (%s)", e)
        elif state["ref"] is None and isinstance(video, torch.Tensor) and video.dim() == 5:
            state["ref"] = _stats(ref_slice(video, lock_ref, dim=2))
            if isinstance(audio, torch.Tensor) and audio.dim() == 3:
                state["ref_audio"] = audio_stats(ref_slice(audio, lock_ref, dim=2))
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


def ref_slice(x: torch.Tensor, mode: str, *, dim: int) -> torch.Tensor:
    """The frames of the first window that define the look-lock reference.  ``window``: all of them (the first
    behaviour).  ``tail``: the last quarter (at least one frame): a first window that drifts (the painting brightened
    115 -> 135 luma over 345 frames on 2026-10-30) may be pulled toward an earlier look by a whole-window reference.
    The owner reported flicker at 15-22 s; the cause and this alternative remain to be tested."""
    if mode != "tail":
        return x
    n = x.shape[dim]
    return x.narrow(dim, n - max(1, n // 4), max(1, n // 4))


def lock_to_reference(ref_stats: tuple[torch.Tensor, torch.Tensor], new_window: torch.Tensor, overlap_v: int,
                      strength: float, *, index: int, ramp: float = 1.0, std_strength: float | None = None) -> torch.Tensor:
    """Per-channel affine of the new window's FREE frames (after the pinned overlap) so that their mean/std move
    toward the reference window's statistics by ``strength`` (1.0 = equal).  Against the drift that compounds window
    after window (luma +41 and a 10x detail metric over 60 s on 2026-10-29): every window restarts exactly from the
    previous tail and then wanders toward the model's brighter, sharper prior; the lock pulls it back to the look of
    the first window.  Latents only; the pinned overlap frames are left untouched."""
    m_ref, s_ref = ref_stats
    free = new_window[:, :, overlap_v:].float()
    m, sd = _stats(free)
    gain = (s_ref / sd.clamp_min(1e-6)).clamp(0.5, 2.0)
    gain = 1.0 + (strength if std_strength is None else std_strength) * (gain - 1.0)   # std_strength 0 = mean only
    m_target = m + strength * (m_ref - m)
    # ramped: no correction at the first free frame (it continues the pinned tail exactly), the full correction at the
    # window's end; a whole-window correction stepped the brightness at the hand-over (-8.5 luma at seam 2, 03:27)
    n_free = free.shape[2]
    # ``ramp``: the fraction of the free frames over which the correction reaches full strength (1.0 = the whole
    # window; 0.25 = full after a quarter, then held: holds the look for most of the window, still continuous)
    n_ramp = max(1, int(round(n_free * max(0.05, min(1.0, ramp)))))
    w = torch.clamp(torch.arange(1, n_free + 1, device=free.device, dtype=torch.float32) / n_ramp, max=1.0).view(1, 1, -1, 1, 1)
    gain_t = 1.0 + w * (gain.view(1, -1, 1, 1, 1) - 1.0)
    shift_t = w * (m_target - m).view(1, -1, 1, 1, 1)
    out = (free - m.view(1, -1, 1, 1, 1)) * gain_t + m.view(1, -1, 1, 1, 1) + shift_t
    m2, s2 = _stats(out)
    logger.info("h3_tp2 look-lock window %d: free frames mean %.4f std %.4f -> %.4f / %.4f (reference %.4f / %.4f, strength %.2f, ramp %.2f)",
                index, m.mean().item(), sd.mean().item(), m2.mean().item(), s2.mean().item(), m_ref.mean().item(), s_ref.mean().item(), strength, ramp)
    locked = new_window.clone()
    locked[:, :, overlap_v:] = out.to(new_window.dtype)
    return locked


def lock_audio_to_reference(ref_stats: tuple[torch.Tensor, torch.Tensor], new_audio: torch.Tensor, overlap_t: int,
                            strength: float, *, index: int, ramp: float = 1.0) -> torch.Tensor:
    """The audio analogue of ``lock_to_reference``: audio latents [C, D, T]; per-(channel, dim) mean/std over time of
    the free steps (after the pinned overlap) pulled toward the first window's, ramped along time.  The generated
    ambience climbed 23 dB over the last two windows of a 60 s clip (2026-10-30 05:30) while the video lock held."""
    m_ref, s_ref = ref_stats
    free = new_audio[..., overlap_t:].float()
    m, sd = free.mean(dim=-1, keepdim=True), free.std(dim=-1, keepdim=True)
    gain = (s_ref / sd.clamp_min(1e-6)).clamp(0.5, 2.0)
    gain = 1.0 + strength * (gain - 1.0)
    shift = strength * (m_ref - m)
    n_free = free.shape[-1]
    n_ramp = max(1, int(round(n_free * max(0.05, min(1.0, ramp)))))
    w = torch.clamp(torch.arange(1, n_free + 1, device=free.device, dtype=torch.float32) / n_ramp, max=1.0)
    gain_t = 1.0 + w * (gain - 1.0)
    out = (free - m) * gain_t + m + w * shift
    logger.info("h3_tp2 audio look-lock window %d: free steps std %.4f -> %.4f (reference %.4f), mean %.4f -> %.4f", index,
                sd.mean().item(), out.std(dim=-1).mean().item(), s_ref.mean().item(), m.mean().item(), out.mean().item())
    locked = new_audio.clone()
    locked[..., overlap_t:] = out.to(new_audio.dtype)
    return locked


def audio_stats(audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = audio.float()
    return a.mean(dim=-1, keepdim=True), a.std(dim=-1, keepdim=True)
