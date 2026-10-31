"""Validate experimental H3 continuation checkpoint shapes; not wired into serving."""
from __future__ import annotations

import torch


def video_t(frames: int) -> int:
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 5 or frames % 17 != 5:
        raise ValueError('video frames must be on the 17n+5 grid')
    return (frames - 5) // 17 * 5 + 2


def audio_t(frame_boundary: int) -> int:
    if not isinstance(frame_boundary, int) or isinstance(frame_boundary, bool) or frame_boundary < 0:
        raise ValueError('audio boundary must be a nonnegative integer')
    # Matches the deployed H3 continuation planner's global 40 Hz clock.
    return round(frame_boundary * 5 / 3)


def planned_window(total_frames: int, full_window: int, overlap: int, window: int) -> tuple[int, int, int]:
    """Return (start, end, overlap) for a one-based continuation window."""
    for value in (total_frames, full_window, overlap):
        video_t(value)
    if (total_frames > 1_000_000 or not 107 <= full_window <= 345 or overlap >= full_window
            or not isinstance(window, int) or isinstance(window, bool) or window < 1):
        raise ValueError('invalid continuation plan')
    end = min(total_frames, full_window)
    start = 0
    current = 1
    while current < window and end < total_frames:
        start = end - overlap
        end = min(total_frames, start + full_window)
        current += 1
    if current != window:
        raise ValueError('checkpoint window is outside the continuation plan')
    return start, end, 0 if current == 1 else overlap


def validate_for_plan(tensors: dict[str, torch.Tensor], *, window: int, total_frames: int, full_window: int,
                      overlap: int, latent_h: int, latent_w: int, channels: int = 24,
                      audio_channels: int = 2, audio_dims: int = 32, require_audio_ref: bool = True) -> None:
    """Validate state against a complete, caller-authenticated continuation plan."""
    start, end, actual_overlap = planned_window(total_frames, full_window, overlap, window)
    validate(tensors, window=window, end_frame=end, window_frames=end - start, overlap_frames=actual_overlap,
             latent_h=latent_h, latent_w=latent_w, channels=channels, audio_channels=audio_channels,
             audio_dims=audio_dims, require_audio_ref=require_audio_ref)


def validate(tensors: dict[str, torch.Tensor], *, window: int, end_frame: int, window_frames: int,
             overlap_frames: int, latent_h: int, latent_w: int, channels: int = 24,
             audio_channels: int = 2, audio_dims: int = 32, require_audio_ref: bool = True) -> None:
    """Fail closed if a rank's stored AV/match state cannot represent the declared window.

    The caller must obtain the window geometry from the request's verified plan,
    independently validate the request/runtime identity, and coordinate ranks.
    """
    ints = (window, end_frame, window_frames, overlap_frames, latent_h, latent_w, channels, audio_channels, audio_dims)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in ints) or window < 1 or not isinstance(require_audio_ref, bool):
        raise ValueError('invalid checkpoint window or tensor dimensions')
    if any(x < 1 for x in (latent_h, latent_w, channels, audio_channels, audio_dims)):
        raise ValueError('latent dimensions must be positive')
    total_t, previous_t = video_t(end_frame), video_t(window_frames)
    if window_frames > end_frame:
        raise ValueError('window extends before the beginning of the clip')
    if window == 1:
        if overlap_frames != 0 or window_frames != end_frame:
            raise ValueError('first window geometry differs')
        previous_overlap = 0
    else:
        previous_overlap = video_t(overlap_frames)
        if overlap_frames >= window_frames:
            raise ValueError('overlap must be smaller than the window')
    required = {'cumulative.video', 'cumulative.audio', 'match.index', 'match.prev', 'match.ref_mean', 'match.ref_std'}
    audio_keys = {'match.audio_mean', 'match.audio_std'}
    if not isinstance(tensors, dict) or set(tensors) not in (required, required | audio_keys):
        raise ValueError('checkpoint state tensors are incomplete or unexpected')
    if require_audio_ref and not audio_keys.issubset(tensors):
        raise ValueError('audio look-lock reference is missing')
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise ValueError('checkpoint state contains a non-tensor value')
    index, video, audio = tensors['match.index'], tensors['cumulative.video'], tensors['cumulative.audio']
    previous, mean, std = tensors['match.prev'], tensors['match.ref_mean'], tensors['match.ref_std']
    if index.ndim != 0 or index.dtype != torch.int64 or index.item() != window:
        raise ValueError('checkpoint match index differs from its completed window')
    if tuple(video.shape) != (1, channels, total_t, latent_h, latent_w):
        raise ValueError('cumulative video shape differs from the window plan')
    if tuple(previous.shape) != (1, channels, previous_t, latent_h, latent_w) or previous.dtype != video.dtype:
        raise ValueError('previous video shape or dtype differs')
    if tuple(audio.shape) != (audio_channels, audio_dims, audio_t(end_frame)):
        raise ValueError('cumulative audio shape differs from the global timeline')
    if (not video.is_floating_point() or not audio.is_floating_point() or mean.dtype != torch.float32
            or std.dtype != torch.float32 or tuple(mean.shape) != (channels,) or std.shape != mean.shape):
        raise ValueError('video/audio dtype or video reference shape differs')
    if any(not torch.isfinite(value).all().item() for name, value in tensors.items() if name != 'match.index'):
        raise ValueError('checkpoint contains non-finite values')
    if (std < 0).any().item():
        raise ValueError('video reference standard deviation is negative')
    if audio_keys.issubset(tensors):
        audio_mean, audio_std = tensors['match.audio_mean'], tensors['match.audio_std']
        if (audio_mean.dtype != torch.float32 or audio_std.dtype != torch.float32
                or tuple(audio_mean.shape) != (audio_channels, audio_dims, 1) or audio_std.shape != audio_mean.shape
                or (audio_std < 0).any().item()):
            raise ValueError('audio reference shape, dtype or standard deviation differs')
    free = previous_t - previous_overlap
    if free < 1 or not torch.equal(video[:, :, -free:], previous[:, :, previous_overlap:]):
        raise ValueError('cumulative video tail does not match the previous window')
