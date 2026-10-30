"""Server-side per-window prompts for long_video continuation (what scripts/render.sh CONT_PROMPTS=auto did client-side).

Upstream encodes one prompt for every continuation window unless the request carries ``continuation_prompts``; with the
same prompt in every window the model re-introduces the prompt's subject in each window (a second "woman in white"
after the seam) and hands over with a visible bump. The wording measured on 2026-10-27/28 (one prompt per window keeps
the motion and the shot continuous, "one single continuous take" removes the hand-over bumps) is applied here when a
request asks for continuation and sends no prompts of its own, so a plain agent request gets it by default.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

FIRST_SUFFIX = ". One single continuous take from a steady camera, no cuts, no transitions."
REST_PREFIX = "The same continuous shot goes on without any cut or transition, the camera keeps moving the same way: "


def window_count(duration_s: float, window_frames: int = 277, overlap_frames: int = 22, fps: int = 24) -> int:
    """Number of continuation windows upstream will render for ``duration_s`` (frames rounded to 17k+5 as upstream)."""
    return window_count_from_frames(int(round(float(duration_s) * fps)), window_frames, overlap_frames)


def window_count_from_frames(frames: int, window_frames: int = 277, overlap_frames: int = 22) -> int:
    """Count windows from the requested, not yet grid-aligned, frame count."""
    frames = ((frames - 5 + 16) // 17) * 17 + 5
    if frames <= window_frames:
        return 1
    return 1 + math.ceil((frames - window_frames) / (window_frames - overlap_frames))


def continuation_prompts(base: str, n_windows: int, extra: str = "", rest_template: str = "") -> list[str]:
    """``rest_template`` (H3_TP2_CONT_PROMPTS_REST) replaces the later windows' wording; ``{base}`` in it is the prompt."""
    base = base.strip()
    extra = extra.strip()
    first = base.rstrip(". ") + FIRST_SUFFIX + (" " + extra if extra else "")
    if rest_template.strip():
        rest = rest_template.strip().replace("{base}", base)
    else:
        rest = REST_PREFIX + base
    rest = rest + (" " + extra if extra else "")
    return [first] + [rest] * (n_windows - 1)


def request_duration(extra_args: dict) -> float | None:
    """Match encoder_processing's field precedence, including explicit nulls.

    An explicit null selects the encoder's num_frames/default path; it must
    not silently fall through to a lower-priority duration.
    """
    target = extra_args.get("target")
    if target is not None and not isinstance(target, Mapping):
        raise ValueError("target must be an object")
    target = target if target is not None else {}
    value = target.get("duration_seconds", extra_args.get("duration_seconds", extra_args.get("duration")))
    return None if value is None else float(value)


def request_window_count(extra_args: dict, num_frames: int | None = None) -> int | None:
    """Count from the encoder's effective duration or, when null, its explicit frame count.

    None means the encoder chooses its task-specific short default; no per-window
    prompts are necessary in that case.
    """
    duration = request_duration(extra_args)
    window = int(extra_args.get("continuation_window_frames", 277))
    overlap = int(extra_args.get("continuation_overlap_frames", 22))
    if duration is not None:
        return window_count(duration, window, overlap)
    if int(num_frames or 1) > 1:
        return window_count_from_frames(int(num_frames), window, overlap)
    return None
