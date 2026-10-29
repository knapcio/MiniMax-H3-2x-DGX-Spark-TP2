"""Server-side per-window prompts for long_video continuation (what scripts/render.sh CONT_PROMPTS=auto did client-side).

Upstream encodes one prompt for every continuation window unless the request carries ``continuation_prompts``; with the
same prompt in every window the model re-introduces the prompt's subject in each window (a second "woman in white"
after the seam) and hands over with a visible bump. The wording measured on 2026-10-27/28 (one prompt per window keeps
the motion and the shot continuous, "one single continuous take" removes the hand-over bumps) is applied here when a
request asks for continuation and sends no prompts of its own, so a plain agent request gets it by default.
"""
from __future__ import annotations

import math

FIRST_SUFFIX = ". One single continuous take from a steady camera, no cuts, no transitions."
REST_PREFIX = "The same continuous shot goes on without any cut or transition, the camera keeps moving the same way: "


def window_count(duration_s: float, window_frames: int = 277, overlap_frames: int = 22, fps: int = 24) -> int:
    """Number of continuation windows upstream will render for ``duration_s`` (frames rounded to 17k+5 as upstream)."""
    frames = int(round(float(duration_s) * fps))
    frames = ((frames - 5 + 16) // 17) * 17 + 5
    if frames <= window_frames:
        return 1
    return 1 + math.ceil((frames - window_frames) / (window_frames - overlap_frames))


def continuation_prompts(base: str, n_windows: int, extra: str = "") -> list[str]:
    base = base.strip()
    extra = extra.strip()
    first = base.rstrip(". ") + FIRST_SUFFIX + (" " + extra if extra else "")
    rest = REST_PREFIX + base + (" " + extra if extra else "")
    return [first] + [rest] * (n_windows - 1)


def request_duration(extra_args: dict) -> float | None:
    for key in ("duration_seconds", "duration"):
        if extra_args.get(key) is not None:
            return float(extra_args[key])
    target = extra_args.get("target") or {}
    if isinstance(target, dict) and target.get("duration_seconds") is not None:
        return float(target["duration_seconds"])
    return None
