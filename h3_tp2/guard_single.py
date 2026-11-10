"""One-window requests go through the continuation loop when a guard is configured, so the guards judge them too.

The speech and identity guards live in ``cont_match.wrap_diffuse``, which upstream reaches only through
``diffuse_continuation`` (``long_video`` requests).  A plain 4-15 s request calls ``diffuse`` directly and is never
judged: the nine single-shot renders of 2026-11-10 ran with ``H3_TP2_SPEECH_GUARD=0.03`` set and no check happened.
With a guard configured this module gives such a request a continuation of exactly one window (its own frame count as
the window, the usual 22-frame overlap that one window never uses), so the request takes the loop's first-window
path: the same arguments as the direct call (no pinned tail, no offset, the request's own seed), plus the guards and
their seed retries.  Requests longer than one window (345 frames) are left alone: splitting them would change the
render.  The same path carries the loop's other per-window levers (``H3_TP2_LADDER_TAIL``), which the direct call
skips (queue137, 2026-11-10: a one-window render with the tail set and no guard was byte for byte the stock one).
``H3_TP2_GUARD_SINGLE=0`` turns this off.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("vllm_omni.h3_tp2.guard_single")

MIN_WINDOW, MAX_WINDOW, GRID, OVERLAP = 107, 345, 17, 22


def one_window(frames) -> tuple[int, int] | None:  # noqa: ANN001
    """The (window, overlap) that makes a request of ``frames`` frames a one-window continuation, or None."""
    try:
        frames = int(frames)
    except (TypeError, ValueError):
        return None
    if not MIN_WINDOW <= frames <= MAX_WINDOW or frames % GRID != GRID - 12:   # 17n+5
        return None
    return frames, OVERLAP


def judge(context: dict, *, log=logger) -> bool:  # noqa: ANN001
    """Give a one-window request without continuation the one-window geometry; returns whether it did."""
    if not isinstance(context, dict) or context.get("continuation") is not None:
        return False
    geometry = one_window(context.get("num_frames"))
    if geometry is None:
        return False
    context["continuation"] = geometry
    log.info("h3_tp2: one-window request of %d frames goes through the continuation loop so that the guards judge it",
             geometry[0])
    return True


def install(cls, *, log=logger) -> None:  # noqa: ANN001
    """Wrap ``cls._prepare_request_inputs`` once."""
    orig = cls._prepare_request_inputs
    if getattr(orig, "_h3_tp2_guard_single", False):
        return

    def _prepare_request_inputs(self, raw_prompt, sampling):  # noqa: ANN001
        context = orig(self, raw_prompt, sampling)
        try:
            judge(context, log=log)
        except Exception as e:  # noqa: BLE001
            log.warning("h3_tp2 guard single skipped (%s)", e)
        return context

    _prepare_request_inputs._h3_tp2_guard_single = True  # type: ignore[attr-defined]
    _prepare_request_inputs.__wrapped__ = orig  # type: ignore[attr-defined]
    cls._prepare_request_inputs = _prepare_request_inputs
