"""Wall time of every window sampling, so a host that slows down under a render shows in the log.

Two samplings of the same size (task, latent shape, audio length, steps, conditioning shapes) do the same work whatever
the seed or the prompt, so they should take the same time.  On the two-Spark stack they sometimes do not: a head process
that sampled a 277-frame 960x544 window in 555 s has been seen to need 700 s for the same window an hour later, for the
rest of the process's life (2026-11-08 and 2026-11-09, `results/20261106-voice/` section 15).  Nothing in the request
causes that and nothing in the output shows it; only the wall time does.

`report` keeps, per process, the fastest sampling seen for each size and logs every sampling against it: an info line
normally, a warning when a sampling is more than `TOLERANCE` slower than the fastest of its size.  No tensors are
touched.  Pure python.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("vllm_omni.h3_tp2.pace")

TOLERANCE = 0.15  # a continued window costs 4-5 % less than the first sampling of a process, re-runs agree within 1 %
_BEST: dict[tuple, float] = {}  # size -> fastest seconds seen in this process, across requests


def key(args: dict) -> tuple:
    """What decides the cost of one sampling: the task, the latent and audio sizes, the steps and the shapes of the
    conditioning inputs (reference images, the hand-over guide).  Seeds, prompts and offsets do not."""
    shapes = args.get("visual_condition_shapes") or []
    return (args.get("task"), args.get("latent_t"), args.get("latent_h"), args.get("latent_w"), args.get("audio_t"),
            args.get("num_steps"), tuple(tuple(int(x) for x in shape) for shape in shapes),
            tuple(int(x) for x in (args.get("audio_condition_lengths") or [])))


def note(size: tuple, seconds: float, best: dict | None = None) -> float | None:
    """Record one sampling.  Returns how much slower it was than the fastest earlier sampling of the same size
    (0.25 = 25 % slower, negative = a new fastest), or None for the first sampling of its size."""
    best = _BEST if best is None else best
    fastest = best.get(size)
    if fastest is None or seconds < fastest:
        best[size] = seconds
    if fastest is None or fastest <= 0:
        return None
    return seconds / fastest - 1.0


REPLAY_FLOOR_S = 5.0   # a sampling that took less than this was replayed from saved latents, not sampled


def report(index: int, attempt: int, args: dict, seconds: float, best: dict | None = None) -> float | None:
    """Log one sampling against the fastest of its size; returns `note`'s value.  Never raises."""
    try:
        if seconds < REPLAY_FLOOR_S:   # H3_TP2_REPLAY_LATENTS: no denoising happened, nothing to time or to compare with
            logger.info("h3_tp2 pace window %d attempt %d: %.1f s (replayed, not timed)", index, attempt, seconds)
            return None
        size = key(args)
        fastest = (_BEST if best is None else best).get(size)
        slower = note(size, seconds, best)
        steps = args.get("num_steps")
        rate = f" ({seconds / int(steps):.1f} s/step)" if steps else ""
        if slower is not None and slower > TOLERANCE:
            logger.warning("h3_tp2 pace window %d attempt %d: %.0f s%s, %.0f %% slower than the fastest sampling of this "
                           "size in this process (%.0f s). The work is the same, so the host slowed down under the render",
                           index, attempt, seconds, rate, 100.0 * slower, fastest)
        else:
            logger.info("h3_tp2 pace window %d attempt %d: %.0f s%s", index, attempt, seconds, rate)
        return slower
    except Exception as e:  # noqa: BLE001  (a diagnostic never breaks a render)
        logger.warning("h3_tp2 pace skipped (%s)", e)
        return None


def clock() -> float:
    return time.perf_counter()
