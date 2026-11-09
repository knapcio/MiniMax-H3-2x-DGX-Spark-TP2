"""A low-noise tail on the sigma ladder (``H3_TP2_LADDER_TAIL``, EXPERIMENT, off by default).

The stock ladder of a 20-step request is a uniform grid in u, shifted by the flow shift: with the video shift of 12
the sigmas are 12u / (1 + 11u), so twelve of the twenty forwards run above sigma 0.92 and the last three start at
0.68, 0.57 and 0.39.  The sampler is deterministic Euler, so a window's fine detail is the x0 prediction made with
39 % noise still on the rows, once, and then handed to the next window as its clean truth.  Hypothesis
(2026-11-09, a code review): that one posterior-mean jump is where the faces and the texture of a chain soften.

``H3_TP2_LADDER_TAIL=0.025,0.0125`` appends those u positions before the final 0, so the last video sigmas become
0.39 -> 0.24 -> 0.13 -> 0 (two more forwards, about +10 % per window).  The first steps are the stock ones on the same
noise and seed, so a paired comparison changes only the end of the ladder.  The audio pin release step is a fraction
of the step count (``audio_pin_steps``); keep it at the same step with ``H3_TP2_CONT_EDIT_AUDIO_STEPS`` when comparing
(0.27 instead of 0.3 for 20 + 2 steps).  Changes every window's output; NOT output-preserving.
"""
from __future__ import annotations

import math


def parse_tail(text: str | None) -> tuple[float, ...] | None:
    """``H3_TP2_LADDER_TAIL`` -> strictly decreasing u positions in (0, 1), or None when the ladder stays."""
    text = (text or "").strip()
    if not text or text in ("0", "off"):
        return None
    values = tuple(float(v) for v in text.split(","))
    if any(not math.isfinite(v) or not 0.0 < v < 1.0 for v in values):
        raise ValueError(f"H3_TP2_LADDER_TAIL={text!r}: positions must be in (0, 1)")
    if any(b >= a for a, b in zip(values, values[1:])):
        raise ValueError(f"H3_TP2_LADDER_TAIL={text!r}: positions must decrease")
    return values


def positions(num_steps: int, tail: tuple[float, ...]) -> list[float]:
    """The stock uniform grid of ``num_steps`` steps with ``tail`` inserted before the final 0."""
    num_steps = int(num_steps)
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    grid = [round(1.0 - i / num_steps, 10) for i in range(num_steps)]   # 1.0 .. 1/num_steps, the final 0 added below
    if tail and tail[0] >= grid[-1]:
        raise ValueError(f"the ladder tail must start below the last stock position {grid[-1]:.4f}, got {tail[0]}")
    return [*grid, *tail, 0.0]


def apply(args: dict, tail: tuple[float, ...] | None) -> dict:
    """The window arguments with the extended ladder (``base_schedule`` + ``num_steps``), or ``args`` itself."""
    if not tail:
        return args
    if args.get("base_schedule") is not None:
        raise ValueError("H3_TP2_LADDER_TAIL cannot extend a checkpoint that pins its own base_schedule")
    grid = positions(int(args["num_steps"]), tail)
    return {**args, "base_schedule": grid, "num_steps": len(grid) - 1}
