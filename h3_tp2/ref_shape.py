"""Reference-image shape rule behind H3_TP2_REF_NO_UPSCALE=<px> (opt-in, not output-preserving).

Upstream resizes every Ref2VA reference image to one short edge (2048 by default, ``H3_TP2_REF_SHORT_EDGE`` lowers
it), so a 512 px character crop next to a 1344x768 scene frame is upscaled to 1024 px and costs as many sequence rows
as the scene.  With ``H3_TP2_REF_NO_UPSCALE=<px>`` a reference whose short edge is below ``px`` keeps its own size
(aligned to the model's 32 px multiple): the small crops stay small, a scene frame above the threshold is normalised to
the short-edge setting as before.  Pure arithmetic, testable on a CPU.
"""

from __future__ import annotations


def align(value: float, multiple: int = 32) -> int:
    """Upstream's rounding of a side length to the model's multiple (never below one multiple)."""
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def native_or_resized(size: tuple[int, int], short_edge: int, keep_below: int, multiple: int = 32) -> tuple[int, int]:
    """(width, height) for a reference of ``size``: its own size (aligned) when the short edge is below ``keep_below``,
    otherwise scaled so that the short edge is ``short_edge`` (upstream's rule)."""
    width, height = int(size[0]), int(size[1])
    short = min(width, height)
    if short <= 0:
        raise ValueError(f"reference image size must be positive, got {width}x{height}")
    if short < keep_below:
        return align(width, multiple), align(height, multiple)
    scale = short_edge / short
    return align(width * scale, multiple), align(height * scale, multiple)
