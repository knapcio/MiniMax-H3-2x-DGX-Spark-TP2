"""Reference images ride with the window in RoPE time (``H3_TP2_REF_FOLLOW_WINDOW=1``, EXPERIMENT, off by default).

A Ref2VA request puts its reference images at the media time origin (``packed_sequence.py``: image blocks sit at
``t_cursor`` from ``media_time_origin``), and a continuation window's target rows are shifted by ``temporal_offset``
(the window's start in 40 Hz RoPE units: 425 per window) while image blocks are explicitly left where they are.  So
window 6 is about 2,100 units from its reference and window 11 about 4,200, against a few hundred inside a training
clip.  Hypothesis (2026-11-09, a code review): the reference's pull on the faces decays with that distance, which is
why identity holds for about 60 s and drifts afterwards.

With the switch the image reference rows get the window's ``temporal_offset`` added to their time coordinate, so
every window sees its references at the distance window 1 saw them.  The text stays where it is (unlike the ``toff0``
mode of ``identity_lock``, which moves the whole window back and costs voice).  A pure function of the layout, the
same on both ranks.  Changes every window after the first; NOT output-preserving.
"""
from __future__ import annotations

import logging


logger = logging.getLogger("vllm_omni.h3_tp2.ref_follow")

IMAGE_TIME_SPAN = 16.0   # image blocks sit at origin, origin + 1, ...: at most 9 of them, well inside this span


def shift_image_rows(packed: dict, temporal_offset: float, origin: float) -> int:
    """Add ``temporal_offset`` to the time coordinate of the image reference rows of ``packed``; returns the count."""
    offset = float(temporal_offset or 0.0)
    if offset == 0.0:
        return 0
    img_pos, update = packed["img_pos"], packed["update_mask"]
    ref_rows = img_pos[~update]
    if ref_rows.numel() == 0:
        return 0
    g = packed["img_position_ids"]
    t = g[ref_rows, 0]
    # image blocks are unshifted at origin + i; latent guides and video references already carry the offset
    image = (t >= float(origin) - 0.5) & (t < float(origin) + IMAGE_TIME_SPAN)
    rows = ref_rows[image]
    if rows.numel() == 0:
        return 0
    g[rows, 0] += offset
    return int(rows.numel())


def install(pp) -> None:  # noqa: ANN001
    """Wrap ``minimax_h3_packed_sequence_ref2va_blocks`` of the pipeline module ``pp``."""
    orig = pp.minimax_h3_packed_sequence_ref2va_blocks
    if getattr(orig, "_h3_tp2_ref_follow", False):
        return

    def minimax_h3_packed_sequence_ref2va_blocks(*a, **kw):  # noqa: ANN001
        packed = orig(*a, **kw)
        try:
            offset = float(kw.get("temporal_offset") or 0.0)
            origin = kw.get("media_time_origin")
            origin = float(kw["text_len"]) if origin is None else float(origin)
            moved = shift_image_rows(packed, offset, origin)
            if moved:
                logger.info("h3_tp2 reference follows the window: %d image reference rows moved by %.1f RoPE units",
                            moved, offset)
        except Exception as e:  # noqa: BLE001
            logger.warning("h3_tp2 reference follow skipped (%s)", e)
        return packed

    minimax_h3_packed_sequence_ref2va_blocks._h3_tp2_ref_follow = True  # type: ignore[attr-defined]
    minimax_h3_packed_sequence_ref2va_blocks.__wrapped__ = orig  # type: ignore[attr-defined]
    pp.minimax_h3_packed_sequence_ref2va_blocks = minimax_h3_packed_sequence_ref2va_blocks


__all__ = ["install", "shift_image_rows"]
