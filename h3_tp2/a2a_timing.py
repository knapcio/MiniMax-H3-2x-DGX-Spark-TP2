"""Opt-in timing of the Ulysses all-to-alls (``H3_TP2_A2A_TIMING=1``).

Wraps ``vllm_omni.diffusion.distributed.comm.all_to_all_4D`` (the permute copies plus the collective) and the
``torch.distributed.all_to_all_single`` call inside it with CUDA events on the current stream, and logs the totals at
the end of every ``diffuse`` phase.  Measurement only: no tensor is touched, the numerics are those of the stock path.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vllm_omni.h3_tp2.a2a")

_outer: list = []  # (start_event, end_event, bytes)
_inner: list = []
_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    import torch
    import torch.distributed as dist
    from vllm_omni.diffusion.distributed import comm

    orig_outer = comm.all_to_all_4D
    orig_inner = dist.all_to_all_single

    def timed_outer(input, *a, **kw):  # noqa: ANN001
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            return orig_outer(input, *a, **kw)
        finally:
            e.record()
            _outer.append((s, e, input.numel() * input.element_size()))

    def timed_inner(output, input, *a, **kw):  # noqa: ANN001
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            return orig_inner(output, input, *a, **kw)
        finally:
            e.record()
            _inner.append((s, e, input.numel() * input.element_size()))

    comm.all_to_all_4D = timed_outer
    comm.dist.all_to_all_single = timed_inner  # comm looks the name up on its ``dist`` module at call time
    _installed = True
    logger.info("h3_tp2: all-to-all timing installed")


def report(phase: str = "diffuse") -> None:
    """Sum the recorded events (synchronising the device first) and log them; clears the record."""
    if not _installed or not _outer:
        return
    import torch

    torch.cuda.synchronize()

    def _sum(rec):
        ms = 0.0
        nbytes = 0
        for s, e, b in rec:
            ms += s.elapsed_time(e)
            nbytes += b
        return ms / 1000.0, nbytes

    o_s, o_b = _sum(_outer)
    i_s, i_b = _sum(_inner)
    logger.info(
        "h3_tp2 a2a %s: calls=%d all_to_all_4D=%.1f s (%.1f GB in) collective=%.1f s (%.1f GB in)",
        phase, len(_outer), o_s, o_b / 1e9, i_s, i_b / 1e9,
    )
    _outer.clear()
    _inner.clear()
