"""Listening to a window before its sampling is over.

The speech guard (``h3_tp2/speech_guard.py``) judges a window after its last denoising step, so a window that slips
costs a whole sampling before it is sampled again.  Every step of the denoising loop already computes the clean
estimate of the audio rows (``x0``) and hands it to ``minimax_h3_euler_eta0_step``; this module taps that call, so the
estimate of chosen steps can be decoded and transcribed while the sampling goes on.

Two uses.  ``H3_TP2_SPEECH_EARLY=<steps>`` only observes (measurement: is a slip already audible at step 8 or 10 of
20?).  ``H3_TP2_SPEECH_EARLY_STOP=<step>[:<max CER>][,...]`` acts on it: a sink may raise ``Stop`` and the sampling in
progress ends there, so an attempt the guard would refuse costs the steps up to the check, not all of them.

The tap only reads: the update function is called with the same arguments and its result is returned untouched, so a
sampling that is not stopped is the sampling without the tap bit for bit.  It is installed once per process and does
nothing outside ``tapping``.

Pure Python apart from the tensors it passes through.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

logger = logging.getLogger("vllm_omni.h3_tp2.speech")

# upstream denoise_loop.py (v0.30.0, lines 319-458): one update of the video rows, then one of the audio rows, per step
CALLS_PER_STEP = 2
AUDIO_CALL = 2

_ACTIVE: "Tap | None" = None


class Stop(Exception):
    """Raised by a sink: give this sampling up at ``step`` (its lines came out with ``cer`` above ``limit``)."""

    def __init__(self, step: int, cer: float, limit: float):
        super().__init__(f"sampling given up at step {step}: CER {cer:.4f} > {limit:g}")
        self.step, self.cer, self.limit = int(step), float(cer), float(limit)


def parse_stops(text: str) -> dict[int, float | None]:
    """``"10:0.03,14"`` -> {10: 0.03, 14: None}: denoising steps (from 1) at which a failing attempt is given up, each
    with its own tolerated character error rate (None: the guard's limit)."""
    out: dict[int, float | None] = {}
    for item in (text or "").replace(" ", "").split(","):
        if not item:
            continue
        key, _, value = item.partition(":")
        try:
            step, limit = int(key), float(value) if value else None
        except ValueError:
            raise ValueError(f"expected <step>[:<max CER>] items separated by commas, got {item!r}") from None
        if step < 1:
            raise ValueError(f"denoising steps start at 1, got {item!r}")
        if limit is not None and not 0.0 < limit <= 1.0:
            raise ValueError(f"the tolerated character error rate is a fraction in (0, 1], got {item!r}")
        out[step] = limit
    return dict(sorted(out.items()))


def parse_steps(text: str) -> tuple[int, ...]:
    """``"6,8,10"`` -> (6, 8, 10): 1-based denoising steps, sorted, without repeats."""
    out = set()
    for item in (text or "").replace(" ", "").split(","):
        if not item:
            continue
        try:
            step = int(item)
        except ValueError:
            raise ValueError(f"expected denoising steps separated by commas, got {item!r}") from None
        if step < 1:
            raise ValueError(f"denoising steps start at 1, got {item!r}")
        out.add(step)
    return tuple(sorted(out))


class Tap:
    """Counts the update calls of one sampling and hands the audio estimate of the chosen steps to ``sink``."""

    def __init__(self, steps, sink):  # noqa: ANN001
        self.steps = frozenset(int(s) for s in steps)
        self.sink = sink
        self.calls = 0

    def seen(self, x0) -> None:  # noqa: ANN001
        self.calls += 1
        if (self.calls - 1) % CALLS_PER_STEP != AUDIO_CALL - 1:
            return
        step = (self.calls - 1) // CALLS_PER_STEP + 1
        if step not in self.steps:
            return
        try:
            self.sink(step, x0)
        except Stop:
            raise
        except Exception as e:  # noqa: BLE001  (a failing check never breaks a render)
            logger.warning("h3_tp2 speech early check skipped at step %d (%s)", step, e)


def install(dl, *, source_ok: bool = True) -> bool:  # noqa: ANN001
    """Wrap ``dl.minimax_h3_euler_eta0_step`` (the upstream ``denoise_loop`` module) once.  ``source_ok`` False (the
    loop is not the reviewed source, so the order of the calls is not known) leaves the module alone."""
    if not source_ok:
        logger.warning("h3_tp2 speech early check: the denoising loop is not the reviewed source, tap not installed")
        return False
    orig = getattr(dl, "minimax_h3_euler_eta0_step", None)
    if orig is None:
        logger.warning("h3_tp2 speech early check: no minimax_h3_euler_eta0_step in the loop module, tap not installed")
        return False
    if getattr(orig, "_h3_tp2_speech_tap", False):
        return True

    def minimax_h3_euler_eta0_step(target, x0, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        tap = _ACTIVE
        if tap is not None:
            tap.seen(x0)
        return orig(target, x0, *args, **kwargs)

    minimax_h3_euler_eta0_step._h3_tp2_speech_tap = True
    minimax_h3_euler_eta0_step.__wrapped__ = orig
    dl.minimax_h3_euler_eta0_step = minimax_h3_euler_eta0_step
    return True


@contextmanager
def tapping(steps, sink):  # noqa: ANN001
    """Tap the sampling that runs inside the block; a no-op without steps or without a sink."""
    global _ACTIVE
    if not steps or sink is None:
        yield None
        return
    previous, tap = _ACTIVE, Tap(steps, sink)
    _ACTIVE = tap
    try:
        yield tap
    finally:
        _ACTIVE = previous
