"""Watch, or keep, the CUDA allocator's pool at the stage boundaries (MEASUREMENT / EXPERIMENT, off by default).

vLLM-Omni v0.30.0 returns the allocator's idle pages to the device at every stage boundary of a request
(``MiniMaxH3Pipeline._release_stage_cache`` after the reference encode, before and after the decode, and
``torch.accelerator.empty_cache()`` whenever a component goes back to the CPU).  On the GB10 the device memory is the
host's memory, so the next stage takes its pages back from whatever the host's free memory looks like at that
moment.  Hypothesis (unproven, 2026-11-09): a pool acquired from fragmented host memory is what makes a render 25 %
slower (seen for whole chains and once from the fourth window of a chain on, each chain in its own freshly booted
head process that had passed the rate probe).  A chain without a guard releases nothing between its windows, so the
hypothesis does not cover every case seen; `observe` is there to find out.

``H3_TP2_KEEP_POOL=observe`` logs every release (reserved and allocated before and after, MemAvailable, the device
allocation counters) and changes nothing.  ``H3_TP2_KEEP_POOL=<GiB>`` skips a release while the host's MemAvailable is
at or above that many GiB, so the pool and its physical pages stay for the life of the process, and releases as stock
below it.  No tensor arithmetic is touched either way; the latent digests are the evidence.
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger("vllm_omni.h3_tp2.pool")

MARK = "_h3_tp2_keep_pool"
GIB = float(1 << 30)
LOG_FIRST = 40      # every release is logged up to here,
LOG_EVERY = 25      # then one in this many (a tiled decode can release often)


def parse(text):
    """``None`` (off), ``("observe", 0.0)`` or ``("keep", <floor GiB>)``."""
    text = (text or "").strip().lower()
    if text in ("", "0", "off", "no", "false"):
        return None
    if text == "observe":
        return ("observe", 0.0)
    try:
        floor = float(text)
    except ValueError:
        raise ValueError(f"H3_TP2_KEEP_POOL must be 'observe' or a MemAvailable floor in GiB, got {text!r}") from None
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError(f"H3_TP2_KEEP_POOL floor must be a positive number of GiB, got {text!r}")
    return ("keep", floor)


def mem_available_gib(path="/proc/meminfo"):
    """The host's MemAvailable in GiB, or ``None`` when it cannot be read."""
    try:
        with open(path, encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / GIB
    except (OSError, ValueError, IndexError):
        return None
    return None


def cuda_stats(torch_module):
    """(reserved GiB, allocated GiB, device allocations, device frees) of the current device, or ``None``."""
    try:
        stats = torch_module.cuda.memory_stats()
        return (stats.get("reserved_bytes.all.current", 0) / GIB, stats.get("allocated_bytes.all.current", 0) / GIB,
                int(stats.get("num_device_alloc", 0)), int(stats.get("num_device_free", 0)))
    except Exception:  # noqa: BLE001 - a measurement must never break a render
        return None


class Gate:
    """Decides and logs what one release does."""

    def __init__(self, mode, floor, *, available=mem_available_gib, stats=lambda: None, log=logger):
        if mode not in ("observe", "keep"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode, self.floor = mode, float(floor)
        self._available, self._stats, self._log = available, stats, log
        self.calls = self.kept = self.released = 0

    def __call__(self, original, where):
        self.calls += 1
        available = self._available()
        # Without a reading there is no evidence that keeping is safe: release as stock.
        keep = self.mode == "keep" and available is not None and available >= self.floor
        before = self._stats()
        if keep:
            self.kept += 1
        else:
            original()
            self.released += 1
        if self.calls <= LOG_FIRST or self.calls % LOG_EVERY == 0:
            after = before if keep else self._stats()
            self._log.info("h3_tp2 pool: release %d via %s %s: reserved %s -> %s GiB, allocated %s GiB, device allocs %s "
                           "frees %s, MemAvailable %s GiB (kept %d, released %d)", self.calls, where,
                           "KEPT" if keep else "done", _f(before, 0), _f(after, 0), _f(after, 1), _i(after, 2),
                           _i(after, 3), "?" if available is None else f"{available:.1f}", self.kept, self.released)
        return keep


def _f(stats, index):
    return "?" if stats is None else f"{stats[index]:.2f}"


def _i(stats, index):
    return "?" if stats is None else str(stats[index])


def install(torch_module, mode, floor, *, available=mem_available_gib, log=logger):
    """Wrap ``torch.cuda.empty_cache`` and ``torch.accelerator.empty_cache`` once; returns the gate."""
    existing = getattr(torch_module, MARK, None)
    if existing is not None:
        return existing
    gate = Gate(mode, floor, available=available, stats=lambda: cuda_stats(torch_module), log=log)
    for owner_name in ("cuda", "accelerator"):
        owner = getattr(torch_module, owner_name, None)
        original = getattr(owner, "empty_cache", None)
        if original is None:
            continue

        def wrapped(*args, _original=original, _where=owner_name, **kwargs):
            gate(lambda: _original(*args, **kwargs), _where)

        wrapped.__wrapped_stock__ = original
        setattr(owner, "empty_cache", wrapped)
    setattr(torch_module, MARK, gate)
    return gate
