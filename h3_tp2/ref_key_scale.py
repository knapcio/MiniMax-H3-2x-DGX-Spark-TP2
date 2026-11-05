"""Reference-key scaling (``H3_TP2_REF_KEY_SCALE``, opt-in, NOT output-preserving): multiply the attention keys of the
reference-image rows of a Ref2VA request by a factor, so every video token attends more strongly to the reference
images (the scene frame and the character crops/cards).  Experiment of 2026-11-05 against identity loss after an
occlusion: with several reference images the identities are held better while visible (+0.1-0.2 face similarity) but
a character re-entering after being hidden still comes back weaker (0.57 against 0.83 before); the hypothesis is that
the reference blocks lose the attention contest against the temporal context at re-entry.

Scaling the keys of the reference rows by ``s`` scales their attention logits by ``s`` (q . (s k) = s (q . k)) for every
query, before the softmax; values are untouched.  ``H3_TP2_REF_KEY_SCALE=1.3`` scales every reference-image block,
``H3_TP2_REF_KEY_SCALE=1,1.5,1.5`` gives each image block (request order: the scene first, then the crops) its own
factor; ``1`` or unset = off.

Layout: upstream packs a Ref2VA request as text | reference blocks (image blocks first, in request order, each
``(latent_h // 2) * (latent_w // 2)`` rows) | reference audio | target audio | target video, so the image rows are
``[text_len, text_len + sum(rows))``; the packer wrapper records them per request (every window of a continuation packs
again).  Strict Ulysses gives each rank the contiguous row span ``[start, start + chunk)`` of that sequence, and the
exact exchange (``ulysses_exact``, the serving default) holds this rank's keys in ``send_k`` [P, M, hl, d] with M = the
local rows: the hook scales ``send_k[:, local reference rows]`` in place before the all-to-all, in every DiT layer.
Without the exact exchange (``H3_TP2_ULYSSES_EXACT=0``) nothing is scaled and a warning says so.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("vllm_omni.h3_tp2.ref_key_scale")

_STATE: dict = {"factors": None, "blocks": None, "seq_len": None, "cache": {}, "logged": set()}


def parse_factors(spec: str) -> list[float] | None:
    """``"1.3"`` -> [1.3]; ``"1,1.5,1.5"`` -> [1.0, 1.5, 1.5]; unset / ``"1"`` -> None (off)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    factors = [float(x) for x in spec.split(",")]
    if any(not (0 < f < 16) for f in factors):
        raise ValueError(f"H3_TP2_REF_KEY_SCALE factors must be in (0, 16), got {spec!r}")
    return None if all(f == 1.0 for f in factors) else factors


def configure(spec: str) -> list[float] | None:
    _STATE["factors"] = parse_factors(spec)
    _STATE["blocks"] = None
    _STATE["cache"] = {}
    return _STATE["factors"]


def image_blocks(text_len: int, ref_blocks) -> list[tuple[int, int]]:  # noqa: ANN001
    """The packed row ranges of the leading image reference blocks: [(start, stop), ...] in request order."""
    out: list[tuple[int, int]] = []
    cursor = int(text_len)
    for raw in ref_blocks or ():
        kind = raw.get("kind", raw.get("type"))
        if kind != "image":
            break   # image blocks come first in the requests this plugin builds; anything after is not an image
        rows = (int(raw["latent_h"]) // 2) * (int(raw["latent_w"]) // 2)
        out.append((cursor, cursor + rows))
        cursor += rows
    return out


def record_layout(text_len: int, ref_blocks, seq_len: int) -> None:  # noqa: ANN001
    """Called by the packer wrapper for every packed Ref2VA sequence."""
    if _STATE["factors"] is None:
        return
    blocks = image_blocks(text_len, ref_blocks)
    _STATE["blocks"] = blocks
    _STATE["seq_len"] = int(seq_len)
    _STATE["cache"] = {}
    key = (tuple(blocks), int(seq_len))
    if key not in _STATE["logged"]:
        _STATE["logged"].add(key)
        logger.warning("h3_tp2 ref key scale %s on %d image reference block(s) %s of a %d-row sequence",
                       _STATE["factors"], len(blocks), blocks, int(seq_len))


def local_scale(start: int, chunk: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor] | None:  # noqa: ANN001
    """(local row indices, per-row factors) for this rank's span, or None when nothing falls into it."""
    blocks, factors = _STATE["blocks"], _STATE["factors"]
    if not blocks or factors is None:
        return None
    key = (int(start), int(chunk), str(device), str(dtype))
    if key in _STATE["cache"]:
        return _STATE["cache"][key]
    idx, fac = [], []
    for i, (a, b) in enumerate(blocks):
        f = factors[min(i, len(factors) - 1)]
        lo, hi = max(a, start), min(b, start + chunk)
        if hi > lo and f != 1.0:
            idx.extend(range(lo - start, hi - start))
            fac.extend([f] * (hi - lo))
    result = None
    if idx:
        result = (torch.tensor(idx, dtype=torch.long, device=device), torch.tensor(fac, dtype=dtype, device=device))
    _STATE["cache"][key] = result
    return result


def scale_send_k(send_k: torch.Tensor, span) -> int:  # noqa: ANN001
    """Scale this rank's reference keys in ``send_k`` [P, M, hl, d] in place; ``span`` = (start, chunk) of the local
    rows in the packed sequence.  Returns the number of rows scaled."""
    if _STATE["factors"] is None or _STATE["blocks"] is None:
        return 0
    start, chunk = span
    if int(chunk) != int(send_k.shape[1]):
        return 0   # not the DiT sequence this layout describes (a self-check sub-call, another shape)
    ls = local_scale(start, chunk, send_k.device, send_k.dtype)
    if ls is None:
        return 0
    idx, fac = ls
    send_k[:, idx] *= fac[None, :, None, None]
    return int(idx.numel())
