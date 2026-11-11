"""The first seconds of window 1's audio as a Ref2VA audio reference for every later window (``H3_TP2_VOICE_ANCHOR``).

Over a long chain the voices drift: the speaker similarity to window 1 falls to 0.75-0.94 by the second minute
(`results/20261109-long-chain-identity/`), the audio analogue of the faces drifting before the crops.  Ref2VA takes
audio references (``ref_blocks`` of kind ``audio``: packed audio-VAE latent rows shown clean, placed at the window's
time origin, so they ride with the window), which a request can give as a voice sample.  This module gives every
window after the first the opening ``seconds`` of the chain's own audio (window 1's latent, already in latent space,
no decode) as such a reference: a voice sample that does not drift because it never changes.

EXPERIMENT (2026-11-11, from a code review): the trained meaning of an audio reference is not documented (a voice to
keep, a sound to continue?), so the first run measures speaker similarity per window, UTMOS and whether the lines
are echoed.  Pure function of the loop's state, the same on both ranks.  Changes every window after the first.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("vllm_omni.h3_tp2.voice_anchor")

AUDIO_HZ = 40.0     # audio latent steps per second (462 steps for an 11.54 s window)
MAX_SECONDS = 15.0


def parse_seconds(text: str | None) -> float | None:
    """``H3_TP2_VOICE_ANCHOR`` -> seconds of window 1's audio to reference, or None when off."""
    text = (text or "").strip()
    if not text or text in ("0", "off"):
        return None
    value = float(text)
    if not 0.5 <= value <= MAX_SECONDS:
        raise ValueError(f"H3_TP2_VOICE_ANCHOR={text!r}: use 0.5..{MAX_SECONDS} seconds")
    return value


def steps(seconds: float) -> int:
    return int(round(float(seconds) * AUDIO_HZ))


def apply(args: dict, audio, seconds: float, pack) -> dict:  # noqa: ANN001
    """The window arguments with the anchor block added first, or ``args`` itself when nothing applies.

    ``audio`` is the chain's audio latent so far (``[channels, dim, T]``; window 1's audio opens it); ``pack`` is
    upstream's ``minimax_h3_pack_audio_latent``.  The anchor goes FIRST among the audio conditions and blocks so
    that the hand-over guide, when there is one, stays last (``cont_match.overlap_audio_t`` reads the last length).
    """
    if audio is None or args.get("task") != "ref2va":
        return args
    t = min(steps(seconds), int(audio.shape[-1]))
    if t <= 0:
        return args
    rows = pack(audio[..., :t])
    out = dict(args)
    existing = out.get("audio_condition")
    out["audio_condition"] = rows if existing is None else _cat(rows, existing)
    out["audio_condition_lengths"] = [t, *(out.get("audio_condition_lengths") or [])]
    blocks = list(out.get("ref_blocks") or [])
    # upstream builds [image, audio(ref_audio_t)] itself when ref_blocks is None; with a list, add the anchor block
    # before any other audio or guide block and after the images
    images = [b for b in blocks if isinstance(b, dict) and b.get("kind") == "image"]
    rest = [b for b in blocks if not (isinstance(b, dict) and b.get("kind") == "image")]
    anchor = {"kind": "audio", "ref_audio_t": t}
    if blocks:
        rest = [b for b in rest if not (isinstance(b, dict) and b.get("kind") == "audio" and int(b.get("ref_audio_t") or 0) == 0)]
        out["ref_blocks"] = [*images, anchor, *rest]
    out["ref_audio_t"] = t
    logger.info("h3_tp2 voice anchor: the first %.1f s of window 1's audio (%d steps, %d rows) given as an audio reference",
                t / AUDIO_HZ, t, int(rows.shape[0]))
    return out


def _cat(a, b):  # noqa: ANN001
    import torch

    return torch.cat((a, b), dim=0)
