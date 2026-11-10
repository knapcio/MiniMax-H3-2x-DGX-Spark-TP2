"""Automatic character cards (``H3_TP2_AUTO_CARDS=<max faces>``, EXPERIMENT, off by default).

A head-and-shoulders crop per character, given as extra Ref2VA reference images next to the scene frame, is what
keeps the faces the same people over two minutes (`results/20261109-long-chain-identity/`: with the scene frame alone
the woman is another person after 90 s; with the crops both faces hold for 118 s, +25 % per step).  Until now the
crops had to be cut by hand with ``scripts/make_character_cards.py`` and passed as ``H3_REF_IMAGES``.

With ``H3_TP2_AUTO_CARDS=2`` a Ref2VA request that carries exactly one image and no video reference gets the crops
made on the server: the faces of the scene frame are detected (MTCNN, the same script, run in the interpreter of the
identity guard, ``H3_TP2_IDENTITY_PY`` or ``~/facenv/bin/python``), the up to N largest faces are cut (2.4 face widths
wide, 3 face heights tall, upscaled to a 512 px short edge, ordered left to right) and appended to the request's
images before upstream prepares the encoder inputs.  Everything downstream (the ``<Picture i>`` text tokens, the
reference blocks, the identity guard's bank, the look-lock) sees them as if the user had sent them.  Upstream builds
the prepared inputs on rank 0 only and broadcasts the result, so nothing has to agree between ranks.

Costs about 5 s of CPU once per request for the detection and about +12 % render time per card.  A request with
several images is left alone (the user chose the references); a frame without a face gets none.  A detection failure
is logged and the request goes on with the scene frame alone.  Changes conditioning; NOT output-preserving.
"""
from __future__ import annotations

import glob
import logging
import os
import subprocess
import tempfile
from collections.abc import Mapping

logger = logging.getLogger("vllm_omni.h3_tp2.auto_cards")

MAX_CARDS = 8          # ref2va accepts 9 images: the scene and up to 8 cards
TIMEOUT_S = 120.0


def parse_max(text: str | None) -> int | None:
    """``H3_TP2_AUTO_CARDS`` -> the largest number of cards to add, or None when off."""
    text = (text or "").strip()
    if not text or text in ("0", "off", "no"):
        return None
    value = int(text)
    if not 1 <= value <= MAX_CARDS:
        raise ValueError(f"H3_TP2_AUTO_CARDS={text!r}: use 1..{MAX_CARDS}")
    return value


def default_python() -> str:
    return os.environ.get("H3_TP2_IDENTITY_PY") or os.path.expanduser("~/facenv/bin/python")


def default_script() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "scripts", "make_character_cards.py")


def make_cards(image, max_cards: int, *, python: str | None = None, script: str | None = None, timeout: float = TIMEOUT_S):  # noqa: ANN001
    """PIL crops of the faces of ``image`` (left to right, at most ``max_cards``), made by the card script."""
    from PIL import Image

    python = python or default_python()
    script = script or default_script()
    with tempfile.TemporaryDirectory(prefix="h3-auto-cards-") as tmp:
        scene = os.path.join(tmp, "scene.png")
        image.save(scene)
        out = os.path.join(tmp, "cards")
        proc = subprocess.run([python, script, scene, "--out-dir", out, "--prefix", "card"], capture_output=True,
                              text=True, timeout=timeout, check=False)
        if proc.returncode != 0:
            if "no face found" in (proc.stderr or ""):
                return []
            raise RuntimeError(f"card script failed (rc {proc.returncode}): {(proc.stderr or '').strip()[-300:]}")
        paths = sorted(glob.glob(os.path.join(out, "card_face*.png")), key=lambda p: int(p.rsplit("face", 1)[1][:-4]))
        cards = []
        for path in paths[:max_cards]:
            with Image.open(path) as card:
                cards.append(card.convert("RGB").copy())
        return cards


def single_image(prompt) -> object | None:  # noqa: ANN001
    """The request's one image value when it has exactly one image and no video reference, else None."""
    if not isinstance(prompt, Mapping):
        return None
    mm = prompt.get("multi_modal_data") or {}
    if not isinstance(mm, Mapping) or mm.get("video"):
        return None
    image = mm.get("image")
    if image is None:
        return None
    if isinstance(image, (list, tuple)):
        if len(image) != 1:
            return None
        image = image[0]
    return image


def with_cards(prompt, cards: list):  # noqa: ANN001
    """A copy of the prompt dict with the cards appended to its image list."""
    mm = dict(prompt.get("multi_modal_data") or {})
    image = mm.get("image")
    first = image[0] if isinstance(image, (list, tuple)) else image
    mm["image"] = [first, *cards]
    return {**prompt, "multi_modal_data": mm}


def install(pp, max_cards: int, *, loader=None, maker=make_cards) -> None:  # noqa: ANN001
    """Wrap ``prepare_encoder_inputs`` of the pipeline module ``pp`` (the name it imported)."""
    orig = pp.prepare_encoder_inputs
    if getattr(orig, "_h3_tp2_auto_cards", False):
        return
    load = loader
    if load is None:
        from vllm_omni.model_executor.models.minimax_h3.preprocessing import load_minimax_h3_images as load

    def prepare_encoder_inputs(prompt, sampling, *, task=None, **kw):  # noqa: ANN001
        if task == "ref2va":
            value = single_image(prompt)
            if value is not None:
                try:
                    scene = load(value)[0]
                    cards = maker(scene, max_cards)
                    if cards:
                        prompt = with_cards(prompt, cards)
                        logger.info("h3_tp2 auto cards: %d face crop(s) added to the one reference image (%s)", len(cards),
                                    ", ".join(f"{c.width}x{c.height}" for c in cards))
                    else:
                        logger.info("h3_tp2 auto cards: no face found in the reference image, none added")
                except Exception as e:  # noqa: BLE001  (the request goes on with the scene frame alone)
                    logger.warning("h3_tp2 auto cards skipped (%s)", e)
        return orig(prompt, sampling, task=task, **kw)

    prepare_encoder_inputs._h3_tp2_auto_cards = True  # type: ignore[attr-defined]
    prepare_encoder_inputs.__wrapped__ = orig  # type: ignore[attr-defined]
    pp.prepare_encoder_inputs = prepare_encoder_inputs


__all__ = ["install", "make_cards", "parse_max", "single_image", "with_cards"]
