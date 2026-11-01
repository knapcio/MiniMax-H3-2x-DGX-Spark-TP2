#!/usr/bin/env python3
"""Report nearly solid or transparent image-edge bands; never crop the image."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


def _line(image: Image.Image, edge: str, index: int) -> Image.Image:
    w, h = image.size
    if edge == "top":
        return image.crop((0, index, w, index + 1))
    if edge == "bottom":
        return image.crop((0, h - index - 1, w, h - index))
    if edge == "left":
        return image.crop((index, 0, index + 1, h))
    if edge == "right":
        return image.crop((w - index - 1, 0, w - index, h))
    raise ValueError("unknown edge")


def _fraction_at_most(image: Image.Image, maximum: int) -> float:
    return sum(image.histogram()[:maximum + 1]) / (image.width * image.height)


def inspect(image: Image.Image, *, tolerance: int = 3, fraction: float = 0.995,
            max_fraction: float = 0.25) -> dict:
    """Count bands matching each edge's representative color, within a limited scan.

    Bands may be intentional content; a truncated scan is especially ambiguous.
    Alpha-only comparison is used when an edge is almost entirely transparent.
    """
    if not 0 <= tolerance <= 32 or not 0.5 <= fraction <= 1 or not 0 < max_fraction <= 0.25:
        raise ValueError("invalid border-check parameters")
    image = image.convert("RGBA")
    if min(image.size) < 4:
        raise ValueError("image must be at least four pixels in each dimension")
    bands = {}
    for edge in ("top", "bottom", "left", "right"):
        first = _line(image, edge, 0)
        transparent = _fraction_at_most(first.getchannel("A"), tolerance) >= fraction
        rgb = first.convert("RGB")
        color = tuple(int(x) for x in ImageStat.Stat(rgb).median)
        maximum = max(1, int((image.height if edge in ("top", "bottom") else image.width) * max_fraction))
        count = 0
        for i in range(maximum):
            line = _line(image, edge, i)
            if transparent:
                matches = _fraction_at_most(line.getchannel("A"), tolerance)
            else:
                color_line = line.convert("RGB")
                differences = ImageChops.difference(color_line, Image.new("RGB", color_line.size, color)).split()
                maximum_difference = ImageChops.lighter(ImageChops.lighter(differences[0], differences[1]), differences[2])
                matches = _fraction_at_most(maximum_difference, tolerance)
                if _fraction_at_most(line.getchannel("A"), 255 - tolerance - 1) > 1 - fraction:
                    matches = 0  # varying or translucent pixels are not an opaque color band
            if matches < fraction:
                break
            count += 1
        bands[edge] = {"pixels": count, "kind": "transparent" if transparent else "solid-color",
                       "rgb": None if transparent else list(color), "scan_limit_reached": count == maximum}
    return {"width": image.width, "height": image.height, "bands": bands,
            "note": "Bands may be intentional. This is a visual-review aid; no pixels were changed."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--tolerance", type=int, default=3, help="maximum per-channel difference, 0..32")
    parser.add_argument("--fraction", type=float, default=.995, help="minimum fraction of matching pixels")
    args = parser.parse_args()
    try:
        path = args.image.expanduser().resolve(strict=True)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with Image.open(path) as source:
            result = inspect(source, tolerance=args.tolerance, fraction=args.fraction)
        result.update({"source": str(path), "source_sha256": digest})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"input-border-check: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
