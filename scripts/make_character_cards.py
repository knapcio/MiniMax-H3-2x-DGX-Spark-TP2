#!/usr/bin/env python3
"""Cut head-and-shoulders reference crops of every character out of a scene reference frame, for a Ref2VA request
with several reference images (`H3_REF_IMAGES=scene.png,crop0.png,...` in scripts/render.sh).

  python scripts/make_character_cards.py scene.png --out-dir refs/            # MTCNN detection (facenet_pytorch)
  python scripts/make_character_cards.py scene.png --out-dir refs/ --box 317,130,415,256 --box 951,180,1034,282
  python scripts/make_character_cards.py scene.png --out-dir refs/ --body       # plus a full-figure card per character

Each crop is 2.4 face widths wide and 3 face heights tall (above the hair to the chest), ordered left to right, and
upscaled so that its short edge is at least --min-edge (default 512; the server refuses references under 256 px).
With ``--body`` a second card per character follows the head cards: 4 face widths wide, from above the hat to about
eight face heights below the face (the whole figure in a medium shot: hair, hat, coat, scarf), clamped to the frame.
Prints the H3_REF_IMAGES value. Measured 2026-11-05 (results/20261105-multiref/): the crops raise every face
similarity measure by 0.1-0.2 while the character is visible; what they do after an occlusion is still being measured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def crop_box(box, width, height, wide=2.4, tall=3.0, above=0.45):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        raise ValueError(f"face box must have positive size, got {box}")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cw, ch = w * wide, h * tall
    left, top = max(0, int(cx - cw / 2)), max(0, int(cy - ch * above))
    right, bottom = min(width, int(cx + cw / 2)), min(height, int(cy + ch * (1 - above)))
    return left, top, right, bottom


def body_box(box, width, height):
    """The full-figure crop of a face box: 4 face widths wide, 0.9 face heights above the face centre to 8 below."""
    return crop_box(box, width, height, wide=4.0, tall=8.9, above=0.9 / 8.9)


def upscale(image: Image.Image, min_edge: int) -> Image.Image:
    short = min(image.size)
    if short >= min_edge:
        return image
    s = min_edge / short
    return image.resize((round(image.width * s), round(image.height * s)), Image.LANCZOS)


def detect(image: Image.Image):
    try:
        from facenet_pytorch import MTCNN
    except ImportError as e:  # pragma: no cover
        raise SystemExit("facenet_pytorch is not installed here; pass --box x0,y0,x1,y1 per face instead") from e
    boxes, probs = MTCNN(keep_all=True, device="cpu").detect(image)
    if boxes is None:
        return []
    return [tuple(float(v) for v in b) for b, p in zip(boxes.tolist(), probs.tolist()) if p is not None and p >= 0.9]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--box", action="append", default=[], help="x0,y0,x1,y1 of a face (repeat); skips detection")
    ap.add_argument("--min-edge", type=int, default=512)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--body", action="store_true", help="also write a full-figure card per character (after the head cards)")
    args = ap.parse_args(argv)
    image = Image.open(args.scene).convert("RGB")
    boxes = [tuple(float(v) for v in b.split(",")) for b in args.box] if args.box else detect(image)
    if not boxes:
        print("no face found", file=sys.stderr)
        return 1
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or Path(args.scene).stem
    paths = [str(Path(args.scene).resolve())]
    ordered = sorted(boxes, key=lambda b: b[0])
    kinds = [("face", crop_box)] + ([("body", body_box)] if args.body else [])
    for kind, fn in kinds:
        for i, box in enumerate(ordered):
            crop = upscale(image.crop(fn(box, image.width, image.height)), args.min_edge)
            path = out / f"{prefix}_{kind}{i}.png"
            crop.save(path)
            print(f"{path} box {[round(v) for v in box]} -> {crop.size[0]}x{crop.size[1]}", file=sys.stderr)
            paths.append(str(path.resolve()))
    print("H3_REF_IMAGES=" + ",".join(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
