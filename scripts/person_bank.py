#!/usr/bin/env python3
"""Whole-person bank for the identity guard: every character as a set of appearance embeddings (head, full figure)
cut from the reference cards, and a score of sampled frames against them.

  python scripts/person_bank.py build --out bank.npz --cards refs/scene_face0.png,refs/scene_body0.png \\
      --cards refs/scene_face1.png,refs/scene_body1.png
  python scripts/person_bank.py score --bank bank.npz [--min-sim 0.6] [--min-height 0.25] frame1.png frame2.png ...

``build`` takes one ``--cards`` list per character (any mix of head-and-shoulders and full-figure crops, see
``scripts/make_character_cards.py --body``) and stores their DINOv2 ViT-S/14 embeddings (self-supervised features:
hair, hat, clothes and face together, not the face alone). ``score`` finds the people in each frame (torchvision
Faster R-CNN MobileNet, COCO "person"), embeds each figure and its upper part, takes the best cosine similarity to any
card of any character, and prints one JSON line: persons (the principal ones: at least ``min-height`` of the frame
tall and at least ``principal-ratio`` of the tallest person in the frame, so extras further back do not count), known
(best >= min-sim), unknown, unknown_frames, unknown_frac (the fraction of FRAMES with an unknown principal person), mean_best,
min_best and per character its best similarity. With ``--save-persons`` the embeddings and similarities of the
principal persons are written for the bank growth (``identity_guard.grow_bank``).

Calibrated 2026-11-06 on the published 120 s shop clip at 960 px: the leads (0.85-0.95 of the frame tall) score
0.67-0.95 against their cards, extras further back (0.25-0.7 of the frame) 0.37-0.55, a second man in a similar dark
coat 0.61 on his upper part.
Runs on a CPU (about 0.5 s per frame); needs torch, torchvision and the DINOv2 hub checkpoint (the face_drift venv).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _models(detector: bool = True):
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True, verbose=False).eval()
    det = None
    if detector:
        from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                                  fasterrcnn_mobilenet_v3_large_320_fpn)

        det = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT).eval()
    return det, dino


def embed(dino, img: Image.Image) -> np.ndarray:
    """Unit DINOv2 class-token embedding of an image (resized to 224x224)."""
    x = torch.from_numpy(np.asarray(img.convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0)
    x = (x.permute(2, 0, 1) - _MEAN) / _STD
    with torch.no_grad():
        e = dino(x[None])[0]
    e = e / e.norm()
    return e.numpy().astype(np.float32)


def persons(det, img: Image.Image, min_prob: float = 0.7):
    """[(x0, y0, x1, y1, prob)] in pixels of the people in ``img``."""
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    with torch.no_grad():
        out = det([x])[0]
    found = []
    for b, label, p in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()):
        if int(label) == 1 and p >= min_prob:
            found.append((b[0], b[1], b[2], b[3], float(p)))
    return found


def person_crops(img: Image.Image, box):
    """(figure, upper part) crops of a person box: the whole box and its top 35 % (head, hat, shoulders)."""
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    figure = img.crop((x0, y0, x1, y1))
    upper = img.crop((x0, y0, x1, y0 + max(8, int((y1 - y0) * 0.35))))
    return figure, upper


def _load(path: str, width: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.width != width:
        img = img.resize((width, max(2, img.height * width // img.width)))
    return img


def build(args) -> int:
    _, dino = _models(detector=False)
    embs, src, who = [], [], []
    for c, cards in enumerate(args.cards):
        for path in cards.split(","):
            embs.append(embed(dino, Image.open(path)))
            src.append(os.path.basename(path))
            who.append(c)
    if not embs:
        print("no cards given", file=sys.stderr)
        return 2
    np.savez(args.out, emb=np.stack(embs), src=np.array(src), who=np.array(who, np.int16))
    print(json.dumps({"bank": args.out, "cards": len(embs), "characters": len(args.cards), "sources": src}))
    return 0


def score_frames(bank: np.ndarray, who: np.ndarray, found_per_frame, min_sim: float) -> dict:
    """Pure part of ``score``: ``found_per_frame`` = per frame a list of (x, height, figure embedding, upper embedding).

    A person is known when the better of the figure's and the upper part's best similarity to any card reaches
    ``min_sim``; ``unknown_frac`` is the fraction of FRAMES with an unknown principal person.
    """
    n_char = int(who.max()) + 1 if len(who) else 0
    best_all, per_char, unknown_frames = [], [-1.0] * n_char, 0
    for found in found_per_frame:
        frame_unknown = False
        for x, h, ef, eu in found:
            sims = np.maximum(bank @ ef, bank @ eu)
            best = float(sims.max())
            best_all.append(best)
            frame_unknown = frame_unknown or best < min_sim
            for c in range(n_char):
                m = sims[who == c]
                if len(m):
                    per_char[c] = max(per_char[c], float(m.max()))
        unknown_frames += int(frame_unknown)
    n = len(best_all)
    known = sum(1 for s in best_all if s >= min_sim)
    return {
        "frames": len(found_per_frame),
        "persons": n,
        "known": known,
        "unknown": n - known,
        "unknown_frames": unknown_frames,
        "unknown_frac": unknown_frames / len(found_per_frame) if found_per_frame else 0.0,
        "mean_best": float(np.mean(best_all)) if n else None,
        "min_best": float(min(best_all)) if n else None,
        "character_best": per_char,
    }


def principal(boxes, height: int, min_height: float, ratio: float):
    """The principal people of a frame: at least ``min_height`` of the frame tall and ``ratio`` of the tallest."""
    tallest = max(((b[3] - b[1]) / height for b in boxes), default=0.0)
    return [b for b in boxes if (b[3] - b[1]) / height >= max(min_height, ratio * tallest)]


def score(args) -> int:
    data = np.load(args.bank, allow_pickle=False)
    bank, who = data["emb"], data["who"]
    det, dino = _models()
    found_per_frame, saved = [], []
    for path in args.frames:
        img = _load(path, args.width)
        found = []
        for box in principal(persons(det, img), img.height, args.min_height, args.principal_ratio):
            h = (box[3] - box[1]) / img.height
            figure, upper = person_crops(img, box)
            ef, eu = embed(dino, figure), embed(dino, upper)
            found.append(((box[0] + box[2]) / 2 / img.width, h, ef, eu))
            saved.append((os.path.basename(path), found[-1][0], h, ef, eu))
        found_per_frame.append(found)
    out = score_frames(bank, who, found_per_frame, args.min_sim)
    if args.save_persons:
        best = [float(np.maximum(bank @ ef, bank @ eu).max()) for _n, _x, _h, ef, eu in saved]
        np.savez(args.save_persons,
                 emb=np.stack([ef for *_r, ef, _eu in saved]) if saved else np.zeros((0, bank.shape[1]), np.float32),
                 emb_upper=np.stack([eu for *_r, eu in saved]) if saved else np.zeros((0, bank.shape[1]), np.float32),
                 best=np.array(best, np.float32), height=np.array([h for _n, _x, h, *_r in saved], np.float32),
                 src=np.array([f"{n}@x={x:.2f}" for n, x, *_r in saved]))
        out["persons_file"] = args.save_persons
    print(json.dumps(out))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--cards", action="append", required=True, help="comma-separated card images of ONE character (repeat)")
    s = sub.add_parser("score")
    s.add_argument("--bank", required=True)
    s.add_argument("--min-sim", type=float, default=0.6)
    s.add_argument("--min-height", type=float, default=0.4, help="ignore people shorter than this fraction of the frame")
    s.add_argument("--principal-ratio", type=float, default=0.75,
                   help="ignore people shorter than this fraction of the tallest person in the frame (extras)")
    s.add_argument("--width", type=int, default=960)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--save-persons", default="")
    s.add_argument("frames", nargs="+")
    args = ap.parse_args()
    torch.set_num_threads(getattr(args, "threads", 8))
    return build(args) if args.cmd == "build" else score(args)


if __name__ == "__main__":
    sys.exit(main())
