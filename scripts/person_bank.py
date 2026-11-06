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
(best >= min-sim), unknown, unjudged, unknown_frames, unknown_frac (the fraction of FRAMES with an unknown principal
person), mean_best, min_best and per character its best similarity. With ``--save-persons`` the embeddings and
similarities of the principal persons are written for the bank growth (``identity_guard.grow_bank``).

A person is judged only when a confident face (MTCNN probability >= ``face-prob``, default 0.98) shows in its upper
part: a figure seen from behind or half hidden scores 0.4-0.6 against frontal cards whoever it is (the park probe of
2026-11-06: the lead's back 0.57, a stranger in a similar coat on the 120 s clip 0.57), so without a face there is
no evidence either way and the person counts as ``unjudged``. With a face, a low figure similarity means the same
face in other hair or clothes, which is what the check is for; the face itself is judged by ``face_bank.py``.

Hair, hat and clothes are judged a second way, because the embeddings do not separate "the same woman in another
coat" from the woman herself (the park seam test of 2026-11-06: an open orange-brown coat, blonder hair, scored
0.86-0.92 against the red-coat card): a **colour signature** of three regions placed relative to the face box
(hair: beside and below the face; torso: 0.8-3.2 face heights below it; legs: 3.2-6) as (hue, saturation, value) of
each region, taken from the cards at build time and from every judged person. The colour distance to the matched
character (``colour_distance``: hue differences weighted by chroma, saturation differences weighted by brightness,
summed over the regions) above ``--max-colour`` (default 15) makes the person ``changed``, which counts as unknown
(the same person: 0-12 on the park and shop clips, the man after his turn up to 14; the other coat and blonder hair
after the park seam 15-27), and it never joins the bank (``best`` saved as -1).

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


def _face_detector():
    from facenet_pytorch import MTCNN

    return MTCNN(keep_all=True, device="cpu")


def face_prob(mtcnn, img: Image.Image) -> float:
    """The best face-detection probability in ``img`` (0 when nothing is found)."""
    _boxes, probs = mtcnn.detect(img)
    if probs is None:
        return 0.0
    return max((float(p) for p in probs if p is not None), default=0.0)


def face_box(mtcnn, img: Image.Image):
    """(probability, (x0, y0, x1, y1)) of the most confident face in ``img``, (0.0, None) when there is none."""
    boxes, probs = mtcnn.detect(img)
    if boxes is None or probs is None:
        return 0.0, None
    best = max(((float(p), b) for p, b in zip(probs, boxes) if p is not None and b is not None), default=(0.0, None),
               key=lambda t: t[0])
    return best[0], (None if best[1] is None else tuple(float(v) for v in best[1]))


REGIONS = ("hair", "torso", "legs")


def colour_signature(img: Image.Image, face, limit=None) -> np.ndarray:
    """(3 regions x [hue degrees, saturation, value]) of the hair, torso and legs regions placed relative to the face
    box ``face`` (x0, y0, x1, y1); NaN rows for regions outside the image or ``limit`` (the person box). The hue is
    the saturation-weighted circular mean, so a dull region does not pull it."""
    x0, y0, x1, y1 = face
    fw, fh = x1 - x0, y1 - y0
    boxes = ((x0 - 0.6 * fw, y1 - 0.2 * fh, x1 + 0.6 * fw, y1 + 0.6 * fh),
             (x0 - 1.2 * fw, y1 + 0.8 * fh, x1 + 1.2 * fw, y1 + 3.2 * fh),
             (x0 - 1.0 * fw, y1 + 3.2 * fh, x1 + 1.0 * fw, y1 + 6.0 * fh))
    lx0, ly0, lx1, ly1 = limit if limit is not None else (0, 0, img.width, img.height)
    lx0, ly0, lx1, ly1 = max(0, lx0), max(0, ly0), min(img.width, lx1), min(img.height, ly1)
    out = np.full((3, 3), np.nan, np.float32)
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    for i, (bx0, by0, bx1, by1) in enumerate(boxes):
        ax0, ay0, ax1, ay1 = int(max(bx0, lx0)), int(max(by0, ly0)), int(min(bx1, lx1)), int(min(by1, ly1))
        if ax1 - ax0 < 4 or ay1 - ay0 < 4 or (ay1 - ay0) < 0.5 * (by1 - by0):
            continue  # less than half of the region is in view: no evidence
        r = a[ay0:ay1, ax0:ax1].reshape(-1, 3)
        mx, mn = r.max(1), r.min(1)
        d = mx - mn
        sat = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0.0)
        h = np.zeros(len(r), np.float32)
        nz = d > 1e-6
        rr, gg, bb = r[nz, 0], r[nz, 1], r[nz, 2]
        dd = d[nz]
        hue = np.where(mx[nz] == rr, (gg - bb) / dd, np.where(mx[nz] == gg, 2.0 + (bb - rr) / dd, 4.0 + (rr - gg) / dd))
        h[nz] = (hue * 60.0) % 360.0
        ang = np.radians(h)
        if sat.sum() > 0:
            mean_hue = np.degrees(np.arctan2((sat * np.sin(ang)).sum(), (sat * np.cos(ang)).sum())) % 360.0
        else:
            mean_hue = 0.0
        out[i] = (mean_hue, float(sat.mean()), float(mx.mean()))
    return out


def colour_distance(a: np.ndarray, cards) -> float:
    """Colour distance of a person's signature ``a`` to a character's ``cards`` (a list of signatures): per region the
    smallest distance to any card that has the region, summed over the regions; NaN when no region is shared.

    A region's distance is the circular hue difference in degrees, weighted by the chroma (saturation x value) of the
    duller side (no weight below 0.10, full from 0.25: the hue of a dark coat or of near-black hair is noise), plus
    30 x the saturation difference weighted by the darker side's value (none below 0.10, full from 0.30: the
    saturation of a near-black region is noise too). Calibrated 2026-11-06 on the park clips and the 120 s shop clip:
    the same person 0-12 (the shop clip's dark clothes 0-5 over 120 s, the park's red coat 3-9, the man after his
    turn up to 14), the other coat and blonder hair after the seam 15-27.
    """
    total, shared = 0.0, 0
    for r, ra in enumerate(a):
        if np.isnan(ra[0]):
            continue
        best = None
        for c in cards:
            rb = c[r]
            if np.isnan(rb[0]):
                continue
            chroma = min(float(ra[1] * ra[2]), float(rb[1] * rb[2]))
            w_hue = min(1.0, max(0.0, (chroma - 0.10) / 0.15))
            w_sat = min(1.0, max(0.0, (min(float(ra[2]), float(rb[2])) - 0.10) / 0.20))
            dh = abs(float(ra[0]) - float(rb[0])) % 360.0
            d = w_hue * min(dh, 360.0 - dh) + w_sat * 30.0 * abs(float(ra[1]) - float(rb[1]))
            best = d if best is None else min(best, d)
        if best is not None:
            total += best
            shared += 1
    return total if shared else float("nan")


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
    mtcnn = _face_detector()
    embs, src, who, cols = [], [], [], []
    for c, cards in enumerate(args.cards):
        for path in cards.split(","):
            img = Image.open(path).convert("RGB")
            embs.append(embed(dino, img))
            src.append(os.path.basename(path))
            who.append(c)
            prob, box = face_box(mtcnn, img)
            cols.append(colour_signature(img, box) if box is not None and prob >= 0.9 else np.full((3, 3), np.nan, np.float32))
    if not embs:
        print("no cards given", file=sys.stderr)
        return 2
    np.savez(args.out, emb=np.stack(embs), src=np.array(src), who=np.array(who, np.int16), col=np.stack(cols))
    print(json.dumps({"bank": args.out, "cards": len(embs), "characters": len(args.cards), "sources": src,
                      "colour_regions": [int(np.sum(~np.isnan(c[:, 0]))) for c in cols]}))
    return 0


def score_frames(bank: np.ndarray, who: np.ndarray, found_per_frame, min_sim: float, face_prob_min: float = 0.98,
                 bank_col=None, max_colour: float = 15.0, changed_out: list | None = None) -> dict:
    """Pure part of ``score``: ``found_per_frame`` = per frame a list of (x, height, figure embedding, upper embedding
    [, face probability [, colour signature]]).

    A person is known when the better of the figure's and the upper part's best similarity to any card reaches
    ``min_sim``, unknown when it does not AND a confident face shows in its upper part (``face_prob_min``), unjudged
    otherwise (seen from behind, half hidden). A known person with a colour signature is compared with the matched
    character's cards (``bank_col``): a colour distance above ``max_colour`` makes it ``changed`` (other hair or
    clothes), which counts as unknown. ``unknown_frac`` is the fraction of FRAMES with an unknown or changed principal
    person. ``changed_out`` (a list) receives one bool per person in order, for the saved persons file.
    """
    n_char = int(who.max()) + 1 if len(who) else 0
    best_all, per_char, unknown_frames, unknown, unjudged, changed = [], [-1.0] * n_char, 0, 0, 0, 0
    colour_all = []
    for found in found_per_frame:
        frame_unknown = False
        for x, h, ef, eu, *rest in found:
            fp = float(rest[0]) if rest else 1.0
            col = rest[1] if len(rest) > 1 else None
            sims = np.maximum(bank @ ef, bank @ eu)
            best = float(sims.max())
            best_all.append(best)
            is_changed = False
            if best < min_sim:
                if fp >= face_prob_min:
                    unknown += 1
                    frame_unknown = True
                else:
                    unjudged += 1
            elif col is not None and bank_col is not None and fp >= face_prob_min:
                c = int(who[int(np.argmax(sims))])
                dist = colour_distance(col, [bank_col[j] for j in range(len(bank_col)) if int(who[j]) == c])
                if not np.isnan(dist):
                    colour_all.append(dist)
                    if dist > max_colour:
                        is_changed = True
                        changed += 1
                        frame_unknown = True
            if changed_out is not None:
                changed_out.append(is_changed)
            for c in range(n_char):
                m = sims[who == c]
                if len(m):
                    per_char[c] = max(per_char[c], float(m.max()))
        unknown_frames += int(frame_unknown)
    n = len(best_all)
    return {
        "frames": len(found_per_frame),
        "persons": n,
        "known": n - unknown - unjudged - changed,
        "unknown": unknown,
        "unjudged": unjudged,
        "changed": changed,
        "unknown_frames": unknown_frames,
        "unknown_frac": unknown_frames / len(found_per_frame) if found_per_frame else 0.0,
        "mean_best": float(np.mean(best_all)) if n else None,
        "min_best": float(min(best_all)) if n else None,
        "character_best": per_char,
        "colour_max": float(max(colour_all)) if colour_all else None,
        "colour_mean": float(np.mean(colour_all)) if colour_all else None,
    }


def principal(boxes, height: int, min_height: float, ratio: float):
    """The principal people of a frame: at least ``min_height`` of the frame tall and ``ratio`` of the tallest."""
    tallest = max(((b[3] - b[1]) / height for b in boxes), default=0.0)
    return [b for b in boxes if (b[3] - b[1]) / height >= max(min_height, ratio * tallest)]


def score(args) -> int:
    data = np.load(args.bank, allow_pickle=False)
    bank, who = data["emb"], data["who"]
    bank_col = data["col"] if "col" in data.files and args.max_colour > 0 else None
    det, dino = _models()
    mtcnn = _face_detector() if args.face_prob > 0 or bank_col is not None else None
    found_per_frame, saved = [], []
    for path in args.frames:
        img = _load(path, args.width)
        found = []
        for box in principal(persons(det, img), img.height, args.min_height, args.principal_ratio):
            h = (box[3] - box[1]) / img.height
            figure, upper = person_crops(img, box)
            ef, eu = embed(dino, figure), embed(dino, upper)
            fp, fbox, col = 1.0, None, None
            if mtcnn is not None:
                fp, fbox = face_box(mtcnn, upper)
                if fbox is not None and bank_col is not None:
                    x0, y0 = int(box[0]), int(box[1])
                    col = colour_signature(img, (fbox[0] + x0, fbox[1] + y0, fbox[2] + x0, fbox[3] + y0),
                                           limit=(x0, y0, int(box[2]), int(box[3])))
            found.append(((box[0] + box[2]) / 2 / img.width, h, ef, eu, fp, col))
            saved.append((os.path.basename(path), found[-1][0], h, ef, eu, fp, col))
        found_per_frame.append(found)
    changed_flags: list = []
    out = score_frames(bank, who, found_per_frame, args.min_sim, args.face_prob, bank_col, args.max_colour, changed_flags)
    if args.save_persons:
        best = [-1.0 if ch else float(np.maximum(bank @ ef, bank @ eu).max())
                for (_n, _x, _h, ef, eu, _fp, _c), ch in zip(saved, changed_flags)]  # a changed look never joins the bank
        nan3 = np.full((3, 3), np.nan, np.float32)
        np.savez(args.save_persons,
                 emb=np.stack([ef for _n, _x, _h, ef, *_r in saved]) if saved else np.zeros((0, bank.shape[1]), np.float32),
                 emb_upper=np.stack([eu for _n, _x, _h, _ef, eu, *_r in saved]) if saved else np.zeros((0, bank.shape[1]), np.float32),
                 best=np.array(best, np.float32), height=np.array([h for _n, _x, h, *_r in saved], np.float32),
                 face_prob=np.array([fp for *_r, fp, _c in saved], np.float32),
                 col=np.stack([nan3 if c is None else c for *_r, c in saved]) if saved else np.zeros((0, 3, 3), np.float32),
                 changed=np.array(changed_flags, bool),
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
    s.add_argument("--face-prob", type=float, default=0.98,
                   help="judge a person only when a face this confident shows in its upper part (0 = judge everyone)")
    s.add_argument("--max-colour", type=float, default=15.0,
                   help="colour distance (hair/torso/legs hue + saturation vs the matched character's cards) above "
                        "which a known person counts as changed; 0 disables the colour check")
    s.add_argument("frames", nargs="+")
    args = ap.parse_args()
    torch.set_num_threads(getattr(args, "threads", 8))
    return build(args) if args.cmd == "build" else score(args)


if __name__ == "__main__":
    sys.exit(main())
