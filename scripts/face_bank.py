#!/usr/bin/env python3
"""Face bank for the identity guard: embeddings of the faces in the reference images, and a score of sampled frames
against them.

  python scripts/face_bank.py build --out bank.npz scene.png crop0.png crop1.png
  python scripts/face_bank.py score --bank bank.npz --min-sim 0.5 [--min-face 0.04] [--save-faces faces.npz] frame1.png ...

``build`` detects every face (MTCNN, probability >= 0.9) in the given images and stores its L2-normalised
InceptionResnetV1 (VGGFace2) embedding. ``score`` detects the faces in the frames, takes for each the best cosine
similarity to the bank and prints one JSON line: faces (detected, at least ``min-face`` of the frame width wide),
known (best >= min-sim), unknown, unknown_frames and unknown_frac (the fraction of frames with an unknown principal
face), mean_best, min_best, and per bank face its best similarity over the frames. A principal face that matches
nobody in the bank over several frames is a new person in the shot, which is what the guard re-samples on.
Calibrated 2026-11-06 on the 11 s occlusion probes at 960x544: the leads are 6-7 % of the frame wide, double
detections 3-4 %, the woman after her re-entry matches her crop at 0.45-0.74, a blurred pass-by frame 0.01-0.11.
Runs on a CPU (about 1 s per frame); needs torch + facenet-pytorch (the face_drift.py venv).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


def _models():
    from facenet_pytorch import MTCNN, InceptionResnetV1

    return MTCNN(keep_all=True, device="cpu", post_process=True), InceptionResnetV1(pretrained="vggface2").eval()


def faces(det, net, img: Image.Image, min_prob: float = 0.9):
    """[(x_centre_frac, width_frac, unit embedding, detection probability)] of the faces in ``img``."""
    boxes, probs = det.detect(img)
    if boxes is None:
        return []
    crops = det.extract(img, boxes, None)
    with torch.no_grad():
        emb = net(crops).numpy()
    out = []
    for b, p, e in zip(boxes, probs, emb):
        if p is None or p < min_prob:
            continue
        out.append((float((b[0] + b[2]) / 2) / img.width, float(b[2] - b[0]) / img.width, e / np.linalg.norm(e), float(p)))
    return out


def _load(path: str, width: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.width != width:
        img = img.resize((width, max(2, img.height * width // img.width)))
    return img


def build(args) -> int:
    det, net = _models()
    embs, src = [], []
    for path in args.images:
        found = faces(det, net, _load(path, args.width))
        for x, w, e, _p in found:
            embs.append(e)
            src.append(f"{path}@x={x:.2f}")
    if not embs:
        print("no face found in the reference images", file=sys.stderr)
        return 2
    np.savez(args.out, emb=np.stack(embs).astype(np.float32), src=np.array(src))
    print(json.dumps({"bank": args.out, "faces": len(embs), "sources": src}))
    return 0


def principal(found, min_face: float):
    """The principal faces of one frame, as (x, width, embedding, probability): at least ``min_face`` of the frame
    width (the leads, not extras further back or double detections); a detection without a probability counts as 1."""
    out = []
    for x, w, e, *rest in found:
        if w >= min_face:
            out.append((x, w, e, float(rest[0]) if rest else 1.0))
    return out


def score_frames(bank: np.ndarray, found_per_frame, min_sim: float, min_face: float, unknown_min_prob: float = 0.98) -> dict:
    """Pure part of ``score``: ``found_per_frame`` = per frame a list of (x, width, embedding[, probability]).

    Only principal faces count (``principal``: the leads, not extras or double detections).
    A face is unknown when its best similarity to the bank is below ``min_sim`` (different people sit below 0.4 in
    VGGFace2 space, the same person in another pose at 0.45-0.9) AND the detector was sure of it (probability at least
    ``unknown_min_prob``): a doubtful detection that matches nobody is most likely not a face (a brass scale on the
    counter of the 120 s shop scene was detected at 0.90-0.94 with similarity -0.1..0.07; its faces sit at 0.99+).
    ``unknown_frac`` is the fraction of FRAMES with an unknown principal face: one blurred pass-by frame does not trip
    the guard, a stranger who stays does.
    """
    best_all, per_bank, unknown_frames = [], [-1.0] * len(bank), 0
    for found in found_per_frame:
        frame_unknown = False
        for x, w, e, p in principal(found, min_face):
            sims = bank @ e
            best = float(sims.max())
            best_all.append(best)
            frame_unknown = frame_unknown or (best < min_sim and p >= unknown_min_prob)
            for i, s in enumerate(sims):
                per_bank[i] = max(per_bank[i], float(s))
        unknown_frames += int(frame_unknown)
    n = len(best_all)
    known = sum(1 for s in best_all if s >= min_sim)
    return {
        "frames": len(found_per_frame),
        "faces": n,
        "known": known,
        "unknown": n - known,
        "unknown_frames": unknown_frames,
        "unknown_frac": unknown_frames / len(found_per_frame) if found_per_frame else 0.0,
        "mean_best": float(np.mean(best_all)) if n else None,
        "min_best": float(min(best_all)) if n else None,
        "bank_best": per_bank,
    }


def principal_faces(bank: np.ndarray, found_per_frame, min_face: float, names=None):
    """Pure: the principal faces of the frames as (embeddings, best similarity to the bank, source label, width,
    probability)."""
    embs, best, src, widths, probs = [], [], [], [], []
    for i, found in enumerate(found_per_frame):
        name = names[i] if names is not None else str(i)
        for x, w, e, p in principal(found, min_face):
            embs.append(np.asarray(e, dtype=np.float32))
            best.append(float((bank @ e).max()))
            src.append(f"{name}@x={x:.2f}")
            widths.append(float(w))
            probs.append(float(p))
    return embs, best, src, widths, probs


def score(args) -> int:
    data = np.load(args.bank, allow_pickle=False)
    bank = data["emb"]
    det, net = _models()
    found_per_frame = [faces(det, net, _load(p, args.width)) for p in args.frames]
    out = score_frames(bank, found_per_frame, args.min_sim, args.min_face, args.unknown_min_prob)
    if args.save_faces:
        # the principal faces with their best similarity, for the guard's bank growth (identity_guard.grow_bank)
        embs, best, src, widths, probs = principal_faces(bank, found_per_frame, args.min_face,
                                                         [os.path.basename(p) for p in args.frames])
        np.savez(args.save_faces, emb=np.stack(embs) if embs else np.zeros((0, bank.shape[1]), np.float32),
                 best=np.array(best, np.float32), src=np.array(src), width=np.array(widths, np.float32),
                 prob=np.array(probs, np.float32))
        out["faces_file"] = args.save_faces
    print(json.dumps(out))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--width", type=int, default=960)
    b.add_argument("images", nargs="+")
    s = sub.add_parser("score")
    s.add_argument("--bank", required=True)
    s.add_argument("--min-sim", type=float, default=0.4)
    s.add_argument("--min-face", type=float, default=0.05, help="ignore faces narrower than this fraction of the frame")
    s.add_argument("--unknown-min-prob", type=float, default=0.98,
                   help="a face counts as unknown only when the detector's probability reaches this")
    s.add_argument("--width", type=int, default=960)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--save-faces", default="", help="write the principal faces' embeddings and best similarities to this .npz")
    s.add_argument("frames", nargs="+")
    args = ap.parse_args()
    torch.set_num_threads(getattr(args, "threads", 8))
    return build(args) if args.cmd == "build" else score(args)


if __name__ == "__main__":
    sys.exit(main())
