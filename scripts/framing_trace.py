#!/usr/bin/env python3
"""The framing of a shot over time: the largest face's size per sampled frame, the camera's zoom between samples,
where the opening settles (the establishing head), and the shot size the model ended on.

A Ref2VA shot opens on its reference's composition and reaches the requested framing by a cut or a push-in; the
scene-cut rule of ``scripts/shot_cuts.py`` sees the cut and not the push-in. This trace sees both: ``scale`` is the
largest face's box height over the frame height (MTCNN, facenet-pytorch), ``zoom`` the per-step scale of a
similarity transform between consecutive samples (ORB + RANSAC, OpenCV; a failed match is a cut). The head ends where
the final framing starts: the first sample at or after the last cut whose face scale is within 20 % of the last
half's median and after which the camera zooms by less than 10 % in total (None for a push-in to the last frame).
The final shot size (wide / medium / close-up by the face scale) says whether a requested close-up
was ignored.

  python scripts/framing_trace.py shot2.mp4 shot4.mp4 --json trace.json
Needs ffmpeg and a venv with torch + facenet-pytorch; OpenCV is optional (no zoom without it).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys

SIZES = ((0.35, "close-up"), (0.22, "medium"), (0.0, "wide"))   # face height / frame height; v3-v5 shots of 2026-11-10/11


def shot_size(scale: float | None) -> str:
    if scale is None:
        return "no face"
    return next(name for floor, name in SIZES if scale >= floor)


def head_end(times: list[float], scales: list[float | None], zooms: list[float | None], cuts: list[float] = (),
             tolerance: float = 0.20, zoom_tolerance: float = 0.10) -> float | None:
    """When the final framing starts: the first sample at or after the last cut whose face scale is within
    ``tolerance`` of the last half's median and after which the camera zooms by less than ``zoom_tolerance`` in
    total. 0 for a static shot; the cut time for a cut; the end of the move for a push-in; None when the shot has no
    face or never arrives (a push-in to the last frame)."""
    if not times:
        return None
    half = [s for s in scales[len(scales) // 2:] if s is not None]
    if not half:
        return None
    median = sorted(half)[len(half) // 2]
    logs = [0.0 if z is None else math.log(z) for z in zooms] + [0.0] * (len(times) - len(zooms))
    remaining = [0.0] * (len(times) + 1)
    for i in range(len(times) - 1, -1, -1):
        remaining[i] = remaining[i + 1] + logs[i]
    last_cut = max(cuts) if cuts else None
    for i, t in enumerate(times):
        s = scales[i]
        if s is None or (last_cut is not None and t < last_cut):
            continue
        if abs(s / median - 1.0) <= tolerance and abs(remaining[i + 1]) <= math.log1p(zoom_tolerance):
            return None if i == len(times) - 1 and len(times) > 1 else t   # arriving on the last sample is not arriving
    return None


def frames(path: str, every: float, width: int):
    """Yield (t_seconds, PIL image) for one frame every ``every`` seconds, scaled to ``width``."""
    from PIL import Image

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", path], check=True, capture_output=True, text=True).stdout.strip()
    w0, h0 = (int(x) for x in probe.split(","))
    h = (h0 * width // w0) // 2 * 2
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps=1/{every},scale={width}:{h}", "-f", "image2pipe",
           "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n, size = 0, width * h * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield n * every, Image.frombytes("RGB", (width, h), buf)
        n += 1
    proc.wait()


def zoom_between(prev, cur) -> tuple[float | None, int]:  # noqa: ANN001
    """(scale of the similarity transform from ``prev`` to ``cur``, inlier count); (None, 0) without OpenCV or a match."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None, 0
    orb = cv2.ORB_create(1500)
    g0, g1 = (cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY) for im in (prev, cur))
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 12 or len(k1) < 12:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
    if len(matches) < 12:
        return None, 0
    p0 = np.float32([k0[m.queryIdx].pt for m in matches])
    p1 = np.float32([k1[m.trainIdx].pt for m in matches])
    m, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    n_in = int(inliers.sum()) if inliers is not None else 0
    if m is None or n_in < 12:
        return None, n_in
    return float(math.hypot(m[0, 0], m[0, 1])), n_in


def trace(path: str, every: float, width: int, detector) -> dict:  # noqa: ANN001
    times, scales, xs, counts, zooms, inliers = [], [], [], [], [], []
    prev = None
    for t, img in frames(path, every, width):
        boxes, probs = detector.detect(img)
        kept = [b for b, p in zip(boxes if boxes is not None else [], probs if probs is not None else []) if p is not None and p >= 0.9]
        if kept:
            b = max(kept, key=lambda b: b[3] - b[1])
            scales.append(float(b[3] - b[1]) / img.height)
            xs.append(float((b[0] + b[2]) / 2) / img.width)
        else:
            scales.append(None)
            xs.append(None)
        counts.append(len(kept))
        z, n_in = zoom_between(prev, img) if prev is not None else (None, 0)
        zooms.append(z)
        inliers.append(n_in)
        times.append(t)
        prev = img
    cuts = [times[i] for i in range(1, len(times)) if zooms[i] is None and inliers[i] < 12 and counts[i] + counts[i - 1] > 0]
    end = head_end(times, scales, zooms, cuts)
    last_half = [s for s in scales[len(scales) // 2:] if s is not None]
    final = sorted(last_half)[len(last_half) // 2] if last_half else None
    total_zoom = math.exp(sum(math.log(z) for z in zooms if z is not None)) if any(z is not None for z in zooms) else None
    return {"clip": path, "every_s": every, "times_s": times, "scale": [None if s is None else round(s, 3) for s in scales],
            "x": [None if x is None else round(x, 3) for x in xs], "faces": counts,
            "zoom": [None if z is None else round(z, 4) for z in zooms], "inliers": inliers,
            "cuts_s": cuts, "head_end_s": end, "final_scale": None if final is None else round(final, 3),
            "final_size": shot_size(final), "opening_scale": next((round(s, 3) for s in scales if s is not None), None),
            "total_zoom": None if total_zoom is None else round(total_zoom, 3)}


def describe(r: dict) -> str:
    sc = [s for s in r["scale"] if s is not None]
    return (f"{r['clip']}: faces {max(r['faces']) if r['faces'] else 0}, scale {r['opening_scale']} -> {r['final_scale']} ({r['final_size']}), "
            f"head ends {r['head_end_s']} s, zoom x{r['total_zoom']}, cuts {' '.join(f'{c:.2f}' for c in r['cuts_s']) or 'none'}, "
            f"{len(sc)}/{len(r['scale'])} samples with a face")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--every", type=float, default=0.25, help="seconds between samples (default 0.25)")
    ap.add_argument("--width", type=int, default=640, help="analysis width (default 640)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    from facenet_pytorch import MTCNN

    det = MTCNN(keep_all=True, device="cpu", post_process=True)
    rows = []
    for clip in args.clips:
        r = trace(clip, args.every, args.width, det)
        rows.append(r)
        print(describe(r))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
