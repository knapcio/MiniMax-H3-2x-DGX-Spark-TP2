#!/usr/bin/env python3
"""Face identity drift of a generated clip against its reference frame.

For every sampled frame the faces are detected (MTCNN) and embedded (InceptionResnetV1, VGGFace2 weights, facenet-pytorch);
each reference face is matched to a detected face (``--match position``: the one closest to its horizontal position,
the default; ``--match embed``: greedy best-similarity assignment, for clips where the characters change sides), and the
cosine similarity of the embeddings is reported per sample. 1.0 = the same embedding; same-person pairs in VGGFace2 are typically > 0.6-0.7,
different people < 0.4. A slow fall over the clip is identity drift; a step at a window hand-over is a seam that changed
the face. Also reports how often each face was not found (a turned or occluded face counts as missing, not as drift).

Usage:
  python scripts/face_drift.py clip.mp4 reference.png [--every 1.0] [--window 277] [--overlap 22] [--json out.json]
Needs ffmpeg on PATH and a venv with torch + facenet-pytorch (CPU is fine: ~1 s per sampled frame).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
from PIL import Image


def window_count(frames_total: int, window: int, overlap: int) -> int:
    """How many windows of `window` frames, stepping `window - overlap`, the renderer used for `frames_total` frames."""
    if window <= overlap or overlap < 0:
        raise ValueError("0 <= overlap < window is required")
    if frames_total <= window:
        return 1
    return -(-(frames_total - window) // (window - overlap)) + 1


def window_of(frame: int, window: int, overlap: int, count: int) -> int:
    """The 1-based window whose own frames hold `frame`: the overlap belongs to the earlier window, and the last window
    keeps its tail (the final `overlap` frames are not a window of their own)."""
    return max(1, min(frame // (window - overlap) + 1, count))


def frame_count(path: str) -> int:
    """Frames of the first video stream (the container's count, else counted by decoding)."""
    for extra in ([], ["-count_frames"]):
        key = "nb_read_frames" if extra else "nb_frames"
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", *extra, "-show_entries", f"stream={key}",
                              "-of", "csv=p=0", path], check=True, capture_output=True, text=True).stdout.strip()
        if out.isdigit():
            return int(out)
    raise RuntimeError(f"could not count the frames of {path}")


def frames(path: str, every: float, width: int):
    """Yield (t_seconds, PIL image) for one frame every `every` seconds, scaled to `width`."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps=1/{every},scale={width}:-2", "-f", "image2pipe",
           "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"]
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", path], check=True, capture_output=True, text=True).stdout.strip()
    w0, h0 = (int(x) for x in probe.split(","))
    h = (h0 * width // w0) // 2 * 2
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = 0
    size = width * h * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield n * every, Image.frombytes("RGB", (width, h), buf)
        n += 1
    proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("reference")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--width", type=int, default=960, help="analysis width (frames are scaled to it)")
    ap.add_argument("--window", type=int, default=277)
    ap.add_argument("--overlap", type=int, default=22)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--json", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--match", choices=("position", "embed"), default="position",
                    help="position: nearest horizontal position (people do not swap places); embed: greedy best similarity")
    args = ap.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    from facenet_pytorch import MTCNN, InceptionResnetV1

    det = MTCNN(keep_all=True, device="cpu", post_process=True)
    net = InceptionResnetV1(pretrained="vggface2").eval()

    def faces(img):
        boxes, probs = det.detect(img)
        if boxes is None:
            return []
        crops = det.extract(img, boxes, None)
        with torch.no_grad():
            emb = net(crops).numpy()
        out = []
        for b, p, e in zip(boxes, probs, emb):
            if p is None or p < 0.9:
                continue
            out.append({"x": float((b[0] + b[2]) / 2) / img.width, "w": float(b[2] - b[0]) / img.width,
                        "emb": e / np.linalg.norm(e)})
        return out

    ref_img = Image.open(args.reference).convert("RGB")
    ref_img = ref_img.resize((args.width, ref_img.height * args.width // ref_img.width))
    ref = sorted(faces(ref_img), key=lambda f: f["x"])
    if not ref:
        print("no face found in the reference", file=sys.stderr)
        return 2
    names = [f"face{i}@x={f['x']:.2f}" for i, f in enumerate(ref)]
    print(f"reference faces: {len(ref)} -> " + ", ".join(names))

    rows = []
    for t, img in frames(args.clip, args.every, args.width):
        found = faces(img)
        sims = [None] * len(ref)
        if found and args.match == "embed":
            # greedy one-to-one assignment by similarity: the best pair first, then the best among the rest
            pairs = sorted(((float(np.dot(f["emb"], rf["emb"])), i, j) for i, rf in enumerate(ref) for j, f in enumerate(found)),
                           reverse=True)
            used_ref, used_found = set(), set()
            for sim, i, j in pairs:
                if i in used_ref or j in used_found:
                    continue
                sims[i] = sim
                used_ref.add(i)
                used_found.add(j)
        elif found:
            for i, rf in enumerate(ref):
                # the candidate whose horizontal position is closest to the reference face (people do not swap places)
                cand = min(found, key=lambda f: abs(f["x"] - rf["x"]))
                if abs(cand["x"] - rf["x"]) <= 0.25:
                    sims[i] = float(np.dot(cand["emb"], rf["emb"]))
        rows.append({"t": round(t, 2), "sims": sims})

    nwin = window_count(frame_count(args.clip), args.window, args.overlap)
    for r in rows:
        r["window"] = window_of(int(round(r["t"] * args.fps)), args.window, args.overlap, nwin)
    print("t(s)  window  " + "  ".join(f"{n:>14}" for n in names))
    for r in rows:
        print(f"{r['t']:5.1f}  {r['window']:>6}  " + "  ".join(f"{s:14.3f}" if s is not None else f"{'missing':>14}" for s in r["sims"]))

    summary = {}
    print("\nper window: mean / min similarity (missing count) per face")
    for w in range(1, nwin + 1):
        sel = [r for r in rows if r["window"] == w]
        line = f"window {w:>2} ({len(sel):>3} samples): "
        summary[w] = []
        for i in range(len(ref)):
            vals = [r["sims"][i] for r in sel if r["sims"][i] is not None]
            miss = sum(1 for r in sel if r["sims"][i] is None)
            if vals:
                line += f"{names[i]} {np.mean(vals):.3f} / {np.min(vals):.3f} ({miss} missing)   "
                summary[w].append({"face": names[i], "mean": float(np.mean(vals)), "min": float(np.min(vals)), "missing": miss})
            else:
                line += f"{names[i]} all missing   "
                summary[w].append({"face": names[i], "mean": None, "min": None, "missing": miss})
        print(line)
    for i in range(len(ref)):
        vals = [r["sims"][i] for r in rows if r["sims"][i] is not None]
        if vals:
            first = np.mean(vals[: max(1, len(vals) // 10)])
            last = np.mean(vals[-max(1, len(vals) // 10):])
            print(f"{names[i]}: first 10 % {first:.3f} -> last 10 % {last:.3f} (drift {last - first:+.3f}); overall mean {np.mean(vals):.3f}, min {np.min(vals):.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"clip": args.clip, "reference": args.reference, "match": args.match, "faces": names, "rows": rows, "windows": summary}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
