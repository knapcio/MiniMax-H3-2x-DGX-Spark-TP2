#!/usr/bin/env python3
"""Sharpness of a long clip per continuation window: the variance of the Laplacian of the luma, whole frame and
optional regions (faces), averaged over each window's own frames.

A chain that degrades shows here before it shows in the latent statistics: on the 118 s chains of 2026-11-09 the frame
went from 186 to 155 and the woman's face from 311 to 196 over eleven windows while the latents kept their spread.
Needs ffmpeg and numpy; reads two frames per second.

  python scripts/window_detail.py clip.mp4 --box woman=520,80,700,240 --box man=220,60,400,220 --json detail.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def window_bounds(total_frames: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """[first new frame, end) of every window: window 1 owns its whole span, later ones what they add."""
    if not 0 <= overlap < window:
        raise ValueError("overlap must be smaller than the window")
    bounds, end = [(0, min(total_frames, window))], min(total_frames, window)
    while end < total_frames:
        start, end = end, min(total_frames, end + window - overlap)
        bounds.append((start, end))
    return bounds


def laplacian_variance(frames):  # noqa: ANN001
    """Per frame variance of the 4-neighbour Laplacian of a [n, h, w] float array."""
    lap = (4.0 * frames[:, 1:-1, 1:-1] - frames[:, :-2, 1:-1] - frames[:, 2:, 1:-1]
           - frames[:, 1:-1, :-2] - frames[:, 1:-1, 2:])
    return lap.var(axis=(1, 2))


def parse_box(text: str) -> tuple[str, tuple[int, int, int, int]]:
    name, _, coords = text.partition("=")
    values = tuple(int(v) for v in coords.split(","))
    if not name or len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise argparse.ArgumentTypeError(f"box must be name=x0,y0,x1,y1 with x1 > x0 and y1 > y0, got {text!r}")
    return name, values  # type: ignore[return-value]


def per_window(values, times, bounds, fps: float) -> list[float | None]:  # noqa: ANN001
    out = []
    for start, end in bounds:
        inside = [v for v, t in zip(values, times) if start / fps <= t < end / fps]
        out.append(round(sum(inside) / len(inside), 1) if inside else None)
    return out


def probe(path: str) -> tuple[int, int, int, float]:
    raw = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,nb_read_packets,r_frame_rate", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    stream = json.loads(raw)["streams"][0]
    num, _, den = stream["r_frame_rate"].partition("/")
    return int(stream["width"]), int(stream["height"]), int(stream["nb_read_packets"]), float(num) / float(den or 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip")
    parser.add_argument("--window-frames", type=int, default=277)
    parser.add_argument("--overlap-frames", type=int, default=22)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--box", action="append", type=parse_box, default=[], help="name=x0,y0,x1,y1 in pixels")
    parser.add_argument("--json", help="write the result here")
    args = parser.parse_args(argv)
    import numpy as np

    width, height, total, fps = probe(args.clip)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", args.clip, "-vf", f"fps={args.sample_fps},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width).astype(np.float32)
    times = [(i + 0.5) / args.sample_fps for i in range(len(frames))]
    bounds = window_bounds(total, args.window_frames, args.overlap_frames)
    result = {"clip": args.clip, "windows": len(bounds), "frame": per_window(laplacian_variance(frames), times, bounds, fps)}
    for name, (x0, y0, x1, y1) in args.box:
        if x1 > width or y1 > height:
            raise SystemExit(f"box {name} leaves the {width}x{height} frame")
        result[name] = per_window(laplacian_variance(frames[:, y0:y1, x0:x1]), times, bounds, fps)
    for key in ["frame", *[name for name, _ in args.box]]:
        print(f"{key}: " + " ".join("-" if v is None else f"{v:.0f}" for v in result[key]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
