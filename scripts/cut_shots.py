#!/usr/bin/env python3
"""Cut rendered shots into one film: the head trim of each shot (``scripts/shot_cuts.py``'s rule, or a value given
by hand), one encoding for all of them, a short audio fade-in at every cut so the room tone does not click, then a
stream-copy concat. The way the comedy clip's v3-v5 were cut (``results/20261110-shots-and-close-ups/``).

  python scripts/cut_shots.py shot1.mp4 shot2.mp4 shot3.mp4 --out film.mp4
  python scripts/cut_shots.py shot*.mp4 --out film.mp4 --trim 5=2.333 --trim 7=1.0   # hand trims by shot number (1-based)
  python scripts/cut_shots.py shot*.mp4 --out film.mp4 --no-head-trim                  # shots as rendered

Needs ffmpeg and ffprobe. The shots should share the frame size; ``--size WxH`` scales them all when they do not.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("shot_cuts", Path(__file__).resolve().with_name("shot_cuts.py"))
shot_cuts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(shot_cuts)  # type: ignore[union-attr]


def parse_trims(items: list[str], count: int) -> dict[int, float]:
    """``N=seconds`` per item, N 1-based and within the shot list."""
    out: dict[int, float] = {}
    for item in items or []:
        number, _, seconds = item.partition("=")
        if not number.isdigit() or not seconds:
            raise argparse.ArgumentTypeError(f"--trim takes N=seconds, got {item!r}")
        n = int(number)
        if not 1 <= n <= count:
            raise argparse.ArgumentTypeError(f"--trim {item!r}: there are {count} shots")
        out[n] = float(seconds)
    return out


def plan(clips: list[str], hand: dict[int, float], head_trim: bool, max_head: float) -> list[tuple[str, float]]:
    """(clip, seconds to trim from its head) per shot: the hand value when given, else the measured rule, else 0."""
    out = []
    for i, clip in enumerate(clips, 1):
        if i in hand:
            trim = hand[i]
        elif head_trim:
            trim = shot_cuts.head_trim(shot_cuts.scene_cuts(clip), shot_cuts.silences(clip), max_head,
                                       shot_cuts.frame_rate(clip))
        else:
            trim = 0.0
        out.append((clip, trim))
    return out


def encode(clip: str, out: str, trim: float, fps: float, size: str | None, crf: int, fade_s: float) -> None:
    vf = [f"fps={fps:g}"]
    if size:
        vf.append(f"scale={size.replace('x', ':')}:flags=lanczos")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", clip, "-ss", f"{trim:.4f}",
                    "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", "-af", f"afade=t=in:st=0:d={fade_s}", out], check=True)


def concat(parts: list[str], out: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
        listing = f.name
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
                        "-c", "copy", out], check=True)
    finally:
        os.unlink(listing)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="the shots in film order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trim", action="append", default=[], help="N=seconds: trim this much from shot N's head")
    ap.add_argument("--no-head-trim", action="store_true", help="no measured head trims (hand --trim values still apply)")
    ap.add_argument("--max-head", type=float, default=2.0, help="a cut later than this is not a head (default 2 s)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--size", help="WxH to scale every shot to (default: as rendered)")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--fade", type=float, default=0.04, help="audio fade-in at each cut in seconds (default 0.04)")
    ap.add_argument("--keep-parts", help="directory to keep the re-encoded shots in (default: a temporary one)")
    args = ap.parse_args(argv)
    try:
        hand = parse_trims(args.trim, len(args.clips))
    except argparse.ArgumentTypeError as e:
        ap.error(str(e))
    rows = plan(args.clips, hand, not args.no_head_trim, args.max_head)
    parts_dir = args.keep_parts or tempfile.mkdtemp(prefix="cut_shots_")
    os.makedirs(parts_dir, exist_ok=True)
    parts = []
    for i, (clip, trim) in enumerate(rows, 1):
        part = os.path.join(parts_dir, f"shot{i:02d}.mp4")
        encode(clip, part, trim, args.fps, args.size, args.crf, args.fade)
        parts.append(part)
        print(f"shot {i}: {clip} trim {trim:.3f} s")
    concat(parts, args.out)
    print(f"wrote {args.out}: {shot_cuts.duration(args.out):.2f} s from {len(parts)} shots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
