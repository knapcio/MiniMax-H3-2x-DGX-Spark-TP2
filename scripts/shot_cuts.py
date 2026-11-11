#!/usr/bin/env python3
"""The model's own cuts and silences in a one-shot clip, and the head trim they imply.

A close-up rendered from a scene frame opens on the reference's composition and cuts to the close-up on its own
(queue144, 2026-11-11: a hard cut at 1.0-2.6 s in three of six shots, not a push-in). When the audio before that cut
is silence the head trims off without touching a word; when the line already started in the head, the head is a shot
of its own and stays. This script measures that per clip and prints the trim: the first scene cut under ``--max-head``
seconds whose leading audio is silence. ``--apply`` writes the trimmed clip (re-encoded; a short fade-in on the audio
keeps the room tone from clicking at the cut). Needs ffmpeg and ffprobe only.

  python scripts/shot_cuts.py shot2.mp4 shot4.mp4            # one line per clip: cuts, silences, trim
  python scripts/shot_cuts.py shot2.mp4 --apply out/shot2.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

FPS_DEFAULT = 24.0


def head_trim(cuts: list[float], silences: list[tuple[float, float]], max_head: float = 2.0,
              fps: float = FPS_DEFAULT) -> float:
    """Seconds to trim from the head: the first scene cut under ``max_head`` whose leading audio is silence
    (a silence that starts at 0 and ends at or after the cut); 0 otherwise. Snapped to a frame."""
    if not cuts:
        return 0.0
    cut = cuts[0]
    if cut >= max_head:
        return 0.0
    lead = next((end for start, end in silences if start < 0.05), 0.0)
    if lead < cut:
        return 0.0
    return round(cut * fps) / fps


def scene_cuts(path: str, threshold: float = 0.3) -> list[float]:
    """Times of the frames ffmpeg's scene detector scores above ``threshold`` (0.3 catches hard cuts, not motion)."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-i", path, "-vf", f"select='gt(scene,{threshold})',showinfo",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    return [float(t) for t in re.findall(r"pts_time:([0-9.]+)", out)]


def silences(path: str, noise_db: float = -32.0, min_s: float = 0.35) -> list[tuple[float, float]]:
    """(start, end) of every stretch quieter than ``noise_db`` for at least ``min_s`` seconds (end = clip end when the
    silence runs out the clip)."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-i", path, "-af", f"silencedetect=n={noise_db}dB:d={min_s}",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    starts = [float(t) for t in re.findall(r"silence_start: ([0-9.]+)", out)]
    ends = [float(t) for t in re.findall(r"silence_end: ([0-9.]+)", out)]
    if len(ends) < len(starts):
        ends.append(duration(path))
    return list(zip(starts, ends))


def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def frame_rate(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    num, _, den = json.loads(out)["streams"][0]["r_frame_rate"].partition("/")
    return float(num) / float(den or 1)


def measure(path: str, max_head: float, threshold: float, noise_db: float) -> dict:
    cuts = scene_cuts(path, threshold)
    quiet = silences(path, noise_db)
    fps = frame_rate(path)
    return {"clip": path, "duration_s": round(duration(path), 3), "cuts_s": [round(c, 3) for c in cuts],
            "silences_s": [[round(a, 2), round(b, 2)] for a, b in quiet],
            "trim_s": head_trim(cuts, quiet, max_head, fps)}


def apply(path: str, out: str, trim: float, fade_s: float = 0.04) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path, "-ss", f"{trim:.4f}", "-c:v", "libx264",
                    "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                    "-af", f"afade=t=in:st=0:d={fade_s}", out], check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--max-head", type=float, default=2.0, help="a cut later than this is not a head (default 2 s)")
    ap.add_argument("--scene", type=float, default=0.3, help="ffmpeg scene score of a cut (default 0.3)")
    ap.add_argument("--noise", type=float, default=-32.0, help="silence threshold in dBFS (default -32)")
    ap.add_argument("--apply", help="write the trimmed clip here (one input clip only)")
    ap.add_argument("--json", help="write the measurements here")
    args = ap.parse_args(argv)
    if args.apply and len(args.clips) != 1:
        ap.error("--apply takes one clip")
    rows = [measure(c, args.max_head, args.scene, args.noise) for c in args.clips]
    for r in rows:
        cuts = " ".join(f"{c:.2f}" for c in r["cuts_s"]) or "none"
        quiet = " ".join(f"{a:.2f}-{b:.2f}" for a, b in r["silences_s"]) or "none"
        print(f"{r['clip']}: {r['duration_s']:.2f} s, cuts {cuts}, silence {quiet}, trim {r['trim_s']:.3f} s")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    if args.apply:
        apply(args.clips[0], args.apply, rows[0]["trim_s"])
        print(f"wrote {args.apply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
