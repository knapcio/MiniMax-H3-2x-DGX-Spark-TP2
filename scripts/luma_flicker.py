#!/usr/bin/env python3
"""Find repeated frame-level luma drops that one-sample-per-second metrics miss."""

import argparse
from fractions import Fraction
import re
import statistics
import subprocess
import sys


def periodic_drops(means, period=17, threshold=2.0, minimum=3):
    """Return runs of consecutive same-phase drops as (frames, signed changes)."""
    groups = []
    for phase in range(period):
        matches = [(i, means[i] - means[i - 1]) for i in range(1, len(means))
                   if i % period == phase and means[i] - means[i - 1] <= -threshold]
        run = []
        for match in matches:
            if run and match[0] != run[-1][0] + period:
                if len(run) >= minimum:
                    groups.append(run)
                run = []
            run.append(match)
        if len(run) >= minimum:
            groups.append(run)
    return sorted(groups, key=lambda run: run[0][0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Video to inspect; requires ffmpeg and ffprobe")
    parser.add_argument("--period", type=int, default=17, help="Expected period in frames")
    parser.add_argument("--threshold", type=float, default=2.0, help="Minimum YAVG drop in one frame")
    parser.add_argument("--minimum", type=int, default=3, help="Minimum consecutive periods")
    args = parser.parse_args()
    if args.period < 1 or args.threshold <= 0 or args.minimum < 2:
        parser.error("period and threshold must be positive, and minimum must be at least 2")

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", args.video],
            check=True, text=True, capture_output=True,
        )
        fps = float(Fraction(probe.stdout.strip()))
        if fps <= 0:
            raise ValueError("invalid frame rate")
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", args.video, "-map", "0:v:0", "-an", "-vf",
             "scale=320:-2:flags=area,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-", "-f", "null", "-"],
            check=True, text=True, capture_output=True,
        )
        means = [float(value) for value in re.findall(r"lavfi\.signalstats\.YAVG=([0-9.]+)", result.stdout)]
        if len(means) < 2:
            raise ValueError("fewer than two decoded video frames")
    except (OSError, subprocess.CalledProcessError, ValueError, ZeroDivisionError) as exc:
        print(f"luma-flicker: cannot inspect video: {exc}", file=sys.stderr)
        return 1

    groups = periodic_drops(means, args.period, args.threshold, args.minimum)
    if not groups:
        print(f"periodic luma drops: none >= {args.threshold:g} YAVG for {args.minimum} successive "
              f"{args.period}-frame periods ({len(means)} frames checked)")
    for group in groups:
        start, end = group[0][0], group[-1][0]
        values = [value for _, value in group]
        print(f"periodic luma drops: frames {start}..{end} ({start / fps:.2f}..{end / fps:.2f} s), "
              f"every {args.period} frames, {len(group)} drops; median {statistics.median(values):+.2f} YAVG, "
              f"range {min(values):+.2f}..{max(values):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
