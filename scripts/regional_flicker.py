#!/usr/bin/env python3
"""Screen for repeated localized frame-change spikes; diagnostic only, not a visual verdict."""
import argparse
from fractions import Fraction
from pathlib import Path
import statistics
import subprocess
import sys

from PIL import Image, ImageChops, ImageStat


WIDTH, HEIGHT = 336, 192
BOXES = [(col * WIDTH // 3, row * HEIGHT // 2, (col + 1) * WIDTH // 3, (row + 1) * HEIGHT // 2)
         for row in range(2) for col in range(3)]
LABELS = ("upper-left", "upper-middle", "upper-right", "lower-left", "lower-middle", "lower-right")


def periodic_spikes(changes, *, period=17, threshold=0.7, ratio=3.0, minimum=3, radius=8):
    """Runs of same-phase spikes against their nearby changes; frame 0 has no predecessor."""
    if period < 1 or threshold <= 0 or ratio <= 1 or minimum < 2 or radius < 1:
        raise ValueError("invalid detector thresholds")
    matches = []
    for frame in range(1, len(changes)):
        lo, hi = max(1, frame - radius), min(len(changes), frame + radius + 1)
        neighbors = changes[lo:frame] + changes[frame + 1:hi]
        if len(neighbors) < radius:
            continue
        baseline = max(0.05, statistics.median(neighbors))
        value = changes[frame]
        if value >= threshold and value >= ratio * baseline:
            matches.append((frame, value, baseline))
    groups = []
    for phase in range(period):
        run = []
        for match in (item for item in matches if item[0] % period == phase):
            if run and match[0] != run[-1][0] + period:
                if len(run) >= minimum:
                    groups.append(run)
                run = []
            run.append(match)
        if len(run) >= minimum:
            groups.append(run)
    return sorted(groups, key=lambda run: run[0][0])


def inspect(path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=avg_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                           check=True, text=True, capture_output=True)
    fps = float(Fraction(probe.stdout.strip()))
    if fps <= 0:
        raise ValueError("invalid frame rate")
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-vf",
               f"scale={WIDTH}:{HEIGHT}:flags=area,format=gray", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    series = [[0.0] for _ in BOXES]
    prev = None
    frames = 0
    try:
        while True:
            raw = proc.stdout.read(WIDTH * HEIGHT)
            if not raw:
                break
            if len(raw) != WIDTH * HEIGHT:
                raise ValueError("incomplete decoded frame")
            current = Image.frombytes("L", (WIDTH, HEIGHT), raw)
            if prev is not None:
                diff = ImageChops.difference(prev, current)
                for i, box in enumerate(BOXES):
                    series[i].append(ImageStat.Stat(diff.crop(box)).mean[0])
            prev = current
            frames += 1
        if proc.wait() != 0 or frames < 2:
            raise ValueError("ffmpeg failed or video has fewer than two frames")
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    return fps, frames, series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--period", type=int, default=17)
    parser.add_argument("--threshold", type=float, default=0.7, help="minimum regional absolute change (0–255 gray)")
    parser.add_argument("--ratio", type=float, default=3.0, help="minimum factor over nearby changes")
    parser.add_argument("--minimum", type=int, default=3, help="minimum consecutive same-phase spikes")
    args = parser.parse_args()
    try:
        fps, frames, series = inspect(args.video)
        count = 0
        for label, changes in zip(LABELS, series, strict=True):
            for group in periodic_spikes(changes, period=args.period, threshold=args.threshold,
                                         ratio=args.ratio, minimum=args.minimum):
                first, last = group[0][0], group[-1][0]
                val = statistics.median(item[1] for item in group)
                ref = statistics.median(item[2] for item in group)
                print(f"{label}: frames {first}..{last} ({first / fps:.2f}..{last / fps:.2f} s), "
                      f"{len(group)} spikes every {args.period} frames, median change {val:.2f}, nearby {ref:.2f}")
                count += 1
        if not count:
            print(f"no qualifying repeated regional frame-change spikes ({frames} frames checked)")
        print("screening candidates only; real motion can also trigger this check, and other defects can be missed")
        return 0
    except (OSError, subprocess.SubprocessError, ValueError, ZeroDivisionError) as exc:
        print(f"regional-flicker: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
