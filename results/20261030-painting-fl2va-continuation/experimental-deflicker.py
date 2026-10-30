#!/usr/bin/env python3
"""Experimental, lossless-encode luma repair of detected 17-frame drops; never overwrites input."""

import argparse
import importlib.util
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DETECTOR = ROOT / "scripts/luma_flicker.py"
spec = importlib.util.spec_from_file_location("luma_flicker", DETECTOR)
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)


def measurements(path):
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-vf",
                           "scale=320:-2:flags=area,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
                           "-f", "null", "-"], text=True, capture_output=True, check=True)
    return [float(v) for v in re.findall(r"lavfi\.signalstats\.YAVG=([0-9.]+)", proc.stdout)]


def make_adjustments(means):
    groups = detector.periodic_drops(means)
    adjustments = {}
    for group in groups:
        for frame, _ in group:
            if frame < 1 or frame + 5 >= len(means):
                raise ValueError("not enough neighboring frames to repair the detected drop")
            left, right = means[frame - 1], means[frame + 5]
            for offset in range(5):
                expected = left + (right - left) * (offset + 1) / 6
                change = max(0.0, min(8.0, expected - means[frame + offset]))
                if change >= 0.5:
                    adjustments[frame + offset] = change
    return groups, adjustments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or args.output.exists():
        parser.error("output must be a new file distinct from the input")
    means = measurements(args.source)
    groups, corrections = make_adjustments(means)
    if not groups or not corrections:
        raise SystemExit("no qualifying repeated drops; no file created")
    expression = "0"
    for frame, change in sorted(corrections.items(), reverse=True):
        # hue's brightness maps b*25.5 to an integer luma offset; the 0.1
        # avoids rounding down a desired integer through floating-point error.
        expression = f"if(eq(n,{frame}),{(int(change + 0.5) + 0.1) / 25.5:.8f},{expression})"
    command = ["ffmpeg", "-v", "error", "-n", "-i", str(args.source), "-map", "0:v:0", "-map", "0:a:0?",
               "-map_metadata", "0", "-vf", f"hue=b='{expression}'", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "0", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
               str(args.output)]
    subprocess.run(command, check=True)
    print(f"{len(groups)} repeated-drop run(s), {len(corrections)} frames adjusted; maximum +{max(corrections.values()):.2f} YAVG")
    print("experimental luma-only alteration; original preserved; audio stream copied; video encoded losslessly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
