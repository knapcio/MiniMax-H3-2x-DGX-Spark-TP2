#!/usr/bin/env python3
"""Compare first-window time in matched 16/60 s 960x576 controls; no promotion."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


BASE = "quality16-960-stream"
LONG = "quality60-960-stream"
TAG16 = "d16_960x576_s20_seed1101_fl2va"
TAG60 = "d60_960x576_s20_seed1101_fl2va"
HELPER_SHA = "3a3eea8a595a8db58f6220550c7fa704a1fa081b18dbf79ab782f32255e28000"


def load_helper(root):
    path = root / "compare-quality-speed-20261030-v2.py"
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError("speed helper changed")
    spec = importlib.util.spec_from_file_location("h3_speed_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(root, helper):
    short, long = helper.read_run(root, BASE, TAG16), helper.read_run(root, LONG, TAG60)
    a, b = short["timing"], long["timing"]
    for key in ("size", "steps", "seed", "task", "image", "frame_indices", "prompt", "audio_reference"):
        if a.get(key) != b.get(key):
            raise ValueError("controls differ in " + key)
    if (a.get("duration"), b.get("duration")) != (16, 60):
        raise ValueError("control durations are not 16 and 60 seconds")
    if (a.get("effective_duration"), b.get("effective_duration")) != (16, 60):
        raise ValueError("effective durations are not 16 and 60 seconds")
    if (a.get("size"), a.get("steps"), a.get("seed"), a.get("task")) != ("960x576", 20, 1101, "fl2va"):
        raise ValueError("unexpected control shape")
    ea, eb = dict(a["extra_params"]), dict(b["extra_params"])
    if ea.pop("duration", None) != 16 or eb.pop("duration", None) != 60 or ea != eb:
        raise ValueError("controls differ beyond their duration")
    if (ea.get("preencode_mp4") is not True or ea.get("long_video") is not True
            or ea.get("long_video_mode") != "continuation" or ea.get("continuation_window_frames", 277) != 277
            or ea.get("continuation_overlap_frames", 22) != 22):
        raise ValueError("controls did not use continuation streaming")
    if not short["diffuse_seconds"] or not long["diffuse_seconds"]:
        raise ValueError("missing first-window phase")
    first_short, first_long = short["diffuse_seconds"][0], long["diffuse_seconds"][0]
    if first_short <= 0 or first_long <= 0:
        raise ValueError("invalid first-window phase")
    return {
        "comparison": "16s vs 60s, same 960x576 FL2VA prompt, 20 steps, seed 1101 and streaming",
        "first_window_seconds": {"16s": first_short, "60s": first_long},
        "first_window_change_percent": round((first_long / first_short - 1) * 100, 2),
        "wall_seconds": {"16s": float(a["elapsed_s"]), "60s": float(b["elapsed_s"])},
        "diffuse_seconds_per_window": {"16s": short["diffuse_seconds"], "60s": long["diffuse_seconds"]},
        "memory": {"16s": short["memory"], "60s": long["memory"]},
        "quality_signals": {"16s": short["quality_signals"], "60s": long["quality_signals"]},
        "caveat": ("One same-boot pair; first windows have 277 frames but request length, cache/runtime state and "
                   "later windows differ. The result cannot alone establish a cause or visual quality."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + (48 * 3600 if args.wait else 0)
    while True:
        try:
            lines = (args.root / "gen.log").read_text().splitlines()
            if any(x.startswith("H3-AUTO-HOLD quality-20261030 ") for x in lines):
                print("quality queue held before both controls", flush=True)
                return 1
            if all(any(x.startswith("H3-AUTO-OK " + name + " ") for x in lines) for name in (BASE, LONG)):
                report = compare(args.root, load_helper(args.root))
                target = args.root / "quality-length-comparison-20261030.json"
                fd, name = tempfile.mkstemp(prefix=".quality-length-", dir=args.root)
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(report, f, indent=2, sort_keys=True)
                        f.write("\n"); f.flush(); os.fsync(f.fileno())
                    os.link(name, target)
                finally:
                    Path(name).unlink(missing_ok=True)
                print("verified length comparison saved:", target, flush=True)
                return 0
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            print("comparison unavailable:", exc, flush=True)
            return 1
        if time.monotonic() >= deadline:
            print("verified controls not yet available", flush=True)
            return 1
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())

