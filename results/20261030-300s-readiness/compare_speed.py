#!/usr/bin/env python3
"""Compare matched 16 s window-size or step-count controls; never promote a setting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import tempfile
import time


TAG = "d16_1344x768_s20_seed1101_fl2va"
TAG10 = "d16_1344x768_s10_seed1101_fl2va"
BASE = "quality16-base"
SMALL = "quality16-win175"
TEN = "quality16-steps10"
DETECTOR_SHA = "15f6267530a8e6051a2f5d27400798adc99d0fe8fe1f115cfd9319d2934c5a63"


def periodic_report(root, video):
    detector = root / "luma-flicker-20261030.py"
    if detector.is_symlink() or hashlib.sha256(detector.read_bytes()).hexdigest() != DETECTOR_SHA:
        raise ValueError("standalone flicker detector changed")
    proc = subprocess.run([sys.executable, str(detector), str(video), "--minimum", "2"],
                          check=True, capture_output=True, text=True, timeout=3600)
    lines = proc.stdout.splitlines()
    if not lines or any(not line.startswith("periodic luma drops:") for line in lines):
        raise ValueError("standalone flicker detector returned no usable report")
    return lines


def read_run(root, name, tag):
    directory = root / name
    if (tag + " ok") not in (directory / "STATUS").read_text().splitlines():
        raise ValueError(name + " did not pass the render verification")
    video = directory / (tag + ".mp4")
    if video.stat().st_size <= 0:
        raise ValueError(name + " has an empty video")
    timing = json.loads((directory / (tag + ".mp4.timing.json")).read_text())
    if str(timing.get("http")) != "200" or timing.get("curl_rc") != 0 or float(timing.get("elapsed_s", 0)) <= 0:
        raise ValueError(name + " has invalid HTTP, curl or wall time")
    memory = {}
    for node in ("head", "worker"):
        rows = (directory / (tag + f".mem-{node}.csv")).read_text().splitlines()
        values = [int(cols[1]) for row in rows if len(cols := row.split(",")) >= 2 and cols[1].isdigit()]
        if len(values) < 2:
            raise ValueError(name + " lacks " + node + " memory samples")
        memory[node] = {"samples": len(values), "min_available_mib": min(values)}
    metrics = (directory / "metrics.txt").read_text().splitlines()
    if not any(line.startswith("cuts (") for line in metrics):
        raise ValueError(name + " lacks quality-screen output")
    flicker = periodic_report(root, video)
    phase_text = (directory / "phases.txt").read_text()
    phases = [float(v) for v in re.findall(r"phase diffuse ([0-9.]+) s", phase_text)]
    return {"timing": timing, "memory": memory, "diffuse_seconds": phases,
            "quality_signals": [line for line in metrics if line.startswith(("seam frame ", "cuts ("))] + flicker}


def compare(root, kind="windows"):
    if kind not in ("windows", "steps"):
        raise ValueError("unknown comparison kind")
    candidate, tag = (SMALL, TAG) if kind == "windows" else (TEN, TAG10)
    a, b = read_run(root, BASE, TAG), read_run(root, candidate, tag)
    ta, tb = a["timing"], b["timing"]
    for key in ("duration", "effective_duration", "size", "seed", "task", "image", "frame_indices", "prompt", "audio_reference"):
        if ta.get(key) != tb.get(key):
            raise ValueError("controls differ in " + key)
    extra_a, extra_b = dict(ta["extra_params"]), dict(tb["extra_params"])
    if kind == "windows":
        if ta.get("steps") != 20 or tb.get("steps") != 20 or extra_b.pop("continuation_window_frames", None) != 175 or extra_a != extra_b:
            raise ValueError("controls differ beyond the 175-frame window")
        labels = ("277", "175")
        description = "16s FL2VA 1344x768, 20 steps, seed 1101: window 175 vs 277"
    else:
        if ta.get("steps") != 20 or tb.get("steps") != 10 or extra_a != extra_b:
            raise ValueError("controls differ beyond the 10-step setting")
        labels = ("20", "10")
        description = "16s FL2VA 1344x768, seed 1101: 10 steps vs 20"
    aw, bw = float(ta["elapsed_s"]), float(tb["elapsed_s"])
    if not a["diffuse_seconds"] or not b["diffuse_seconds"]:
        raise ValueError("missing per-window phase times")
    return {
        "comparison": description,
        "wall_seconds": {labels[0]: aw, labels[1]: bw},
        "wall_change_percent": round((bw / aw - 1) * 100, 2),
        "diffuse_seconds_per_window": {labels[0]: a["diffuse_seconds"], labels[1]: b["diffuse_seconds"]},
        "memory": {labels[0]: a["memory"], labels[1]: b["memory"]},
        "quality_signals": {labels[0]: a["quality_signals"], labels[1]: b["quality_signals"]},
        "caveat": "Single same-boot comparison; the changed setting changes the samples. Review visual quality and repeat before promotion.",
    }


def ready(root, name):
    lines = (root / "gen.log").read_text().splitlines()
    return any(line.startswith("H3-AUTO-OK " + name + " ") for line in lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--kind", choices=("windows", "steps"), default="windows")
    parser.add_argument("--wait", action="store_true", help="wait up to 48 hours for both verified cases")
    args = parser.parse_args()
    deadline = time.monotonic() + (48 * 3600 if args.wait else 0)
    while True:
        try:
            candidate = SMALL if args.kind == "windows" else TEN
            if ready(args.root, BASE) and ready(args.root, candidate):
                result = compare(args.root, args.kind)
                filename = "quality-speed-comparison-20261030.json" if args.kind == "windows" else "quality-steps-comparison-20261030.json"
                output = args.root / filename
                fd, name = tempfile.mkstemp(prefix=".quality-speed-", dir=args.root)
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(result, f, indent=2, sort_keys=True)
                        f.write("\n")
                        f.flush()
                        os.fsync(f.fileno())
                    os.link(name, output)  # atomic, never overwrites an earlier report
                finally:
                    Path(name).unlink(missing_ok=True)
                print("verified comparison saved:", output, flush=True)
                return 0
            if "H3-AUTO-HOLD quality-20261030 " in (args.root / "gen.log").read_text():
                print("quality queue held before both cases completed", flush=True)
                return 1
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            print("comparison unavailable:", exc, flush=True)
            return 1
        if time.monotonic() >= deadline:
            print("verified controls not yet available", flush=True)
            return 1
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
