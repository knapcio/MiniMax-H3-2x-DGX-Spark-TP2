#!/usr/bin/env python3
"""Summarize two completed same-boot 16 s window-size controls; never promotes a setting."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time


TAG = "d16_1344x768_s20_seed1101_fl2va"
BASE = "quality16-base"
SMALL = "quality16-win175"


def read_run(root, name):
    directory = root / name
    if (TAG + " ok") not in (directory / "STATUS").read_text().splitlines():
        raise ValueError(name + " did not pass the render verification")
    if (directory / (TAG + ".mp4")).stat().st_size <= 0:
        raise ValueError(name + " has an empty video")
    timing = json.loads((directory / (TAG + ".mp4.timing.json")).read_text())
    if str(timing.get("http")) != "200" or timing.get("curl_rc") != 0 or float(timing.get("elapsed_s", 0)) <= 0:
        raise ValueError(name + " has invalid HTTP, curl or wall time")
    memory = {}
    for node in ("head", "worker"):
        rows = (directory / (TAG + f".mem-{node}.csv")).read_text().splitlines()
        values = [int(cols[1]) for row in rows if len(cols := row.split(",")) >= 2 and cols[1].isdigit()]
        if len(values) < 2:
            raise ValueError(name + " lacks " + node + " memory samples")
        memory[node] = {"samples": len(values), "min_available_mib": min(values)}
    metrics = (directory / "metrics.txt").read_text().splitlines()
    if not any(line.startswith("cuts (") for line in metrics) or not any(line.startswith("periodic luma drops:") for line in metrics):
        raise ValueError(name + " lacks quality-screen output")
    phase_text = (directory / "phases.txt").read_text()
    phases = [float(v) for v in re.findall(r"phase diffuse ([0-9.]+) s", phase_text)]
    return {"timing": timing, "memory": memory, "diffuse_seconds": phases,
            "quality_signals": [line for line in metrics if line.startswith(("seam frame ", "cuts (", "periodic luma drops:"))]}


def compare(root):
    a, b = read_run(root, BASE), read_run(root, SMALL)
    ta, tb = a["timing"], b["timing"]
    for key in ("duration", "effective_duration", "size", "steps", "seed", "task", "image", "frame_indices", "prompt", "audio_reference"):
        if ta.get(key) != tb.get(key):
            raise ValueError("controls differ in " + key)
    extra_a, extra_b = dict(ta["extra_params"]), dict(tb["extra_params"])
    if extra_b.pop("continuation_window_frames", None) != 175 or extra_a != extra_b:
        raise ValueError("controls differ beyond the 175-frame window")
    aw, bw = float(ta["elapsed_s"]), float(tb["elapsed_s"])
    if not a["diffuse_seconds"] or not b["diffuse_seconds"]:
        raise ValueError("missing per-window phase times")
    return {
        "comparison": "16s FL2VA 1344x768, 20 steps, seed 1101: window 175 vs 277",
        "wall_seconds": {"277": aw, "175": bw},
        "wall_change_percent": round((bw / aw - 1) * 100, 2),
        "diffuse_seconds_per_window": {"277": a["diffuse_seconds"], "175": b["diffuse_seconds"]},
        "memory": {"277": a["memory"], "175": b["memory"]},
        "quality_signals": {"277": a["quality_signals"], "175": b["quality_signals"]},
        "caveat": "Single same-boot comparison; different window sizes change the samples. Review visual quality and repeat before promotion.",
    }


def ready(root, name):
    lines = (root / "gen.log").read_text().splitlines()
    return any(line.startswith("H3-AUTO-OK " + name + " ") for line in lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--wait", action="store_true", help="wait up to 48 hours for both verified cases")
    args = parser.parse_args()
    deadline = time.monotonic() + (48 * 3600 if args.wait else 0)
    while True:
        try:
            if ready(args.root, BASE) and ready(args.root, SMALL):
                result = compare(args.root)
                output = args.root / "quality-speed-comparison-20261030.json"
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
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print("comparison unavailable:", exc, flush=True)
            return 1
        if time.monotonic() >= deadline:
            print("verified controls not yet available", flush=True)
            return 1
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
