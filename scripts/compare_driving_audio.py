#!/usr/bin/env python3
"""Compare a driving audio track with sparse portions of rendered audio.

The measurement finds the best waveform offset in each sample. It screens for
missing, shifted or altered source audio; it does not assess speech quality,
speaker identity, or whether a character's mouth is synchronized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


def track_timing(path: Path) -> dict:
    """Describe audio start relative to the first video, or container, timeline."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time:format=start_time",
         "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=60,
    )
    data = json.loads(probe.stdout)
    streams = data.get("streams", [])

    def seconds(value):
        try:
            n = float(value)
            return n if math.isfinite(n) else None
        except (TypeError, ValueError):
            return None

    audio = next((seconds(x.get("start_time")) for x in streams if x.get("codec_type") == "audio"), None)
    video = next((seconds(x.get("start_time")) for x in streams if x.get("codec_type") == "video"), None)
    container = seconds(data.get("format", {}).get("start_time"))
    origin = video if video is not None else container
    assumption = None
    if audio is None and origin is None and not any(x.get("codec_type") == "video" for x in streams):
        # Ordinary WAV files often report neither start time; their first
        # decoded sample defines the source audio's zero in this comparison.
        audio = origin = 0.0
        assumption = "audio-only input has no start timestamps; decoded start treated as zero"
    return {"audio_start_s": audio, "timeline_origin_s": origin,
            "audio_relative_to_timeline_s": None if audio is None or origin is None else round(audio - origin, 6),
            "assumption": assumption}


def decode(path: Path, sample_rate: int, cap_seconds: float) -> np.ndarray:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(path), "-map", "0:a:0",
         "-vn", "-ac", "1", "-ar", str(sample_rate), "-t", str(cap_seconds),
         "-f", "f32le", "pipe:1"],
        check=True, capture_output=True, timeout=1800,
    )
    if not result.stdout or len(result.stdout) % 4:
        raise ValueError("missing or incomplete decoded audio")
    data = np.frombuffer(result.stdout, dtype="<f4")
    if not np.isfinite(data).all():
        raise ValueError("decoded audio contains nonfinite samples")
    return data


def compare_segment(reference: np.ndarray, output: np.ndarray, start: int, length: int,
                    max_shift: int) -> dict:
    """Find signed normalized correlation for one reference interval.

    Positive shift means that the source is later in the output. Both inputs
    are already mono at the same sample rate. Correlations are calculated for
    the full reference interval, so small overlaps cannot appear as good hits.
    """
    if start < 0 or length < 2 or max_shift < 0 or start + length > min(len(reference), len(output)):
        raise ValueError("invalid segment bounds")
    x = reference[start:start + length].astype(np.float64)
    x_centered = x - x.mean()
    x_rms = float(np.sqrt(np.mean(x_centered * x_centered)))
    nominal = output[start:start + length].astype(np.float64)
    nominal -= nominal.mean()
    result = {"reference_rms": x_rms, "output_nominal_rms": float(np.sqrt(np.mean(nominal * nominal)))}
    if x_rms < 1e-4:
        result.update({"status": "low_reference_energy", "shift_samples": None, "correlation": None,
                       "nominal_correlation": None, "fitted_gain": None})
        return result

    left = max(0, start - max_shift)
    right = min(len(output), start + length + max_shift)
    y = output[left:right].astype(np.float64)
    count = len(y) - length + 1
    result["search_shift_samples"] = [left - start, left + count - 1 - start]
    nfft = 1 << (len(y) + length - 2).bit_length()
    products = np.fft.irfft(np.fft.rfft(y, nfft) * np.fft.rfft(x_centered[::-1], nfft), nfft)
    dots = products[length - 1:length - 1 + count]
    sums = np.concatenate(([0.0], np.cumsum(y)))
    squares = np.concatenate(([0.0], np.cumsum(y * y)))
    sums = sums[length:] - sums[:-length]
    squares = squares[length:] - squares[:-length]
    energy = np.maximum(0.0, squares - sums * sums / length)
    x_energy = float(np.dot(x_centered, x_centered))
    valid = energy > length * 1e-8
    if not np.any(valid):
        result.update({"status": "low_output_energy", "shift_samples": None, "correlation": None,
                       "nominal_correlation": None, "fitted_gain": None})
        return result
    scores = np.full(count, -np.inf)
    scores[valid] = dots[valid] / np.sqrt(x_energy * energy[valid])
    best = int(np.argmax(scores))
    nominal_index = start - left
    result.update({"status": "measured", "shift_samples": left + best - start,
                   "correlation": float(np.clip(scores[best], -1, 1)),
                   "nominal_correlation": float(np.clip(scores[nominal_index], -1, 1)) if valid[nominal_index] else None,
                   "fitted_gain": float(dots[best] / x_energy)})
    return result


def compare(reference: np.ndarray, output: np.ndarray, sample_rate: int, *, window_seconds: float = 3,
            max_shift_seconds: float = 1, samples: int = 3) -> list[dict]:
    if (sample_rate < 1 or not math.isfinite(window_seconds) or window_seconds <= 0 or
            not math.isfinite(max_shift_seconds) or max_shift_seconds < 0 or samples < 1):
        raise ValueError("invalid sampling parameters")
    common = min(len(reference), len(output))
    length = min(common, int(round(window_seconds * sample_rate)))
    if length < sample_rate // 2:
        raise ValueError("at least half a second of common audio is required")
    shift = int(round(max_shift_seconds * sample_rate))
    # Leave room on both sides when possible, so a delayed final window is
    # measured rather than falsely reported as unmatched at the file edge.
    margin = min(shift, max(0, (common - length) // 2))
    starts = sorted(set(int(x) for x in np.linspace(margin, common - length - margin, samples)))
    rows = []
    for start in starts:
        row = compare_segment(reference, output, start, length, shift)
        row["start_seconds"] = round(start / sample_rate, 4)
        row["duration_seconds"] = round(length / sample_rate, 4)
        row["shift_seconds"] = None if row["shift_samples"] is None else round(row["shift_samples"] / sample_rate, 4)
        for key in ("reference_rms", "output_nominal_rms", "correlation", "nominal_correlation", "fitted_gain"):
            if row[key] is not None:
                row[key] = round(row[key], 5)
        rows.append(row)
    return rows


def sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path, help="rendered video or audio")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--window-seconds", type=float, default=3)
    parser.add_argument("--max-shift-seconds", type=float, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--cap-seconds", type=float, default=600, help="decode at most this many seconds per input")
    parser.add_argument("--report", type=Path, help="write JSON to a new file instead of stdout")
    args = parser.parse_args()
    try:
        if args.sample_rate < 1000 or not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0:
            raise ValueError("invalid sample rate or duration cap")
        reference = decode(args.reference, args.sample_rate, args.cap_seconds)
        output = decode(args.output, args.sample_rate, args.cap_seconds)
        reference_timing, output_timing = track_timing(args.reference), track_timing(args.output)
        rows = compare(reference, output, args.sample_rate, window_seconds=args.window_seconds,
                       max_shift_seconds=args.max_shift_seconds, samples=args.samples)
        reference_start = reference_timing["audio_relative_to_timeline_s"]
        output_start = output_timing["audio_relative_to_timeline_s"]
        for row in rows:
            shift = row["shift_seconds"]
            row["presentation_shift_seconds"] = (None if shift is None or reference_start is None or output_start is None
                                                  else round(shift + output_start - reference_start, 4))
        result = {"reference": str(args.reference), "reference_sha256": sha(args.reference),
                  "output": str(args.output), "output_sha256": sha(args.output), "sample_rate": args.sample_rate,
                  "reference_decoded_seconds": round(len(reference) / args.sample_rate, 4),
                  "output_decoded_seconds": round(len(output) / args.sample_rate, 4),
                  "decode_cap_seconds": args.cap_seconds,
                  "reference_timing": reference_timing, "output_timing": output_timing, "segments": rows,
                  "limitations": ("Sparse mono waveform comparisons, not a listening or lip-sync verdict. "
                                  "Quiet source segments have no reliable offset; correlated music, "
                                  "mixed sound or reordered material can confuse the search. "
                                  "Presentation shifts rely on container timestamps and may be unavailable. "
                                  "Material beyond the decode cap is not checked.")}
        text = json.dumps(result, indent=2) + "\n"
        if args.report:
            with args.report.open("x") as f:
                f.write(text)
        else:
            print(text, end="")
        return 0
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"compare-driving-audio: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
