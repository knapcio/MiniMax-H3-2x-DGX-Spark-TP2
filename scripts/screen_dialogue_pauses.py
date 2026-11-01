#!/usr/bin/env python3
"""Measure near-silent intervals in a clip; this is not a judgment of pacing.

The audio is decoded with its timestamps reset. A quiet interval is one where
all channels stay below the chosen peak threshold for at least the chosen
duration. Background sound can mask a conversational pause, and quiet passages
may be intentional; listening is still needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


def sha(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def screen(path: Path, *, threshold_db: float = -45, minimum_s: float = 0.8) -> dict:
    if not math.isfinite(threshold_db) or not -100 <= threshold_db < 0:
        raise ValueError('threshold must be between -100 and 0 dBFS')
    if not math.isfinite(minimum_s) or not 0.05 <= minimum_s <= 60:
        raise ValueError('minimum duration must be between 0.05 and 60 s')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                            'stream=duration:format=duration', '-of', 'json', str(path)],
                           check=True, capture_output=True, text=True, timeout=60)
    metadata = json.loads(probe.stdout)
    if len(metadata['streams']) != 1:
        raise ValueError('expected an audio stream')
    duration_value = metadata['streams'][0].get('duration') or metadata['format'].get('duration')
    duration = float(duration_value)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid audio duration')
    command = ['ffmpeg', '-hide_banner', '-nostats', '-nostdin', '-i', str(path), '-map', '0:a:0', '-vn',
               '-af', f'asetpts=PTS-STARTPTS,silencedetect=noise={threshold_db}dB:d={minimum_s}', '-f', 'null', '-']
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    events = re.findall(r'silence_(start|end):\s*([-+0-9.eE]+)', result.stderr)
    intervals = []
    start = None
    for kind, value in events:
        timestamp = float(value)
        if not math.isfinite(timestamp):
            raise ValueError('invalid silence timestamp')
        if kind == 'start':
            if start is not None:
                raise ValueError('overlapping silence intervals')
            start = max(0.0, timestamp)
        else:
            if start is None or timestamp < start:
                raise ValueError('unmatched silence interval')
            end = min(duration, timestamp)
            if end < start:
                raise ValueError('silence interval exceeds audio duration')
            intervals.append({'start_s': round(start, 3), 'end_s': round(end, 3),
                              'duration_s': round(end - start, 3)})
            start = None
    if start is not None:
        raise ValueError('unfinished silence interval')
    total = sum(row['duration_s'] for row in intervals)
    return {'source_sha256': sha(path), 'audio_duration_s': round(duration, 3),
            'threshold_dbfs': threshold_db, 'minimum_s': minimum_s,
            'quiet_intervals': intervals, 'quiet_total_s': round(total, 3),
            'quiet_fraction': round(min(1.0, total / duration), 4),
            'longest_quiet_s': max((row['duration_s'] for row in intervals), default=0.0),
            'limitations': ('Quiet audio is not automatically poor pacing. Background sound can hide pauses; '
                            'listen to judge dialogue and ambience.')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clip', type=Path)
    parser.add_argument('--threshold-db', type=float, default=-45)
    parser.add_argument('--minimum-s', type=float, default=.8)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        data = screen(args.clip, threshold_db=args.threshold_db, minimum_s=args.minimum_s)
        with args.report.open('x') as out:
            json.dump(data, out, ensure_ascii=False, indent=2)
            out.write('\n')
        print(f"quiet: {data['quiet_total_s']:.2f}/{data['audio_duration_s']:.2f} s; "
              f"longest interval {data['longest_quiet_s']:.2f} s")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f'screen-dialogue-pauses: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
