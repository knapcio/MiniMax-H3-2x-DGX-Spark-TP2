#!/usr/bin/env python3
"""Split timed speech intervals for time-resolved source/output diagnostics.

The result is input for screen_speech_levels.py. This only plans intervals;
it does not process audio or judge whether a level change is audible.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

MIN_TAIL = Decimal('0.01')
MAX_SLICES = 1000


def seconds(value: object, label: str) -> Decimal:
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError(f'{label} must be a finite JSON number')
    return Decimal(str(value))


def plan(intervals: object, slice_seconds: float = 0.5) -> dict:
    step = seconds(slice_seconds, 'slice size')
    if not Decimal('0.1') <= step <= Decimal('60'):
        raise ValueError('slice size must be between 0.1 and 60 seconds')
    if not isinstance(intervals, list) or not 1 <= len(intervals) <= 1000:
        raise ValueError('intervals must contain 1..1000 objects')
    rows = []
    for parent, item in enumerate(intervals, 1):
        if not isinstance(item, dict):
            raise ValueError('each interval must be an object')
        start = seconds(item.get('start'), 'interval start')
        end = seconds(item.get('end'), 'interval end')
        if not Decimal('0') <= start < end <= Decimal('3600') or end - start < MIN_TAIL:
            raise ValueError('interval must lie within 0..3600 seconds and last at least 0.01 seconds')
        speaker = item.get('speaker')
        if speaker is not None and (not isinstance(speaker, str) or len(speaker) > 256):
            raise ValueError('speaker must be a string of at most 256 characters')
        position = start
        index = 0
        while position < end:
            stop = min(position + step, end)
            # Avoid tiny final bins that can round to zero decoded samples.
            if Decimal('0') < end - stop < MIN_TAIL:
                stop = end
            if len(rows) >= MAX_SLICES:
                raise ValueError('slice plan exceeds 1000 intervals')
            index += 1
            rows.append({'start': float(position), 'end': float(stop), 'speaker': speaker,
                         'source_interval_index': parent, 'slice_index': index})
            position = stop
    return {'schema': 'h3-dialogue-slices-v1', 'slice_seconds': float(step),
            'source_interval_count': len(intervals), 'dialogue': rows,
            'limitations': ('Slices preserve the scheduled intervals and may include silence; a low-energy '
                            'source slice is not a quality pass. The screen compares RMS, not perceived '
                            'loudness. Audio timing differences can change short-slice ratios; inspect '
                            'waveform alignment separately. Splitting is not a listening or lip-sync test.')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervals', type=Path, required=True, help='JSON list or object with dialogue list')
    parser.add_argument('--slice-seconds', type=float, default=0.5)
    parser.add_argument('--output', type=Path, required=True, help='new JSON file, usable by the level screen')
    args = parser.parse_args()
    created = False
    try:
        # A FIFO must not block while being identified as an invalid input.
        fd = os.open(args.intervals, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as f:
            before = os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('interval input must be a regular file')
            raw = f.read(1024 * 1024 + 1)
            after = os.fstat(f.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError('interval input changed while reading')
        if len(raw) > 1024 * 1024:
            raise ValueError('interval input exceeds 1 MiB')
        data = json.loads(raw)
        intervals = data.get('dialogue') if isinstance(data, dict) else data
        result = plan(intervals, args.slice_seconds)
        result['interval_input_sha256'] = hashlib.sha256(raw).hexdigest()
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        print(f"planned {len(result['dialogue'])} slices from {result['source_interval_count']} speech intervals")
        return 0
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        if created:
            args.output.unlink(missing_ok=True)
        print(f'slice-dialogue-intervals: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
