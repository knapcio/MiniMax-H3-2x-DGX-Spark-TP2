#!/usr/bin/env python3
"""Check whether each timed source-audio interval survives in a rendered clip.

Waveform correlation screens missing or shifted speech; it does not assess
voice naturalness, speaker identity, exact lip synchronization or picture.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import compare_driving_audio as audio


def assess(source: Path, output: Path, intervals: list[dict], *, rate: int = 8000,
           window: float = 2.5, padding: float = .15, search: float = .75,
           min_correlation: float = .65, max_delay: float = .25) -> dict:
    values = (window, padding, search, min_correlation, max_delay)
    if (not isinstance(intervals, list) or not intervals or len(intervals) > 1000 or rate < 1000
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
            or window < .5 or padding < 0 or search < 0 or not -1 <= min_correlation <= 1
            or not 0 <= max_delay <= search):
        raise ValueError('invalid intervals or screening thresholds')
    checked = []
    for item in intervals:
        if not isinstance(item, dict):
            raise ValueError('interval must be an object')
        start, end = item.get('start'), item.get('end')
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end
                or end - start - padding < .5):
            raise ValueError('each interval needs finite start/end and at least half a second after padding')
        checked.append((float(start), float(end), item.get('speaker')))
    cap = max(end for _, end, _ in checked) + search + 1
    reference, rendered = audio.decode(source, rate, cap), audio.decode(output, rate, cap)
    source_time, output_time = audio.track_timing(source), audio.track_timing(output)
    a, b = source_time['audio_relative_to_timeline_s'], output_time['audio_relative_to_timeline_s']
    rows = []
    for start, end, speaker in checked:
        position = round((start + padding) * rate)
        length = round(min(window, end - start - padding) * rate)
        sample = audio.compare_segment(reference, rendered, position, length, round(search * rate))
        shift = None if sample['shift_samples'] is None else sample['shift_samples'] / rate
        presentation_shift = None if shift is None or a is None or b is None else shift + b - a
        corr = sample['correlation']
        passed = (sample['status'] == 'measured' and corr is not None and corr >= min_correlation
                  and presentation_shift is not None and abs(presentation_shift) <= max_delay)
        rows.append({'start_s': start, 'end_s': end, 'speaker': speaker, 'status': sample['status'],
                     'correlation': None if corr is None else round(corr, 5),
                     'shift_seconds': None if shift is None else round(shift, 5),
                     'presentation_shift_seconds': None if presentation_shift is None else round(presentation_shift, 5),
                     'pass': passed})
    return {'source_sha256': audio.sha(source), 'output_sha256': audio.sha(output), 'pass': all(x['pass'] for x in rows),
            'source_timing': source_time, 'output_timing': output_time, 'sample_rate': rate,
            'thresholds': {'min_correlation': min_correlation, 'max_presentation_shift_s': max_delay,
                           'search_s': search, 'window_s': window, 'padding_s': padding},
            'intervals': rows, 'limitations': ('A sparse waveform match is not a naturalness, speaker-identity, '
                                              'visual-continuity or lip-synchronization verdict.')}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--intervals', type=Path, required=True, help='JSON list or object with a dialogue list')
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--min-correlation', type=float, default=.65)
    p.add_argument('--max-delay', type=float, default=.25)
    args = p.parse_args()
    try:
        data = json.loads(args.intervals.read_text())
        intervals = data.get('dialogue') if isinstance(data, dict) else data
        result = assess(args.source, args.output, intervals, min_correlation=args.min_correlation,
                        max_delay=args.max_delay)
        with args.report.open('x') as f:
            json.dump(result, f, indent=2, ensure_ascii=False); f.write('\n')
        print(f"timed-audio: {'PASS' if result['pass'] else 'HOLD'}; {len(result['intervals'])} intervals")
        return 0 if result['pass'] else 1
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'timed-audio: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
