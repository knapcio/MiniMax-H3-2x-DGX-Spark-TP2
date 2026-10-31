#!/usr/bin/env python3
"""Compare encoded global luma changes by temporal phase in two video clips."""

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


spec = importlib.util.spec_from_file_location('analyze_vae_trace', Path(__file__).with_name('analyze_vae_trace.py'))
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def metadata(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                             'stream=width,height,avg_frame_rate', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)['streams']
    if len(streams) != 1:
        raise ValueError('expected one selected video stream')
    stream = streams[0]
    rate = Fraction(stream['avg_frame_rate'])
    if rate <= 0 or int(stream['width']) <= 0 or int(stream['height']) <= 0:
        raise ValueError('invalid video geometry or frame rate')
    return {'width': int(stream['width']), 'height': int(stream['height']), 'fps': str(rate)}


def compare(source, candidate, *, start=1, end=None, period=17):
    source, candidate = Path(source).resolve(strict=True), Path(candidate).resolve(strict=True)
    before = [digest(source), digest(candidate)]
    geometry = metadata(source)
    if geometry != metadata(candidate):
        raise ValueError('video dimensions or average frame rates differ')
    first, second = analyzer.encoded_means(source), analyzer.encoded_means(candidate)
    if len(first) != len(second):
        raise ValueError('decoded frame counts differ')
    if [digest(source), digest(candidate)] != before:
        raise ValueError('a source changed during analysis')
    return {'source': str(source), 'candidate': str(candidate), 'source_sha256': before[0],
            'candidate_sha256': before[1], 'video': geometry, 'decoded_frames': len(first),
            'frame_interval': [start, len(first) if end is None else end], 'period': period,
            'source_phases': analyzer.phase_changes(first, start=start, end=end, period=period),
            'candidate_phases': analyzer.phase_changes(second, start=start, end=end, period=period),
            'caveat': ('Descriptive global luma changes aligned by frame index, not a visual-quality verdict or proof '
                       'of cause. Average frame rates and counts do not prove identical frame timestamps; '
                       'real scene motion or lighting may also generate periodic changes.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--start', type=int, default=1, help='first transition frame, inclusive')
    parser.add_argument('--end', type=int, help='last transition frame, exclusive')
    parser.add_argument('--period', type=int, default=17)
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.source, args.candidate, start=args.start, end=args.end, period=args.period), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError) as exc:
        print(f'compare-temporal-phases: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
