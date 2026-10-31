#!/usr/bin/env python3
"""Describe regional frame changes by retiming phase; this does not judge visual quality."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys


spec = importlib.util.spec_from_file_location('regional_flicker', Path(__file__).with_name('regional_flicker.py'))
regional = importlib.util.module_from_spec(spec); spec.loader.exec_module(regional)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def phase_summary(series, period):
    if not 2 <= period <= 60 or len(series) < period * 3:
        raise ValueError('period must be 2..60 and have at least three frames per phase')
    phases = []
    for phase in range(period):
        values = [series[i] for i in range(1, len(series)) if i % period == phase]
        if len(values) < 2:
            raise ValueError('not enough frame transitions in a phase')
        phases.append({'phase': phase, 'count': len(values), 'median': statistics.median(values)})
    return phases


def inspect(path, period):
    path = Path(path).expanduser().resolve(strict=True)
    before = digest(path)
    fps, frames, series = regional.inspect(path)
    if digest(path) != before:
        raise ValueError('video changed during analysis')
    return {'video': str(path), 'sha256': before, 'fps': fps, 'frames': frames, 'period': period,
            'regions': {name: phase_summary(values, period)
                        for name, values in zip(regional.LABELS, series, strict=True)},
            'caveat': ('Median absolute grayscale frame changes at each phase are descriptive. '
                       'Real motion and lighting can affect them; even values do not establish visual quality, '
                       'and uneven values do not by themselves prove visible judder.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--period', type=int, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.video, args.period), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError) as exc:
        print('retime-phase-diagnostics:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
