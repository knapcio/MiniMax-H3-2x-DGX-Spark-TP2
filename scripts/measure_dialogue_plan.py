#!/usr/bin/env python3
"""Measure speech occupancy and gaps in an author-supplied dialogue plan.

This does not inspect generated audio: a scheduled line may not be spoken as
planned, and intervals without scheduled dialogue may contain music or sound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import sys
import unicodedata

WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
MAX_PLAN_BYTES = 16 * 1024**2


def number(value, name):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError(name + ' must be a finite number')
    return value


def measure(seconds, dialogue):
    seconds = number(seconds, 'duration')
    if not 0 < seconds <= 86400:
        raise ValueError('duration must be greater than zero and at most 86400 seconds')
    if not isinstance(dialogue, list) or len(dialogue) > 10000:
        raise ValueError('dialogue must be a list of at most 10000 items')
    lines = []
    for original_index, item in enumerate(dialogue, 1):
        if not isinstance(item, dict):
            raise ValueError('each dialogue item must be an object')
        start, end = number(item.get('start'), 'start'), number(item.get('end'), 'end')
        if not 0 <= start < end <= seconds:
            raise ValueError('dialogue interval must be inside the requested duration')
        text, speaker = item.get('text'), item.get('speaker', '')
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError('dialogue text must be a nonempty string of at most 4000 characters')
        if not isinstance(speaker, str) or len(speaker) > 4000:
            raise ValueError('speaker must be a string of at most 4000 characters')
        lines.append({'input_index': original_index, 'start_s': start, 'end_s': end,
                      'speaker': speaker, 'words': len(WORD.findall(unicodedata.normalize('NFC', text)))})
    lines.sort(key=lambda item: (item['start_s'], item['end_s'], item['input_index']))
    if any(a['end_s'] > b['start_s'] for a, b in zip(lines, lines[1:])):
        raise ValueError('overlapping dialogue intervals are not supported')
    gaps = []
    previous_end = 0
    for i, line in enumerate(lines):
        if line['start_s'] > previous_end:
            gaps.append({'kind': 'opening' if i == 0 else 'between_lines',
                         'start_s': previous_end, 'end_s': line['start_s'],
                         'duration_s': round(line['start_s'] - previous_end, 6)})
        previous_end = line['end_s']
    if previous_end < seconds:
        gaps.append({'kind': 'closing' if lines else 'entire_clip',
                     'start_s': previous_end, 'end_s': seconds,
                     'duration_s': round(seconds - previous_end, 6)})
    occupied = sum(item['end_s'] - item['start_s'] for item in lines)
    words = sum(item['words'] for item in lines)
    between = [item['duration_s'] for item in gaps if item['kind'] == 'between_lines']
    return {'duration_s': seconds, 'utterances': len(lines), 'words_approx': words,
            'words_per_minute_of_clip_approx': round(words * 60 / seconds, 3),
            'scheduled_dialogue_s': round(occupied, 6),
            'scheduled_dialogue_fraction': round(occupied / seconds, 6),
            'unscheduled_dialogue_s': round(seconds - occupied, 6),
            'longest_unscheduled_dialogue_s': max((g['duration_s'] for g in gaps), default=0),
            'positive_between_line_gap_count': len(between),
            'positive_between_line_gap_median_s': statistics.median(between) if between else None,
            'lines': lines, 'gaps': gaps,
            'limitations': ('Scheduled intervals and approximate Unicode word counts only. '
                            'No generated audio is inspected and an interval without scheduled dialogue '
                            'is not necessarily silent. This does not judge naturalness, acting, '
                            'intelligibility or lip synchronization.')}


def read_plan(path):
    """Read one bounded regular-file snapshot, never block on a FIFO."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_PLAN_BYTES:
            raise ValueError('dialogue input must be a regular file of at most 16 MiB')
        with os.fdopen(fd, 'rb', closefd=False) as source:
            raw = source.read(MAX_PLAN_BYTES + 1)
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if len(raw) > MAX_PLAN_BYTES or len(raw) != before.st_size or identity(before) != identity(after):
            raise ValueError('dialogue input changed while reading')
        return json.loads(raw.decode('utf-8')), hashlib.sha256(raw).hexdigest()
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--dialogue', type=Path, required=True,
                        help='JSON list of objects with start, end, text and optional speaker')
    parser.add_argument('--report', type=Path, required=True, help='new JSON report')
    args = parser.parse_args()
    try:
        dialogue, digest = read_plan(args.dialogue)
        result = measure(args.seconds, dialogue)
        result['source_plan_sha256'] = digest
        with args.report.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
        print(f"{result['utterances']} lines; {result['scheduled_dialogue_s']} s scheduled dialogue; "
              f"longest unscheduled interval {result['longest_unscheduled_dialogue_s']} s")
        return 0
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print('measure-dialogue-plan:', type(exc).__name__ + ':', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
