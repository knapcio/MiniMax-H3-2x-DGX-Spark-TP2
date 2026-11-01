#!/usr/bin/env python3
"""Plan one connected story as explicit prompts for H3 continuation windows.

The stable context is repeated, while timed story beats appear only in windows
whose new frames intersect them. This does not enforce a cut, an exact time,
identity or narrative continuity. Generated footage still needs evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import estimate_continuation as geometry


def nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be nonempty text')
    return value


def validate_story(seconds: float, continuity: str, beats: list[dict]) -> tuple[str, int, int, list[dict]]:
    """Validate and quantize one global timeline, without allocating prompts."""
    continuity = nonempty(continuity, 'continuity')
    aligned = geometry.aligned_frames(seconds)
    requested = round(seconds * 24)
    if not isinstance(beats, list) or not 1 <= len(beats) <= 1000:
        raise ValueError('beats must contain 1..1000 entries')
    checked = []
    for item in beats:
        if not isinstance(item, dict) or set(item) != {'start', 'end', 'action'}:
            raise ValueError('each beat must contain start, end and action')
        start, end = item['start'], item['end']
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= seconds):
            raise ValueError('beat times must be finite, ordered and inside the requested duration')
        # The output is an integer number of 24 fps frames, so its actual end
        # can be slightly earlier than a fractional requested duration. Never
        # silently discard or truncate a beat in that rounded-off tail.
        if math.floor(start * 24) >= requested or end * 24 > requested + 1e-9:
            raise ValueError('beat extends beyond the duration rounded to 24 fps frames')
        checked.append({'start': start, 'end': end, 'action': nonempty(item['action'], 'action'),
                        'first_frame': math.floor(start * 24), 'end_frame': min(requested, math.ceil(end * 24))})
    if any(a['start'] > b['start'] or a['end'] > b['start'] for a, b in zip(checked, checked[1:])):
        raise ValueError('beats must be ordered and cannot overlap')
    return continuity, requested, aligned, checked


def plan(seconds: float, continuity: str, beats: list[dict], *, window: int = 277, overlap: int = 22) -> dict:
    continuity, requested, aligned, checked = validate_story(seconds, continuity, beats)
    windows = geometry.plan(aligned, window, overlap)

    prompts, reports = [], []
    for index, (start, end) in enumerate(windows):
        new_start = 0 if index == 0 else windows[index - 1][1]
        new_end = min(end, requested)
        active = [(j, beat) for j, beat in enumerate(checked)
                  if beat['first_frame'] < new_end and beat['end_frame'] > new_start]
        directions = []
        for _, beat in active:
            prefix = 'Continue the action already underway: ' if beat['first_frame'] < new_start else 'Next: '
            directions.append(prefix + beat['action'])
        if not directions:
            directions = ['No new story beat is specified; continue from the current state.']
        prompts.append(continuity + '\n' + '\n'.join(directions))
        reports.append({'index': index + 1, 'frames': [start, end], 'new_frames': [new_start, new_end],
                        'beat_indices': [j + 1 for j, _ in active],
                        'multiple_beats_in_window': len(active) > 1})
    assigned = {index for report in reports for index in report['beat_indices']}
    if assigned != set(range(1, len(checked) + 1)):
        raise ValueError('one or more beats have no output frames')
    extra = {'long_video': True, 'long_video_mode': 'continuation', 'preencode_mp4': True,
             'continuation_window_frames': window, 'continuation_overlap_frames': overlap,
             'continuation_prompts': prompts}
    return {'requested_seconds': seconds, 'fps': 24, 'requested_frames': requested,
            'aligned_internal_frames': aligned, 'window_frames': window, 'overlap_frames': overlap,
            'windows': reports, 'beats': checked, 'extra_params': extra,
            'extra_params_json_bytes_default_python': len(json.dumps(extra).encode('utf-8')),
            'limitations': ('These are per-window text directions, not frame-accurate timing or a guarantee of '
                            'story, identity or visual continuity. Multiple beats in one window can mix or occur '
                            'out of order. Audio, transitions and playback quality require separate verification.')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', required=True, type=float)
    parser.add_argument('--continuity-file', required=True, type=Path, help='stable UTF-8 character and setting context')
    parser.add_argument('--beats', required=True, type=Path, help='ordered JSON array of start/end/action objects')
    parser.add_argument('--window', type=int, default=277)
    parser.add_argument('--overlap', type=int, default=22)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--extra-output', required=True, type=Path)
    args = parser.parse_args()
    try:
        continuity = args.continuity_file.read_bytes().decode('utf-8')
        beats = json.loads(args.beats.read_bytes().decode('utf-8'))
        result = plan(args.seconds, continuity, beats, window=args.window, overlap=args.overlap)
        if args.report.exists() or args.extra_output.exists() or args.report == args.extra_output:
            raise FileExistsError('output already exists or paths are the same')
        with args.report.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
        try:
            with args.extra_output.open('x', encoding='utf-8') as f:
                json.dump(result['extra_params'], f, ensure_ascii=False, indent=2); f.write('\n')
        except BaseException:
            args.report.unlink(missing_ok=True)
            raise
        print(f"planned {len(result['windows'])} windows; per-window timing and quality remain unverified")
        return 0
    except (OSError, ValueError, OverflowError, TypeError) as exc:
        print(f'plan-story-windows: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
