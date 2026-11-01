#!/usr/bin/env python3
"""Prepare explicit per-window dialogue prompts for an H3 continuation.

This is a planning aid, not a guarantee that the model speaks on schedule.
Supply a separate time-aligned audio reference when exact audio timing matters.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import estimate_continuation as geometry


def phrase(value, name, *, required=True):
    if value is None and not required:
        return ''
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > 4000:
        raise ValueError(name + ' must be a short nonempty string' if required else name + ' must be a short string')
    value = value.strip()
    if '<' in value or '>' in value or any(ord(char) < 32 for char in value):
        raise ValueError(name + ' cannot contain markup or control characters')
    return value


def plan(seconds, scene, dialogue, *, window=277, overlap=22, lock_source=False):
    scene = phrase(scene, 'scene')
    if not isinstance(dialogue, list) or len(dialogue) > 1000:
        raise ValueError('dialogue must be a list with at most 1000 items')
    if type(lock_source) is not bool:
        raise ValueError('lock_source must be boolean')
    aligned = geometry.aligned_frames(seconds)
    windows = geometry.plan(aligned, window, overlap)
    requested = round(seconds * 24)
    # Each global frame is generated as new content once. Later windows also
    # see their overlap, but repeating its dialogue in the prompt is risky.
    boundaries = [(0 if i == 0 else windows[i - 1][1], min(end, requested))
                  for i, (_, end) in enumerate(windows)]
    items = []
    for entry in dialogue:
        if not isinstance(entry, dict) or set(entry) - {'start', 'end', 'text', 'speaker', 'language'}:
            raise ValueError('each dialogue item must contain only start, end, text, speaker and language')
        start, end = entry.get('start'), entry.get('end')
        if (type(start) not in (int, float) or type(end) not in (int, float) or
                not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= seconds):
            raise ValueError('dialogue times must be finite, ordered and inside the requested duration')
        text = phrase(entry.get('text'), 'text')
        speaker = phrase(entry.get('speaker'), 'speaker', required=False)
        language = phrase(entry.get('language', 'Polish'), 'language')
        first, last = math.floor(start * 24), math.ceil(end * 24)
        owners = [i for i, (a, b) in enumerate(boundaries) if first >= a and last <= b]
        if len(owners) != 1:
            raise ValueError(f'dialogue at {start:g}–{end:g} s crosses a generated-window boundary; move or split it')
        items.append({'start': start, 'end': end, 'text': text, 'speaker': speaker,
                      'language': language, 'window': owners[0]})
    items.sort(key=lambda x: (x['start'], x['end']))
    if any(a['end'] > b['start'] for a, b in zip(items, items[1:])):
        raise ValueError('overlapping dialogue is not supported by this planner')
    prompts = []
    for index, _window in enumerate(windows):
        sentences = []
        for item in (x for x in items if x['window'] == index):
            who = f"{item['speaker']}: " if item['speaker'] else ''
            sentences.append(f"{who}<d>[{item['language']}] {item['text']}</d>")
        direction = (' '.join(sentences) if sentences else 'No new dialogue is spoken in this part.')
        prompts.append(scene + ' ' + direction)
    extra = {'long_video': True, 'long_video_mode': 'continuation', 'preencode_mp4': True,
             # The prompt assignments and the server must use the same frame
             # boundaries, even when their window counts happen to match.
             'continuation_window_frames': window, 'continuation_overlap_frames': overlap,
             'continuation_prompts': prompts}
    if lock_source:
        extra['audio_mode'] = 'lock_source'
    return {'requested_seconds': seconds, 'fps': 24, 'window_frames': window, 'overlap_frames': overlap,
            'requested_frames': requested, 'aligned_internal_frames': aligned,
            'windows': [{'index': i + 1, 'frames': [start, end], 'new_frames': list(boundaries[i]),
                         'new_seconds': [round(boundaries[i][0] / 24, 4), round(boundaries[i][1] / 24, 4)],
                         'dialogue_items': [j + 1 for j, x in enumerate(items) if x['window'] == i]}
                        for i, (start, end) in enumerate(windows)],
            'dialogue': [{**x, 'window': x['window'] + 1} for x in items], 'extra_params': extra,
            'limitations': ('Text assignment prevents accidental prompt repetition, not generated speech repetition '
                            'or exact timing. Audio conditioning, visuals, speech and lip synchronization need testing.')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--scene', required=True, help='scene description without dialogue markup')
    parser.add_argument('--dialogue', type=Path, required=True, help='JSON list of time-stamped dialogue objects')
    parser.add_argument('--window', type=int, default=277)
    parser.add_argument('--overlap', type=int, default=22)
    parser.add_argument('--lock-source', action='store_true', help='include audio_mode=lock_source in extra_params')
    parser.add_argument('--report', type=Path, required=True, help='write full plan to a new JSON file')
    parser.add_argument('--extra-output', type=Path, required=True, help='write just extra_params to a new JSON file')
    args = parser.parse_args()
    try:
        result = plan(args.seconds, args.scene, json.loads(args.dialogue.read_text()), window=args.window,
                      overlap=args.overlap, lock_source=args.lock_source)
        # Publish neither output if either path already exists. Neither file
        # is used by a running request automatically.
        if args.report.exists() or args.extra_output.exists() or args.report == args.extra_output:
            raise FileExistsError('output already exists or paths are the same')
        with args.report.open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
        try:
            with args.extra_output.open('x') as f:
                json.dump(result['extra_params'], f, ensure_ascii=False, indent=2); f.write('\n')
        except BaseException:
            args.report.unlink(missing_ok=True)
            raise
        return 0
    except (OSError, ValueError, OverflowError) as exc:
        print('plan-dialogue-windows:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
