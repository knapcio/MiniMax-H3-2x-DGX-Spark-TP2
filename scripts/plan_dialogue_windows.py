#!/usr/bin/env python3
"""Prepare explicit per-window dialogue prompts for an H3 continuation.

This is a planning aid, not a guarantee that the model speaks on schedule.
Supply a separate time-aligned audio reference when exact audio timing matters.

The report also counts the words each window has to speak in its new content
and warns when a window is denser than --max-words-per-second (default 2.75):
on 2026-11-05 (results/20261105-polish-pacing) three Polish lines per window
were all spoken at 2.26-2.64 words/s of new content, but the 6 s tail window,
three short lines at 2.85 words/s, dropped half a line.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

import estimate_continuation as geometry

WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
DEFAULT_MAX_WORDS_PER_SECOND = 2.75
# Below this the model has spare time in the window and fills it with pauses or invented speech: the one-line-per-
# window arms (about 0.7-1.0 words/s) babbled in 2 of 12 windows; a window with no lines at all babbled in the 2.4 s
# tail of the 192-frame arm (2026-11-05). 1.7 words/s was spoken cleanly.
DEFAULT_MIN_WORDS_PER_SECOND = 1.0


def phrase(value, name, *, required=True):
    if value is None and not required:
        return ''
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > 4000:
        raise ValueError(name + ' must be a short nonempty string' if required else name + ' must be a short string')
    value = value.strip()
    if '<' in value or '>' in value or any(ord(char) < 32 for char in value):
        raise ValueError(name + ' cannot contain markup or control characters')
    return value


def plan(seconds, scene, dialogue, *, window=277, overlap=22, lock_source=False,
         max_words_per_second=DEFAULT_MAX_WORDS_PER_SECOND, min_words_per_second=DEFAULT_MIN_WORDS_PER_SECOND):
    scene = phrase(scene, 'scene')
    if max_words_per_second is not None and (type(max_words_per_second) not in (int, float) or
                                             not math.isfinite(max_words_per_second) or max_words_per_second <= 0):
        raise ValueError('max_words_per_second must be a positive finite number or None')
    if min_words_per_second is not None and (type(min_words_per_second) not in (int, float) or
                                             not math.isfinite(min_words_per_second) or min_words_per_second <= 0):
        raise ValueError('min_words_per_second must be a positive finite number or None')
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
                      'language': language, 'window': owners[0],
                      'words': len(WORD.findall(unicodedata.normalize('NFC', text)))})
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
    # Words per second of each window's NEW content: the model has only that
    # span to speak the window's lines. A short tail window needs fewer lines.
    densities = []
    for i, (a, b) in enumerate(boundaries):
        words = sum(x['words'] for x in items if x['window'] == i)
        new_seconds = (b - a) / 24
        densities.append((words, round(words / new_seconds, 3) if new_seconds > 0 else None))
    warnings = []
    if max_words_per_second is not None:
        for i, (words, density) in enumerate(densities):
            if density is not None and density > max_words_per_second:
                a, b = boundaries[i]
                tail = (' (the last, partial window: give it fewer lines or make the clip a whole number of windows)'
                        if i == len(boundaries) - 1 and b - a < boundaries[0][1] - boundaries[0][0] else '')
                warnings.append(f'window {i + 1}: {words} words in {(b - a) / 24:.1f} s of new content = {density:g} '
                                f'words/s, above {max_words_per_second:g}{tail}')
    if min_words_per_second is not None:
        for i, (words, density) in enumerate(densities):
            if density is not None and density < min_words_per_second:
                a, b = boundaries[i]
                warnings.append(f'window {i + 1}: {words} words in {(b - a) / 24:.1f} s of new content = {density:g} '
                                f'words/s, below {min_words_per_second:g}: the model fills spare time with pauses or '
                                f'invented speech; give the window a line or shorten the clip to end with the previous one')
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
                         'dialogue_items': [j + 1 for j, x in enumerate(items) if x['window'] == i],
                         'words_approx': densities[i][0], 'words_per_new_second_approx': densities[i][1]}
                        for i, (start, end) in enumerate(windows)],
            'max_words_per_second': max_words_per_second, 'min_words_per_second': min_words_per_second,
            'warnings': warnings,
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
    parser.add_argument('--max-words-per-second', type=float, default=DEFAULT_MAX_WORDS_PER_SECOND,
                        help='warn (stderr and the report) when a window must speak more words per second of its new '
                             'content than this; 0 disables the check')
    parser.add_argument('--min-words-per-second', type=float, default=DEFAULT_MIN_WORDS_PER_SECOND,
                        help='warn when a window has fewer words per second of its new content than this (spare time '
                             'becomes pauses or invented speech); 0 disables the check')
    parser.add_argument('--report', type=Path, required=True, help='write full plan to a new JSON file')
    parser.add_argument('--extra-output', type=Path, required=True, help='write just extra_params to a new JSON file')
    args = parser.parse_args()
    try:
        result = plan(args.seconds, args.scene, json.loads(args.dialogue.read_text()), window=args.window,
                      overlap=args.overlap, lock_source=args.lock_source,
                      max_words_per_second=args.max_words_per_second or None,
                      min_words_per_second=args.min_words_per_second or None)
        for warning in result['warnings']:
            print('plan-dialogue-windows: warning:', warning, file=sys.stderr)
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
