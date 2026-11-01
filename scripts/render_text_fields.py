#!/usr/bin/env python3
"""Snapshot and prepare H3 text fields without putting their contents in argv.

The byte limit is a multipart transport budget, not a model token guarantee.
Used by render.sh; this helper neither submits requests nor starts the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read_text(path: Path, limit: int) -> str:
    if not path.is_file():
        raise ValueError(f'text input must be a regular file: {path}')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'text input exceeds field limit {limit} bytes: {path}')
    return data.decode('utf-8')  # Preserve CRLF, trailing newlines and all literal punctuation.


def write_field(path: Path, text: str, limit: int) -> dict:
    data = text.encode('utf-8')
    if len(data) > limit:
        raise ValueError(f'{path.name} field needs {len(data)} bytes, limit is {limit}; '
                         'raise H3_TEXT_MAX_FIELD_BYTES only if the server is configured for it')
    with path.open('xb') as stream:
        stream.write(data)
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def prepare(args):
    if not 1 <= args.limit <= 64 * 1024 * 1024:
        raise ValueError('H3_TEXT_MAX_FIELD_BYTES must be between 1 and 67108864')
    prompt = read_text(args.prompt, args.limit)
    raw_extra = read_text(args.extra, args.limit)
    extra = {'task': 'ref2va' if args.task == 'ref2va' else ('fl2va' if args.image else 't2va'),
             'duration': json.loads(args.duration), 'audio_flow_shift': json.loads(args.audio_shift)}
    if extra['task'] == 'fl2va':
        extra['frame_indices'] = {'first': [0], 'last': [-1], 'both': [0, -1]}[args.where]
    overrides = json.loads(raw_extra) if raw_extra else {}
    if not isinstance(overrides, dict):
        raise ValueError('extra_params must be a JSON object')
    extra.update(overrides)
    target = extra.get('target')
    if target is not None and not isinstance(target, dict):
        raise ValueError('target must be an object')
    if (args.auto and extra.get('long_video') and extra.get('long_video_mode', 'continuation') == 'continuation'
            and not extra.get('continuation_prompts')):
        duration = (target or {}).get('duration_seconds', extra.get('duration_seconds', extra.get('duration')))
        if duration is not None:
            frames = int(round(float(duration) * 24))
            frames = ((frames - 5 + 16) // 17) * 17 + 5
            window = int(extra.get('continuation_window_frames', 277))
            overlap = int(extra.get('continuation_overlap_frames', 22))
            if window <= overlap:
                raise ValueError('continuation window must exceed overlap')
            count = 1 if frames <= window else 1 + math.ceil((frames - window) / (window - overlap))
            words = read_text(args.directory / 'words.input', args.limit).strip()
            template = read_text(args.directory / 'rest.input', args.limit).strip()
            first = prompt.rstrip('. ') + '. One single continuous take from a steady camera, no cuts, no transitions.'
            rest = (template.replace('{base}', prompt) if template else
                    'The same continuous shot goes on without any cut or transition, the camera keeps moving the same way: ' + prompt)
            if words:
                first += ' ' + words
                rest += ' ' + words
            # Reject an impossible final field before allocating a huge list.
            # JSON punctuation/escaping can make it larger; the exact check follows.
            if (count > args.limit // 3 or
                    len(first.encode('utf-8')) + (count - 1) * len(rest.encode('utf-8')) > args.limit):
                raise ValueError(f'generated continuation prompts exceed field limit {args.limit} bytes')
            extra['continuation_prompts'] = [first] + [rest] * (count - 1)
    # Both files are private snapshots. curl and timing read these same bytes,
    # so later edits of caller-owned files cannot change the submitted request.
    metadata = {'limit_bytes': args.limit,
                'prompt': write_field(args.directory / 'prompt.txt', prompt, args.limit),
                'extra_params': write_field(args.directory / 'extra.json', json.dumps(extra, ensure_ascii=False), args.limit)}
    (args.directory / 'text-meta.json').write_text(json.dumps(metadata), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--extra', type=Path, required=True)
    parser.add_argument('--duration', required=True)
    parser.add_argument('--audio-shift', required=True)
    parser.add_argument('--image', default='')
    parser.add_argument('--where', choices=['first', 'last', 'both'], default='first')
    parser.add_argument('--task', default='')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--limit', type=int, default=1048576)
    prepare(parser.parse_args())
