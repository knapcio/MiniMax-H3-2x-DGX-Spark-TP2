#!/usr/bin/env python3
"""Plan expected scene state across separate H3 continuation requests.

State changes are author-supplied expectations, not observations of generated
frames. A renderer must inspect each result before treating them as facts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import stat
import sys

import estimate_continuation as geometry
import plan_segmented_story as segmented

MAX_WINDOWS = 512
MAX_STATE_BYTES = 16 * 1024


def read_input(path: Path, limit: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('story and context inputs must be regular files')
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after):
        raise ValueError('story or context changed while reading')
    if len(raw) > limit:
        raise ValueError('story or context exceeds the planning input limit')
    return raw


def check_state_size(value: dict[str, str]) -> None:
    if sum(len(key.encode('utf-8')) + len(item.encode('utf-8')) for key, item in value.items()) > MAX_STATE_BYTES:
        raise ValueError('combined scene state exceeds 16 KiB')


def state_map(value: object, field: str, *, allow_empty: bool = False) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > 64 or (not value and not allow_empty):
        raise ValueError(f'{field} must contain {"0" if allow_empty else "1"}..64 state entries')
    result = {}
    for name, description in value.items():
        if (not isinstance(name, str) or not name.strip() or len(name) > 128
                or not isinstance(description, str) or not description.strip() or len(description) > 512):
            raise ValueError(f'{field} state names and values must be short nonempty strings')
        result[name] = description
    check_state_size(result)
    return result


def plan(seconds: float, continuity: str, initial_state: dict, beats: list[dict], segments: list[int], *,
         window: int = 277, overlap: int = 22, reference: str = 'last-frame') -> dict:
    if type(seconds) not in (int, float) or not 0 < seconds <= 3600 or not math.isfinite(seconds):
        raise ValueError('duration must be finite and within 0..3600 seconds')
    if not isinstance(continuity, str) or not continuity.strip() or len(continuity.encode('utf-8')) > 64 * 1024:
        raise ValueError('continuity must be nonempty text of at most 64 KiB')
    initial = state_map(initial_state, 'initial_state')
    if not isinstance(beats, list) or not 1 <= len(beats) <= 1000:
        raise ValueError('beats must contain 1..1000 entries')
    if (not isinstance(segments, list) or not 1 <= len(segments) <= MAX_WINDOWS
            or any(type(frames) is not int or frames < 96 for frames in segments)):
        raise ValueError('segments must contain 1..512 integer frame counts of at least 96 frames')
    # Bound the planner before constructing prompts. The underlying geometry
    # validates window/overlap and handles frame alignment for each request.
    if sum(segments) != round(seconds * 24):
        raise ValueError('segment frames must sum to the requested frame count')
    count = 0
    for frames in segments:
        count += len(geometry.plan(geometry.aligned_frames(frames / 24), window, overlap))
        if count > MAX_WINDOWS:
            raise ValueError('stateful plan exceeds 512 windows')
    simple = []
    transitions = []
    assumed = initial.copy()
    action_bytes = 0
    for beat in beats:
        if (not isinstance(beat, dict) or not {'start', 'end', 'action'}.issubset(beat)
                or set(beat) - {'start', 'end', 'action', 'before', 'after'}):
            raise ValueError('each beat needs start/end/action and optional before/after state maps')
        if not isinstance(beat['action'], str) or not beat['action'].strip():
            raise ValueError('each action must be nonempty text')
        action_bytes += len(beat['action'].encode('utf-8'))
        if action_bytes > 256 * 1024:
            raise ValueError('combined actions exceed 256 KiB')
        before = state_map(beat.get('before', {}), 'before', allow_empty=True)
        after = state_map(beat.get('after', {}), 'after', allow_empty=True)
        if any(key not in initial for key in before.keys() | after.keys()):
            raise ValueError('state transitions must use keys from initial_state')
        if any(assumed[key] != value for key, value in before.items()):
            raise ValueError('beat before-state conflicts with preceding planned state')
        assumed.update(after)
        check_state_size(assumed)
        simple.append({key: beat[key] for key in ('start', 'end', 'action')})
        transitions.append({'before': before, 'after': after})
    result = segmented.plan(seconds, continuity, simple, segments, window=window, overlap=overlap, reference=reference)
    for segment in result['segments']:
        for report, prompt_index in zip(segment['windows'], range(len(segment['windows'])), strict=True):
            new_start = report['new_global_frames'][0]
            current = initial.copy()
            ongoing = []
            uncertain_keys = set()
            for beat_index, (beat, transition) in enumerate(zip(result['beats'], transitions, strict=True), 1):
                if beat['end_frame'] <= new_start:
                    current.update(transition['after'])
                elif beat['first_frame'] < new_start < beat['end_frame']:
                    ongoing.append(beat_index)
                    uncertain_keys.update(transition['before'])
                    uncertain_keys.update(transition['after'])
            # A before-value describes the start of the action, not its middle.
            # Do not assert it after an ongoing action has already begun.
            stable = {key: value for key, value in current.items() if key not in uncertain_keys}
            report['last_completed_planned_state'] = current
            report['assumed_stable_state_at_new_start'] = stable
            report['ongoing_beat_indices'] = ongoing
            report['in_progress_state_keys'] = sorted(uncertain_keys)
            report['state_verified_from_video'] = False
            state_lines = '\n'.join(f'{name}: {description}' for name, description in stable.items())
            original = segment['extra_params']['continuation_prompts'][prompt_index]
            # Preserve the existing planner's exact beat directions after the
            # same context; do not repeat an action after its planned end.
            prefix = continuity + '\n'
            if not original.startswith(prefix):
                raise ValueError('unexpected underlying prompt format')
            state_direction = ('At the start of the new portion, the expected stable scene state is:\n'
                               + state_lines + '\n') if stable else ''
            if uncertain_keys:
                state_direction += ('For the ongoing action, continue smoothly from the provided frames without '
                                    'resetting these changing elements: ' + ', '.join(sorted(uncertain_keys)) + '.\n')
            directions, with_targets = [], []
            for beat_index in report['beat_indices']:
                beat = result['beats'][beat_index - 1]
                direction = ('Continue the action already underway: ' if beat['first_frame'] < new_start else 'Next: ') + beat['action']
                directions.append(direction)
                target = transitions[beat_index - 1]['after']
                if target:
                    target_lines = '\n'.join(f'{name}: {description}' for name, description in target.items())
                    direction += '\nWhen this action is complete, the intended state is:\n' + target_lines
                with_targets.append(direction)
            if not directions:
                directions = with_targets = ['No new story beat is specified; continue from the current state.']
            if original[len(prefix):] != '\n'.join(directions):
                raise ValueError('unexpected underlying beat directions')
            segment['extra_params']['continuation_prompts'][prompt_index] = prefix + state_direction + '\n'.join(with_targets)
        segment['extra_params_json_bytes_default_python'] = len(json.dumps(segment['extra_params']).encode('utf-8'))
    result['initial_state'] = initial
    result['planned_transitions'] = transitions
    result['state_status'] = 'author-supplied expectation, not verified from generated video'
    result['limitations'] += (' Planned state is applied only after a beat ends in the timeline. '
                              'It may be false if the model ignored an earlier action; compare actual frames '
                              'before relying on it. Prompt directions do not enforce precise action timing.')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--continuity-file', type=Path, required=True)
    parser.add_argument('--story', type=Path, required=True, help='JSON object with initial_state and beats')
    parser.add_argument('--segment-frames', nargs='+', type=int, required=True)
    parser.add_argument('--window', type=int, default=277)
    parser.add_argument('--overlap', type=int, default=22)
    parser.add_argument('--reference', choices=('last-frame', 'explicit'), default='last-frame')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    created = False
    try:
        raw_story = read_input(args.story, 1024 * 1024)
        raw_context = read_input(args.continuity_file, 64 * 1024)
        story = json.loads(raw_story)
        if not isinstance(story, dict) or set(story) != {'initial_state', 'beats'}:
            raise ValueError('story must contain exactly initial_state and beats')
        result = plan(args.seconds, raw_context.decode('utf-8'), story['initial_state'], story['beats'],
                      args.segment_frames, window=args.window, overlap=args.overlap, reference=args.reference)
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
        print(f"planned {len(result['segments'])} requests; scene state is not verified from video")
        return 0
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        if created:
            args.output.unlink(missing_ok=True)
        print(f'plan-stateful-story: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
