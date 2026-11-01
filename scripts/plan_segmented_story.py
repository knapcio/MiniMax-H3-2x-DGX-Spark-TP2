#!/usr/bin/env python3
"""Map one story onto separately rendered, frame-aligned H3 requests.

This creates a plan only. It does not render, extract references or guarantee
that independently generated requests are visually or narratively continuous.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import estimate_continuation as geometry
import plan_story_windows as story


def plan(seconds: float, continuity: str, beats: list[dict], segments: list[int], *,
         window: int = 277, overlap: int = 22, reference: str = 'last-frame') -> dict:
    continuity, requested, _aligned, checked = story.validate_story(seconds, continuity, beats)
    if (not isinstance(segments, list) or not 1 <= len(segments) <= 1000
            or any(type(n) is not int or n < 96 for n in segments)):
        raise ValueError('segments must contain 1..1000 integer frame counts of at least 96 frames each')
    if sum(segments) != requested:
        raise ValueError('segment frames must sum to the requested duration rounded to 24 fps')
    if reference not in ('last-frame', 'explicit'):
        raise ValueError('reference must be last-frame or explicit')

    result = []
    offset = 0
    assigned: set[int] = set()
    for segment_index, frames in enumerate(segments, 1):
        aligned = geometry.aligned_frames(frames / 24)
        windows = geometry.plan(aligned, window, overlap)
        reports, prompts = [], []
        for index, (start, end) in enumerate(windows):
            new_start = 0 if index == 0 else windows[index - 1][1]
            new_end = min(end, frames)
            global_start, global_end = offset + new_start, offset + new_end
            active = [(j, beat) for j, beat in enumerate(checked, 1)
                      if beat['first_frame'] < global_end and beat['end_frame'] > global_start]
            assigned.update(j for j, _ in active)
            directions = [
                ('Continue the action already underway: ' if beat['first_frame'] < global_start else 'Next: ')
                + beat['action'] for _, beat in active
            ] or ['No new story beat is specified; continue from the current state.']
            prompts.append(continuity + '\n' + '\n'.join(directions))
            reports.append({'index': index + 1, 'local_frames': [start, end],
                            'new_global_frames': [global_start, global_end],
                            'beat_indices': [j for j, _ in active],
                            'multiple_beats_in_window': len(active) > 1})
        if segment_index == 1:
            reference_plan = {'kind': 'initial-request-input'}
        elif reference == 'explicit':
            reference_plan = {'kind': 'user-supplied'}
        else:
            previous_frames = segments[segment_index - 2]
            reference_plan = {'kind': 'previous-segment-last-frame', 'segment': segment_index - 1,
                              'source': 'verified-exact-duration-output',
                              'expected_frames': previous_frames, 'frame_index_zero_based': previous_frames - 1}
        extra = {'long_video': True, 'long_video_mode': 'continuation', 'preencode_mp4': True,
                 'continuation_window_frames': window, 'continuation_overlap_frames': overlap,
                 'continuation_prompts': prompts}
        result.append({'index': segment_index, 'global_frames': [offset, offset + frames],
                       'expected_output_frames': frames, 'duration_seconds': frames / 24,
                       'aligned_internal_frames': aligned, 'first_frame_reference_plan': reference_plan,
                       'windows': reports, 'extra_params': extra,
                       'extra_params_json_bytes_default_python': len(json.dumps(extra).encode('utf-8'))})
        offset += frames
    if assigned != set(range(1, len(checked) + 1)):
        raise ValueError('one or more beats have no output frames')
    return {'requested_seconds': seconds, 'fps': 24, 'requested_frames': requested,
            'frame_ranges': 'zero-based, end-exclusive', 'window_frames': window, 'overlap_frames': overlap,
            'beats': checked, 'segments': result,
            'limitations': ('A plan only: no media or reference frame is extracted and no request is sent. '
                            'The caller must provide each reference, map audio to the global timeline and verify '
                            'the resulting video. Per-window directions are not frame-accurate and cannot guarantee '
                            'identity, story order, timing, lip sync or visual continuity.')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', required=True, type=float)
    parser.add_argument('--continuity-file', required=True, type=Path)
    parser.add_argument('--beats', required=True, type=Path, help='ordered JSON array of start/end/action objects')
    parser.add_argument('--segment-frames', required=True, nargs='+', type=int,
                        help='24 fps output frame counts; each is at least 96 frames')
    parser.add_argument('--window', type=int, default=277)
    parser.add_argument('--overlap', type=int, default=22)
    parser.add_argument('--reference', choices=('last-frame', 'explicit'), default='last-frame')
    parser.add_argument('--output', required=True, type=Path, help='new private JSON plan')
    args = parser.parse_args()
    created = False
    try:
        continuity = args.continuity_file.read_bytes().decode('utf-8')
        beats = json.loads(args.beats.read_bytes().decode('utf-8'))
        result = plan(args.seconds, continuity, beats, args.segment_frames, window=args.window,
                      overlap=args.overlap, reference=args.reference)
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print(f"planned {len(result['segments'])} requests; reference extraction and quality remain unverified")
        return 0
    except (OSError, ValueError, OverflowError, TypeError) as exc:
        if created:
            args.output.unlink(missing_ok=True)
        print(f'plan-segmented-story: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
