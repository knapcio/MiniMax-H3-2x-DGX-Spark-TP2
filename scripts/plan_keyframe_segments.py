#!/usr/bin/env python3
"""Plan exact-duration, single-window two-keyframe segments; no model access.

Each complete H3 segment is on the 17n+5 frame grid. This planner never trims
the last frames or claims the server will follow the requested end image.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import sys

import estimate_continuation as geometry
import snapshot_first_last

FPS = 24
MIN_FRAMES = 107  # first 17n+5 length at or above H3's four-second minimum
MAX_TOTAL_FRAMES = 1_000_000


def seconds_to_frames(value: str) -> int:
    if len(value) > 128:
        raise ValueError('seconds value is too long')
    try:
        seconds = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError('seconds must be a finite decimal') from exc
    if not seconds.is_finite() or seconds <= 0:
        raise ValueError('seconds must be positive and finite')
    if abs(seconds.as_tuple().exponent) > 100 or seconds > MAX_TOTAL_FRAMES:
        raise ValueError('seconds value is outside supported bounds')
    frames = Fraction(seconds) * FPS
    if frames.denominator != 1:
        raise ValueError('seconds must be exactly on the 24 fps frame grid; use --frames for repeating decimals')
    return frames.numerator


def plan_segments(total: int, *, window: int = 277, overlap: int = 22, sample_rate: int = 32000,
                  drop_boundary_frame: bool = False) -> dict:
    if type(total) is not int or not 1 <= total <= MAX_TOTAL_FRAMES:
        raise ValueError('target frames must be an integer in 1..1000000')
    if type(sample_rate) is not int or not 1 <= sample_rate <= 384000:
        raise ValueError('sample rate must be an integer in 1..384000')
    if type(drop_boundary_frame) is not bool:
        raise ValueError('drop_boundary_frame must be boolean')
    # Reuse the actual client geometry validator, including window/overlap bounds.
    geometry.plan(MIN_FRAMES, window, overlap)
    drop = int(drop_boundary_frame)
    lower = max(1, (total - drop + window - drop - 1) // (window - drop))
    upper = (total - drop) // (MIN_FRAMES - drop)
    count = next((n for n in range(lower, upper + 1)
                  if n * MIN_FRAMES <= total + drop * (n - 1) <= n * window
                  and (total + drop * (n - 1) - 5 * n) % 17 == 0), None)
    result = {
        'target_frames': total,
        'fps': FPS,
        'target_seconds_exact': str(Fraction(total, FPS)),
        'sample_rate': sample_rate,
        'window_frames': window,
        'overlap_frames': overlap,
        'drop_boundary_frame': drop_boundary_frame,
        'possible_preserving_endpoints': count is not None,
        'caveat': ('Geometry only. Boundary frames need not be identical even when a previous endpoint is used '
                   'as reference. Dropping a boundary frame is an editing choice; source audio cuts may fall '
                   'between samples and need explicit alignment. Conditioning, actions and runtime need checks.'),
    }
    if count is None:
        result['reason'] = ('No sum of 17n+5 source segments between 107 frames and the selected window, '
                            'with the chosen boundary policy, equals the target. A different strategy is needed.')
        return result
    increments = (total + drop * (count - 1) - count * MIN_FRAMES) // 17
    quotient, remainder = divmod(increments, count)
    lengths = [MIN_FRAMES + 17 * (quotient + (i < remainder)) for i in range(count)]
    segments = []
    cursor = 0
    for index, frames in enumerate(lengths, 1):
        seconds = frames / FPS
        contract = snapshot_first_last.validate_request(
            {'task': 'fl2va', 'frame_indices': [0, -1], 'continuation_window_frames': window,
             'continuation_overlap_frames': overlap}, seconds)
        if contract['requested_frames'] != frames or contract['alignment_tail_frames'] != 0:
            raise ValueError('segment does not round-trip through the two-keyframe client')
        discarded = drop if index > 1 else 0
        end = cursor + frames - discarded
        # Round cumulative boundaries, not each segment independently; this bounds drift.
        sample_start = round(Fraction(cursor * sample_rate, FPS))
        sample_end = round(Fraction(end * sample_rate, FPS))
        segments.append({'index': index, 'start_frame': cursor, 'end_frame': end, 'frames': frames,
                         'discard_prefix_frames': discarded, 'retained_frames': frames - discarded,
                         'request_duration_seconds': seconds, 'request_duration_exact': str(Fraction(frames, FPS)),
                         'source_audio_cut_start_samples_exact': str(Fraction(discarded * sample_rate, FPS)),
                         'source_audio_end_samples_exact': str(Fraction(frames * sample_rate, FPS)),
                         'timeline_start_sample': sample_start, 'timeline_end_sample': sample_end,
                         'timeline_audio_samples': sample_end - sample_start})
        cursor = end
    if cursor != total or any(s['frames'] > window for s in segments):
        raise ValueError('internal segment plan error')
    result.update(segment_count=count, target_audio_samples=round(Fraction(total * sample_rate, FPS)),
                  segments=segments)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--seconds', help='exact decimal duration on the 24 fps frame grid')
    target.add_argument('--frames', type=int, help='exact target frame count')
    parser.add_argument('--window', type=int, default=277)
    parser.add_argument('--overlap', type=int, default=22)
    parser.add_argument('--sample-rate', type=int, default=32000)
    parser.add_argument('--drop-boundary-frame', action='store_true',
                        help='plan to discard frame 0 of each segment after the first')
    args = parser.parse_args()
    try:
        frames = seconds_to_frames(args.seconds) if args.seconds is not None else args.frames
        result = plan_segments(frames, window=args.window, overlap=args.overlap, sample_rate=args.sample_rate,
                               drop_boundary_frame=args.drop_boundary_frame)
        print(json.dumps(result, indent=2))
        return 0 if result['possible_preserving_endpoints'] else 2
    except (ValueError, OverflowError) as exc:
        print(f'plan-keyframe-segments: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
