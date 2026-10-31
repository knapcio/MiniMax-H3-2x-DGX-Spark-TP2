#!/usr/bin/env python3
"""Plan H3 continuation windows and report observed full-window cost, not wall time."""
import argparse
import json
import math
import sys


def aligned_frames(seconds):
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 4:
        raise ValueError('duration must be finite and at least 4 seconds')
    requested = round(seconds * 24)
    if not 1 <= requested <= 1_000_000:
        raise ValueError('requested frame count must be 1..1000000')
    return requested + (5 - requested) % 17


def plan(total, window, overlap):
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (total, window, overlap)):
        raise ValueError('frame counts must be integers')
    if not 5 <= total <= 1_000_000 or total % 17 != 5:
        raise ValueError('total frames must be on the 17n+5 grid')
    if not 107 <= window <= 345 or window % 17 != 5:
        raise ValueError('window must be 107..345 frames on the 17n+5 grid')
    if not 5 <= overlap < window or overlap % 17 != 5:
        raise ValueError('overlap must be smaller than the window and on the 17n+5 grid')
    end = min(total, window)
    windows = [(0, end)]
    while end < total:
        start = end - overlap
        end = min(total, start + window)
        windows.append((start, end))
    return windows


def summarize(total, window, overlap, width, height, full_window_seconds=None):
    if (any(not isinstance(x, int) or isinstance(x, bool) for x in (width, height)) or
            width <= 0 or height <= 0 or width % 32 or height % 32):
        raise ValueError('height and width must be positive multiples of 32')
    if full_window_seconds is not None and (not math.isfinite(full_window_seconds) or full_window_seconds <= 0):
        raise ValueError('observed full-window time must be finite and positive')
    windows = plan(total, window, overlap)
    latent_t = lambda frames: (frames - 5) // 17 * 5 + 2
    # One cumulative video and one previous-window video per checkpoint.
    # Float32, B=1, 24 channels, spatial compression 16; audio and metadata excluded.
    bytes_per_t = 24 * (height // 16) * (width // 16) * 4
    cumulative = [bytes_per_t * latent_t(end) for _, end in windows]
    previous = [bytes_per_t * latent_t(end - start) for start, end in windows]
    full = sum(end - start == window for start, end in windows)
    short = [end - start for start, end in windows if end - start < window]
    return {'window_frames': window, 'overlap_frames': overlap, 'aligned_total_frames': total,
            'windows': len(windows), 'handovers': len(windows) - 1, 'full_windows': full,
            'short_window_frames': short, 'observed_full_window_seconds': full_window_seconds,
            'full_windows_seconds_at_observed_rate': (round(full * full_window_seconds, 1)
                                                       if full_window_seconds is not None else None),
            'final_cumulative_video_mib_float32': round(cumulative[-1] / 1024**2, 2),
            'last_previous_window_video_mib_float32': round(previous[-1] / 1024**2, 2),
            'all_checkpoint_video_mib_per_rank_float32': round((sum(cumulative) + sum(previous)) / 1024**2, 2),
            'caveat': ('Full-window multiplication excludes short windows, decode, setup, retries and runtime variation; '
                       'observations from different positions or requests are not controlled speed tests. '
                       'Checkpoint storage is uncompressed video-tensor arithmetic per rank, excludes audio, metadata, '
                       'temporary copies and filesystem overhead, and is not a peak-memory measurement.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--overlap', type=int, default=22)
    parser.add_argument('--window', type=int, action='append', required=True)
    parser.add_argument('--sample', action='append', default=[], metavar='WINDOW=SECONDS',
                        help='one observed full-window denoise time; no interpolation is performed')
    args = parser.parse_args()
    try:
        samples = {}
        for sample in args.sample:
            key, value = sample.split('=', 1)
            number, duration = int(key), float(value)
            if number in samples:
                raise ValueError('duplicate observed window size')
            samples[number] = duration
        if set(samples) - set(args.window):
            raise ValueError('sample has no matching requested window')
        total = aligned_frames(args.seconds)
        data = [summarize(total, window, args.overlap, args.width, args.height, samples.get(window))
                for window in args.window]
        print(json.dumps({'requested_seconds': args.seconds, 'fps': 24, 'results': data}, indent=2))
        return 0
    except (ValueError, OverflowError) as exc:
        print('estimate-continuation:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
