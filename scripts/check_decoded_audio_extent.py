#!/usr/bin/env python3
"""Measure decoded audio coverage of a caller-supplied presentation interval.

Container/packet duration is not proof that the audio decodes to that extent.
The expected interval must come from separately verified video presentation
times. This does not judge audibility, waveform identity or lip sync.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

import audio_frame_timing


def numeric(value, name):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError(name + ' must be a finite number')
    return Fraction(str(value))


def assess(data, expected_start, expected_end, max_missing=None):
    start = numeric(expected_start, 'expected start'); end = numeric(expected_end, 'expected end')
    if not -60 <= start < end <= 86400:
        raise ValueError('expected interval is outside supported limits')
    tolerance = None if max_missing is None else numeric(max_missing, 'maximum missing')
    if tolerance is not None and not 0 <= tolerance <= 10:
        raise ValueError('maximum missing must be 0..10 seconds')
    if not isinstance(data, dict):
        raise ValueError('decoded frame report must be an object')
    streams, frames = data.get('streams'), data.get('frames')
    if (not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict)
            or not isinstance(frames, list) or not frames):
        raise ValueError('one audio stream and decoded frames are required')
    stream = streams[0]
    raw_rate = stream.get('sample_rate')
    if not (type(raw_rate) is int or (type(raw_rate) is str and raw_rate.isascii() and raw_rate.isdigit())):
        raise ValueError('invalid decoded sample rate')
    rate = int(raw_rate)
    if not 1000 <= rate <= 384000:
        raise ValueError('unsupported decoded sample rate')
    raw_base = stream.get('time_base')
    if type(raw_base) is not str:
        raise ValueError('invalid audio time base')
    try:
        time_base = Fraction(raw_base)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError('invalid audio time base') from exc
    if time_base <= 0:
        raise ValueError('invalid audio time base')
    if any(not isinstance(frame, dict) or type(frame.get('pts')) is not int
           or type(frame.get('nb_samples')) is not int or frame['nb_samples'] <= 0 for frame in frames):
        raise ValueError('missing or invalid decoded-frame PTS/sample count')
    timing = audio_frame_timing.assess_frames(data)
    if timing['status'] != 'measured' or not timing['continuous']:
        raise ValueError('decoded audio frame timestamps are unavailable or discontinuous')
    total = sum(frame['nb_samples'] for frame in frames)
    actual_start = frames[0]['pts'] * time_base
    actual_end = frames[-1]['pts'] * time_base + Fraction(frames[-1]['nb_samples'], rate)
    missing_start = max(Fraction(0), actual_start - start)
    missing_end = max(Fraction(0), end - actual_end)
    status = ('MEASURED' if tolerance is None else
              'FAIL' if max(missing_start, missing_end) > tolerance else 'PASS')
    return {'status': status, 'sample_rate_hz': rate, 'decoded_frames': len(frames),
            'decoded_samples': total, 'decoded_start_s': float(actual_start), 'decoded_end_s': float(actual_end),
            'expected_start_s': float(start), 'expected_end_s': float(end),
            'missing_start_s': float(missing_start), 'missing_end_s': float(missing_end),
            'allowed_missing_s': None if tolerance is None else float(tolerance),
            'frame_timing': timing,
            'limitations': ('Expected presentation interval is caller supplied and must be verified separately. '
                            'Decoded coverage is not intelligibility, waveform identity, perceived quality or lip sync. '
                            'Extra decoded priming/padding outside the requested interval is reported by endpoints, '
                            'not treated as missing audio.')}


def stable_sha(path: Path) -> str:
    """Hash a stable regular file without blocking on a substituted FIFO."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 8 * 1024**3:
            raise ValueError('input is nonregular or exceeds the 8 GiB limit')
        digest = hashlib.sha256(); total = 0
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                total += len(chunk)
                if total > 8 * 1024**3:
                    raise ValueError('input grew beyond the 8 GiB limit')
                digest.update(chunk)
        after = os.fstat(fd)
        identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if identity(before) != identity(after) or total != before.st_size:
            raise ValueError('input changed while hashing')
        return digest.hexdigest()
    finally:
        os.close(fd)


def snapshot_file(path: Path, destination: Path) -> str:
    """Copy from one bounded regular-file descriptor and hash its bytes."""
    source_fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 8 * 1024**3:
            raise ValueError('input is nonregular or exceeds the 8 GiB limit')
        target_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256(); total = 0
        with os.fdopen(source_fd, 'rb', closefd=False) as source, os.fdopen(target_fd, 'wb') as target:
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > 8 * 1024**3:
                    raise ValueError('input grew beyond the 8 GiB limit')
                digest.update(chunk); target.write(chunk)
        after = os.fstat(source_fd)
        identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if identity(before) != identity(after) or total != before.st_size:
            raise ValueError('input changed while taking the snapshot')
        destination.chmod(0o400)
        return digest.hexdigest()
    finally:
        os.close(source_fd)


def probe(path):
    command = ['/bin/sh', '-c', 'ulimit -f 65536 || exit 2; exec "$@"', 'ffprobe-limit',
               'ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_frames', '-show_streams',
               '-show_entries', 'frame=pts,nb_samples:stream=sample_rate,time_base', '-of', 'json', str(path)]
    with tempfile.TemporaryFile() as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.DEVNULL, check=True, timeout=1800)
        if stream.tell() > 16 * 1024**2:
            raise ValueError('decoded-frame report exceeds 16 MiB')
        stream.seek(0)
        return json.load(stream)


def inspect(path: Path, expected_start: float, expected_end: float, max_missing: float | None = None):
    with tempfile.TemporaryDirectory(prefix='h3-audio-extent-') as directory:
        snapshot = Path(directory) / ('media' + path.suffix)
        digest = snapshot_file(path, snapshot)
        result = assess(probe(snapshot), expected_start, expected_end, max_missing)
        if stable_sha(snapshot) != digest:
            raise ValueError('media snapshot changed during inspection')
    if stable_sha(path) != digest:
        raise ValueError('original media changed during inspection')
    return {'media_sha256': digest, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('media', type=Path)
    parser.add_argument('--expected-start', type=float, default=0.0)
    parser.add_argument('--expected-end', type=float, required=True)
    parser.add_argument('--max-missing', type=float, help='caller-selected maximum missing time at either end, in seconds')
    parser.add_argument('--report', type=Path, required=True, help='new JSON report')
    args = parser.parse_args()
    try:
        result = inspect(args.media, args.expected_start, args.expected_end, args.max_missing)
        with args.report.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
        print('decoded audio extent:', result['status'], 'samples:', result['decoded_samples'],
              'missing at end:', result['missing_end_s'], 's')
        return 1 if result['status'] == 'FAIL' else 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, subprocess.SubprocessError) as exc:
        print('decoded-audio-extent:', type(exc).__name__ + ':', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
