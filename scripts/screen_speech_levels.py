#!/usr/bin/env python3
"""Measure relative RMS for timed source and rendered speech intervals.

RMS is not perceived loudness, intelligibility, speaker identity or lip sync.
Optional thresholds are caller-selected; there is no calibrated default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

import numpy as np
import audio_frame_timing
import compare_driving_audio as audio


def snapshot(path: Path, destination: Path) -> str:
    """Copy one stable regular file into the private directory and bind its hash."""
    # Opening a FIFO in blocking mode can hang before fstat can reject it.
    # O_NONBLOCK has no effect on ordinary regular-file reads.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 8 * 1024**3:
            raise ValueError('input is nonregular or exceeds the 8 GiB snapshot limit')
        target = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256(); total = 0
        with os.fdopen(fd, 'rb', closefd=False) as source, os.fdopen(target, 'wb') as out:
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > 8 * 1024**3:
                    raise ValueError('input grew beyond the 8 GiB snapshot limit')
                digest.update(chunk); out.write(chunk)
        after = os.fstat(fd)
        identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError('input changed while taking the snapshot')
        destination.chmod(0o400)
        return digest.hexdigest()
    finally:
        os.close(fd)


def inspect_frame_timing(path: Path) -> dict:
    """Bound ffprobe's frame JSON on disk before parsing it into memory.

    The shell applies a 65536-block output-file limit, then execs ffprobe with
    literal positional arguments. It never evaluates the media path as code.
    The additional 16 MiB check is independent of the shell's block size.
    """
    command = ['/bin/sh', '-c', 'ulimit -f 65536 || exit 2; exec "$@"', 'ffprobe-limit',
               'ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_frames', '-show_streams',
               '-show_entries', 'frame=pts,nb_samples:stream=sample_rate,time_base', '-of', 'json', str(path)]
    with tempfile.TemporaryFile() as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.DEVNULL, check=True, timeout=1800)
        if stream.tell() > 16 * 1024**2:
            raise ValueError('audio PTS report exceeds the 16 MiB limit')
        stream.seek(0)
        return audio_frame_timing.assess_frames(json.load(stream))


def decode_first_channel(path: Path, rate: int, cap: float) -> np.ndarray:
    result = subprocess.run(
        ['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(path), '-map', '0:a:0',
         '-af', 'pan=mono|c0=c0', '-ar', str(rate), '-t', str(cap), '-f', 'f32le', 'pipe:1'],
        capture_output=True, check=True, timeout=1800)
    if not result.stdout or len(result.stdout) % 4:
        raise ValueError('missing or incomplete decoded audio')
    result = np.frombuffer(result.stdout, dtype='<f4')
    if not np.isfinite(result).all():
        raise ValueError('decoded audio has nonfinite samples')
    return result


def assess(source: Path, output: Path, intervals: list[dict], *, rate: int = 32000,
           min_relative_db: float | None = None, max_relative_db: float | None = None) -> dict:
    if (not isinstance(intervals, list) or not 1 <= len(intervals) <= 1000
            or type(rate) is not int or not 1000 <= rate <= 48000):
        raise ValueError('invalid intervals or sample rate')
    for value in (min_relative_db, max_relative_db):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not -120 <= value <= 60):
            raise ValueError('relative RMS thresholds must be finite and within -120..60 dB')
    if min_relative_db is not None and max_relative_db is not None and min_relative_db > max_relative_db:
        raise ValueError('minimum relative RMS exceeds maximum')
    checked = []
    for item in intervals:
        if not isinstance(item, dict):
            raise ValueError('interval must be an object')
        start, end = item.get('start'), item.get('end')
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= 3600):
            raise ValueError('interval times must be finite and within 0..3600 seconds')
        checked.append((float(start), float(end), item.get('speaker')))
    with tempfile.TemporaryDirectory(prefix='h3-speech-levels-') as d:
        directory = Path(d)
        src_copy, out_copy = directory / ('source' + source.suffix), directory / ('output' + output.suffix)
        source_sha = snapshot(source, src_copy); output_sha = snapshot(output, out_copy)
        result = assess_snapshots(src_copy, out_copy, checked, rate=rate,
                                  min_relative_db=min_relative_db, max_relative_db=max_relative_db)
        if audio.sha(src_copy) != source_sha or audio.sha(out_copy) != output_sha:
            raise ValueError('private snapshot changed during analysis')
    if audio.sha(source) != source_sha or audio.sha(output) != output_sha:
        raise ValueError('input changed during analysis')
    return {'source_sha256': source_sha, 'output_sha256': output_sha, **result}


def assess_snapshots(source: Path, output: Path, checked: list[tuple[float, float, object]], *,
                     rate: int, min_relative_db: float | None, max_relative_db: float | None) -> dict:
    tracks = {'source': audio.track_timing(source), 'output': audio.track_timing(output)}
    frames = {'source': inspect_frame_timing(source), 'output': inspect_frame_timing(output)}
    offsets = {key: audio_frame_timing.relative_start(frames[key], tracks[key]) for key in tracks}
    if any(value is None or not math.isfinite(value) for value in offsets.values()):
        raise ValueError('decoded audio PTS are unavailable or discontinuous')
    cap = max(end for _, end, _ in checked) + max(abs(x) for x in offsets.values()) + 1
    if cap * rate > 20_000_000:
        raise ValueError('analysis exceeds 20000000 samples per input; lower the sample rate or split the intervals')
    reference, rendered = decode_first_channel(source, rate, cap), decode_first_channel(output, rate, cap)
    rows = []
    for start, end, speaker in checked:
        slices = []
        for samples, key in ((reference, 'source'), (rendered, 'output')):
            first, last = round((start - offsets[key]) * rate), round((end - offsets[key]) * rate)
            if not 0 <= first < last <= len(samples):
                raise ValueError(f'{key} interval outside decoded audio')
            segment = samples[first:last].astype(np.float64)
            slices.append(float(np.sqrt(np.mean(segment * segment))))
        ref_rms, out_rms = slices
        if ref_rms < 1e-6:
            status, relative = 'low_source_energy', None
        elif out_rms == 0:
            status, relative = 'silent_output', None
        else:
            status, relative = 'measured', 20 * math.log10(out_rms / ref_rms)
        failed = bool(status == 'silent_output' and min_relative_db is not None)
        if relative is not None:
            failed = ((min_relative_db is not None and relative < min_relative_db) or
                      (max_relative_db is not None and relative > max_relative_db))
        rows.append({'start_s': start, 'end_s': end, 'speaker': speaker, 'status': status,
                     'source_rms': ref_rms, 'output_rms': out_rms, 'output_over_source_db': relative,
                     'outside_selected_threshold': failed})
    if (any(x['status'] == 'low_source_energy' for x in rows) or
            ((min_relative_db is None and max_relative_db is not None) and
             any(x['status'] == 'silent_output' for x in rows))):
        status = 'INCOMPLETE'
    elif any(x['outside_selected_threshold'] for x in rows):
        status = 'FAIL'
    elif min_relative_db is not None or max_relative_db is not None:
        status = 'PASS'
    else:
        status = 'MEASURED'
    return {'status': status,
            'sample_rate': rate, 'channel': 'first decoded channel of each input',
            'source_timing': tracks['source'], 'output_timing': tracks['output'], 'frame_timing': frames,
            'thresholds': {'min_relative_db': min_relative_db, 'max_relative_db': max_relative_db},
            'intervals': rows,
            'limitations': ('First-channel RMS over the full scheduled interval, not perceptual loudness, '
                            'intelligibility or lip sync. Channel layout, silence and small speech shifts affect '
                            'the ratio. PTS continuity is screened; waveform identity and video are not. '
                            'Caller-selected thresholds are not calibrated quality judgments.')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--intervals', type=Path, required=True, help='JSON list or object with dialogue list')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--sample-rate', type=int, default=32000)
    parser.add_argument('--min-relative-db', type=float)
    parser.add_argument('--max-relative-db', type=float)
    args = parser.parse_args()
    try:
        data = json.loads(args.intervals.read_text())
        intervals = data.get('dialogue') if isinstance(data, dict) else data
        result = assess(args.source, args.output, intervals, rate=args.sample_rate,
                        min_relative_db=args.min_relative_db, max_relative_db=args.max_relative_db)
        with args.report.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        print('speech-levels:', result['status'], ';', len(result['intervals']), 'intervals')
        return 1 if result['status'] == 'FAIL' else 2 if result['status'] == 'INCOMPLETE' else 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print('speech-levels:', type(exc).__name__ + ':', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
