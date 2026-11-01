#!/usr/bin/env python3
"""Compare an edited film's sampled video and audio to its ordered source shots.

This independent diagnostic can detect accidental omissions, reordering and
audio displacement. Sparse samples cannot establish the quality, continuity,
performance or lip synchronization of the resulting film.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import compare_driving_audio as audio


WIDTH, HEIGHT = 160, 90


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def video_info(path: Path) -> dict:
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=width,height,avg_frame_rate,nb_frames,duration,start_time', '-of', 'json', str(path)],
                           check=True, capture_output=True, text=True, timeout=60)
    streams = json.loads(probe.stdout)['streams']
    if len(streams) != 1:
        raise ValueError('expected exactly one video stream')
    stream = streams[0]
    return {'width': int(stream['width']), 'height': int(stream['height']),
            'fps': Fraction(stream['avg_frame_rate']), 'frames': int(stream['nb_frames']),
            'duration': float(stream['duration']), 'start': float(stream.get('start_time', 0))}


def video_sample(path: Path, seconds: float) -> np.ndarray:
    result = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-ss', f'{seconds:.9f}',
                             '-i', str(path), '-map', '0:v:0', '-frames:v', '1',
                             '-vf', f'scale={WIDTH}:{HEIGHT}:flags=area,format=rgb24',
                             '-f', 'rawvideo', 'pipe:1'], check=True, capture_output=True, timeout=120)
    if len(result.stdout) != WIDTH * HEIGHT * 3:
        raise ValueError('could not decode one complete comparison frame')
    return np.frombuffer(result.stdout, dtype=np.uint8).astype(np.float32)


def validate_timing(path: Path, *, frames: int, fps: int, geometry: tuple[int, int] | None = None) -> dict:
    info = video_info(path)
    if (info['fps'] != fps or info['frames'] != frames or abs(info['duration'] - frames / fps) > 1 / fps
            or abs(info['start']) > 1 / fps or (geometry and (info['width'], info['height']) != geometry)):
        raise ValueError('video timing or geometry differs: ' + str(path))
    timing = audio.track_timing(path)
    relative = timing['audio_relative_to_timeline_s']
    if relative is None or abs(relative) > .02:
        raise ValueError('audio start is unknown or not aligned to the video timeline: ' + str(path))
    return info


def assess(film: Path, shots: list[Path], *, shot_seconds: float | list[float], fps: int) -> dict:
    if not shots or len(shots) > 500 or fps < 1:
        raise ValueError('invalid shot list, rate or duration')
    durations = ([shot_seconds] * len(shots) if isinstance(shot_seconds, (int, float))
                 else list(shot_seconds))
    if len(durations) != len(shots) or any(type(x) not in (int, float) or not math.isfinite(x) or x < 1
                                             for x in durations):
        raise ValueError('provide one duration or one duration per shot')
    frame_counts = [round(seconds * fps) for seconds in durations]
    if any(abs(frames / fps - seconds) > 1e-8 for frames, seconds in zip(frame_counts, durations, strict=True)):
        raise ValueError('each shot duration must contain a whole number of frames')
    total = sum(frame_counts) / fps
    info = validate_timing(film, frames=sum(frame_counts), fps=fps)
    geometry = (info['width'], info['height'])
    rate = 8000
    film_audio = audio.decode(film, rate, total + 1)
    if len(film_audio) < int((total - .1) * rate):
        raise ValueError('film audio shorter than expected')
    rows = []
    offset_frames = 0
    for i, (shot, shot_seconds, frame_count) in enumerate(zip(shots, durations, frame_counts, strict=True)):
        validate_timing(shot, frames=frame_count, fps=fps, geometry=geometry)
        sample_frames = sorted(set(min(frame_count - 1, int(frame_count * fraction)) for fraction in (.05, .5, .95)))
        samples = []
        for frame in sample_frames:
            local = frame / fps
            reference = video_sample(shot, local)
            actual = video_sample(film, (offset_frames + frame) / fps)
            samples.append({'time_within_shot_s': round(local, 4),
                            'rgb_mean_absolute_difference': round(float(np.mean(np.abs(reference - actual))), 4)})
        ref_audio = audio.decode(shot, rate, shot_seconds + 1)
        start = round(offset_frames * rate / fps)
        end = round((offset_frames + frame_count) * rate / fps)
        segment = film_audio[start:end]
        compared = audio.compare(ref_audio, segment, rate, window_seconds=min(3, shot_seconds * .5),
                                 max_shift_seconds=min(.5, shot_seconds * .1), samples=3)
        measured = [x for x in compared if x['status'] == 'measured']
        rows.append({'shot': i + 1, 'seconds': shot_seconds, 'source': str(shot), 'source_sha256': sha(shot), 'video_samples': samples,
                     'audio_samples': compared,
                     'median_audio_correlation': round(statistics.median(x['correlation'] for x in measured), 5) if measured else None,
                     'median_audio_shift_s': round(statistics.median(x['shift_seconds'] for x in measured), 4) if measured else None})
        offset_frames += frame_count
    return {'film': str(film), 'film_sha256': sha(film), 'shot_count': len(shots), 'fps': fps,
            'shot_seconds': durations[0] if all(x == durations[0] for x in durations) else None,
            'shot_durations': durations, 'declared_film_frames': info['frames'], 'shots': rows,
            'limitations': ('Only three sampled frames and up to three sparse mono audio windows per shot; '
                            'differences may include re-encoding. This does not rate the film, detect every '
                            'assembly error or establish acting, motion, speech quality or lip sync.')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('film', type=Path)
    parser.add_argument('--shot', action='append', type=Path, required=True, help='ordered source; repeat once per shot')
    parser.add_argument('--shot-seconds', type=float, action='append', required=True,
                        help='one duration for every shot, or repeat once per shot in the same order')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        durations = args.shot_seconds[0] if len(args.shot_seconds) == 1 else args.shot_seconds
        result = assess(args.film, args.shot, shot_seconds=durations, fps=args.fps)
        with args.report.open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'verify-assembled-shots: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
