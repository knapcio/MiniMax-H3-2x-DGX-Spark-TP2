#!/usr/bin/env python3
"""Create a separate, frame-identical video with bounded per-line audio gains.

This is an optional listening experiment. RMS matching does not establish
perceived loudness, naturalness, voice identity or accurate lip sync.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import compare_driving_audio as audio
import exact_duration


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def decode_stereo(path: Path, seconds: float, rate: int = 48000) -> np.ndarray:
    result = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(path), '-map', '0:a:0',
                             '-vn', '-ac', '2', '-ar', str(rate), '-t', str(seconds), '-f', 'f32le', 'pipe:1'],
                            check=True, capture_output=True, timeout=1800)
    if len(result.stdout) % 8 or not result.stdout:
        raise ValueError('missing or incomplete stereo audio')
    data = np.frombuffer(result.stdout, dtype='<f4').reshape(-1, 2)
    if not np.isfinite(data).all():
        raise ValueError('nonfinite audio')
    return data


def balance(video: Path, intervals: list[dict], destination: Path, *, target_rms: float = .09,
            max_gain: float = 2.0, line_peak: float = .65, base_gain: float = .7,
            margin: float = .12, ramp: float = .08) -> dict:
    video = video.resolve(strict=True); destination = Path(os.path.abspath(destination.expanduser()))
    if video == destination or os.path.lexists(destination):
        raise ValueError('output must be new and distinct from input')
    numbers = (target_rms, max_gain, line_peak, base_gain, margin, ramp)
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers) or
            not .01 <= target_rms <= .2 or not 1 <= max_gain <= 3 or not .1 <= line_peak <= .7 or
            not .1 <= base_gain <= 1 or not 0 <= margin <= 1 or not .01 <= ramp <= .5 or
            not isinstance(intervals, list) or not intervals or len(intervals) > 1000):
        raise ValueError('invalid intervals or gain settings')
    meta = exact_duration.probe(video); videos = [s for s in meta['streams'] if s['codec_type'] == 'video']
    if len(videos) != 1:
        raise ValueError('expected one video stream')
    v = videos[0]; frames = int(v.get('nb_frames') or 0)
    if (frames < 1 or Fraction(v['avg_frame_rate']) != 24 or Fraction(v['r_frame_rate']) != 24 or
            v.get('pix_fmt') != 'yuv420p' or abs(float(v.get('start_time', 0))) > .001):
        raise ValueError('video must be zero-start yuv420p at 24 fps')
    seconds = frames / 24
    if seconds > 600:
        raise ValueError('experiment supports at most 600 seconds')
    timing = audio.track_timing(video)
    if timing['audio_relative_to_timeline_s'] is None or abs(timing['audio_relative_to_timeline_s']) > .02:
        raise ValueError('audio start is unknown or offset')
    lines = []
    for x in intervals:
        if not isinstance(x, dict):
            raise ValueError('each interval must be an object')
        start, end = x.get('start'), x.get('end')
        if (type(start) not in (int, float) or type(end) not in (int, float) or not math.isfinite(start) or
                not math.isfinite(end) or not 0 <= start < end <= seconds or end - start < .5):
            raise ValueError('invalid timed line')
        lines.append((float(start), float(end), x.get('speaker')))
    lines.sort(key=lambda x: x[0])
    if any(a[1] + 2 * margin > b[0] for a, b in zip(lines, lines[1:])):
        raise ValueError('line intervals are too close for independent gain ramps')
    original_sha = sha(video); rate = 48000; data = decode_stereo(video, seconds + .1, rate)
    target_samples = round(seconds * rate)
    if len(data) < target_samples - round(.1 * rate):
        raise ValueError('audio is shorter than the video by more than 100 ms')
    padded = np.zeros((target_samples, 2), dtype=np.float32)
    padded[:min(len(data), target_samples)] = data[:target_samples]
    envelope = np.full(target_samples, base_gain, dtype=np.float32); measurements = []
    for start, end, speaker in lines:
        a, b = round(start * rate), round(end * rate)
        signal = padded[a:b].astype(np.float64); rms = float(np.sqrt(np.mean(signal * signal)))
        peak = float(np.max(np.abs(signal)))
        if rms < .001 or peak < .005:
            raise ValueError('line has insufficient audio energy to balance')
        gain = min(target_rms / rms, max_gain, line_peak / peak)
        if gain < .1:
            raise ValueError('line requires an unexpectedly large reduction')
        left = max(0, a - round(margin * rate)); right = min(target_samples, b + round(margin * rate))
        fade = min(round(ramp * rate), a - left, right - b)
        envelope[left:a] = base_gain
        if fade: envelope[a-fade:a] = np.linspace(base_gain, gain, fade, endpoint=False)
        envelope[a:b] = gain
        if fade: envelope[b:b+fade] = np.linspace(gain, base_gain, fade, endpoint=False)
        measurements.append({'speaker': speaker, 'start_s': start, 'end_s': end,
                             'input_rms': round(rms, 6), 'input_peak': round(peak, 6), 'gain': round(gain, 6)})
    processed = padded * envelope[:, None]
    if not np.isfinite(processed).all():
        raise ValueError('invalid processed samples')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.dialogue-balance-', suffix='.mp4', dir=destination.parent)
    os.close(fd); temporary = Path(name)
    try:
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-y', '-i', str(video),
                        '-f', 'f32le', '-ar', str(rate), '-ac', '2', '-i', 'pipe:0', '-map', '0:v:0', '-map', '1:a:0',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', '-ar', str(rate), '-t', f'{seconds:.9f}',
                        '-movflags', '+faststart', str(temporary)], input=processed.astype('<f4').tobytes(),
                       check=True, capture_output=True, timeout=10800)
        output = exact_duration.probe(temporary)
        out_v = [x for x in output['streams'] if x['codec_type'] == 'video']; out_a = [x for x in output['streams'] if x['codec_type'] == 'audio']
        if (len(out_v) != 1 or len(out_a) != 1 or int(out_v[0].get('nb_frames') or 0) != frames or
                (out_v[0]['width'], out_v[0]['height'], out_v[0]['pix_fmt']) != (v['width'], v['height'], v['pix_fmt']) or
                Fraction(out_v[0]['avg_frame_rate']) != 24 or abs(float(out_v[0].get('start_time', 0))) > .001 or
                abs(float(output['format']['duration']) - seconds) > .04 or float(out_a[0].get('duration') or 0) < seconds - .04):
            raise ValueError('output timing or stream properties differ')
        for key in ('sample_aspect_ratio', 'color_range', 'color_space', 'color_transfer', 'color_primaries'):
            value = v.get(key)
            if value not in (None, 'N/A', 'unknown', 'unspecified') and out_v[0].get(key) != value:
                raise ValueError('video metadata changed: ' + key)
        if exact_duration.frame_hashes(video, frames) != exact_duration.frame_hashes(temporary, frames):
            raise ValueError('decoded video frame or timestamp changed')
        output_timing = audio.track_timing(temporary)
        if output_timing['audio_relative_to_timeline_s'] is None or abs(output_timing['audio_relative_to_timeline_s']) > .02:
            raise ValueError('output audio start differs')
        mono = audio.decode(temporary, rate, seconds + .1)
        mono_peak = float(np.max(np.abs(mono)))
        if mono_peak > .98:
            raise ValueError('decoded mono sample peak exceeds 0.98; no output published')
        for row in measurements:
            a, b = round(row['start_s'] * rate), round(row['end_s'] * rate)
            sample = mono[a:min(b,len(mono))].astype(np.float64)
            if not len(sample): raise ValueError('missing output audio interval')
            row['output_mono_rms'] = round(float(np.sqrt(np.mean(sample * sample))), 6)
        if sha(video) != original_sha:
            raise ValueError('input changed during processing')
        result = {'input_sha256': original_sha, 'output_sha256': sha(temporary), 'seconds': seconds, 'frames': frames,
                  'decoded_video_frames_identical': True, 'output_mono_sample_peak': round(mono_peak, 6),
                  'target_stereo_rms': target_rms, 'base_gain': base_gain, 'lines': measurements,
                  'limitations': 'Experimental relative RMS gains and decoded sample peaks are not perceptual loudness, true-peak, lip-sync or quality verdicts.'}
        os.link(temporary, destination)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--intervals', type=Path, required=True); p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    try:
        if os.path.lexists(args.report): raise FileExistsError('report exists')
        value = json.loads(args.intervals.read_text()); intervals = value.get('dialogue') if isinstance(value, dict) else value
        result = balance(args.video, intervals, args.output)
        try:
            with args.report.open('x') as f: json.dump(result, f, indent=2, ensure_ascii=False); f.write('\n')
        except BaseException:
            args.output.unlink(missing_ok=True); raise
        print(f"saved {args.output}: {result['frames']} identical decoded video frames; experimental audio balance")
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'balance-dialogue-audio: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
