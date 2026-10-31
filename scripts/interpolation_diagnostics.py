#!/usr/bin/env python3
"""Screen interpolated video triplets for unusual frame-to-frame luma changes.

This is a review aid. A high score can be real motion, and small local warps
can be missed; the screen is not a visual-quality verdict.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from PIL import Image, ImageChops, ImageStat


WIDTH, HEIGHT = 336, 192
BOXES = {
    'whole': (0, 0, WIDTH, HEIGHT),
    'upper-left': (0, 0, 112, 96), 'upper-middle': (112, 0, 224, 96),
    'upper-right': (224, 0, 336, 96), 'lower-left': (0, 96, 112, 192),
    'lower-middle': (112, 96, 224, 192), 'lower-right': (224, 96, 336, 192),
    'ground': (0, 120, WIDTH, HEIGHT),
}


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def triplet(first, middle, last):
    if (first.size != (WIDTH, HEIGHT) or middle.size != first.size or last.size != first.size or
            any(image.mode != 'L' for image in (first, middle, last))):
        raise ValueError('unexpected diagnostic frame dimensions')
    differences = [ImageChops.difference(first, middle), ImageChops.difference(middle, last),
                   ImageChops.difference(first, last)]
    out = {}
    for name, box in BOXES.items():
        a, b, direct = (ImageStat.Stat(image.crop(box)).mean[0] for image in differences)
        total = a + b
        out[name] = {
            'first_half': a, 'second_half': b, 'direct': direct,
            'excess': max(0.0, total - direct),
            'imbalance': abs(a - b) / total if total > 0 else 0.0,
        }
    return out


def inspect(video, *, top=10):
    video = Path(video).expanduser().resolve(strict=True)
    if not 1 <= top <= 100:
        raise ValueError('top must be in 1..100')
    before = digest(video)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=avg_frame_rate,nb_frames', '-of', 'json', str(video)],
                           check=True, capture_output=True, text=True, timeout=60)
    streams = json.loads(probe.stdout)['streams']
    if len(streams) != 1 or Fraction(streams[0]['avg_frame_rate']) != 24:
        raise ValueError('expected one 24 fps video stream')
    count = int(streams[0].get('nb_frames') or 0)
    if count < 6 or count % 2:
        raise ValueError('expected an even output frame count of at least six')
    cmd = ['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(video), '-map', '0:v:0', '-an',
           '-vf', f'scale={WIDTH}:{HEIGHT}:flags=area,format=gray',
           '-fps_mode', 'passthrough', '-pix_fmt', 'gray', '-f', 'rawvideo', 'pipe:1']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = WIDTH * HEIGHT
    def read():
        raw = proc.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            raise ValueError('incomplete diagnostic video frame')
        return Image.frombytes('L', (WIDTH, HEIGHT), raw)
    records = []
    try:
        first = read()
        for frame in range(0, count - 2, 2):
            middle, last = read(), read()
            values = triplet(first, middle, last)
            region = max(values, key=lambda name: values[name]['excess'])
            records.append({'middle_frame': frame + 1, 'seconds': (frame + 1) / 24,
                            'largest_region': region, 'largest_excess': values[region]['excess'],
                            'regions': values})
            first = last
        read()  # final odd frame; there is no later source frame to compare
        if proc.stdout.read(1):
            raise ValueError('more decoded frames than declared')
        if proc.wait() != 0:
            raise ValueError('video decode failed: ' + proc.stderr.read().decode(errors='replace').strip())
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.stdout.close()
        proc.wait()
        proc.stderr.close()
    if digest(video) != before:
        raise ValueError('video changed during analysis')
    summaries = {}
    for name in BOXES:
        rows = [record['regions'][name] for record in records]
        summaries[name] = {
            'median_first_half': statistics.median(row['first_half'] for row in rows),
            'median_second_half': statistics.median(row['second_half'] for row in rows),
            'median_direct': statistics.median(row['direct'] for row in rows),
            'median_excess': statistics.median(row['excess'] for row in rows),
            'max_excess': max(row['excess'] for row in rows),
        }
    return {
        'video': str(video), 'video_sha256': before, 'frames': count, 'triplets': len(records),
        'diagnostic_size': [WIDTH, HEIGHT], 'region_summaries': summaries,
        'highest_excess_triplets': sorted(records, key=lambda row: row['largest_excess'], reverse=True)[:top],
        'caveat': ('Excess is mean absolute change from first to middle plus middle to last minus the direct '
                   'first-to-last change, on downscaled grayscale regions. High values can be legitimate motion, '
                   'and small local interpolation warps may be missed. This is not a quality or causality verdict.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.video, top=args.top), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError) as exc:
        print(f'interpolation-diagnostics: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
