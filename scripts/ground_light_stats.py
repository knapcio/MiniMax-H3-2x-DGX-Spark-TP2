#!/usr/bin/env python3
"""Sample lower-frame brightness over time in a fixed-camera clip; diagnostic only."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import statistics
import subprocess
import sys

from PIL import Image, ImageStat


WIDTH, HEIGHT = 336, 192


def inspect(path: Path, top: float = 0.625) -> dict:
    if not 0 <= top < 1:
        raise ValueError('region top must be in [0, 1)')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=avg_frame_rate,nb_frames', '-of', 'json', str(path)],
                           check=True, capture_output=True, text=True, timeout=60)
    streams = json.loads(probe.stdout)['streams']
    if len(streams) != 1 or Fraction(streams[0]['avg_frame_rate']) != 24:
        raise ValueError('expected one 24 fps video stream')
    result = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-map', '0:v:0', '-an',
                             '-vf', r'select=not(mod(n\,24)),scale=336:192:flags=area,format=gray',
                             '-fps_mode', 'passthrough', '-f', 'rawvideo', '-'],
                            check=True, capture_output=True, timeout=3600)
    frame_bytes = WIDTH * HEIGHT
    if not result.stdout or len(result.stdout) % frame_bytes:
        raise ValueError('incomplete decoded frames')
    count = len(result.stdout) // frame_bytes
    frames = streams[0].get('nb_frames')
    if frames and frames != 'N/A' and count != (int(frames) + 23) // 24:
        raise ValueError('unexpected number of one-second samples')
    y = int(HEIGHT * top)
    boxes = [(0, y, WIDTH, HEIGHT)] + [(i * WIDTH // 3, y, (i + 1) * WIDTH // 3, HEIGHT) for i in range(3)]
    series = [[] for _ in boxes]
    for offset in range(0, len(result.stdout), frame_bytes):
        image = Image.frombytes('L', (WIDTH, HEIGHT), result.stdout[offset:offset + frame_bytes])
        for values, box in zip(series, boxes, strict=True):
            values.append(round(ImageStat.Stat(image.crop(box)).mean[0], 2))
    mean = series[0]
    return {
        'sample_seconds': list(range(count)),
        'ground_top_fraction': top,
        'ground_gray_mean': mean,
        'ground_thirds_gray_mean': dict(zip(('left', 'middle', 'right'), series[1:], strict=True)),
        'ground_gray_range': round(max(mean) - min(mean), 2),
        'mean_absolute_change_from_first': round(statistics.mean(abs(x - mean[0]) for x in mean), 2),
        'max_absolute_one_second_change': round(max((abs(b - a) for a, b in zip(mean, mean[1:])), default=0), 2),
        'caveat': ('One grayscale mean per second in a fixed image region; changing content and shadows both affect it. '
                   'This does not track clouds or establish shadow alignment, motion quality or absence of flicker.'),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--ground-top', type=float, default=0.625)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.video, args.ground_top), indent=2))
        return 0
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, ZeroDivisionError, json.JSONDecodeError) as exc:
        print(f'ground-light-stats: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
