#!/usr/bin/env python3
"""Screen a join between two clips using decoded RGB frames near the boundary.

This flags large visual changes for review; an intentional cut, camera motion,
a dissolve, or a gradual change can make this diagnostic misleading. It does
not establish narrative continuity, identity, acting or image quality.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

WIDTH, HEIGHT = 320, 180


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def probe(path: Path) -> dict:
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                             'stream=width,height,avg_frame_rate,nb_frames', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True, timeout=60)
    rows = json.loads(result.stdout)['streams']
    if len(rows) != 1:
        raise ValueError('expected exactly one video stream')
    row = rows[0]
    return {'width': int(row['width']), 'height': int(row['height']),
            'fps': Fraction(row['avg_frame_rate']), 'frames': int(row['nb_frames'])}


def read_frames(path: Path, start: int, count: int) -> np.ndarray:
    end = start + count - 1
    video_filter = f'select=between(n\\,{start}\\,{end}),scale={WIDTH}:{HEIGHT}:flags=area,format=rgb24'
    result = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(path), '-vf', video_filter,
                             '-fps_mode', 'passthrough', '-frames:v', str(count), '-f', 'rawvideo', 'pipe:1'],
                            check=True, capture_output=True, timeout=1800)
    expected = count * HEIGHT * WIDTH * 3
    if len(result.stdout) != expected:
        raise ValueError('could not decode the expected number of frames')
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(count, HEIGHT, WIDTH, 3)


def assess(before: Path, after: Path, *, frames: int = 12) -> tuple[dict, np.ndarray]:
    if type(frames) is not int or not 2 <= frames <= 120:
        raise ValueError('frames must be an integer in 2..120')
    a, b = probe(before), probe(after)
    if ((a['width'], a['height'], a['fps']) != (b['width'], b['height'], b['fps']) or
            a['fps'] <= 0 or min(a['frames'], b['frames']) < frames):
        raise ValueError('clip geometry/rate differs or a clip is too short')
    left = read_frames(before, a['frames'] - frames, frames).astype(np.float32)
    right = read_frames(after, 0, frames).astype(np.float32)
    boundary = float(np.mean(np.abs(left[-1] - right[0])))
    local = [*np.mean(np.abs(np.diff(left, axis=0)), axis=(1, 2, 3)),
             *np.mean(np.abs(np.diff(right, axis=0)), axis=(1, 2, 3))]
    median = float(statistics.median(local))
    result = {'before_sha256': sha(before), 'after_sha256': sha(after), 'fps': str(a['fps']),
              'sample_frames_each_side': frames, 'boundary_rgb_mae_255': round(boundary, 4),
              'local_median_rgb_mae_255': round(median, 4), 'boundary_over_local_median_floor_1': round(boundary / max(1, median), 3),
              'attention_suggested': boundary >= 15 and boundary / max(1, median) >= 3,
              'limitations': ('A large jump can be an intentional cut or legitimate motion; a low jump does not '
                              'prove continuous action, faces, story or stable quality.')}
    sheet = np.stack([left[max(0, frames - 3)], left[-2], left[-1], right[0], right[1], right[min(2, frames - 1)]])
    return result, sheet.astype(np.uint8)


def write_sheet(frames: np.ndarray, destination: Path) -> None:
    canvas = Image.new('RGB', (3 * WIDTH, 2 * (HEIGHT + 24)), (18, 23, 35)); draw = ImageDraw.Draw(canvas)
    labels = ('before -3', 'before -2', 'before -1', 'after 0', 'after 1', 'after 2')
    for index, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        x, y = index % 3 * WIDTH, index // 3 * (HEIGHT + 24)
        canvas.paste(Image.fromarray(frame), (x, y)); draw.text((x + 5, y + HEIGHT + 4), label, fill='white')
    canvas.save(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path); parser.add_argument('after', type=Path)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--report', type=Path, required=True); parser.add_argument('--contact', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.report.exists() or args.contact.exists() or args.report == args.contact:
            raise FileExistsError('output already exists or paths are the same')
        result, sheet = assess(args.before, args.after, frames=args.frames)
        write_sheet(sheet, args.contact)
        try:
            with args.report.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
        except BaseException:
            args.contact.unlink(missing_ok=True); raise
        print('join screen:', 'review suggested' if result['attention_suggested'] else 'no large frame jump detected')
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'screen-segment-join: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
