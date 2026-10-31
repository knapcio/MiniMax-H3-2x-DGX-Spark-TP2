#!/usr/bin/env python3
"""Verify decoded-frame, chroma and audio invariants of an experimental luma repair."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageChops


spec = importlib.util.spec_from_file_location('exact_duration', Path(__file__).with_name('exact_duration.py'))
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)


def compare_frame(left: bytes, right: bytes, width: int, height: int, unchanged_above: int = 0) -> tuple[bool, int]:
    y_size = width * height
    if len(left) != y_size * 3 // 2 or len(right) != len(left):
        raise ValueError('incomplete YUV420 frame')
    if left[y_size:] != right[y_size:]:
        raise ValueError('chroma bytes changed')
    if not 0 <= unchanged_above <= height:
        raise ValueError('invalid protected row count')
    if left[:width * unchanged_above] != right[:width * unchanged_above]:
        raise ValueError('luma changed in protected rows')
    if left[:y_size] == right[:y_size]:
        return False, 0
    a = Image.frombytes('L', (width, height), left[:y_size])
    b = Image.frombytes('L', (width, height), right[:y_size])
    return True, ImageChops.difference(a, b).getextrema()[1]


def inspect(source: Path, repaired: Path, limit: int = 8, unchanged_above: int = 0) -> dict:
    if source.resolve() == repaired.resolve() or not 1 <= limit <= 32:
        raise ValueError('distinct files and a luma limit in 1..32 are required')
    a, b = exact.probe(source), exact.probe(repaired)
    def video(meta):
        items = [s for s in meta['streams'] if s['codec_type'] == 'video']
        if len(items) != 1 or items[0].get('pix_fmt') != 'yuv420p':
            raise ValueError('expected one YUV420 video stream')
        return items[0]
    va, vb = video(a), video(b)
    keys = ('width', 'height', 'r_frame_rate', 'avg_frame_rate', 'nb_frames', 'start_time',
            'sample_aspect_ratio', 'color_range', 'color_space', 'color_transfer', 'color_primaries')
    if any(va.get(key) != vb.get(key) for key in keys):
        raise ValueError('video format, timing or color metadata differ')
    width, height = int(va['width']), int(va['height'])
    if width % 2 or height % 2:
        raise ValueError('expected even video dimensions')
    if not 0 <= unchanged_above <= height:
        raise ValueError('invalid protected row count')
    if any(len([s for s in meta['streams'] if s['codec_type'] == 'audio' and s.get('codec_name') == 'aac']) != 1
           or len([s for s in meta['streams'] if s['codec_type'] == 'audio']) != 1 for meta in (a, b)):
        raise ValueError('expected one AAC stream in each file')
    if exact.audio_packets(source) != exact.audio_packets(repaired):
        raise ValueError('audio packet data or timing differ')
    def frame_times(path):
        proc = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                               '-show_entries', 'frame=best_effort_timestamp_time,duration_time,pkt_duration_time',
                               '-of', 'json', str(path)], check=True, capture_output=True, text=True)
        frames = json.loads(proc.stdout)['frames']
        times = [(f.get('best_effort_timestamp_time'), f.get('duration_time', f.get('pkt_duration_time')))
                 for f in frames]
        if any(not t[0] or t[0] == 'N/A' for t in times):
            raise ValueError('video frame timestamps unavailable')
        return times
    if frame_times(source) != frame_times(repaired):
        raise ValueError('video frame timestamps or durations differ')
    def decoder(path):
        return subprocess.Popen(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-map', '0:v:0',
                                 '-an', '-fps_mode', 'passthrough', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    left, right = decoder(source), decoder(repaired)
    frame_size = width * height * 3 // 2
    changed, maximum, count = [], 0, 0
    try:
        while True:
            x, y = left.stdout.read(frame_size), right.stdout.read(frame_size)
            if not x and not y:
                break
            differs, max_delta = compare_frame(x, y, width, height, unchanged_above)
            if differs:
                changed.append(count)
                maximum = max(maximum, max_delta)
                if maximum > limit:
                    raise ValueError('luma change exceeds the limit')
            count += 1
        codes = (left.wait(), right.wait())
        if any(codes) or count != int(va['nb_frames']):
            details = (left.stderr.read() + right.stderr.read()).decode(errors='replace').strip()
            raise ValueError('decode failed or frame count differs' + (': ' + details if details else ''))
    finally:
        for process in (left, right):
            if process.poll() is None:
                process.terminate()
            process.stdout.close()
            process.wait()
            process.stderr.close()
    def digest(path):
        with path.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    return {'source_sha256': digest(source), 'repaired_sha256': digest(repaired), 'frames': count,
            'changed_luma_frames': changed, 'max_luma_change': maximum,
            'chroma_identical': True, 'video_timing_identical': True, 'aac_packets_identical': True,
            'unchanged_top_luma_rows': unchanged_above,
            'caveat': 'Luma edits can still alter real motion and details; this check does not establish visual quality.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('repaired', type=Path)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--unchanged-above', type=int, default=0, metavar='ROWS',
                        help='require the top ROWS of every decoded luma plane to be unchanged')
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.source, args.repaired, args.limit, args.unchanged_above), indent=2))
        return 0
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, ZeroDivisionError) as exc:
        print(f'verify-luma-repair: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
