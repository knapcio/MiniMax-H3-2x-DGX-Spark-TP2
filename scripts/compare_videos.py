#!/usr/bin/env python3
"""Make an explicitly labelled side-by-side comparison of exact 24 fps clips."""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont


spec = importlib.util.spec_from_file_location('exact_duration', Path(__file__).with_name('exact_duration.py'))
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_info(path, frames):
    info = exact.probe(path)
    videos = [stream for stream in info['streams'] if stream['codec_type'] == 'video']
    audios = [stream for stream in info['streams'] if stream['codec_type'] == 'audio']
    if (len(videos) != 1 or len(audios) != 1 or audios[0].get('codec_name') != 'aac'
            or int(videos[0].get('nb_frames', 0)) != frames or Fraction(videos[0]['avg_frame_rate']) != 24
            or Fraction(videos[0]['r_frame_rate']) != 24 or abs(float(videos[0].get('start_time', 0))) > .001):
        raise ValueError('expected an exact, zero-start 24 fps video with one AAC stream')
    return info


def create(left, right, destination, *, frames, left_title, right_title):
    left, right, destination = Path(left).absolute(), Path(right).absolute(), Path(destination).absolute()
    if (not isinstance(frames, int) or isinstance(frames, bool) or not 1 <= frames <= 100_000
            or left == right or destination in (left, right) or os.path.lexists(destination)):
        raise ValueError('inputs must be distinct and output new, with a valid frame count')
    if any(not isinstance(title, str) or not title.strip() or '\n' in title or len(title) > 42
           for title in (left_title, right_title)):
        raise ValueError('comparison titles must be one line and at most 42 characters')
    before = (digest(left), digest(right))
    for path in (left, right):
        source_info(path, frames)
        exact.frame_hashes(path, frames)  # decode and require continuous 24 fps timestamps
    source_audio = exact.audio_packets(left)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.h3-comparison-', dir=destination.parent) as temp:
        root = Path(temp); header = root / 'header.png'; output = root / 'comparison.mp4'
        canvas = Image.new('RGB', (1280, 66), '#111827'); draw = ImageDraw.Draw(canvas)
        font_path = '/System/Library/Fonts/Helvetica.ttc'
        try:
            small = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
        except OSError:
            small = ImageFont.load_default()
        def fitted_font(title):
            for size in range(24, 11, -1):
                try: font = ImageFont.truetype(font_path, size)
                except OSError: font = ImageFont.load_default()
                if draw.textbbox((0, 0), title, font=font)[2] <= 608:
                    return font
            raise ValueError('comparison title is too wide')
        draw.text((16, 5), left_title, font=fitted_font(left_title), fill='white')
        draw.text((656, 5), right_title, font=fitted_font(right_title), fill='white')
        draw.text((16, 39), 'Dźwięk z lewego filmu. Porównanie eksperymentalne.', font=small, fill='#cbd5e1')
        canvas.save(header)
        scale = ('scale=640:360:force_original_aspect_ratio=decrease:force_divisible_by=2,'
                 'pad=640:360:(ow-iw)/2:(oh-ih)/2:color=black')
        filters = (f'[0:v]{scale}[a];[1:v]{scale}[b];[a][b]hstack=inputs=2,'
                   'pad=1280:426:0:66:color=black[base];[base][2:v]overlay=0:0:shortest=1[v]')
        command = ['ffmpeg', '-v', 'error', '-xerror', '-i', str(left), '-i', str(right), '-loop', '1',
                   '-framerate', '24', '-i', str(header), '-filter_complex', filters, '-map', '[v]', '-map', '0:a:0',
                   '-frames:v', str(frames), '-fps_mode', 'passthrough', '-c:v', 'libx264', '-preset', 'veryfast',
                   '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-t', f'{frames / 24:.9f}',
                   '-movflags', '+faststart', str(output)]
        subprocess.run(command, check=True, timeout=14_400)
        info = source_info(output, frames)
        video = next(stream for stream in info['streams'] if stream['codec_type'] == 'video')
        if (video['width'], video['height']) != (1280, 426):
            raise ValueError('comparison geometry differs')
        if abs(float(info['format']['duration']) - frames / 24) > .001:
            raise ValueError('comparison duration differs')
        exact.frame_hashes(output, frames)
        if exact.audio_packets(output) != source_audio:
            raise ValueError('comparison does not preserve left AAC packets and timing')
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(output), '-map', '0:a:0', '-f', 'null', '-'],
                       check=True, timeout=3600)
        if (digest(left), digest(right)) != before:
            raise ValueError('a source changed during comparison')
        result = {'path': str(destination), 'sha256': digest(output), 'frames': frames,
                  'left_sha256': before[0], 'right_sha256': before[1],
                  'audio': 'AAC packets and timing copied from the left input',
                  'caveat': 'The side-by-side is scaled and recompressed; assess both original clips too.'}
        os.link(output, destination)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--left-title', required=True)
    parser.add_argument('--right-title', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(create(args.left, args.right, args.output, frames=args.frames,
                                left_title=args.left_title, right_title=args.right_title), indent=2))
        return 0
    except (OSError, ValueError, KeyError, ZeroDivisionError, subprocess.SubprocessError) as exc:
        print('compare-videos:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
