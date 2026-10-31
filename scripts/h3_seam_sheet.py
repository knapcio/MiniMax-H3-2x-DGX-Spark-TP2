#!/usr/bin/env python3
"""Make a frame-accurate still-image overview around planned H3 handovers."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
exact = load('exact_duration_for_seam_sheet', HERE / 'exact_duration.py')
planner = load('estimate_continuation_for_seam_sheet', HERE / 'estimate_continuation.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make(source, destination, *, seconds, window=277, overlap=22, delta=3):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if source == destination or os.path.lexists(destination):
        raise ValueError('output must be new and distinct from the source')
    if not isinstance(delta, int) or isinstance(delta, bool) or not 1 <= delta <= 24:
        raise ValueError('frame offset must be an integer in 1..24')
    aligned = planner.aligned_frames(seconds)
    frames = round(seconds * 24)
    windows = planner.plan(aligned, window, overlap)
    before_hash = digest(source)
    info = exact.probe(source)
    videos = [item for item in info['streams'] if item['codec_type'] == 'video']
    if len(videos) != 1 or int(videos[0].get('nb_frames', 0)) != frames or videos[0].get('avg_frame_rate') != '24/1':
        raise ValueError('source must be an exact 24 fps clip of the requested length')
    exact.frame_hashes(source, frames)
    boundaries = [end for _, end in windows[:-1] if delta <= end and end + delta < frames]
    if not boundaries or len(boundaries) > 80:
        raise ValueError('expected 1..80 handovers with surrounding frames inside the clip')
    indices = [index for boundary in boundaries for index in (boundary - delta, boundary, boundary + delta)]
    if indices != sorted(set(indices)):
        raise ValueError('selected frame indices overlap or are out of order')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.h3-seams-', dir=destination.parent) as temp:
        root = Path(temp)
        selection = '+'.join(f'eq(n\\,{index})' for index in indices)
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(source), '-map', '0:v:0', '-an', '-vf',
                        f'select={selection},scale=320:-2', '-fps_mode', 'passthrough', str(root / 'frame-%03d.png')],
                       check=True, timeout=7200)
        images = sorted(root.glob('frame-*.png'))
        if len(images) != len(indices) or digest(source) != before_hash:
            raise ValueError('selected frame count differs or source changed')
        with Image.open(images[0]) as image:
            width, height = image.size
        canvas = Image.new('RGB', (3 * width, 72 + len(boundaries) * (height + 28)), '#111827')
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
            small = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
        except OSError:
            font = ImageFont.load_default(); small = font
        draw.text((10, 5), 'Przed połączeniem                  Granica — orientacyjnie                  Po połączeniu',
                  font=font, fill='white')
        draw.text((10, 36), 'Klatki nie oceniają płynności, migania ani zgodności cieni z chmurami.',
                  font=small, fill='#cbd5e1')
        for i, path in enumerate(images):
            row, column = divmod(i, 3)
            x, y = column * width, 72 + row * (height + 28)
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError('thumbnail dimensions differ')
                canvas.paste(image.convert('RGB'), (x, y))
            draw.text((x + 6, y + height + 4), f'{indices[i] / 24:.3f} s · klatka {indices[i]}',
                      font=small, fill='white')
        png = root / 'seams.png'; canvas.save(png)
        with Image.open(png) as image: image.verify()
        if digest(source) != before_hash:
            raise ValueError('source changed while creating the sheet')
        result = {'source': str(source), 'source_sha256': before_hash, 'output': str(destination),
                  'output_sha256': digest(png), 'source_frames': frames, 'aligned_internal_frames': aligned,
                  'window': window, 'overlap': overlap, 'boundaries': boundaries, 'sample_frames': indices,
                  'caveat': ('Boundaries are inferred from the window plan. Still frames do not establish motion, '
                             'flicker, cloud-shadow alignment or overall quality.')}
        os.link(png, destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--window', type=int, default=277)
    parser.add_argument('--overlap', type=int, default=22)
    parser.add_argument('--delta', type=int, default=3)
    args = parser.parse_args()
    try:
        print(json.dumps(make(args.source, args.output, seconds=args.seconds, window=args.window,
                              overlap=args.overlap, delta=args.delta), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError) as exc:
        print('h3-seam-sheet:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
