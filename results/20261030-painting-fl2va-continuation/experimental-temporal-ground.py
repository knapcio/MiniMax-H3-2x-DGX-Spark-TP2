#!/usr/bin/env python3
"""Experimental symmetric temporal smoothing of lower-frame lighting, not cloud tracking.

The filter adjusts only low-spatial-frequency luma in a soft ground mask. It
may suppress genuine movement or create temporal smearing; visual review is
required. The input and audio are preserved and the output must be a new file.
"""
import argparse
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageFilter


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('soft_ground', HERE / 'experimental-soft-ground.py')
soft_ground = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soft_ground)
LOW_WIDTH, LOW_HEIGHT = 168, 96


def weights(radius, sigma):
    if not isinstance(radius, int) or not 1 <= radius <= 48 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('invalid temporal filter parameters')
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    values = np.exp(-0.5 * (offsets / sigma) ** 2)
    return values / values.sum()


def smoothed_difference(fields, index, kernel):
    """Centered low-frequency temporal estimate minus the current field."""
    if fields.ndim != 3 or not 0 <= index < len(fields) or len(kernel) % 2 != 1:
        raise ValueError('invalid field sequence or kernel')
    radius = len(kernel) // 2
    result = np.zeros_like(fields[0], dtype=np.float32)
    for j, value in enumerate(kernel):
        other = min(max(index + j - radius, 0), len(fields) - 1)
        result += fields[other] * value
    return result - fields[index]


def boundary_weight(index, count, radius):
    """Fade correction to zero at the first and last frame."""
    if count < 1 or not 0 <= index < count or radius < 1:
        raise ValueError('invalid boundary position')
    return min(1.0, index / radius, (count - 1 - index) / radius)


def adjust_luma(frame, low_difference, mask, strength, limit):
    """Add only the bounded smoothed light correction, leaving chroma to the caller."""
    if frame.ndim != 2 or mask.shape != frame.shape or not 0 <= strength <= 1 or not 1 <= limit <= 32:
        raise ValueError('invalid luma, mask, strength or correction limit')
    h, w = frame.shape
    image = Image.fromarray(low_difference.astype(np.float32))
    enlarged = np.asarray(image.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
    correction = np.rint(np.clip(enlarged * strength * mask, -limit, limit)).astype(np.int16)
    return np.clip(frame.astype(np.int16) + correction, 0, 255).astype(np.uint8)


def low_fields(source, expected_frames, spatial_radius):
    if not 1 <= expected_frames <= 8000 or not 0 <= spatial_radius <= 32:
        raise ValueError('unsupported number of frames or spatial blur radius')
    fields = np.empty((expected_frames, LOW_HEIGHT, LOW_WIDTH), dtype=np.float32)
    decode = ['ffmpeg', '-v', 'error', '-xerror', '-i', str(source), '-map', '0:v:0', '-an', '-vf',
              f'scale={LOW_WIDTH}:{LOW_HEIGHT}:flags=area,format=gray', '-fps_mode', 'passthrough',
              '-pix_fmt', 'gray', '-f', 'rawvideo', 'pipe:1']
    count = 0
    for raw in soft_ground.read_frames(decode, LOW_WIDTH * LOW_HEIGHT):
        if count >= expected_frames:
            raise ValueError('decoded more frames than declared')
        image = Image.frombytes('L', (LOW_WIDTH, LOW_HEIGHT), raw)
        if spatial_radius:
            image = image.filter(ImageFilter.GaussianBlur(spatial_radius))
        fields[count] = np.asarray(image, dtype=np.float32)
        count += 1
    if count != expected_frames:
        raise ValueError('decoded frame count differs from source metadata')
    return fields


def render(source, output, *, radius=12, sigma=6, spatial_radius=4, strength=1.0, limit=6,
           start=.62, end=.77):
    source, output = Path(source).absolute(), Path(output).absolute()
    if source == output or os.path.lexists(output):
        raise ValueError('output must be a new file distinct from source')
    if not 0 <= start < end <= 1 or not 0 <= strength <= 1 or not 1 <= limit <= 32:
        raise ValueError('invalid mask or correction parameters')
    kernel = weights(radius, sigma)
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames', '-of', 'json', str(source)],
                           check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout)['streams']
    if len(streams) != 1:
        raise ValueError('expected one video stream')
    info = streams[0]
    w, h, n = int(info['width']), int(info['height']), int(info['nb_frames'])
    if (info['pix_fmt'] != 'yuv420p' or w % 2 or h % 2 or Fraction(info['avg_frame_rate']) != 24
            or Fraction(info['r_frame_rate']) != 24):
        raise ValueError('expected even-sized, constant 24 fps yuv420p video')
    fields = low_fields(source, n, spatial_radius)
    mask = np.asarray(soft_ground.lower_mask(w, h, start, end), dtype=np.float32) / 255.0
    y_bytes, frame_bytes = w * h, w * h * 3 // 2
    decode = ['ffmpeg', '-v', 'error', '-xerror', '-i', str(source), '-map', '0:v:0', '-an',
              '-fps_mode', 'passthrough', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', 'pipe:1']
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.h3-temporal-ground-', suffix='.mp4', dir=output.parent)
    os.close(fd); temp = Path(name)
    encode = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'yuv420p', '-video_size', f'{w}x{h}',
              '-framerate', '24', '-i', 'pipe:0', '-i', str(source), '-map', '0:v:0', '-map', '1:a:0?',
              '-map_metadata', '1', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '0', '-pix_fmt', 'yuv420p',
              '-c:a', 'copy', '-movflags', '+faststart', '-f', 'mp4', str(temp)]
    encoder = None
    try:
        encoder = subprocess.Popen(encode, stdin=subprocess.PIPE)
        count = 0
        for raw in soft_ground.read_frames(decode, frame_bytes):
            if count >= n:
                raise ValueError('second decode produced too many frames')
            luma = np.frombuffer(raw, dtype=np.uint8, count=y_bytes).reshape(h, w)
            delta = smoothed_difference(fields, count, kernel) * boundary_weight(count, n, radius)
            adjusted = adjust_luma(luma, delta, mask, strength, limit)
            encoder.stdin.write(adjusted.tobytes() + raw[y_bytes:])
            count += 1
        encoder.stdin.close()
        if encoder.wait() != 0 or count != n:
            raise RuntimeError('encoding failed or frame count differs')
        os.link(temp, output)
    except BaseException:
        if encoder is not None:
            try: encoder.stdin.close()
            except (BrokenPipeError, OSError, ValueError): pass
            if encoder.poll() is None: encoder.terminate()
            encoder.wait()
        raise
    finally:
        temp.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    count = render(args.source, args.output)
    print(f'experimental centered ground-light smoothing: {count} frames; max +/-6 luma; review required')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
