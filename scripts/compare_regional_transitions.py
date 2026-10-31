#!/usr/bin/env python3
"""Compare signed regional brightness changes at repeated source-frame spikes."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys

from PIL import Image, ImageStat


spec = importlib.util.spec_from_file_location('regional_flicker', Path(__file__).with_name('regional_flicker.py'))
regional = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regional)


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def shape(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                             'stream=width,height,r_frame_rate,avg_frame_rate,start_time,nb_frames', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)['streams']
    if len(streams) != 1:
        raise ValueError('expected one video stream')
    info = streams[0]
    if (Fraction(info['r_frame_rate']) != 24 or Fraction(info['avg_frame_rate']) != 24 or
            abs(float(info.get('start_time', 0))) > .001 or int(info['nb_frames']) < 2):
        raise ValueError('expected a zero-start, constant 24 fps video with a frame count')
    return int(info['width']), int(info['height']), int(info['nb_frames'])


def signed_changes(path, targets, expected_frames):
    targets = sorted(set(targets))
    if any(frame <= 0 or frame >= expected_frames for frame in targets):
        raise ValueError('target frame is outside the video')
    needed = set(targets) | {frame - 1 for frame in targets}
    command = ['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-map', '0:v:0', '-an', '-vf',
               f'scale={regional.WIDTH}:{regional.HEIGHT}:flags=area,format=gray', '-pix_fmt', 'gray',
               '-fps_mode', 'passthrough', '-f', 'rawvideo', 'pipe:1']
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    means = {}
    count = 0
    try:
        while True:
            raw = proc.stdout.read(regional.WIDTH * regional.HEIGHT)
            if not raw:
                break
            if len(raw) != regional.WIDTH * regional.HEIGHT:
                raise ValueError('incomplete decoded frame')
            if count in needed:
                frame = Image.frombytes('L', (regional.WIDTH, regional.HEIGHT), raw)
                means[count] = [ImageStat.Stat(frame.crop(box)).mean[0] for box in regional.BOXES]
            count += 1
        if proc.wait() != 0 or count != expected_frames or set(means) != needed:
            raise ValueError('decoded frame count or target frames differ')
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    return {frame: [current - previous for current, previous in zip(means[frame], means[frame - 1], strict=True)]
            for frame in targets}


def compare(source, candidate, min_regions=3):
    source, candidate = Path(source), Path(candidate)
    if not 1 <= min_regions <= len(regional.BOXES):
        raise ValueError('min_regions must be between 1 and 6')
    before_hash, after_hash = digest(source), digest(candidate)
    source_shape, candidate_shape = shape(source), shape(candidate)
    if source_shape != candidate_shape:
        raise ValueError('source and candidate geometry or frame counts differ')
    _fps, frames, series = regional.inspect(source)
    if frames != source_shape[2]:
        raise ValueError('source frame count differs')
    counts = Counter()
    for changes in series:
        detected = {frame for group in regional.periodic_spikes(changes) for frame, _value, _baseline in group}
        counts.update(detected)
    targets = sorted(frame for frame, count in counts.items() if count >= min_regions)
    source_changes = signed_changes(source, targets, frames)
    candidate_changes = signed_changes(candidate, targets, frames)
    if digest(source) != before_hash or digest(candidate) != after_hash:
        raise ValueError('a video changed during analysis')
    rows = [{'frame': frame, 'seconds': frame / 24, 'regions_detected': counts[frame],
             'source_signed_gray': dict(zip(regional.LABELS, source_changes[frame], strict=True)),
             'candidate_signed_gray': dict(zip(regional.LABELS, candidate_changes[frame], strict=True))}
            for frame in targets]
    old_values = [x for values in source_changes.values() for x in values]
    new_values = [x for values in candidate_changes.values() for x in values]
    return {'source': str(source), 'source_sha256': before_hash, 'candidate': str(candidate),
            'candidate_sha256': after_hash, 'frames': frames, 'min_regions': min_regions,
            'source_negative_transitions': sum(x < 0 for x in old_values),
            'candidate_negative_transitions': sum(x < 0 for x in new_values),
            'source_median_signed_gray': statistics.median(old_values) if old_values else None,
            'candidate_median_signed_gray': statistics.median(new_values) if new_values else None,
            'selected_transitions': rows,
            'caveat': ('Signed means are sampled from six grayscale regions at repeated source changes. '
                       'They can include actual motion or light, do not prove a decoder cause, and do not establish visual quality.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--min-regions', type=int, default=3)
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.source, args.candidate, args.min_regions), indent=2))
        return 0
    except (OSError, ValueError, KeyError, ZeroDivisionError, subprocess.SubprocessError) as exc:
        print('compare-regional-transitions:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
