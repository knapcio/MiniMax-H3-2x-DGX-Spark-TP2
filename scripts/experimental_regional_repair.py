#!/usr/bin/env python3
"""Make a bounded, separate regional-flicker experiment from a verified H3 clip.

The detector is a screen for repeated changes, not a visual-quality test. The
correction uses neighboring frames and may change real lighting or motion.
"""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / 'results/20261030-painting-fl2va-continuation/experimental-regional-deflicker.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regional = load('regional_flicker', ROOT / 'scripts/regional_flicker.py')
repair = load('regional_repair', REPAIR)
verify = load('verify_luma_repair', ROOT / 'scripts/verify_luma_repair.py')
transitions = load('compare_regional_transitions', ROOT / 'scripts/compare_regional_transitions.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_info(source, expected_sha, expected_frames, width, height):
    if not source.is_file() or source.is_symlink() or digest(source) != expected_sha:
        raise ValueError('source missing, symbolic, or checksum mismatch')
    info = verify.exact.probe(source)
    videos = [x for x in info['streams'] if x['codec_type'] == 'video']
    audios = [x for x in info['streams'] if x['codec_type'] == 'audio']
    if len(videos) != 1 or len(audios) != 1 or audios[0].get('codec_name') != 'aac':
        raise ValueError('expected one video and one AAC audio stream')
    video = videos[0]
    if (video.get('pix_fmt') != 'yuv420p' or int(video.get('nb_frames', 0)) != expected_frames
            or (video.get('width'), video.get('height')) != (width, height)
            or Fraction(video['r_frame_rate']) != 24 or Fraction(video['avg_frame_rate']) != 24
            or abs(float(video.get('start_time', 0))) > .001):
        raise ValueError('source frame count, geometry, format, or rate differs')
    return info


def choose_events(frames, series):
    events = repair.regional_events_from_series(frames, series, regional.periodic_spikes)
    if any(frame < 0 or frame >= frames for frame in events):
        raise ValueError('selected frame outside source')
    if len(events) * 10 > frames * 3:
        raise ValueError('selected frames exceed 30 percent; refusing broad correction')
    return events


def build(source, expected_sha, video, report, *, expected_frames, width, height, check_bundle=lambda: None):
    source = Path(source).absolute()
    video, report = Path(video).absolute(), Path(report).absolute()
    if source in (video, report) or video == report or video.parent != report.parent:
        raise ValueError('outputs must be distinct and in the same directory')
    if any(os.path.lexists(p) for p in (video, report)):
        raise FileExistsError('output already exists')
    if shutil.disk_usage(video.parent).free < 8 * 1024**3:
        raise ValueError('less than 8 GiB free for local analysis')
    check_bundle()
    source_info(source, expected_sha, expected_frames, width, height)
    fps, frames, series = regional.inspect(source)
    if fps != 24 or frames != expected_frames:
        raise ValueError('screening frame count or rate differs')
    events = choose_events(frames, series)
    if not events:
        if digest(source) != expected_sha:
            raise ValueError('source changed during screening')
        check_bundle()
        return False
    with tempfile.TemporaryDirectory(prefix='.h3-regional-retry-', dir=video.parent) as temp:
        temporary = Path(temp) / 'candidate.mp4'
        subprocess.run([sys.executable, str(REPAIR), str(source), str(temporary), '--local-periodic',
                        '--radius', '4', '--limit', '8'], check=True, timeout=14400)
        result = verify.inspect(source, temporary, 8)
        if (result['source_sha256'] != expected_sha or result['repaired_sha256'] != digest(temporary)
                or result['frames'] != expected_frames
                or not set(result['changed_luma_frames']).issubset(events)):
            raise ValueError('repair changed an unselected frame or verification checksums differ')
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(temporary), '-map', '0:v:0',
                        '-map', '0:a:0', '-f', 'null', '-'], check=True, timeout=7200)
        comparison = transitions.compare(source, temporary)
        if (comparison['source_sha256'] != expected_sha
                or comparison['candidate_sha256'] != result['repaired_sha256']):
            raise ValueError('transition comparison checksums differ')
        comparison['candidate'] = str(video)
        check_bundle()
        if digest(source) != expected_sha or digest(temporary) != result['repaired_sha256']:
            raise ValueError('a video changed during verification')
        payload = {'source': str(source), 'candidate': str(video), 'source_sha256': expected_sha,
                   'method': 'dominant repeated regional phase; radius 4; maximum +/-8 luma',
                   'selected_frames': sorted(events), 'invariants': result, 'transition_comparison': comparison,
                   'caveat': ('Experimental; repeated changes may be real light or motion. '
                              'Neither the screen nor bounded luma proves visual quality. Playback review required.')}
        temp_report = Path(temp) / 'verification.json'
        with temp_report.open('x') as out:
            json.dump(payload, out, indent=2, sort_keys=True)
            out.write('\n'); out.flush(); os.fsync(out.fileno())
        os.link(temporary, video)
        try:
            os.link(temp_report, report)
        except BaseException:
            if os.path.samestat(video.stat(), temporary.stat()):
                video.unlink()
            raise
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('sha256')
    parser.add_argument('video', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    args = parser.parse_args()
    try:
        created = build(args.source, args.sha256, args.video, args.report, expected_frames=args.frames,
                        width=args.width, height=args.height)
        print('experimental copy and report saved' if created else 'no qualifying events; no copy created')
        return 0
    except (OSError, ValueError, KeyError, ZeroDivisionError, subprocess.SubprocessError) as exc:
        print('experimental-regional-repair:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
