#!/usr/bin/env python3
"""Make a separate video with its driving audio, verifying every video frame.

This is explicit post-processing. It does not establish lip synchronization,
correct speaker identity, intelligibility or the quality of the source audio.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import exact_duration
import compare_driving_audio as audio
import audio_frame_timing


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def replace(video: Path, source: Path, destination: Path) -> dict:
    video = video.resolve(strict=True); source = source.resolve(strict=True)
    destination = Path(os.path.abspath(destination.expanduser()))
    if video == source or destination in (video, source) or os.path.lexists(destination):
        raise ValueError('destination must be a new file distinct from both inputs')
    video_sha, source_sha = sha(video), sha(source)
    metadata = exact_duration.probe(video)
    videos = [x for x in metadata['streams'] if x['codec_type'] == 'video']
    if len(videos) != 1:
        raise ValueError('input requires exactly one video stream')
    stream = videos[0]
    fps = Fraction(stream['avg_frame_rate']); frames = int(stream.get('nb_frames') or 0)
    if (fps != 24 or Fraction(stream['r_frame_rate']) != 24 or frames <= 0 or
            stream.get('pix_fmt') != 'yuv420p' or abs(float(stream.get('start_time', 0))) > .001):
        raise ValueError('input must be zero-start yuv420p 24 fps with a known frame count')
    duration = frames / 24
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.driving-audio-', suffix='.mp4', dir=destination.parent)
    os.close(fd); temporary = Path(name)
    snapshot_dir = None
    try:
        snapshot_dir = Path(tempfile.mkdtemp(prefix='.driving-source-', dir=destination.parent))
        source_snapshot = snapshot_dir / ('source' + source.suffix)
        shutil.copyfile(source, source_snapshot)
        if sha(source_snapshot) != source_sha:
            raise ValueError('driving audio changed while taking its snapshot')
        # Validate and transcode the same private bytes. An original file that
        # changes and is restored while ffmpeg runs cannot alter this input.
        source_timing = audio.track_timing(source_snapshot)
        source_frames = audio_frame_timing.inspect(source_snapshot)
        source_start = audio_frame_timing.relative_start(source_frames, source_timing)
        if source_start is None or abs(source_start) > .02:
            raise ValueError('driving audio timestamps are discontinuous, unknown or offset by more than 20 ms')
        filt = f'[1:a:0]aresample=32000,apad=whole_dur={duration:.9f},atrim=duration={duration:.9f},asetpts=N/SR/TB[a]'
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-y', '-i', str(video), '-i', str(source_snapshot),
                        '-filter_complex', filt, '-map', '0:v:0', '-map', '[a]', '-c:v', 'copy',
                        '-c:a', 'aac', '-b:a', '128k', '-ar', '32000', '-t', f'{duration:.9f}',
                        '-movflags', '+faststart', str(temporary)], check=True, timeout=10800)
        output = exact_duration.probe(temporary)
        out_v = [x for x in output['streams'] if x['codec_type'] == 'video']
        out_a = [x for x in output['streams'] if x['codec_type'] == 'audio']
        if (len(out_v) != 1 or len(out_a) != 1 or int(out_v[0].get('nb_frames') or 0) != frames or
                (out_v[0]['width'], out_v[0]['height'], out_v[0]['pix_fmt']) !=
                (stream['width'], stream['height'], stream['pix_fmt']) or
                Fraction(out_v[0]['avg_frame_rate']) != 24 or abs(float(out_v[0].get('start_time', 0))) > .001 or
                abs(float(output['format']['duration']) - duration) > .04 or
                float(out_a[0].get('duration') or 0) < duration - .04):
            raise ValueError('replacement media structure differs')
        for key in ('sample_aspect_ratio', 'color_range', 'color_space', 'color_transfer', 'color_primaries'):
            value = stream.get(key)
            if value not in (None, 'N/A', 'unknown', 'unspecified') and out_v[0].get(key) != value:
                raise ValueError('video metadata changed: ' + key)
        if exact_duration.frame_hashes(video, frames) != exact_duration.frame_hashes(temporary, frames):
            raise ValueError('a decoded video frame or timestamp changed')
        timing = audio.track_timing(temporary)
        output_frames = audio_frame_timing.inspect(temporary)
        output_start = audio_frame_timing.relative_start(output_frames, timing)
        if output_start is None or abs(output_start) > .02:
            raise ValueError('replacement audio timestamps are discontinuous, unknown or not aligned')
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(temporary), '-map', '0:v:0',
                        '-map', '0:a:0', '-f', 'null', '-'], check=True, timeout=3600)
        if sha(video) != video_sha or sha(source) != source_sha or sha(source_snapshot) != source_sha:
            raise ValueError('an input or its snapshot changed during processing')
        result = {'input_video_sha256': video_sha, 'driving_audio_sha256': source_sha, 'output_sha256': sha(temporary),
                  'frames': frames, 'seconds': duration, 'decoded_video_frames_identical': True,
                  'audio_timing': timing, 'source_audio_frame_timing': source_frames,
                  'output_audio_frame_timing': output_frames,
                  'audio': 'source transcoded to AAC; replaces H3-generated audio',
                  'limitations': 'Video frame identity does not prove intelligibility, identity or lip sync.'}
        os.link(temporary, destination)
        return result
    finally:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            if snapshot_dir is not None:
                shutil.rmtree(snapshot_dir)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path); p.add_argument('audio', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    try:
        if os.path.lexists(args.report):
            raise FileExistsError('report already exists')
        result = replace(args.video, args.audio, args.output)
        try:
            with args.report.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
        except BaseException:
            args.output.unlink(missing_ok=True); raise
        print(f"saved {args.output}: {result['frames']} identical decoded video frames, replacement AAC audio")
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'replace-driving-audio: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
