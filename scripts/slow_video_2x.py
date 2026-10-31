#!/usr/bin/env python3
"""Create an opt-in, 2x-slower H3 video with interpolated frames and retimed audio.

This slows every part of the picture, not only people. Intermediate frames may
warp or smear, and retimed audio may sound unnatural. It does not establish
quality or cloud-shadow alignment. The source is never overwritten.
"""
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
import time


spec = importlib.util.spec_from_file_location('exact_duration', Path(__file__).with_name('exact_duration.py'))
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def selected_hashes(path, expression):
    filters = f"select={expression},setpts=N/(24*TB)"
    result = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(path),
                             '-map', '0:v:0', '-vf', filters, '-fps_mode', 'passthrough',
                             '-pix_fmt', 'yuv420p', '-f', 'framemd5', '-'],
                            check=True, capture_output=True, text=True, timeout=7200)
    return [line.split(',')[-1].strip() for line in result.stdout.splitlines()
            if line and not line.startswith('#')]


def audio_format(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                             'stream=channels,channel_layout', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True, timeout=60)
    streams = json.loads(result.stdout)['streams']
    if len(streams) != 1 or int(streams[0].get('channels') or 0) <= 0:
        raise ValueError('audio channel metadata unavailable')
    return streams[0]


def verify_timestamps(path, count):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                             '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True, timeout=7200)
    frames = json.loads(result.stdout)['frames']
    if len(frames) != count:
        raise ValueError('video timestamp count differs')
    try:
        errors = [abs(Fraction(item['best_effort_timestamp_time']) - Fraction(i, 24))
                  for i, item in enumerate(frames)]
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise ValueError('video frame timestamps unavailable') from exc
    if max(errors, default=Fraction(0)) > Fraction(1, 1_000_000):
        raise ValueError('video timestamps are not continuous 24 fps')


def create(source, destination):
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(os.path.abspath(Path(destination).expanduser()))
    if os.path.lexists(destination) or source == destination.resolve():
        raise ValueError('output must be a new file distinct from the input')
    before = digest(source)
    meta = exact.probe(source)
    videos = [s for s in meta['streams'] if s.get('codec_type') == 'video']
    audios = [s for s in meta['streams'] if s.get('codec_type') == 'audio']
    if len(videos) != 1 or len(audios) > 1 or len(meta['streams']) != 1 + len(audios):
        raise ValueError('expected one video, at most one audio and no other streams')
    video = videos[0]
    frames = int(video.get('nb_frames') or 0)
    width, height = int(video.get('width', 0)), int(video.get('height', 0))
    if (frames < 3 or video.get('pix_fmt') != 'yuv420p' or width <= 0 or height <= 0 or
            width % 2 or height % 2 or
            Fraction(video.get('r_frame_rate', '0/1')) != 24 or
            Fraction(video.get('avg_frame_rate', '0/1')) != 24 or
            abs(float(video.get('start_time', 0))) > 0.001 or
            video.get('sample_aspect_ratio') not in (None, '1:1', 'N/A')):
        raise ValueError('expected at least three zero-start, square-pixel YUV420 frames at 24 fps')
    source_seconds = Fraction(frames, 24)
    if audios:
        audio = audios[0]
        if (audio.get('codec_name') != 'aac' or int(audio.get('sample_rate') or 0) <= 0 or
                abs(float(audio.get('start_time', 0))) > 0.001 or not audio.get('duration') or
                abs(float(audio['duration']) - float(source_seconds)) > 0.1):
            raise ValueError('expected near-video-length, zero-start AAC audio')
        source_audio_format = audio_format(source)
    out_frames = frames * 2
    duration = float(Fraction(out_frames, 24))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.h3-slow2x-', suffix='.mp4', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    started = time.monotonic()
    try:
        # Pad two source frames before motion interpolation so the penultimate
        # intermediate frame has future context. Replace every even frame with
        # its exact source frame and repeat the final frame for the last odd one.
        filters = (
            '[0:v]split=2[raw][warp];'
            '[raw]setpts=2*(PTS-STARTPTS)[original];'
            '[warp]tpad=stop_mode=clone:stop=2,setpts=2*(PTS-STARTPTS),'
            'minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,'
            f'trim=end_frame={out_frames},setpts=N/(24*TB)[mi];'
            f"[mi][original]overlay=format=yuv420:enable='not(mod(n,2))+eq(n,{out_frames - 1})':"
            'eof_action=repeat:repeatlast=1,format=yuv420p[v]'
        )
        if audios:
            filters += f';[0:a]atempo=0.5,apad,atrim=duration={duration:.9f},asetpts=PTS-STARTPTS[a]'
        cmd = ['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-y', '-i', str(source),
               '-filter_complex', filters, '-map', '[v]']
        if audios:
            cmd += ['-map', '[a]', '-c:a', 'aac', '-b:a', '128k']
        cmd += ['-map_metadata', '0', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '0',
                '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough']
        for key, option in (('color_range', '-color_range'), ('color_space', '-colorspace'),
                            ('color_transfer', '-color_trc'), ('color_primaries', '-color_primaries')):
            if video.get(key) and video[key] != 'unknown':
                cmd += [option, video[key]]
        cmd += ['-t', f'{duration:.9f}', '-movflags', '+faststart', str(temporary)]
        subprocess.run(cmd, check=True, timeout=max(600, int(duration * 120)))
        out = exact.probe(temporary)
        result_video = [s for s in out['streams'] if s.get('codec_type') == 'video']
        result_audio = [s for s in out['streams'] if s.get('codec_type') == 'audio']
        if len(result_video) != 1 or len(result_audio) != len(audios):
            raise ValueError('unexpected output streams')
        rv = result_video[0]
        if ((rv.get('width'), rv.get('height')) != (video['width'], video['height']) or
                rv.get('pix_fmt') != 'yuv420p' or Fraction(rv.get('avg_frame_rate', '0/1')) != 24 or
                Fraction(rv.get('r_frame_rate', '0/1')) != 24 or abs(float(rv.get('start_time', 0))) > 0.001 or
                rv.get('sample_aspect_ratio') not in (None, '1:1', 'N/A') or
                int(rv.get('nb_frames') or 0) != out_frames or
                abs(float(out['format']['duration']) - duration) > 0.001):
            raise ValueError('output video geometry, frame count, rate or duration differs')
        for key in ('color_range', 'color_space', 'color_transfer', 'color_primaries'):
            if video.get(key) not in (None, 'unknown') and rv.get(key) != video[key]:
                raise ValueError('video color metadata differs')
        if audios:
            ra = result_audio[0]
            if (ra.get('codec_name') != 'aac' or ra.get('sample_rate') != audios[0].get('sample_rate') or
                    abs(float(ra.get('start_time', 0))) > 0.001 or not ra.get('duration') or
                    abs(float(ra['duration']) - duration) > 0.05):
                raise ValueError('retimed audio timing or format differs')
            if audio_format(temporary) != source_audio_format:
                raise ValueError('audio channels or layout differ')
        verify_timestamps(source, frames)
        verify_timestamps(temporary, out_frames)
        original_hashes = exact.frame_hashes(source, frames)
        even_hashes = selected_hashes(temporary, r'not(mod(n\,2))')
        final_hashes = selected_hashes(temporary, fr'eq(n\,{out_frames - 1})')
        if len(even_hashes) != frames or even_hashes != original_hashes:
            raise ValueError('an even output frame differs from its source frame')
        if final_hashes != [original_hashes[-1]]:
            raise ValueError('last interpolated frame does not repeat the last source frame')
        decode = ['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(temporary), '-map', '0:v:0']
        if audios:
            decode += ['-map', '0:a:0']
        subprocess.run(decode + ['-f', 'null', '-'], check=True, timeout=7200)
        if digest(source) != before:
            raise ValueError('source changed during processing')
        output_sha = digest(temporary)
        os.link(temporary, destination)
        return {
            'source': str(source), 'output': str(destination), 'source_sha256': before,
            'output_sha256': output_sha, 'source_frames': frames, 'output_frames': out_frames,
            'source_seconds': float(source_seconds), 'output_seconds': duration,
            'processing_seconds': round(time.monotonic() - started, 2),
            'retained_even_frames_identical': frames, 'last_frame_repeats_source': True,
            'video_timestamps': 'continuous 24 fps on source and output',
            'audio': ('time-stretched 2x and reencoded to AAC, with end padding as needed' if audios else 'none'),
            'audio_format': source_audio_format if audios else None,
            'caveat': ('All picture motion is slowed, not only people; synthesized odd frames may warp or smear. '
                       'Retimed audio may sound unnatural and does not establish event synchronization. '
                       'Source duration and narrative coverage are not increased by new model generation.'),
        }
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(create(args.source, args.output), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError) as exc:
        print(f'slow-video-2x: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
