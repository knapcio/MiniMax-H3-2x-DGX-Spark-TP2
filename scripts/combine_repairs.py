#!/usr/bin/env python3
"""Combine two aligned H3 repairs at a boundary where neighboring frames match.

Both videos must have the same constant 24 fps geometry and identical AAC packets.
The result is a new losslessly encoded copy, checked against every selected frame.
This does not establish visual quality or fix cloud-shadow alignment.
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


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exact_duration', HERE/'exact_duration.py')
exact = importlib.util.module_from_spec(spec); spec.loader.exec_module(exact)
spec = importlib.util.spec_from_file_location('slow_video', HERE/'slow_video_2x.py')
slow = importlib.util.module_from_spec(spec); spec.loader.exec_module(slow)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def video_and_audio(meta):
    streams = meta['streams']
    videos = [x for x in streams if x.get('codec_type') == 'video']
    audios = [x for x in streams if x.get('codec_type') == 'audio']
    if len(videos) != 1 or len(audios) != 1 or len(streams) != 2:
        raise ValueError('expected exactly one video and one audio stream')
    video, audio = videos[0], audios[0]
    count = int(video.get('nb_frames') or 0)
    if (count < 5 or video.get('pix_fmt') != 'yuv420p' or
            Fraction(video.get('avg_frame_rate', '0/1')) != 24 or
            Fraction(video.get('r_frame_rate', '0/1')) != 24 or
            abs(float(video.get('start_time', 0))) > .001 or
            video.get('sample_aspect_ratio') not in (None, '1:1', 'N/A') or
            audio.get('codec_name') != 'aac' or abs(float(audio.get('start_time', 0))) > .001):
        raise ValueError('expected zero-start, square-pixel 24 fps YUV420 and AAC')
    return video, audio, count


def combine(left, right, output, cut):
    left, right = Path(left).expanduser().resolve(strict=True), Path(right).expanduser().resolve(strict=True)
    output = Path(os.path.abspath(Path(output).expanduser()))
    if os.path.lexists(output) or output in (left, right):
        raise ValueError('output must be a new file distinct from both inputs')
    before = [digest(left), digest(right)]
    lv, la, count = video_and_audio(exact.probe(left))
    rv, ra, other_count = video_and_audio(exact.probe(right))
    keys = ('width', 'height', 'pix_fmt', 'sample_aspect_ratio', 'color_range', 'color_space',
            'color_transfer', 'color_primaries')
    if count != other_count or any(lv.get(k) != rv.get(k) for k in keys) or la.get('sample_rate') != ra.get('sample_rate'):
        raise ValueError('input video geometry, color or audio rate differs')
    if not isinstance(cut, int) or not 2 <= cut <= count - 3:
        raise ValueError('cut must leave at least two frames on each side')
    slow.verify_timestamps(left, count); slow.verify_timestamps(right, count)
    left_frames, right_frames = exact.frame_hashes(left, count), exact.frame_hashes(right, count)
    if left_frames[cut-2:cut+3] != right_frames[cut-2:cut+3]:
        raise ValueError('five frames around the cut must match')
    packets = exact.audio_packets(left)
    if packets != exact.audio_packets(right):
        raise ValueError('input AAC packets or timestamps differ')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.h3-combine-', suffix='.mp4', dir=output.parent); os.close(fd)
    temp = Path(name)
    try:
        filters = (f'[0:v]trim=end_frame={cut},setpts=PTS-STARTPTS[a];'
                   f'[1:v]trim=start_frame={cut},setpts=PTS-STARTPTS[b];'
                   '[a][b]concat=n=2:v=1:a=0,setpts=N/(24*TB)[v]')
        cmd = ['ffmpeg','-v','error','-xerror','-nostdin','-y','-i',str(left),'-i',str(right),
               '-filter_complex',filters,'-map','[v]','-map','0:a:0','-map_metadata','0',
               '-c:v','libx264','-preset','veryfast','-crf','0','-pix_fmt','yuv420p',
               '-fps_mode','passthrough','-enc_time_base:v','1:24','-c:a','copy']
        for key, flag in (('color_range','-color_range'),('color_space','-colorspace'),
                          ('color_transfer','-color_trc'),('color_primaries','-color_primaries')):
            if lv.get(key) not in (None, 'unknown'): cmd += [flag, lv[key]]
        duration = float(Fraction(count,24))
        subprocess.run(cmd+['-t',f'{duration:.9f}','-movflags','+faststart',str(temp)],check=True,timeout=14400)
        ov, oa, out_count = video_and_audio(exact.probe(temp))
        if (out_count != count or any(ov.get(k) != lv.get(k) for k in keys) or
                oa.get('sample_rate') != la.get('sample_rate')):
            raise ValueError('output stream geometry or color differs')
        if exact.frame_hashes(temp, count) != left_frames[:cut] + right_frames[cut:]:
            raise ValueError('output frame differs from the selected input frame')
        slow.verify_timestamps(temp, count)
        if exact.audio_packets(temp) != packets:
            raise ValueError('output AAC packets or timestamps differ')
        subprocess.run(['ffmpeg','-v','error','-xerror','-nostdin','-i',str(temp),'-map','0:v:0',
                        '-map','0:a:0','-f','null','-'],check=True,timeout=7200)
        if [digest(left), digest(right)] != before:
            raise ValueError('an input changed during processing')
        output_hash = digest(temp)
        os.link(temp, output)
        return {'left':str(left),'right':str(right),'output':str(output),'input_sha256':before,
                'output_sha256':output_hash,'frames':count,'duration_seconds':duration,'cut_frame':cut,
                'cut_seconds':cut/24,'frames_before_from_left':cut,'frames_after_from_right':count-cut,
                'neighboring_matching_frames':list(range(cut-2,cut+3)), 'aac_packets_identical':True,
                'caveat':'The later repair may alter real motion or detail; playback review remains necessary.'}
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path); parser.add_argument('right', type=Path)
    parser.add_argument('output', type=Path); parser.add_argument('--cut-frame', type=int, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(combine(args.left, args.right, args.output, args.cut_frame),indent=2)); return 0
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError) as exc:
        print('combine-repairs:',exc,file=sys.stderr); return 1


if __name__ == '__main__': sys.exit(main())
