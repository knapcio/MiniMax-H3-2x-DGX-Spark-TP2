#!/usr/bin/env python3
"""Make a separate, verified, linearly levelled AAC copy of a quiet H3 clip."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


def run(args, *, timeout=7200):
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=timeout)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def probe(path):
    result = run(['ffprobe', '-v', 'error', '-show_data_hash', 'sha256', '-show_entries',
                  'stream=index,codec_type,codec_name,width,height,extradata_hash,sample_rate,channels,time_base,start_time,duration,nb_frames',
                  '-of', 'json', str(path)])
    streams = json.loads(result.stdout)['streams']
    if len(streams) != 2 or [x['codec_type'] for x in streams] != ['video', 'audio']:
        raise ValueError('expected exactly one video stream followed by one audio stream')
    video, audio = streams
    if int(audio['channels']) not in (1, 2) or int(audio['sample_rate']) < 8000:
        raise ValueError('only mono or stereo audio at 8 kHz or higher is supported')
    for stream in streams:
        if not math.isfinite(float(stream['start_time'])) or not math.isfinite(float(stream['duration'])):
            raise ValueError('missing or invalid stream timing')
    return video, audio


def loudness(path, target, lra, peak):
    result = run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(path), '-map', '0:a:0', '-af',
                  f'loudnorm=I={target}:LRA={lra}:TP={peak}:print_format=json', '-f', 'null', '-'])
    matches = re.findall(r'\{\s*"input_i".*?\}', result.stderr, flags=re.DOTALL)
    if not matches:
        raise ValueError('ffmpeg did not return a loudness measurement')
    values = json.loads(matches[-1])
    for key in ('input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset'):
        if not math.isfinite(float(values[key])):
            raise ValueError('audio is silent or its loudness measurement is invalid')
    return values


def video_packets(path, time_base):
    result = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets', '-show_data_hash', 'sha256',
                  '-show_entries', 'packet=pts,dts,duration,size,flags,data_hash', '-of', 'json', str(path)])
    base = Fraction(time_base)
    packets = json.loads(result.stdout)['packets']
    if not packets:
        raise ValueError('video has no packets')
    return [(Fraction(int(p['pts'])) * base, Fraction(int(p['dts'])) * base,
             Fraction(int(p['duration'])) * base, p['size'], p['flags'], p['data_hash']) for p in packets]


def create(source, destination, *, target=-20.0, max_gain=30.0):
    source = Path(source).resolve(strict=True)
    destination = Path(os.path.abspath(destination))
    report_path = destination.with_suffix('.audio-normalization.json')
    if destination.suffix.lower() != '.mp4' or source in (destination, report_path):
        raise ValueError('choose a different output ending in .mp4')
    if os.path.lexists(destination) or os.path.lexists(report_path):
        raise FileExistsError('output or report already exists')
    if not -36 <= target <= -14 or not 0 < max_gain <= 30:
        raise ValueError('target must be -36..-14 LUFS; maximum gain must be 0..30 dB')
    video, audio = probe(source)
    source_hash = digest(source)
    measured = loudness(source, target, 11, -2)
    input_i = float(measured['input_i'])
    gain = target - input_i
    if abs(gain) > max_gain or float(measured['input_tp']) + gain > -2.25:
        raise ValueError('level change exceeds limit or would require peak limiting')
    lra = max(11, math.ceil(float(measured['input_lra']) + 1))
    if lra > 50:
        raise ValueError('audio dynamic range exceeds the linear normalizer limit')
    filt = (f'loudnorm=I={target}:LRA={lra}:TP=-2:measured_I={measured["input_i"]}:'
            f'measured_TP={measured["input_tp"]}:measured_LRA={measured["input_lra"]}:'
            f'measured_thresh={measured["input_thresh"]}:offset={measured["target_offset"]}:'
            'linear=true:print_format=json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.h3-audio-', dir=destination.parent) as tmp:
        candidate = Path(tmp) / 'candidate.mp4'
        result = run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(source), '-map', '0:v:0', '-map', '0:a:0',
                      '-map_metadata', '0', '-c:v', 'copy', '-af', filt, '-c:a', 'aac', '-b:a', '192k',
                      '-ar', audio['sample_rate'], '-movflags', '+faststart', str(candidate)])
        match = re.findall(r'\{\s*"input_i".*?\}', result.stderr, flags=re.DOTALL)
        if not match or json.loads(match[-1]).get('normalization_type') != 'linear':
            raise ValueError('ffmpeg could not apply a linear level change')
        video_after, audio_after = probe(candidate)
        if any(key not in video or video[key] != video_after.get(key)
               for key in ('codec_name', 'width', 'height', 'extradata_hash')):
            raise ValueError('video codec metadata differs')
        if video_packets(source, video['time_base']) != video_packets(candidate, video_after['time_base']):
            raise ValueError('video packets or their timestamps differ')
        sample_tolerance = 1024 / int(audio['sample_rate'])
        if (audio_after['sample_rate'] != audio['sample_rate'] or audio_after['channels'] != audio['channels'] or
                abs(float(audio['start_time']) - float(audio_after['start_time'])) > sample_tolerance or
                abs(float(audio['duration']) - float(audio_after['duration'])) > sample_tolerance):
            raise ValueError('audio stream format or timing differs')
        run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(candidate), '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'])
        after = loudness(candidate, target, 11, -2)
        if abs(float(after['input_i']) - target) > 1 or float(after['input_tp']) > -1.5:
            raise ValueError('encoded audio loudness or true peak is outside the verified bounds')
        candidate_hash = digest(candidate)
        if digest(source) != source_hash:
            raise ValueError('source changed during processing')
        report = {'source': str(source), 'output': str(destination), 'source_sha256': source_hash,
                  'output_sha256': candidate_hash, 'target_lufs': target, 'gain_estimate_db': gain,
                  'before': measured, 'after': after, 'video_codec_metadata_identical': True,
                  'video_packets_and_timestamps_identical': True,
                  'audio_timing_tolerance_seconds': sample_tolerance,
                  'caveat': 'Experimental: re-encoded audio and increased level can make existing noise audible; '
                            'listening review is required.'}
        temporary_report = Path(tmp) / 'report.json'
        with temporary_report.open('x') as out:
            json.dump(report, out, indent=2, sort_keys=True)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.link(candidate, destination)
        try:
            os.link(temporary_report, report_path)
        except BaseException:
            if os.path.samestat(destination.stat(), candidate.stat()):
                destination.unlink()
            raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    parser.add_argument('--target-lufs', type=float, default=-20.0)
    parser.add_argument('--max-gain-db', type=float, default=30.0)
    args = parser.parse_args()
    print(json.dumps(create(args.source, args.output, target=args.target_lufs, max_gain=args.max_gain_db), indent=2))


if __name__ == '__main__':
    main()
