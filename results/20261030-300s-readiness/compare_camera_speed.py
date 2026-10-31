#!/usr/bin/env python3
"""Compare matched 30 s camera-wording and step-count experiments without promoting either."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('compare_speed', HERE / 'compare_speed.py')
speed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed)
spec = importlib.util.spec_from_file_location('exact_duration', HERE.parents[1] / 'scripts/exact_duration.py')
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)
NAMES = ('quality30-camera-default-960x544-20261031', 'quality30-camera-locked-960x544-20261031',
         'quality30-camera-locked-s10-960x544-20261031')
TAGS = ('d30_960x544_s20_seed1101_fl2va', 'd30_960x544_s20_seed1101_fl2va',
        'd30_960x544_s10_seed1101_fl2va')


def compare(root):
    runs = [speed.read_run(root, name, tag) for name, tag in zip(NAMES, TAGS, strict=True)]
    times = [run['timing'] for run in runs]
    for key in ('duration', 'effective_duration', 'size', 'seed', 'task', 'image', 'frame_indices',
                'prompt', 'audio_reference'):
        if any(item.get(key) != times[0].get(key) for item in times[1:]):
            raise ValueError('requests differ in ' + key)
    if (times[0].get('duration'), times[0].get('size'), times[0].get('seed'), times[0].get('task')) != (30, '960x544', 1101, 'fl2va'):
        raise ValueError('unexpected experiment shape or task')
    if [t.get('steps') for t in times] != [20, 20, 10]:
        raise ValueError('unexpected step counts')
    base_extra, locked_extra, fast_extra = (dict(t['extra_params']) for t in times)
    if (base_extra.get('long_video') is not True or base_extra.get('long_video_mode') != 'continuation'
            or base_extra.get('preencode_mp4') is not True or base_extra.get('frame_indices') != [0]):
        raise ValueError('unexpected streaming or first-frame parameters')
    if base_extra.get('continuation_prompts') is not None or fast_extra != locked_extra:
        raise ValueError('default prompt or step-control parameters differ')
    prompts = locked_extra.pop('continuation_prompts', None)
    prompt = times[0]['prompt'].strip()
    first = prompt.rstrip('. ') + '. One single continuous take from a steady camera, no cuts, no transitions.'
    rest = 'The same continuous shot goes on without any cut or transition, the camera remains locked on a tripod: ' + prompt
    if prompts != [first, rest, rest] or locked_extra != base_extra:
        raise ValueError('camera controls differ beyond the expected continuation wording')
    image_hashes = [(root / name / 'source.sha256').read_text().strip() for name in NAMES]
    if len(set(image_hashes)) != 1 or not re.fullmatch('[0-9a-f]{64}', image_hashes[0]):
        raise ValueError('source-image checksums differ or are invalid')
    pids = [(root / name / 'service-pids.txt').read_text().strip() for name in NAMES]
    if len(set(pids)) != 1 or not re.fullmatch(r'[1-9][0-9]* [1-9][0-9]*', pids[0]):
        raise ValueError('head/agent process identities differ')
    if any(len(run['diffuse_seconds']) < 3 for run in runs):
        raise ValueError('missing per-window timing measurements')
    baseline_video = root / NAMES[0] / (TAGS[0] + '.mp4')
    locked_video = root / NAMES[1] / (TAGS[1] + '.mp4')
    baseline_frames = exact.frame_hashes(baseline_video, 277)
    locked_frames = exact.frame_hashes(locked_video, 277)
    elapsed = [float(t['elapsed_s']) for t in times]
    labels = ('default20', 'locked20', 'locked10')
    return {
        'source_image_sha256': image_hashes[0], 'head_agent_pids': pids[0],
        'wall_seconds': dict(zip(labels, elapsed, strict=True)),
        'locked_10_vs_20_wall_change_percent': round((elapsed[2] / elapsed[1] - 1) * 100, 2),
        'diffuse_seconds_per_window': {label: run['diffuse_seconds'] for label, run in zip(labels, runs, strict=True)},
        'memory': {label: run['memory'] for label, run in zip(labels, runs, strict=True)},
        'quality_signals': {label: run['quality_signals'] for label, run in zip(labels, runs, strict=True)},
        'pre_overlap_frames_compared': 255,
        'pre_overlap_frames_equal': sum(a == b for a, b in zip(baseline_frames[:255], locked_frames[:255], strict=True)),
        'overlap_frames_compared': 22,
        'overlap_frames_equal': sum(a == b for a, b in zip(baseline_frames[255:], locked_frames[255:], strict=True)),
        'caveat': ('One set with matching head/agent service PIDs; worker-process continuity is not independently verified. '
                   'Prompt and step changes can alter motion, light and audio; '
                   'the final 22 compared frames are the continuation overlap and decode can use temporal context. '
                   'Screens do not establish visual quality. Review playback before changing defaults.'),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    try:
        print(json.dumps(compare(args.root),indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,ZeroDivisionError,subprocess.SubprocessError) as exc:
        print('compare-camera-speed:',exc,file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
