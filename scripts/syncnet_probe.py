#!/usr/bin/env python3
"""Optional SyncNet screen with a deliberately delayed-audio control.

The face crop comes from Apple Vision detections, not the upstream SyncNet
tracker. Results are exploratory; wrong speaker selection, short utterances,
language, pose and synthetic audio can all make the score misleading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

WEIGHTS_SHA = '961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442'
CODE_SHA = {'SyncNetInstance.py': 'bd722bf844700d6a923dec2f10d7cacd65c0c6b22d09aa5d44d2509660e900d6',
            'SyncNetModel.py': '41b85aed0006546cb2d6dc9180c5bc3dac737ab80eff643022e397a349c30c22'}


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def crop_from_landmarks(path: Path, width: int, height: int, duration: float, *, start: float = 0) -> dict:
    data = json.loads(path.read_text())
    rows = [row for row in data['samples'] if start <= row.get('time_s_approx', math.inf) < start + duration]
    if not rows:
        raise ValueError('no face samples in speech interval')
    boxes = [row['face_box'] for row in rows if row.get('detected') and row.get('face_box')]
    if not boxes or len(boxes) / len(rows) < .9:
        raise ValueError('face was not detected in at least 90% of sampled frames')
    if any(len(box) != 4 or any(not math.isfinite(v) or v < 0 or v > 1 for v in box) for box in boxes):
        raise ValueError('invalid face box')
    x, y, w, h = (statistics.median(values) for values in zip(*boxes, strict=True))
    if x + w > 1 or y + h > 1 or min(w, h) <= 0:
        raise ValueError('face box outside frame')
    centers = [((a + c / 2) * width, (1 - b - d / 2) * height) for a, b, c, d in boxes]
    cx, cy = (x + w / 2) * width, (1 - y - h / 2) * height
    half = max(w * width, h * height) / 2
    spread = max(math.hypot(a - cx, b - cy) for a, b in centers) / (2 * half)
    if spread > .35:
        raise ValueError('face position varies too much for the fixed crop')
    side = int(round(2 * half * 1.4))
    side -= side % 2
    if side < 32 or side > min(width, height):
        raise ValueError('face crop does not fit')
    left = max(0, min(width - side, int(round(cx - half * 1.4))))
    top = max(0, min(height - side, int(round(cy - half))))
    left -= left % 2
    top -= top % 2
    return {'x': left, 'y': top, 'side': side, 'detections': len(boxes), 'samples': len(rows),
            'max_center_shift_in_face_widths': round(spread, 4),
            'samples_with_multiple_faces': sum(row.get('face_count', 0) > 1 for row in rows)}


def make_crop(source: Path, dest: Path, crop: dict, duration: float, audio_delay_ms: int, *, start: float = 0) -> None:
    video = (f"[0:v:0]trim=start={start}:duration={duration},setpts=PTS-STARTPTS,fps=25,"
             f"crop={crop['side']}:{crop['side']}:{crop['x']}:{crop['y']},"
             'scale=224:224:flags=bicubic,setsar=1[v]')
    delay = f'adelay={audio_delay_ms}:all=1,' if audio_delay_ms else ''
    # adelay may generate silence packets without PTS after a mid-file trim.
    # Reconstruct timestamps before atrim so those packets are not discarded.
    audio = (f'[0:a:0]atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS,aresample=16000,'
             f'{delay}asetpts=N/SR/TB,apad=whole_dur={duration},atrim=duration={duration},asetpts=N/SR/TB[a]')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-n', '-i', str(source),
                    '-filter_complex', video + ';' + audio, '-map', '[v]', '-map', '[a]', '-t', str(duration),
                    '-c:v', 'ffv1', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le', str(dest)],
                   check=True, timeout=180)


def evaluate(video: Path, landmarks: Path, duration: float, *, upstream: Path, weights: Path, start: float = 0) -> dict:
    upstream = upstream.resolve(strict=True)
    weights = weights.resolve(strict=True)
    for name, expected in CODE_SHA.items():
        if sha(upstream / name) != expected:
            raise ValueError('SyncNet source differs from the pinned digest: ' + name)
    if sha(weights) != WEIGHTS_SHA:
        raise ValueError('SyncNet weights differ from the pinned digest')
    if not math.isfinite(start) or start < 0 or not 2 <= duration <= 15:
        raise ValueError('speech sample must be 2–15 seconds')
    # Both streams must have the same exact duration at the model's 25 fps.
    requested_duration = duration
    duration = math.ceil(duration * 25) / 25
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'stream=width,height,duration', '-of', 'json', str(video)],
                           check=True, capture_output=True, text=True, timeout=60)
    stream = json.loads(probe.stdout)['streams'][0]
    if float(stream['duration']) + .001 < start + duration:
        raise ValueError('speech sample exceeds video duration')
    crop = crop_from_landmarks(landmarks, int(stream['width']), int(stream['height']), duration, start=start)
    if 'SyncNetInstance' in sys.modules or 'SyncNetModel' in sys.modules:
        raise ValueError('SyncNet modules must not be preloaded from a different location')
    sys.path.insert(0, str(upstream))
    import torch
    from SyncNetInstance import SyncNetInstance

    torch.set_num_threads(4)
    model = SyncNetInstance(device='cpu')
    model.loadParameters(str(weights))  # pinned upstream uses torch.load(weights_only=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix='h3-syncnet-') as d, torch.inference_mode():
        directory = Path(d)
        for name, delay in (('original', 0), ('audio_delayed_200ms', 200)):
            cropped = directory / (name + '.mkv')
            make_crop(video, cropped, crop, duration, delay, start=start)
            opt = argparse.Namespace(tmp_dir=str(directory / 'work'), reference=name, batch_size=8, vshift=15)
            offset, confidence, distances = model.evaluate(opt, videofile=str(cropped))
            if not math.isfinite(float(confidence)):
                raise ValueError('SyncNet returned nonfinite confidence')
            results[name] = {'offset_frames_25fps': int(offset), 'confidence': round(float(confidence), 4),
                             'windows': len(distances)}
    delta = results['audio_delayed_200ms']['offset_frames_25fps'] - results['original']['offset_frames_25fps']
    return {'video': str(video), 'video_sha256': sha(video), 'face_report_sha256': sha(landmarks),
            'model_sha256': WEIGHTS_SHA, 'code_sha256': CODE_SHA, 'start_seconds': start,
            'requested_duration_seconds': requested_duration,
            'duration_seconds': duration, 'face_crop': crop,
            'results': results, 'control_offset_delta_frames': delta,
            'control_recovered_expected_shift': abs(delta + 5) <= 1,
            'limitations': ('Exploratory fixed crop and largest-face selection; this is not a verified speaker '
                            'identity, lip-sync verdict or naturalness score. The control tests response to '
                            'an artificial delay, not accuracy on the original speech.')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('face_report', type=Path)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--start', type=float, default=0, help='speech interval start in seconds')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--upstream-dir', type=Path, required=True, help='pinned joonson/syncnet_python checkout')
    parser.add_argument('--weights', type=Path, required=True, help='pinned syncnet_v2.model')
    args = parser.parse_args()
    try:
        result = evaluate(args.video, args.face_report, args.duration, upstream=args.upstream_dir,
                          weights=args.weights, start=args.start)
        with args.report.open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps(result, indent=2))
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f'syncnet-probe: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
