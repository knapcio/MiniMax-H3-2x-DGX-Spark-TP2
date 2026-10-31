#!/usr/bin/env python3
"""Collect a verified 300 s retry and run isolated, guarded follow-up comparisons on the head.

Only a successful, fully verified predecessor can release follow-ups. The original
failure logs, predecessor outputs and older queue event history are never modified.
"""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


TAG = 'd300_960x544_s20_seed1101_fl2va'
IMAGE_SHA = '567138222c7e0793e8e774a80b188695f862ca2430ae70a969444fec23cbdd5d'
START = '2026-10-31 21:42:00'
WORKER = '10.100.96.5'


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def mark(root, line):
    with (root / 'aspect-events-20261031.log').open('a') as stream:
        stream.write(f'{line} {dt.datetime.now().astimezone().isoformat(timespec="seconds")}\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def verify_bundle(bundle, expected):
    manifest = bundle / 'manifest.json'
    if sha(manifest) != expected:
        raise ValueError('bundle manifest differs from pinned checksum')
    content = json.loads(manifest.read_text())
    if not isinstance(content, dict) or not content:
        raise ValueError('invalid bundle manifest')
    for name, digest in content.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or len(digest) != 64:
            raise ValueError('invalid bundle entry')
        path = bundle / relative
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError('bundle file missing or changed: ' + name)


def probe_media(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                             'stream=codec_type,width,height,avg_frame_rate,nb_frames:format=duration',
                             '-of', 'json', str(path)], check=True, capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout)


def validate_retry(retry, root, *, image_sha=IMAGE_SHA, media_probe=probe_media):
    if (retry / 'runner.exit').read_text().strip() != '0':
        raise ValueError('300 s runner failed')
    events = (retry / 'events.log').read_text().splitlines()
    if not any(s.startswith('H3-RETRY300-START ') for s in events) or not any(
            s.startswith('H3-RETRY300-END rc=0 ') for s in events):
        raise ValueError('300 s terminal markers missing')
    if (TAG + ' ok') not in (retry / 'STATUS').read_text().splitlines():
        raise ValueError('300 s STATUS does not report success')
    if 'verify: OK' not in (retry / (TAG + '.verify.txt')).read_text().splitlines():
        raise ValueError('full media verification missing')
    image = root / 'koninck.png'
    if sha(image) != image_sha:
        raise ValueError('source image changed')
    data = json.loads((retry / (TAG + '.mp4.timing.json')).read_text())
    if (data.get('http') not in ('200', 200) or data.get('curl_rc') != 0 or
            (data.get('duration'), data.get('size'), data.get('steps'), data.get('seed'), data.get('task'))
            != (300, '960x544', 20, 1101, 'fl2va') or data.get('image') != str(image)):
        raise ValueError('300 s request settings or response differ')
    extra = data.get('extra_params', {})
    if (extra.get('long_video') is not True or extra.get('long_video_mode') != 'continuation' or
            extra.get('preencode_mp4') is not True):
        raise ValueError('300 s continuation settings differ')
    minimum = {}
    for node in ('head', 'worker'):
        path = retry / (TAG + f'.mem-{node}.csv')
        with path.open(newline='') as stream:
            values = [int(row[1]) for row in csv.reader(stream) if len(row) >= 2 and row[1].isdigit()]
        if len(values) < 2 or min(values) < 8192:
            raise ValueError('300 s memory samples missing or below 8 GiB: ' + node)
        minimum[node] = min(values)
    video = retry / (TAG + '.mp4')
    if not video.is_file() or video.stat().st_size == 0:
        raise ValueError('300 s video missing')
    meta = media_probe(video)
    streams = meta['streams']
    videos = [x for x in streams if x.get('codec_type') == 'video']
    audios = [x for x in streams if x.get('codec_type') == 'audio']
    if (len(videos) != 1 or len(audios) != 1 or
            (videos[0].get('width'), videos[0].get('height'), videos[0].get('avg_frame_rate'))
            != (960, 544, '24/1') or int(videos[0].get('nb_frames', 0)) < 7200):
        raise ValueError('300 s stream geometry or frame count differs')
    duration = float(meta['format']['duration'])
    if not 299.4 <= duration <= 301:
        raise ValueError('300 s container duration outside expected range')
    return {'native_sha256': sha(video), 'source_image_sha256': image_sha,
            'native_frames': int(videos[0]['nb_frames']), 'native_duration': duration,
            'elapsed_s': data.get('elapsed_s'), 'min_memavailable_mib': minimum}


def validate_controls_and_links(root, results, retry):
    links = {'koninck.png': results / 'koninck.png',
             'experiment300-960x544-20261031': retry}
    for duration in (10, 15):
        name = f'quality{duration}-direct-960x544-20261031'
        tag = f'd{duration}_960x544_s20_seed1101_fl2va'
        folder = results / name
        links[name] = folder
        if ((tag + ' ok') not in (folder / 'STATUS').read_text().splitlines() or
                'verify: OK' not in (folder / (tag + '.verify.txt')).read_text().splitlines() or
                not (folder / (tag + '.mp4')).is_file() or (folder / (tag + '.mp4')).stat().st_size == 0):
            raise ValueError(f'{duration} s control lacks verified media')
        if (folder / 'source.sha256').read_text().strip() != IMAGE_SHA:
            raise ValueError(f'{duration} s control image hash differs')
        data = json.loads((folder / (tag + '.mp4.timing.json')).read_text())
        if (data.get('http') not in ('200', 200) or data.get('curl_rc') != 0 or
                (data.get('duration'), data.get('size'), data.get('steps'), data.get('seed'), data.get('task'))
                != (duration, '960x544', 20, 1101, 'fl2va') or
                data.get('image') != str(results / 'koninck.png')):
            raise ValueError(f'{duration} s control request differs')
    for duration in (16, 60):
        name = f'quality{duration}-960x544-20261031'
        folder = results / name
        links[name] = folder
        if (folder / 'source.sha256').read_text().strip() != IMAGE_SHA:
            raise ValueError(f'{duration} s control image hash differs')
    for name, expected in links.items():
        link = root / name
        if not link.is_symlink() or link.resolve(strict=True) != expected.resolve(strict=True):
            raise ValueError('isolated input link differs: ' + name)


def ensure_no_xid(root, phase):
    command = ['journalctl', '-k', '--since', START, '--no-pager', '-o', 'short-iso']
    for node, args in (('head', command), ('worker', ['ssh', '-n', '-o', 'BatchMode=yes', '-o',
                            'ConnectTimeout=8', WORKER, shlex.join(command)])):
        result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)
        lines = [line for line in result.stdout.splitlines() if 'NVRM: Xid' in line]
        path = root / f'xid-{node}-{phase}.txt'
        with path.open('x') as out:
            out.write('\n'.join(lines) + ('\n' if lines else 'no Xid lines in checked interval\n'))
        if lines:
            raise ValueError(f'{node} kernel recorded Xid; inspect {path.name}')


def run_logged(command, path, *, env=None, timeout):
    with path.open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=timeout)


def make_exact_and_screens(root, retry, bundle):
    video = retry / (TAG + '.mp4')
    exact = root / 'd300_koninck_retry_960x544_EXACT300s.mp4'
    if os.path.lexists(exact):
        raise ValueError('exact 300 s output already exists')
    py = str(Path.home() / 'h3-long/.venv/bin/python')
    run_logged([py, str(bundle / 'exact_duration.py'), str(video), str(exact), '300'],
               root / 'exact300.log', timeout=7200)
    run_logged([py, str(bundle / 'luma_flicker.py'), str(exact), '--minimum', '2'],
               root / 'luma-flicker-300.txt', timeout=3600)
    run_logged([py, str(bundle / 'regional_flicker.py'), str(exact), '--minimum', '2'],
               root / 'regional-flicker-300.txt', timeout=3600)
    run_logged([py, str(bundle / 'ground_light_stats.py'), str(exact)],
               root / 'ground-light-300.txt', timeout=3600)
    return exact


def run_queue(root, bundle, phase, script, prefix):
    env = os.environ.copy()
    env.update({prefix + '_RESULTS': str(root), prefix + '_BENCH': str(bundle / 'scripts/bench-matrix.sh'),
                prefix + '_METRICS': str(bundle / 'scripts/clip-metrics.sh'),
                prefix + '_TXN': str(Path.home() / 'h3-tp2-results/file-txn-fc14580.py'),
                prefix + '_FLICKER': str(bundle / 'luma_flicker.py'),
                'H3_MEMWATCH_WORKER_DIR': 'h3-tp2-results/memwatch/retry-followups-20261031',
                'H3_MEMWATCH_PER_OUTPUT': '1'})
    mark(root, 'H3-RETRY-FOLLOWUP-START ' + phase)
    run_logged(['bash', str(bundle / script)], root / (phase + '.log'), env=env, timeout=46000)
    ensure_no_xid(root, phase)
    mark(root, 'H3-RETRY-FOLLOWUP-OK ' + phase)


def runner_alive(retry):
    pid = int((retry / 'runner.pid').read_text().strip())
    if pid < 2:
        raise ValueError('invalid 300 s runner PID')
    proc = Path('/proc') / str(pid)
    try:
        command = (proc / 'cmdline').read_bytes().split(b'\0')
        state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != 'Z' and str(retry / 'run.sh').encode() in command


def wait_for_retry(retry, max_wait, interval):
    deadline = time.monotonic() + max_wait
    missing = 0
    while time.monotonic() < deadline:
        if (retry / 'runner.exit').is_file():
            return
        missing = 0 if runner_alive(retry) else missing + 1
        if missing >= 2:
            if (retry / 'runner.exit').is_file():
                return
            raise ValueError('300 s runner vanished without a terminal status')
        time.sleep(interval)
    raise TimeoutError('300 s runner did not report an exit before watcher deadline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='new isolated follow-up directory')
    parser.add_argument('--results', type=Path, required=True, help='original h3-tp2-results directory')
    parser.add_argument('--retry', type=Path, required=True, help='existing 300 s retry directory')
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--max-wait', type=int, default=32400)
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    root, results, retry = args.root.resolve(strict=True), args.results.resolve(strict=True), args.retry.resolve(strict=True)
    bundle = root / 'bundle'
    phase = 'initialization'
    with (root / 'followups.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('follow-up watcher already running', file=sys.stderr)
            return 2
        try:
            if args.interval < 1 or args.max_wait < 1:
                raise ValueError('invalid watcher interval or deadline')
            verify_bundle(bundle, args.manifest_sha)
            mark(root, 'H3-RETRY-FOLLOWUPS-WAITING')
            phase = 'predecessor'
            wait_for_retry(retry, args.max_wait, args.interval)
            info = validate_retry(retry, results)
            validate_controls_and_links(root, results, retry)
            if shutil.disk_usage(root).free < 16 * 1024 ** 3:
                raise ValueError('less than 16 GiB available for postprocessing')
            ensure_no_xid(root, 'predecessor')
            # The isolated event history contains only facts verified above.
            mark(root, 'H3-ASPECT-SHORT-DONE verified-existing-10s-and-15s')
            mark(root, 'H3-ASPECT-OK experiment300-960x544-20261031')
            phase = 'exact-and-screens'
            exact = make_exact_and_screens(root, retry, bundle)
            info['exact_sha256'] = sha(exact)
            info['note'] = 'Structural and detector checks only; visual/audio quality remains unreviewed.'
            with (root / 'retry300-summary.json').open('x') as out:
                json.dump(info, out, indent=2, sort_keys=True); out.write('\n')
            ensure_no_xid(root, 'postprocessing')
            mark(root, 'H3-ASPECT-EXPERIMENT300-DONE structural-media-verification-only')
            for phase, script, prefix in (
                ('camera-speed', 'queue-camera-speed-ab-20261031.sh', 'H3_AB'),
                ('window175', 'queue-window175-20261031.sh', 'H3_W175'),
                ('human-pace', 'queue-human-pace-direct-bench-20261031.sh', 'H3_PACE'),
            ):
                verify_bundle(bundle, args.manifest_sha)
                run_queue(root, bundle, phase, script, prefix)
            mark(root, 'H3-RETRY-FOLLOWUPS-DONE quality-review-required')
            return 0
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            print(f'follow-ups held at {phase}: {exc}', file=sys.stderr, flush=True)
            try:
                mark(root, f'H3-RETRY-FOLLOWUPS-HOLD phase={phase} reason={type(exc).__name__}')
            except OSError:
                pass
            return 1


if __name__ == '__main__':
    sys.exit(main())
