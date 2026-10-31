#!/usr/bin/env python3
"""Run one isolated attention benchmark only after the guarded H3 follow-ups.

The serving API remains available. A new request, a changed service, insufficient
memory, or a failed health check aborts only this benchmark; sampling cannot
eliminate the brief race between an incoming request and the next guard check.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request


START = '2026-10-31 21:42:00'
WORKER = '10.100.96.5'
URL = 'http://127.0.0.1:18093'
PREDECESSOR_HOLDS = ('H3-ASPECT-HOLD ', 'H3-RETRY-FOLLOWUPS-HOLD ')
NODE_SCRIPT = r'''import json, pathlib, subprocess, sys
service, guard = sys.argv[1:]
for unit in (service, guard):
    subprocess.run(['systemctl','--user','is-active','--quiet',unit],check=True,timeout=5)
pid = int(subprocess.run(['systemctl','--user','show',service,'-p','MainPID','--value'],check=True,capture_output=True,text=True,timeout=5).stdout.strip())
mem = {s.split()[0].rstrip(':'): int(s.split()[1]) for s in pathlib.Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')}
gpu = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=8).stdout.strip()
print(json.dumps({'pid':pid,'mem_kib':mem['MemAvailable'],'gpu':int(gpu)}))'''


class Hold(Exception):
    pass


class Busy(Exception):
    pass


def event_state(lines):
    if any(line.startswith(PREDECESSOR_HOLDS) for line in lines):
        return 'hold'
    required = ('H3-RETRY-FOLLOWUPS-DONE ', 'H3-ASPECT-CLOUDLIGHT-DONE ')
    return 'ready' if all(any(line.startswith(prefix) for line in lines) for prefix in required) else 'wait'


def metrics_idle(data):
    values = {'running': [], 'waiting': []}
    pattern = re.compile(r'^vllm_omni:num_requests_(running|waiting)\{[^}]*\}\s+(\S+)')
    for line in data.splitlines():
        match = pattern.match(line)
        if match:
            try:
                value = float(match.group(2))
            except ValueError:
                raise Hold('malformed request metric') from None
            if not math.isfinite(value) or value < 0:
                raise Hold('invalid request metric')
            values[match.group(1)].append(value)
    if not all(values.values()):
        raise Hold('request metrics missing')
    return all(value == 0 for group in values.values() for value in group)


def verified_bundle(bundle, manifest_sha):
    manifest = bundle / 'manifest.json'
    if manifest.is_symlink() or hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_sha:
        raise Hold('bundle manifest checksum differs')
    data = json.loads(manifest.read_text())
    if set(data) != {'attention_probe.py', 'queue-attention-probe.py'}:
        raise Hold('unexpected bundle contents')
    for name, expected in data.items():
        path = bundle / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise Hold('bundle file missing or changed: ' + name)


def run(command, timeout=20):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout).stdout


def node_snapshot(worker=False):
    service = 'h3-tp2-agent.service' if worker else 'h3-tp2-head.service'
    guard = 'h3-tp2-memguard-worker.service' if worker else 'h3-tp2-memguard-head.service'
    command = ['python3', '-c', NODE_SCRIPT, service, guard]
    if worker:
        command = ['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', WORKER, shlex.join(command)]
    data = json.loads(run(command, timeout=25))
    if data['pid'] < 2 or data['mem_kib'] < 0 or not 0 <= data['gpu'] <= 100:
        raise Hold('invalid node snapshot')
    return data


def request_state():
    def get(path):
        with urllib.request.urlopen(URL + path, timeout=5) as response:
            if response.status != 200:
                raise Hold('head API returned non-200')
            return response.read().decode()
    get('/health')
    idle = metrics_idle(get('/metrics'))
    videos = json.loads(get('/v1/videos?limit=1'))
    if videos.get('data') != [] or videos.get('has_more') is not False:
        idle = False
    return idle


def check_xids():
    command = ['journalctl', '-k', '--since', START, '--no-pager', '-o', 'short-iso']
    for node, cmd in (('head', command), ('worker', ['ssh', '-n', '-o', 'BatchMode=yes', '-o',
                            'ConnectTimeout=8', WORKER, shlex.join(command)])):
        output = run(cmd, timeout=60)
        if 'NVRM: Xid' in output:
            raise Hold(node + ' kernel reports an Xid since retry start')


def check_pair(pids, *, idle_gpu, minimum_kib):
    if not request_state():
        raise Busy('API has an active or queued request')
    snapshots = [node_snapshot(), node_snapshot(worker=True)]
    if [x['pid'] for x in snapshots] != pids:
        raise Hold('serving process PID changed')
    if any(x['mem_kib'] < minimum_kib for x in snapshots):
        raise Busy('available memory below guard threshold')
    if idle_gpu and any(x['gpu'] >= 10 for x in snapshots):
        raise Busy('GPU pair is busy')
    return snapshots


def mark(root, message):
    text = f'{message} {dt.datetime.now().astimezone().isoformat(timespec="seconds")}\n'
    with (root / 'attention-probe-events.log').open('a') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    print(text, end='', flush=True)


def stop_child(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=15)


def monitor_child(child, pids, root, timeout):
    deadline, xid_next = time.monotonic() + timeout, time.monotonic()
    try:
        while child.poll() is None:
            if time.monotonic() >= deadline:
                raise Hold('benchmark exceeded time limit')
            check_pair(pids, idle_gpu=False, minimum_kib=12 * 1024 ** 2)
            if time.monotonic() >= xid_next:
                check_xids(); xid_next = time.monotonic() + 30
            time.sleep(3)
        if child.returncode:
            raise Hold('benchmark exited with status ' + str(child.returncode))
        check_pair(pids, idle_gpu=False, minimum_kib=12 * 1024 ** 2)
        check_xids()
    except BaseException:
        stop_child(child)
        raise


def validate_result(path):
    data = json.loads(path.read_text())
    if data.get('capability') != [12, 1] or len(data.get('cases', [])) != 4:
        raise Hold('benchmark result incomplete')
    for case in data['cases']:
        if (not case.get('cudnn', {}).get('timing', {}).get('milliseconds') or
                set(case.get('backends', {})) != {'flashinfer_fa2', 'flashinfer_fmha_v2_fp16_qk'}):
            raise Hold('benchmark case incomplete')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--wait-hours', type=int, default=72)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    bundle = root / 'attention-probe-bundle-20261031'
    phase = 'initialization'
    def interrupted(signum, _frame):
        raise Hold('interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (root / 'attention-probe.lock').open('a') as own_lock:
        try:
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('attention watcher already active', file=sys.stderr); return 2
        try:
            if not 1 <= args.wait_hours <= 96:
                raise Hold('invalid wait period')
            verified_bundle(bundle, args.bundle_sha)
            mark(root, 'H3-ATTN-PROBE-WAITING')
            phase = 'predecessors'
            deadline = time.monotonic() + 3600 * args.wait_hours
            while True:
                state = event_state((root / 'aspect-events-20261031.log').read_text().splitlines())
                if state == 'hold':
                    raise Hold('a predecessor held; no benchmark')
                if state == 'ready':
                    break
                if time.monotonic() >= deadline:
                    raise Hold('predecessor deadline elapsed')
                time.sleep(30)
            with (root / 'followups.lock').open('a') as followups_lock, \
                    (root / 'cloud-light-aspect-20261031.lock').open('a') as cloud_lock:
                while True:
                    try:
                        fcntl.flock(followups_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(cloud_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        fcntl.flock(followups_lock, fcntl.LOCK_UN)
                        if time.monotonic() >= deadline:
                            raise Hold('predecessor locks did not clear')
                        time.sleep(10)
                phase = 'preflight'
                pids = [int(x) for x in (root / 'quality30-camera-locked-960x544-20261031/service-pids.txt').read_text().split()]
                if len(pids) != 2 or min(pids) < 2:
                    raise Hold('invalid baseline service PIDs')
                if shutil.disk_usage(root).free < 8 * 1024 ** 3:
                    raise Hold('less than 8 GiB disk available')
                check_xids(); verified_bundle(bundle, args.bundle_sha)
                consecutive = 0
                while consecutive < 3:
                    if time.monotonic() >= deadline:
                        raise Hold('idle pair not available before deadline')
                    try:
                        check_pair(pids, idle_gpu=True, minimum_kib=24 * 1024 ** 2)
                        consecutive += 1
                    except Busy:
                        consecutive = 0
                    if consecutive < 3:
                        time.sleep(10)
                if event_state((root / 'aspect-events-20261031.log').read_text().splitlines()) != 'ready':
                    raise Hold('predecessor state changed')
                verified_bundle(bundle, args.bundle_sha)
                check_xids()
                check_pair(pids, idle_gpu=True, minimum_kib=24 * 1024 ** 2)
                output = root / 'attention-probe-result-20261031.json'
                if os.path.lexists(output):
                    raise Hold('benchmark result already exists')
                phase = 'benchmark'
                env = os.environ.copy()
                env['PATH'] = str(Path.home() / 'h3-long/.venv/bin') + ':/usr/local/cuda/bin:' + env.get('PATH', '')
                env['FLASHINFER_WORKSPACE_BASE'] = str(root / 'attention-probe-workspace-20261031')
                env['MAX_JOBS'] = '2'; env['CMAKE_BUILD_PARALLEL_LEVEL'] = '2'
                env['OMP_NUM_THREADS'] = '1'; env['OPENBLAS_NUM_THREADS'] = '1'
                mark(root, 'H3-ATTN-PROBE-START synthetic-sm121')
                with (root / 'attention-probe-run.log').open('x') as log:
                    child = None
                    try:
                        child = subprocess.Popen([str(Path.home() / 'h3-long/.venv/bin/python'),
                                                  str(bundle / 'attention_probe.py'), str(output)],
                                                 stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                        monitor_child(child, pids, root, timeout=3600)
                    finally:
                        if child is not None:
                            stop_child(child)
                validate_result(output)
                mark(root, 'H3-ATTN-PROBE-DONE synthetic-sm121')
                return 0
        except (Exception, KeyboardInterrupt) as exc:
            mark(root, f'H3-ATTN-PROBE-HOLD phase={phase} reason={type(exc).__name__}:{str(exc)[:240]}')
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
