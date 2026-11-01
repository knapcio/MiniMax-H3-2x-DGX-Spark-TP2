#!/usr/bin/env python3
"""Compare checkpoint input hashing in two local revisions, CPU only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
CHILD = r'''
import importlib.util,json,resource,sys,time,torch
spec=importlib.util.spec_from_file_location('checkpoint',sys.argv[1]); module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module); torch.set_num_threads(1)
x=torch.full((int(sys.argv[2])*1024*1024//4,),0.25,dtype=torch.float32)
before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; start=time.perf_counter()
digest=module.fingerprint_inputs({'seed':1101},{'input':x},max_bytes=(int(sys.argv[2])+16)*1024**2)
elapsed=time.perf_counter()-start; after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
unit=1024**2 if sys.platform=='darwin' else 1024
print(json.dumps({'sha256':digest,'seconds':round(elapsed,6),
                  'peak_rss_increase_mib':round(max(0,after-before)/unit,2),'torch':torch.__version__}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-ref', required=True, help='Git revision containing the earlier implementation')
    parser.add_argument('--mib', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.mib <= 1024 or not 1 <= args.repeats <= 10:
        parser.error('mib must be 1..1024 and repeats must be 1..10')
    old = subprocess.run(['git', 'show', f'{args.before_ref}:h3_tp2/cont_checkpoint.py'], cwd=ROOT,
                         check=True, capture_output=True).stdout
    candidate = ROOT / 'h3_tp2/cont_checkpoint.py'
    env = os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    with tempfile.TemporaryDirectory(prefix='h3-fingerprint-bench-') as temp:
        baseline = Path(temp) / 'before.py'; baseline.write_bytes(old)
        paths = {'before': baseline, 'after': candidate}; measurements = {name: [] for name in paths}
        for _ in range(args.repeats):
            for name, path in paths.items():
                proc = subprocess.run([sys.executable, '-c', CHILD, str(path), str(args.mib)], env=env,
                                      check=True, capture_output=True, text=True, timeout=120)
                measurements[name].append(json.loads(proc.stdout.strip().splitlines()[-1]))
        if len({item['sha256'] for values in measurements.values() for item in values}) != 1:
            raise ValueError('fingerprints differ')
        result = {'platform': platform.platform(), 'python': sys.version.split()[0], 'input_mib': args.mib,
                  'before_ref': args.before_ref, 'before_code_sha256': hashlib.sha256(old).hexdigest(),
                  'after_code_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(), 'samples': measurements,
                  'median_seconds': {name: statistics.median(x['seconds'] for x in items)
                                     for name, items in measurements.items()},
                  'median_peak_rss_increase_mib': {name: statistics.median(x['peak_rss_increase_mib'] for x in items)
                                                   for name, items in measurements.items()},
                  'caveat': ('CPU contiguous-input fingerprint only. RSS is process high-water increase after input '
                             'allocation; it is not GPU memory, checkpoint write cost or render speed.')}
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
