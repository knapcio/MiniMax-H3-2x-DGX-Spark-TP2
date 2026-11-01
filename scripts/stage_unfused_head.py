#!/usr/bin/env python3
"""Stage an experimental head launcher that disables fused SwiGLU after env.sh.

This only creates a new private directory containing two scripts. It does not
change a service, start a process, verify a GPU, or load an adapter. Caller
must validate idle admission, the loaded plugin and both ranks before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


SOURCE_LINE = b'source "$(dirname "$0")/env.sh"\n'
OVERRIDE = b'export H3_TP2_FP8_FUSE_SWIGLU=0  # staged LoRA compatibility experiment\n'
SHA = re.compile(r'^[0-9a-f]{64}$')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_file(path, expected):
    if not isinstance(expected, str) or not SHA.fullmatch(expected):
        raise ValueError('expected checksum must be lowercase SHA-256')
    if path.is_symlink() or not path.is_file():
        raise ValueError('source must be a regular non-symlink file')
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError('source checksum differs')
    return data


def write_new(path, data, mode):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def stage(source, output, *, head_sha, env_sha):
    source, output = Path(source), Path(output)
    if source.is_symlink() or not source.is_dir():
        raise ValueError('source directory must be real')
    head = checked_file(source / 'start-head.sh', head_sha)
    env = checked_file(source / 'env.sh', env_sha)
    if head.count(SOURCE_LINE) != 1:
        raise ValueError('head launcher must source env.sh exactly once')
    staged_head = head.replace(SOURCE_LINE, SOURCE_LINE + OVERRIDE, 1)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    write_new(output / 'env.sh', env, 0o600)
    write_new(output / 'start-head.sh', staged_head, 0o700)
    fd = os.open(output, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return {'source_head_sha256': head_sha, 'source_env_sha256': env_sha,
            'staged_head_sha256': digest(staged_head), 'staged_env_sha256': digest(env),
            'override': 'H3_TP2_FP8_FUSE_SWIGLU=0 after sourcing env.sh',
            'limitations': 'Not started or installed; actual rank-local patch state and service safety remain unverified.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--head-sha256', required=True)
    parser.add_argument('--env-sha256', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(stage(args.source_dir, args.output_dir, head_sha=args.head_sha256, env_sha=args.env_sha256),
                         indent=2))
    except (OSError, ValueError) as exc:
        print('stage-unfused-head:', exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
