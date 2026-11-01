#!/usr/bin/env python3
"""Validate and snapshot two FL2VA keyframes for a single-window request.

This prepares files only; no HTTP/model access. The server and model still
need independent qualification for two-image requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import estimate_continuation as geometry

MAX_IMAGE_BYTES = 30 * 1024**2  # limit in the inspected server snapshot


def validate_request(extra, seconds):
    if not isinstance(extra, dict) or extra.get('task') != 'fl2va':
        raise ValueError('two keyframes require task fl2va')
    indices = extra.get('frame_indices')
    if not isinstance(indices, list) or len(indices) != 2 or any(type(x) is not int for x in indices) or indices != [0, -1]:
        raise ValueError('two keyframes require exact frame_indices [0, -1]')
    target = extra.get('target')
    if target is not None and not isinstance(target, dict):
        raise ValueError('target must be an object')
    duration = (target or {}).get('duration_seconds', extra.get('duration_seconds', extra.get('duration', seconds)))
    if isinstance(duration, bool) or duration is None:
        raise ValueError('effective duration must be explicit')
    try:
        total = geometry.aligned_frames(float(duration))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('invalid effective duration') from exc
    window = extra.get('continuation_window_frames', 277)
    overlap = extra.get('continuation_overlap_frames', 22)
    windows = geometry.plan(total, window, overlap)
    if len(windows) != 1:
        raise ValueError('two-keyframe client currently supports one continuation window only')
    requested = round(float(duration) * 24)
    return {'effective_seconds': float(duration), 'requested_frames': requested, 'aligned_frames': total,
            'alignment_tail_frames': total - requested, 'window_frames': window, 'overlap_frames': overlap}


def snapshot(source, destination):
    fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_IMAGE_BYTES:
            raise ValueError('each image must be a nonempty regular file at most 30 MiB')
        target = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256(); total = 0
        with os.fdopen(fd, 'rb', closefd=False) as src, os.fdopen(target, 'wb') as dst:
            while chunk := src.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise ValueError('image grew beyond 30 MiB')
                digest.update(chunk); dst.write(chunk)
        after = os.fstat(fd)
        identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if identity(before) != identity(after) or total != before.st_size:
            raise ValueError('image changed during snapshot')
        return {'bytes': total, 'sha256': digest.hexdigest()}
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('first', 'last', 'first-output', 'last-output', 'extra', 'metadata'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--seconds', required=True, type=float)
    args = parser.parse_args()
    try:
        info = validate_request(json.loads(args.extra.read_bytes()), args.seconds)
        first = snapshot(args.first, args.first_output)
        last = snapshot(args.last, args.last_output)
        with args.metadata.open('x', encoding='utf-8') as stream:
            json.dump({'first': first, 'last': last, **info}, stream, indent=2, allow_nan=False); stream.write('\n')
        return 0
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print('snapshot-first-last:', type(exc).__name__ + ':', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
