#!/usr/bin/env python3
"""Two-node CPU-only smoke of checkpoint receipt exchange; no model or GPU resume."""
import argparse
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile

import torch
import torch.distributed as dist


def load_checkpoint(path):
    spec = importlib.util.spec_from_file_location('distributed_checkpoint_test_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gather_json(value, world):
    """Exchange bounded JSON bytes without unpickling data from another rank."""
    encoded = json.dumps(value, separators=(',', ':'), sort_keys=True).encode()
    limit = 16_384
    if len(encoded) > limit:
        raise ValueError('test receipt payload too large')
    length = torch.tensor([len(encoded)], dtype=torch.int64)
    lengths = [torch.zeros_like(length) for _ in range(world)]
    dist.all_gather(lengths, length)
    if any(x.item() < 0 or x.item() > limit for x in lengths):
        raise ValueError('received invalid payload length')
    payload = torch.zeros(limit, dtype=torch.uint8)
    payload[:len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)
    received = [torch.zeros_like(payload) for _ in range(world)]
    dist.all_gather(received, payload)
    return [json.loads(bytes(tensor[:size.item()].tolist())) for tensor, size in zip(received, lengths, strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int)
    parser.add_argument('--port-file', type=Path)
    parser.add_argument('--module', type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available() or not dist.is_gloo_available():
        raise RuntimeError('test requires CUDA hidden and the Gloo CPU backend')
    if (args.rank == 0 and (args.port is not None or args.port_file is None)) or (
            args.rank == 1 and (args.port is None or args.port_file is not None)):
        raise ValueError('rank 0 requires a new port file; rank 1 requires its port')
    torch.set_num_threads(1)
    checkpoint = load_checkpoint(args.module)
    timeout = timedelta(seconds=45)
    store = dist.TCPStore(args.host, 0 if args.rank == 0 else args.port, 2, args.rank == 0,
                          timeout=timeout, wait_for_workers=False)
    if args.rank == 0:
        with args.port_file.open('x') as out:
            out.write(str(store.port) + '\n'); out.flush(); os.fsync(out.fileno())
    dist.init_process_group('gloo', store=store, rank=args.rank, world_size=2, timeout=timeout)
    run_id, request_sha = 'a' * 32, 'b' * 64
    try:
        with tempfile.TemporaryDirectory(prefix='h3-dist-checkpoint-cpu-') as temp:
            root = Path(temp)
            windows = (1, 2) if args.rank == 0 else (1, 3)
            for window in windows:
                checkpoint.save(root, run_id=run_id, request_sha256=request_sha, rank=args.rank, world_size=2,
                                window=window, tensors={'video': torch.tensor([args.rank, window], dtype=torch.float32)},
                                max_bytes=1024 * 1024)
            def read(wanted):
                return [checkpoint.load(root, run_id=run_id, request_sha256=request_sha, rank=args.rank,
                                        world_size=2, window=window, max_bytes=1024 * 1024)[1] for window in wanted]
            rank_receipts = gather_json(read(windows), 2)
            both = dict(enumerate(rank_receipts))
            if checkpoint.latest_common(both, run_id=run_id, request_sha256=request_sha, world_size=2) != 1:
                raise AssertionError('ranks failed to choose their common checkpoint')
            if args.rank == 1:
                path = root / run_id / 'rank-0001-window-000001.safetensors'
                with path.open('r+b') as out:
                    out.seek(-1, 2); original = out.read(1); out.seek(-1, 2); out.write(bytes([original[0] ^ 1]))
                try:
                    read((1,))
                except ValueError as exc:
                    if 'checksum' not in str(exc): raise
                else:
                    raise AssertionError('corrupted checkpoint was accepted')
                valid = read((3,))
            else:
                valid = read(windows)
            rank_receipts = gather_json(valid, 2)
            if checkpoint.latest_common(dict(enumerate(rank_receipts)), run_id=run_id,
                                        request_sha256=request_sha, world_size=2) is not None:
                raise AssertionError('ranks selected a non-common or corrupted checkpoint')
            dist.barrier()
            print(json.dumps({'host': socket.gethostname(), 'rank': args.rank, 'backend': 'gloo',
                              'common_before_corruption': 1, 'common_after_corruption': None,
                              'cuda_available': torch.cuda.is_available()}), flush=True)
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
