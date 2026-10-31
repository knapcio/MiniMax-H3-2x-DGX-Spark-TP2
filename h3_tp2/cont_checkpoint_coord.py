"""Experimental rank coordination for continuation checkpoints; not wired into serving."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Callable

import torch
import torch.distributed as dist

from h3_tp2 import cont_checkpoint as storage


logger = logging.getLogger('vllm_omni.h3_tp2.cont_checkpoint')
MAX_PAYLOAD = 1_048_576


class CheckpointCoordinationError(RuntimeError):
    """At least one rank could not complete a coordinated checkpoint operation."""


def _group(group, device):
    if not dist.is_initialized():
        raise ValueError('distributed process group is not initialized')
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    if world < 1:
        raise ValueError('invalid distributed world size')
    device = torch.device(device)
    backend = str(dist.get_backend(group)).lower()
    if backend not in ('gloo', 'nccl'):
        raise ValueError('checkpoint exchange supports only Gloo or NCCL')
    if backend == 'nccl' and device.type != 'cuda':
        raise ValueError('NCCL checkpoint exchange requires a CUDA device')
    if backend == 'gloo' and device.type != 'cpu':
        raise ValueError('Gloo checkpoint exchange requires a CPU device')
    return rank, world, device


def _gather(value, *, group, device):  # noqa: ANN001
    """Collect bounded JSON without deserializing pickle from other ranks."""
    _rank, world, device = _group(group, device)
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError, RecursionError):
        # All ranks still enter the exchange; identity validation rejects this
        # sentinel on all of them instead of leaving peers in a collective.
        raw = b'{"serialization_error":true}'
    # Send the length first on every rank, including an oversized sender; no
    # rank raises before peers have entered the same initial collective.
    length = torch.tensor([len(raw)], dtype=torch.int64, device=device)
    lengths = [torch.zeros_like(length) for _ in range(world)]
    dist.all_gather(lengths, length, group=group)
    sizes = [int(item.item()) for item in lengths]
    if any(size < 1 or size > MAX_PAYLOAD for size in sizes):
        raise CheckpointCoordinationError('checkpoint exchange payload exceeds limit')
    payload = torch.zeros(max(sizes), dtype=torch.uint8, device=device)
    payload[:len(raw)] = torch.tensor(list(raw), dtype=torch.uint8, device=device)
    received = [torch.zeros_like(payload) for _ in range(world)]
    dist.all_gather(received, payload, group=group)
    return [json.loads(bytes(tensor[:size].cpu().tolist()))
            for tensor, size in zip(received, sizes, strict=True)]


def _verify_messages(messages, *, run_id, request_sha256, world, window=None):  # noqa: ANN001
    if len(messages) != world:
        raise CheckpointCoordinationError('checkpoint rank count differs')
    for rank, message in enumerate(messages):
        if (not isinstance(message, dict) or type(message.get('rank')) is not int or message.get('rank') != rank
                or message.get('run_id') != run_id
                or message.get('request_sha256') != request_sha256 or not isinstance(message.get('ok'), bool)
                or (window is not None and message.get('window') != window)):
            raise CheckpointCoordinationError('checkpoint rank, request, or window identity differs')


def publish(root: Path, *, run_id: str, request_sha256: str, window: int,
            tensors: dict[str, torch.Tensor], max_bytes: int, group=None, device='cpu') -> dict:
    """Each rank saves locally; all must report success before the caller advances."""
    rank, world, device = _group(group, device)
    local_receipt = None
    error = None
    try:
        local_receipt = storage.save(root, run_id=run_id, request_sha256=request_sha256, rank=rank,
                                     world_size=world, window=window, tensors=tensors, max_bytes=max_bytes)
    except Exception as exc:  # a peer still needs to hear that this rank failed
        error = type(exc).__name__
        logger.warning('checkpoint save failed locally on rank %d (%s)', rank, error)
    messages = _gather({'rank': rank, 'run_id': run_id, 'request_sha256': request_sha256, 'window': window,
                        'ok': error is None, 'error': error}, group=group, device=device)
    _verify_messages(messages, run_id=run_id, request_sha256=request_sha256, world=world, window=window)
    if not all(message['ok'] for message in messages):
        failed = [message['rank'] for message in messages if not message['ok']]
        raise CheckpointCoordinationError(f'checkpoint save failed on ranks {failed}')
    assert local_receipt is not None
    return local_receipt


def _local_windows(root: Path, run_id: str, rank: int) -> list[int]:
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError('checkpoint root is not a real directory')
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError('checkpoint root is not a real directory')
    directory = root / run_id
    if directory.is_symlink():
        raise ValueError('checkpoint run is not a real directory')
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError('checkpoint run is not a real directory')
    pattern = re.compile(rf'rank-{rank:04d}-window-(\d{{6,7}})\.json')
    windows = []
    for path in directory.glob(f'rank-{rank:04d}-window-*.json'):
        match = pattern.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise ValueError('invalid checkpoint receipt path')
        windows.append(int(match.group(1)))
        if len(windows) > 65_536:
            raise ValueError('too many checkpoint windows')
    return sorted(windows)


def restore_latest(root: Path, *, run_id: str, request_sha256: str, max_bytes: int,
                   validate: Callable[[dict[str, torch.Tensor], int], None], group=None, device='cpu'):
    """Load the newest checkpoint verified on every rank, falling back on corruption."""
    rank, world, device = _group(group, device)
    error = None
    windows = []
    try:
        if not callable(validate):
            raise ValueError('a state validator is required')
        storage._identity(run_id, request_sha256, rank, world, 1)
        windows = _local_windows(root, run_id, rank)
    except Exception as exc:
        error = type(exc).__name__
        logger.warning('checkpoint inventory failed locally on rank %d (%s)', rank, error)
    messages = _gather({'rank': rank, 'run_id': run_id, 'request_sha256': request_sha256,
                        'ok': error is None, 'windows': windows, 'error': error}, group=group, device=device)
    _verify_messages(messages, run_id=run_id, request_sha256=request_sha256, world=world)
    if not all(message['ok'] for message in messages):
        raise CheckpointCoordinationError('checkpoint inventory failed on at least one rank')
    if any(not isinstance(message.get('windows'), list) or any(type(w) is not int or not 1 <= w <= 1_000_000
                                                               for w in message['windows']) for message in messages):
        raise CheckpointCoordinationError('invalid checkpoint window inventory')
    common = set(messages[0]['windows'])
    for message in messages[1:]:
        common.intersection_update(message['windows'])
    for window in sorted(common, reverse=True):
        tensors = receipt = None
        error = None
        try:
            tensors, receipt = storage.load(root, run_id=run_id, request_sha256=request_sha256, rank=rank,
                                            world_size=world, window=window, max_bytes=max_bytes)
            validate(tensors, window)
        except Exception as exc:
            error = type(exc).__name__
            logger.warning('checkpoint window %d rejected locally on rank %d (%s)', window, rank, error)
            tensors = receipt = None
        statuses = _gather({'rank': rank, 'run_id': run_id, 'request_sha256': request_sha256, 'window': window,
                            'ok': error is None, 'error': error}, group=group, device=device)
        _verify_messages(statuses, run_id=run_id, request_sha256=request_sha256, world=world, window=window)
        if all(status['ok'] for status in statuses):
            assert tensors is not None and receipt is not None
            return window, tensors, receipt
    return None
