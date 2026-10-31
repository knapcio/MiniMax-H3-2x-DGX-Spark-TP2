"""Experimental building blocks for continuation checkpoints; not wired into serving.

The caller must fingerprint every result-affecting request and runtime input,
capture all continuation state on each rank, and coordinate a common completed
window before resuming. This module by itself cannot make a render resumable.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch
from safetensors.torch import load_file, save_file


VERSION = 1
NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_.-]{0,127}$')
SHA256 = re.compile(r'^[0-9a-f]{64}$')
RUN_ID = re.compile(r'^[0-9a-f]{32}$')


def _hash_file(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _validate_tensors(tensors: dict[str, torch.Tensor], max_bytes: int) -> int:
    if (not isinstance(tensors, dict) or not tensors or len(tensors) > 1024 or not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool) or max_bytes < 1):
        raise ValueError('nonempty tensors and a positive size limit are required')
    total = 0
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not NAME.fullmatch(name) or not isinstance(tensor, torch.Tensor):
            raise ValueError('tensor names or values are invalid')
        if tensor.layout != torch.strided or tensor.numel() == 0:
            raise ValueError('only nonempty strided tensors are supported')
        total += tensor.numel() * tensor.element_size()
    if total > max_bytes:
        raise ValueError('checkpoint tensor size exceeds limit')
    return total


def fingerprint_inputs(metadata: dict[str, Any], tensors: dict[str, torch.Tensor], *, max_bytes: int) -> str:
    """Fingerprint supplied inputs with shape/dtype; completeness is the caller's responsibility."""
    if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
        raise ValueError('metadata must have string keys')
    _validate_tensors(tensors, max_bytes)
    scalar_bytes = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    hasher = hashlib.sha256(b'h3-continuation-input-v1\0')
    hasher.update(len(scalar_bytes).to_bytes(8, 'big')); hasher.update(scalar_bytes)
    for name, tensor in sorted(tensors.items()):
        cpu = tensor.detach().to('cpu').contiguous()
        info = json.dumps([name, str(cpu.dtype), list(cpu.shape)], separators=(',', ':')).encode()
        hasher.update(len(info).to_bytes(8, 'big')); hasher.update(info)
        data = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        hasher.update(len(data).to_bytes(8, 'big')); hasher.update(data)
    return hasher.hexdigest()


def _identity(run_id: str, request_sha256: str, rank: int, world_size: int, window: int) -> None:
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError('run ID must be 32 lowercase hexadecimal characters')
    if not isinstance(request_sha256, str) or not SHA256.fullmatch(request_sha256):
        raise ValueError('request fingerprint must be SHA-256')
    if (any(not isinstance(x, int) or isinstance(x, bool) for x in (rank, world_size, window))
            or not 1 <= world_size <= 1024 or not 0 <= rank < world_size or not 1 <= window <= 1_000_000):
        raise ValueError('invalid rank, world size, or completed-window number')


def _directory(root: Path, run_id: str, *, create: bool) -> Path:
    root = Path(root).absolute()
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('checkpoint root must be a real directory')
    directory = root / run_id
    if create:
        directory.mkdir(exist_ok=True, mode=0o700)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('checkpoint run must be a real directory')
    return directory


def _paths(directory: Path, rank: int, window: int) -> tuple[Path, Path]:
    stem = f'rank-{rank:04d}-window-{window:06d}'
    return directory / (stem + '.safetensors'), directory / (stem + '.json')


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(root: Path, *, run_id: str, request_sha256: str, rank: int, world_size: int, window: int,
         tensors: dict[str, torch.Tensor], max_bytes: int) -> dict[str, Any]:
    """Publish one immutable rank/window checkpoint; never replaces an earlier file."""
    _identity(run_id, request_sha256, rank, world_size, window)
    total = _validate_tensors(tensors, max_bytes)
    directory = _directory(root, run_id, create=True)
    data_path, receipt_path = _paths(directory, rank, window)
    with (directory / '.write.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.lexists(data_path) or os.path.lexists(receipt_path):
            raise FileExistsError('checkpoint data or receipt already exists')
        stat = os.statvfs(directory)
        if stat.f_bavail * stat.f_frsize < total + 64 * 1024**2:
            raise ValueError('insufficient free space for checkpoint')
        cpu = {name: tensor.detach().to('cpu', copy=True).contiguous() for name, tensor in tensors.items()}
        specs = {name: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                        'bytes': tensor.numel() * tensor.element_size()} for name, tensor in sorted(cpu.items())}
        data_fd, data_temp_name = tempfile.mkstemp(prefix='.h3-state-', suffix='.safetensors', dir=directory)
        os.close(data_fd)
        try:
            receipt_fd, receipt_temp_name = tempfile.mkstemp(prefix='.h3-receipt-', suffix='.json', dir=directory)
        except BaseException:
            Path(data_temp_name).unlink(missing_ok=True)
            raise
        data_temp, receipt_temp = Path(data_temp_name), Path(receipt_temp_name)
        data_published = receipt_published = False
        try:
            save_file(cpu, str(data_temp))
            with data_temp.open('rb') as stream:
                os.fsync(stream.fileno())
            receipt = {'version': VERSION, 'run_id': run_id, 'request_sha256': request_sha256,
                       'rank': rank, 'world_size': world_size, 'window': window,
                       'data_sha256': _hash_file(data_temp), 'data_bytes': data_temp.stat().st_size,
                       'tensor_bytes': total, 'tensors': specs}
            receipt_stream = os.fdopen(receipt_fd, 'w')
            receipt_fd = -1
            with receipt_stream as stream:
                json.dump(receipt, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
                stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            os.link(data_temp, data_path); data_published = True
            _sync_directory(directory)
            os.link(receipt_temp, receipt_path); receipt_published = True
            _sync_directory(directory)
            return receipt
        except BaseException:
            if data_published and not receipt_published and os.path.samestat(data_path.stat(), data_temp.stat()):
                data_path.unlink()
                _sync_directory(directory)
            raise
        finally:
            if receipt_fd >= 0:
                os.close(receipt_fd)
            data_temp.unlink(missing_ok=True)
            receipt_temp.unlink(missing_ok=True)


def load(root: Path, *, run_id: str, request_sha256: str, rank: int, world_size: int, window: int,
         max_bytes: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load only a complete, hash-checked rank/window pair; no pickle execution."""
    _identity(run_id, request_sha256, rank, world_size, window)
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError('invalid size limit')
    directory = _directory(root, run_id, create=False)
    data_path, receipt_path = _paths(directory, rank, window)
    if data_path.is_symlink() or receipt_path.is_symlink() or not data_path.is_file() or not receipt_path.is_file():
        raise ValueError('checkpoint pair is missing or symbolic')
    if receipt_path.stat().st_size > 1024 * 1024:
        raise ValueError('checkpoint receipt is too large')
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    identity = {'version': VERSION, 'run_id': run_id, 'request_sha256': request_sha256,
                'rank': rank, 'world_size': world_size, 'window': window}
    if not isinstance(receipt, dict) or type(receipt.get('version')) is not int:
        raise ValueError('checkpoint identity differs')
    _identity(receipt.get('run_id'), receipt.get('request_sha256'), receipt.get('rank'),
              receipt.get('world_size'), receipt.get('window'))
    if any(receipt.get(k) != v for k, v in identity.items()):
        raise ValueError('checkpoint identity differs')
    size = data_path.stat().st_size
    if size > max_bytes + 1024 * 1024 or receipt.get('data_bytes') != size:
        raise ValueError('checkpoint data size differs or exceeds limit')
    if receipt.get('data_sha256') != _hash_file(data_path):
        raise ValueError('checkpoint data checksum differs')
    # Detach from the safetensors file mapping so later file modifications do
    # not silently change a state already handed to the caller.
    tensors = {name: tensor.clone() for name, tensor in load_file(str(data_path), device='cpu').items()}
    total = _validate_tensors(tensors, max_bytes)
    specs = {name: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                    'bytes': tensor.numel() * tensor.element_size()} for name, tensor in sorted(tensors.items())}
    if receipt.get('tensor_bytes') != total or receipt.get('tensors') != specs:
        raise ValueError('checkpoint tensor metadata differs')
    if receipt_bytes != receipt_path.read_bytes() or receipt.get('data_sha256') != _hash_file(data_path):
        raise ValueError('checkpoint changed during loading')
    return tensors, receipt


def latest_common(receipts_by_rank: dict[int, list[dict[str, Any]]], *, run_id: str,
                  request_sha256: str, world_size: int) -> int | None:
    """Find the highest common window among independently verified rank receipts."""
    _identity(run_id, request_sha256, 0, world_size, 1)
    if set(receipts_by_rank) != set(range(world_size)):
        raise ValueError('every rank must provide its independently verified receipts')
    common = None
    for rank, receipts in receipts_by_rank.items():
        windows = set()
        for item in receipts:
            if not isinstance(item, dict):
                raise ValueError('invalid receipt')
            window = item.get('window')
            _identity(item.get('run_id'), item.get('request_sha256'), item.get('rank'), item.get('world_size'), window)
            if type(item.get('version')) is not int or any(item.get(k) != v for k, v in {'version': VERSION, 'run_id': run_id,
                    'request_sha256': request_sha256, 'rank': rank, 'world_size': world_size}.items()):
                raise ValueError('rank receipt identity differs')
            if window in windows:
                raise ValueError('duplicate window on rank')
            windows.add(window)
        common = windows if common is None else common.intersection(windows)
    return max(common) if common else None
