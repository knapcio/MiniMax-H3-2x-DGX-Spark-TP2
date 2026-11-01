"""Experimental checkpoint session for a continuation loop; not wired into serving."""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from h3_tp2 import cont_checkpoint as storage
from h3_tp2 import cont_checkpoint_coord as coordination
from h3_tp2 import cont_checkpoint_state as state_check


logger = logging.getLogger('vllm_omni.h3_tp2.cont_checkpoint')
RESERVED_KEY = '__h3_checkpoint_state_schema'
OUTPUT_KEYS = frozenset(('height', 'width', 'fps', 'audio_sample_rate', 'preencode_mp4',
                         'preencode_batch_frames', 'video_codec_options', 'num_outputs'))


class Session:
    """Rank-coordinated state snapshots for an already verified request/deployment.

    Every participating rank must call the same methods in the same order.
    The caller must authenticate a complete deployment digest, own synchronized
    input tensors, and abort the request if a checkpoint operation fails. The
    caller must verify the actually active request-specific adapters on each
    rank and supply their complete digest, including adapter weights, loader
    behavior and scale, or explicitly supply None when no adapter is active.
    The separate execution-state digest must cover dynamic behavior such as
    persistent inactive wrappers and effective bindings, even with no adapter.
    Only a single output is supported: previously finished outputs are not saved.
    """

    @classmethod
    def begin(cls, root: Path, *, run_id: str, kwargs: dict, text_conditioning, options: dict,
              output: dict, deployment_sha256: str, dynamic_adapters_sha256: str | None,
              execution_state_sha256: str, max_bytes: int, group=None, device='cpu', channels=24,
              audio_channels=2, audio_dims=32):
        rank, world, device = coordination._group(group, device)
        error = None
        request_sha = None
        try:
            if not Path(root).is_absolute():
                raise ValueError('checkpoint root must be absolute')
            if not isinstance(kwargs, dict) or RESERVED_KEY in kwargs:
                raise ValueError('invalid request or reserved checkpoint key')
            if dynamic_adapters_sha256 is not None and (not isinstance(dynamic_adapters_sha256, str)
                                                        or not storage.SHA256.fullmatch(dynamic_adapters_sha256)):
                raise ValueError('active adapter identity must be SHA-256 or explicitly absent')
            if not isinstance(execution_state_sha256, str) or not storage.SHA256.fullmatch(execution_state_sha256):
                raise ValueError('verified dynamic execution-state identity must be SHA-256')
            dimensions = (kwargs.get('num_frames'), kwargs.get('latent_h'), kwargs.get('latent_w'),
                          channels, audio_channels, audio_dims)
            if any(type(value) is not int or value < 1 for value in dimensions):
                raise ValueError('invalid checkpoint state dimensions')
            if not isinstance(options, dict) or type(options.get('lock_audio')) is not bool:
                raise ValueError('invalid checkpoint audio-lock option')
            if not isinstance(output, dict) or set(output) != OUTPUT_KEYS:
                raise ValueError('checkpoint output configuration is incomplete')
            if (type(output['height']) is not int or type(output['width']) is not int
                    or output['height'] != kwargs['latent_h'] * 16 or output['width'] != kwargs['latent_w'] * 16
                    or type(output['fps']) is not int or output['fps'] != 24
                    or type(output['audio_sample_rate']) is not int or output['audio_sample_rate'] != 32000
                    or type(output['preencode_mp4']) is not bool or type(output['preencode_batch_frames']) is not int
                    or output['preencode_batch_frames'] < 1 or type(output['num_outputs']) is not int
                    or output['num_outputs'] != 1):
                raise ValueError('checkpoint output geometry, rate or encoding configuration differs')
            state_check.planned_window(kwargs['num_frames'], options['window_frames'], options['overlap_frames'], 1)
            identity_kwargs = {**kwargs, RESERVED_KEY: {'channels': channels, 'audio_channels': audio_channels,
                                                         'audio_dims': audio_dims, 'output': output,
                                                         'dynamic_adapters_sha256': dynamic_adapters_sha256,
                                                         'execution_state_sha256': execution_state_sha256}}
            request_sha = storage.fingerprint_request(identity_kwargs, text_conditioning, options,
                                                        deployment_sha256=deployment_sha256, max_bytes=max_bytes)
            storage._identity(run_id, request_sha, rank, world, 1)
        except Exception as exc:
            error = type(exc).__name__
            logger.warning('checkpoint request fingerprint failed on rank %d (%s)', rank, error)
        messages = coordination._gather({'rank': rank, 'ok': error is None, 'run_id': run_id,
                                         'request_sha256': request_sha, 'error': error}, group=group, device=device)
        if (len(messages) != world or any(not isinstance(item, dict) or type(item.get('rank')) is not int
                                          or item.get('rank') != i or item.get('ok') is not True
                                          or item.get('run_id') != run_id or item.get('request_sha256') != request_sha
                                          for i, item in enumerate(messages))):
            raise coordination.CheckpointCoordinationError('ranks could not agree on checkpoint request identity')
        assert request_sha is not None
        return cls(root, run_id, request_sha, kwargs['num_frames'], options['window_frames'],
                   options['overlap_frames'], kwargs['latent_h'], kwargs['latent_w'], channels,
                   audio_channels, audio_dims, options['lock_audio'], max_bytes, group, device)

    def __init__(self, root, run_id, request_sha, total_frames, full_window, overlap, latent_h, latent_w,
                 channels, audio_channels, audio_dims, audio_lock, max_bytes, group, device):  # noqa: ANN001
        self.root = Path(root)
        self.run_id, self.request_sha = run_id, request_sha
        self.total_frames, self.full_window, self.overlap = total_frames, full_window, overlap
        self.latent_h, self.latent_w, self.channels = latent_h, latent_w, channels
        self.audio_channels, self.audio_dims, self.audio_lock = audio_channels, audio_dims, audio_lock
        self.max_bytes, self.group, self.device = max_bytes, group, device

    def validate(self, tensors, window):  # noqa: ANN001
        state_check.validate_for_plan(tensors, window=window, total_frames=self.total_frames,
                                      full_window=self.full_window, overlap=self.overlap,
                                      latent_h=self.latent_h, latent_w=self.latent_w, channels=self.channels,
                                      audio_channels=self.audio_channels, audio_dims=self.audio_dims,
                                      require_audio_ref=self.audio_lock)

    def save_window(self, window, video, audio, wrapped_diffuse):  # noqa: ANN001
        """Capture only after cumulative AV and wrapped-match state are committed."""
        snapshot = {}
        try:
            snapshot = wrapped_diffuse.snapshot_state()
            snapshot['cumulative.video'] = video.detach().to('cpu', copy=True)
            snapshot['cumulative.audio'] = audio.detach().to('cpu', copy=True)
            self.validate(snapshot, window)
        except Exception as exc:
            logger.warning('local checkpoint snapshot failed (%s)', type(exc).__name__)
            snapshot = {}  # storage rejects this, and the coordinated failure reaches every rank
        return coordination.publish(self.root, run_id=self.run_id, request_sha256=self.request_sha,
                                    window=window, tensors=snapshot, max_bytes=self.max_bytes,
                                    group=self.group, device=self.device)

    def restore(self, previous_run_id, wrapped_diffuse, *, tensor_device):  # noqa: ANN001
        """Restore a common window, copy it to this fresh run ID, and return cumulative AV."""
        rank, world, _ = coordination._group(self.group, self.device)
        messages = coordination._gather({'rank': rank, 'run_id': self.run_id, 'previous_run_id': previous_run_id},
                                        group=self.group, device=self.device)
        if (len(messages) != world or any(not isinstance(item, dict) or item.get('rank') != i
                                          or item.get('run_id') != self.run_id
                                          or item.get('previous_run_id') != previous_run_id
                                          for i, item in enumerate(messages)) or previous_run_id == self.run_id):
            raise coordination.CheckpointCoordinationError('ranks disagree on source or new checkpoint run')
        restored = coordination.restore_latest(self.root, run_id=previous_run_id, request_sha256=self.request_sha,
                                                max_bytes=self.max_bytes, validate=self.validate,
                                                group=self.group, device=self.device)
        if restored is None:
            raise coordination.CheckpointCoordinationError('no valid checkpoint is common to every rank')
        window, tensors, _receipt = restored
        # A new ID is mandatory: a failed rank can have left an immutable
        # later window in the old run. Keep it and fork from the common state.
        coordination.publish(self.root, run_id=self.run_id, request_sha256=self.request_sha, window=window,
                             tensors=tensors, max_bytes=self.max_bytes, group=self.group, device=self.device)
        video = audio = None
        error = None
        try:
            video = tensors['cumulative.video'].to(device=tensor_device, copy=True)
            audio = tensors['cumulative.audio'].to(device=tensor_device, copy=True)
            wrapped_diffuse.restore_state({key: value for key, value in tensors.items() if key.startswith('match.')},
                                          device=tensor_device)
        except Exception as exc:
            error = type(exc).__name__
            logger.warning('checkpoint state restore failed on rank %d (%s)', rank, error)
        messages = coordination._gather({'rank': rank, 'run_id': self.run_id, 'window': window,
                                         'ok': error is None, 'error': error}, group=self.group, device=self.device)
        if (len(messages) != world or any(not isinstance(item, dict) or item.get('rank') != i
                                          or item.get('run_id') != self.run_id or item.get('window') != window
                                          or item.get('ok') is not True for i, item in enumerate(messages))):
            raise coordination.CheckpointCoordinationError('checkpoint state could not be restored on every rank')
        assert video is not None and audio is not None
        return window, video, audio
