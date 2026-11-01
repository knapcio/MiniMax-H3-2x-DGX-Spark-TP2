"""Experimental checkpointed continuation loop; not wired into H3 serving.

The caller owns and synchronizes the full request, authenticates the deployment
on both ranks, supplies a fresh run ID, and must abort on checkpoint failure.
All ranks must enter the same checkpoint mode with the same working process
group and corresponding collective device.
Durable job recovery, cancellation and real-model/GPU validation are not solved.
"""
from __future__ import annotations

import torch

from h3_tp2 import cont_checkpoint as storage
from h3_tp2 import cont_checkpoint_coord as coordination
from h3_tp2 import cont_checkpoint_session as sessions
from h3_tp2 import cont_match as match


def run(diffuse, kwargs, *, api, options, text_conditioning=None, checkpoint=None):  # noqa: ANN001
    """Run the same continuation hand-over with optional, explicit rank checkpoints.

    ``api`` is the vLLM-Omni continuation module. ``checkpoint`` contains the
    arguments for ``Session.begin`` other than kwargs/options/conditioning,
    and optionally ``previous_run_id`` and ``tensor_device``. This prototype
    intentionally has no environment switch and does not modify serving.
    """
    if checkpoint is not None and not isinstance(checkpoint, dict):
        raise ValueError('every rank must supply checkpoint settings as an object')
    setup = dict(checkpoint) if checkpoint is not None else None
    previous = setup.pop('previous_run_id', None) if setup is not None else None
    tensor_device = setup.pop('tensor_device', None) if setup is not None else None
    error = None
    try:
        if not isinstance(options, dict) or set(options) != storage.REQUEST_OPTIONS:
            raise ValueError('complete continuation options are required')
        if not options['crossfade'] and not options['edit']:
            raise ValueError('the upstream-only continuation branch is not supported by this prototype')
        if setup is not None:
            required = {'root', 'run_id', 'output', 'deployment_sha256', 'dynamic_adapters_sha256', 'max_bytes'}
            allowed = required | {'group', 'device', 'channels', 'audio_channels', 'audio_dims'}
            if not required.issubset(setup) or set(setup) - allowed:
                raise ValueError('checkpoint settings are incomplete or unexpected')
            if previous is not None and tensor_device is None:
                raise ValueError('a target tensor device is required for restore')
        window_frames, overlap_frames = options['window_frames'], options['overlap_frames']
        wrapped = match.wrap_diffuse(diffuse, apply=options['apply_match'], overlap_v_hint=api._video_t(overlap_frames),
                                     retries=options['retries'], max_ratio=options['max_ratio'], lock=options['lock'],
                                     lock_strength=options['lock_strength'], lock_ramp=options['lock_ramp'],
                                     lock_audio=options['lock_audio'], lock_std=options['lock_std'], lock_ref=options['lock_ref'])
        windows = api.plan_continuation_windows(kwargs['num_frames'], window_frames, overlap_frames)
        if text_conditioning is not None and len(text_conditioning) != len(windows):
            raise api.OmniClientError('MiniMax H3 requires one text conditioning per continuation window')
        media_time_origin = max(item[0].shape[0] for item in text_conditioning) if text_conditioning else None
        source_rows = kwargs.get('locked_audio_rows')
        source = None if source_rows is None else source_rows.reshape(2, kwargs['audio_t'], 32)
    except Exception as exc:
        error = exc
    if setup is not None:
        group, device = setup.get('group'), setup.get('device', 'cpu')
        rank, world, device = coordination._group(group, device)
        messages = coordination._gather({'rank': rank, 'ok': error is None,
                                          'previous_run_id': previous, 'error': type(error).__name__ if error else None},
                                         group=group, device=device)
        if (len(messages) != world or any(not isinstance(item, dict) or item.get('rank') != index
                                          or item.get('ok') is not True or item.get('previous_run_id') != previous
                                          for index, item in enumerate(messages))):
            raise coordination.CheckpointCoordinationError('ranks could not agree on continuation preflight')
    elif error is not None:
        raise error
    video = audio = None
    completed = 0
    session = None
    if setup is not None:
        session = sessions.Session.begin(kwargs=kwargs, text_conditioning=text_conditioning, options=options, **setup)
        if previous is not None:
            completed, video, audio = session.restore(previous, wrapped, tensor_device=tensor_device)
            if not 1 <= completed <= len(windows):
                raise ValueError('restored window is outside the continuation plan')
    for index, window in enumerate(windows):
        if index < completed:
            continue
        frames = window.end - window.start
        args = {**kwargs, 'num_frames': frames, 'latent_t': api._video_t(frames),
                'audio_t': window.audio_end - window.audio_start, 'temporal_offset': window.start * (40.0 / 24.0)}
        if options['seed_step'] and index and kwargs.get('seed') is not None:
            args['seed'] = int(kwargs['seed']) + index * options['seed_step']
        if index and kwargs.get('task') in ('fl2va', 't2va'):
            args.update({'task': 't2va', 'keyframe_frame_indices': None, 'visual_condition': None,
                         'visual_condition_shape': None, 'visual_condition_shapes': None, 'ref_blocks': None,
                         'audio_condition': None, 'ref_audio_t': None, 'audio_condition_lengths': None})
        if text_conditioning is not None:
            args['text_embeddings'], args['text_tags'] = text_conditioning[index]
            args['media_time_origin'] = media_time_origin
        if source is not None:
            args['locked_audio_rows'] = source[:, window.audio_start:window.audio_end].reshape(-1, 32)
        overlap_v = api._video_t(window.overlap) if window.overlap else 0
        if video is not None and audio is not None:
            guide_video = api.minimax_h3_patchify_video_latent(video[:, :, -overlap_v:], patch_size=(1, 2, 2))
            guide_audio = api.minimax_h3_pack_audio_latent(audio[..., -window.overlap_audio_t:])
            if options['edit'] != 2:
                for name, guide in (('visual_condition', guide_video), ('audio_condition', guide_audio)):
                    existing = kwargs.get(name)
                    args[name] = guide if existing is None else torch.cat((existing, guide), dim=0)
                args['visual_condition_shapes'] = [*(kwargs.get('visual_condition_shapes') or []),
                                                   (overlap_v, kwargs['latent_h'], kwargs['latent_w'])]
                args['audio_condition_lengths'] = [*(kwargs.get('audio_condition_lengths') or []), window.overlap_audio_t]
                args['ref_blocks'] = [*(kwargs.get('ref_blocks') or []),
                                      {'kind': 'latent_guide', 'latent_t': overlap_v, 'latent_h': kwargs['latent_h'],
                                       'latent_w': kwargs['latent_w'], 'ref_audio_t': window.overlap_audio_t}]
            if options['edit']:
                device = video.device
                rows_per_frame = (video.shape[3] // 2) * (video.shape[4] // 2)
                n_rows, n_guide = args['latent_t'] * rows_per_frame, guide_video.shape[0]
                clean = torch.zeros((n_rows, guide_video.shape[1]), device=device, dtype=torch.float32)
                clean[:n_guide] = guide_video.float()
                mask = torch.ones(n_rows, device=device, dtype=torch.float32); mask[:n_guide] = 0.0
                args['video_edit_clean_rows'], args['video_edit_mask_rows'], args['video_edit_restore_mask_rows'] = clean, mask, mask.clone()
                ov_a, a_t = window.overlap_audio_t, args['audio_t']
                clean_a = torch.zeros((2 * a_t, guide_audio.shape[1]), device=device, dtype=torch.float32)
                mask_a = torch.ones(2 * a_t, device=device, dtype=torch.float32)
                for channel in range(2):
                    clean_a[channel * a_t:channel * a_t + ov_a] = guide_audio[channel * ov_a:(channel + 1) * ov_a].float()
                    mask_a[channel * a_t:channel * a_t + ov_a] = 0.0
                args['audio_edit_clean_rows'], args['audio_edit_mask_rows'], args['audio_edit_restore_mask_rows'] = clean_a, mask_a, mask_a.clone()
        sampled_video, sampled_audio = wrapped(**args)
        if video is None:
            video, audio = sampled_video, sampled_audio
        else:
            video = (match.crossfade_tail(video, sampled_video, overlap_v) if options['crossfade']
                     else torch.cat((video, sampled_video[:, :, overlap_v:]), dim=2))
            audio = torch.cat((audio, sampled_audio[..., window.overlap_audio_t:]), dim=-1)
        if session is not None:
            session.save_window(index + 1, video, audio, wrapped)
    if video is None or audio is None or video.shape[2] != kwargs['latent_t'] or audio.shape[-1] != kwargs['audio_t']:
        raise RuntimeError('MiniMax H3 continuation produced inconsistent cumulative AV lengths')
    return video, audio
