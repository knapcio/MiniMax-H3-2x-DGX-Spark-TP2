#!/usr/bin/env python3
"""Two-node CPU-only smoke of checkpoint receipt exchange; no model or GPU resume."""
import argparse
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

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
    parser.add_argument('--package-root', type=Path, help='also test the staged checkpoint coordinator')
    parser.add_argument('--match-module', type=Path, help='also test continuation match-state restoration')
    parser.add_argument('--state-module', type=Path, help='also validate H3-shaped continuation state')
    parser.add_argument('--session', action='store_true', help='also exercise the staged checkpoint session')
    parser.add_argument('--loop-module', type=Path, help='also test the staged continuation loop against the actual patched loop')
    parser.add_argument('--crash-test', action='store_true', help='terminate rank 1 during a CPU collective')
    args = parser.parse_args()
    if torch.cuda.is_available() or not dist.is_gloo_available():
        raise RuntimeError('test requires CUDA hidden and the Gloo CPU backend')
    if (args.rank == 0 and (args.port is not None or args.port_file is None)) or (
            args.rank == 1 and (args.port is None or args.port_file is not None)):
        raise ValueError('rank 0 requires a new port file; rank 1 requires its port')
    torch.set_num_threads(1)
    checkpoint = load_checkpoint(args.module)
    coordinator = None
    if args.package_root is not None:
        sys.path.insert(0, str(args.package_root))
        from h3_tp2 import cont_checkpoint_coord as coordinator
    if args.crash_test and coordinator is None:
        raise ValueError('crash test requires the staged coordinator package')
    if args.match_module is not None and coordinator is None:
        raise ValueError('match-state test requires the staged coordinator package')
    if args.state_module is not None and args.match_module is None:
        raise ValueError('shape test requires match-state restoration')
    if args.session and args.state_module is None:
        raise ValueError('session test requires the state-shape test')
    if args.loop_module is not None and not args.session:
        raise ValueError('loop test requires the checkpoint session test')
    match = load_checkpoint(args.match_module) if args.match_module is not None else None
    state_check = load_checkpoint(args.state_module) if args.state_module is not None else None
    session_module = None
    if args.session:
        from h3_tp2 import cont_checkpoint_session as session_module
    loop_test = None
    if args.loop_module is not None:
        spec = importlib.util.spec_from_file_location('distributed_h3_loop_test', args.loop_module)
        loop_test = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loop_test
        spec.loader.exec_module(loop_test)
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
            if args.crash_test:
                dist.barrier()
                if args.rank == 1:
                    os._exit(17)  # intentional loss of only this test process
                started = time.monotonic()
                try:
                    coordinator.publish(root, run_id=run_id, request_sha256=request_sha, window=1,
                                        tensors={'video': torch.tensor([1.0])}, max_bytes=1024 * 1024)
                except RuntimeError:
                    print(json.dumps({'host': socket.gethostname(), 'rank': args.rank, 'peer_lost': True,
                                      'seconds_to_error': round(time.monotonic() - started, 3),
                                      'cuda_available': torch.cuda.is_available()}), flush=True)
                    return 0
                raise AssertionError('lost rank was treated as a committed checkpoint')
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
            if coordinator is not None:
                coordinator_smoke(coordinator, args.rank, root)
            if match is not None:
                match_state_smoke(coordinator, match, args.rank, root)
            if state_check is not None:
                realistic_state_smoke(coordinator, match, state_check, args.rank, root)
            if session_module is not None:
                session_smoke(coordinator, session_module, match, state_check, args.rank, root)
            if loop_test is not None:
                loop_session_smoke(coordinator, loop_test, args.rank, root)
            dist.barrier()
            print(json.dumps({'host': socket.gethostname(), 'rank': args.rank, 'backend': 'gloo',
                              'common_before_corruption': 1, 'common_after_corruption': None,
                              'coordinator_checked': coordinator is not None,
                              'match_state_checked': match is not None,
                              'h3_shape_checked': state_check is not None,
                              'session_checked': session_module is not None,
                              'loop_checked': loop_test is not None,
                              'cuda_available': torch.cuda.is_available()}), flush=True)
    finally:
        dist.destroy_process_group()
    return 0


def coordinator_smoke(coordinator, rank, root):
    """A partial save must fall back, then fork to a fresh run before advancing."""
    run_id, fork_id, request_sha = 'c' * 32, 'e' * 32, 'd' * 64
    common = {'run_id': run_id, 'request_sha256': request_sha, 'max_bytes': 1024 * 1024}
    def state(window):
        return {'video': torch.tensor([rank, window], dtype=torch.float32)}
    def validate(tensors, window):
        if set(tensors) != {'video'} or not torch.equal(tensors['video'], state(window)['video']):
            raise ValueError('state contents do not match window and rank')
    coordinator.publish(root, window=1, tensors=state(1), **common)
    original_save = coordinator.storage.save
    try:
        if rank == 1:
            def fail_second(*_args, **_kwargs):
                raise OSError('injected rank-1 checkpoint failure')
            coordinator.storage.save = fail_second
        try:
            coordinator.publish(root, window=2, tensors=state(2), **common)
        except coordinator.CheckpointCoordinationError:
            pass
        else:
            raise AssertionError('partial checkpoint save was treated as committed')
    finally:
        coordinator.storage.save = original_save
    restored = coordinator.restore_latest(root, validate=validate, **common)
    if restored is None or restored[0] != 1:
        raise AssertionError('partial save did not fall back to the prior common window')
    # Existing window-2 files are immutable. Copy the restored state to a new
    # run ID on both ranks before sampling and publishing the next window.
    fork = {**common, 'run_id': fork_id}
    coordinator.publish(root, window=1, tensors=restored[1], **fork)
    coordinator.publish(root, window=2, tensors=state(2), **fork)
    if rank == 1:
        path = root / fork_id / 'rank-0001-window-000002.safetensors'
        with path.open('r+b') as out:
            out.seek(-1, 2); original = out.read(1); out.seek(-1, 2); out.write(bytes([original[0] ^ 1]))
    dist.barrier()
    fallback = coordinator.restore_latest(root, validate=validate, **fork)
    if fallback is None or fallback[0] != 1:
        raise AssertionError('corrupt latest checkpoint did not fall back on both ranks')
    if rank == 1:
        path = root / fork_id / 'rank-0001-window-000001.safetensors'
        with path.open('r+b') as out:
            out.seek(-1, 2); original = out.read(1); out.seek(-1, 2); out.write(bytes([original[0] ^ 1]))
    dist.barrier()
    if coordinator.restore_latest(root, validate=validate, **fork) is not None:
        raise AssertionError('corruption on one rank was accepted by the group')
    mismatched = {**common, 'run_id': 'f' * 32, 'request_sha256': request_sha if rank == 0 else '9' * 64}
    try:
        coordinator.publish(root, window=1, tensors=state(1), **mismatched)
    except coordinator.CheckpointCoordinationError:
        pass
    else:
        raise AssertionError('different request fingerprints were accepted')
    payload = {'rank': rank, 'value': 'ok' if rank == 0 else object()}
    exchanged = coordinator._gather(payload, group=None, device='cpu')
    if exchanged[1] != {'serialization_error': True}:
        raise AssertionError('a rank serialization error did not reach both processes')
    try:
        coordinator.restore_latest(root, validate=validate if rank == 0 else None, **fork)
    except coordinator.CheckpointCoordinationError:
        pass
    else:
        raise AssertionError('a missing validator on one rank was accepted')


def match_state_smoke(coordinator, match, rank, root):
    """Compare uninterrupted and recovered synthetic AV latents on both ranks."""
    run_id, fork_id, request_sha = '1' * 32, '2' * 32, '3' * 64
    common = {'run_id': run_id, 'request_sha256': request_sha, 'max_bytes': 1024 * 1024}
    first_v = torch.arange(64, dtype=torch.float32).reshape(1, 2, 8, 2, 2) / 20 + rank
    next_v = first_v + 3; next_v[:, :, :2] = first_v[:, :, -2:]
    first_a = torch.arange(96, dtype=torch.float32).reshape(2, 6, 8) / 20 + rank
    next_a = first_a + 3; next_a[..., :2] = first_a[..., -2:]
    def wrapper(samples):
        values = iter(samples)
        return match.wrap_diffuse(lambda **_kw: tuple(x.clone() for x in next(values)), apply=False,
                                  lock='first', lock_audio=True, lock_ramp=.25, max_ratio=100)
    def validate(tensors, window):
        if (tensors.get('match.index') is None or tensors['match.index'].item() != window
                or tuple(tensors['cumulative.video'].shape) != ((1, 2, 8, 2, 2) if window == 1 else (1, 2, 14, 2, 2))
                or tuple(tensors['cumulative.audio'].shape) != ((2, 6, 8) if window == 1 else (2, 6, 14))):
            raise ValueError('synthetic cumulative state differs from its window')
    run = wrapper(((first_v, first_a), (next_v, next_a)))
    first = run(seed=1)
    first_state = {**run.snapshot_state(), 'cumulative.video': first[0], 'cumulative.audio': first[1]}
    coordinator.publish(root, window=1, tensors=first_state, **common)
    args = {'seed': 2, 'ref_blocks': [{'kind': 'latent_guide', 'latent_t': 2}], 'audio_condition_lengths': [2]}
    second = run(**args)
    cumulative_video = torch.cat((first[0], second[0][:, :, 2:]), dim=2)
    cumulative_audio = torch.cat((first[1], second[1][..., 2:]), dim=-1)
    second_state = {**run.snapshot_state(), 'cumulative.video': cumulative_video, 'cumulative.audio': cumulative_audio}
    original_save = coordinator.storage.save
    try:
        if rank == 1:
            def fail_second(*_args, **_kwargs):
                raise OSError('injected rank-1 checkpoint failure')
            coordinator.storage.save = fail_second
        try:
            coordinator.publish(root, window=2, tensors=second_state, **common)
        except coordinator.CheckpointCoordinationError:
            pass
        else:
            raise AssertionError('partial synthetic AV checkpoint was accepted')
    finally:
        coordinator.storage.save = original_save
    restored = coordinator.restore_latest(root, validate=validate, **common)
    if restored is None or restored[0] != 1:
        raise AssertionError('synthetic AV state did not roll back')
    resume = wrapper(((next_v, next_a),))
    resume.restore_state({k: v for k, v in restored[1].items() if k.startswith('match.')}, device='cpu')
    again = resume(**args)
    resumed_video = torch.cat((restored[1]['cumulative.video'], again[0][:, :, 2:]), dim=2)
    resumed_audio = torch.cat((restored[1]['cumulative.audio'], again[1][..., 2:]), dim=-1)
    if not torch.equal(cumulative_video, resumed_video) or not torch.equal(cumulative_audio, resumed_audio):
        raise AssertionError('resumed synthetic AV differs from uninterrupted output')
    fork = {**common, 'run_id': fork_id}
    coordinator.publish(root, window=1, tensors=restored[1], **fork)
    resumed_state = {**resume.snapshot_state(), 'cumulative.video': resumed_video, 'cumulative.audio': resumed_audio}
    coordinator.publish(root, window=2, tensors=resumed_state, **fork)
    final = coordinator.restore_latest(root, validate=validate, **fork)
    if final is None or final[0] != 2 or not torch.equal(final[1]['cumulative.video'], cumulative_video):
        raise AssertionError('the resumed synthetic AV checkpoint was not reusable')


def realistic_state_smoke(coordinator, match, checker, rank, root):
    """Use the deployed H3 temporal formulas with synthetic CPU latents."""
    storage = coordinator.storage
    options = {key: None for key in storage.REQUEST_OPTIONS}
    options.update({'window_frames': 107, 'overlap_frames': 22, 'seed_step': 1, 'lock_audio': True})
    request = {'seed': 1101, 'num_frames': 192, 'latent_h': 2, 'latent_w': 2,
               'text_embeddings': torch.tensor([[1., 2.]]), 'ref_blocks': [{'kind': 'image'}]}
    request_sha = storage.fingerprint_request(request, [(torch.tensor([[1.]]), torch.tensor([2]))], options,
                                              deployment_sha256='a' * 64, max_bytes=1024 * 1024)
    values = coordinator._gather({'rank': rank, 'sha': request_sha}, group=None, device='cpu')
    if len({value['sha'] for value in values}) != 1:
        raise AssertionError('synthetic logical request fingerprints differ across ranks')
    run_id, fork_id = '4' * 32, '5' * 32
    common = {'run_id': run_id, 'request_sha256': request_sha, 'max_bytes': 1024 * 1024}
    video_t = checker.video_t(107); audio_first = checker.audio_t(107)
    video_overlap = checker.video_t(22); audio_overlap = checker.audio_t(107) - checker.audio_t(85)
    first_v = torch.arange(2 * video_t * 2 * 2, dtype=torch.float32).reshape(1, 2, video_t, 2, 2) / 100 + rank
    next_v = first_v + 2; next_v[:, :, :video_overlap] = first_v[:, :, -video_overlap:]
    first_a = torch.arange(2 * 6 * audio_first, dtype=torch.float32).reshape(2, 6, audio_first) / 100 + rank
    next_a = first_a + 2; next_a[..., :audio_overlap] = first_a[..., -audio_overlap:]
    def wrapper(samples):
        values = iter(samples)
        return match.wrap_diffuse(lambda **_kw: tuple(x.clone() for x in next(values)), apply=False,
                                  lock='first', lock_audio=True, lock_ramp=.25, max_ratio=100)
    def validate(tensors, window):
        checker.validate_for_plan(tensors, window=window, total_frames=192, full_window=107, overlap=22,
                                  latent_h=2, latent_w=2, channels=2, audio_channels=2, audio_dims=6)
    run = wrapper(((first_v, first_a), (next_v, next_a)))
    first = run(seed=1101)
    state1 = {**run.snapshot_state(), 'cumulative.video': first[0], 'cumulative.audio': first[1]}
    validate(state1, 1); coordinator.publish(root, window=1, tensors=state1, **common)
    args = {'seed': 1102, 'ref_blocks': [{'kind': 'latent_guide', 'latent_t': video_overlap}],
            'audio_condition_lengths': [audio_overlap]}
    second = run(**args)
    total_v = torch.cat((first[0], second[0][:, :, video_overlap:]), dim=2)
    total_a = torch.cat((first[1], second[1][..., audio_overlap:]), dim=-1)
    state2 = {**run.snapshot_state(), 'cumulative.video': total_v, 'cumulative.audio': total_a}
    validate(state2, 2)
    if rank == 1:
        state2 = {name: tensor.clone() for name, tensor in state2.items()}
        state2['match.prev'][:, :, -1, 0, 0] += 1  # valid file hash, semantically inconsistent tail
    coordinator.publish(root, window=2, tensors=state2, **common)
    restored = coordinator.restore_latest(root, validate=validate, **common)
    if restored is None or restored[0] != 1:
        raise AssertionError('semantic mismatch did not trigger a common rollback')
    resumed = wrapper(((next_v, next_a),))
    resumed.restore_state({key: value for key, value in restored[1].items() if key.startswith('match.')}, device='cpu')
    again = resumed(**args)
    out_v = torch.cat((restored[1]['cumulative.video'], again[0][:, :, video_overlap:]), dim=2)
    out_a = torch.cat((restored[1]['cumulative.audio'], again[1][..., audio_overlap:]), dim=-1)
    if not torch.equal(total_v, out_v) or not torch.equal(total_a, out_a):
        raise AssertionError('H3-shaped synthetic recovery differs from uninterrupted output')
    fork = {**common, 'run_id': fork_id}
    coordinator.publish(root, window=1, tensors=restored[1], **fork)
    repaired = {**resumed.snapshot_state(), 'cumulative.video': out_v, 'cumulative.audio': out_a}
    validate(repaired, 2); coordinator.publish(root, window=2, tensors=repaired, **fork)
    final = coordinator.restore_latest(root, validate=validate, **fork)
    if final is None or final[0] != 2:
        raise AssertionError('validated resumed H3-shaped checkpoint was not selected')


def session_smoke(coordinator, sessions, match, checker, rank, root):
    """Exercise request agreement, partial-save rollback and fresh-session restoration."""
    options = {key: None for key in coordinator.storage.REQUEST_OPTIONS}
    options.update({'window_frames': 107, 'overlap_frames': 22, 'seed_step': 1, 'lock_audio': True})
    kwargs = {'seed': 1101, 'num_frames': 192, 'latent_h': 2, 'latent_w': 2,
              'text_embeddings': torch.tensor([[1., 2.]])}
    conditioning = [(torch.tensor([[1.]]), torch.tensor([2]))]
    output = {'height': 32, 'width': 32, 'fps': 24, 'audio_sample_rate': 32000, 'preencode_mp4': True,
              'preencode_batch_frames': 17, 'video_codec_options': {'preset': 'ultrafast'}, 'num_outputs': 1}
    common = {'kwargs': kwargs, 'text_conditioning': conditioning, 'options': options,
              'output': output, 'deployment_sha256': 'a' * 64, 'max_bytes': 1024 * 1024,
              'channels': 2, 'audio_channels': 2, 'audio_dims': 6}
    differing = {**common, 'kwargs': {**kwargs, 'seed': 1101 if rank == 0 else 1102}}
    try:
        sessions.Session.begin(root, run_id='8' * 32, **differing)
    except coordinator.CheckpointCoordinationError:
        pass
    else:
        raise AssertionError('different request fingerprints were accepted by the session')
    differing_output = {**common, 'output': {**output, 'preencode_batch_frames': 17 if rank == 0 else 34}}
    try:
        sessions.Session.begin(root, run_id='a' * 32, **differing_output)
    except coordinator.CheckpointCoordinationError:
        pass
    else:
        raise AssertionError('different output settings were accepted by the session')
    multiple_outputs = {**common, 'output': {**output, 'num_outputs': 2}}
    try:
        sessions.Session.begin(root, run_id='b' * 32, **multiple_outputs)
    except coordinator.CheckpointCoordinationError:
        pass
    else:
        raise AssertionError('multiple outputs lack saved prior results and must be rejected')
    first_t, audio_first = checker.video_t(107), checker.audio_t(107)
    overlap_t = checker.video_t(22); audio_overlap = checker.audio_t(107) - checker.audio_t(85)
    first_v = torch.arange(2 * first_t * 2 * 2, dtype=torch.float32).reshape(1, 2, first_t, 2, 2) / 100 + rank
    next_v = first_v + 2; next_v[:, :, :overlap_t] = first_v[:, :, -overlap_t:]
    first_a = torch.arange(2 * 6 * audio_first, dtype=torch.float32).reshape(2, 6, audio_first) / 100 + rank
    next_a = first_a + 2; next_a[..., :audio_overlap] = first_a[..., -audio_overlap:]
    def wrapper(samples):
        values = iter(samples)
        return match.wrap_diffuse(lambda **_kw: tuple(x.clone() for x in next(values)), apply=False,
                                  lock='first', lock_audio=True, lock_ramp=.25, max_ratio=100)
    session = sessions.Session.begin(root, run_id='6' * 32, **common)
    run = wrapper(((first_v, first_a), (next_v, next_a)))
    first = run(seed=1101)
    session.save_window(1, first[0], first[1], run)
    args = {'seed': 1102, 'ref_blocks': [{'kind': 'latent_guide', 'latent_t': overlap_t}],
            'audio_condition_lengths': [audio_overlap]}
    second = run(**args)
    total_v = torch.cat((first[0], second[0][:, :, overlap_t:]), dim=2)
    total_a = torch.cat((first[1], second[1][..., audio_overlap:]), dim=-1)
    bad_v = total_v.clone()
    if rank == 1: bad_v[:, :, -1, 0, 0] += 1
    try:
        session.save_window(2, bad_v, total_a, run)
    except coordinator.CheckpointCoordinationError:
        pass
    else:
        raise AssertionError('invalid rank-local cumulative state was accepted')
    fork = sessions.Session.begin(root, run_id='7' * 32, **common)
    resumed = wrapper(((next_v, next_a),))
    window, saved_v, saved_a = fork.restore('6' * 32, resumed, tensor_device='cpu')
    if window != 1:
        raise AssertionError('session did not restore the last common valid window')
    again = resumed(**args)
    out_v = torch.cat((saved_v, again[0][:, :, overlap_t:]), dim=2)
    out_a = torch.cat((saved_a, again[1][..., audio_overlap:]), dim=-1)
    if not torch.equal(total_v, out_v) or not torch.equal(total_a, out_a):
        raise AssertionError('session recovery differs from uninterrupted synthetic AV')
    fork.save_window(2, out_v, out_a, resumed)
    fork2 = sessions.Session.begin(root, run_id='9' * 32, **common)
    fresh = wrapper(())
    final, final_v, final_a = fork2.restore('7' * 32, fresh, tensor_device='cpu')
    if final != 2 or not torch.equal(total_v, final_v) or not torch.equal(total_a, final_a):
        raise AssertionError('session could not recover its completed synthetic window')


def loop_session_smoke(coordinator, helper, rank, root):
    """A failed rank-local save rolls the actual-shaped loop back on both ranks."""
    opts = helper.options(); kwargs, texts = helper.request(); expected_calls = []
    expected = helper.production_loop(opts)(helper.fake_diffuse(expected_calls), kwargs, window_frames=277,
                                              overlap_frames=22, text_conditioning=texts)
    output = {'height': 32, 'width': 32, 'fps': 24, 'audio_sample_rate': 32000, 'preencode_mp4': True,
              'preencode_batch_frames': 17, 'video_codec_options': {'preset': 'ultrafast'}, 'num_outputs': 1}
    base = {'root': root, 'output': output, 'deployment_sha256': 'a' * 64, 'max_bytes': 8 * 1024**2,
            'channels': 2, 'audio_channels': 2, 'audio_dims': 32}
    original_save = coordinator.storage.save
    def fail_second(*args, **kw):
        if rank == 1 and kw.get('window') == 2:
            raise OSError('injected loop checkpoint failure on rank 1')
        return original_save(*args, **kw)
    first_calls = []
    try:
        coordinator.storage.save = fail_second
        try:
            helper.loop.run(helper.fake_diffuse(first_calls), kwargs, api=helper.api, options=opts,
                            text_conditioning=texts, checkpoint={**base, 'run_id': '0' * 32})
        except coordinator.CheckpointCoordinationError:
            pass
        else:
            raise AssertionError('partial loop checkpoint was accepted')
    finally:
        coordinator.storage.save = original_save
    if first_calls != expected_calls[:2]:
        raise AssertionError('checkpoint failure occurred at the wrong window')
    resumed_calls = []
    resumed = helper.loop.run(helper.fake_diffuse(resumed_calls), kwargs, api=helper.api, options=opts,
                              text_conditioning=texts, checkpoint={**base, 'run_id': '3' * 32,
                                                                   'previous_run_id': '0' * 32, 'tensor_device': 'cpu'})
    if resumed_calls != expected_calls[1:] or any(not torch.equal(a, b) for a, b in zip(expected, resumed)):
        raise AssertionError('two-rank loop recovery differs from the uninterrupted synthetic AV')
    for mode in ('conditioning', 'resume_mode'):
        bad_texts = texts[:-1] if mode == 'conditioning' and rank == 1 else texts
        config = {**base, 'run_id': '9' * 32}
        if mode == 'resume_mode' and rank == 1:
            config.update(previous_run_id='0' * 32, tensor_device='cpu')
        try:
            helper.loop.run(helper.fake_diffuse([]), kwargs, api=helper.api, options=opts,
                            text_conditioning=bad_texts, checkpoint=config)
        except coordinator.CheckpointCoordinationError:
            pass
        else:
            raise AssertionError('rank-local ' + mode + ' mismatch did not abort preflight on both ranks')


if __name__ == '__main__':
    raise SystemExit(main())
