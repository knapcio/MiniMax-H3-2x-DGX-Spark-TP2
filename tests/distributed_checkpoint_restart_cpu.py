#!/usr/bin/env python3
"""One CPU/Gloo rank of the continuation checkpoint process-restart test.

The parent starts two ranks twice. All inputs and diffusion are synthetic;
this does not exercise the model, GPU/NCCL, or the serving process.
"""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

from h3_tp2 import cont_checkpoint_session as sessions


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('continuation_checkpoint_fixture', HERE / 'test_cont_checkpoint_loop.py')
fixture = importlib.util.module_from_spec(spec); sys.modules[spec.name] = fixture; spec.loader.exec_module(fixture)


class StoppedAfterSave(Exception):
    pass


def av_digest(video, audio):
    digest = hashlib.sha256()
    for value in (video, audio):
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('save', 'resume'), required=True)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--common-window', type=int, required=True)
    parser.add_argument('--stop-window', type=int, default=2)
    parser.add_argument('--total-frames', type=int, default=617)
    parser.add_argument('--wait-after-save', action='store_true')
    args = parser.parse_args()
    if torch.cuda.is_available() or not dist.is_gloo_available():
        raise RuntimeError('this test requires CUDA hidden and the Gloo CPU backend')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{args.port}', rank=args.rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        opts = fixture.options(); kwargs, texts = fixture.request(total=args.total_frames); expected_calls = []
        expected = fixture.production_loop(opts)(fixture.fake_diffuse(expected_calls), kwargs,
                                                  window_frames=277, overlap_frames=22, text_conditioning=texts)
        if not 1 <= args.common_window <= args.stop_window < len(expected_calls):
            raise ValueError('invalid restart window for synthetic plan')
        output = {'height': 32, 'width': 32, 'fps': 24, 'audio_sample_rate': 32000, 'preencode_mp4': True,
                  'preencode_batch_frames': 17, 'video_codec_options': {'preset': 'ultrafast'}, 'num_outputs': 1}
        base = {'root': args.root / f'rank-{args.rank}', 'output': output, 'deployment_sha256': 'a' * 64,
                'dynamic_adapters_sha256': None,
                'max_bytes': 8 * 1024**2, 'channels': 2, 'audio_channels': 2, 'audio_dims': 32}
        calls = []
        if args.phase == 'save':
            original = sessions.Session.save_window
            def save_then_stop(session, window, video, audio, wrapped):
                original(session, window, video, audio, wrapped)
                if window == args.stop_window: raise StoppedAfterSave()
            sessions.Session.save_window = save_then_stop
            try:
                try:
                    fixture.loop.run(fixture.fake_diffuse(calls), kwargs, api=fixture.api, options=opts,
                                     text_conditioning=texts, checkpoint={**base, 'run_id': '1' * 32})
                except StoppedAfterSave:
                    pass
                else:
                    raise AssertionError('test did not stop after the requested saved window')
            finally:
                sessions.Session.save_window = original
            if calls != expected_calls[:args.stop_window]:
                raise AssertionError('save phase rendered the wrong windows')
        else:
            actual = fixture.loop.run(fixture.fake_diffuse(calls), kwargs, api=fixture.api, options=opts,
                                      text_conditioning=texts,
                                      checkpoint={**base, 'run_id': '2' * 32, 'previous_run_id': '1' * 32,
                                                  'tensor_device': 'cpu'})
            if calls != expected_calls[args.common_window:]:
                raise AssertionError('resume rendered the wrong windows')
            if any(not torch.equal(a, b) for a, b in zip(actual, expected, strict=True)):
                raise AssertionError('resumed synthetic AV differs from the uninterrupted loop')
        record = {'phase': args.phase, 'rank': args.rank, 'pid': os.getpid(), 'common_window': args.common_window,
                  'rendered_windows': len(calls), 'total_windows': len(expected_calls),
                  'expected_av_sha256': av_digest(*expected)}
        with (args.root / f'{args.phase}-rank-{args.rank}.json').open('x') as out:
            json.dump(record, out, sort_keys=True); out.write('\n'); out.flush(); os.fsync(out.fileno())
        dist.barrier()
        if args.phase == 'save' and args.wait_after_save:
            with (args.root / f'waiting-rank-{args.rank}').open('x') as out:
                out.write('checkpoint committed\n'); out.flush(); os.fsync(out.fileno())
            while True: time.sleep(60)
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
