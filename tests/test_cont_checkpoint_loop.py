"""Compare the prototype with the actual patched loop using CPU stand-ins."""
import ast
from dataclasses import dataclass
from datetime import timedelta
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from h3_tp2 import cont_checkpoint_coord as coordination
from h3_tp2 import cont_checkpoint_loop as loop
from h3_tp2 import cont_checkpoint_session as sessions
from h3_tp2 import cont_match as match


ROOT = Path(__file__).resolve().parents[1]


def video_t(frames):
    return (frames - 5) // 17 * 5 + 2


def audio_t(frames):
    return round(frames * 5 / 3)


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    overlap: int

    @property
    def audio_start(self): return audio_t(self.start)
    @property
    def audio_end(self): return audio_t(self.end)
    @property
    def overlap_audio_t(self): return audio_t(self.start + self.overlap) - self.audio_start


def plan(total, size, overlap):
    end = min(total, size); windows = [Window(0, end, 0)]
    while end < total:
        start = end - overlap; end = min(total, start + size); windows.append(Window(start, end, overlap))
    return windows


def patch_video(x, *, patch_size):
    assert patch_size == (1, 2, 2)
    n, c, t, h, w = x.shape
    packed = x.reshape(n, c, t, 1, h // 2, 2, w // 2, 2)
    return torch.einsum('nctrhpwq->nthwcrpq', packed).reshape(n * t * (h // 2) * (w // 2), c * 4).contiguous()


def unpack_video(rows, t, channels=2):
    packed = rows.reshape(1, t, 1, 1, channels, 1, 2, 2)
    return torch.einsum('nthwcrpq->nctrhpwq', packed).reshape(1, channels, t, 2, 2).contiguous()


def pack_audio(x):
    return x.permute(0, 2, 1).reshape(-1, x.shape[1]).contiguous()


api = types.SimpleNamespace(_video_t=video_t, plan_continuation_windows=plan,
                            minimax_h3_patchify_video_latent=patch_video, minimax_h3_pack_audio_latent=pack_audio,
                            OmniClientError=ValueError, logger=types.SimpleNamespace(info=lambda *_a, **_k: None))


def production_loop(options):
    tree = ast.parse((ROOT / 'h3_tp2/plugin.py').read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_dc'
             and any(arg.arg == 'window_frames' for arg in node.args.kwonlyargs)]
    if len(nodes) != 1:
        raise AssertionError('actual patched continuation loop was not found uniquely')
    env = {'torch': torch, '_cm': match, '_ct': api, '_apply_match': options['apply_match'],
           '_xfade': options['crossfade'], '_edit': options['edit'], '_retries': options['retries'],
           '_max_ratio': options['max_ratio'], '_lock': options['lock'], '_lock_strength': options['lock_strength'],
           '_lock_ramp': options['lock_ramp'], '_lock_audio': options['lock_audio'], '_lock_std': options['lock_std'],
           '_lock_ref': options['lock_ref'], '_lock_classes': options['lock_classes'], '_seed_step': options['seed_step'],
           '_lock_robust': options.get('lock_robust', False), '_edit_audio': options.get('edit_audio', True),
           '_max_cut': options.get('max_cut', 0.0), '_ig_cfg': options.get('identity_cfg'),
           '_make_identity_fn': options.get('make_identity_fn', lambda _diffuse, _overlap: None)}
    module = ast.fix_missing_locations(ast.Module(body=[nodes[0]], type_ignores=[]))
    exec(compile(module, str(ROOT / 'h3_tp2/plugin.py'), 'exec'), env)  # actual function body, synthetic dependencies
    return env['_dc']


def options(**overrides):
    result = {'window_frames': 277, 'overlap_frames': 22, 'apply_match': False, 'crossfade': True, 'edit': 2,
              'retries': 1, 'max_ratio': 100., 'lock': 'first', 'lock_strength': 1., 'lock_ramp': .25,
              'lock_audio': True, 'lock_std': None, 'lock_ref': 'window', 'lock_classes': True, 'seed_step': 1}
    result.update(overrides); return result


def request(total=617):
    kwargs = {'num_frames': total, 'latent_t': video_t(total), 'latent_h': 2, 'latent_w': 2,
              'audio_t': audio_t(total), 'seed': 1101, 'task': 'fl2va',
              'text_embeddings': torch.tensor([[1., 2.]]), 'text_tags': torch.tensor([1]),
              'visual_condition': torch.ones(1, 8), 'visual_condition_shapes': [(1, 2, 2)],
              'audio_condition': torch.ones(2, 32), 'audio_condition_lengths': [1],
              'ref_blocks': [{'kind': 'image'}], 'keyframe_frame_indices': [0],
              'locked_audio_rows': torch.arange(2 * audio_t(total) * 32, dtype=torch.float32).reshape(-1, 32)}
    texts = [(torch.full((i + 1, 2), float(i)), torch.tensor([i])) for i in range(len(plan(total, 277, 22)))]
    return kwargs, texts


def fake_diffuse(calls):
    def sample(**args):
        digest = hashlib.sha256()
        for key, value in sorted(args.items()):
            digest.update(key.encode())
            if isinstance(value, torch.Tensor):
                digest.update(str((tuple(value.shape), value.dtype)).encode())
                digest.update(value.detach().contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode())
        calls.append((args['seed'], args['num_frames'], args['temporal_offset'], args['task'],
                      args['text_embeddings'].shape[0], args.get('media_time_origin'), digest.hexdigest()))
        gen = torch.Generator(device='cpu').manual_seed(args['seed'])
        video = torch.randn(1, 2, args['latent_t'], 2, 2, generator=gen)
        audio = torch.randn(2, 32, args['audio_t'], generator=gen)
        if args.get('video_edit_clean_rows') is not None:
            rows = patch_video(video, patch_size=(1, 2, 2)); mask = args['video_edit_mask_rows'].unsqueeze(1)
            video = unpack_video(rows * mask + args['video_edit_clean_rows'] * (1 - mask), args['latent_t'])
        if args.get('audio_edit_clean_rows') is not None:
            rows = pack_audio(audio); mask = args['audio_edit_mask_rows'].unsqueeze(1)
            audio = (rows * mask + args['audio_edit_clean_rows'] * (1 - mask)).reshape(2, args['audio_t'], 32).permute(0, 2, 1).contiguous()
        return video, audio
    return sample


class InterruptedAfterCheckpoint(Exception): pass


class ContinuationCheckpointLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_available() or not dist.is_gloo_available():
            raise RuntimeError('CPU-only test requires CUDA hidden and Gloo')
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        dist.init_process_group('gloo', init_method='file://' + str(Path(cls.tmp.name) / 'store'),
                                rank=0, world_size=1, timeout=timedelta(seconds=20))

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group(); cls.tmp.cleanup()

    def test_prototype_matches_the_actual_uncheckpointed_loop(self):
        cases = [({}, 'fl2va', True), ({'edit': 1}, 'fl2va', True), ({'edit': 0}, 'fl2va', True),
                 ({'crossfade': False}, 'fl2va', True), ({'seed_step': 0}, 'fl2va', True),
                 ({'lock': ''}, 'fl2va', True), ({'max_ratio': .01}, 'fl2va', True),
                 ({}, 'ref2va', True), ({}, 'fl2va', False)]
        for change, task, have_texts in cases:
            with self.subTest(change=change, task=task, have_texts=have_texts):
                opts = options(**change); kwargs, texts = request(); kwargs['task'] = task
                texts = texts if have_texts else None
                expected_calls = []; actual_calls = []
                expected = production_loop(opts)(fake_diffuse(expected_calls), kwargs, window_frames=277,
                                                  overlap_frames=22, text_conditioning=texts)
                actual = loop.run(fake_diffuse(actual_calls), kwargs, api=api, options=opts, text_conditioning=texts)
                self.assertEqual(expected_calls, actual_calls)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    def test_300_second_window_plan_matches_actual_loop(self):
        opts = options(); kwargs, texts = request(7213); expected_calls = []; actual_calls = []
        expected = production_loop(opts)(fake_diffuse(expected_calls), kwargs, window_frames=277,
                                          overlap_frames=22, text_conditioning=texts)
        actual = loop.run(fake_diffuse(actual_calls), kwargs, api=api, options=opts, text_conditioning=texts)
        self.assertEqual(len(expected_calls), 29)
        self.assertEqual(expected_calls, actual_calls)
        self.assertEqual((expected_calls[0][0], expected_calls[-1][0]), (1101, 1129))
        self.assertEqual((actual[0].shape[2], actual[1].shape[-1]), (video_t(7213), audio_t(7213)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    def test_edit_audio_off_pins_only_the_video_prefix(self):
        kwargs, texts = request()
        seen = []

        def spy(**args):
            seen.append({k: args.get(k) for k in ('video_edit_mask_rows', 'audio_edit_mask_rows', 'audio_condition')})
            return fake_diffuse([])(**args)
        production_loop(options(edit_audio=False))(spy, kwargs, window_frames=277, overlap_frames=22, text_conditioning=texts)
        self.assertEqual(len(seen), 3)
        for later in seen[1:]:
            self.assertIsNotNone(later['video_edit_mask_rows'])
            self.assertEqual((later['video_edit_mask_rows'] == 0).sum().item(), video_t(22) * 1)
            self.assertIsNone(later['audio_edit_mask_rows'])
            self.assertIsNone(later['audio_condition'])

    def test_upstream_only_branch_is_refused(self):
        kwargs, texts = request()
        with self.assertRaisesRegex(ValueError, 'upstream-only'):
            loop.run(fake_diffuse([]), kwargs, api=api, options=options(edit=0, crossfade=False), text_conditioning=texts)

    def test_resume_after_second_saved_window_matches_uninterrupted_av(self):
        opts = options(); kwargs, texts = request(); expected_calls = []
        expected = production_loop(opts)(fake_diffuse(expected_calls), kwargs, window_frames=277,
                                          overlap_frames=22, text_conditioning=texts)
        output = {'height': 32, 'width': 32, 'fps': 24, 'audio_sample_rate': 32000, 'preencode_mp4': True,
                  'preencode_batch_frames': 17, 'video_codec_options': {'preset': 'ultrafast'}, 'num_outputs': 1}
        with tempfile.TemporaryDirectory() as folder:
            base = {'root': Path(folder), 'output': output, 'deployment_sha256': 'a' * 64,
                    'dynamic_adapters_sha256': None, 'execution_state_sha256': 'e' * 64, 'max_bytes': 8 * 1024**2,
                    'channels': 2, 'audio_channels': 2, 'audio_dims': 32}
            begin = sessions.Session.begin
            class InterruptingSession:
                def __init__(self, inner): self.inner = inner
                def save_window(self, index, *args):
                    self.inner.save_window(index, *args)
                    if index == 2: raise InterruptedAfterCheckpoint()
            first_calls = []
            with patch.object(sessions.Session, 'begin', side_effect=lambda **kw: InterruptingSession(begin(**kw))):
                with self.assertRaises(InterruptedAfterCheckpoint):
                    loop.run(fake_diffuse(first_calls), kwargs, api=api, options=opts, text_conditioning=texts,
                             checkpoint={**base, 'run_id': '1' * 32})
            self.assertEqual(first_calls, expected_calls[:2])
            resume_calls = []
            restored = loop.run(fake_diffuse(resume_calls), kwargs, api=api, options=opts, text_conditioning=texts,
                                checkpoint={**base, 'run_id': '2' * 32, 'previous_run_id': '1' * 32, 'tensor_device': 'cpu'})
            self.assertEqual(resume_calls, expected_calls[2:])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, restored)))
            final_calls = []
            final = loop.run(fake_diffuse(final_calls), kwargs, api=api, options=opts, text_conditioning=texts,
                             checkpoint={**base, 'run_id': '3' * 32, 'previous_run_id': '2' * 32, 'tensor_device': 'cpu'})
            self.assertEqual(final_calls, [])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, final)))
            with self.assertRaises(coordination.CheckpointCoordinationError):
                loop.run(fake_diffuse([]), kwargs, api=api, options=options(seed_step=2), text_conditioning=texts,
                         checkpoint={**base, 'run_id': '4' * 32, 'previous_run_id': '1' * 32, 'tensor_device': 'cpu'})
            # Request modifiers are not present in the denoise kwargs. A new
            # active adapter must not resume a no-adapter checkpoint even when
            # every tensor, seed, schedule, and output setting is otherwise equal.
            with self.assertRaises(coordination.CheckpointCoordinationError):
                loop.run(fake_diffuse([]), kwargs, api=api, options=opts, text_conditioning=texts,
                         checkpoint={**base, 'dynamic_adapters_sha256': 'b' * 64, 'run_id': '5' * 32,
                                     'previous_run_id': '1' * 32, 'tensor_device': 'cpu'})
            with self.assertRaises(coordination.CheckpointCoordinationError):
                loop.run(fake_diffuse([]), kwargs, api=api, options=opts, text_conditioning=texts,
                         checkpoint={**base, 'dynamic_adapters_sha256': 'bad', 'run_id': '6' * 32})
            with self.assertRaises(coordination.CheckpointCoordinationError):
                loop.run(fake_diffuse([]), kwargs, api=api, options=opts, text_conditioning=texts,
                         checkpoint={**base, 'execution_state_sha256': None, 'run_id': 'b' * 32})
            omitted = {key: value for key, value in base.items() if key != 'dynamic_adapters_sha256'}
            with self.assertRaises(coordination.CheckpointCoordinationError):
                loop.run(fake_diffuse([]), kwargs, api=api, options=opts, text_conditioning=texts,
                         checkpoint={**omitted, 'run_id': '7' * 32})
            active = {**base, 'dynamic_adapters_sha256': 'b' * 64}
            active_result = loop.run(fake_diffuse([]), kwargs, api=api, options=opts, text_conditioning=texts,
                                     checkpoint={**active, 'run_id': '8' * 32})
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, active_result)))
            no_calls = []
            with self.assertRaises(coordination.CheckpointCoordinationError):
                loop.run(fake_diffuse(no_calls), kwargs, api=api, options=opts, text_conditioning=texts,
                         checkpoint={**active, 'dynamic_adapters_sha256': 'c' * 64, 'run_id': '9' * 32,
                                     'previous_run_id': '8' * 32, 'tensor_device': 'cpu'})
            self.assertEqual(no_calls, [])
            resumed_active = loop.run(fake_diffuse(no_calls), kwargs, api=api, options=opts, text_conditioning=texts,
                                      checkpoint={**active, 'run_id': 'a' * 32, 'previous_run_id': '8' * 32,
                                                  'tensor_device': 'cpu'})
            self.assertEqual(no_calls, [])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, resumed_active)))
            # Even with no adapter active, cached wrappers may change which
            # implementation executes; require the verified runtime identity.
            with self.assertRaises(coordination.CheckpointCoordinationError):
                loop.run(fake_diffuse(no_calls), kwargs, api=api, options=opts, text_conditioning=texts,
                         checkpoint={**base, 'execution_state_sha256': 'f' * 64, 'run_id': 'c' * 32,
                                     'previous_run_id': '1' * 32, 'tensor_device': 'cpu'})
            self.assertEqual(no_calls, [])


if __name__ == '__main__':
    unittest.main()
