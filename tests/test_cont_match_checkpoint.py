"""CPU stand-in for match-state storage and restoration; not a distributed GPU test."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, env, fallback):
    path = Path(os.environ.get(env, ROOT / fallback))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


match = load('match_checkpoint_test', 'H3_MATCH_TEST_SCRIPT', 'h3_tp2/cont_match.py')
checkpoint = load('checkpoint_match_test', 'H3_CHECKPOINT_TEST_SCRIPT', 'h3_tp2/cont_checkpoint.py')


class MatchCheckpointTests(unittest.TestCase):
    @staticmethod
    def samples():
        first_video = torch.arange(64, dtype=torch.float32).reshape(1, 2, 8, 2, 2) / 20
        next_video = first_video + 3
        next_video[:, :, :2] = first_video[:, :, -2:]
        first_audio = torch.arange(96, dtype=torch.float32).reshape(2, 6, 8) / 20
        next_audio = first_audio + 3
        next_audio[..., :2] = first_audio[..., -2:]
        return (first_video, first_audio), (next_video, next_audio)

    @staticmethod
    def wrapper(samples):
        frames = iter(samples)
        return match.wrap_diffuse(lambda **_kw: tuple(x.clone() for x in next(frames)), apply=False,
                                  lock='first', lock_audio=True, lock_ramp=.25, max_ratio=100)

    def test_saved_match_state_matches_uninterrupted_next_window(self):
        first, second = self.samples()
        uninterrupted = self.wrapper((first, second))
        previous = uninterrupted(seed=1)
        state = uninterrupted.snapshot_state()
        state['match.prev'][0, 0, 0, 0, 0] = 999
        self.assertNotEqual(previous[0][0, 0, 0, 0, 0].item(), 999)
        state = uninterrupted.snapshot_state()
        state.update({'cumulative.video': previous[0], 'cumulative.audio': previous[1]})
        with tempfile.TemporaryDirectory() as root:
            identity = {'run_id': '1' * 32, 'request_sha256': '2' * 64, 'rank': 0,
                        'world_size': 2, 'window': 1, 'max_bytes': 1024 * 1024}
            checkpoint.save(Path(root), tensors=state, **identity)
            restored, _ = checkpoint.load(Path(root), **identity)
        resumed = self.wrapper((second,))
        resumed.restore_state({k: v for k, v in restored.items() if k.startswith('match.')}, device='cpu')
        args = {'seed': 2, 'ref_blocks': [{'kind': 'latent_guide', 'latent_t': 2}], 'audio_condition_lengths': [2]}
        expected, actual = uninterrupted(**args), resumed(**args)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, actual)))
        expected_video = torch.cat((previous[0], expected[0][:, :, 2:]), dim=2)
        restored_video = torch.cat((restored['cumulative.video'], actual[0][:, :, 2:]), dim=2)
        expected_audio = torch.cat((previous[1], expected[1][..., 2:]), dim=-1)
        restored_audio = torch.cat((restored['cumulative.audio'], actual[1][..., 2:]), dim=-1)
        self.assertTrue(torch.equal(expected_video, restored_video))
        self.assertTrue(torch.equal(expected_audio, restored_audio))
        self.assertEqual(resumed.snapshot_state()['match.index'].item(), 2)

    def test_malformed_state_is_rejected_without_mutating_fresh_wrapper(self):
        first, second = self.samples()
        original = self.wrapper((first,)); original(seed=1)
        state = original.snapshot_state()
        restored = self.wrapper((second,))
        missing = dict(state); missing.pop('match.ref_std')
        with self.assertRaisesRegex(ValueError, 'incomplete'): restored.restore_state(missing, device='cpu')
        invalid = dict(state); invalid['match.ref_std'] = torch.full_like(state['match.ref_std'], -1)
        with self.assertRaisesRegex(ValueError, 'standard deviation'): restored.restore_state(invalid, device='cpu')
        empty = dict(state)
        empty['match.audio_mean'] = torch.empty((0, 6, 1), dtype=torch.float32)
        empty['match.audio_std'] = torch.empty((0, 6, 1), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, 'audio'): restored.restore_state(empty, device='cpu')
        restored.restore_state(state, device='cpu')
        with self.assertRaisesRegex(ValueError, 'fresh wrapper'): restored.restore_state(state, device='cpu')


if __name__ == '__main__':
    unittest.main()
