"""CPU-only checks for checkpoint AV geometry and continuation state."""
import importlib.util
import os
from pathlib import Path
import unittest

import torch


SCRIPT = Path(os.environ.get('H3_STATE_TEST_SCRIPT', Path(__file__).resolve().parents[1] / 'h3_tp2/cont_checkpoint_state.py'))
spec = importlib.util.spec_from_file_location('cont_checkpoint_state_test', SCRIPT)
subject = importlib.util.module_from_spec(spec); spec.loader.exec_module(subject)


class ContinuationStateTests(unittest.TestCase):
    @staticmethod
    def state(window):
        end = 107 if window == 1 else 192
        t, prev_t = subject.video_t(end), subject.video_t(107)
        video = torch.arange(1 * 2 * t * 2 * 2, dtype=torch.float32).reshape(1, 2, t, 2, 2)
        previous = video[:, :, -prev_t:].clone()
        return {'cumulative.video': video, 'cumulative.audio': torch.zeros(2, 6, subject.audio_t(end)),
                'match.index': torch.tensor(window, dtype=torch.int64), 'match.prev': previous,
                'match.ref_mean': torch.zeros(2), 'match.ref_std': torch.ones(2),
                'match.audio_mean': torch.zeros(2, 6, 1), 'match.audio_std': torch.ones(2, 6, 1)}

    @staticmethod
    def validate(tensors, window):
        subject.validate_for_plan(tensors, window=window, total_frames=192, full_window=107, overlap=22,
                                  latent_h=2, latent_w=2, channels=2, audio_channels=2, audio_dims=6)

    def test_first_and_second_window_states_pass(self):
        self.validate(self.state(1), 1)
        self.validate(self.state(2), 2)
        self.assertEqual(subject.video_t(7213), 2122)
        self.assertEqual(subject.audio_t(7213), round(7213 * 5 / 3))
        self.assertEqual(subject.planned_window(7213, 277, 22, 29), (7140, 7213, 22))
        with self.assertRaises(ValueError): subject.planned_window(7213, 277, 22, 30)

    def test_wrong_index_audio_or_video_tail_is_rejected(self):
        for mode in ('index', 'audio', 'tail', 'nan', 'std'):
            state = self.state(2)
            if mode == 'index': state['match.index'] = torch.tensor(1)
            if mode == 'audio': state['cumulative.audio'] = state['cumulative.audio'][..., :-1]
            if mode == 'tail': state['match.prev'][:, :, -1, 0, 0] += 1
            if mode == 'nan': state['cumulative.video'][0, 0, 0, 0, 0] = float('nan')
            if mode == 'std': state['match.audio_std'][0, 0, 0] = -1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.validate(state, 2)

    def test_missing_audio_reference_and_invalid_geometry_are_rejected(self):
        state = self.state(1); state.pop('match.audio_mean'); state.pop('match.audio_std')
        with self.assertRaisesRegex(ValueError, 'audio look-lock'): self.validate(state, 1)
        with self.assertRaises(ValueError): subject.video_t(7200)
        with self.assertRaises(ValueError):
            subject.validate(self.state(1), window=1, end_frame=107, window_frames=107, overlap_frames=22,
                             latent_h=2, latent_w=2, channels=2, audio_dims=6)


if __name__ == '__main__':
    unittest.main()
