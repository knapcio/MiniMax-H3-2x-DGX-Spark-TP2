import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('snapshot_first_last', SCRIPTS / 'snapshot_first_last.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class TwoFrameGeometry(unittest.TestCase):
    def test_aligned_tail_and_exact_grid(self):
        base = {'task': 'fl2va', 'frame_indices': [0, -1]}
        padded = module.validate_request(base, 10)
        self.assertEqual((padded['requested_frames'], padded['aligned_frames'], padded['alignment_tail_frames']),
                         (240, 243, 3))
        exact = module.validate_request(base, 10.125)
        self.assertEqual((exact['requested_frames'], exact['aligned_frames'], exact['alignment_tail_frames']),
                         (243, 243, 0))

    def test_single_window_boundary_and_custom_geometry(self):
        base = {'task': 'fl2va', 'frame_indices': [0, -1]}
        self.assertEqual(module.validate_request(base, 277 / 24)['aligned_frames'], 277)
        with self.assertRaisesRegex(ValueError, 'one continuation window'):
            module.validate_request(base, 278 / 24)
        custom = {**base, 'continuation_window_frames': 345, 'continuation_overlap_frames': 22}
        self.assertEqual(module.validate_request(custom, 345 / 24)['aligned_frames'], 345)
        with self.assertRaisesRegex(ValueError, 'one continuation window'):
            module.validate_request(custom, 346 / 24)

    def test_effective_duration_precedence_and_invalid_frame_types(self):
        base = {'task': 'fl2va', 'frame_indices': [0, -1], 'duration': 10}
        self.assertEqual(module.validate_request({**base, 'duration_seconds': 9,
                                                  'target': {'duration_seconds': 8}}, 4)['effective_seconds'], 8)
        for update in ({'frame_indices': [False, -1]}, {'frame_indices': [0, 1]}, {'task': 'ref2va'},
                       {'target': {'duration_seconds': None}}, {'duration': float('nan')},
                       {'continuation_window_frames': True}, {'continuation_overlap_frames': 277}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                module.validate_request({**base, **update}, 10)


if __name__ == '__main__':
    unittest.main()
