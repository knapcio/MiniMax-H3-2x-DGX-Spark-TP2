"""CPU-only guard tests; distributed coordination is exercised separately."""
from pathlib import Path
import tempfile
import unittest

from h3_tp2 import cont_checkpoint_coord as subject


class CheckpointCoordinatorGuards(unittest.TestCase):
    def test_requires_initialized_group(self):
        with self.assertRaisesRegex(ValueError, 'not initialized'):
            subject._group(None, 'cpu')

    def test_inventory_refuses_symlinks_and_malformed_receipts(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            run = 'a' * 32
            self.assertEqual(subject._local_windows(root, run, 0), [])
            (root / run).mkdir()
            directory = root / run
            (directory / 'rank-0000-window-000001.json').write_text('{}')
            (directory / 'rank-0000-window-1000000.json').write_text('{}')
            self.assertEqual(subject._local_windows(root, run, 0), [1, 1_000_000])
            bad = directory / 'rank-0000-window-bad.json'
            bad.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'receipt path'): subject._local_windows(root, run, 0)
            bad.unlink()
            link = directory / 'rank-0000-window-000002.json'
            link.symlink_to(directory / 'absent')
            with self.assertRaisesRegex(ValueError, 'receipt path'): subject._local_windows(root, run, 0)
            dangling_root = root / 'dangling'
            dangling_root.symlink_to(root / 'absent')
            with self.assertRaisesRegex(ValueError, 'root'): subject._local_windows(dangling_root, run, 0)

    def test_rank_and_request_mismatch_are_rejected(self):
        messages = [{'rank': 0, 'run_id': 'a', 'request_sha256': 'b', 'window': 1, 'ok': True},
                    {'rank': 1, 'run_id': 'a', 'request_sha256': 'b', 'window': 1, 'ok': True}]
        subject._verify_messages(messages, run_id='a', request_sha256='b', world=2, window=1)
        for broken in ({**messages[1], 'rank': True}, {**messages[1], 'request_sha256': 'other'},
                       {**messages[1], 'window': 2}, {'serialization_error': True}):
            with self.subTest(broken=broken), self.assertRaises(subject.CheckpointCoordinationError):
                subject._verify_messages([messages[0], broken], run_id='a', request_sha256='b', world=2, window=1)


if __name__ == '__main__':
    unittest.main()
