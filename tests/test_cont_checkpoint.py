"""CPU-only tests of checkpoint storage primitives, not end-to-end resume."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch


SCRIPT = Path(os.environ.get('H3_CHECKPOINT_TEST_SCRIPT', Path(__file__).resolve().parents[1] / 'h3_tp2/cont_checkpoint.py'))
spec = importlib.util.spec_from_file_location('cont_checkpoint_test', SCRIPT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class ContinuationCheckpointTests(unittest.TestCase):
    RUN = '1' * 32
    REQUEST = '2' * 64
    LIMIT = 1024 * 1024

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = {'video': torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2),
                      'audio': torch.arange(8, dtype=torch.bfloat16),
                      'ref.mean': torch.tensor([1, 2], dtype=torch.float32)}

    def save(self, rank=0, window=1):
        return subject.save(self.root, run_id=self.RUN, request_sha256=self.REQUEST, rank=rank, world_size=2,
                            window=window, tensors=self.state, max_bytes=self.LIMIT)

    def load(self, rank=0, window=1, request=None):
        return subject.load(self.root, run_id=self.RUN, request_sha256=request or self.REQUEST, rank=rank,
                            world_size=2, window=window, max_bytes=self.LIMIT)

    def test_round_trip_and_no_overwrite(self):
        first = self.save()
        self.state['audio'][0] = 123
        restored, receipt = self.load()
        self.assertEqual(first, receipt)
        self.assertEqual(restored['audio'][0].item(), 0)
        self.assertTrue(torch.equal(restored['video'], self.state['video']))
        self.assertEqual(restored['video'].dtype, torch.float32)
        self.assertEqual(restored['audio'].dtype, torch.bfloat16)
        with self.assertRaises(FileExistsError): self.save()
        self.assertEqual(self.load()[1], first)

    def test_input_fingerprint_covers_shape_dtype_values_and_metadata(self):
        t = torch.arange(8, dtype=torch.float32)
        def fingerprint(metadata, tensor):
            return subject.fingerprint_inputs(metadata, {'x': tensor}, max_bytes=self.LIMIT)
        reference = fingerprint({'seed': 3}, t)
        self.assertEqual(reference, fingerprint({'seed': 3}, t.clone()))
        self.assertNotEqual(reference, fingerprint({'seed': 4}, t))
        self.assertNotEqual(reference, fingerprint({'seed': 3}, t.reshape(2, 4)))
        self.assertNotEqual(reference, fingerprint({'seed': 3}, t.to(torch.float16)))
        self.assertNotEqual(reference, fingerprint({'seed': 3}, t + 1))
        self.assertEqual(len(fingerprint({}, torch.tensor(1, dtype=torch.bfloat16))), 64)

    def test_data_corruption_and_foreign_request_are_rejected(self):
        self.save()
        with self.assertRaisesRegex(ValueError, 'identity'): self.load(request='3' * 64)
        path = self.root / self.RUN / 'rank-0000-window-000001.safetensors'
        with path.open('r+b') as stream:
            stream.seek(-1, 2); original = stream.read(1); stream.seek(-1, 2)
            stream.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, 'checksum'): self.load()

    def test_incomplete_pair_and_oversized_state_are_rejected(self):
        self.save()
        receipt = self.root / self.RUN / 'rank-0000-window-000001.json'
        receipt.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'): self.load()
        with self.assertRaisesRegex(ValueError, 'size exceeds'):
            subject.save(self.root, run_id=self.RUN, request_sha256=self.REQUEST, rank=0, world_size=2,
                         window=2, tensors=self.state, max_bytes=1)

    def test_receipt_publish_failure_does_not_destroy_prior_checkpoint(self):
        first = self.save(window=1)
        original_link = os.link
        def fail_receipt(src, dst):
            if str(dst).endswith('.json'): raise OSError('injected receipt failure')
            return original_link(src, dst)
        with mock.patch.object(subject.os, 'link', side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError, 'injected'): self.save(window=2)
        directory = self.root / self.RUN
        self.assertFalse((directory / 'rank-0000-window-000002.safetensors').exists())
        self.assertFalse((directory / 'rank-0000-window-000002.json').exists())
        self.assertEqual(self.load(window=1)[1], first)

    def test_latest_common_window_rejects_mixed_ranks_or_runs(self):
        for rank, window in ((0, 1), (0, 2), (1, 1)):
            self.save(rank=rank, window=window)
        receipts = {0: [self.load(rank=0, window=1)[1], self.load(rank=0, window=2)[1]],
                    1: [self.load(rank=1, window=1)[1]]}
        self.assertEqual(subject.latest_common(receipts, run_id=self.RUN, request_sha256=self.REQUEST, world_size=2), 1)
        with self.assertRaisesRegex(ValueError, 'every rank'):
            subject.latest_common({0: receipts[0]}, run_id=self.RUN, request_sha256=self.REQUEST, world_size=2)
        wrong = dict(receipts[1][0], run_id='4' * 32)
        with self.assertRaisesRegex(ValueError, 'identity'):
            subject.latest_common({0: receipts[0], 1: [wrong]}, run_id=self.RUN,
                                  request_sha256=self.REQUEST, world_size=2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            subject.latest_common({0: receipts[0] + [receipts[0][0]], 1: receipts[1]}, run_id=self.RUN,
                                  request_sha256=self.REQUEST, world_size=2)

    def test_receipt_metadata_tampering_is_rejected(self):
        self.save()
        path = self.root / self.RUN / 'rank-0000-window-000001.json'
        data = json.loads(path.read_text()); data['tensors']['video']['shape'] = [999]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'metadata'): self.load()

    def test_change_during_load_is_rejected_and_returned_state_is_not_mapped(self):
        self.save()
        path = self.root / self.RUN / 'rank-0000-window-000001.safetensors'
        restored, _ = self.load()
        expected = restored['audio'].clone()
        original_load = subject.load_file
        def change_after_read(*args, **kwargs):
            loaded = original_load(*args, **kwargs)
            with path.open('r+b') as stream:
                stream.seek(-1, 2); original = stream.read(1); stream.seek(-1, 2)
                stream.write(bytes([original[0] ^ 1]))
            return loaded
        with mock.patch.object(subject, 'load_file', side_effect=change_after_read):
            with self.assertRaisesRegex(ValueError, 'changed during loading'): self.load()
        self.assertTrue(torch.equal(restored['audio'], expected))


if __name__ == '__main__':
    unittest.main()
