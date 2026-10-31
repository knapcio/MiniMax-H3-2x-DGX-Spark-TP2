"""CPU checks for attention-probe error accounting; no GPU benchmark."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import Mock

import torch


SCRIPT = Path(os.environ.get('H3_ATTENTION_PROBE_TEST_SCRIPT', Path(__file__).resolve().parents[1] / 'scripts/attention_probe.py'))
spec = importlib.util.spec_from_file_location('attention_probe_test', SCRIPT)
subject = importlib.util.module_from_spec(spec); spec.loader.exec_module(subject)


class AttentionProbeTests(unittest.TestCase):
    def test_fmha_uses_auto_dispatch(self):
        flashinfer = Mock()
        workspace = object()
        self.assertIs(subject.fmha_wrapper(flashinfer, workspace),
                      flashinfer.BatchPrefillWithRaggedKVCacheWrapper.return_value)
        flashinfer.BatchPrefillWithRaggedKVCacheWrapper.assert_called_once_with(
            workspace, kv_layout='NHD', backend='auto')

    def test_chunked_error_statistics_and_nonfinite_values(self):
        a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        b = torch.tensor([[2., 2.], [1., float('nan')], [5., 9.]])
        result = subject.error_stats(a, b, chunk=1)
        self.assertEqual(result['values'], 6)
        self.assertEqual(result['nonfinite_values'], 1)
        self.assertEqual(result['max_abs'], 3)
        self.assertAlmostEqual(result['mean_abs'], 6 / 5)
        self.assertAlmostEqual(result['relative_l2'], (14 / 75) ** .5)

    def test_bad_shapes_and_fatal_cuda_classification(self):
        with self.assertRaises(ValueError): subject.error_stats(torch.ones(2), torch.ones(3))
        with self.assertRaises(ValueError): subject.error_stats(torch.ones(2), torch.ones(2), chunk=0)
        self.assertTrue(subject.fatal_cuda(RuntimeError('CUDA error: an illegal memory access was encountered')))
        self.assertTrue(subject.fatal_cuda(RuntimeError('CUDA error: unknown error')))
        self.assertTrue(subject.fatal_cuda(RuntimeError('misaligned address')))
        self.assertFalse(subject.fatal_cuda(RuntimeError('CUDA out of memory')))
        self.assertFalse(subject.fatal_cuda(ValueError('backend unsupported')))


if __name__ == '__main__':
    unittest.main()
