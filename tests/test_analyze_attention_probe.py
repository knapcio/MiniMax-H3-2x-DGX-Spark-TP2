"""CPU-only checks for the synthetic attention result report."""
import copy
import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / 'scripts/analyze_attention_probe.py'
spec = importlib.util.spec_from_file_location('attention_analysis_test', path)
subject = importlib.util.module_from_spec(spec); spec.loader.exec_module(subject)


def example():
    cases = []
    for seq, scale in ((512, 1.), (512, 4.), (512, 16.), (43392, 1.)):
        n = seq * 28 * 128
        err = {'values': n, 'nonfinite_values': 0, 'max_abs': .1, 'mean_abs': .01, 'relative_l2': .02}
        def backend(name, fp16):
            item = {'status': 'ok', 'metadata': {'backend': name, 'fp16_qk_reduction': fp16},
                    'timing': {'milliseconds': [8., 10., 12.], 'median_ms': 10.}, 'vs_cudnn': copy.deepcopy(err)}
            if seq == 512: item['vs_fp32_math'] = copy.deepcopy(err)
            return item
        base = {'timing': {'milliseconds': [10., 20., 30.], 'median_ms': 20.}}
        if seq == 512: base['vs_fp32_math'] = copy.deepcopy(err)
        cases.append({'sequence': seq, 'heads': 28, 'head_dim': 128, 'qk_scale': scale, 'cudnn': base,
                      'backends': {'flashinfer_fa2': backend('fa2', False),
                                   'flashinfer_fmha_v2_fp16_qk': backend('fmha_v2', True)}})
    return {'device': 'test SM121', 'capability': [12, 1], 'torch': 'test', 'cudnn': 92000, 'cases': cases}


class AttentionAnalysisTests(unittest.TestCase):
    def test_report_keeps_speed_and_fidelity_separate(self):
        result = subject.analyze(example())
        self.assertEqual(result['cases'][(43392, 1.)]['backends']['flashinfer_fa2']['speed_ratio_vs_cudnn'], 2.)
        text = subject.report(result)
        self.assertIn('do not establish whole-render speed or output quality', text)
        self.assertIn('FP16 QK reduction', text)
        self.assertIn('2.000', text)

    def test_unavailable_and_nonfinite_are_reported_not_hidden(self):
        data = example()
        data['cases'][-1]['backends']['flashinfer_fa2'] = {'status': 'unavailable', 'error': 'missing compiler'}
        data['cases'][-1]['backends']['flashinfer_fmha_v2_fp16_qk']['vs_cudnn']['nonfinite_values'] = 5
        result = subject.analyze(data); text = subject.report(result)
        self.assertIn('flashinfer_fa2 (unavailable)', text)
        self.assertEqual(result['cases'][(43392, 1.)]['backends']['flashinfer_fmha_v2_fp16_qk']['vs_cudnn']['nonfinite_pairs'], 5)

    def test_rejects_wrong_dispatch_nonfinite_time_and_duplicate_case(self):
        for mode in ('dispatch', 'fa2_precision', 'time', 'duplicate', 'shape', 'median', 'nonfinite_count'):
            data = example()
            if mode == 'dispatch': data['cases'][0]['backends']['flashinfer_fmha_v2_fp16_qk']['metadata']['backend'] = 'fa2'
            if mode == 'fa2_precision': data['cases'][0]['backends']['flashinfer_fa2']['metadata']['fp16_qk_reduction'] = True
            if mode == 'time': data['cases'][0]['cudnn']['timing']['milliseconds'][0] = float('nan')
            if mode == 'duplicate': data['cases'][-1] = copy.deepcopy(data['cases'][0])
            if mode == 'shape': data['cases'][0]['heads'] = 56
            if mode == 'median': data['cases'][0]['cudnn']['timing']['median_ms'] = 21.
            if mode == 'nonfinite_count': data['cases'][0]['cudnn']['vs_fp32_math']['nonfinite_values'] = -1
            with self.subTest(mode=mode), self.assertRaises(ValueError): subject.analyze(data)


if __name__ == '__main__':
    unittest.main()
