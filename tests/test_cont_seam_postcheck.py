"""CPU-only diagnostics of actual returned tensors, not model/video quality."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from h3_tp2 import cont_match as cm

if __package__:
    from .test_cont_checkpoint_loop import production_loop, options, request, fake_diffuse
else:
    from test_cont_checkpoint_loop import production_loop, options, request, fake_diffuse


def synthetic_run(*, dtype=torch.float32, locked_source=False, apply=False, retries=0, max_ratio=100.):
    spatial = torch.tensor([[-.2, .2], [.2, -.2]])[None, None, None]
    supplied = [None]
    calls, outputs, raw_windows = [], [], []

    def diffuse(**kwargs):
        calls.append(kwargs['seed'])
        return tuple(value.clone() for value in supplied[0])

    wrapped = cm.wrap_diffuse(diffuse, apply=apply, overlap_v_hint=7, retries=retries,
                              max_ratio=max_ratio, lock='first', lock_ramp=.25, lock_audio=True)
    previous = None
    for index, length in enumerate((82, 82, 22)):
        if previous is None:
            video = torch.arange(length)[None, None, :, None, None] * .02 + spatial
        else:
            free = previous[:, :, -1:] + torch.arange(1, length - 6)[None, None, :, None, None] * .02
            video = torch.cat((previous[:, :, -7:], free), dim=2)
        audio = torch.arange(80, dtype=torch.float32).reshape(2, 4, 10) / 20 + index * 3
        supplied[0] = (video.to(dtype), audio.to(dtype))
        raw_windows.append(tuple(value.clone() for value in supplied[0]))
        args = {'seed': 100 + index}
        if index:
            args.update(video_edit_mask_rows=torch.zeros(7), audio_condition_lengths=[2])
        if locked_source:
            args['locked_audio_rows'] = torch.zeros(20, 4)
        outputs.append(wrapped(**args))
        previous = outputs[-1][0]
    return outputs, wrapped.snapshot_state(), calls, raw_windows


class PostcheckTests(unittest.TestCase):
    def assert_bits(self, left, right):
        self.assertEqual((left.shape, left.dtype), (right.shape, right.dtype))
        self.assertTrue(torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                                    right.contiguous().reshape(-1).view(torch.uint8)))

    def assert_runs_equal(self, left, right):
        self.assertEqual(left[2], right[2])
        for a, b in zip(left[0], right[0]):
            for x, y in zip(a, b):
                self.assert_bits(x, y)
        self.assertEqual(set(left[1]), set(right[1]))
        for key in left[1]:
            self.assert_bits(left[1][key], right[1][key])

    def test_default_and_non_one_values_do_not_invoke_any_extra_measurement(self):
        for value in (None, '0', 'true', ''):
            with self.subTest(value=value), patch.dict(os.environ):
                os.environ.pop('H3_TP2_SEAM_POSTCHECK', None)
                if value is not None:
                    os.environ['H3_TP2_SEAM_POSTCHECK'] = value
                with patch.object(cm, '_log_post_correction_seam') as post, \
                        patch.object(cm, 'seam_ratio', wraps=cm.seam_ratio) as metric:
                    result = synthetic_run()
                post.assert_not_called()
                self.assertEqual(metric.call_count, 2)  # existing raw retry scores only
                self.assertEqual(result[2], [100, 101, 102])

    def test_reproducer_logs_returned_ratio_after_audio_and_before_crossfade(self):
        events = []
        original_audio, original_metric = cm.lock_audio_to_reference, cm.seam_ratio
        def audio(*args, **kwargs):
            events.append('audio')
            return original_audio(*args, **kwargs)
        def metric(*args):
            events.append('metric')
            return original_metric(*args)
        with patch.dict(os.environ, {'H3_TP2_SEAM_POSTCHECK': '1'}), \
                patch.object(cm, 'lock_audio_to_reference', side_effect=audio), \
                patch.object(cm, 'seam_ratio', side_effect=metric), \
                self.assertLogs(cm.logger.name, level='INFO') as logs:
            outputs, _, calls, raw = synthetic_run(retries=2, max_ratio=1.3)
        self.assertEqual(events, ['metric', 'audio', 'metric'] * 2)
        self.assertEqual(calls, [100, 101, 102])
        self.assertAlmostEqual(original_metric(raw[1][0], 7)[2], 1., places=4)
        self.assertAlmostEqual(original_metric(outputs[1][0], 7)[2], 3.0465444, places=5)
        post = [line for line in logs.output if 'seam postcheck' in line]
        self.assertEqual(len(post), 2)
        self.assertIn('window 2: returned window after match/look-lock stage', post[0])
        self.assertIn('before downstream crossfade', post[0])
        self.assertIn('ratio 3.05 (diagnostic only; not the retry score)', post[0])

    def test_enabled_preserves_bits_rng_seed_retries_audio_and_checkpoint_state(self):
        for dtype in (torch.float32, torch.float16):
            for locked in (False, True):
                for apply in (False, True):
                    with self.subTest(dtype=dtype, locked=locked, apply=apply):
                        results = []
                        rng = torch.get_rng_state().clone()
                        for flag in ('0', '1'):
                            with patch.dict(os.environ, {'H3_TP2_SEAM_POSTCHECK': flag}), \
                                    self.assertLogs(cm.logger.name, level='WARNING'):
                                results.append(synthetic_run(dtype=dtype, locked_source=locked,
                                                             apply=apply, retries=2, max_ratio=0.))
                        self.assert_runs_equal(*results)
                        self.assertEqual(results[0][2], [100, 101, 8020, 15939, 102, 8021, 15940])
                        self.assert_bits(rng, torch.get_rng_state())
                        for output, raw in zip(results[1][0][1:], results[1][3][1:]):
                            if locked:
                                self.assert_bits(output[1], raw[1])
                            else:
                                self.assertFalse(torch.equal(output[1], raw[1]))

    def test_failed_post_measurement_keeps_normal_audio_correction_and_state(self):
        with patch.dict(os.environ, {'H3_TP2_SEAM_POSTCHECK': '0'}):
            baseline = synthetic_run()
        original = cm.seam_ratio
        count = 0
        def fail_post(*args):
            nonlocal count
            count += 1
            if count % 2 == 0:
                raise RuntimeError('synthetic post measurement failure')
            return original(*args)
        with patch.dict(os.environ, {'H3_TP2_SEAM_POSTCHECK': '1'}), \
                patch.object(cm, 'seam_ratio', side_effect=fail_post), \
                self.assertLogs(cm.logger.name, level='WARNING') as logs:
            failed = synthetic_run()
        self.assert_runs_equal(baseline, failed)
        self.assertEqual(count, 4)
        self.assertEqual(sum('postcheck' in line and 'skipped' in line for line in logs.output), 2)

    def test_diagnostic_logger_failure_does_not_break_render(self):
        with patch.dict(os.environ, {'H3_TP2_SEAM_POSTCHECK': '0'}):
            baseline = synthetic_run()
        info, warning = cm.logger.info, cm.logger.warning
        def guarded(original):
            def log(message, *args, **kwargs):
                if 'seam postcheck' in message:
                    raise RuntimeError('synthetic logging failure')
                return original(message, *args, **kwargs)
            return log
        with patch.dict(os.environ, {'H3_TP2_SEAM_POSTCHECK': '1'}), \
                patch.object(cm.logger, 'info', side_effect=guarded(info)), \
                patch.object(cm.logger, 'warning', side_effect=guarded(warning)):
            failed = synthetic_run()
        self.assert_runs_equal(baseline, failed)

    def test_nonfinite_measurement_is_reported_as_skipped(self):
        video = torch.ones(1, 1, 8, 2, 2)
        before = video.clone()
        with patch.object(cm, 'seam_ratio', return_value=(1., float('nan'), float('nan'))), \
                self.assertLogs(cm.logger.name, level='WARNING') as logs:
            cm._log_post_correction_seam(video, 2, 2)
        self.assertIn('nonfinite', logs.output[0])
        self.assert_bits(before, video)

    def test_actual_edited_plugin_loop_has_identical_assembled_outputs_with_crossfade(self):
        for crossfade in (False, True):
            values, calls = [], []
            for flag in ('0', '1'):
                samples = []
                kwargs, texts = request()
                with patch.dict(os.environ, {'H3_TP2_SEAM_POSTCHECK': flag}), \
                        patch.object(cm, '_log_post_correction_seam', wraps=cm._log_post_correction_seam) as post:
                    values.append(production_loop(options(crossfade=crossfade))(
                        fake_diffuse(samples), kwargs, window_frames=277, overlap_frames=22,
                        text_conditioning=texts))
                self.assertEqual(post.call_count, 2 if flag == '1' else 0)
                calls.append(samples)
            self.assertEqual(*calls)
            for a, b in zip(*values):
                self.assert_bits(a, b)

    def test_actual_diagnostic_only_plugin_hook_also_reads_the_switch(self):
        source = Path(__file__).resolve().parents[1] / 'h3_tp2/plugin.py'
        hooks = sorted((node for node in ast.walk(ast.parse(source.read_text()))
                        if isinstance(node, ast.FunctionDef) and node.name == '_dc'), key=lambda n: n.lineno)
        def upstream(wrapped, kwargs, **options):
            wrapped(seed=1)
            return wrapped(seed=2, ref_blocks=[{'kind': 'latent_guide', 'latent_t': 2}])
        env = {'_cm': cm, '_apply_match': False, '_orig_match_dc': upstream}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[hooks[-1]], type_ignores=[])), str(source), 'exec'), env)
        def diffuse(**kwargs):
            return torch.arange(32, dtype=torch.float32).reshape(1, 1, 8, 2, 2), torch.zeros(2, 4, 10)
        with patch.dict(os.environ, {'H3_TP2_SEAM_POSTCHECK': '1'}), \
                patch.object(cm, '_log_post_correction_seam', wraps=cm._log_post_correction_seam) as post:
            env['_dc'](diffuse, {}, window_frames=277, overlap_frames=22)
        self.assertEqual(post.call_count, 1)


if __name__ == '__main__':
    unittest.main()
