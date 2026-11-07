"""The step-gated audio pin is the real edit before the release step and the free sampler from it on."""
import types
import unittest

import torch

from h3_tp2 import audio_pin_steps as aps


def rf_v_to_x0(rows, velocity, t):
    return rows - t * velocity


class FakeEdit:
    def __init__(self):
        self.prepared, self.x0_calls = 0, 0

    def to(self, *, device, dtype=torch.float32):
        return self

    def prepare(self, state_rows, timestep, condition_timestep, *, sigma):
        self.prepared += 1
        return torch.zeros_like(state_rows), torch.full((state_rows.shape[0],), 0.5)

    def x0(self, model_rows, velocity, timestep):
        self.x0_calls += 1
        return torch.full_like(model_rows, 7.0)


class ParseAndRelease(unittest.TestCase):
    def test_parse(self):
        self.assertIsNone(aps.parse_fraction(None))
        self.assertIsNone(aps.parse_fraction(""))
        self.assertIsNone(aps.parse_fraction("1"))
        self.assertEqual(aps.parse_fraction("0.3"), 0.3)
        for bad in ("0", "1.5", "-0.1"):
            with self.assertRaises(ValueError):
                aps.parse_fraction(bad)

    def test_release_step_keeps_one_pinned_and_one_free_step(self):
        self.assertEqual(aps.release_step(0.3, 20), 6)
        self.assertEqual(aps.release_step(0.5, 37), 19)
        self.assertEqual(aps.release_step(0.01, 20), 1)
        self.assertEqual(aps.release_step(0.99, 20), 19)


class Gated(unittest.TestCase):
    def test_real_edit_before_release_then_free_sampler(self):
        inner = FakeEdit()
        edit = aps.gate(inner, [0.0] * 21, 0.3, rf_v_to_x0).to(device=torch.device("cpu"))
        rows = torch.arange(6.0).reshape(3, 2)
        vel = torch.ones_like(rows)
        for step in range(20):
            sigma = 1.0 - step / 20
            model_rows, target = edit.prepare(rows, 1.0 - sigma, 0.95, sigma=sigma)
            x0 = edit.x0(model_rows, vel, 1.0 - sigma)
            if step < 6:
                self.assertTrue(torch.equal(model_rows, torch.zeros_like(rows)))
                self.assertTrue(torch.all(target == 0.5))
                self.assertTrue(torch.all(x0 == 7.0))
            else:
                self.assertTrue(torch.equal(model_rows, rows))
                self.assertTrue(torch.allclose(target, torch.full((3,), min(1.0 - sigma, 0.95))))
                self.assertTrue(torch.allclose(x0, rows - (1.0 - sigma) * vel))
        self.assertEqual((inner.prepared, inner.x0_calls), (6, 6))

    def test_gate_leaves_nothing_or_one_step_alone(self):
        self.assertIsNone(aps.gate(None, [0.0] * 21, 0.3, rf_v_to_x0))
        inner = FakeEdit()
        self.assertIs(aps.gate(inner, [0.0, 1.0], 0.3, rf_v_to_x0), inner)

    def test_install_wraps_the_loop_the_pipeline_calls(self):
        seen = {}

        def loop(*, model, audio_edit=None, sigmas_audio=None, **kwargs):
            seen.update(model=model, audio_edit=audio_edit, sigmas_audio=sigmas_audio)
            return "rows"
        dl = types.SimpleNamespace(minimax_h3_denoise_loop=loop, minimax_h3_rf_v_to_x0=rf_v_to_x0)
        pp = types.SimpleNamespace(minimax_h3_denoise_loop=loop)
        aps.install(dl, pp, 0.3)
        inner = FakeEdit()
        self.assertEqual(pp.minimax_h3_denoise_loop(model="m", audio_edit=inner, sigmas_audio=[0.0] * 21), "rows")
        self.assertIsInstance(seen["audio_edit"], aps.StepGatedEdit)
        self.assertIs(seen["audio_edit"].inner, inner)
        self.assertEqual(seen["audio_edit"].release, 6)
        self.assertIs(dl.minimax_h3_denoise_loop, pp.minimax_h3_denoise_loop)
        self.assertEqual(pp.minimax_h3_denoise_loop(model="m", audio_edit=None, sigmas_audio=[0.0] * 21), "rows")
        self.assertIsNone(seen["audio_edit"])


class LoggerReachesTheWorkerLog(unittest.TestCase):
    def test_logger_is_under_the_vllm_omni_namespace(self):
        # the worker only configures vllm_omni.* loggers; a module-named logger dropped the release line (queue114)
        self.assertTrue(aps.logger.name.startswith("vllm_omni."), aps.logger.name)


if __name__ == "__main__":
    unittest.main()
