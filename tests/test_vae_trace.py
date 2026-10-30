from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from h3_tp2.h3_output import _blend_trace_context
from h3_tp2.vae_trace import trace_temporal_blends


class Model:
    clip_length = 17
    processor = SimpleNamespace(transform_rev=SimpleNamespace(mean=[0, 0, 0], std=[1, 1, 1]))

    def blend(self, a, b, blend_extent, dim):
        weight = torch.arange(blend_extent, dtype=b.dtype).view(1, 1, -1, 1, 1) / blend_extent
        return torch.cat((a[:, :, -blend_extent:] * (1 - weight) + b[:, :, :blend_extent] * weight,
                          b[:, :, blend_extent:]), dim=dim)


class VaeTraceTests(unittest.TestCase):
    def test_trace_is_opt_in(self):
        with patch.dict("os.environ", {"H3_TP2_VAE_BLEND_TRACE": "0"}):
            with _blend_trace_context(object()):
                pass

    def test_temporal_trace_does_not_change_output_and_restores_class_method(self):
        model = Model()
        a = torch.full((1, 3, 5, 16, 16), 0.25)
        b = torch.full((1, 3, 8, 16, 16), 0.75)
        expected = model.blend(a, b, 5, -3)
        with self.assertLogs("vllm_omni.h3_tp2.vae_trace", level="INFO") as logs:
            with trace_temporal_blends(model):
                actual = model.blend(a, b, 5, -3)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(a, torch.full_like(a, 0.25)))
        self.assertTrue(torch.equal(b, torch.full_like(b, 0.75)))
        self.assertNotIn("blend", vars(model))
        self.assertIn("approx-frame 17", logs.output[0])

    def test_existing_instance_method_restored_after_decode_error(self):
        model = Model()
        original = model.blend
        model.blend = original
        with self.assertRaisesRegex(ValueError, "decode failed"):
            with trace_temporal_blends(model):
                raise ValueError("decode failed")
        self.assertIs(model.blend, original)

    def test_missing_diagnostics_do_not_change_output(self):
        model = Model()
        model.processor = None
        a = torch.ones((1, 3, 5, 8, 8))
        b = torch.zeros((1, 3, 8, 8, 8))
        expected = model.blend(a, b, 5, -3)
        with self.assertLogs("vllm_omni.h3_tp2.vae_trace", level="WARNING"):
            with trace_temporal_blends(model):
                actual = model.blend(a, b, 5, -3)
        self.assertTrue(torch.equal(expected, actual))

    def test_module_method_is_restored_and_unavailable_model_is_safe(self):
        class ModuleModel(Model, torch.nn.Module):
            def __init__(self):
                torch.nn.Module.__init__(self)

        model = ModuleModel()
        with trace_temporal_blends(model):
            self.assertIn("blend", vars(model))
        self.assertNotIn("blend", vars(model))
        with self.assertLogs("vllm_omni.h3_tp2.vae_trace", level="WARNING"):
            with trace_temporal_blends(object()):
                pass


if __name__ == "__main__":
    unittest.main()
