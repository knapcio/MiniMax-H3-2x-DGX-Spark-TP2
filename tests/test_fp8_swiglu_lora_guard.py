"""CPU stand-ins for adapter preservation in the optional MLP fusion."""
import unittest
from unittest.mock import Mock, patch

import torch

from h3_tp2 import fp8_swiglu as fusion


class LoRABase:
    pass


class Linear:
    def __init__(self, width):
        self.width = width
        self.output_size_per_partition = width

    def __call__(self, x):
        value = x[..., :self.width].clone()
        if getattr(fusion._tls, 'stash_next', False):
            fusion._tls.stash = (value, None, None, None)
            fusion._tls.stash_next = False
            return torch.zeros_like(value), None  # the real fused hook returns an uninitialized placeholder
        return value, None


class WrappedLinear(LoRABase):
    def __init__(self, width, delta):
        self.base = Linear(width)
        self.delta = delta
        self.output_size_per_partition = width

    def __call__(self, x):
        value, _ = self.base(x)
        return value + self.delta, None


def activation(x):
    half = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :half]) * x[..., half:]


def make_mlp(fc1, fc2):
    class MLP:
        def __init__(self):
            self.fc1, self.fc2, self.act_fn = fc1, fc2, activation

        def forward(self, x):
            hidden, _ = self.fc1(x)
            out, _ = self.fc2(self.act_fn(hidden))
            return out

        __call__ = lambda self, x: self.forward(x)
    return MLP


class LoRAGuardTests(unittest.TestCase):
    def setUp(self):
        fusion._tls.stash_next = False
        fusion._tls.stash = None
        self.x = torch.tensor([[0.5, -0.3, 1.0, 0.8]])

    def test_fc1_adapter_delta_is_not_discarded(self):
        cls = make_mlp(WrappedLinear(4, torch.tensor([[0.5, 0.3, -0.2, 0.4]])), Linear(2))
        model = cls(); expected = model(self.x)
        fusion._install_on(cls, level=1, lora_layer_type=LoRABase)
        with patch.object(fusion, 'scale_swiglu', side_effect=lambda x, *_: activation(x)) as fused:
            self.assertTrue(torch.equal(model(self.x), expected))
            fused.assert_not_called()
        self.assertFalse(torch.equal(expected, activation(self.x)))  # the adapter had a real effect

    def test_fc2_adapter_is_not_bypassed_by_level_two(self):
        cls = make_mlp(Linear(4), WrappedLinear(2, torch.tensor([[1.0, -0.5]])))
        model = cls(); expected = model(self.x)
        fusion._install_on(cls, level=2, lora_layer_type=LoRABase)
        with patch.object(fusion, 'scale_swiglu_quant', new=Mock()) as quant, \
                patch.object(fusion, '_fc2_direct', new=Mock()) as direct:
            self.assertTrue(torch.equal(model(self.x), expected))
            quant.assert_not_called(); direct.assert_not_called()
        self.assertFalse(torch.equal(expected, activation(self.x)))

    def test_unwrapped_mlp_still_uses_the_fused_path(self):
        cls = make_mlp(Linear(4), Linear(2)); model = cls(); expected = model(self.x)
        fusion._install_on(cls, level=1, lora_layer_type=LoRABase)
        with patch.object(fusion, 'scale_swiglu', side_effect=lambda x, *_: activation(x)) as fused:
            self.assertTrue(torch.equal(model(self.x), expected))
            fused.assert_called_once()


if __name__ == '__main__':
    unittest.main()
