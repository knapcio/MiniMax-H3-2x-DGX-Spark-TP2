"""Isolated GPU guard check using vLLM-Omni's real diffusion LoRA apply method.

The base linear is synthetic; this is not a full H3 or Turbo validation.
"""
import os
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip('needs an idle CUDA device', allow_module_level=True)

pytest.importorskip('vllm_omni.diffusion.lora.layers.base_linear')
from vllm.lora.layers import BaseLayerWithLoRA  # noqa: E402
from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA  # noqa: E402

from h3_tp2 import fp8_swiglu as fusion  # noqa: E402


def test_staged_fusion_module_is_loaded():
    expected = os.environ.get('H3_FUSION_TEST_EXPECTED')
    if expected:
        assert Path(fusion.__file__).resolve() == Path(expected).resolve()


class SyntheticQuant:
    def apply(self, layer, x, bias=None):
        value = x @ layer.weight
        if getattr(fusion._tls, 'stash_next', False):
            m, n = value.shape
            fusion._tls.stash = (value.float(), torch.ones(m, device=x.device), torch.ones(n, device=x.device), None)
            fusion._tls.stash_next = False
            return torch.empty_like(value)
        return value


class SyntheticLinear(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.register_buffer('weight', torch.randn(in_size, out_size, device='cuda', dtype=torch.bfloat16) / 8)
        self.quant_method = SyntheticQuant()
        self.output_size_per_partition = out_size

    def forward(self, x):
        return self.quant_method.apply(self, x), None


class RealApplyWrapper(DiffusionBaseLinearLayerWithLoRA):
    """Use the real apply() with small, explicitly initialized LoRA weights."""
    def __init__(self, base, in_size, out_size):
        torch.nn.Module.__init__(self)
        self.base_layer = base
        self.lora_a_stacked = (torch.randn(1, 1, 4, in_size, device='cuda', dtype=torch.bfloat16) / 4,)
        self.lora_b_stacked = (torch.randn(1, 1, out_size, 4, device='cuda', dtype=torch.bfloat16) / 4,)
        self.output_slices = (out_size,)
        self.lora_config = None
        self._diffusion_lora_active_slices = (True,)

    def forward(self, x):
        return self.apply(x), None


def activation(x):
    half = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :half].float()).to(x.dtype) * x[..., half:]


def make_mlp(fc1, fc2):
    class MLP:
        def __init__(self):
            self.fc1, self.fc2, self.act_fn = fc1, fc2, activation

        def forward(self, x):
            hidden, _ = self.fc1(x)
            output, _ = self.fc2(self.act_fn(hidden))
            return output
    return MLP


@torch.no_grad()
@pytest.mark.parametrize('wrapped_projection,level', [('fc1', 1), ('fc2', 2)])
def test_real_lora_apply_is_preserved(wrapped_projection, level, monkeypatch):
    torch.manual_seed(123)
    x = torch.randn(8, 32, device='cuda', dtype=torch.bfloat16)
    fc1, fc2 = SyntheticLinear(32, 64), SyntheticLinear(32, 32)
    wrapped = RealApplyWrapper(fc1 if wrapped_projection == 'fc1' else fc2, 32,
                               64 if wrapped_projection == 'fc1' else 32)
    plain = make_mlp(fc1, fc2)()
    cls = make_mlp(wrapped if wrapped_projection == 'fc1' else fc1,
                   wrapped if wrapped_projection == 'fc2' else fc2)
    model = cls(); expected = model.forward(x); without_adapter = plain.forward(x)
    assert not torch.equal(expected, without_adapter), 'test adapter must have a nonzero effect'
    fusion._install_on(cls, level=level, lora_layer_type=BaseLayerWithLoRA)
    monkeypatch.setattr(fusion, 'scale_swiglu', lambda *_a: pytest.fail('wrapped fc1 must not use fused output'))
    monkeypatch.setattr(fusion, 'scale_swiglu_quant', lambda *_a: pytest.fail('wrapped fc2 must not use level 2'))
    monkeypatch.setattr(fusion, '_fc2_direct', lambda *_a: pytest.fail('wrapped fc2 must not be bypassed'))
    assert torch.equal(model.forward(x), expected)


@torch.no_grad()
def test_unwrapped_gpu_path_still_uses_real_triton_fusion():
    if not fusion._HAVE_TRITON:
        pytest.skip('Triton unavailable')
    torch.manual_seed(456)
    x = torch.randn(8, 32, device='cuda', dtype=torch.bfloat16)
    cls = make_mlp(SyntheticLinear(32, 64), SyntheticLinear(32, 32))
    model = cls(); expected = model.forward(x)
    fusion._install_on(cls, level=1, lora_layer_type=BaseLayerWithLoRA)
    actual = model.forward(x)
    torch.cuda.synchronize()
    assert torch.allclose(actual.float(), expected.float(), atol=0.01, rtol=0.02)
