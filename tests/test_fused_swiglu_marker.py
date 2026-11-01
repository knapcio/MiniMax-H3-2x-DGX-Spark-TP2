"""The startup marker is diagnostic, and must never import a model just to inspect it."""
import sys
from types import ModuleType
from unittest.mock import patch

from h3_tp2.plugin import fused_swiglu_marker


MODULE = 'vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer'


def test_missing_module_and_class_are_unknown():
    with patch.dict(sys.modules, {MODULE: None}):
        assert fused_swiglu_marker() == 'unknown'
    with patch.dict(sys.modules, {MODULE: object()}):
        assert fused_swiglu_marker() == 'unknown'
    lazy = ModuleType(MODULE)
    def fail_if_queried(_name):
        raise AssertionError('diagnostic triggered a lazy module lookup')
    lazy.__getattr__ = fail_if_queried
    with patch.dict(sys.modules, {MODULE: lazy}):
        assert fused_swiglu_marker() == 'unknown'


def test_class_marker_distinguishes_absent_present_and_unexpected():
    module = ModuleType(MODULE)
    class MLP:
        pass
    module.MiniMaxH3MLP = MLP
    with patch.dict(sys.modules, {MODULE: module}):
        assert fused_swiglu_marker() == 'absent'
        MLP._h3_tp2_fused_swiglu = True
        assert fused_swiglu_marker() == 'present'
        MLP._h3_tp2_fused_swiglu = 'unexpected'
        assert fused_swiglu_marker() == 'unknown'
