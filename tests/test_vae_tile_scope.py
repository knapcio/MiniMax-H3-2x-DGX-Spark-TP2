"""CPU-only behavioral coverage; actual VAE/CUDA still requires qualification."""
import ast
import contextvars
import logging
import os
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

from h3_tp2 import vae_tile_scope


def fixture():
    class Model:
        stack_tiling = False
        training = False

        def __init__(self):
            self.calls = []
            self.decoder = types.SimpleNamespace(mask_enabled=False)

        def decode(self, batch):
            self.calls.append(('decode', len(batch)))
            return batch

        def encode(self, batch):
            self.calls.append(('encode', len(batch)))
            return batch

        def _run_tile_tasks(self, tiles, indices, forward, stacking, cls_agg=None):
            if stacking and indices:
                return forward([tiles[i] for i in indices])
            return [forward([tiles[i]])[0] for i in indices]

    class Host:
        def __init__(self, model): self.model = model
        def decode_latent(self, action): return action(self)
        def decode_with_chunks(self, action): return self.decode_latent(action)

    return Host, Model


def run_tiles(model, *, encode=False):
    forward = model.encode if encode else model.decode
    aggregator = object() if encode else None
    return model._run_tile_tasks(list(range(11)), list(range(11)), forward, model.stack_tiling, aggregator)


class ScopedVaeTiles(unittest.TestCase):
    def test_decode_batches_but_encode_and_original_flag_are_unchanged(self):
        Host, Model = fixture(); vae_tile_scope.install(Host, 4); model = Model(); host = Host(model)
        result = host.decode_with_chunks(lambda h: (run_tiles(h.model), run_tiles(h.model, encode=True)))
        self.assertEqual(result, (list(range(11)), list(range(11))))
        self.assertEqual(model.calls, [('decode', 4), ('decode', 4), ('decode', 3)] + [('encode', 1)] * 11)
        self.assertFalse(model.stack_tiling)
        model.calls.clear(); run_tiles(model)
        self.assertEqual(model.calls, [('decode', 1)] * 11)

    def test_concurrent_encode_never_inherits_decode_scope(self):
        Host, Model = fixture(); vae_tile_scope.install(Host, 4); model = Model(); host = Host(model)
        entered, release = threading.Event(), threading.Event()
        errors = []
        def worker():
            try: host.decode_latent(lambda h: (entered.set(), release.wait(3), run_tiles(h.model)))
            except BaseException as exc: errors.append(exc)
        thread = threading.Thread(target=worker); thread.start()
        try:
            self.assertTrue(entered.wait(2))
            run_tiles(model, encode=True)
            self.assertEqual(model.calls, [('encode', 1)] * 11)
            self.assertFalse(model.stack_tiling)
        finally:
            release.set(); thread.join(3)
        self.assertFalse(thread.is_alive()); self.assertEqual(errors, [])
        self.assertEqual(model.calls[-3:], [('decode', 4), ('decode', 4), ('decode', 3)])

    def test_copied_context_revoked_and_training_mask_respected(self):
        Host, Model = fixture(); vae_tile_scope.install(Host, 4); model = Model(); host = Host(model)
        copied = host.decode_latent(lambda h: contextvars.copy_context())
        copied.run(run_tiles, model)
        self.assertEqual(model.calls, [('decode', 1)] * 11)
        model.calls.clear(); model.training = True; model.decoder.mask_enabled = True
        host.decode_latent(lambda h: run_tiles(h.model))
        self.assertEqual(model.calls, [('decode', 1)] * 11)

    def test_reinstall_and_indirect_override_fail_closed(self):
        Host, Model = fixture(); vae_tile_scope.install(Host, 4)
        model = Model(); host = Host(model); host.decode_latent(lambda h: run_tiles(h.model))
        wrapper = Model._run_tile_tasks
        for _ in range(100):
            vae_tile_scope.install(Host, 4); host.decode_latent(lambda h: run_tiles(h.model))
            self.assertIs(Model._run_tile_tasks, wrapper)
        with self.assertRaises(ValueError): vae_tile_scope.install(Host, 1)
        class Mixin:
            def decode_latent(self, action): return action(self)
        class Indirect(Mixin, Host): pass
        class Leaf(Indirect): pass
        with self.assertRaises(ValueError): vae_tile_scope.install(Leaf, 4)

    def test_actual_plugin_opt_in_branch_and_default(self):
        path = Path(__file__).resolve().parents[1] / 'h3_tp2/plugin.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and 'H3_TP2_VAE_STACK_TILES' in ast.unparse(node.test)
                      and any(isinstance(item, ast.Constant) and item.value == '1' for item in ast.walk(node.test)))
        code = compile(ast.Module(body=[branch], type_ignores=[]), str(path), 'exec')
        Host, Model = fixture(); original = Host.decode_latent
        modules = {}
        for name in ('vllm_omni', 'vllm_omni.diffusion', 'vllm_omni.diffusion.models',
                     'vllm_omni.diffusion.models.minimax_h3'):
            module = types.ModuleType(name); module.__path__ = []; modules[name] = module
        modules['vllm_omni.diffusion.models.minimax_h3'].vae = types.SimpleNamespace(MiniMaxH3VideoVAE=Host)
        namespace = {'os': os, 'logger': logging.getLogger('test-vae-branch')}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {'H3_TP2_VAE_STACK_TILES': '0'}, clear=False):
            exec(code, namespace)
            self.assertIs(Host.decode_latent, original)
            os.environ['H3_TP2_VAE_STACK_TILES'] = '2'; os.environ['H3_TP2_VAE_TILE_BATCH'] = '4'
            exec(code, namespace)
            first = Host.decode_latent
            exec(code, namespace)
            self.assertIs(Host.decode_latent, first)
            model = Model(); Host(model).decode_latent(lambda h: run_tiles(h.model))
            self.assertEqual(model.calls, [('decode', 4), ('decode', 4), ('decode', 3)])
            os.environ['H3_TP2_VAE_TILE_BATCH'] = '5'
            with self.assertRaises(ValueError): exec(code, namespace)

    def test_actual_plugin_legacy_mode_remains_available(self):
        path = Path(__file__).resolve().parents[1] / 'h3_tp2/plugin.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and 'H3_TP2_VAE_STACK_TILES' in ast.unparse(node.test)
                      and any(isinstance(item, ast.Constant) and item.value == '1' for item in ast.walk(node.test)))
        code = compile(ast.Module(body=[branch], type_ignores=[]), str(path), 'exec')
        Host, Model = fixture()
        modules = {'torch': types.ModuleType('torch')}
        for name in ('vllm_omni', 'vllm_omni.diffusion', 'vllm_omni.diffusion.models',
                     'vllm_omni.diffusion.models.minimax_h3'):
            module = types.ModuleType(name); module.__path__ = []; modules[name] = module
        modules['vllm_omni.diffusion.models.minimax_h3'].vae = types.SimpleNamespace(MiniMaxH3VideoVAE=Host)
        namespace = {'os': os, 'logger': logging.getLogger('test-vae-legacy')}
        with patch.dict(sys.modules, modules), patch.dict(os.environ,
                {'H3_TP2_VAE_STACK_TILES': '1', 'H3_TP2_VAE_TILE_BATCH': '4'}, clear=False):
            exec(code, namespace)
            model = Model(); Host(model).decode_latent(lambda h: run_tiles(h.model))
            self.assertEqual(model.calls, [('decode', 4), ('decode', 4), ('decode', 3)])
            self.assertFalse(model.stack_tiling)


if __name__ == '__main__':
    unittest.main()
