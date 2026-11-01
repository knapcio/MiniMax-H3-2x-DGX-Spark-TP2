"""CPU regression for the actual nested continuation hooks in plugin.py.

No torch/model import is needed: the source bodies are compiled unchanged into
one installer scope, reproducing the original shared-cell recursion hazard.
"""
import ast
import logging
from pathlib import Path
import types
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'h3_tp2/plugin.py'


def install_hooks(upstream, *, legacy=True, rebind=False, step=1, fixed=None):
    tree = ast.parse(SOURCE.read_text())
    hooks = sorted((node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == '_dc'), key=lambda n: n.lineno)
    if len(hooks) != 3:
        raise AssertionError('expected experiment, edited-loop and diagnostic continuation hooks')
    first, _, diagnostic = hooks
    bindings = sorted((node for node in ast.walk(tree)
                       if isinstance(node, ast.Assign) and len(node.targets) == 1
                       and isinstance(node.targets[0], ast.Name)
                       and isinstance(node.value, ast.Attribute) and node.value.attr == 'diffuse_continuation'),
                      key=lambda n: n.lineno)
    if len(bindings) != 2:
        raise AssertionError('expected exactly two original-callable bindings')
    installer = ast.parse('def install(upstream, replacement):\n pass').body[0]
    installer.body = []
    if legacy:
        installer.body += [bindings[0], first]
        installer.body += ast.parse('_pp9e.diffuse_continuation = _dc').body
    installer.body += [bindings[1], diagnostic]
    if rebind:
        for binding in bindings:
            installer.body += ast.parse(binding.targets[0].id + ' = replacement').body
    installer.body += ast.parse('return _dc').body
    wrap_calls = []
    def wrap(function, **options):
        wrap_calls.append(options)
        return function
    env = {'_cm': types.SimpleNamespace(wrap_diffuse=wrap), '_apply_match': False,
           '_anchor_v': 0, '_step': step, '_fixed': fixed, '_cap': None,
           '_cont': types.SimpleNamespace(diffuse_continuation=upstream),
           '_pp9e': types.SimpleNamespace(diffuse_continuation=upstream),
           'logger': logging.getLogger(__name__)}
    module = ast.fix_missing_locations(ast.Module(body=[installer], type_ignores=[]))
    exec(compile(module, str(SOURCE), 'exec'), env)
    def replacement(*args, **kwargs):
        raise AssertionError('a later binding replaced the captured predecessor')
    return env['install'](upstream, replacement), wrap_calls


class HookBinding(unittest.TestCase):
    def exercise(self, *, legacy=True, rebind=False, step=1, fixed=None):
        calls, samples = [], []
        def upstream(diffuse, kwargs, **options):
            calls.append(options)
            result = None
            for i in range(3):
                result = diffuse(seed=kwargs['seed'], temporal_offset=425.0 * i)
            return result
        def diffuse(**kwargs):
            samples.append(kwargs)
            return 'video', 'audio'
        hook, wrapped = install_hooks(upstream, legacy=legacy, rebind=rebind, step=step, fixed=fixed)
        self.assertEqual(hook(diffuse, {'seed': 42}, window_frames=277, overlap_frames=22), ('video', 'audio'))
        self.assertEqual(calls, [{'window_frames': 277, 'overlap_frames': 22}])
        self.assertEqual(wrapped, [{'apply': False}])
        return samples

    def test_explicit_seed_experiment_and_edit_off_do_not_recurse(self):
        samples = self.exercise()
        self.assertEqual([x['seed'] for x in samples], [42, 43, 44])
        self.assertEqual([x['temporal_offset'] for x in samples], [0, 425, 850])

    def test_diagnostic_only_calls_original_once(self):
        self.assertEqual([x['seed'] for x in self.exercise(legacy=False)], [42, 42, 42])

    def test_later_rebinding_cannot_change_either_predecessor(self):
        self.assertEqual([x['seed'] for x in self.exercise(rebind=True)], [42, 43, 44])
        self.assertEqual([x['seed'] for x in self.exercise(legacy=False, rebind=True)], [42, 42, 42])

    def test_experiment_parameters_still_reach_diffuse(self):
        samples = self.exercise(step=2, fixed=0.0)
        self.assertEqual([x['seed'] for x in samples], [42, 44, 46])
        self.assertEqual([x['temporal_offset'] for x in samples], [0, 0, 0])


if __name__ == '__main__': unittest.main(verbosity=2)
