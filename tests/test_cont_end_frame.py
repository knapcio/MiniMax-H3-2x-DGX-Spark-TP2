"""CPU tests of the actual continuation body and its later-hook precedence."""
import ast
import logging
from pathlib import Path
import unittest

if __package__:
    from .test_cont_checkpoint_loop import production_loop, options, request, fake_diffuse
else:
    from test_cont_checkpoint_loop import production_loop, options, request, fake_diffuse


SOURCE = Path(__file__).resolve().parents[1] / 'h3_tp2/plugin.py'


def selected_loop(legacy, **changes):
    opts = options(edit=2, crossfade=False, retries=0, lock='', **changes)
    actual = production_loop(opts)
    if not legacy:
        return actual
    # Execute the actual definitions in installation order: the experimental
    # wrapper is installed first, then the edited continuation body replaces it.
    tree = ast.parse(SOURCE.read_text())
    hooks = sorted((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_dc'),
                   key=lambda n: n.lineno)
    if len(hooks) != 3:
        raise AssertionError('continuation hooks changed')
    env = dict(actual.__globals__)
    def upstream(*args, **kwargs):
        raise AssertionError('edited-loop request unexpectedly escaped to the legacy loop')
    env.update(_orig_experiment_dc=upstream, _anchor_v=0, _step=1, _fixed=0., _cap=None,
               logger=logging.getLogger(__name__))
    body = [hooks[0], *ast.parse('_orig_match_dc = _dc').body, hooks[1]]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(SOURCE), 'exec'), env)
    return env['_dc']


class FinalFrameContract(unittest.TestCase):
    def run_case(self, indices, *, total=617, task='fl2va', legacy=False):
        kwargs, texts = request(total)
        kwargs['task'] = task; kwargs['keyframe_frame_indices'] = indices
        calls = []
        loop = selected_loop(legacy)
        return loop, kwargs, texts, calls

    def test_multiwindow_final_frame_fails_before_first_diffuse_with_either_hook_order(self):
        for indices in ([-1], [0, -1]):
            for legacy in (False, True):
                with self.subTest(indices=indices, legacy=legacy):
                    loop, kwargs, texts, calls = self.run_case(indices, legacy=legacy)
                    with self.assertRaisesRegex(ValueError, 'final-frame conditioning.*not yet supported'):
                        loop(fake_diffuse(calls), kwargs, window_frames=277, overlap_frames=22,
                             text_conditioning=texts)
                    self.assertEqual(calls, [])

    def test_multiwindow_start_only_still_runs(self):
        for legacy in (False, True):
            loop, kwargs, texts, calls = self.run_case([0], legacy=legacy)
            loop(fake_diffuse(calls), kwargs, window_frames=277, overlap_frames=22, text_conditioning=texts)
            self.assertEqual(len(calls), 3)

    def test_single_window_final_frame_is_unchanged_including_exact_boundary(self):
        for total in (107, 277):
            for indices in ([-1], [0, -1], [0]):
                for legacy in (False, True):
                    with self.subTest(total=total, indices=indices, legacy=legacy):
                        loop, kwargs, texts, calls = self.run_case(indices, total=total, legacy=legacy)
                        loop(fake_diffuse(calls), kwargs, window_frames=277, overlap_frames=22,
                             text_conditioning=texts)
                        self.assertEqual(len(calls), 1)

    def test_first_grid_point_after_window_is_rejected(self):
        loop, kwargs, texts, calls = self.run_case([-1], total=294)
        with self.assertRaisesRegex(ValueError, 'multi-window FL2VA'):
            loop(fake_diffuse(calls), kwargs, window_frames=277, overlap_frames=22, text_conditioning=texts)
        self.assertEqual(calls, [])

    def test_ref2va_is_not_subject_to_fl2va_guard(self):
        for legacy in (False, True):
            loop, kwargs, texts, calls = self.run_case([0, -1], task='ref2va', legacy=legacy)
            loop(fake_diffuse(calls), kwargs, window_frames=277, overlap_frames=22, text_conditioning=texts)
            self.assertEqual(len(calls), 3)


if __name__ == '__main__': unittest.main(verbosity=2)
