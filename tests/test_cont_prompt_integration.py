"""Exercise the actual plugin's prompt wrapper without starting the GPU stack."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

from h3_tp2 import cont_prompts


class PromptWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "h3_tp2/plugin.py"
        tree = ast.parse(path.read_text())
        wrapper = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_plc")
        module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        scope = {"_cp": cont_prompts, "_extra_words": "", "_rest_template": "", "logger": logging.getLogger(__name__),
                 "_orig_plc": lambda _self, _raw, sampling, *a, **kw: sampling.extra_args}
        exec(compile(module, str(path), "exec"), scope)
        cls.wrapper = staticmethod(scope["_plc"])
        cls.pipeline = SimpleNamespace(_extract_prompt=lambda raw: (raw, None))

    def test_direct_frame_count_gets_correct_number_of_prompts(self):
        sampling = SimpleNamespace(extra_args={"long_video": True, "duration": None}, num_frames=7200)
        result = self.wrapper(self.pipeline, "painting", sampling)
        self.assertEqual(len(result["continuation_prompts"]), 29)

    def test_target_duration_takes_precedence_over_other_fields_and_frames(self):
        sampling = SimpleNamespace(extra_args={"long_video": True, "duration": 30,
                                               "target": {"duration_seconds": 60}}, num_frames=7200)
        result = self.wrapper(self.pipeline, "painting", sampling)
        self.assertEqual(len(result["continuation_prompts"]), 6)

    def test_explicit_prompts_are_untouched(self):
        custom = ["first", "second"]
        sampling = SimpleNamespace(extra_args={"long_video": True, "duration": 30,
                                               "continuation_prompts": custom}, num_frames=7200)
        self.assertIs(self.wrapper(self.pipeline, "painting", sampling)["continuation_prompts"], custom)


if __name__ == "__main__":
    unittest.main()
