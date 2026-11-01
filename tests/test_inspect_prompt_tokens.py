import hashlib
import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'inspect_prompt_tokens.py'
SPEC = importlib.util.spec_from_file_location('inspect_prompt_tokens', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    model_max_length = 262144

    def __init__(self):
        self.calls = []

    def __call__(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return {'input_ids': list(range(len(text.split())))}


class PromptTokenInspection(unittest.TestCase):
    def test_preserves_literal_unicode_and_counts_each_window(self):
        prompts = ['Wokulski mówi; „dzień dobry”.\r\n', 'Izabela słucha.'] * 15
        tokenizer = FakeTokenizer()
        result = MODULE.inspect(prompts, tokenizer)
        self.assertEqual(tokenizer.calls, [(s, False) for s in prompts])
        self.assertEqual(result['window_count'], 30)
        self.assertEqual(result['windows'][0]['utf8_bytes'], len(prompts[0].encode('utf-8')))
        self.assertEqual(result['windows'][0]['prompt_sha256'], hashlib.sha256(prompts[0].encode('utf-8')).hexdigest())
        self.assertEqual(result['sum_text_tokens_across_windows'], sum(len(s.split()) for s in prompts))
        self.assertIn('not a verified safe', result['limitations'])

    def test_plan_shape_and_invalid_inputs(self):
        prompts = ['jeden', 'dwa']
        self.assertEqual(MODULE.strings_from_plan({'extra_params': {'continuation_prompts': prompts}}), prompts)
        self.assertEqual(MODULE.strings_from_plan({'continuation_prompts': prompts}), prompts)
        for value in (None, [], {'continuation_prompts': []}, {'extra_params': None},
                      {'continuation_prompts': ['a', None]}, {'continuation_prompts': ['   ']}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.strings_from_plan(value)


if __name__ == '__main__':
    unittest.main()
