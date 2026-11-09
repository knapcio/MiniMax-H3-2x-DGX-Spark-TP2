"""Pure-python check of the sampling pace record (h3_tp2/pace.py)."""

import unittest

from h3_tp2 import pace


def window(**over):
    args = {"task": "ref2va", "latent_t": 70, "latent_h": 34, "latent_w": 60, "audio_t": 289, "num_steps": 20,
            "seed": 3601, "temporal_offset": 0.0, "visual_condition_shapes": [(1, 64, 64)], "audio_condition_lengths": []}
    args.update(over)
    return args


class KeyTests(unittest.TestCase):
    def test_seed_prompt_and_offset_do_not_change_the_size(self):
        self.assertEqual(pace.key(window()), pace.key(window(seed=11520, temporal_offset=425.0, text_embeddings="x")))

    def test_shape_steps_and_conditioning_do(self):
        base = pace.key(window())
        for over in ({"latent_t": 40}, {"latent_h": 48}, {"num_steps": 50}, {"audio_t": 100}, {"task": "fl2va"},
                     {"visual_condition_shapes": [(1, 64, 64), (6, 34, 60)]}, {"audio_condition_lengths": [22]}):
            self.assertNotEqual(base, pace.key(window(**over)), over)

    def test_missing_arguments_are_tolerated(self):
        self.assertEqual(pace.key({}), (None, None, None, None, None, None, (), ()))


class NoteTests(unittest.TestCase):
    def test_first_of_a_size_has_no_reference(self):
        best = {}
        self.assertIsNone(pace.note(("a",), 581.0, best))
        self.assertEqual(best, {("a",): 581.0})

    def test_slower_and_faster_than_the_fastest(self):
        best = {}
        pace.note(("a",), 555.0, best)
        self.assertAlmostEqual(pace.note(("a",), 694.0, best), 0.2505, places=3)
        self.assertEqual(best[("a",)], 555.0)          # a slow sampling does not move the reference
        self.assertAlmostEqual(pace.note(("a",), 550.0, best), -0.009, places=3)
        self.assertEqual(best[("a",)], 550.0)          # a faster one does
        self.assertIsNone(pace.note(("b",), 100.0, best))  # another size has its own reference


class ReportTests(unittest.TestCase):
    def test_warns_only_past_the_tolerance(self):
        best = {}
        with self.assertLogs("vllm_omni.h3_tp2.pace", level="INFO") as logs:
            self.assertIsNone(pace.report(1, 1, window(), 555.0, best))
            self.assertAlmostEqual(pace.report(2, 1, window(seed=2), 580.0, best), 0.045, places=3)
            self.assertAlmostEqual(pace.report(3, 2, window(seed=3), 694.0, best), 0.2505, places=3)
        levels = [record.levelname for record in logs.records]
        self.assertEqual(levels, ["INFO", "INFO", "WARNING"])
        self.assertIn("555 s (27.8 s/step)", logs.output[0])
        self.assertIn("window 3 attempt 2: 694 s (34.7 s/step), 25 % slower", logs.output[2])
        self.assertIn("(555 s)", logs.output[2])

    def test_a_broken_argument_never_raises(self):
        with self.assertLogs("vllm_omni.h3_tp2.pace", level="WARNING") as logs:
            self.assertIsNone(pace.report(1, 1, window(visual_condition_shapes=[("x",)]), 10.0, {}))
        self.assertIn("pace skipped", logs.output[0])

    def test_no_steps_no_rate(self):
        with self.assertLogs("vllm_omni.h3_tp2.pace", level="INFO") as logs:
            pace.report(1, 1, window(num_steps=None), 10.0, {})
        self.assertNotIn("s/step", logs.output[0])


if __name__ == "__main__":
    unittest.main()
