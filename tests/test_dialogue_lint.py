"""Dialogue lint: budget, tags, repeats."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dialogue_lint as dl  # noqa: E402


class DialogueLint(unittest.TestCase):
    def test_budget_and_seconds_per_window(self):
        prompts = ["scene <d>[Polish] jeden dwa trzy.</d>", "scene <d>[Polish] " + " ".join(["słowo"] * 31) + "</d>", "no lines"]
        r = dl.lint(prompts, window_frames=277, overlap_frames=22, fps=24.0, budget=30)
        self.assertEqual([w["words"] for w in r["windows"]], [3, 31, 0])
        self.assertAlmostEqual(r["windows"][0]["seconds"], 11.54, places=2)
        self.assertAlmostEqual(r["windows"][1]["seconds"], 10.62, places=2)
        self.assertEqual([w["over_budget"] for w in r["windows"]], [False, True, False])
        self.assertEqual(r["silent_windows"], [3])
        self.assertFalse(r["ok"])

    def test_tags_and_repeats_are_reported(self):
        prompts = ["<d>[Polish] Dzień dobry.</d> <d>[English] Hello.</d>", "<d>[Polish] Dzień dobry.</d> <d>", "<d>[Polish] Fine.</d>"]
        r = dl.lint(prompts, window_frames=277, overlap_frames=22, fps=24.0, budget=30)
        self.assertIn("mixed language tags ['English', 'Polish']", r["windows"][0]["problems"])
        self.assertIn("unbalanced <d> tags", r["windows"][1]["problems"])
        self.assertTrue(any(p.startswith("line also in window 1") for p in r["windows"][1]["problems"]))
        self.assertEqual(r["windows"][2]["problems"], [])
        self.assertFalse(r["ok"])
        self.assertTrue(dl.lint(["<d>[Polish] Dobrze.</d>"], window_frames=277, overlap_frames=22, fps=24.0, budget=30)["ok"])


if __name__ == "__main__":
    unittest.main()
