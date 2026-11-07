"""Per-window word error rate against the dialogue in the prompts."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import script_wer  # noqa: E402


class ScriptWer(unittest.TestCase):
    def test_dialogue_lines_drop_the_language_tag(self):
        self.assertEqual(script_wer.script_lines("scene <d>[Polish] Dzień dobry.</d> more <d>Hello  there</d>"),
                         ["Dzień dobry.", "Hello there"])

    def test_window_spans_give_the_overlap_to_the_earlier_window(self):
        spans = script_wer.window_spans(3, 277, 22, 24.0)
        self.assertEqual([round(s, 3) for s, _ in spans], [0.0, 10.625, 21.25])
        self.assertAlmostEqual(spans[-1][1], (2 * 255 + 277) / 24.0)
        with self.assertRaises(ValueError):
            script_wer.window_spans(1, 22, 22, 24.0)

    def test_segments_go_to_the_window_of_their_midpoint(self):
        prompts = ["<d>[Polish] To list od pani.</d>", "<d>[Polish] Dziewczyna pisze, że przyszła.</d>", "no speech"]
        segments = [{"start_s": 1.0, "end_s": 3.0, "text": "To list od pani."},
                    {"start_s": 12.0, "end_s": 15.0, "text": "Dziewczy napisze, że przysła."}]
        result = script_wer.score(prompts, segments, window_frames=277, overlap_frames=22, fps=24.0)
        rows = result["windows"]
        self.assertEqual([r["window"] for r in rows], [1, 2])
        self.assertEqual(rows[0]["word_error_rate"], 0.0)
        self.assertEqual(rows[1]["word_error_rate"], 0.75)
        self.assertEqual(rows[1]["missing"], ["dziewczyna", "pisze", "przyszła"])
        self.assertEqual(result["mean_word_error_rate"], 0.375)


if __name__ == "__main__":
    unittest.main()
