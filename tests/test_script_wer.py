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

    def test_errors_are_charged_to_the_window_of_the_script_word(self):
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
        self.assertEqual(result["clip_word_error_rate"], 0.375)

    def test_a_segment_fusing_two_windows_lines_costs_nothing(self):
        prompts = ["<d>To list od pani.</d>", "<d>Dziewczyna pisze, że przyszła.</d>"]
        fused = [{"start_s": 1.0, "end_s": 12.0, "text": "To list od pani, dziewczyna pisze"},
                 {"start_s": 12.0, "end_s": 14.0, "text": "że przyszła."}]
        rows = script_wer.score(prompts, fused, window_frames=277, overlap_frames=22, fps=24.0)["windows"]
        self.assertEqual([r["word_error_rate"] for r in rows], [0.0, 0.0])
        self.assertEqual([r["heard_words"] for r in rows], [4, 4])

    def test_an_inserted_word_goes_to_the_window_heard_before_it(self):
        prompts = ["<d>To list od pani.</d>", "<d>Dziewczyna pisze, że przyszła.</d>"]
        segments = [{"start_s": 1.0, "end_s": 4.0, "text": "To list od pani proszę"},
                    {"start_s": 12.0, "end_s": 14.0, "text": "Dziewczyna pisze, że przyszła."}]
        rows = script_wer.score(prompts, segments, window_frames=277, overlap_frames=22, fps=24.0)["windows"]
        self.assertEqual([r["word_error_rate"] for r in rows], [0.25, 0.0])
        self.assertEqual(rows[0]["extra"], ["proszę"])
        self.assertEqual(rows[0]["missing"], [])

    def test_respellings_in_the_prompt_are_scored_as_the_real_word(self):
        import tempfile
        prompts = ["<d>[Polish] Morze byli zajęci.</d>"]
        segments = [{"start_s": 0.5, "end_s": 2.0, "text": "Może byli zajęci."}]
        plain = script_wer.score(prompts, segments, window_frames=277, overlap_frames=22, fps=24.0)["windows"][0]
        self.assertEqual(plain["word_error_rate"], round(1 / 3, 4))
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write("# prompt spelling, real word\nmorze może\npieniondze pieniądze\n")
        table = script_wer.load_respellings(Path(handle.name))
        self.assertEqual(table, {"morze": "może", "pieniondze": "pieniądze"})
        fixed = script_wer.score(prompts, segments, window_frames=277, overlap_frames=22, fps=24.0, respellings=table)
        self.assertEqual(fixed["windows"][0]["word_error_rate"], 0.0)

    def test_a_moved_word_boundary_moves_only_the_word_rate(self):
        prompts = ["<d>[Polish] Dziewczyna pisze, że przyszła.</d>", "<d>[Polish] Czynsz może poczekać.</d>"]
        segments = [{"start_s": 1.0, "end_s": 3.0, "text": "Dziewczy napisze, że przyszła."},
                    {"start_s": 12.0, "end_s": 15.0, "text": "Czyż może poczekać."}]
        rows = script_wer.score(prompts, segments, window_frames=277, overlap_frames=22, fps=24.0)["windows"]
        self.assertEqual(rows[0]["word_error_rate"], 0.5)
        self.assertEqual(rows[0]["char_error_rate"], 0.0)
        self.assertEqual(rows[1]["word_error_rate"], round(1 / 3, 4))
        self.assertEqual(rows[1]["char_error_rate"], round(3 / 18, 4))
        self.assertEqual(script_wer.char_distance("czynsz", "czyż"), 3)

    def test_align_lists_the_edit_operations_in_order(self):
        ops = script_wer.align(["a", "b", "c"], ["a", "x", "c", "d"])
        self.assertEqual([op for op, _, _ in ops], ["ok", "sub", "ok", "ins"])
        self.assertEqual([op for op, _, _ in script_wer.align(["a", "b"], ["b"])], ["del", "ok"])
        self.assertEqual(sum(op != "ok" for op, _, _ in ops), 2)

if __name__ == "__main__":
    unittest.main()
