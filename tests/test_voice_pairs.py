"""Pairing the same spoken lines of two renders."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import voice_pairs  # noqa: E402


def seg(w, utmos, text):
    return {"w": w, "utmos": utmos, "text": text}


class VoicePairs(unittest.TestCase):
    def test_lines_pair_by_text_each_used_once(self):
        a = [seg(1, 3.9, "To list od pani Nowakowej."), seg(2, 3.5, "Dziewczyna pisze, że przyszła."), seg(2, 3.0, "Proszę.")]
        b = [seg(2, 3.8, "Dziewczy napisze, że przyszła."), seg(1, 3.7, "To list od pani Nowakowej"), seg(3, 2.0, "Zupełnie inne zdanie tutaj.")]
        pairs = voice_pairs.pair_lines(a, b)
        self.assertEqual([(x["text"][:7], y["text"][:7]) for x, y, _ in pairs], [("To list", "To list"), ("Dziewcz", "Dziewcz")])

    def test_compare_reports_per_window_differences(self):
        a = [seg(1, 3.9, "To list od pani."), seg(2, 3.5, "Dziewczyna pisze."), seg(2, 3.1, "Może byli zajęci.")]
        b = [seg(1, 3.9, "To list od pani."), seg(2, 3.0, "Dziewczyna pisze."), seg(2, 3.2, "Może byli zajęci.")]
        r = voice_pairs.compare(a, b)
        self.assertEqual((r["paired"], r["b_better"], r["a_better"]), (3, 1, 1))
        self.assertEqual([(w["window"], w["mean_diff"]) for w in r["windows"]], [(1, 0.0), (2, -0.2)])
        self.assertAlmostEqual(r["mean_diff"], -0.133, places=3)


if __name__ == "__main__":
    unittest.main()
