"""Semantic examples for the optional speech-quality screen."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("speech_qa", Path(__file__).resolve().parents[1] / "scripts/speech_qa.py")
speech_qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speech_qa)


class WordErrorTests(unittest.TestCase):
    def test_polish_case_and_punctuation_do_not_change_meaningful_words(self):
        self.assertEqual(speech_qa.word_error_rate("Dzień dobry, panie!", "dzień dobry panie"), 0)

    def test_missing_speech_and_missing_reference_are_not_a_success(self):
        self.assertEqual(speech_qa.word_error_rate("Dzień dobry", ""), 1)
        with self.assertRaises(ValueError):
            speech_qa.word_error_rate("", "Dzień dobry")

    def test_insertions_and_omissions_count(self):
        self.assertAlmostEqual(speech_qa.word_error_rate("raz dwa trzy", "raz trzy cztery"), 2 / 3)

    def test_diacritics_are_not_ignored(self):
        self.assertEqual(speech_qa.word_error_rate("było", "bylo"), 1)


if __name__ == "__main__":
    unittest.main()
