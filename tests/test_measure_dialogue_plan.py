import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import unicodedata

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/measure_dialogue_plan.py'
spec = importlib.util.spec_from_file_location('measure_dialogue_plan', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DialoguePlanMeasurements(unittest.TestCase):
    def test_equivalent_unicode_normalizations_count_words_equally(self):
        text = 'Zażółć gęślą jaźń.'
        nfc = module.measure(3, [{'start': 0, 'end': 2, 'text': text}])
        nfd = module.measure(3, [{'start': 0, 'end': 2, 'text': unicodedata.normalize('NFD', text)}])
        self.assertEqual(nfc['words_approx'], 3)
        self.assertEqual(nfc['words_approx'], nfd['words_approx'])

    def test_unsorted_lines_measure_gaps_and_polish_words(self):
        lines = [{'start': 6, 'end': 8, 'speaker': 'I', 'text': 'Do widzenia.'},
                 {'start': 3, 'end': 5, 'speaker': 'W', 'text': 'Zażółć gęślą, panie!'}]
        result = module.measure(10, lines)
        self.assertEqual(result['utterances'], 2)
        self.assertEqual(result['words_approx'], 5)
        self.assertEqual(result['words_per_minute_of_clip_approx'], 30)
        self.assertEqual(result['scheduled_dialogue_s'], 4)
        self.assertEqual(result['scheduled_dialogue_fraction'], .4)
        self.assertEqual(result['unscheduled_dialogue_s'], 6)
        self.assertEqual([x['duration_s'] for x in result['gaps']], [3, 1, 2])
        self.assertEqual(result['positive_between_line_gap_median_s'], 1)
        self.assertEqual([x['input_index'] for x in result['lines']], [2, 1])

    def test_adjacent_lines_and_no_dialogue(self):
        result = module.measure(2, [{'start': 0, 'end': 1, 'text': 'Raz'},
                                    {'start': 1, 'end': 2, 'text': 'Dwa'}])
        self.assertEqual(result['gaps'], [])
        self.assertEqual(result['positive_between_line_gap_count'], 0)
        self.assertIsNone(result['positive_between_line_gap_median_s'])
        empty = module.measure(2, [])
        self.assertEqual(empty['gaps'][0]['kind'], 'entire_clip')
        self.assertEqual(empty['longest_unscheduled_dialogue_s'], 2)

    def test_rejects_overlap_invalid_times_and_malformed_items(self):
        bad = [
            (0, []), (float('nan'), []), (10, [None]),
            (10, [{'start': True, 'end': 1, 'text': 'tak'}]),
            (10, [{'start': 0, 'end': float('inf'), 'text': 'tak'}]),
            (10, [{'start': 0, 'end': 11, 'text': 'tak'}]),
            (10, [{'start': 0, 'end': 1, 'text': ''}]),
            (10, [{'start': 0, 'end': 2, 'text': 'raz'}, {'start': 1, 'end': 3, 'text': 'dwa'}]),
        ]
        for seconds, lines in bad:
            with self.subTest(seconds=seconds, lines=lines), self.assertRaises(ValueError):
                module.measure(seconds, lines)

    def test_cli_writes_new_report_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'dialogue.json'; report = root / 'report.json'
            source.write_text(json.dumps([{'start': 1, 'end': 2, 'text': 'Dzień dobry'}]), encoding='utf-8')
            command = [sys.executable, '-B', str(SCRIPT), '--seconds', '3', '--dialogue', str(source),
                       '--report', str(report)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            saved = report.read_bytes()
            self.assertEqual(json.loads(saved)['unscheduled_dialogue_s'], 2)
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual(report.read_bytes(), saved)

    def test_cli_refuses_fifo_and_oversized_file_without_blocking(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source.json'; report = root / 'report.json'
            os.mkfifo(source)
            command = [sys.executable, '-B', str(SCRIPT), '--seconds', '3', '--dialogue', str(source),
                       '--report', str(report)]
            result = subprocess.run(command, capture_output=True, timeout=3)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(report.exists())
            source.unlink()
            with source.open('wb') as stream:
                stream.truncate(module.MAX_PLAN_BYTES + 1)
            result = subprocess.run(command, capture_output=True, timeout=3)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(report.exists())


if __name__ == '__main__':
    unittest.main()
