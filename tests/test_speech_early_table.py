"""scripts/speech_early_table.py: the journal lines of the early speech checks become a table and stop counts."""

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("speech_early_table", Path(__file__).resolve().parents[1] / "scripts/speech_early_table.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

JOURNAL = """
Nov 09 x python[1]: INFO h3_tp2 speech early check window 1 attempt 1 step 6 of 20: CER 0.0400 (5 of 125 chars), 4.0 s: To list
Nov 09 x python[1]: INFO h3_tp2 speech early check window 1 attempt 1 step 10 of 20: CER 0.0000 (0 of 125 chars), 4.1 s: To list
Nov 09 x python[1]: INFO h3_tp2 speech check window 1 attempt 1 (seed 3601): CER 0.0000 over 125 script chars, score 0.00
Nov 09 x python[1]: INFO h3_tp2 speech early check window 2 attempt 1 step 6 of 20: CER 0.0900 (11 of 122 chars), 4.0 s: Dziewczyna
Nov 09 x python[1]: INFO h3_tp2 speech early check window 2 attempt 1 step 10 of 20: CER 0.0410 (5 of 122 chars), 4.0 s: Dziewczyna
Nov 09 x python[1]: INFO h3_tp2 speech check window 2 attempt 1 (seed 3602): CER 0.0328 over 122 script chars, score 3.28
Nov 09 x python[1]: INFO h3_tp2 speech early check window 2 attempt 2 step 6 of 20: CER 0.0160 (2 of 122 chars), 4.0 s: Dziewczyna
Nov 09 x python[1]: INFO h3_tp2 speech early check window 2 attempt 2 step 10 of 20: CER 0.0000 (0 of 122 chars), 4.0 s: Dziewczyna
Nov 09 x python[1]: INFO h3_tp2 speech check window 2 attempt 2 (seed 10521): CER 0.0000 over 122 script chars, score 0.00
Nov 09 x python[1]: WARNING h3_tp2 speech guard summary: 2 windows judged, 1 re-sampled (w2 0.033 > 0.000), best attempts CER max 0.000
Nov 09 x python[1]: INFO h3_tp2 speech early check window 1 attempt 1 step 10 of 20: CER 0.0240 (3 of 125 chars), 4.1 s: To list
Nov 09 x python[1]: INFO h3_tp2 speech check window 1 attempt 1 (seed 3701): CER 0.0240 over 125 script chars, score 2.40
"""


class TableTests(unittest.TestCase):
    def test_every_sampling_is_one_record_and_a_summary_line_starts_the_next_request(self):
        records = tool.parse(JOURNAL.splitlines())
        self.assertEqual([(r["request"], r["window"], r["attempt"], r["seed"], r["final"]) for r in records],
                         [(1, 1, 1, 3601, 0.0), (1, 2, 1, 3602, 0.0328), (1, 2, 2, 10521, 0.0), (2, 1, 1, 3701, 0.024)])
        self.assertEqual(records[1]["early"], {6: 0.09, 10: 0.041})
        self.assertEqual(records[3]["early"], {10: 0.024})

    def test_stop_counts_per_step_and_limit(self):
        got = {(v["step"], v["limit"]): v for v in tool.verdicts(tool.parse(JOURNAL.splitlines()), 0.01, [0.01, 0.03])}
        # step 6 was heard for three samplings: one fails in the end (0.09), two pass (0.04 and 0.016)
        self.assertEqual((got[6, 0.01]["stopped_failing"], got[6, 0.01]["failing"]), (1, 1))
        self.assertEqual((got[6, 0.01]["stopped_passing"], got[6, 0.01]["passing"]), (2, 2))
        self.assertEqual((got[6, 0.03]["stopped_passing"], got[6, 0.03]["passing"]), (1, 2))
        # step 10 was heard for all four: two fail in the end (0.041 and 0.024 early), two pass (both 0.000 early)
        self.assertEqual((got[10, 0.01]["stopped_failing"], got[10, 0.01]["failing"]), (2, 2))
        self.assertEqual((got[10, 0.03]["stopped_failing"], got[10, 0.03]["stopped_passing"]), (1, 0))

    def test_the_command_prints_the_table_and_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            journal, out = os.path.join(d, "journal.txt"), os.path.join(d, "out.json")
            Path(journal).write_text(JOURNAL, encoding="utf-8")
            text = io.StringIO()
            with redirect_stdout(text):
                self.assertEqual(tool.main([journal, "--limits", "0.03", "--json", out]), 0)
            self.assertIn("request window attempt     s6    s10  final", text.getvalue())
            self.assertIn("      1      2       1  0.090  0.041  0.033", text.getvalue())
            self.assertIn("      2      1       1      -  0.024  0.024", text.getvalue())
            self.assertIn("step 10 limit 0.03 : stops 1 of 2 failing, 0 of 2 passing", text.getvalue())
            self.assertEqual(len(json.loads(Path(out).read_text())["records"]), 4)

    def test_a_journal_without_checks_says_so(self):
        text = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            journal = os.path.join(d, "journal.txt")
            Path(journal).write_text("nothing here\n", encoding="utf-8")
            with redirect_stdout(text):
                self.assertEqual(tool.main([journal]), 1)
        self.assertIn("no speech check lines found", text.getvalue())


if __name__ == "__main__":
    unittest.main()
