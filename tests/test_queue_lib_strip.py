"""The queue library must strip every continuation experiment variable between boots.

`restart_head` rebuilds ~/.h3-tp2.env from the lines that do NOT match STRIP plus the arm's own exports.  A variable the
plugin reads but STRIP does not match survives every later boot: H3_TP2_CONT_LOCK_ROBUST=1 from one arm stayed in the
head's environment for the next queue's control arm (2026-11-06).
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# request-level switches that are not experiment arms
KEEP = {"H3_TP2_CONT_PROMPTS"}


class StripCoversContinuationVariables(unittest.TestCase):
    def test_every_continuation_variable_of_the_plugin_is_stripped(self):
        lib = (ROOT / "scripts" / "queue-lib.sh").read_text()
        pattern = re.search(r"^STRIP='([^']*)'", lib, flags=re.M).group(1)
        names = set(re.findall(r"H3_TP2_CONT_[A-Z0-9_]+", (ROOT / "h3_tp2" / "plugin.py").read_text())) - KEEP
        self.assertTrue(names)
        missed = sorted(n for n in names if not re.search(pattern, f"export {n}=1"))
        self.assertEqual(missed, [])


if __name__ == "__main__":
    unittest.main()
