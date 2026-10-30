"""Exercise render.sh's actual form and timing record without contacting a server."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class RenderDurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bindir = self.root / "bin"
        bindir.mkdir()
        (bindir / "python3").symlink_to(sys.executable)
        date = bindir / "date"
        date.write_text("#!/bin/sh\nprintf '1.000000001\\n'\n")
        date.chmod(0o755)
        curl = bindir / "curl"
        curl.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
fields = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "-F"]
pathlib.Path(os.environ["H3_TEST_CAPTURE"]).write_text(json.dumps(fields))
pathlib.Path(args[args.index("-o") + 1]).write_bytes(b"fake video")
pathlib.Path(args[args.index("-D") + 1]).write_text("HTTP/1.1 200 OK\\n")
print("200", end="")
''')
        curl.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(H3_VENV=str(self.root), H3_TEST_CAPTURE=str(self.root / "fields.json"),
                        CONT_PROMPTS="auto", TASK="t2va")

    def run_render(self, extra):
        env = dict(self.env, EXTRA_JSON=json.dumps(extra))
        out = self.root / "result.mp4"
        result = subprocess.run(["bash", str(REPO / "scripts/render.sh"), str(out), "30", "1344", "768",
                                 "20", "1101", "A landscape."], env=env, text=True, capture_output=True)
        if result.returncode:
            return result, None, None
        fields = json.loads((self.root / "fields.json").read_text())
        sent = json.loads(next(s.removeprefix("extra_params=") for s in fields if s.startswith("extra_params=")))
        timing = json.loads((self.root / "result.mp4.timing.json").read_text())
        return result, sent, timing

    def test_target_wins_and_produces_29_prompts_for_300_seconds(self):
        result, sent, timing = self.run_render({"long_video": True, "long_video_mode": "continuation",
                                                "duration_seconds": 60, "target": {"duration_seconds": 300}})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(sent["continuation_prompts"]), 29)
        self.assertEqual(timing["effective_duration"], 300.0)

    def test_explicit_null_does_not_use_lower_priority_duration(self):
        result, sent, timing = self.run_render({"long_video": True, "long_video_mode": "continuation",
                                                "target": {"duration_seconds": None}})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("continuation_prompts", sent)
        self.assertIsNone(timing["effective_duration"])

    def test_invalid_target_fails_before_submitting(self):
        result, _, _ = self.run_render({"long_video": True, "target": "not-an-object"})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "fields.json").exists())

    def test_explicit_prompts_are_not_replaced(self):
        prompts = ["custom first", "custom continuation"]
        result, sent, _ = self.run_render({"long_video": True, "continuation_prompts": prompts})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sent["continuation_prompts"], prompts)


if __name__ == "__main__":
    unittest.main()
