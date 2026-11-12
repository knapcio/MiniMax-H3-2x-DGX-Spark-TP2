"""Exercise render.sh's actual form and timing record without contacting a server."""

import json
import hashlib
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
import base64, json, os, pathlib, shlex, sys
args = sys.argv[1:]
fields = []
for i, arg in enumerate(args[:-1]):
    if arg not in ("-F", "--form-string"):
        continue
    field = args[i + 1]
    name, value = field.split("=", 1)
    if arg == "-F" and value.startswith("<"):
        field = name + "=" + pathlib.Path(shlex.split(value[1:])[0]).read_bytes().decode("utf-8")
    fields.append(field)
for field in fields:
    if field.startswith("audio_reference="):
        payload = json.loads(field.split("=", 1)[1])
        pathlib.Path(os.environ["H3_TEST_AUDIO_CAPTURE"]).write_bytes(base64.b64decode(payload["audio_url"].split(",", 1)[1]))
pathlib.Path(os.environ["H3_TEST_CAPTURE"]).write_text(json.dumps(fields))
pathlib.Path(args[args.index("-o") + 1]).write_bytes(b"fake video")
pathlib.Path(args[args.index("-D") + 1]).write_text("HTTP/1.1 200 OK\\n")
print("200", end="")
''')
        curl.chmod(0o755)
        self.env = os.environ.copy()
        for name in ("H3_VIDEO_FLOW_SHIFT", "H3_AUDIO_FLOW_SHIFT", "H3_LORA_JSON", "CONT_PROMPTS_REST", "CONT_PROMPTS_EXTRA", "H3_AUDIO_FILE", "H3_AUDIO_MAX_FIELD_BYTES", "AUDIO_URL", "H3_PROMPT_FILE", "H3_EXTRA_JSON_FILE", "H3_TEXT_MAX_FIELD_BYTES"):
            self.env.pop(name, None)
        # HOME = the temp root: env.sh sources $HOME/.h3-tp2.env, and a serving host's file sets H3_VENV (the stub curl
        # would fall off PATH and the request would reach the real API)
        self.env.update(HOME=str(self.root), H3_VENV=str(self.root), H3_TEST_CAPTURE=str(self.root / "fields.json"),
                        H3_TEST_AUDIO_CAPTURE=str(self.root / "audio.bin"), TMPDIR=str(self.root),
                        CONT_PROMPTS="auto", TASK="t2va")

    def run_render(self, extra, **updates):
        env = dict(self.env, EXTRA_JSON=json.dumps(extra), **updates)
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
        prompts = ["custom first", "custom continuation", "custom end"]   # three windows at the default 30 s
        result, sent, _ = self.run_render({"long_video": True, "continuation_prompts": prompts})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sent["continuation_prompts"], prompts)

    def test_custom_later_prompt_does_not_repeat_first_window_content(self):
        result, sent, _ = self.run_render({"long_video": True},
                                          CONT_PROMPTS_REST="The same scene continues in silence.")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("A landscape.", sent["continuation_prompts"][0])
        self.assertEqual(sent["continuation_prompts"][1:], ["The same scene continues in silence."] * 2)

    def test_explicit_prompts_win_over_later_prompt_override(self):
        prompts = ["first", "later", "later"]   # one per window: the default 30 s request is three windows
        result, sent, _ = self.run_render({"long_video": True, "continuation_prompts": prompts},
                                          CONT_PROMPTS_REST="ignored")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sent["continuation_prompts"], prompts)

    def test_audio_file_larger_than_single_argument_is_submitted_and_recorded_by_hash(self):
        source = self.root / "voice.wav"
        data = bytes(range(256)) * 12300 + b"12"  # crosses the streaming chunk boundary
        source.write_bytes(data)
        result, _, timing = self.run_render({"audio_mode": "lock_source"}, H3_AUDIO_FILE=str(source),
                                            H3_AUDIO_MAX_FIELD_BYTES=str(8 * 1024 * 1024))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "audio.bin").read_bytes(), data)
        self.assertEqual(timing["audio_reference_file"]["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(timing["audio_reference_file"]["bytes"], len(data))
        self.assertIsNone(timing["audio_reference"])
        self.assertFalse(list(self.root.glob("h3-audio.*")))

    def test_audio_over_default_server_field_limit_fails_before_request(self):
        source = self.root / "voice.wav"
        source.write_bytes(b"a" * (1024 * 1024))
        result, _, _ = self.run_render({}, H3_AUDIO_FILE=str(source))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("limit is 1048576", result.stderr)
        self.assertFalse((self.root / "fields.json").exists())
        self.assertFalse(list(self.root.glob("h3-audio.*")))

    def test_conflicting_audio_inputs_fail_before_request(self):
        source = self.root / "voice.wav"
        source.write_bytes(b"audio")
        result, _, _ = self.run_render({}, H3_AUDIO_FILE=str(source), AUDIO_URL="https://example.invalid/a.wav")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "fields.json").exists())

    def test_baseline_sampler_defaults_are_preserved(self):
        result, sent, timing = self.run_render({})
        self.assertEqual(result.returncode, 0, result.stderr)
        fields = json.loads((self.root / "fields.json").read_text())
        self.assertIn("flow_shift=12", fields)
        self.assertFalse(any(field.startswith("lora=") for field in fields))
        self.assertEqual(sent["audio_flow_shift"], 3.0)
        self.assertEqual(timing["video_flow_shift_form"], 12.0)
        self.assertIsNone(timing["lora"])

    def test_adapter_and_shifts_are_sent_and_path_not_recorded(self):
        adapter = {"name": "test-adapter", "path": "/path/to/private.safetensors", "scale": 1.0}
        result, sent, timing = self.run_render({}, H3_VIDEO_FLOW_SHIFT="6", H3_AUDIO_FLOW_SHIFT="3",
                                                H3_LORA_JSON=json.dumps(adapter))
        self.assertEqual(result.returncode, 0, result.stderr)
        fields = json.loads((self.root / "fields.json").read_text())
        self.assertIn("flow_shift=6", fields)
        self.assertEqual(json.loads(next(field[5:] for field in fields if field.startswith("lora="))), adapter)
        self.assertEqual(sent["audio_flow_shift"], 3)
        self.assertEqual(timing["video_flow_shift_after_extra_merge"], 6.0)
        self.assertEqual(timing["lora"]["name"], "test-adapter")
        self.assertNotIn("private.safetensors", json.dumps(timing))

    def test_extra_params_can_override_form_shift_and_is_recorded(self):
        result, sent, timing = self.run_render({"flow_shift": 8}, H3_VIDEO_FLOW_SHIFT="6")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sent["flow_shift"], 8)
        self.assertEqual(timing["video_flow_shift_form"], 6.0)
        self.assertEqual(timing["video_flow_shift_after_extra_merge"], 8)

    def test_invalid_sampler_or_adapter_is_not_submitted(self):
        bad = ({"H3_VIDEO_FLOW_SHIFT": "NaN"}, {"H3_AUDIO_FLOW_SHIFT": "0"},
               {"H3_LORA_JSON": "[]"}, {"H3_LORA_JSON": '{"name":"missing-path"}'},
               {"H3_LORA_JSON": '{"name":"x","path":"/a","scale":NaN}'},
               {"H3_LORA_JSON": '{"name":"x","path":"/a","int_id":0}'})
        for updates in bad:
            with self.subTest(updates=updates):
                result, _, _ = self.run_render({}, **updates)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "fields.json").exists())


if __name__ == "__main__":
    unittest.main()
