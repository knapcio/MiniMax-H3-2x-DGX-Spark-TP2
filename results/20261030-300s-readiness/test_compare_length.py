import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import types
import unittest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compare_length", HERE / "compare_length.py")
length = importlib.util.module_from_spec(spec)
spec.loader.exec_module(length)


class LengthComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        common = {"http": "200", "curl_rc": 0, "size": "960x576", "steps": 20, "seed": 1101,
                  "task": "fl2va", "image": "painting.png", "prompt": "matched", "frame_indices": "first",
                  "audio_reference": None}
        extra = {"task": "fl2va", "long_video": True, "long_video_mode": "continuation", "preencode_mp4": True}
        self.runs = {}
        for name, seconds, wall, phases in ((length.BASE, 16, 1300, [900, 300]),
                                             (length.LONG, 60, 4400, [1050, 950, 900])):
            timing = dict(common, duration=seconds, effective_duration=seconds, elapsed_s=wall,
                          extra_params=dict(extra, duration=seconds))
            self.runs[name] = {"timing": timing, "diffuse_seconds": phases,
                               "memory": {"head": {"min_available_mib": 15000}},
                               "quality_signals": ["cuts (scene score > 0.1): none"]}
        self.helper = types.SimpleNamespace(read_run=lambda _root, name, _tag: self.runs[name])

    def test_matched_pair_reports_first_window_and_caveat(self):
        report = length.compare(self.root, self.helper)
        self.assertEqual(report["first_window_seconds"], {"16s": 900, "60s": 1050})
        self.assertEqual(report["first_window_change_percent"], 16.67)
        self.assertEqual(report["wall_seconds"]["60s"], 4400)
        self.assertIn("cannot alone establish", report["caveat"])

    def test_aspect_corrected_canvas_can_be_compared_without_changing_default(self):
        original = length.CANVAS
        self.addCleanup(setattr, length, "CANVAS", original)
        length.CANVAS = "960x544"
        for run in self.runs.values():
            run["timing"]["size"] = "960x544"
        report = length.compare(self.root, self.helper)
        self.assertIn("960x544", report["comparison"])
        self.assertEqual(report["first_window_seconds"], {"16s": 900, "60s": 1050})

    def test_prompt_or_sampling_change_is_rejected(self):
        for key, value in (("prompt", "different"), ("seed", 42), ("size", "1344x768")):
            with self.subTest(key=key):
                original = self.runs[length.LONG]["timing"][key]
                self.runs[length.LONG]["timing"][key] = value
                with self.assertRaises(ValueError):
                    length.compare(self.root, self.helper)
                self.runs[length.LONG]["timing"][key] = original

    def test_duration_and_continuation_changes_are_rejected(self):
        baseline = copy.deepcopy(self.runs)
        for mutation in ("duration", "effective", "extra", "window", "missing_phase"):
            with self.subTest(mutation=mutation):
                self.runs = copy.deepcopy(baseline)
                if mutation == "duration": self.runs[length.LONG]["timing"]["duration"] = 30
                if mutation == "effective": self.runs[length.LONG]["timing"]["effective_duration"] = 30
                if mutation == "extra": self.runs[length.LONG]["timing"]["extra_params"]["other"] = True
                if mutation == "window":
                    for run in self.runs.values(): run["timing"]["extra_params"]["continuation_window_frames"] = 175
                if mutation == "missing_phase": self.runs[length.LONG]["diffuse_seconds"] = []
                with self.assertRaises(ValueError):
                    length.compare(self.root, self.helper)

    def test_helper_hash_is_checked(self):
        helper = self.root / "compare-quality-speed-20261030-v2.py"
        shutil.copy2(HERE / "compare_speed.py", helper)
        self.assertTrue(callable(length.load_helper(self.root).read_run))
        helper.write_text("different")
        with self.assertRaisesRegex(ValueError, "helper changed"):
            length.load_helper(self.root)


if __name__ == "__main__":
    unittest.main()

