import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("compare_speed", Path(__file__).with_name("compare_speed.py"))
speed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed)


class SpeedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, seconds, window in ((speed.BASE, 1000, None), (speed.SMALL, 800, 175)):
            d = self.root / name
            d.mkdir()
            (d / "STATUS").write_text(speed.TAG + " ok\n")
            (d / (speed.TAG + ".mp4")).write_bytes(b"video")
            extra = {"task": "fl2va", "long_video": True}
            if window is not None:
                extra["continuation_window_frames"] = window
            timing = {"http": "200", "curl_rc": 0, "elapsed_s": seconds, "duration": 16,
                      "effective_duration": 16, "size": "1344x768", "steps": 20, "seed": 1101,
                      "task": "fl2va", "image": "painting.png", "prompt": "same prompt", "extra_params": extra}
            (d / (speed.TAG + ".mp4.timing.json")).write_text(json.dumps(timing))
            for node in ("head", "worker"):
                (d / (speed.TAG + f".mem-{node}.csv")).write_text("1,16000,0,0\n2,15000,0,0\n")
            (d / "metrics.txt").write_text("cuts (scene score > 0.1): none\nperiodic luma drops: none\n")
            (d / "phases.txt").write_text("phase diffuse 500.0 s\nphase diffuse 100.0 s\n")

    def test_matched_controls_produce_speed_and_memory_without_promotion(self):
        report = speed.compare(self.root)
        self.assertEqual(report["wall_change_percent"], -20.0)
        self.assertEqual(report["memory"]["175"]["worker"]["min_available_mib"], 15000)
        self.assertIn("Review visual quality", report["caveat"])

    def test_changed_prompt_makes_comparison_invalid(self):
        path = self.root / speed.SMALL / (speed.TAG + ".mp4.timing.json")
        data = json.loads(path.read_text())
        data["prompt"] = "different prompt"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "prompt"):
            speed.compare(self.root)

    def test_missing_memory_or_invalid_status_rejected(self):
        (self.root / speed.SMALL / (speed.TAG + ".mem-worker.csv")).write_text("1,15000,0,0\n")
        with self.assertRaisesRegex(ValueError, "memory samples"):
            speed.compare(self.root)
        (self.root / speed.SMALL / "STATUS").write_text(speed.TAG + " render-failed\n")
        with self.assertRaisesRegex(ValueError, "verification"):
            speed.compare(self.root)


if __name__ == "__main__":
    unittest.main()
