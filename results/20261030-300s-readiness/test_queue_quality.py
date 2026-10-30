import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("queue-quality.sh")
DETECTOR = Path(__file__).resolve().parents[2] / "scripts/luma_flicker.py"


class QueueQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.results = self.root / "results"
        self.results.mkdir()
        (self.results / "gen.log").write_text("")
        (self.results / "koninck.png").write_bytes(b"painting")
        (self.root / "meminfo").write_text("MemAvailable: 30000000 kB\n")
        (self.root / "proc/123").mkdir(parents=True)
        self.timeouts(28800, 28800, 29400)
        self.command("flock", "exit 0")
        self.command("nvidia-smi", "printf '%s\\n' \"${H3_TEST_GPU:-0}\"")
        self.command("systemctl", '''
case " $* " in
  *" show "*) printf '123\\n'; exit 0;;
  *" h3-tp2-memguard-head.service "*) [ "${H3_TEST_HEAD_GUARD:-ok}" = ok ]; exit $?;;
esac
exit 0''')
        self.command("ssh", '''
case "$*" in
  *h3-tp2-agent.service*) [ "${H3_TEST_WORKER_GUARD:-ok}" = ok ]; exit $?;;
  *nvidia-smi*) printf '0\\n'; exit 0;;
  *MemAvailable*) printf '30000000\\n'; exit 0;;
esac
exit 1''')
        self.command("curl", '''
case "$*" in
  */health*) printf 200;;
  */metrics*) printf 'vllm_omni:num_requests_running{x="0"} 0\\nvllm_omni:num_requests_waiting{x="0"} 0\\n';;
  *) exit 1;;
esac''')
        self.command("journalctl", "printf 'phase diffuse %s s\\n' \"${H3_TEST_PHASE:-300}\"")
        self.command("ffprobe", "printf '24/1\\n'")
        self.command("ffmpeg", '''
[ "${H3_TEST_FLICKER_FAIL:-0}" != 1 ] || exit 1
for i in $(seq 0 39); do
  value=100
  if [ "${H3_TEST_FLICKER:-0}" = 1 ]; then
    if ((i >= 34)); then value=94; elif ((i >= 17)); then value=97; fi
  fi
  printf 'lavfi.signalstats.YAVG=%s\\n' "$value"
done''')
        self.command("timeout", "shift 3; exec \"$@\"")
        self.bench = self.root / "bench.sh"
        self.bench.write_text('''#!/bin/bash
set -e
name=${OUT##*/}
printf '%s\\n' "$name" >> "$H3_TEST_CALLS"
[ "$name" != "${H3_TEST_FAIL:-}" ] || exit 1
IFS=: read -r dur w h steps seed <<< "$1"
tag="d${dur}_${w}x${h}_s${steps}_seed${seed}_fl2va"
mkdir -p "$OUT"
printf '%s ok\\n' "$tag" > "$OUT/STATUS"
printf 'fake mp4' > "$OUT/$tag.mp4"
printf '1,%s,0,0\\n2,%s,0,0\\n' "${H3_TEST_MIN_MEM:-15000}" "${H3_TEST_MIN_MEM:-15000}" > "$OUT/$tag.mem-head.csv"
cp "$OUT/$tag.mem-head.csv" "$OUT/$tag.mem-worker.csv"
python3 - "$OUT/args.json" "$PROMPT" "$EXTRA_JSON" "$steps" <<'PY'
import json,sys
open(sys.argv[1], 'w').write(json.dumps({'prompt':sys.argv[2], 'extra':json.loads(sys.argv[3]), 'steps':int(sys.argv[4])}))
PY
''')
        self.metrics = self.root / "metrics.sh"
        self.metrics.write_text('''#!/bin/bash
if [ "${H3_TEST_CUT:-0}" = 1 ]; then echo 'cuts (scene score > 0.1): frame 1';
else echo 'cuts (scene score > 0.1): none'; fi
''')
        self.env = os.environ.copy()
        self.env.update(PATH=str(self.bin) + os.pathsep + self.env["PATH"], H3_QUALITY_RESULTS=str(self.results),
                        H3_QUALITY_BENCH=str(self.bench), H3_QUALITY_METRICS=str(self.metrics),
                        H3_QUALITY_PYTHON="python3", H3_QUALITY_FLICKER=str(DETECTOR),
                        H3_QUALITY_MEMINFO=str(self.root / "meminfo"),
                        H3_QUALITY_PROC_ROOT=str(self.root / "proc"), H3_QUALITY_SLEEP_S="0",
                        H3_QUALITY_IDLE_SLEEP_S="0", H3_QUALITY_WAIT_ATTEMPTS="1", H3_QUALITY_IDLE_ATTEMPTS="3",
                        H3_TEST_CALLS=str(self.root / "calls"))

    def command(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)

    def timeouts(self, executor, wave, api):
        values = [f"H3_TP2_REQUEST_TIMEOUT_S={executor}", f"VLLM_OMNI_DLO_DP_WAVE_TIMEOUT={wave}",
                  f"VLLM_OMNI_VIDEO_SYNC_TIMEOUT={api}"]
        (self.root / "proc/123/environ").write_bytes(("\0".join(values) + "\0").encode())

    def ready(self, status=True):
        tag = "d60_1344x768_s20_seed1101_fl2va"
        directory = self.results / "koninck60-fl2va"
        directory.mkdir()
        (directory / "STATUS").write_text(f"{tag} {'ok' if status else 'render-failed'}\n")
        (directory / (tag + ".mp4")).write_bytes(b"fake")
        (self.results / "gen.log").write_text("CALIB41-DONE 2026-10-30 17:00:00 CET\n")

    def run_queue(self, **overrides):
        env = dict(self.env, **overrides)
        return subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True, timeout=20)

    def calls(self):
        p = self.root / "calls"
        return p.read_text().splitlines() if p.exists() else []

    def test_waits_without_verified_predecessor(self):
        self.ready(status=False)
        result = self.run_queue()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_refuses_inactive_guards_or_busy_gpu(self):
        for override in ({"H3_TEST_HEAD_GUARD": "off"}, {"H3_TEST_WORKER_GUARD": "off"}, {"H3_TEST_GPU": "40"}):
            with self.subTest(override=override):
                # Each invocation sees the same verified predecessor and creates no output.
                if not (self.results / "koninck60-fl2va").exists():
                    self.ready()
                self.assertNotEqual(self.run_queue(**override).returncode, 0)
                self.assertEqual(self.calls(), [])

    def test_runs_controls_and_guarded_300(self):
        self.ready()
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(self.calls(), ["quality16-base", "quality16-steps10", "quality16-win175", "quality16-slowfig", "quality5-stream",
                                        "quality16-stream", "quality16-960-stream", "quality300-960-stream"])
        def args(name):
            return json.loads((self.results / name / "args.json").read_text())
        base, short, slow, stream, long = (args(name) for name in ("quality16-base", "quality16-win175",
                                          "quality16-slowfig", "quality16-stream", "quality300-960-stream"))
        self.assertEqual(base["prompt"], short["prompt"])
        self.assertEqual(base["prompt"], stream["prompt"])
        self.assertEqual(args("quality16-steps10")["prompt"], base["prompt"])
        self.assertEqual(args("quality16-steps10")["extra"], base["extra"])
        self.assertEqual(args("quality16-steps10")["steps"], 10)
        self.assertEqual(base["steps"], 20)
        self.assertEqual(short["extra"]["continuation_window_frames"], 175)
        self.assertTrue(stream["extra"]["preencode_mp4"])
        self.assertIn("move very slowly", slow["prompt"])
        self.assertIn("without strong moving cloud shadows", long["prompt"])
        report = (self.results / "quality16-960-stream/luma-flicker.txt").read_text()
        self.assertIn("for 2 successive 17-frame periods", report)
        self.assertNotIn("periodic luma drops", (self.results / "quality16-960-stream/metrics.txt").read_text())

    def test_300_skipped_when_quality_speed_memory_or_timeouts_fail(self):
        for override in ({"H3_TEST_CUT": "1"}, {"H3_TEST_PHASE": "1000"}, {"H3_TEST_MIN_MEM": "5000"},
                         {"H3_TEST_FLICKER": "1"}, {"H3_TEST_FLICKER_FAIL": "1"},
                         {"H3_QUALITY_FLICKER": "/nonexistent/detector.py"}):
            with self.subTest(override=override):
                with tempfile.TemporaryDirectory() as nested:
                    # Use a new fixture per scenario to keep the no-overwrite check meaningful.
                    test = QueueQualityTests()
                    test.setUp()
                    try:
                        test.ready()
                        result = test.run_queue(**override)
                        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                        self.assertNotIn("quality300-960-stream", test.calls())
                        self.assertIn("H3-AUTO-SKIP-300", (test.results / "gen.log").read_text())
                    finally:
                        test.doCleanups()
        self.ready()
        self.timeouts(14400, 14400, 15000)
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertNotIn("quality300-960-stream", self.calls())

    def test_failed_case_stops_later_work(self):
        self.ready()
        result = self.run_queue(H3_TEST_FAIL="quality5-stream")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls()[-1], "quality5-stream")
        self.assertNotIn("quality300-960-stream", self.calls())
        self.assertIn("H3-AUTO-HOLD", (self.results / "gen.log").read_text())

    def test_existing_output_is_not_overwritten(self):
        self.ready()
        (self.results / "quality16-base").mkdir()
        result = self.run_queue()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
