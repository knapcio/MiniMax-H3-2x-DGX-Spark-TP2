import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPT = HERE / "queue-vae-trace.sh"
TXN = HERE / "file_txn.py"
FLICKER = REPO / "scripts/luma_flicker.py"


class VaeTraceQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-trace-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"; self.bin.mkdir()
        self.results = self.root / "results"; self.results.mkdir()
        self.dropin = self.root / "run/h3-tp2-head.service.d/trace.conf"
        self.state = self.root / "state"; self.state.write_text("active")
        self.proc = self.root / "proc/123"; self.proc.mkdir(parents=True)
        (self.proc / "environ").write_bytes(b"H3_TP2_VAE_BLEND_TRACE=0\0")
        (self.root / "meminfo").write_text("MemAvailable: 30000000 kB\n")
        (self.root / "head.env").write_text("# no trace setting\n")
        (self.results / "koninck.png").write_bytes(b"painting")
        self.output_case("patch-smoke5", "d5_960x576_s10_seed0")
        self.output_case("patch-duration16", "d16_1344x768_s20_seed1101_fl2va")
        (self.results / "gen.log").write_text("H3-AUTO-PATCH-DONE fc14580 2026-10-31 09:00 CET\n"
                                               "H3-AUTO-LONG300-SKIPPED prior success 2026-10-31 09:01 CET\n")
        self.command("flock", "exit 0")
        self.command("python3", '''
if [ "$1" = "$H3_TRACE_TXN" ]; then [ "${H3_TEST_BAD_TXN:-0}" != 1 ]; exit $?; fi
if [ "$1" = "$H3_TRACE_FLICKER" ]; then echo 'periodic luma drops: none'; exit 0; fi
exec "$H3_TEST_REAL_PYTHON" "$@"''')
        self.command("nvidia-smi", '''
if [ -e "$H3_TEST_BUSY_MARKER" ]; then printf '50\\n'; else printf '%s\\n' "${H3_TEST_GPU:-0}"; fi''')
        self.command("systemctl", '''
printf '%s\\n' "$*" >> "$H3_TEST_SYSTEMCTL"
case " $* " in
  *" show "*" DropInPaths "*) exit 0;;
  *" show "*" ActiveState "*) state=$(cat "$H3_TEST_STATE"); [ "$state" = stopped ] && state=inactive; printf '%s\\n' "$state"; exit 0;;
  *" show "*) if [ "$(cat "$H3_TEST_STATE")" = stopped ]; then printf '0\\n'; else printf '123\\n'; fi; exit 0;;
  *" stop h3-tp2-head.service "*)
    if [ "${H3_TEST_STOP_ACTIVATING:-0}" = 1 ]; then printf activating > "$H3_TEST_STATE"; else printf stopped > "$H3_TEST_STATE"; fi; exit 0;;
  *" start h3-tp2-head.service "*)
    if [ "${H3_TEST_START_ACTIVATING:-0}" = 1 ]; then printf activating > "$H3_TEST_STATE"; exit 1; fi
    if [ "${H3_TEST_FAIL_START_ONCE:-0}" = 1 ] && [ ! -e "$H3_TEST_FAIL_MARKER" ]; then touch "$H3_TEST_FAIL_MARKER"; exit 1; fi
    printf active > "$H3_TEST_STATE"
    if [ -e "$H3_TRACE_DROPIN" ]; then printf 'H3_TP2_VAE_BLEND_TRACE=1\\0' > "$H3_TRACE_PROC_ROOT/123/environ"
    else printf 'H3_TP2_VAE_BLEND_TRACE=0\\0' > "$H3_TRACE_PROC_ROOT/123/environ"; fi
    exit 0;;
  *" is-active "*" h3-tp2-head.service "*) [ "$(cat "$H3_TEST_STATE")" = active ]; exit $?;;
  *" is-active "*" h3-tp2-memguard-head.service "*) [ "${H3_TEST_GUARD:-ok}" = ok ]; exit $?;;
esac
exit 0''')
        self.command("ssh", '''
case "$*" in
  *h3-tp2-agent.service*) [ "${H3_TEST_WORKER:-ok}" = ok ]; exit $?;;
  *sha256sum*) if [ "${H3_TEST_REMOTE_HASH_BAD:-0}" = 1 ]; then echo wrong; else sha256sum "$H3_TRACE_TXN"; fi; exit 0;;
  *verify-new*) [ "${H3_TEST_REMOTE_BAD:-0}" != 1 ]; exit $?;;
  *nvidia-smi*) printf '0\\n'; exit 0;;
  *MemAvailable*) printf '30000000\\n'; exit 0;;
esac
exit 1''')
        self.command("curl", '''
case "$*" in
  */health*) if [ "$(cat "$H3_TEST_STATE")" = active ]; then printf 200; else printf 000; fi;;
  */metrics*) printf 'vllm_omni:num_requests_running{x="0"} 0\\nvllm_omni:num_requests_waiting{x="0"} 0\\n';;
  */v1/videos*) if [ "${H3_TEST_JOBS:-0}" = 1 ] || [ -e "$H3_TEST_JOBS_MARKER" ]; then
      printf '{"data":[{}],"has_more":false}'; else printf '{"data":[],"has_more":false}'; fi;;
  *) exit 1;;
esac''')
        self.command("journalctl", '''
case "${H3_TEST_TRACE:-good}" in
  good) printf 'h3_tp2 VAE temporal blend 1 approx-frame 17: old [1] new [2] blended [1]\\n';;
  warning) printf 'h3_tp2 VAE temporal blend trace skipped (error)\\n';;
  missing) printf 'other log\\n';;
esac''')
        self.command("timeout", 'shift 3; exec "$@"')
        self.command("ffmpeg", "printf 'SHA256=abcdef\\n'")
        self.metrics = self.root / "metrics.sh"; self.metrics.write_text("#!/bin/bash\necho metrics\n")
        self.bench = self.root / "bench.sh"
        self.bench.write_text('''#!/bin/bash
set -e
name=${OUT##*/}
printf '%s\\n' "$name" >> "$H3_TEST_CASES"
if [ "$name" = "${H3_TEST_FAIL_CASE:-}" ]; then
  [ "${H3_TEST_BUSY_ON_FAIL:-0}" != 1 ] || touch "$H3_TEST_BUSY_MARKER"
  [ "${H3_TEST_JOBS_ON_FAIL:-0}" != 1 ] || touch "$H3_TEST_JOBS_MARKER"
  [ "${H3_TEST_CHANGE_DROPIN:-0}" != 1 ] || printf 'foreign\\n' > "$H3_TRACE_DROPIN"
  exit 1
fi
IFS=: read -r dur w h steps seed <<< "$1"
suffix=""; [ "$TASK" = fl2va ] && suffix=_fl2va
tag="d${dur}_${w}x${h}_s${steps}_seed${seed}${suffix}"
mkdir -p "$OUT"
printf '%s ok\\n' "$tag" > "$OUT/STATUS"
printf video > "$OUT/$tag.mp4"
printf '%s\\n' "${EXTRA_JSON:-}" > "$OUT/extra.json"
''')
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        H3_TRACE_RESULTS=str(self.results), H3_TRACE_LIVE=str(self.root / "live"),
                        H3_TRACE_STAGE=str(self.root / "stage"), H3_TRACE_BACKUP=str(self.root / "backup"),
                        H3_TRACE_TXN=str(TXN), H3_TRACE_BENCH=str(self.bench), H3_TRACE_METRICS=str(self.metrics),
                        H3_TRACE_PYTHON="python3", H3_TRACE_FLICKER=str(FLICKER),
                        H3_TRACE_ENV_FILE=str(self.root / "head.env"), H3_TRACE_PROC_ROOT=str(self.root / "proc"),
                        H3_TRACE_MEMINFO=str(self.root / "meminfo"), H3_TRACE_DROPIN=str(self.dropin),
                        H3_TRACE_WAIT_ATTEMPTS="1", H3_TRACE_IDLE_ATTEMPTS="3", H3_TRACE_READY_ATTEMPTS="3",
                        H3_TRACE_WAIT_S="0", H3_TRACE_IDLE_S="0", H3_TRACE_READY_S="0",
                        H3_TEST_REAL_PYTHON=sys.executable, H3_TEST_STATE=str(self.state),
                        H3_TEST_SYSTEMCTL=str(self.root / "systemctl.log"), H3_TEST_CASES=str(self.root / "cases"),
                        H3_TEST_FAIL_MARKER=str(self.root / "fail-marker"), H3_TEST_BUSY_MARKER=str(self.root / "busy"),
                        H3_TEST_JOBS_MARKER=str(self.root / "jobs"))

    def command(self, name, body):
        p = self.bin / name; p.write_text("#!/bin/bash\n" + body + "\n"); p.chmod(0o755)

    def output_case(self, folder, tag):
        p = self.results / folder; p.mkdir()
        (p / "STATUS").write_text(tag + " ok\n")
        (p / (tag + ".mp4")).write_bytes(b"video")

    def run_queue(self, **overrides):
        return subprocess.run(["bash", str(SCRIPT)], env=dict(self.env, **overrides), text=True,
                              capture_output=True, timeout=25)

    def systemctl(self):
        p = self.root / "systemctl.log"
        return p.read_text() if p.exists() else ""

    def cases(self):
        p = self.root / "cases"
        return p.read_text().splitlines() if p.exists() else []

    def test_success_restores_default_after_smoke_and_history(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.cases(), ["trace-smoke5", "trace-history30"])
        self.assertEqual(self.state.read_text(), "active")
        self.assertFalse(self.dropin.exists())
        self.assertIn("H3_TP2_VAE_BLEND_TRACE=0", (self.proc / "environ").read_text())
        self.assertIn("H3-AUTO-TRACE-DONE", (self.results / "gen.log").read_text())
        self.assertIn('"continuation_window_frames":345', (self.results / "trace-history30/extra.json").read_text())
        self.assertEqual(self.systemctl().count("stop h3-tp2-head.service"), 2)

    def test_predecessor_without_verified_output_does_not_stop(self):
        (self.results / "patch-duration16/STATUS").write_text("failed\n")
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertNotIn("stop h3-tp2-head.service", self.systemctl())

    def test_long300_must_finish_and_holds_propagate(self):
        (self.results / "gen.log").write_text("H3-AUTO-PATCH-DONE fc14580 2026-10-31 09:00 CET\n")
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertNotIn("stop h3-tp2-head.service", self.systemctl())
        (self.results / "gen.log").write_text("H3-AUTO-HOLD long300 phase=healthy rc=1\n")
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertNotIn("stop h3-tp2-head.service", self.systemctl())

    def test_busy_guards_jobs_and_bad_remote_all_prevent_stop(self):
        for override in ({"H3_TEST_GPU": "40"}, {"H3_TEST_GUARD": "bad"}, {"H3_TEST_WORKER": "bad"},
                         {"H3_TEST_JOBS": "1"}, {"H3_TEST_REMOTE_BAD": "1"}, {"H3_TEST_REMOTE_HASH_BAD": "1"}):
            with self.subTest(override=override):
                result = self.run_queue(**override)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("stop h3-tp2-head.service", self.systemctl())

    def test_conflicting_env_or_existing_output_prevents_stop(self):
        (self.root / "head.env").write_text("export H3_TP2_VAE_BLEND_TRACE=0\n")
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertNotIn("stop h3-tp2-head.service", self.systemctl())
        (self.root / "head.env").write_text("# blank\n")
        (self.results / "trace-smoke5").mkdir()
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertNotIn("stop h3-tp2-head.service", self.systemctl())

    def test_foreign_dropin_and_changed_detector_prevent_stop(self):
        self.dropin.parent.mkdir(parents=True)
        self.dropin.write_text("foreign\n")
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertEqual(self.dropin.read_text(), "foreign\n")
        self.assertNotIn("stop h3-tp2-head.service", self.systemctl())
        self.dropin.unlink()
        bad = self.root / "bad.py"; bad.write_text("different\n")
        self.assertNotEqual(self.run_queue(H3_TRACE_FLICKER=str(bad)).returncode, 0)
        self.assertNotIn("stop h3-tp2-head.service", self.systemctl())

    def test_failed_start_restores_original(self):
        result = self.run_queue(H3_TEST_FAIL_START_ONCE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.state.read_text(), "active")
        self.assertFalse(self.dropin.exists())
        self.assertIn("H3-AUTO-HOLD vae-trace", (self.results / "gen.log").read_text())

    def test_activating_head_is_never_treated_as_stopped(self):
        result = self.run_queue(H3_TEST_START_ACTIVATING="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.state.read_text(), "activating")
        self.assertTrue(self.dropin.exists())
        self.assertEqual(self.systemctl().count("stop h3-tp2-head.service"), 1)

    def test_nonquiescent_stop_never_writes_dropin(self):
        result = self.run_queue(H3_TEST_STOP_ACTIVATING="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.state.read_text(), "activating")
        self.assertFalse(self.dropin.exists())

    def test_missing_or_broken_trace_stops_before_expensive_history_and_restores(self):
        for trace in ("missing", "warning"):
            with self.subTest(trace=trace):
                test = VaeTraceQueueTests(); test.setUp()
                try:
                    result = test.run_queue(H3_TEST_TRACE=trace)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(test.cases(), ["trace-smoke5"])
                    self.assertEqual(test.state.read_text(), "active")
                    self.assertFalse(test.dropin.exists())
                finally:
                    test.doCleanups()

    def test_new_busy_work_or_async_job_prevents_cleanup_stop(self):
        for mode in ("H3_TEST_BUSY_ON_FAIL", "H3_TEST_JOBS_ON_FAIL"):
            with self.subTest(mode=mode):
                test = VaeTraceQueueTests(); test.setUp()
                try:
                    result = test.run_queue(H3_TEST_FAIL_CASE="trace-smoke5", **{mode: "1"})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(test.state.read_text(), "active")
                    self.assertTrue(test.dropin.exists())
                    self.assertEqual(test.systemctl().count("stop h3-tp2-head.service"), 1)
                    self.assertIn("H3-AUTO-HOLD vae-trace", (test.results / "gen.log").read_text())
                finally:
                    test.doCleanups()

    def test_failed_long_case_restores_and_keeps_original(self):
        result = self.run_queue(H3_TEST_FAIL_CASE="trace-history30")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.state.read_text(), "active")
        self.assertFalse(self.dropin.exists())
        self.assertIn("H3-AUTO-HOLD vae-trace", (self.results / "gen.log").read_text())

    def test_changed_dropin_leaves_active_head_untouched(self):
        result = self.run_queue(H3_TEST_FAIL_CASE="trace-smoke5", H3_TEST_CHANGE_DROPIN="1", PYTHONOPTIMIZE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.state.read_text(), "active")
        self.assertEqual(self.dropin.read_text(), "foreign\n")
        self.assertEqual(self.systemctl().count("stop h3-tp2-head.service"), 1)


if __name__ == "__main__":
    unittest.main()
