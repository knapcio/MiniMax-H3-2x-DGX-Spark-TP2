import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPT = HERE / "deploy-prompt-fc14580.sh"
TXN = HERE / "file_txn.py"
FILES = ("h3_tp2/plugin.py", "h3_tp2/cont_prompts.py", "scripts/render.sh", "h3_tp2/h3_output.py")


class PromptDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"; self.bin.mkdir()
        self.results = self.root / "results"; self.results.mkdir()
        self.live, self.stage, self.backup = (self.root / x for x in ("live", "stage", "backup"))
        self.wlive, self.wstage, self.wbackup = (self.root / x for x in ("wlive", "wstage", "wbackup"))
        for root in (self.live, self.stage, self.wlive, self.wstage):
            (root / "h3_tp2").mkdir(parents=True)
            (root / "scripts").mkdir()
        for path in FILES:
            old = subprocess.run(["git", "-C", str(REPO), "show", "0ed86ba:" + path], capture_output=True, check=True).stdout
            new = (REPO / path).read_bytes()
            for root in (self.live, self.wlive):
                (root / path).write_bytes(old)
            for root in (self.stage, self.wstage):
                (root / path).write_bytes(new)
        for root in (self.stage, self.wstage):
            shutil.copy2(REPO / "h3_tp2/vae_trace.py", root / "h3_tp2/vae_trace.py")
        tag = "d16_960x576_s20_seed1101_fl2va"
        (self.results / "quality16-960-stream").mkdir()
        (self.results / "quality16-960-stream/STATUS").write_text(tag + " ok\n")
        (self.results / "quality16-960-stream" / (tag + ".mp4")).write_bytes(b"video")
        (self.results / "gen.log").write_text("H3-AUTO-SKIP-300 quality guard 2026-10-30 23:00 CET\n")
        (self.results / "koninck.png").write_bytes(b"painting")
        (self.results / "quality-20261030.lock").touch()
        (self.root / "state").write_text("active")
        (self.root / "meminfo").write_text("MemAvailable: 30000000 kB\n")
        (self.root / "proc/123").mkdir(parents=True)
        (self.root / "proc/123/environ").write_bytes(
            b"H3_TP2_REQUEST_TIMEOUT_S=28800\0VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=28800\0VLLM_OMNI_VIDEO_SYNC_TIMEOUT=29400\0")
        self.command("flock", "exit 0")
        self.command("nvidia-smi", "printf '%s\\n' \"${H3_TEST_GPU:-0}\"")
        self.command("systemctl", '''
printf '%s\\n' "$*" >> "$H3_TEST_SYSTEMCTL"
case " $* " in
  *" show "*) printf '123\\n'; exit 0;;
  *" stop h3-tp2-head.service "*) printf stopped > "$H3_TEST_STATE"; exit 0;;
  *" start h3-tp2-head.service "*)
    if [ "${H3_TEST_FAIL_START_ONCE:-0}" = 1 ] && [ ! -e "$H3_TEST_FAIL_MARKER" ]; then touch "$H3_TEST_FAIL_MARKER"; exit 1; fi
    printf active > "$H3_TEST_STATE"; exit 0;;
  *" is-active "*" h3-tp2-head.service "*) [ "$(cat "$H3_TEST_STATE")" = active ]; exit $?;;
esac
exit 0''')
        self.command("ssh", '''
case "$*" in
  *sha256sum*) sha256sum "$H3_PATCH_TXN"; exit $?;;
  *"test ! -e"*) [ ! -e "$H3_TEST_WBACKUP" ]; exit $?;;
  *file_txn.py*)
    case "$*" in
      *"'verify-new'"*) op=verify-new;; *"'verify'"*) op=verify;;
      *"'apply'"*) op=apply;; *"'restore'"*) op=restore;; *) exit 3;;
    esac
    if [ "$op" = apply ] && [ "${H3_TEST_REMOTE_APPLY_FAIL:-0}" = 1 ]; then exit 1; fi
    if [ "$op" = restore ] && [ "${H3_TEST_REMOTE_RESTORE_FAIL:-0}" = 1 ]; then exit 1; fi
    python3 "$H3_PATCH_TXN" "$op" --live "$H3_TEST_WLIVE" --stage "$H3_TEST_WSTAGE" --backup "$H3_TEST_WBACKUP"; exit $?;;
  *h3-tp2-agent.service*) exit 0;;
  *nvidia-smi*) printf '0\\n'; exit 0;;
  *MemAvailable*) printf '30000000\\n'; exit 0;;
esac
exit 1''')
        self.command("curl", '''
case "$*" in
  */health*) if [ "$(cat "$H3_TEST_STATE")" = active ] && [ "${H3_TEST_BAD_HEALTH:-0}" != 1 ]; then printf 200; else printf 000; fi;;
  */metrics*) printf 'vllm_omni:num_requests_running{x="0"} 0\\nvllm_omni:num_requests_waiting{x="0"} 0\\n';;
  */v1/videos*) if [ "${H3_TEST_JOBS:-0}" = 1 ]; then printf '{"data":[{}],"has_more":false}'; else printf '{"data":[],"has_more":false}'; fi;;
  *) exit 1;;
esac''')
        self.command("journalctl", "printf 'continuation prompts generated for 2 windows (16.0 s)\\n'")
        self.command("timeout", "shift 3; exec \"$@\"")
        self.bench = self.root / "bench.sh"
        self.bench.write_text('''#!/bin/bash
set -e
[ "${H3_TEST_BENCH_FAIL:-0}" != 1 ] || exit 1
IFS=: read -r dur w h steps seed <<< "$1"
suffix=""; [ "$TASK" = fl2va ] && suffix=_fl2va
tag="d${dur}_${w}x${h}_s${steps}_seed${seed}${suffix}"
mkdir -p "$OUT"
printf '%s ok\\n' "$tag" > "$OUT/STATUS"
printf 'video' > "$OUT/$tag.mp4"
if [ "$TASK" = fl2va ]; then
  printf '{"effective_duration":16,"extra_params":{"duration_seconds":30,"target":{"duration_seconds":16}}}' > "$OUT/$tag.mp4.timing.json"
fi
''')
        self.env = os.environ.copy()
        self.env.update(PATH=str(self.bin) + os.pathsep + self.env["PATH"], H3_PATCH_RESULTS=str(self.results),
                        H3_PATCH_TXN=str(TXN), H3_PATCH_LIVE=str(self.live), H3_PATCH_STAGE=str(self.stage),
                        H3_PATCH_BACKUP=str(self.backup), H3_PATCH_BENCH=str(self.bench),
                        H3_PATCH_MEMINFO=str(self.root / "meminfo"), H3_PATCH_PROC_ROOT=str(self.root / "proc"),
                        H3_PATCH_WAIT_ATTEMPTS="1", H3_PATCH_IDLE_ATTEMPTS="3", H3_PATCH_READY_ATTEMPTS="3",
                        H3_PATCH_WAIT_S="0", H3_PATCH_IDLE_S="0", H3_PATCH_READY_S="0", H3_TEST_STATE=str(self.root / "state"),
                        H3_TEST_SYSTEMCTL=str(self.root / "systemctl.log"), H3_TEST_FAIL_MARKER=str(self.root / "fail-marker"),
                        H3_TEST_WLIVE=str(self.wlive), H3_TEST_WSTAGE=str(self.wstage), H3_TEST_WBACKUP=str(self.wbackup))

    def command(self, name, body):
        p = self.bin / name
        p.write_text("#!/bin/bash\n" + body + "\n")
        p.chmod(0o755)

    def run_deploy(self, **overrides):
        return subprocess.run(["bash", str(SCRIPT)], env=dict(self.env, **overrides), text=True,
                              capture_output=True, timeout=20)

    def test_completed_queue_deploys_both_and_runs_verified_smokes(self):
        result = self.run_deploy()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        for root in (self.live, self.wlive):
            self.assertEqual((root / "h3_tp2/plugin.py").read_bytes(), (REPO / "h3_tp2/plugin.py").read_bytes())
            self.assertTrue((root / "h3_tp2/vae_trace.py").is_file())
        self.assertIn("H3-AUTO-PATCH-DONE", (self.results / "gen.log").read_text())
        self.assertEqual((self.root / "state").read_text(), "active")

    def test_existing_async_job_or_busy_gpu_prevents_stop(self):
        for overrides in ({"H3_TEST_JOBS": "1"}, {"H3_TEST_GPU": "40"}):
            with self.subTest(overrides=overrides):
                result = self.run_deploy(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.backup.exists())
                log = (self.root / "systemctl.log").read_text()
                self.assertNotIn("stop h3-tp2-head.service", log)

    def test_failed_new_head_rolls_back_both_nodes_and_restarts_old_head(self):
        old = (self.live / "h3_tp2/plugin.py").read_bytes()
        result = self.run_deploy(H3_TEST_FAIL_START_ONCE="1")
        self.assertNotEqual(result.returncode, 0)
        for root in (self.live, self.wlive):
            self.assertEqual((root / "h3_tp2/plugin.py").read_bytes(), old)
            self.assertFalse((root / "h3_tp2/vae_trace.py").exists())
        self.assertEqual((self.root / "state").read_text(), "active")
        self.assertIn("H3-AUTO-PATCH-ROLLBACK-OK", (self.results / "gen.log").read_text())

    def test_remote_apply_failure_restores_old_service(self):
        result = self.run_deploy(H3_TEST_REMOTE_APPLY_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.live / "h3_tp2/vae_trace.py").exists())
        self.assertEqual((self.root / "state").read_text(), "active")

    def test_preflight_refuses_changed_live_code_before_stopping(self):
        (self.live / "h3_tp2/plugin.py").write_bytes(b"different live revision")
        result = self.run_deploy()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.backup.exists())
        self.assertFalse((self.root / "systemctl.log").exists())

    def test_failed_restore_keeps_head_stopped(self):
        result = self.run_deploy(H3_TEST_FAIL_START_ONCE="1", H3_TEST_REMOTE_RESTORE_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / "state").read_text(), "stopped")
        self.assertIn("rollback incomplete", result.stdout)

    def test_smoke_failure_leaves_healthy_head_and_records_hold(self):
        result = self.run_deploy(H3_TEST_BENCH_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / "state").read_text(), "active")
        self.assertTrue((self.live / "h3_tp2/vae_trace.py").is_file())
        self.assertIn("H3-AUTO-HOLD prompt-patch-fc14580 phase=healthy", (self.results / "gen.log").read_text())

    def test_new_head_that_cannot_be_proven_idle_is_not_stopped_again(self):
        # Break health only after the first stop, so preflight can pass.
        curl = self.bin / "curl"
        code = curl.read_text().replace('[ "${H3_TEST_BAD_HEALTH:-0}" != 1 ]',
                                        '[ "${H3_TEST_BAD_HEALTH:-0}" != 1 ] || [ ! -e "$H3_TEST_FAIL_MARKER" ]')
        curl.write_text(code)
        # Fake systemctl touches the marker on start while still starting successfully.
        systemctl = self.bin / "systemctl"
        code = systemctl.read_text().replace('printf active > "$H3_TEST_STATE"; exit 0;;',
                                             'touch "$H3_TEST_FAIL_MARKER"; printf active > "$H3_TEST_STATE"; exit 0;;')
        systemctl.write_text(code)
        result = self.run_deploy(H3_TEST_BAD_HEALTH="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / "state").read_text(), "active")
        self.assertIn("not verifiably idle", result.stdout)
        self.assertNotIn("H3-AUTO-PATCH-ROLLBACK-OK", (self.results / "gen.log").read_text())

    def test_unfinished_predecessor_does_not_stop_head(self):
        (self.results / "gen.log").write_text("")
        result = self.run_deploy()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "systemctl.log").exists())
        self.assertFalse(self.backup.exists())

    def test_300_success_marker_without_its_file_cannot_trigger_restart(self):
        (self.results / "gen.log").write_text("H3-AUTO-QUALITY-DONE 2026-10-31 09:00 CET\n")
        result = self.run_deploy()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "systemctl.log").exists())
        self.assertFalse(self.backup.exists())


if __name__ == "__main__":
    unittest.main()
