"""restart_head_gated must really restart after a slow probe.

`restart_head` skips the restart when the head is healthy and already runs the requested env and code.  After a slow
boot probe that is exactly the state of the boot to be replaced, so the "restart" of the gate was a no-op and the
slow boot was probed three times and kept.  Runs the real functions in bash with stub commands; no head needed.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB = """#!/bin/bash
echo "$(basename "$0") $*" >> "$HOME/calls.log"
case "$(basename "$0")" in
  curl) printf 200;;
  systemctl) [ "$1 $2" = "--user is-active" ] && exit 0;;
esac
exit 0
"""
PROBE = """#!/bin/bash
n=$(cat "$HOME/probes" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "$HOME/probes"
[ $n -le ${SLOW_PROBES:-0} ] && exit 2
exit 0
"""


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class GatedRestartTests(unittest.TestCase):
    def run_gate(self, slow_probes):
        home = Path(tempfile.mkdtemp(prefix="h3-queue-lib-"))
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        (home / "h3-tp2-results").mkdir()
        (home / "h3-tp2" / "scripts").mkdir(parents=True)
        (home / "bin").mkdir()
        for name in ("curl", "systemctl", "sudo", "ssh", "journalctl", "sleep"):
            stub = home / "bin" / name
            stub.write_text(STUB)
            stub.chmod(0o755)
        (home / "h3-tp2" / "scripts" / "boot-probe.sh").write_text(PROBE)
        lib = (ROOT / "scripts" / "queue-lib.sh").read_text()
        # the env file the library would write for an arm without experiment lines, and matching code revisions
        base = lib.split("BASEENV='", 1)[1].split("'", 1)[0]
        (home / ".h3-tp2.env").write_text("export H3_MODEL=/weights/FL2VA\n" + base + "\n")
        (home / "h3-tp2" / ".synced-rev").write_text("abc\n")
        (home / "h3-tp2-results" / ".head-rev").write_text("abc\n")
        env = dict(os.environ, HOME=str(home), PATH=f"{home / 'bin'}:{os.environ['PATH']}", SLOW_PROBES=str(slow_probes))
        res = subprocess.run(["bash", "-c", f". {ROOT / 'scripts' / 'queue-lib.sh'}; restart_head_gated ''; echo rc=$?"],
                             env=env, capture_output=True, text=True, timeout=60)
        calls = (home / "calls.log").read_text().splitlines() if (home / "calls.log").exists() else []
        return res.stdout, calls, int((home / "probes").read_text())

    def test_a_healthy_head_with_the_same_env_is_not_restarted(self):
        out, calls, probes = self.run_gate(slow_probes=0)
        self.assertIn("no restart", out)
        self.assertIn("rc=0", out)
        self.assertEqual(probes, 1)
        self.assertFalse([c for c in calls if c.startswith("systemctl --user stop")])

    def test_a_slow_probe_forces_a_real_restart(self):
        out, calls, probes = self.run_gate(slow_probes=1)
        self.assertIn("slow boot (attempt 1), restarting", out)
        self.assertEqual(out.count("no restart"), 1)                      # only the first call took the shortcut
        self.assertEqual(len([c for c in calls if c.startswith("systemctl --user stop")]), 1)
        self.assertEqual(len([c for c in calls if c.startswith("systemctl --user start")]), 1)
        self.assertEqual(probes, 2)
        self.assertIn("rc=0", out)

    def test_three_slow_boots_restart_twice_and_keep_the_last(self):
        out, calls, probes = self.run_gate(slow_probes=3)
        self.assertEqual(len([c for c in calls if c.startswith("systemctl --user stop")]), 2)
        self.assertEqual(probes, 3)
        self.assertIn("three slow boots in a row", out)
        self.assertIn("rc=2", out)


if __name__ == "__main__":
    unittest.main()
