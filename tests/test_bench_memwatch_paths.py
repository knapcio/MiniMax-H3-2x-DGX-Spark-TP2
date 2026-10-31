"""Worker memory sampling can be isolated without changing the default directory."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


BENCH = Path(__file__).resolve().parents[1] / 'scripts/bench-matrix.sh'


class BenchMemwatchPathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scripts = self.root / 'scripts'; self.scripts.mkdir()
        shutil.copy(BENCH, self.scripts / 'bench-matrix.sh')
        (self.scripts / 'env.sh').write_text('WORKER_IP=worker\n')
        (self.scripts / 'memwatch.sh').write_text('#!/bin/bash\nexit 0\n')
        (self.scripts / 'render.sh').write_text('#!/bin/bash\nprintf video > "$1"\n')
        (self.scripts / 'verify-output.sh').write_text('#!/bin/bash\necho verified\n')
        bindir = self.root / 'bin'; bindir.mkdir()
        ssh = bindir / 'ssh'
        ssh.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$H3_TEST_SSH_LOG"\n'
                       'case "$*" in *" cat "*) printf "samples=2 min_MemAvailable_MiB=9000\\n1,9000,0,0\\n2,9000,0,0\\n";; esac\n')
        ssh.chmod(0o755)
        self.log = self.root / 'ssh.log'
        self.env = dict(os.environ, HOME=str(self.root), PATH=f'{bindir}:{os.environ["PATH"]}',
                        H3_TEST_SSH_LOG=str(self.log), TASK='fl2va')

    def run_case(self, name, **changes):
        env = dict(self.env, OUT=str(self.root / name), **changes)
        return subprocess.run(['bash', str(self.scripts / 'bench-matrix.sh'), '5:960:544:20:1101'],
                              env=env, capture_output=True, text=True, timeout=15)

    def test_default_keeps_existing_worker_directory(self):
        result = self.run_case('normal')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('~/h3-tp2-results/memwatch/d5_960x544_s20_seed1101.mem-worker.csv', self.log.read_text())

    def test_opt_in_separates_each_output(self):
        for name in ('case-a', 'case-b'):
            result = self.run_case(name, H3_MEMWATCH_WORKER_DIR='h3-tp2-results/memwatch/new-series',
                                   H3_MEMWATCH_PER_OUTPUT='1')
            self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text()
        self.assertIn('~/h3-tp2-results/memwatch/new-series/case-a/d5_', calls)
        self.assertIn('~/h3-tp2-results/memwatch/new-series/case-b/d5_', calls)

    def test_unsafe_worker_path_fails_before_ssh(self):
        for path in ('/absolute', '../other', 'ok/../../other', 'ok;false'):
            with self.subTest(path=path):
                result = self.run_case('unsafe', H3_MEMWATCH_WORKER_DIR=path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('invalid worker memory directory', result.stderr)
                self.assertFalse(self.log.exists())


if __name__ == '__main__':
    unittest.main()
