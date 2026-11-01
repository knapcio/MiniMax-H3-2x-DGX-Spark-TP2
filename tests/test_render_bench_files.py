"""Exercise the matrix wrapper with inert local render/memwatch/SSH stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class BenchFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root / 'scripts'
        scripts.mkdir()
        shutil.copyfile(REPO / 'scripts/bench-matrix.sh', scripts / 'bench-matrix.sh')
        (scripts / 'env.sh').write_text('export PATH="$TEST_BIN:$PATH"\nWORKER_IP=unused\n')
        (scripts / 'memwatch.sh').write_text('exit 0\n')
        (scripts / 'verify-output.sh').write_text('exit 0\n')
        (scripts / 'render.sh').write_text('''exec "$TEST_PY" -c '
import json,os,sys
from pathlib import Path
Path(os.environ["TEST_CAPTURE"]).write_text(json.dumps({"args":sys.argv[1:],"file":os.environ.get("H3_PROMPT_FILE"),"prompt_exported":"PROMPT" in os.environ}))
' "$@"
''')
        bindir = self.root / 'bin'
        bindir.mkdir()
        ssh = bindir / 'ssh'
        ssh.write_text('#!/bin/sh\nprintf stub >> "$TEST_SSH_CAPTURE"\nexit 0\n')
        ssh.chmod(0o755)
        self.env = os.environ.copy()
        for name in ('PROMPT', 'H3_PROMPT_FILE', 'H3_MEMWATCH_PER_OUTPUT', 'H3_MEMWATCH_WORKER_DIR'):
            self.env.pop(name, None)
        self.env.update(TEST_BIN=str(bindir), TEST_PY=sys.executable, TEST_CAPTURE=str(self.root / 'captured.json'),
                        TEST_SSH_CAPTURE=str(self.root / 'ssh.txt'), OUT=str(self.root / 'output'))

    def call(self, **updates):
        return subprocess.run(['bash', str(self.root / 'scripts/bench-matrix.sh'), '300:960:544:20:3101'],
                              env=dict(self.env, **updates), capture_output=True, timeout=20)

    def test_file_mode_passes_empty_positional_placeholder_without_exporting_prompt(self):
        path = self.root / 'long-prompt.txt'
        path.write_text('scene;\n' * 30000)
        result = self.call(H3_PROMPT_FILE=str(path), PROMPT='')
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        capture = json.loads((self.root / 'captured.json').read_text())
        self.assertEqual(capture['args'][6], '')
        self.assertEqual(capture['file'], str(path))
        self.assertFalse(capture['prompt_exported'])

    def test_no_file_preserves_default_and_custom_prompts(self):
        for text in ('', 'Custom; Polish scene.'):
            with self.subTest(text=text):
                result = self.call(PROMPT=text)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                capture = json.loads((self.root / 'captured.json').read_text())
                self.assertEqual(capture['args'][6], text or (
                    'Slow drifting clouds over rolling green hills at golden hour, soft wind in the grass, '
                    'gentle light moving across the valley, calm, photoreal, steady camera.'))

    def test_conflict_fails_before_any_ssh_or_render(self):
        result = self.call(H3_PROMPT_FILE='prompt.txt', PROMPT='conflicting text')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'ssh.txt').exists())
        self.assertFalse((self.root / 'captured.json').exists())


if __name__ == '__main__':
    unittest.main()
