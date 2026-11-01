"""Validate staged launch precedence using a fake executable, never a service."""
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from scripts.stage_unfused_head import stage


REPO = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UnfusedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.source = self.root / 'source'; self.source.mkdir()
        (self.source / 'start-head.sh').write_bytes((REPO / 'scripts/start-head.sh').read_bytes())
        bindir = self.root / 'runtime/bin'; bindir.mkdir(parents=True)
        vllm = bindir / 'vllm'
        vllm.write_text('#!/bin/bash\nprintf "%s\\n" "$H3_TP2_FP8_FUSE_SWIGLU" "$H3_TP2_ROLE" "$@"\n')
        vllm.chmod(0o700)
        vals = {'H3_TP2_FP8_FUSE_SWIGLU': '1', 'H3_VENV': str(self.root / 'runtime'), 'H3_MODEL': '/fake/weights',
                'H3_API_HOST': '127.0.0.1', 'H3_API_PORT': '18093', 'HEAD_IP': '127.0.0.1'}
        (self.source / 'env.sh').write_text(''.join(f'export {key}={shlex.quote(value)}\n' for key, value in vals.items()))
        self.head_sha = sha(self.source / 'start-head.sh'); self.env_sha = sha(self.source / 'env.sh')

    def run_head(self, path):
        env = dict(os.environ, H3_TP2_FP8_FUSE_SWIGLU='0')
        return subprocess.run(['bash', str(path / 'start-head.sh'), '--test-argument'], env=env, text=True,
                              capture_output=True, check=True).stdout.splitlines()

    def test_override_follows_sourced_site_values_without_changing_other_arguments(self):
        original = self.run_head(self.source)
        output = self.root / 'staged'
        receipt = stage(self.source, output, head_sha=self.head_sha, env_sha=self.env_sha)
        staged = self.run_head(output)
        self.assertEqual(original[0], '1')
        self.assertEqual(staged[0], '0')
        self.assertEqual(original[1:], staged[1:])
        self.assertEqual(staged[1], 'head')
        self.assertEqual(receipt['staged_head_sha256'], sha(output / 'start-head.sh'))
        self.assertEqual(receipt['staged_env_sha256'], self.env_sha)
        self.assertEqual((output.stat().st_mode & 0o777), 0o700)

    def test_refuses_changed_source_existing_output_and_symlink(self):
        out = self.root / 'staged'
        with self.assertRaises(ValueError):
            stage(self.source, out, head_sha='0' * 64, env_sha=self.env_sha)
        self.assertFalse(out.exists())
        out.mkdir()
        with self.assertRaises(FileExistsError):
            stage(self.source, out, head_sha=self.head_sha, env_sha=self.env_sha)
        (self.source / 'env.sh').unlink()
        (self.source / 'env.sh').symlink_to(self.source / 'start-head.sh')
        with self.assertRaises(ValueError):
            stage(self.source, self.root / 'other', head_sha=self.head_sha, env_sha=self.env_sha)


if __name__ == '__main__':
    unittest.main()
