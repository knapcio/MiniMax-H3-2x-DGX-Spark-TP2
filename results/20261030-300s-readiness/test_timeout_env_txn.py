import importlib.util
import os
from pathlib import Path
import stat
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("timeout_env_txn.py")
spec = importlib.util.spec_from_file_location("timeout_env_txn", SCRIPT)
txn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(txn)


class TimeoutEnvTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3-timeout-env-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / "head.env"
        self.meta = self.root / "meta.json"
        self.content = ("# unrelated text preserved\nEXAMPLE_SETTING=stay\n" +
                        "".join(f"export {name}={value}\n" for name, value in txn.OLD.items())).encode()
        self.env.write_bytes(self.content)
        self.env.chmod(0o640)

    def test_apply_and_restore_change_only_timeouts(self):
        txn.prepare(self.env, self.meta)
        record = self.meta.read_text()
        self.assertNotIn("EXAMPLE_SETTING", record)
        self.assertEqual(stat.S_IMODE(self.meta.stat().st_mode), 0o600)
        self.assertEqual(txn.state(self.env, self.meta), "old")
        txn.apply(self.env, self.meta)
        self.assertEqual(txn.state(self.env, self.meta), "new")
        self.assertIn(b"EXAMPLE_SETTING=stay", self.env.read_bytes())
        self.assertEqual(stat.S_IMODE(self.env.stat().st_mode), 0o640)
        txn.restore(self.env, self.meta)
        self.assertEqual(self.env.read_bytes(), self.content)
        txn.restore(self.env, self.meta)  # idempotent

    def test_changed_env_refuses_apply_and_restore(self):
        txn.prepare(self.env, self.meta)
        self.env.write_bytes(self.content + b"EXTERNAL=1\n")
        with self.assertRaises(ValueError): txn.apply(self.env, self.meta)
        self.env.write_bytes(self.content)
        txn.apply(self.env, self.meta)
        changed = self.env.read_bytes() + b"EXTERNAL=2\n"
        self.env.write_bytes(changed)
        self.assertEqual(txn.state(self.env, self.meta), "changed")
        with self.assertRaises(ValueError): txn.restore(self.env, self.meta)
        self.assertEqual(self.env.read_bytes(), changed)

    def test_duplicate_dynamic_missing_or_wrong_timeout_refused(self):
        values = [
            self.content + b"H3_TP2_REQUEST_TIMEOUT_S=28800\n",
            self.content.replace(b"H3_TP2_REQUEST_TIMEOUT_S=28800", b'H3_TP2_REQUEST_TIMEOUT_S="${TIMEOUT}"'),
            self.content.replace(b"H3_TP2_REQUEST_TIMEOUT_S=28800", b"OTHER=28800"),
            self.content.replace(b"H3_TP2_REQUEST_TIMEOUT_S=28800", b"H3_TP2_REQUEST_TIMEOUT_S=2"),
            self.content.replace(b"H3_TP2_REQUEST_TIMEOUT_S=28800", b"H3_TP2_REQUEST_TIMEOUT_S=028800"),
        ]
        for value in values:
            with self.subTest(value=value):
                self.env.write_bytes(value)
                with self.assertRaises(ValueError): txn.prepare(self.env, self.meta)
                self.assertFalse(self.meta.exists())
                self.assertEqual(self.env.read_bytes(), value)

    def test_existing_metadata_and_symlinks_are_never_overwritten(self):
        self.meta.write_text("foreign")
        with self.assertRaises(FileExistsError): txn.prepare(self.env, self.meta)
        self.assertEqual(self.meta.read_text(), "foreign")
        self.meta.unlink()
        real = self.root / "real.env"; self.env.rename(real); self.env.symlink_to(real)
        with self.assertRaises(ValueError): txn.prepare(self.env, self.meta)
        self.assertEqual(real.read_bytes(), self.content)

    def test_file_changed_immediately_before_replace_is_preserved(self):
        data, before = txn.snapshot(self.env)
        changed = data + b"EXTERNAL=1\n"
        self.env.write_bytes(changed)
        with self.assertRaises(ValueError): txn.atomic_write(self.env, b"replacement", before)
        self.assertEqual(self.env.read_bytes(), changed)

    def test_crlf_whitespace_and_commented_assignments_are_preserved(self):
        original = ("# H3_TP2_REQUEST_TIMEOUT_S=not used\r\n" +
                    "".join(f"  export {name} = {value}  \r\n" for name, value in txn.OLD.items())).encode()
        self.env.write_bytes(original)
        txn.prepare(self.env, self.meta)
        txn.apply(self.env, self.meta)
        txn.restore(self.env, self.meta)
        self.assertEqual(self.env.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
