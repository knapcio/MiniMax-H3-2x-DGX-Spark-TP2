import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("file_txn", Path(__file__).with_name("file_txn.py"))
txn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(txn)


class FileTransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.live, self.stage, self.backup = (self.root / name for name in ("live", "stage", "backup"))
        for root in (self.live, self.stage):
            (root / "pkg").mkdir(parents=True)
        self.old = {"pkg/module.py": b"VALUE = 1\n", "pkg/existing.py": b"old\n"}
        self.new = {"pkg/module.py": b"VALUE = 2\n", "pkg/existing.py": b"new\n", "pkg/new.py": b"added\n"}
        h = lambda value: hashlib.sha256(value).hexdigest()
        self.spec = {name: (h(self.old[name]) if name in self.old else None, h(value)) for name, value in self.new.items()}
        for name, value in self.old.items():
            (self.live / name).write_bytes(value)
        for name, value in self.new.items():
            (self.stage / name).write_bytes(value)

    def test_apply_and_restore_preserve_bytes_and_new_file_existence(self):
        txn.apply(self.live, self.stage, self.backup, self.spec)
        self.assertTrue(all((self.live / name).read_bytes() == value for name, value in self.new.items()))
        txn.verify_new(self.live, self.spec)
        txn.restore(self.live, self.backup, self.spec)
        self.assertTrue(all((self.live / name).read_bytes() == value for name, value in self.old.items()))
        self.assertFalse((self.live / "pkg/new.py").exists())

    def test_rejects_mismatched_source_before_mutation(self):
        (self.stage / "pkg/module.py").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "staged file"):
            txn.apply(self.live, self.stage, self.backup, self.spec)
        self.assertFalse(self.backup.exists())
        self.assertEqual((self.live / "pkg/module.py").read_bytes(), self.old["pkg/module.py"])

    def test_partial_failure_restores_prior_files(self):
        original = txn.atomic_copy
        count = 0
        def fail_once(source, destination):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("injected failure")
            return original(source, destination)
        with patch.object(txn, "atomic_copy", side_effect=fail_once), self.assertRaisesRegex(OSError, "injected"):
            txn.apply(self.live, self.stage, self.backup, self.spec)
        self.assertTrue(all((self.live / name).read_bytes() == value for name, value in self.old.items()))
        self.assertFalse((self.live / "pkg/new.py").exists())

    def test_restore_refuses_unexpected_external_change(self):
        txn.apply(self.live, self.stage, self.backup, self.spec)
        (self.live / "pkg/existing.py").write_bytes(b"external change")
        with self.assertRaisesRegex(ValueError, "unexpected file"):
            txn.restore(self.live, self.backup, self.spec)
        self.assertEqual((self.live / "pkg/module.py").read_bytes(), self.new["pkg/module.py"])

    def test_refuses_symlink_and_existing_backup(self):
        (self.live / "pkg/existing.py").unlink()
        (self.live / "pkg/existing.py").symlink_to(self.stage / "pkg/existing.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            txn.apply(self.live, self.stage, self.backup, self.spec)
        (self.live / "pkg/existing.py").unlink()
        (self.live / "pkg/existing.py").write_bytes(self.old["pkg/existing.py"])
        self.backup.mkdir()
        with self.assertRaises(FileExistsError):
            txn.apply(self.live, self.stage, self.backup, self.spec)

    def test_python_bytecode_cache_does_not_mask_a_same_size_change(self):
        def load():
            s = importlib.util.spec_from_file_location("dynamic_h3_test", self.live / "pkg/module.py")
            m = importlib.util.module_from_spec(s)
            s.loader.exec_module(m)
            return m.VALUE
        self.assertEqual(load(), 1)
        txn.apply(self.live, self.stage, self.backup, self.spec)
        self.assertEqual(load(), 2)
        txn.restore(self.live, self.backup, self.spec)
        self.assertEqual(load(), 1)


if __name__ == "__main__":
    unittest.main()
