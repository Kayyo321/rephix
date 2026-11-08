"""Critical installation transitions, failures, locking, and deletion boundaries."""

import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rephix import installation as manager


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "managed"
        self.wheel = self.base / "rephix-0.3.0-py3-none-any.whl"
        self.wheel.write_bytes(b"test wheel; subprocess boundary is mocked")
        self.args = Namespace(
            prefix=str(self.root),
            wheel=str(self.wheel),
            extras="tui",
            wheelhouse=None,
            constraints=None,
            operation="install",
        )
        self.addCleanup(patch.stopall)
        patch.object(manager.venv.EnvBuilder, "create", lambda _, path: path.mkdir()).start()
        patch.object(manager, "_run", return_value="dependency==1.0").start()
        self.validate = patch.object(manager, "_validate", return_value="0.3.0").start()

    def install(self):
        return manager.install(self.args)

    def test_update_rollback_and_selective_uninstall_preserve_user_data(self):
        first = self.install()["active"]
        for category in manager.CATEGORIES:
            (self.root / category / "keep").write_text(category)
        unrelated = self.root / "user-notes.txt"
        unrelated.write_text("keep this too")
        self.args.operation, self.args.extras = "update", None
        second = self.install()["active"]
        state = manager._read(self.root)
        self.assertEqual(state["previous"], first)
        self.assertEqual(state["slots"][second]["extras"], ["tui"])
        rolled = manager.rollback(self.args)
        self.assertEqual(rolled["active"], first)
        self.assertEqual(rolled["previous"], second)
        for category in manager.CATEGORIES:
            setattr(self.args, "purge_" + category, category == "evidence")
        report = manager.uninstall(self.args)
        self.assertTrue(report["uninstalled"])
        self.assertEqual(report["purged"], ["evidence"])
        self.assertEqual(manager._read(self.root)["slots"], {})
        self.assertFalse((self.root / "evidence").exists())
        self.assertEqual(unrelated.read_text(), "keep this too")
        for category in ("data", "config", "checkpoints"):
            self.assertEqual((self.root / category / "keep").read_text(), category)

    def test_bootstrap_can_be_removed_while_preserving_user_files(self):
        self.install()
        for category in manager.CATEGORIES:
            setattr(self.args, "purge_" + category, False)
            (self.root / category / "user-file").write_text("keep")
        self.args.remove_bootstrap = True
        report = manager.uninstall(self.args)
        self.assertTrue(report["bootstrap_removed"])
        self.assertFalse((self.root / manager.MANIFEST).exists())
        self.assertFalse((self.root / "manager.py").exists())
        self.assertFalse((self.root / "launch.py").exists())
        self.assertTrue((self.root / ".installation.lock").exists())
        for category in manager.CATEGORIES:
            self.assertEqual((self.root / category / "user-file").read_text(), "keep")

    def test_failed_or_cancelled_update_keeps_active_manifest_unchanged(self):
        self.install()
        self.args.operation = "update"
        before = (self.root / manager.MANIFEST).read_bytes()
        for error in (RuntimeError("failed dependency check"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.validate.side_effect = error
                with self.assertRaises(type(error)):
                    self.install()
                self.assertEqual((self.root / manager.MANIFEST).read_bytes(), before)
                self.assertEqual(len(list(self.root.glob("env-*"))), 1)

    def test_force_interrupted_slot_can_be_pruned_without_losing_rollback(self):
        first = self.install()["active"]
        self.args.operation = "update"
        second = self.install()["active"]
        state = manager._read(self.root)
        orphan = "env-" + "a" * 32
        state["slots"][orphan] = {"version": None, "extras": [], "status": "preparing"}
        (self.root / orphan).mkdir()
        manager._save(self.root, state)
        self.args.apply = False
        self.assertEqual(manager.prune(self.args)["would_remove"], [orphan])
        self.assertTrue((self.root / orphan).exists())
        self.args.apply = True
        self.assertEqual(manager.prune(self.args)["removed"], [orphan])
        self.assertTrue((self.root / first).exists())
        self.assertTrue((self.root / second).exists())

    def test_current_environment_cannot_be_deleted_or_deactivated(self):
        first = self.install()["active"]
        for category in manager.CATEGORIES:
            setattr(self.args, "purge_" + category, False)
        before = (self.root / manager.MANIFEST).read_bytes()
        with patch.object(manager.sys, "prefix", str(self.root / first)):
            with self.assertRaisesRegex(ValueError, "managed launcher"):
                manager.uninstall(self.args)
        self.assertEqual((self.root / manager.MANIFEST).read_bytes(), before)

    def test_unrelated_prefix_future_schema_and_corrupt_metadata_fail_closed(self):
        self.root.mkdir()
        (self.root / "valuable").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "unrelated"):
            self.install()
        (self.root / "valuable").unlink()
        self.install()
        before = json.loads((self.root / manager.MANIFEST).read_text())
        for change in ({"schema_version": 2}, {"active": []}, {"slots": {"../outside": {}}}):
            manager._save(self.root, {**before, **change})
            original = (self.root / manager.MANIFEST).read_bytes()
            with self.assertRaises(ValueError):
                manager.inspect_installation(Namespace(prefix=str(self.root), verify=False))
            self.assertEqual((self.root / manager.MANIFEST).read_bytes(), original)

    def test_concurrent_operations_are_rejected(self):
        self.install()
        with manager._lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "holds the lock"):
                manager.rollback(self.args)

    def test_symlink_purge_and_slot_deletion_never_follow_external_paths(self):
        first = self.install()["active"]
        external = self.base / "external"
        external.mkdir()
        sentinel = external / "important"
        sentinel.write_text("preserve")
        evidence = self.root / "evidence"
        evidence.rmdir()
        try:
            evidence.symlink_to(external, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable")
        for category in manager.CATEGORIES:
            setattr(self.args, "purge_" + category, category == "evidence")
        before = (self.root / manager.MANIFEST).read_bytes()
        with self.assertRaisesRegex(ValueError, "symlink"):
            manager.uninstall(self.args)
        self.assertEqual((self.root / manager.MANIFEST).read_bytes(), before)
        self.assertEqual(sentinel.read_text(), "preserve")
        evidence.unlink()
        shutil.rmtree(self.root / first)
        (self.root / first).symlink_to(external, target_is_directory=True)
        self.args.purge_evidence = False
        with self.assertRaisesRegex(ValueError, "regular directory"):
            manager.uninstall(self.args)
        self.assertEqual((self.root / manager.MANIFEST).read_bytes(), before)
        self.assertEqual(sentinel.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
