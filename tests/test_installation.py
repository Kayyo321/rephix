"""Critical installation transitions, failures, locking, and deletion boundaries."""

import contextlib
import io
import json
import shutil
import subprocess
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
            release=None,
            force=True,
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

    def test_launcher_reports_corrupt_manifest_without_a_traceback(self):
        self.install()
        (self.root / manager.MANIFEST).write_text("{corrupted")
        result = subprocess.run(
            [manager.sys.executable, str(self.root / "launch.py"), "--version"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Cannot read managed installation", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_venv_bootstrap_failure_is_a_clear_cli_error(self):
        from rephix.cli import main

        error = subprocess.CalledProcessError(1, ["python", "-m", "ensurepip"])
        output = io.StringIO()
        with (
            patch.object(manager.venv.EnvBuilder, "create", side_effect=error),
            contextlib.redirect_stderr(output),
        ):
            self.assertEqual(
                main(
                    [
                        "installation",
                        "install",
                        "--prefix",
                        str(self.root),
                        "--wheel",
                        str(self.wheel),
                    ]
                ),
                1,
            )
        self.assertIn("rephix:", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())
        self.assertIsNone(manager._read(self.root)["active"])
        self.assertEqual(manager._read(self.root)["slots"], {})

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


class BasePythonTests(unittest.TestCase):
    def test_base_python_from_inside_a_venv_is_outside_it(self):
        import venv

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "env"
            venv.EnvBuilder(with_pip=False).create(root)
            python = root / ("Scripts/python.exe" if manager.os.name == "nt" else "bin/python")
            source = Path(manager.__file__).read_text(encoding="utf-8")
            script = Path(directory) / "probe.py"
            source = source.replace('if __name__ == "__main__":', "if False:")
            script.write_text(source + "\nprint(_base_python())\n", encoding="utf-8")
            result = subprocess.run(
                [str(python), "-I", str(script)], capture_output=True, text=True, check=True
            )
            # Unresolved: a venv interpreter is often a symlink to the base itself.
            base = Path(result.stdout.strip()).absolute()
            self.assertNotIn(root.absolute(), base.parents)
            self.assertTrue(base.is_file())
            self.assertTrue(Path(manager._base_python()).is_file())


class HttpChannelErrorTests(unittest.TestCase):
    def test_truncated_transfer_is_a_clean_channel_error(self):
        import http.client

        with patch("urllib.request.urlopen", side_effect=http.client.IncompleteRead(b"x")):
            with self.assertRaisesRegex(RuntimeError, "Cannot reach the release channel"):
                manager._http_get("https://example.invalid", 10)


class WindowsLauncherTemplateTests(unittest.TestCase):
    def test_template_falls_back_to_launcher_files_for_older_pip(self):
        import types

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("t64.exe", "t64-arm.exe", "t32.exe"):
                (folder / name).write_bytes(b"launcher-" + name.encode())
            fake = types.SimpleNamespace(__file__=str(folder / "scripts.py"))  # no WRAPPERS
            package = types.ModuleType("pip._vendor.distlib")
            package.scripts = fake
            with patch.dict(
                "sys.modules", {"pip._vendor.distlib": package, "pip._vendor.distlib.scripts": fake}
            ):
                self.assertTrue(manager._windows_launcher_template().startswith(b"launcher-t"))


class ReleaseChannelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.wheel_bytes = b"wheel bytes"
        digest = __import__("hashlib").sha256(self.wheel_bytes).hexdigest()
        self.release = {
            "tag_name": "v9.8.7",
            "draft": False,
            "prerelease": False,
            "html_url": "https://example.invalid/r",
            "assets": [
                {
                    "name": "rephix-9.8.7-py3-none-any.whl",
                    "size": len(self.wheel_bytes),
                    "browser_download_url": "https://example.invalid/w",
                },
                {"name": "SHA256SUMS", "browser_download_url": "https://example.invalid/s"},
            ],
        }
        self.files = {
            "https://example.invalid/w": self.wheel_bytes,
            "https://example.invalid/s": f"{digest}  rephix-9.8.7-py3-none-any.whl\n".encode(),
        }

        def fake_get(url, limit):
            if url.endswith("/latest") or "/tags/" in url:
                return json.dumps(self.release).encode()
            return self.files[url]

        patcher = patch.object(manager, "_http_get", side_effect=fake_get)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_latest_stable_release_resolves_and_verifies(self):
        resolved = manager.resolve_release("latest")
        self.assertEqual(resolved["version"], "9.8.7")
        path, digest = manager.download_release(resolved, self.base)
        self.assertEqual(path.read_bytes(), self.wheel_bytes)
        self.assertEqual(len(digest), 64)

    def test_checksum_mismatch_and_prereleases_are_rejected(self):
        self.files["https://example.invalid/w"] = b"tampered"
        self.release["assets"][0]["size"] = len(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "Integrity check failed"):
            manager.download_release(manager.resolve_release("latest"), self.base)
        self.assertEqual(list(self.base.iterdir()), [])
        self.release["prerelease"] = True
        # A prerelease is installable only when requested explicitly by its tag.
        self.assertEqual(manager.resolve_release("v9.8.7")["tag"], "v9.8.7")
        self.release["draft"] = True
        with self.assertRaisesRegex(RuntimeError, "draft"):
            manager.resolve_release("v9.8.7")
        self.release["draft"] = False
        with self.assertRaisesRegex(ValueError, "latest or a tag"):
            manager.resolve_release("main")

    def test_install_requires_exactly_one_source_and_update_is_idempotent(self):
        args = Namespace(
            prefix=str(self.base / "m"),
            wheel=None,
            release=None,
            extras=None,
            wheelhouse=None,
            constraints=None,
            operation="install",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            manager.install(args)
        wheel = self.base / "rephix-9.8.7-py3-none-any.whl"
        wheel.write_bytes(self.wheel_bytes)
        with (
            patch.object(manager.venv.EnvBuilder, "create", lambda _, path: path.mkdir()),
            patch.object(manager, "_run", return_value="x==1"),
            patch.object(manager, "_validate", return_value="9.8.7"),
        ):
            args.wheel = str(wheel)
            first = manager.install(args)
            args.operation, args.force = "update", False
            again = manager.install(args)
            self.assertTrue(again["already_current"])
            self.assertIn("bootstrap_refreshed", again)  # repairs still run when current
            self.assertEqual(again["active"], first["active"])
            self.assertEqual(len(list((self.base / "m").glob("env-*"))), 1)
            state = manager._read(self.base / "m")
            state["slots"][first["active"]]["version"] = "0.3.0"
            manager._save(self.base / "m", state)
            report = manager.check_update(Namespace(prefix=str(self.base / "m"), release="latest"))
            self.assertTrue(report["update_available"])
            self.assertEqual(report["command"], "rephix installation update --release v9.8.7")


if __name__ == "__main__":
    unittest.main()
