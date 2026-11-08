"""Verify real managed install/update/rollback/removal with disposable environments.

Pass wheels outside a new --work-dir. No existing installations are modified.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from rephix.installation import CATEGORIES
from rephix.io import write_json


def evaluate(args):
    root = args.work_dir.resolve()
    if root.exists():
        raise ValueError("Use a new disposable work directory.")
    root.mkdir(parents=True)
    prefix = root / "managed install"
    manager = Path(__file__).resolve().parents[1] / "src/rephix/installation.py"
    base = getattr(sys, "_base_executable", sys.executable)
    commands = []

    def run(argv, expected=0):
        result = subprocess.run(
            [str(a) for a in argv],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=root,
        )
        if result.returncode != expected:
            raise AssertionError(
                f"Unexpected exit {result.returncode}: {result.stdout}\n{result.stderr}"
            )
        commands.append({"exit": result.returncode})
        return result.stdout

    def launch(*argv, expected=0):
        return run([base, prefix / "launch.py", *argv], expected=expected)

    def state():
        return json.loads((prefix / "installation.json").read_text())

    def snapshot():
        return {
            str(p.relative_to(prefix)): hashlib.sha256(p.read_bytes()).hexdigest()
            for c in CATEGORIES
            for p in (prefix / c).rglob("*")
            if p.is_file()
        }

    old_wheel = (args.old_wheel or args.wheel).resolve()
    wheel = args.wheel.resolve()
    initial = json.loads(run([base, manager, "install", "--prefix", prefix, "--wheel", old_wheel]))
    initial_version = launch("--version").strip()
    for category in CATEGORIES:
        (prefix / category / "existing-user-artifact").write_text(f"preserve {category}\n")
    (prefix / "unrelated-note").write_text("user file, not managed software\n")
    original_data = snapshot()
    before_failure = (prefix / "installation.json").read_bytes()
    broken = root / "rephix-9.9.9-py3-none-any.whl"
    broken.write_bytes(b"invalid wheel for a failed upgrade test")
    launch("installation", "update", "--wheel", broken, expected=1)
    assert (prefix / "installation.json").read_bytes() == before_failure
    assert launch("--version").strip() == initial_version
    upgraded = json.loads(launch("installation", "update", "--wheel", wheel, "--extras", "tui"))
    updated_version = launch("--version").strip()
    assert upgraded["previous"] == initial["active"]
    assert snapshot() == original_data
    launch("installation", "inspect", "--verify")
    launch("ui", "--plain")
    launch("doctor")
    launch("installation", "rollback")
    assert launch("--version").strip() == initial_version
    assert snapshot() == original_data
    launch("installation", "rollback")
    assert launch("--version").strip() == updated_version
    runtime_lease_verified = False
    if os.name != "nt":
        import pty
        import termios
        import time

        master, slave = pty.openpty()
        child = subprocess.Popen(
            [base, str(prefix / "launch.py"), "ui", "--color", "never"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env={**os.environ, "TERM": "xterm-256color"},
        )
        try:
            for _ in range(100):
                if not (termios.tcgetattr(slave)[3] & termios.ICANON):
                    break
                if child.poll() is not None:
                    raise AssertionError("Workbench exited before runtime lease check")
                time.sleep(0.05)
            else:
                raise AssertionError("Workbench failed to acquire the terminal")
            before_busy = (prefix / "installation.json").read_bytes()
            launch("installation", "uninstall", expected=1)
            assert (prefix / "installation.json").read_bytes() == before_busy
            runtime_lease_verified = True
            os.write(master, b"\x03")
            assert child.wait(timeout=5) == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            os.close(master)
            os.close(slave)
    # Reinstalling the same release with no extras argument inherits tui.
    repeated = json.loads(launch("installation", "update", "--wheel", wheel))
    assert repeated["slots"][repeated["active"]]["extras"] == ["tui"]
    plan = json.loads(launch("installation", "prune"))
    assert plan["would_remove"] == [initial["active"]]
    launch("installation", "prune", "--apply")
    assert len(state()["slots"]) == 2
    launch("installation", "rollback")
    assert launch("--version").strip() == updated_version
    assert snapshot() == original_data
    retained = json.loads(launch("installation", "uninstall"))
    assert state()["active"] is None and state()["slots"] == {}
    assert snapshot() == original_data
    launch("--version", expected=1)
    # The retained bootstrap allows reinstall without wiping data or configuration.
    launch("installation", "install", "--wheel", wheel)
    assert snapshot() == original_data
    purged = json.loads(
        launch(
            "installation",
            "uninstall",
            "--purge-data",
            "--purge-config",
            "--purge-evidence",
            "--purge-checkpoints",
            "--remove-bootstrap",
        )
    )
    assert all(not (prefix / c).exists() for c in CATEGORIES)
    assert (prefix / "unrelated-note").read_text() == "user file, not managed software\n"
    assert not (prefix / "installation.json").exists()
    assert not (prefix / "launch.py").exists()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "initial_version": initial_version,
        "updated_version": updated_version,
        "initial_wheel_sha256": hashlib.sha256(old_wheel.read_bytes()).hexdigest(),
        "updated_wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "failed_upgrade_preserved_manifest_and_version": True,
        "updates_preserved_user_file_sha256": original_data,
        "inherited_extras": ["tui"],
        "rollback_validated": True,
        "active_workbench_blocked_uninstall": runtime_lease_verified,
        "prune_retained_current_and_previous": True,
        "default_uninstall_retained": retained["retained"],
        "reinstall_preserved_data": True,
        "explicit_purge_removed": purged["purged"],
        "unrelated_user_file_retained": True,
        "explicit_bootstrap_removal_verified": purged["bootstrap_removed"],
        "commands_checked": len(commands),
        "temporary_environments_remaining": len(list(prefix.glob("env-*"))),
        "windows": os.name == "nt",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--old-wheel", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
