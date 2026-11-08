"""Standard-library bootstrap and transactional, explicitly scoped installation lifecycle.

Can run directly before Rephix is installed:
python src/rephix/installation.py install --prefix PATH --wheel PATH.whl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import venv
from contextlib import contextmanager
from pathlib import Path

SCHEMA = 1
MANIFEST = "installation.json"
CATEGORIES = ("data", "config", "evidence", "checkpoints")
EXTRAS = {"tui", "hf", "diffusion", "gguf", "analysis"}
SLOT = re.compile(r"env-[0-9a-f]{32}\Z")


def _atomic(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save(root, state):
    _atomic(root / MANIFEST, json.dumps(state, indent=2, allow_nan=False) + "\n")


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return path


def _root(value):
    if not value:
        raise ValueError("Choose --prefix, or run the managed installation's launcher.")
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise ValueError("The installation prefix must not be a symlink.")
    root = raw.resolve()
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError("Use a dedicated installation directory, not a filesystem root or home.")
    return root


def _read(root):
    try:
        state = json.loads(_regular(root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read managed installation: {root}: {exc}") from exc
    if (
        not isinstance(state, dict)
        or type(state.get("schema_version")) is not int
        or state["schema_version"] != SCHEMA
        or set(state) != {"schema_version", "active", "previous", "slots"}
    ):
        raise ValueError(
            "Unsupported installation schema; use a compatible installer. No changes made."
        )
    slots = state.get("slots")
    if not isinstance(slots, dict) or any(not SLOT.fullmatch(key) for key in slots):
        raise ValueError("Invalid installation environment records.")
    for record in slots.values():
        if (
            not isinstance(record, dict)
            or record.get("status") not in ("preparing", "ready")
            or not isinstance(record.get("extras"), list)
            or any(not isinstance(extra, str) or extra not in EXTRAS for extra in record["extras"])
        ):
            raise ValueError("Invalid installation environment metadata.")
    for key in ("active", "previous"):
        if state[key] is not None and (
            not isinstance(state[key], str)
            or state[key] not in slots
            or slots[state[key]]["status"] != "ready"
        ):
            raise ValueError(f"Invalid {key} environment in installation manifest.")
    return state


def _slot(root, name):
    if not isinstance(name, str) or not SLOT.fullmatch(name):
        raise ValueError("Invalid environment identifier.")
    path = root / name
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"Environment must be a regular directory: {path}")
    return path


def _python(path):
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


@contextmanager
def _lock(root):
    path = root / ".installation.lock"
    if path.is_symlink():
        raise ValueError("Installation lock must not be a symlink.")
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "Rephix is running or another installation operation holds the lock. Retry after it exits."
            ) from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _run(argv):
    # Keep pip progress and diagnostics separate from the CLI's JSON result.
    result = subprocess.run(
        [str(a) for a in argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    if result.returncode:
        raise RuntimeError(
            f"Installation check failed (exit {result.returncode}):\n{result.stdout[-12000:]}"
        )
    return result.stdout.strip()


def _validate(path):
    python = _python(path)
    _run([python, "-I", "-m", "pip", "check"])
    version = _run([python, "-I", "-m", "rephix", "--version"])
    report = json.loads(_run([python, "-I", "-m", "rephix", "demo"]))
    if report["full_logits"]["parameter_relative_error"] > 1e-10:
        raise RuntimeError("The new environment failed the reconstruction smoke check.")
    return version


def _bootstrap(root):
    # Stable launcher runs with the base Python, so Windows can remove inactive
    # environments without holding their python.exe open. It forwards signals
    # to the child by sharing the terminal's foreground process group.
    launcher = """import json, os, pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parent
os.environ['REPHIX_INSTALL_ROOT'] = str(root)
if sys.argv[1:2] == ['installation']:
    argv = [sys.executable, str(root / 'manager.py'), *sys.argv[2:]]
else:
    lease = (root / '.installation.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            lease.seek(0)
            msvcrt.locking(lease.fileno(), msvcrt.LK_NBRLCK, 1)
        else:
            import fcntl
            fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        sys.exit('An installation operation is running. Retry after it exits.')
    state = json.loads((root / 'installation.json').read_text(encoding='utf-8'))
    name = state.get('active')
    import re
    if state.get('schema_version') != 1 or not isinstance(name, str) or not re.fullmatch(r'env-[0-9a-f]{32}', name):
        sys.exit('Rephix is uninstalled or its installation manifest is invalid.')
    slot = root / name
    if slot.is_symlink():
        sys.exit('Invalid installation environment: symlink.')
    python = slot / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    argv = [str(python), '-I', '-m', 'rephix', *sys.argv[1:]]
try:
    child = subprocess.Popen(argv)
    while True:
        try:
            sys.exit(child.wait())
        except KeyboardInterrupt:
            continue
except OSError as exc:
    sys.exit('Cannot launch Rephix: ' + str(exc))
"""
    _atomic(root / "launch.py", launcher)
    _atomic(root / "manager.py", Path(__file__).read_text(encoding="utf-8"))
    base = getattr(sys, "_base_executable", sys.executable)
    if os.name == "nt":
        if any(c in base for c in ('"', "%", "\n", "\r")):
            raise ValueError("Python path cannot be safely represented by the Windows launcher.")
        _atomic(root / "rephix.cmd", f'@"{base}" "%~dp0launch.py" %*\n')
    else:
        # A shell wrapper supports Python paths containing spaces without moving venvs.
        import shlex

        _atomic(
            root / "rephix",
            "#!/bin/sh\nexec " + shlex.quote(base) + ' "$(dirname "$0")/launch.py" "$@"\n',
        )
        (root / "rephix").chmod(0o755)


def _report(root, state):
    return {
        "prefix": str(root),
        "schema_version": state["schema_version"],
        "active": state["active"],
        "previous": state["previous"],
        "slots": state["slots"],
        "launcher": str(root / ("rephix.cmd" if os.name == "nt" else "rephix")),
        "user_directories": {name: str(root / name) for name in CATEGORIES},
    }


def install(args):
    root = _root(args.prefix)
    wheel = _regular(Path(args.wheel).expanduser().resolve())
    if wheel.suffix != ".whl" or not wheel.name.startswith("rephix-"):
        raise ValueError("Supply a built Rephix .whl file with --wheel.")
    if root == wheel.parent or root in wheel.parents:
        raise ValueError("Keep the installation wheel outside the managed installation.")
    if args.extras is not None:
        extras = sorted(set(filter(None, args.extras.split(","))))
        if set(extras) - EXTRAS:
            raise ValueError("Supported extras: " + ", ".join(sorted(EXTRAS)))
    else:
        extras = None
    if not root.exists():
        root.mkdir(parents=True)
    elif not (root / MANIFEST).exists() and any(
        p.name != ".installation.lock" for p in root.iterdir()
    ):
        raise ValueError("The prefix contains unrelated files. Choose a new dedicated directory.")
    with _lock(root):
        if (root / MANIFEST).exists():
            state = _read(root)
        else:
            state = {"schema_version": SCHEMA, "active": None, "previous": None, "slots": {}}
            _save(root, state)
        if not state["slots"] and state["active"] is None:
            _bootstrap(root)
        if args.operation == "install" and state["active"]:
            raise ValueError("Rephix is already installed here; use installation update.")
        if args.operation == "update" and not state["active"]:
            raise ValueError("No active installation; use installation install.")
        if extras is None:
            extras = state["slots"][state["active"]]["extras"] if state["active"] else []
        name = "env-" + uuid.uuid4().hex
        path = _slot(root, name)
        record = {
            "version": None,
            "wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "extras": extras,
            "status": "preparing",
        }
        # Record the slot before creating it so forced interruption leaves a
        # discoverable inactive environment, removable with prune.
        state["slots"][name] = record
        _save(root, state)
        try:
            venv.EnvBuilder(with_pip=True).create(path)
            command = [
                _python(path),
                "-I",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
            ]
            if args.wheelhouse:
                command += ["--no-index", "--find-links", str(Path(args.wheelhouse).resolve())]
            if args.constraints:
                command += ["--constraint", str(_regular(Path(args.constraints).resolve()))]
            requirement = str(wheel) + ("[" + ",".join(extras) + "]" if extras else "")
            _run([*command, requirement])
            record["version"] = _validate(path)
            _atomic(
                path / "requirements.lock.txt",
                _run([_python(path), "-I", "-m", "pip", "freeze", "--all"]) + "\n",
            )
            record["status"] = "ready"
        except BaseException:
            # Preserve the old active pointer, including on Ctrl+C. If a partial
            # Windows environment cannot be removed, its preparing record remains.
            try:
                if path.exists():
                    shutil.rmtree(path)
                del state["slots"][name]
                _save(root, state)
            except OSError:
                pass
            raise
        for category in CATEGORIES:
            directory = root / category
            if not directory.exists():
                directory.mkdir()
        state["previous"], state["active"] = state["active"], name
        try:
            _save(root, state)
        except OSError as exc:
            raise RuntimeError(
                "Activation could not be confirmed; inspect the installation before retrying."
            ) from exc
        return _report(root, state)


def inspect_installation(args):
    root = _root(args.prefix)
    state = _read(root)
    report = _report(root, state)
    if args.verify and state["active"]:
        report["verified_version"] = _validate(_slot(root, state["active"]))
    return report


def rollback(args):
    root = _root(args.prefix)
    with _lock(root):
        state = _read(root)
        previous = state["previous"]
        if previous is None:
            raise ValueError("No previous installation is available for rollback.")
        _validate(_slot(root, previous))
        state["active"], state["previous"] = previous, state["active"]
        _save(root, state)
        return _report(root, state)


def _removal_paths(root, names):
    # Validate every target before the first deletion. Never follow a root slot symlink.
    paths = [(name, _slot(root, name)) for name in names]
    current = Path(sys.prefix).resolve()
    for _, path in paths:
        if current == path.resolve() or path.resolve() in current.parents:
            raise ValueError(
                "Run this operation through the managed launcher outside its environment."
            )
    return paths


def _remove_slots(root, state, names):
    for name, path in _removal_paths(root, names):
        if path.exists():
            shutil.rmtree(path)
        del state["slots"][name]
        _save(root, state)


def prune(args):
    root = _root(args.prefix)
    with _lock(root):
        state = _read(root)
        names = [
            name for name in state["slots"] if name not in (state["active"], state["previous"])
        ]
        if args.apply:
            _remove_slots(root, state, names)
        return {
            "prefix": str(root),
            "removed" if args.apply else "would_remove": names,
            "previous_retained": state["previous"],
        }


def uninstall(args):
    root = _root(args.prefix)
    with _lock(root):
        state = _read(root)
        categories = [name for name in CATEGORIES if getattr(args, "purge_" + name)]
        for name in categories:
            path = root / name
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise ValueError(f"Refusing to purge a non-directory or symlink: {path}")
        _removal_paths(root, list(state["slots"]))
        bootstrap_files = [
            root / name for name in ("rephix", "rephix.cmd", "launch.py", "manager.py", MANIFEST)
        ]
        if getattr(args, "remove_bootstrap", False):
            for path in bootstrap_files:
                if path.exists() or path.is_symlink():
                    _regular(path)
        # Disable new launches atomically, retaining enough state to retry a
        # failed or interrupted uninstall. Existing user directories are separate.
        state["active"] = state["previous"] = None
        _save(root, state)
        _remove_slots(root, state, list(state["slots"]))
        for name in categories:
            path = root / name
            if path.exists():
                shutil.rmtree(path)
        if getattr(args, "remove_bootstrap", False):
            for path in bootstrap_files:
                path.unlink(missing_ok=True)
        return {
            "prefix": str(root),
            "uninstalled": True,
            "bootstrap_removed": getattr(args, "remove_bootstrap", False),
            "coordination_lock_retained": True,
            "purged": categories,
            "retained": [n for n in CATEGORIES if n not in categories],
            "external_paths_touched": False,
        }


def _commands(parser):
    commands = parser.add_subparsers(dest="operation", required=True)
    for name, function, description in (
        ("install", install, "install a wheel in an isolated managed environment"),
        ("update", install, "validate a fresh environment before activating it"),
        ("inspect", inspect_installation, "show installation and retained user directories"),
        ("rollback", rollback, "activate the previous verified environment"),
        ("prune", prune, "remove obsolete environments, preserving rollback"),
        ("uninstall", uninstall, "remove software and optionally selected user directories"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--prefix", default=os.environ.get("REPHIX_INSTALL_ROOT"))
        command.set_defaults(func=function)
        if name in ("install", "update"):
            command.add_argument("--wheel", required=True, help="local built Rephix wheel")
            command.add_argument(
                "--extras", help="comma-separated extras; update inherits by default"
            )
            command.add_argument("--constraints", help="pip constraints file")
            command.add_argument(
                "--wheelhouse", help="offline wheel directory; disables index access"
            )
        elif name == "inspect":
            command.add_argument("--verify", action="store_true")
        elif name == "prune":
            command.add_argument(
                "--apply", action="store_true", help="apply the displayed removal plan"
            )
        elif name == "uninstall":
            command.add_argument(
                "--remove-bootstrap",
                action="store_true",
                help="also remove launcher/manager/manifest; retain the coordination lock",
            )
            for category in CATEGORIES:
                command.add_argument(
                    "--purge-" + category,
                    action="store_true",
                    help=f"remove only PREFIX/{category}; default retains it",
                )


def add_subcommands(commands):
    _commands(
        commands.add_parser("installation", help="manage isolated installs, updates and removal")
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    _commands(parser)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(args.func(args), indent=2, allow_nan=False))
    except KeyboardInterrupt:
        print("rephix: installation interrupted; inspect or retry the operation", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"rephix: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
