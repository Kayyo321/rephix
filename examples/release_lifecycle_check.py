"""Verify release artifacts through install, upgrade, failure, removal and reinstall.

The check never uses the checkout's source tree for the product under test. It
installs only from the supplied release assets, the published v0.2.0 wheel and a
0.3.0 wheel built from its verified commit, into disposable directories.

Channel modes:
  --channel mirror   serve the candidate assets from a local release-API mirror with
                     the same JSON shape as GitHub (before publication);
  --channel github   use the real GitHub Releases API (after publication).

Usage:
  python -I examples/release_lifecycle_check.py --assets DIR --old-wheel-02 W02 \
      --old-wheel-03 W03 --installer-03 installation.py@553bfea --work-dir NEW \
      --channel mirror --output result.json
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

RESULTS = []
CATEGORIES = ("data", "config", "evidence", "checkpoints")


LAST = {}


def record(name, passed, **evidence):
    if not passed and LAST:
        evidence.setdefault("last_command", LAST.get("argv"))
        evidence.setdefault("last_exit", LAST.get("exit"))
        evidence.setdefault("last_stdout", LAST.get("stdout", "")[-800:])
        evidence.setdefault("last_stderr", LAST.get("stderr", "")[-1500:])
    RESULTS.append({"check": name, "passed": bool(passed), **evidence})
    print(("PASS " if passed else "FAIL ") + name, flush=True)
    return passed


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, env=None, cwd=None, expect=0):
    completed = subprocess.run(
        [str(a) for a in argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=cwd,
    )
    ok = completed.returncode == expect and "Traceback" not in completed.stderr
    LAST.update(
        argv=[str(a) for a in argv],
        exit=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    return ok, completed


def bin_path(venv, name):
    return (
        venv
        / ("Scripts" if os.name == "nt" else "bin")
        / (name + (".exe" if os.name == "nt" else ""))
    )


def launcher(prefix):
    if os.name == "nt":
        return prefix / ("rephix.exe" if (prefix / "rephix.exe").is_file() else "rephix.cmd")
    return prefix / "rephix"


class Mirror:
    """A GitHub-shaped release API for the candidate, plus a prerelease decoy."""

    def __init__(self, assets, version, corrupt=False):
        self.assets = assets
        self.version = version
        self.corrupt = corrupt
        handler = self._handler()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def api(self):
        return f"http://127.0.0.1:{self.port}/repos/Kayyo321/rephix/releases"

    def release_json(self, tag, prerelease=False):
        base = f"http://127.0.0.1:{self.port}/download/{tag}/"
        return {
            "tag_name": tag,
            "draft": False,
            "prerelease": prerelease,
            "html_url": base,
            "assets": [
                {
                    "name": p.name,
                    "size": p.stat().st_size,
                    "browser_download_url": base + p.name,
                    "digest": "sha256:" + sha256(p),
                }
                for p in sorted(self.assets.iterdir())
            ],
        }

    def _handler(self):
        mirror = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                tag = "v" + mirror.version
                if self.path.endswith("/releases/latest") or self.path.endswith("/tags/" + tag):
                    body = json.dumps(mirror.release_json(tag)).encode()
                elif self.path.endswith("/tags/v9.9.9"):
                    body = json.dumps(mirror.release_json("v9.9.9", prerelease=True)).encode()
                elif self.path.startswith("/download/"):
                    name = self.path.rsplit("/", 1)[1]
                    path = mirror.assets / name
                    if not path.is_file():
                        self.send_error(404)
                        return
                    body = path.read_bytes()
                    if mirror.corrupt and name.endswith(".whl"):
                        body = body[:-1] + bytes([body[-1] ^ 1])
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


def user_files(prefix):
    return {
        str(p.relative_to(prefix)): sha256(p)
        for c in CATEGORIES
        for p in (prefix / c).rglob("*")
        if p.is_file()
    }


def shell_integrations():
    """Files and settings a PATH-modifying installer would touch."""
    home = Path.home()
    names = [
        ".bashrc",
        ".bash_profile",
        ".profile",
        ".zshrc",
        ".zprofile",
        ".config/fish/config.fish",
    ]
    state = {n: sha256(home / n) for n in names if (home / n).is_file()}
    state["PATH"] = os.environ.get("PATH", "")
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            try:
                state["HKCU_Path"] = winreg.QueryValueEx(key, "Path")[0]
            except FileNotFoundError:
                state["HKCU_Path"] = None
    return state


def representative_workflow(rephix, work):
    """Core recovery demo and a small text-only recovery through a local oracle."""
    ok, demo = run([rephix, "demo", "--seed", "7"])
    report = json.loads(demo.stdout) if ok else {}
    demo_ok = ok and report["full_logits"]["parameter_relative_error"] < 1e-10
    folder = work / "workflow ü"
    folder.mkdir(exist_ok=True)
    vocab = list("abcdef")
    import random

    rng = random.Random(3)
    theta = [[rng.gauss(0, 1) for _ in range(7)] for _ in vocab]
    # Written with the installed environment's NumPy to avoid depending on the checkout.
    code = "import json,sys,numpy as np; np.savez(sys.argv[1], theta=np.array(json.loads(sys.argv[2])))"
    env_python = _env_python(rephix)
    subprocess.run(
        [str(env_python), "-I", "-c", code, str(folder / "t.npz"), json.dumps(theta)], check=True
    )
    (folder / "server.json").write_text(
        json.dumps(
            {
                "kind": "ngram",
                "vocabulary": vocab,
                "order": 1,
                "parameters": str(folder / "t.npz"),
                "seed": 1,
            }
        ),
        encoding="utf-8",
    )
    (folder / "features.json").write_text(
        json.dumps({"kind": "ngram", "vocabulary": vocab, "order": 1, "prompt_length": 3}),
        encoding="utf-8",
    )
    (folder / "target.json").write_text(
        json.dumps(
            {"argv": [str(rephix), "text-target-serve", "--config", str(folder / "server.json")]}
        ),
        encoding="utf-8",
    )
    ok, recovered = run(
        [
            rephix,
            "text-recover",
            "--features",
            folder / "features.json",
            "--target",
            folder / "target.json",
            "--output",
            folder / "export",
            "--contexts",
            "40",
            "--samples-per-context",
            "3000",
            "--bootstrap",
            "79",
            "--alpha",
            "0.05",
        ]
    )
    status = json.loads(recovered.stdout)["status"] if ok else recovered.stderr[-300:]
    ok2, reloaded = run(
        [
            rephix,
            "text-model",
            "generate",
            "--model",
            folder / "export",
            "--prompt",
            "abc",
            "--steps",
            "3",
        ]
    )
    return demo_ok and ok and status == "recovered_in_class" and ok2, status


def _env_python(rephix):
    rephix = Path(rephix)
    if rephix.name.startswith("rephix") and rephix.parent.name in ("bin", "Scripts"):
        return bin_path(rephix.parent.parent, "python")
    prefix = rephix.parent
    state = json.loads((prefix / "installation.json").read_text())
    return prefix / state["active"] / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def wheel_metadata_version(wheel):
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        for line in archive.read(name).decode().splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--old-wheel-02", type=Path, required=True)
    parser.add_argument("--old-wheel-03", type=Path, required=True)
    parser.add_argument("--installer-03", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--channel", choices=("mirror", "github"), default="mirror")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if work.exists():
        raise SystemExit("use a new work directory")
    work.mkdir(parents=True)
    assets = args.assets.resolve()
    base = getattr(sys, "_base_executable", sys.executable)
    integrations_before = shell_integrations()

    # 1. Integrity and version agreement of the artifacts themselves.
    sums = {}
    for line in (assets / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        sums[name.lstrip("*")] = digest
    listed = sorted(sums)
    present = sorted(p.name for p in assets.iterdir() if p.name != "SHA256SUMS")
    record(
        "assets: SHA256SUMS lists every asset and every digest matches",
        listed == present and all(sha256(assets / n) == d for n, d in sums.items()),
        listed=listed,
        present=present,
    )
    wheel = next(assets.glob("rephix-*-py3-none-any.whl"))
    version = wheel.name.split("-")[1]
    info = json.loads((assets / "BUILD-INFO.json").read_text())
    installer = assets / "rephix-installer.py"
    record(
        "assets: wheel name, wheel metadata, sdist and build info agree on the version",
        wheel_metadata_version(wheel) == version == info["version"]
        and (assets / f"rephix-{version}.tar.gz").is_file(),
        version=version,
        build=info,
    )

    env = dict(os.environ)
    mirror = None
    if args.channel == "mirror":
        mirror = Mirror(assets, version)
        env["REPHIX_RELEASES_API"] = mirror.api
    else:
        env.pop("REPHIX_RELEASES_API", None)

    # 2. Fresh install through the documented installer script and release discovery.
    prefix = work / "fresh app ü"
    ok, out = run(
        [
            base,
            "-I",
            installer,
            "install",
            "--prefix",
            prefix,
            "--release",
            "latest",
            "--extras",
            "tui",
        ],
        env=env,
    )
    report = json.loads(out.stdout) if ok else {}
    record(
        "fresh install: installer resolves the latest stable release and verifies it",
        ok and report.get("release", {}).get("tag") == "v" + version,
        stderr=out.stderr[-500:],
        release=report.get("release"),
    )
    rephix = launcher(prefix)
    ok, out = run([rephix, "--version"], env=env)
    record(
        "fresh install: launcher reports the release version",
        ok and out.stdout.strip() == version,
        reported=out.stdout.strip(),
    )
    slot = report.get("slots", {}).get(report.get("active"), {})
    if os.name == "nt":
        record(
            "fresh install: native rephix.exe launcher created", (prefix / "rephix.exe").is_file()
        )
    record(
        "fresh install: manifest records the verified wheel digest",
        slot.get("wheel_sha256") == sha256(wheel) and slot.get("version") == version,
    )
    passed, status = representative_workflow(rephix, work)
    record(
        "fresh install: representative workflow (affine demo, text-only recovery, export)",
        passed,
        status=status,
    )
    ok, out = run([rephix, "installation", "check-update"], env=env)
    record(
        "fresh install: check-update reports current",
        ok and json.loads(out.stdout)["update_available"] is False,
        stdout=out.stdout[-300:],
    )
    ok, out = run([rephix, "installation", "update", "--release", "latest"], env=env)
    record(
        "repeat: update to the same release is a no-op",
        ok
        and json.loads(out.stdout).get("already_current") is True
        and len(list(prefix.glob("env-*"))) == 1,
    )
    before = (prefix / "installation.json").read_bytes()
    ok, out = run(
        [base, "-I", installer, "install", "--prefix", prefix, "--release", "latest"],
        env=env,
        expect=1,
    )
    record(
        "repeat: second install is refused without changes",
        ok
        and "already installed" in out.stderr
        and before == (prefix / "installation.json").read_bytes(),
    )
    ok, out = run([rephix, "installation", "update", "--release", "v9.9.9"], env=env, expect=1)
    if args.channel == "mirror":
        record(
            "channel: prereleases are never selected",
            ok and "not a published stable" in out.stderr,
            stderr=out.stderr[-300:],
        )

    # 3. Upgrade a real pip installation of the published v0.2.0 release.
    venv = work / "pip env"
    subprocess.run([base, "-m", "venv", str(venv)], check=True)
    pip = [bin_path(venv, "python"), "-I", "-m", "pip", "--disable-pip-version-check"]
    ok, _ = run([*pip, "install", args.old_wheel_02])
    old = bin_path(venv, "rephix")
    ok2, out = run([old, "--version"])
    record(
        "upgrade 0.2 (pip): published v0.2.0 installs and runs",
        ok and ok2 and out.stdout.strip() == "0.2.0",
        wheel_sha256=sha256(args.old_wheel_02),
    )
    data = work / "pip user data"
    data.mkdir()
    run([old, "demo", "--output", data / "demo-0.2.json"])
    evidence = data / "evidence.jsonl"
    evidence.write_text(
        json.dumps({"id": "e1", "prompt": "Hi", "output": "Hello", "metadata": {"group_id": "g"}})
        + "\n"
    )
    data_before = {p.name: sha256(p) for p in data.iterdir()}
    ok, _ = run([*pip, "install", "--upgrade", wheel])
    ok2, out = run([old, "--version"])
    ok3, val = run([old, "validate", evidence])
    record(
        "upgrade 0.2 (pip): documented pip upgrade reaches the release and reads 0.2 evidence",
        ok
        and ok2
        and out.stdout.strip() == version
        and ok3
        and json.loads(val.stdout)["observations"] == 1
        and data_before == {p.name: sha256(p) for p in data.iterdir()},
    )
    ok, _ = run([*pip, "uninstall", "-y", "rephix"])
    record(
        "uninstall (pip): package removed, user files untouched",
        ok and not old.exists() and data_before == {p.name: sha256(p) for p in data.iterdir()},
    )

    # 4. Upgrade a managed 0.3.0 installation made by the 0.3.0 installer.
    for route in ("old-manager", "new-installer"):
        managed = work / f"managed 0.3 {route}"
        ok, out = run(
            [
                base,
                "-I",
                args.installer_03,
                "install",
                "--prefix",
                managed,
                "--wheel",
                args.old_wheel_03,
                "--extras",
                "tui",
            ]
        )
        rephix3 = launcher(managed)
        ok2, ver = run([rephix3, "--version"])
        record(
            f"upgrade 0.3 ({route}): 0.3.0 managed install",
            ok and ok2 and ver.stdout.strip() == "0.3.0",
            stderr=out.stderr[-300:],
        )
        for category in CATEGORIES:
            (managed / category / "kept ü.txt").write_text(
                f"{category} from 0.3\n", encoding="utf-8"
            )
        snapshot = user_files(managed)
        if route == "old-manager":
            # The 0.3 manager has no discovery: users download and verify, then update.
            ok, out = run([rephix3, "installation", "update", "--wheel", wheel], env=env)
        else:
            ok, out = run(
                [base, "-I", installer, "update", "--prefix", managed, "--release", "latest"],
                env=env,
            )
        ok2, ver = run([rephix3, "--version"])
        record(
            f"upgrade 0.3 ({route}): update activates the release and preserves user data",
            ok and ok2 and ver.stdout.strip() == version and user_files(managed) == snapshot,
            stderr=out.stderr[-400:],
        )
        refreshed = (managed / "manager.py").read_text(encoding="utf-8")
        has_discovery = "resolve_release" in refreshed
        record(
            f"upgrade 0.3 ({route}): manager adopted from the release"
            if route == "new-installer"
            else "upgrade 0.3 (old-manager): legacy manager kept and still works",
            has_discovery if route == "new-installer" else not has_discovery,
        )
        ok, out = run([rephix3, "installation", "rollback"], env=env)
        ok2, ver = run([rephix3, "--version"])
        record(
            f"upgrade 0.3 ({route}): rollback returns to 0.3.0",
            ok and ver.stdout.strip() == "0.3.0",
        )
        run([rephix3, "installation", "rollback"], env=env)
        if route == "new-installer":
            # 5. Failed updates: corrupt download and unreachable channel leave 0.4 active.
            state_before = (managed / "installation.json").read_bytes()
            if mirror is not None:
                bad = Mirror(assets, version, corrupt=True)
                bad_env = {**env, "REPHIX_RELEASES_API": bad.api}
                ok, out = run(
                    [rephix3, "installation", "update", "--release", "latest", "--force"],
                    env=bad_env,
                    expect=1,
                )
                record(
                    "failed update: corrupted download is rejected without changes",
                    ok
                    and "Integrity check failed" in out.stderr
                    and state_before == (managed / "installation.json").read_bytes(),
                    stderr=out.stderr[-300:],
                )
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                closed = probe.getsockname()[1]
            dead_env = {**env, "REPHIX_RELEASES_API": f"http://127.0.0.1:{closed}/r"}
            ok, out = run(
                [rephix3, "installation", "update", "--release", "latest", "--force"],
                env=dead_env,
                expect=1,
            )
            record(
                "failed update: unreachable channel is reported without changes",
                ok
                and "Cannot reach the release channel" in out.stderr
                and state_before == (managed / "installation.json").read_bytes(),
            )
            broken = work / f"rephix-{version}.post9-py3-none-any.whl"
            broken.write_bytes(b"not a wheel")
            ok, out = run([rephix3, "installation", "update", "--wheel", broken], expect=1)
            ok2, ver = run([rephix3, "--version"])
            record(
                "failed update: invalid wheel leaves the active version usable",
                ok and ok2 and ver.stdout.strip() == version,
            )
            # 6. Uninstall, retention, reinstall.
            ok, out = run([rephix3, "installation", "uninstall"])
            report = json.loads(out.stdout) if ok else {}
            record(
                "uninstall: environments removed, user directories retained",
                ok
                and not list(managed.glob("env-*"))
                and user_files(managed) == snapshot
                and report.get("retained") == list(CATEGORIES),
            )
            ok, out = run([rephix3, "--version"], expect=1)
            record("uninstall: launcher reports the uninstalled state", ok)
            ok, out = run([rephix3, "installation", "install", "--release", "latest"], env=env)
            ok2, ver = run([rephix3, "--version"])
            record(
                "reinstall: retained bootstrap reinstalls the release with data intact",
                ok and ok2 and ver.stdout.strip() == version and user_files(managed) == snapshot,
            )
            (managed / "unrelated.txt").write_text("not managed")
            ok, out = run(
                [
                    rephix3,
                    "installation",
                    "uninstall",
                    "--purge-evidence",
                    "--purge-checkpoints",
                    "--remove-bootstrap",
                ]
            )
            report = json.loads(out.stdout) if ok else {}
            record(
                "purge: only requested directories and the bootstrap are removed",
                ok
                and not (managed / "evidence").exists()
                and not (managed / "checkpoints").exists()
                and (managed / "data" / "kept ü.txt").is_file()
                and (managed / "config" / "kept ü.txt").is_file()
                and (managed / "unrelated.txt").read_text() == "not managed"
                and not (managed / "installation.json").exists(),
            )
            ok, out = run(
                [
                    base,
                    "-I",
                    installer,
                    "install",
                    "--prefix",
                    work / "after purge",
                    "--release",
                    "latest",
                ],
                env=env,
            )
            record("reinstall: fresh prefix after bootstrap removal", ok)
        else:
            run([rephix3, "installation", "uninstall"])

    integrations_after = shell_integrations()
    record(
        "integrations: PATH, shell startup files and user environment untouched",
        integrations_before == integrations_after,
    )
    failed = [r["check"] for r in RESULTS if not r["passed"]]
    result = {
        "channel": args.channel,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "version": version,
        "assets": {n: sums[n] for n in listed},
        "old_wheels": {"0.2.0": sha256(args.old_wheel_02), "0.3.0": sha256(args.old_wheel_03)},
        "checks": RESULTS,
        "failed": failed,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.output.write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} lifecycle checks passed")
    shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
