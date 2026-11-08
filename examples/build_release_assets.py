"""Build the complete, checksummed release asset set from one commit.

Writes into --output: the wheel, the sdist, rephix-installer.py (the standalone
standard-library installer), BUILD-INFO.json and SHA256SUMS covering every file.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise SystemExit("refusing to build release assets from a modified checkout:\n" + status)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    match = re.search(r'^version = "([^"]+)"$', (REPO / "pyproject.toml").read_text(), re.M)
    version = match.group(1)
    init = (REPO / "src/rephix/__init__.py").read_text()
    if f'__version__ = "{version}"' not in init:
        raise SystemExit("pyproject and rephix.__version__ disagree")
    subprocess.run([sys.executable, "-m", "build", "--outdir", str(output)], cwd=REPO, check=True)
    shutil.copyfile(REPO / "src/rephix/installation.py", output / "rephix-installer.py")
    expected = {
        f"rephix-{version}-py3-none-any.whl",
        f"rephix-{version}.tar.gz",
        "rephix-installer.py",
    }
    if {p.name for p in output.iterdir()} != expected:
        raise SystemExit(f"unexpected build outputs: {sorted(p.name for p in output.iterdir())}")
    info = {
        "version": version,
        "tag": "v" + version,
        "commit": commit,
        "python": sys.version.split()[0],
    }
    (output / "BUILD-INFO.json").write_text(json.dumps(info, indent=1) + "\n")
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(output.iterdir())
    ]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
