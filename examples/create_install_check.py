"""Create Mode from a packaged wheel in a fresh environment (no checkout on the path).

Installs the wheel with the ``hf`` extra into a new virtual environment and then:

* builds the download-free fixture;
* runs traces (stopping after the fit, then resuming) and a corpus with an instruction;
* verifies both exports;
* uninstalls Rephix and reloads an export with plain Transformers, to show it does not
  depend on Rephix.

Usage:
  python -I examples/create_install_check.py --wheel dist/rephix-0.4.1-py3-none-any.whl \
      --work-dir NEW_DIR --output result.json [--torch-index-url URL]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

RESULTS = []

_RELOAD = r"""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = sys.argv[1]
assert not any(m == "rephix" or m.startswith("rephix.") for m in sys.modules)
try:
    import rephix  # noqa: F401
    installed = True
except ImportError:
    installed = False
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path).eval()
ids = tokenizer("The answer", return_tensors="pt")["input_ids"]
out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=8,
                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(json.dumps({"rephix_importable": installed, "generated_tokens": int(out.shape[1] - ids.shape[1])}))
"""


def record(name, passed, **evidence):
    RESULTS.append({"check": name, "passed": bool(passed), **evidence})
    print(("PASS " if passed else "FAIL ") + name, flush=True)


def run(argv, expect=0, **kwargs):
    completed = subprocess.run(
        [str(a) for a in argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )
    return completed.returncode == expect, completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-index-url")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if work.exists():
        raise SystemExit("use a new work directory")
    work.mkdir(parents=True)
    venv = work / "env ü"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    rephix = scripts / ("rephix.exe" if os.name == "nt" else "rephix")
    pip = [python, "-m", "pip", "--disable-pip-version-check", "-q"]
    if args.torch_index_url:
        ok, out = run([*pip, "install", "torch", "--index-url", args.torch_index_url])
        record("install: torch from the requested index", ok, stderr=out.stderr[-400:])
    ok, out = run([*pip, "install", f"{args.wheel.resolve()}[hf]"])
    record("install: wheel with the hf extra in a fresh environment", ok, stderr=out.stderr[-600:])
    version = args.wheel.name.split("-")[1]
    ok, out = run([rephix, "--version"], cwd=work)
    record(
        "version: installed package reports the wheel version",
        ok and out.stdout.strip() == version,
        reported=out.stdout.strip(),
    )
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    env.pop("PYTHONPATH", None)
    fixture = work / "fixture"
    ok, out = run([rephix, "create-fixture", "--output", fixture], cwd=work, env=env)
    record("fixture: download-free base and evidence", ok, stderr=out.stderr[-400:])
    common = ["--profile", "cpu-smoke", "--device", "cpu"]
    traces = work / "run traces"
    started = time.monotonic()
    ok, out = run(
        [
            rephix,
            "--threads",
            "2",
            "create",
            "--base",
            fixture / "base",
            "--evidence",
            fixture / "traces.jsonl",
            "--output",
            traces,
            *common,
            "--stop-after",
            "fit",
        ],
        cwd=work,
        env=env,
    )
    paused = ok and json.loads(out.stdout)["status"] == "paused"
    record("traces: run stops after the fit as an interruption", paused, stderr=out.stderr[-600:])
    ok, out = run([rephix, "--threads", "2", "create-resume", traces], cwd=work, env=env)
    done = json.loads(out.stdout) if ok else {}
    record(
        "traces: resume completes and exports",
        ok and done.get("status") == "complete",
        stderr=out.stderr[-600:],
        seconds=round(time.monotonic() - started, 1),
    )
    corpus = work / "run corpus"
    ok, out = run(
        [
            rephix,
            "--threads",
            "2",
            "create",
            "--base",
            fixture / "base",
            "--corpus",
            fixture / "corpus.txt",
            "--prompt",
            "Keep the formal voice",
            "--output",
            corpus,
            *common,
        ],
        cwd=work,
        env=env,
    )
    corpus_done = json.loads(out.stdout) if ok else {}
    record(
        "corpus: instruction-guided corpus-only run completes",
        ok and corpus_done.get("status") == "complete",
        stderr=out.stderr[-600:],
    )
    exports = [
        Path(done.get("export", traces / "export")),
        Path(corpus_done.get("export", corpus / "export")),
    ]
    for export in exports:
        ok, out = run([rephix, "create-export", "--verify", export], cwd=work, env=env)
        record(f"export: {export.parent.name} verifies", ok and json.loads(out.stdout)["valid"])
        verification = json.loads((export / "export-verification.json").read_text())
        record(
            f"export: {export.parent.name} schema, confinement and fresh reload",
            verification["passed"],
        )
    ok, out = run([rephix, "create-inspect", traces, "--verify"], cwd=work, env=env)
    record(
        "inspect: completed run verifies", ok and json.loads(out.stdout)["verification"]["passed"]
    )
    ok, out = run([*pip, "uninstall", "-y", "rephix"])
    record("uninstall: rephix removed from the environment", ok)
    ok, out = run([python, "-I", "-c", _RELOAD, exports[0]], cwd=work, env=env)
    info = json.loads(out.stdout.strip().splitlines()[-1]) if ok else {}
    record(
        "independence: export loads and generates with plain Transformers, Rephix absent",
        ok and info.get("rephix_importable") is False and info.get("generated_tokens") == 8,
        stderr=out.stderr[-400:],
    )
    failed = [r["check"] for r in RESULTS if not r["passed"]]
    args.output.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "wheel": args.wheel.name,
                "checks": RESULTS,
                "failed": failed,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} Create Mode install checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
