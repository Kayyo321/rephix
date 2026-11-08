"""End-to-end MPS verification through Rephix's real command-line entry points.

Run on Apple silicon with a usable Metal device (not a virtualized runner GPU):

  python examples/mps_check.py --work-dir /abs/new-dir --output mps-result.json

Every MPS step runs with PYTORCH_ENABLE_MPS_FALLBACK unset, so any operator without
an MPS kernel raises instead of silently running on the CPU; reports must also show
``accelerator.device == "mps"`` with no fallback operators and nonzero Metal driver
allocation. CPU runs of the same commands are the numerical reference. With
``--expect-unavailable`` (hosts without a usable Metal device) the script instead
verifies the explicit unavailable-device error and CPU behaviour.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

CHECKS = []


def check(name, passed, **evidence):
    CHECKS.append({"check": name, "passed": bool(passed), **evidence})
    print(("PASS " if passed else "FAIL ") + name, flush=True)
    return passed


STATE_FILE = "training_state.pt"
DEVICE = "mps"  # --dry-run-cpu substitutes cpu to validate the harness wiring elsewhere.


def rephix(*argv, expect=0):
    argv = [DEVICE if a == "mps" else a for a in argv]
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_ENABLE_MPS_FALLBACK"}
    completed = subprocess.run(
        [sys.executable, "-m", "rephix", *map(str, argv)],
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != expect:
        raise AssertionError(
            f"rephix {' '.join(map(str, argv))} exited {completed.returncode}:\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout) if expect == 0 and completed.stdout.strip() else completed


def on_mps(report, name):
    if DEVICE != "mps":
        return True
    accelerator = report.get("accelerator", {})
    return check(
        f"{name}: executed on MPS without CPU fallback",
        accelerator.get("device") == "mps"
        and accelerator.get("cpu_fallback_enabled") is False
        and accelerator.get("cpu_fallback_operators") == []
        and accelerator.get("mps_driver_allocated_bytes", 0) > 0,
        accelerator=accelerator,
    )


def environment():
    import torch

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    try:
        info["chip"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        info["macos"] = platform.mac_ver()[0]
    except OSError:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


def unavailable(work):
    trial = work / "trial"
    rephix("offline-trial", "--output", trial, "--steps", "2")
    completed = rephix(
        "fit",
        "--train",
        trial / "splits/train.jsonl",
        "--validation",
        trial / "splits/validation.jsonl",
        "--teacher",
        trial / "fixture" / "model",
        "--output",
        work / "fit",
        "--steps",
        "2",
        "--device",
        "mps",
        expect=1,
    )
    check(
        "unavailable: --device mps fails with an explicit message and no traceback",
        "MPS was requested but is unavailable" in completed.stderr
        and "Traceback" not in completed.stderr,
        stderr=completed.stderr[-400:],
    )
    report = rephix(
        "evaluate",
        "--data",
        trial / "splits/test.jsonl",
        "--model",
        trial / "fixture" / "model",
        "--device",
        "auto",
    )
    check(
        "unavailable: --device auto runs on CPU and reports no accelerator",
        "accelerator" not in report,
    )


def available(work):
    trial = work / "trial"
    rephix("offline-trial", "--output", trial, "--steps", "2")
    splits = trial / "splits"
    common = [
        "--train",
        splits / "train.jsonl",
        "--validation",
        splits / "validation.jsonl",
        "--teacher",
        trial / "fixture" / "model",
        "--steps",
        "40",
        "--eval-every",
        "10",
        "--save-every",
        "5",
        "--seed",
        "3",
        "--learning-rate",
        "0.003",
        "--max-seq-length",
        "64",
    ]
    reference = rephix("fit", *common, "--output", work / "fit-cpu", "--device", "cpu")
    mps = rephix("fit", *common, "--output", work / "fit-mps", "--device", "mps")
    on_mps(mps, "train (fit)")
    cpu_loss = reference["best_validation"]["continuation_cross_entropy"]
    mps_loss = mps["best_validation"]["continuation_cross_entropy"]
    # Float32 kernels differ between backends and the difference compounds over 40
    # optimizer steps; 2% relative agreement of validation loss is the stated bound.
    check(
        "train (fit): validation loss within 2% of the CPU reference",
        abs(cpu_loss - mps_loss) <= 0.02 * abs(cpu_loss),
        cpu=cpu_loss,
        mps=mps_loss,
    )

    # Interrupt an MPS run after a durable checkpoint, then resume to the same target.
    output = work / "fit-mps-resume"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rephix",
            "fit",
            *map(str, common),
            "--output",
            str(output),
            "--device",
            DEVICE,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline and process.poll() is None:
        if (output / STATE_FILE).exists():
            break
        time.sleep(0.2)
    time.sleep(1)
    process.send_signal(signal.SIGINT)
    _, stderr = process.communicate(timeout=120)
    check(
        "cancel: SIGINT stops MPS training with a resumable exit",
        process.returncode in (130, 0) and "Traceback" not in stderr,
        exit=process.returncode,
        stderr=stderr[-300:],
    )
    resumed = rephix("fit", *common, "--output", output, "--device", "mps", "--resume")
    on_mps(resumed, "resume (fit --resume)")
    check(
        "resume: resumed MPS run reaches the same validation loss as uninterrupted MPS",
        abs(resumed["best_validation"]["continuation_cross_entropy"] - mps_loss)
        <= 1e-3 * abs(mps_loss),
        resumed=resumed["best_validation"]["continuation_cross_entropy"],
        uninterrupted=mps_loss,
    )

    best = work / "fit-mps"  # resolved through best_checkpoint.json
    evaluations = {}
    for device in ("cpu", "mps"):
        evaluations[device] = rephix(
            "evaluate",
            "--data",
            splits / "test.jsonl",
            "--model",
            best,
            "--device",
            device,
            "--generate",
            "--max-new-tokens",
            "8",
            "--max-seq-length",
            "64",
        )
    on_mps(evaluations["mps"], "inference (evaluate --generate)")
    ce = [evaluations[d].get("continuation_cross_entropy") for d in ("cpu", "mps")]
    check(
        "inference: test cross-entropy within 1e-3 relative of CPU on the same checkpoint",
        None not in ce and abs(ce[0] - ce[1]) <= 1e-3 * abs(ce[0]),
        cpu=ce[0],
        mps=ce[1],
    )

    # Checkpoint save/load on MPS: layer recovery exports a checkpoint that is then audited.
    recovered = rephix(
        "recover-layers",
        "--model",
        trial / "fixture" / "model",
        "--output",
        work / "recovered-mps",
        "--device",
        "mps",
        "--dtype",
        "float32",
    )
    on_mps(recovered, "checkpoint recovery (recover-layers)")
    audit = rephix(
        "audit-weights",
        trial / "fixture" / "model",
        work / "recovered-mps",
        "--rtol",
        "1e-4",
        "--atol",
        "1e-6",
    )
    check(
        "checkpoint recovery: every stored value within float32 tolerance of the source",
        audit.get("coverage") == 1.0 and audit.get("fraction_within_tolerance") == 1.0,
        audit={k: audit.get(k) for k in ("coverage", "fraction_within_tolerance", "relative_l2")},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-unavailable", action="store_true")
    parser.add_argument(
        "--dry-run-cpu",
        action="store_true",
        help="run the MPS steps on CPU to validate the harness (not evidence)",
    )
    args = parser.parse_args()
    global DEVICE
    if args.dry_run_cpu:
        DEVICE = "cpu"
    work = args.work_dir.resolve()
    work.mkdir(parents=True)
    info = environment()
    try:
        if args.expect_unavailable:
            check("host: MPS unavailable as expected", not info["mps_available"], **info)
            unavailable(work)
        else:
            if args.dry_run_cpu or check("host: usable MPS device", info["mps_available"], **info):
                available(work)
    except Exception as exc:  # noqa: BLE001 - record the failing step
        check("harness completed", False, error=repr(exc)[-3000:])
    failed = [c["check"] for c in CHECKS if not c["passed"]]
    args.output.write_text(
        json.dumps({"environment": info, "checks": CHECKS, "failed": failed}, indent=1) + "\n"
    )
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} MPS checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
