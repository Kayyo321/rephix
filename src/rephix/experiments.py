"""Reproducible experiment grids and validation-only comparison reports.

The runner does not read test sets or choose a winning configuration.  Each fit
selects its checkpoint on the supplied validation evidence; final test evaluation
is a separate action once settings are fixed.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import io
import itertools
import json
import math
import re
import time
from argparse import Namespace
from pathlib import Path

from .evidence import assert_disjoint, read_observations
from .io import atomic_write_text, write_json

_DEFAULTS = {
    "seed": 42,
    "steps": 100,
    "learning_rate": 5e-5,
    "topk_weight": 1.0,
    "max_seq_length": 256,
    "init": "random",
    "trainable": "all",
    "student_config": None,
    "student_revision": None,
    "gradient_accumulation_steps": 1,
    "eval_every": 10,
    "save_every": 10,
}
_PARAMETERS = set(_DEFAULTS) | {"selection_metric"}
_TOP_LEVEL = {
    "schema_version",
    "teacher",
    "train",
    "validation",
    "revision",
    "device",
    "defaults",
    "grid",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_object)


def _atomic_json(path, value):
    write_json(path, value)


def _hash_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unknown_keys(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} keys: {', '.join(sorted(unknown))}")


def _path(value, base, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path string")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _model_reference(value, base, label):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a model ID or local path")
    candidate = Path(value).expanduser()
    relative = base / candidate
    # Existing paths and explicitly relative paths are local.  Otherwise an
    # owner/model string remains a Hugging Face model ID.
    if candidate.is_absolute() or value.startswith((".", "~")) or relative.exists():
        return _path(value, base, label)
    return value


def _validate_parameters(parameters):
    for key in (
        "steps",
        "max_seq_length",
        "gradient_accumulation_steps",
        "eval_every",
        "save_every",
    ):
        if type(parameters[key]) is not int or parameters[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(parameters["seed"]) is not int or parameters["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in ("learning_rate", "topk_weight"):
        value = parameters[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be a finite number")
    if parameters["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= parameters["topk_weight"] <= 1:
        raise ValueError("topk_weight must be between zero and one")
    if parameters.get("selection_metric", "reconstruction_objective") not in (
        "reconstruction_objective",
        "continuation_cross_entropy",
    ):
        raise ValueError("Unsupported checkpoint selection metric")
    if not isinstance(parameters["init"], str) or parameters["init"] not in {
        "random",
        "pretrained",
    }:
        raise ValueError("init must be random or pretrained")
    if not isinstance(parameters["trainable"], str) or parameters["trainable"] not in {
        "all",
        "head",
    }:
        raise ValueError("trainable must be all or head")
    if parameters["student_config"] is not None and parameters["init"] != "random":
        raise ValueError("student_config requires random initialization")
    if parameters["student_revision"] is not None and not isinstance(
        parameters["student_revision"], str
    ):
        raise ValueError("student_revision must be a commit string or null")
    if parameters["student_config"] is None and parameters["student_revision"] is not None:
        raise ValueError("student_revision requires student_config")


def load_sweep_config(path):
    """Resolve a strict schema-v1 config and expand its deterministic grid."""
    path = Path(path).resolve()
    config = _read_json(path)
    _unknown_keys(config, _TOP_LEVEL, "sweep config")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for required in ("teacher", "train", "validation"):
        if required not in config:
            raise ValueError(f"Missing sweep key: {required}")
    defaults = config.get("defaults", {})
    grid = config.get("grid", {})
    _unknown_keys(defaults, _PARAMETERS, "defaults")
    _unknown_keys(grid, _PARAMETERS, "grid")
    if not isinstance(config.get("device", "auto"), str) or config.get("device", "auto") not in {
        "auto",
        "cpu",
        "cuda",
        "mps",
    }:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if config.get("revision") is not None and not isinstance(config["revision"], str):
        raise ValueError("revision must be a string or null")
    combinations = 1
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid.{key} must be a nonempty list")
        combinations *= len(values)
    if combinations > 1024:
        raise ValueError("Sweep exceeds 1024 runs; split it into bounded experiments")
    shared = {
        "teacher": _model_reference(config["teacher"], path.parent, "teacher"),
        "train": _path(config["train"], path.parent, "train"),
        "validation": _path(config["validation"], path.parent, "validation"),
        "revision": config.get("revision"),
        "device": config.get("device", "auto"),
    }
    if shared["teacher"] is None:
        raise ValueError("teacher must be a model ID or local path")
    parameters = []
    seen = set()
    keys = sorted(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        point = {**_DEFAULTS, **defaults, **dict(zip(keys, values))}
        point["student_config"] = _model_reference(
            point["student_config"], path.parent, "student_config"
        )
        _validate_parameters(point)
        identity = _hash_json(point)
        if identity in seen:
            raise ValueError("Grid contains duplicate configurations")
        seen.add(identity)
        parameters.append(point)
    return {"schema_version": 1, **shared, "configurations": parameters}


def evidence_summary(records):
    """Count recorded evidence, without estimating tokens from character counts."""
    token_records = [record for record in records if record.completion_token_ids is not None]
    known_tokens = sum(len(record.completion_token_ids) for record in token_records)
    top_tokens = sum(
        len(record.top_logprobs) for record in records if record.top_logprobs is not None
    )
    return {
        "examples": len(records),
        "observed_completion_tokens": known_tokens if len(token_records) == len(records) else None,
        "known_completion_tokens": known_tokens,
        "examples_without_token_ids": len(records) - len(token_records),
        "top_logprob_tokens": top_tokens,
        "top_logprob_token_coverage": top_tokens / known_tokens if known_tokens else None,
        "output_characters": sum(len(record.output) for record in records),
    }


def _source_identity(reference, revision=None):
    """Fingerprint local model inputs; remote references require a pinned revision."""
    if reference is None:
        return None
    root = Path(reference)
    if not root.exists():
        if root.is_absolute() or reference.startswith((".", "~")):
            raise ValueError(f"Local model source does not exist: {reference}")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise ValueError(
                f"Remote model source requires an immutable 40-hex revision: {reference}"
            )
        return {"reference": reference, "kind": "remote_reference", "revision": revision.lower()}
    paths = (
        [root]
        if root.is_file()
        else sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        )
    )
    files = {
        str(path.relative_to(root)) if root.is_dir() else root.name: _hash_file(path)
        for path in paths
    }
    return {"reference": str(root.resolve()), "kind": "local", "files": files}


def sweep(args):
    """Run or continue a bounded grid, preserving input and artifact identities."""
    from . import hf

    specification = load_sweep_config(args.config)
    train = read_observations(specification["train"])
    validation = read_observations(specification["validation"])
    assert_disjoint(train, validation)
    identity = {
        "specification": specification,
        "train_sha256": _hash_file(specification["train"]),
        "validation_sha256": _hash_file(specification["validation"]),
        "model_sources": [
            _source_identity(reference, revision)
            for reference, revision in sorted(
                {
                    (specification["teacher"], specification["revision"]),
                    *(
                        (point["student_config"], point["student_revision"])
                        for point in specification["configurations"]
                        if point["student_config"] is not None
                    ),
                },
                key=lambda pair: (pair[0], pair[1] or ""),
            )
        ],
    }
    fingerprint = _hash_json(identity)
    output = Path(args.output).resolve()
    manifest_path = output / "sweep.json"
    resume = getattr(args, "resume", False)
    expected_runs = [
        {
            "id": "run-" + _hash_json({"sweep": fingerprint, "parameters": point})[:16],
            "parameters": point,
        }
        for point in specification["configurations"]
    ]
    if manifest_path.exists():
        if not resume:
            raise ValueError("Sweep output exists; use --resume or a fresh output directory")
        manifest = _read_json(manifest_path)
        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("runs"), list)
            or any(not isinstance(run, dict) for run in manifest.get("runs", []))
        ):
            raise ValueError("Sweep manifest must contain a runs list of objects")
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError(
                "Sweep configuration or evidence changed; start a fresh output directory"
            )
        if (
            any(manifest.get(key) != value for key, value in identity.items())
            or [
                {"id": run.get("id"), "parameters": run.get("parameters")}
                for run in manifest.get("runs", [])
                if isinstance(run, dict)
            ]
            != expected_runs
        ):
            raise ValueError("Sweep manifest configuration is inconsistent with its fingerprint")
        for run in manifest["runs"]:
            elapsed = run.get("elapsed_seconds")
            status = run.get("status")
            if (
                not isinstance(status, str)
                or status
                not in {
                    "pending",
                    "running",
                    "interrupted",
                    "failed",
                    "completed",
                }
                or (
                    isinstance(elapsed, bool)
                    or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed)
                    or elapsed < 0
                )
            ):
                raise ValueError("Sweep manifest contains invalid run state or elapsed time")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Sweep output is nonempty without its manifest")
        manifest = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            **identity,
            "train_evidence": evidence_summary(train),
            "validation_evidence": evidence_summary(validation),
            "runs": [],
            "interpretation": "Validation comparisons only; no test evidence is read or selected on.",
        }
        for expected in expected_runs:
            manifest["runs"].append(
                {
                    **expected,
                    "status": "pending",
                    "elapsed_seconds": 0.0,
                }
            )
        _atomic_json(manifest_path, manifest)
    shared = {
        key: specification[key] for key in ("teacher", "train", "validation", "device", "revision")
    }
    for run in manifest["runs"]:
        destination = output / run["id"]
        metrics_path = destination / "metrics.json"
        if run["status"] == "completed":
            if not metrics_path.is_file() or _hash_file(metrics_path) != run.get("metrics_sha256"):
                raise ValueError(f"Completed run metrics changed or are missing: {run['id']}")
            continue
        resumable = (destination / "training_state.pt").is_file()
        if destination.exists() and any(destination.iterdir()) and not resumable:
            raise ValueError(
                f"Incomplete run lacks training_state.pt: {run['id']}; preserve it and choose a fresh sweep"
            )
        run["status"] = "running"
        run.pop("error", None)
        _atomic_json(manifest_path, manifest)
        started = time.monotonic()
        try:
            metrics = hf.fit(
                Namespace(**shared, **run["parameters"], output=str(destination), resume=resumable)
            )
            if not metrics_path.exists():
                _atomic_json(metrics_path, metrics)
            run["metrics_sha256"] = _hash_file(metrics_path)
            run["status"] = "completed"
        except BaseException as exc:
            run["status"] = (
                "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            )
            run["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            run["elapsed_seconds"] += time.monotonic() - started
            _atomic_json(manifest_path, manifest)
    # Tables are useful even without plotting dependencies; plots are opt-in.
    comparison = create_report([str(output)], output / "report", plot=False)
    return {
        "output": str(output),
        "completed_runs": len(manifest["runs"]),
        "fingerprint": fingerprint,
        "report": comparison,
    }


def _metric(value, label):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    return value


def _discover_runs(inputs):
    found = {}
    for pattern in inputs:
        matches = glob.glob(str(pattern), recursive=True)
        if not matches:
            raise ValueError(f"Run path matched nothing: {pattern}")
        for match in matches:
            path = Path(match).resolve()
            paths = [path] if path.is_file() else sorted(path.rglob("metrics.json"))
            for metrics in paths:
                if metrics.name != "metrics.json":
                    raise ValueError(f"Expected metrics.json or a directory: {metrics}")
                payload = _read_json(metrics)
                if not isinstance(payload, dict):
                    raise ValueError(f"Training metrics must be a JSON object: {metrics}")
                if "best_validation" not in payload or "baseline_validation" not in payload:
                    continue
                if not isinstance(payload["best_validation"], dict) or not isinstance(
                    payload["baseline_validation"], dict
                ):
                    raise ValueError(f"Validation metrics must be JSON objects: {metrics}")
                if not isinstance(payload.get("history", []), list) or any(
                    not isinstance(point, dict)
                    or not isinstance(point.get("validation"), dict)
                    or type(point.get("step")) is not int
                    for point in payload.get("history", [])
                ):
                    raise ValueError(
                        f"Training history must contain step/validation objects: {metrics}"
                    )
                found[str(metrics)] = (metrics, payload)
    if not found:
        raise ValueError("No completed training metrics found")
    return [found[key] for key in sorted(found)]


def _run_context(metrics_path):
    manifest_path = metrics_path.parent.parent / "sweep.json"
    if not manifest_path.is_file():
        return {}, None
    manifest = _read_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("runs"), list)
        or any(not isinstance(run, dict) for run in manifest.get("runs", []))
    ):
        raise ValueError(f"Invalid sweep manifest: {manifest_path}")
    matches = [run for run in manifest.get("runs", []) if run.get("id") == metrics_path.parent.name]
    if not matches:
        return {}, None
    run = matches[0]
    if not isinstance(run.get("parameters"), dict) or not isinstance(
        manifest.get("train_evidence"), dict
    ):
        raise ValueError(f"Invalid sweep run metadata: {manifest_path}")
    if run.get("status") != "completed":
        raise ValueError(
            f"Run is not marked completed in its sweep manifest: {metrics_path.parent.name}"
        )
    if _hash_file(metrics_path) != run.get("metrics_sha256"):
        raise ValueError(f"Run metrics do not match sweep manifest: {metrics_path.parent.name}")
    return {
        **run.get("parameters", {}),
        "elapsed_seconds": run.get("elapsed_seconds"),
        **manifest.get("train_evidence", {}),
    }, manifest.get("fingerprint")


def _comparison_row(path, metrics, context, sweep_fingerprint):
    baseline = metrics["baseline_validation"]
    best = metrics["best_validation"]
    row = {
        "run_id": path.parent.name,
        "run_path": str(path.parent),
        "sweep_fingerprint": sweep_fingerprint,
        "initialization": metrics.get("initialization", context.get("init")),
        "trainable": metrics.get("trainable", context.get("trainable", "all")),
        "seed": metrics.get("seed", context.get("seed")),
        "steps": metrics.get("steps", context.get("steps")),
        "best_step": metrics.get("best_step"),
        "learning_rate": metrics.get("learning_rate", context.get("learning_rate")),
        "topk_weight": metrics.get("topk_weight", context.get("topk_weight")),
        "max_seq_length": metrics.get("max_seq_length", context.get("max_seq_length")),
        "checkpoint_selection_metric": metrics.get("checkpoint_selection_metric"),
        "baseline_validation_cross_entropy": _metric(
            baseline.get("continuation_cross_entropy"), "baseline loss"
        ),
        "best_validation_cross_entropy": _metric(
            best.get("continuation_cross_entropy"), "best loss"
        ),
        "best_validation_token_agreement": _metric(
            best.get("teacher_token_agreement"), "agreement"
        ),
        "best_validation_selection_score": _metric(
            best.get(metrics.get("checkpoint_selection_metric", "")), "selection score"
        ),
        "train_examples": metrics.get("train_examples", context.get("examples")),
        "observed_completion_tokens": context.get("observed_completion_tokens"),
        "top_logprob_tokens": context.get("top_logprob_tokens"),
        "elapsed_seconds": context.get("elapsed_seconds", metrics.get("elapsed_seconds")),
        "device": metrics.get("execution_device"),
    }
    return row


def _format(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _plot_learning_curves(runs, output):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "status": "unavailable",
            "reason": 'Install plotting with pip install -e ".[analysis]"',
        }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for path, metrics in runs:
        history = [
            {"step": 0, "validation": metrics["baseline_validation"]},
            *metrics.get("history", []),
        ]
        steps = [point["step"] for point in history]
        label = f"{path.parent.name} (seed={metrics.get('seed', '?')}, weight={metrics.get('topk_weight', '?')})"
        losses = [
            _metric(point["validation"].get("continuation_cross_entropy"), "curve loss")
            for point in history
        ]
        agreement = [
            _metric(point["validation"].get("teacher_token_agreement"), "curve agreement")
            for point in history
        ]
        axes[0].plot(steps, losses, marker=".", label=label)
        axes[1].plot(steps, agreement, marker=".", label=label)
    axes[0].set_ylabel("Validation continuation cross entropy (nats/token)")
    axes[1].set_ylabel("Validation recorded-token agreement")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.set_xlabel("Optimizer steps")
        axis.grid(alpha=0.25)
    if len(runs) <= 12:
        axes[0].legend(fontsize=6)
    figure.suptitle("Validation comparisons; held-out test evidence is not used")
    destination = output / "learning-curves.svg"
    figure.savefig(destination, metadata={"Date": None})
    plt.close(figure)
    return {"status": "created", "path": str(destination)}


def create_report(inputs, output, *, plot=False):
    """Write portable comparison artifacts without choosing a winning model."""
    runs = _discover_runs(inputs)
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, metrics in runs:
        context, sweep_fingerprint = _run_context(path)
        rows.append(_comparison_row(path, metrics, context, sweep_fingerprint))
    plot_result = _plot_learning_curves(runs, destination) if plot else {"status": "not_requested"}
    payload = {
        "schema_version": 1,
        "interpretation": "Validation comparisons only; no winner or test-set result is inferred.",
        "runs": rows,
        "plot": plot_result,
    }
    _atomic_json(destination / "comparison.json", payload)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(destination / "comparison.csv", stream.getvalue())
    columns = [
        "run_id",
        "initialization",
        "seed",
        "topk_weight",
        "steps",
        "best_validation_cross_entropy",
        "best_validation_token_agreement",
        "observed_completion_tokens",
        "elapsed_seconds",
    ]
    lines = [
        "# Experiment comparison",
        "",
        "Validation metrics describe the checkpoint selected by each run. No test evidence was read and no winning configuration was selected.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(_format(row[key]) for key in columns) + " |" for row in rows)
    lines.extend(
        [
            "",
            "Token counts are recorded evidence counts, not optimization exposures. Unknown counts are shown as —.",
            "",
        ]
    )
    atomic_write_text(destination / "comparison.md", "\n".join(lines))
    return {"output": str(destination), "runs": len(rows), "plot": plot_result}


def report(args):
    return create_report(args.runs, args.output, plot=getattr(args, "plot", False))


def add_subcommands(subparsers):
    parser = subparsers.add_parser(
        "sweep", help="Run or resume a bounded, validation-only experiment grid"
    )
    parser.add_argument(
        "--config", required=True, help="Schema-v1 sweep JSON; paths are relative to this file"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(func=sweep)

    parser = subparsers.add_parser(
        "report", help="Export validation comparisons as JSON, CSV, Markdown and optional SVG"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run/sweep directories, metrics.json files, or quoted glob patterns",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot", action="store_true")
    parser.set_defaults(func=report)
