"""Add only the chosen affine inputs needed to complete a passive trace design."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .hf import _deps, _device
from .io import file_sha256, write_json
from .trace_recovery import (
    _AFFINE,
    _array,
    _destination,
    _layout,
    _publish,
    _trace_description,
    _widths,
)


def _evidence_hashes(directory, *, embedding=False):
    names = ("seen.npy", "y.npy") if embedding else ("x.npy", "y.npy")
    return {name: file_sha256(directory / name) for name in names}


def collect_trace_calibration(traces, output, call_layer, *, max_probes=128, max_embedding_rows=0):
    """Probe missing affine directions and, if budgeted, missing embedding IDs.

    The callback takes a module path and either an activation matrix or an
    integer ID vector. It returns a NumPy output matrix in the trace dtype.
    The entire request is planned before any oracle calls.
    """
    from .trace_solver import plan_affine_calibration

    if any(type(limit) is not int or limit < 0 for limit in (max_probes, max_embedding_rows)):
        raise ValueError("probe budgets must be nonnegative integers")
    traces, output = Path(traces), _destination(output)
    description, dtype = _trace_description(traces)
    selected = []
    affine_rows = embedding_rows = full_basis_rows = affine_operators = 0
    for index, operator in enumerate(description["layout"]):
        kind = operator["kind"]
        if kind not in _AFFINE | {"embedding"}:
            continue
        name = operator["path"]
        width, out_width = _widths(operator)
        directory = traces / "train" / f"{index:04d}"
        if kind == "embedding":
            seen = _array(directory / "seen.npy", np.dtype("bool"))
            if seen.shape != (operator["weight_shape"][0],):
                raise ValueError(f"{name}: embedding coverage shape mismatch")
            inputs = np.flatnonzero(~seen).astype(np.int64)
            if embedding_rows + len(inputs) > max_embedding_rows:
                raise ValueError(
                    f"{name}: {len(inputs)} missing IDs exceed remaining embedding budget"
                )
            plan = {"probe_rows": len(inputs)}
            embedding_rows += len(inputs)
        else:
            affine_operators += 1
            full_basis_rows += width + operator["has_bias"]
            x = _array(directory / "x.npy", dtype)
            if x.ndim != 2 or x.shape[1] != width:
                raise ValueError(f"{name}: passive trace input width disagrees with operator")
            try:
                inputs, plan = plan_affine_calibration(
                    x, has_bias=operator["has_bias"], max_rows=max_probes - affine_rows
                )
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc
            affine_rows += len(inputs)
        if len(inputs):
            metadata = {
                "operator_index": index,
                "path": name,
                "kind": kind,
                "passive_sha256": _evidence_hashes(directory, embedding=kind == "embedding"),
                **plan,
            }
            selected.append((inputs, out_width, metadata))

    with tempfile.TemporaryDirectory(prefix=".rephix-calibration-", dir=output.parent) as tmp:
        staged = Path(tmp) / "calibration"
        staged.mkdir()
        for inputs, out_width, metadata in selected:
            directory = staged / f"{metadata['operator_index']:04d}"
            directory.mkdir()
            np.save(directory / "x.npy", inputs, allow_pickle=False)
            outputs = call_layer(metadata["path"], inputs)
            if (
                not isinstance(outputs, np.ndarray)
                or outputs.shape != (len(inputs), out_width)
                or outputs.dtype != dtype
                or not np.isfinite(outputs).all()
            ):
                raise ValueError(f"{metadata['path']}: invalid calibration oracle output")
            np.save(directory / "y.npy", outputs, allow_pickle=False)
            metadata["chosen_sha256"] = _evidence_hashes(directory)
        report = {
            "schema_version": 1,
            "observation_mode": "rank_and_coverage_completing_layer_calls",
            "trace_manifest_sha256": file_sha256(traces / "manifest.json"),
            "dtype": dtype.name,
            "max_probes": max_probes,
            "max_embedding_rows": max_embedding_rows,
            "chosen_rows": affine_rows + embedding_rows,
            "affine_rows": affine_rows,
            "embedding_rows": embedding_rows,
            "affine_operators": affine_operators,
            "full_affine_basis_rows": full_basis_rows,
            "operators": [metadata for _, _, metadata in selected],
        }
        write_json(staged / "manifest.json", report)
        _publish(staged, output)
    return report


def load_trace_calibration(calibration, traces, layout, dtype):
    """Validate additional observations against the passive arrays they extend."""
    calibration = Path(calibration)
    manifest = json.loads((calibration / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("observation_mode") != "rank_and_coverage_completing_layer_calls"
        or manifest.get("dtype") != dtype.name
        or manifest.get("trace_manifest_sha256") != file_sha256(traces / "manifest.json")
        or not isinstance(manifest.get("operators"), list)
    ):
        raise ValueError("calibration manifest does not match the passive traces")
    paired = {}
    affine_rows = embedding_rows = 0
    for entry in manifest["operators"]:
        if not isinstance(entry, dict):
            raise ValueError("invalid calibration operator entry")
        index = entry.get("operator_index")
        if type(index) is not int or not 0 <= index < len(layout) or index in paired:
            raise ValueError("invalid or duplicate calibration operator index")
        operator = layout[index]
        kind = operator["kind"]
        embedding = kind == "embedding"
        if (
            kind not in _AFFINE | {"embedding"}
            or entry.get("kind") != kind
            or entry.get("path") != operator["path"]
        ):
            raise ValueError("calibration operator does not match the trace")
        folder = f"{index:04d}"
        directory = calibration / folder
        passive = traces / "train" / folder
        if entry.get("passive_sha256") != _evidence_hashes(
            passive, embedding=embedding
        ) or entry.get("chosen_sha256") != _evidence_hashes(directory):
            raise ValueError(f"{operator['path']}: calibration evidence hash mismatch")
        x = _array(directory / "x.npy", np.dtype("int64") if embedding else dtype)
        y = _array(directory / "y.npy", dtype)
        width, out_width = _widths(operator)
        count = entry.get("probe_rows")
        if (
            type(count) is not int
            or count < 1
            or x.shape != ((count,) if embedding else (count, width))
            or y.shape != (count, out_width)
        ):
            raise ValueError(f"{operator['path']}: calibration shape mismatch")
        if embedding:
            seen = _array(passive / "seen.npy", np.dtype("bool"))
            if not np.array_equal(x, np.flatnonzero(~seen)):
                raise ValueError(
                    f"{operator['path']}: calibration IDs must cover exactly missing rows"
                )
            embedding_rows += count
        else:
            affine_rows += count
        paired[index] = (x, y)
    totals = {
        "chosen_rows": affine_rows + embedding_rows,
        "affine_rows": affine_rows,
        "embedding_rows": embedding_rows,
    }
    if any(manifest.get(name) != count for name, count in totals.items()):
        raise ValueError("calibration row totals do not match their arrays")
    return paired, {**totals, "calibrated_operators": len(paired)}


def calibrate_local_trace(args):
    traces = Path(args.traces)
    source = Path(args.model)
    description, dtype = _trace_description(traces)
    if file_sha256(source / "config.json") != description.get("config_sha256"):
        raise ValueError("local model configuration does not match the passive traces")
    torch, AutoConfig, AutoModel, _ = _deps()
    device = _device(torch, args.device)
    opts = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(source, **opts)
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("calibration requires a dense, nonquantized model")
    model, loading = AutoModel.from_pretrained(
        source,
        config=config,
        use_safetensors=True,
        torch_dtype=getattr(torch, dtype.name),
        output_loading_info=True,
        **opts,
    )
    if any(loading[key] for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"calibration model did not fully load: {loading}")
    if _layout(torch, model) != description["layout"]:
        raise ValueError("calibration model layout does not match the passive traces")
    model.eval().to(device)

    def observe(path, inputs):
        with torch.inference_mode():
            activation = torch.from_numpy(inputs).to(device)
            return model.get_submodule(path)(activation).cpu().numpy()

    return collect_trace_calibration(
        traces,
        args.output,
        observe,
        max_probes=args.max_probes,
        max_embedding_rows=args.max_embedding_rows,
    )


def add_subcommands(commands):
    calibrate = commands.add_parser(
        "calibrate-layer-traces", help="probe missing affine directions and embedding IDs"
    )
    calibrate.add_argument("--model", required=True, help="local oracle checkpoint")
    calibrate.add_argument("--traces", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--max-probes", type=int, default=128, help="total chosen affine rows")
    calibrate.add_argument("--max-embedding-rows", type=int, default=0)
    calibrate.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    calibrate.set_defaults(func=calibrate_local_trace)
