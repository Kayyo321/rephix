"""Recover a decoder from activation traces of ordinary forward passes."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from .endpoint_head import _forward_kwargs
from .hf import _deps, _device, _load_tokenizer
from .io import file_sha256, write_json
from .layer_recovery import _plan

_DTYPES = {"float32": np.float32, "float64": np.float64}
_AFFINE = {"linear", "conv1d"}


def _layout(torch, model):
    plan = _plan(torch, model)
    layout = []
    for path, module, pending, kind in plan:
        bias = getattr(module, "bias", None)
        if set(pending) != ({"weight"} if bias is None else {"weight", "bias"}):
            raise ValueError(f"{path}: partially shared operators are unsupported")
        shape = tuple(module.weight.shape)
        if kind not in _AFFINE | {"embedding"} and len(shape) != 1:
            raise ValueError(f"{path}: only vector normalization is supported")
        layout.append(
            {"path": path, "kind": kind, "weight_shape": list(shape), "has_bias": bias is not None}
        )
    return layout


def _widths(operator):
    shape = operator["weight_shape"]
    if operator["kind"] == "conv1d":
        return shape[0], shape[1]
    if operator["kind"] == "linear":
        return shape[1], shape[0]
    return shape[-1], shape[-1]


def _contexts(path, limit, vocabulary):
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 2
        or not all(values.shape)
        or not np.issubdtype(values.dtype, np.integer)
    ):
        raise ValueError("contexts must be a nonempty two-dimensional integer .npy array")
    if (
        (limit is not None and values.shape[1] > limit)
        or values.min() < 0
        or values.max() >= vocabulary
    ):
        raise ValueError("context length or token ID exceeds model limits")
    return values


class _Capture:
    def __init__(self, directory, operator, rows, dtype):
        directory.mkdir()
        self.directory = directory
        self.operator = operator
        self.rows = rows
        self.count = 0
        self.x = None
        if operator["kind"] == "embedding":
            shape = operator["weight_shape"]
            self.seen = np.zeros(shape[0], dtype=np.bool_)
            self.y = np.lib.format.open_memmap(
                directory / "y.npy", "w+", dtype=dtype, shape=tuple(shape)
            )
        else:
            self.seen = None
            width, outputs = _widths(operator)
            self.x = np.lib.format.open_memmap(
                directory / "x.npy", "w+", dtype=dtype, shape=(rows, width)
            )
            self.y = np.lib.format.open_memmap(
                directory / "y.npy", "w+", dtype=dtype, shape=(rows, outputs)
            )

    @property
    def done(self):
        return self.count == (len(self.seen) if self.seen is not None else self.rows)

    def hook(self, module, args, output):
        if self.done:
            return
        if not args or not hasattr(args[0], "shape") or not hasattr(output, "shape"):
            raise ValueError(f"{self.operator['path']}: trace needs one tensor input and output")
        x = args[0]
        if self.seen is not None:
            if output.shape[:-1] != x.shape or output.shape[-1] != self.y.shape[1]:
                raise ValueError(f"{self.operator['path']}: embedding trace shape mismatch")
            ids, first = np.unique(x.reshape(-1).cpu().numpy(), return_index=True)
            fresh = ~self.seen[ids]
            if fresh.any():
                flat = output.reshape(-1, self.y.shape[1])
                self.y[ids[fresh]] = flat[first[fresh].tolist()].cpu().numpy()
                self.seen[ids[fresh]] = True
                self.count += int(fresh.sum())
        else:
            if x.shape[:-1] != output.shape[:-1] or (x.shape[-1], output.shape[-1]) != (
                self.x.shape[1],
                self.y.shape[1],
            ):
                raise ValueError(f"{self.operator['path']}: input and output trace shapes disagree")
            n = min(self.rows - self.count, x.numel() // self.x.shape[1])
            rows = slice(self.count, self.count + n)
            self.x[rows] = x.reshape(-1, self.x.shape[1])[:n].cpu().numpy()
            self.y[rows] = output.reshape(-1, self.y.shape[1])[:n].cpu().numpy()
            self.count += n

    def finish(self):
        if self.seen is not None:
            np.save(self.directory / "seen.npy", self.seen, allow_pickle=False)
        elif self.count != self.rows and self.operator["kind"] not in _AFFINE:
            raise ValueError(
                f"{self.operator['path']}: expected {self.rows} traced rows, saw {self.count}"
            )
        self.y.flush()
        if self.x is not None:
            self.x.flush()


def _capture_pass(
    torch, model, layout, contexts, directory, *, fit_multiple, val_rows, batch_size, dtype
):
    captures = []
    handles = []
    head_capture = None
    output_head = model.get_output_embeddings()
    last_only = _forward_kwargs(model)
    directory.mkdir()
    try:
        for index, operator in enumerate(layout):
            if val_rows is not None and operator["kind"] == "embedding":
                captures.append(None)
                continue
            width, _ = _widths(operator)
            required = (
                fit_multiple * (width + operator["has_bias"]) if operator["kind"] in _AFFINE else 64
            )
            rows = min(contexts.size, required if val_rows is None else val_rows)
            capture = _Capture(directory / f"{index:04d}", operator, rows, dtype)
            module = model.get_submodule(operator["path"])
            handles.append(module.register_forward_hook(capture.hook))
            if module is output_head:
                head_capture = capture
            captures.append(capture)
        with torch.inference_mode():
            for start in range(0, len(contexts), batch_size):
                ids = torch.from_numpy(
                    np.array(contexts[start : start + batch_size], dtype=np.int64)
                )
                kwargs = last_only if head_capture is None or head_capture.done else {}
                model(input_ids=ids.to(model.device), use_cache=False, **kwargs)
        for capture in captures:
            if capture is not None:
                capture.finish()
    finally:
        for handle in handles:
            handle.remove()
    return [None if capture is None else capture.count for capture in captures]


def _destination(output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _publish(staged, output):
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    os.rename(staged, output)


def capture_layer_traces(
    model_path,
    train,
    validation,
    output,
    *,
    device="cpu",
    dtype="float64",
    fit_multiple=2,
    validation_rows=128,
    batch_size=8,
):
    """Capture primitive inputs and outputs without invoking isolated layers."""
    source = Path(model_path)
    output = _destination(output)
    config_path = source / "config.json"
    if not config_path.is_file():
        raise ValueError("trace capture needs a local checkpoint with config.json")
    if dtype not in _DTYPES:
        raise ValueError("dtype must be float32 or float64")
    if any(
        type(value) is not int or value < 1 for value in (fit_multiple, validation_rows, batch_size)
    ):
        raise ValueError("fit multiple, validation rows and batch size must be positive integers")

    torch, AutoConfig, AutoModel, _ = _deps()
    device = _device(torch, device)
    opts = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(source, **opts)
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("trace capture requires a dense, nonquantized model")
    model, loading = AutoModel.from_pretrained(
        source,
        config=config,
        use_safetensors=True,
        torch_dtype=getattr(torch, dtype),
        output_loading_info=True,
        **opts,
    )
    if any(loading[key] for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"source checkpoint did not fully load: {loading}")
    layout = _layout(torch, model)
    vocabulary = model.get_input_embeddings().num_embeddings
    limit = getattr(config, "max_position_embeddings", None)
    training = _contexts(train, limit, vocabulary)
    heldout = _contexts(validation, limit, vocabulary)
    seen = {tuple(row) for row in training}
    if any(tuple(row) in seen for row in heldout):
        raise ValueError("training and validation share a token context")
    model.eval().to(device)
    common = {"fit_multiple": fit_multiple, "batch_size": batch_size, "dtype": _DTYPES[dtype]}
    with tempfile.TemporaryDirectory(prefix=".rephix-trace-", dir=output.parent) as tmp:
        staged = Path(tmp) / "traces"
        staged.mkdir()
        train_counts = _capture_pass(
            torch, model, layout, training, staged / "train", val_rows=None, **common
        )
        validation_counts = _capture_pass(
            torch, model, layout, heldout, staged / "validation", val_rows=validation_rows, **common
        )
        report = {
            "schema_version": 1,
            "observation_mode": "passive_layer_inputs_and_outputs",
            "model_type": config.model_type,
            "config_sha256": file_sha256(config_path),
            "dtype": dtype,
            "capture_device": device,
            "layout": layout,
            "train_counts": train_counts,
            "validation_counts": validation_counts,
            "train_contexts": len(training),
            "train_input_tokens": training.size,
            "validation_contexts": len(heldout),
            "validation_input_tokens": heldout.size,
            "source_ignored_keys": loading["unexpected_keys"],
        }
        write_json(staged / "manifest.json", report)
        _publish(staged, output)
    return report


def _normalized(torch, module, kind, values):
    from torch.nn import functional as F

    x = torch.tensor(values)
    if kind == "hf_rms_norm":
        normalized = x.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        return (normalized * torch.rsqrt(variance + module.variance_epsilon)).to(x.dtype).numpy()
    if kind == "rms_norm":
        return F.rms_norm(x, tuple(module.weight.shape), eps=module.eps).numpy()
    return F.layer_norm(x, tuple(module.weight.shape), eps=module.eps).numpy()


def _array(path, dtype):
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.dtype != dtype:
        raise ValueError(f"{path}: unexpected array dtype")
    return values


def _trace_arrays(directory, dtype, count):
    x, y = (_array(directory / name, dtype) for name in ("x.npy", "y.npy"))
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not 0 <= count <= len(x):
        raise ValueError(f"{directory}: trace row count or array shape is invalid")
    return x[:count], y[:count]


def _fit_operator(
    torch, module, operator, train, train_count, validation, dtype, gate, calibration
):
    from .trace_solver import fit_trace_affine, fit_trace_normalization

    kind = operator["kind"]
    if kind == "embedding":
        y = _array(train / "y.npy", dtype)
        seen = _array(train / "seen.npy", np.dtype("bool"))
        if y.shape != tuple(module.weight.shape) or seen.shape != y.shape[:1]:
            raise ValueError("embedding trace shape mismatch")
        observed_ids = int(seen.sum())
        if calibration is not None:
            ids, observed = calibration
            y = np.array(y, copy=True)
            y[ids] = observed
            observed_ids += len(ids)
        if observed_ids != len(seen):
            raise ValueError(f"embedding token coverage {observed_ids}/{len(seen)}")
        if not np.isfinite(y).all():
            raise ValueError("embedding trace contains nonfinite values")
        return y, None, {"observed_token_ids": observed_ids, "observed_rows": observed_ids}

    x, y = _trace_arrays(train, dtype, train_count)
    val_x, val_y = validation
    input_width, output_width = _widths(operator)
    if any(array.ndim != 2 or array.shape[1] != input_width for array in (x, val_x)) or any(
        array.ndim != 2 or array.shape[1] != output_width for array in (y, val_y)
    ):
        raise ValueError("trace feature or output width disagrees with model")
    opts = {"has_bias": operator["has_bias"], "max_validation_rmse": gate}
    if kind in _AFFINE:
        if calibration is not None:
            opts.update(calibration_x=calibration[0], calibration_y=calibration[1])
        weight, bias, metrics = fit_trace_affine(x, y, val_x, val_y, **opts)
        return (weight.T if kind == "linear" else weight), bias, metrics
    return fit_trace_normalization(
        _normalized(torch, module, kind, x),
        y,
        _normalized(torch, module, kind, val_x),
        val_y,
        **opts,
    )


def _runtime_validation(torch, module, validation):
    features, observed = validation
    squared = 0.0
    with torch.inference_mode():
        for start in range(0, len(features), 16):
            rows = slice(start, start + 16)
            predicted = module(torch.tensor(features[rows])).double().numpy()
            residual = predicted - observed[rows]
            squared += float(np.vdot(residual, residual))
    return math.sqrt(squared / observed.size)


def _trace_description(traces):
    manifest = json.loads((traces / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or (manifest.get("observation_mode") != "passive_layer_inputs_and_outputs")
    ):
        raise ValueError("unsupported trace manifest")
    precision = manifest.get("dtype")
    if not isinstance(precision, str) or precision not in _DTYPES:
        raise ValueError("unsupported trace precision")
    if not isinstance(manifest.get("layout"), list) or not manifest["layout"]:
        raise ValueError("trace manifest has no operator layout")
    layout = manifest["layout"]
    for key in ("train_counts", "validation_counts"):
        counts = manifest.get(key)
        if not isinstance(counts, list) or len(counts) != len(layout):
            raise ValueError(f"trace manifest has no valid {key}")
        for operator, count in zip(layout, counts):
            if not isinstance(operator, dict) or not isinstance(operator.get("kind"), str):
                raise ValueError("trace manifest contains an invalid operator")
            if key == "validation_counts" and operator["kind"] == "embedding":
                if count is not None:
                    raise ValueError("embedding operators do not have validation trace rows")
            elif type(count) is not int or count < 0:
                raise ValueError(f"trace manifest has an invalid entry in {key}")
    return manifest, np.dtype(_DTYPES[precision])


def recover_layer_traces(
    config_path,
    traces,
    output,
    *,
    max_validation_rmse=1e-5,
    include_tokenizer=True,
    calibration=None,
):
    """Reconstruct a checkpoint from traces, optional chosen rows, and configuration."""
    config_path, traces = Path(config_path), Path(traces)
    output = _destination(output)
    if not math.isfinite(max_validation_rmse) or max_validation_rmse < 0:
        raise ValueError("max validation RMSE must be finite and nonnegative")
    manifest, dtype = _trace_description(traces)
    precision = dtype.name
    if file_sha256(config_path / "config.json") != manifest.get("config_sha256"):
        raise ValueError("trace configuration hash does not match the supplied model configuration")

    torch, AutoConfig, AutoModel, AutoTokenizer = _deps()
    from transformers import GenerationConfig

    opts = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(config_path, **opts)
    model = AutoModel.from_config(
        config, trust_remote_code=False, torch_dtype=getattr(torch, precision)
    ).eval()
    layout = _layout(torch, model)
    if layout != manifest.get("layout") or config.model_type != manifest.get("model_type"):
        raise ValueError("trace operator layout does not match the supplied architecture")
    chosen, chosen_validation = {}, {}
    calibration_summary = None
    if calibration is not None:
        from .trace_calibration import load_trace_calibration

        chosen, chosen_validation, calibration_summary = load_trace_calibration(
            calibration, traces, layout, dtype, manifest
        )
    modules = []
    with torch.no_grad():
        for index, operator in enumerate(layout):
            name = operator["path"]
            module = model.get_submodule(name)
            folder = f"{index:04d}"
            try:
                validation = None
                if operator["kind"] != "embedding":
                    val_count = manifest["validation_counts"][index]
                    if val_count:
                        validation = _trace_arrays(traces / "validation" / folder, dtype, val_count)
                    elif index in chosen_validation:
                        validation = chosen_validation[index]
                    else:
                        raise ValueError(
                            "no held-out trace rows; additional validation evidence is required"
                        )
                weight, bias, metrics = _fit_operator(
                    torch,
                    module,
                    operator,
                    traces / "train" / folder,
                    manifest["train_counts"][index],
                    validation,
                    dtype,
                    max_validation_rmse,
                    chosen.get(index),
                )
                module.weight.copy_(torch.from_numpy(np.array(weight, copy=True)))
                if bias is not None:
                    module.bias.copy_(torch.from_numpy(bias))
                if operator["kind"] != "embedding":
                    runtime_error = _runtime_validation(torch, module, validation)
                    if not math.isfinite(runtime_error) or runtime_error > max_validation_rmse:
                        raise ValueError(
                            f"candidate held-out RMSE {runtime_error:.6g} exceeds "
                            f"{max_validation_rmse:.6g} or is nonfinite"
                        )
                    metrics["validation_runtime_rmse"] = runtime_error
                    if index in chosen_validation:
                        metrics["validation_mode"] = "chosen_holdout"
            except (ValueError, OSError) as exc:
                raise ValueError(f"{name}: {exc}") from exc
            modules.append({**operator, **metrics})
    if (config_path / "generation_config.json").is_file():
        model.generation_config = GenerationConfig.from_pretrained(
            config_path, local_files_only=True
        )
    report = {
        "schema_version": 1,
        "observation_mode": manifest["observation_mode"],
        "model_type": config.model_type,
        "dtype": precision,
        "checkpoint": str(output),
        "max_validation_rmse": max_validation_rmse,
        "unique_parameter_elements_recovered": sum(p.numel() for p in model.parameters()),
        "operators": modules,
    }
    if calibration_summary is not None:
        report["calibration"] = calibration_summary
    with tempfile.TemporaryDirectory(prefix=".rephix-trace-recovery-", dir=output.parent) as tmp:
        staged = Path(tmp) / "checkpoint"
        model.save_pretrained(staged, safe_serialization=True)
        if include_tokenizer:
            tokenizer = _load_tokenizer(AutoTokenizer, config_path, model_config=config, **opts)
            tokenizer.save_pretrained(staged)
        write_json(staged / "trace-recovery.json", report)
        _publish(staged, output)
    return report


def add_subcommands(commands):
    capture = commands.add_parser(
        "capture-layer-traces", help="observe local model layers during ordinary inference"
    )
    capture.add_argument("--model", required=True)
    capture.add_argument("--train", required=True, help="training contexts .npy")
    capture.add_argument("--validation", required=True, help="validation contexts .npy")
    capture.add_argument("--output", required=True)
    capture.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    capture.add_argument("--dtype", choices=tuple(_DTYPES), default="float64")
    capture.add_argument("--fit-multiple", type=int, default=2)
    capture.add_argument("--validation-rows", type=int, default=128)
    capture.add_argument("--batch-size", type=int, default=8)
    capture.set_defaults(
        func=lambda a: capture_layer_traces(
            a.model,
            a.train,
            a.validation,
            a.output,
            device=a.device,
            dtype=a.dtype,
            fit_multiple=a.fit_multiple,
            validation_rows=a.validation_rows,
            batch_size=a.batch_size,
        )
    )
    recover = commands.add_parser(
        "recover-layer-traces", help="export a known architecture from saved activation traces"
    )
    recover.add_argument("--model-config", required=True, help="directory with model configuration")
    recover.add_argument("--traces", required=True)
    recover.add_argument("--output", required=True)
    recover.add_argument("--calibration", help="optional rank-completing affine evidence")
    recover.add_argument("--max-validation-rmse", type=float, default=1e-5)
    recover.add_argument("--without-tokenizer", action="store_true")
    recover.set_defaults(
        func=lambda a: recover_layer_traces(
            a.model_config,
            a.traces,
            a.output,
            max_validation_rmse=a.max_validation_rmse,
            include_tokenizer=not a.without_tokenizer,
            calibration=a.calibration,
        )
    )
