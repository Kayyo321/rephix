"""Recover native Diffusers components with resumable chosen-layer evidence."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from .batch_norm_recovery import BATCH_NORM_TYPES, observe_batch_norm
from .hf import _device
from .io import write_json
from .layer_bundle import bundle_binding, open_bundle, prepare_bundle, source_fingerprint
from .layer_journal import LayerJournal
from .layer_recovery import (
    _JOURNAL_FILENAME,
    _PROBE_PLAN_VERSION,
    _check_disjoint,
    _check_output,
    _evidence_report,
    _plan,
    _qualified_name,
    _validate_probe_options,
    recover_from_layer_calls,
)
from .layer_staging import LayerStager
from .storage_dtypes import (
    apply_buffer_dtypes,
    apply_parameter_dtypes,
    source_buffer_dtypes,
    source_parameter_dtypes,
)
from .weight_audit import _inventory, _unique_json_object
from .weight_norm_recovery import (
    WEIGHT_NORM_PARAMETRIZATIONS,
    observe_weight_norm,
)

_MODEL_CLASSES = frozenset(
    {"AutoencoderKL", "AutoencoderTiny", "UNet2DModel", "UNet2DConditionModel"}
)
_RUNTIME_SCHEMA_VERSION = 1
_EXTRA_ASSETS = ("scheduler_config.json",)
_DTYPE_NAMES = ("float16", "bfloat16", "float32", "float64")


def _deps():
    try:
        import diffusers
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'Install optional dependencies with: pip install -e ".[diffusion]"'
        ) from exc
    return torch, diffusers


def _native_class(diffusers, config):
    if not isinstance(config, dict):
        raise ValueError("Diffusers component config must be a JSON object")
    name = config.get("_class_name")
    if not isinstance(name, str) or name not in _MODEL_CLASSES:
        raise ValueError("unsupported native Diffusers component class")
    if config.get("quantization_config") is not None:
        raise ValueError("diffusion recovery requires a dense nonquantized component")
    return getattr(diffusers, name)


def _source_config(source):
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError("diffusion recovery requires a local component directory with config.json")
    if any(
        (source / name).exists() for name in ("model.safetensors", "model.safetensors.index.json")
    ):
        raise ValueError(
            "Diffusers recovery requires an unambiguous native diffusion checkpoint layout"
        )
    layouts = (
        source / "diffusion_pytorch_model.safetensors",
        source / "diffusion_pytorch_model.safetensors.index.json",
    )
    if sum(path.is_file() for path in layouts) != 1:
        raise ValueError("expected one native diffusion safetensors file or shard index")
    config = json.loads(
        (source / "config.json").read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(config, dict):
        raise ValueError("Diffusers component config must be a JSON object")
    return config


def _load_dtype(torch, source, setting):
    if setting in {"float32", "float64"}:
        return getattr(torch, setting)
    if setting != "auto":
        raise ValueError("diffusion recovery dtype must be auto, float32, or float64")
    kinds = {entry.dtype for entry in _inventory(source).values()}
    floats = kinds & {"F16", "BF16", "F32", "F64"}
    if "F64" in floats:
        return torch.float64
    if floats == {"F16"}:
        return torch.float16
    if floats == {"BF16"}:
        return torch.bfloat16
    return torch.float32


def _parameter_aliases(source, model):
    """Retain original names for native deprecated attention projections."""
    inventory = _inventory(source)
    aliases = {}
    replacements = (
        (".to_q.", ".query."),
        (".to_k.", ".key."),
        (".to_v.", ".value."),
        (".to_out.0.", ".proj_attn."),
    )
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name in inventory:
            continue
        matches = [
            name.replace(current, old)
            for current, old in replacements
            if current in name and name.replace(current, old) in inventory
        ]
        if len(matches) != 1:
            raise ValueError(f"{name}: source parameter name has no native archive mapping")
        stored = matches[0]
        if inventory[stored].shape != tuple(parameter.shape):
            raise ValueError(f"{stored}: legacy projection shape requires unsupported conversion")
        aliases[name] = stored
    state = {aliases.get(name, name): value for name, value in model.state_dict().items()}
    if state.keys() != inventory.keys():
        raise ValueError("source archive state does not match the native component layout")
    if any(tuple(state[name].shape) != entry.shape for name, entry in inventory.items()):
        raise ValueError("source archive tensor shapes disagree with the native component layout")
    return aliases


def _mirror_dtypes(torch, candidate, target):
    for name, parameter in candidate.named_parameters():
        parameter.data = parameter.detach().to(target.get_parameter(name).dtype)
    for name, value in candidate.named_buffers():
        if value.is_floating_point():
            value.data = value.detach().to(target.get_buffer(name).dtype)


def _restore_probe_dtypes(torch, model, metadata):
    apply_parameter_dtypes(torch, model, metadata["parameter_dtype_overrides"])
    apply_buffer_dtypes(torch, model, metadata["buffer_dtype_overrides"])


def _copy_extras(source, destination):
    for name in _EXTRA_ASSETS:
        path = source / name
        if path.is_file():
            shutil.copyfile(path, destination / name)


def _publish(candidate, output, report, assets):
    torch, _ = _deps()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rephix-diffusion-", dir=output.parent) as temporary:
        staged = Path(temporary) / "checkpoint"
        apply_parameter_dtypes(torch, candidate, report.get("export_parameter_dtypes", {}))
        apply_buffer_dtypes(torch, candidate, report.get("export_buffer_dtypes", {}))
        state = candidate.state_dict()
        aliases = report.get("serialized_parameter_aliases", {})
        for current, stored in aliases.items():
            if current not in state or stored in state:
                raise ValueError("invalid Diffusers serialized parameter alias")
            state[stored] = state.pop(current)
        # ModelMixin builds its own state dictionary and ignores a state_dict
        # keyword. Supply the renamed archive through its native save path.
        save_options = {}
        if aliases:
            # Native deprecated-key conversion only runs on unsharded files.
            save_options["max_shard_size"] = sum(
                value.numel() * value.element_size() for value in state.values()
            )
        with patch.object(candidate, "state_dict", return_value=state):
            candidate.save_pretrained(staged, safe_serialization=True, **save_options)
        _copy_extras(assets, staged)
        write_json(staged / "layer-recovery.json", report)
        _check_output(output)
        os.rename(staged, output)


def recover_diffusion_model(
    source,
    output,
    *,
    device="cpu",
    dtype="float32",
    batch_size=128,
    stage_layers=False,
    evidence_dir=None,
    restore_source_dtypes=False,
):
    """Reconstruct a local native Diffusers component from its layer responses."""
    source, output = Path(source), Path(output)
    _validate_probe_options(batch_size, "embedding")
    _check_output(output)
    _check_disjoint(source, output)
    if evidence_dir is not None:
        evidence_dir = Path(evidence_dir)
        _check_disjoint(source, evidence_dir)
        _check_disjoint(output, evidence_dir)
    config = _source_config(source)
    torch, diffusers = _deps()
    model_class = _native_class(diffusers, config)
    device = _device(torch, device)
    scaffold_dtype = _load_dtype(torch, source, dtype)
    target, loading = model_class.from_pretrained(
        source,
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=False,
        torch_dtype=scaffold_dtype,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Diffusers source did not fully load: {loading}")
    target.eval()
    aliases = _parameter_aliases(source, target)
    if dtype == "auto":
        apply_parameter_dtypes(
            torch, target, source_parameter_dtypes(torch, source, target, aliases)
        )
        apply_buffer_dtypes(torch, target, source_buffer_dtypes(torch, source, target))
    oracle_dtype = target.dtype
    candidate = model_class.from_config(dict(target.config)).to(dtype=scaffold_dtype).eval()
    _mirror_dtypes(torch, candidate, target)
    _plan(torch, candidate, direct=True)
    stager = LayerStager(torch, device) if stage_layers and device != "cpu" else None
    if not stage_layers:
        target.to(device)
    provenance = {
        "runtime": "diffusers",
        "architecture": type(target).__name__,
        "oracle_device": device,
        "oracle_dtype": str(oracle_dtype),
        "oracle_placement": "staged" if stage_layers else "resident",
        "serialized_parameter_aliases": aliases,
    }
    if restore_source_dtypes:
        provenance["export_parameter_dtypes"] = source_parameter_dtypes(
            torch, source, candidate, aliases
        )
        provenance["export_buffer_dtypes"] = source_buffer_dtypes(torch, source, candidate)
    context = nullcontext()
    export_assets = source
    if evidence_dir is not None:
        metadata = {
            "runtime": "diffusers",
            "runtime_schema_version": _RUNTIME_SCHEMA_VERSION,
            "probe_plan_version": _PROBE_PLAN_VERSION,
            "source_files": source_fingerprint(source),
            "load_dtype": dtype,
            "batch_size": batch_size,
            "parameter_dtype_overrides": {
                name: str(value.dtype)
                for name, value in candidate.named_parameters()
                if value.dtype != scaffold_dtype
            },
            "buffer_dtype_overrides": {
                name: str(value.dtype)
                for name, value in candidate.named_buffers()
                if value.is_floating_point() and value.dtype != scaffold_dtype
            },
            "provenance": provenance,
        }
        if scaffold_dtype != oracle_dtype:
            metadata["scaffold_dtype"] = str(scaffold_dtype)

        def save_scaffold(destination):
            target.save_config(destination)
            _copy_extras(source, destination)

        _, export_assets = prepare_bundle(evidence_dir, metadata, save_scaffold)
        context = LayerJournal(
            evidence_dir / _JOURNAL_FILENAME, torch, binding=bundle_binding(evidence_dir)
        )

    def observe(path, inputs):
        path, _, operation = path.partition(":")
        module = target.get_submodule(path)
        if stager is not None:
            (module,) = stager.get(module)
        name = _qualified_name(module)
        if name in BATCH_NORM_TYPES:
            return observe_batch_norm(torch, module, inputs.to(device), operation)
        if name == WEIGHT_NORM_PARAMETRIZATIONS:
            return observe_weight_norm(torch, module, inputs.to(device), operation)
        return module(inputs.to(device))

    try:
        with context as journal:
            report = recover_from_layer_calls(
                candidate, observe, batch_size=batch_size, _journal=journal
            )
            if journal is not None:
                journal.finish()
                report.update(_evidence_report(journal, evidence_dir))
    finally:
        if stager is not None:
            stager.clear()
    report.update(
        provenance,
        source=str(source),
        checkpoint=str(output),
        oracle_staged_groups=stager.moves if stager is not None else 0,
        oracle_peak_staged_state_bytes=stager.peak_state_bytes if stager is not None else 0,
        oracle_host_layer_calls=(
            report.get("evidence_observations_collected", report["layer_calls"])
            if device == "cpu"
            else 0
        ),
    )
    target = None
    _publish(candidate, output, report, export_assets)
    return report


def replay_diffusion_evidence(evidence_dir, output):
    """Rebuild a native Diffusers checkpoint using only a sealed evidence bundle."""
    evidence_dir, output = Path(evidence_dir), Path(output)
    _check_output(output)
    _check_disjoint(evidence_dir, output)
    metadata, assets = open_bundle(evidence_dir)
    if (
        metadata.get("runtime") != "diffusers"
        or metadata.get("runtime_schema_version") != _RUNTIME_SCHEMA_VERSION
        or metadata.get("probe_plan_version") != _PROBE_PLAN_VERSION
    ):
        raise ValueError("unsupported Diffusers evidence runtime or probe plan")
    torch, diffusers = _deps()
    config = json.loads(
        (assets / "config.json").read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
    )
    model_class = _native_class(diffusers, config)
    provenance = metadata["provenance"]
    dtypes = {str(getattr(torch, name)): getattr(torch, name) for name in _DTYPE_NAMES}
    scaffold_dtype = metadata.get("scaffold_dtype", provenance["oracle_dtype"])
    candidate = model_class.from_config(config).to(dtype=dtypes[scaffold_dtype]).eval()
    if type(candidate).__name__ != provenance["architecture"]:
        raise ValueError("Diffusers evidence architecture disagrees with the scaffold")
    _restore_probe_dtypes(torch, candidate, metadata)
    with LayerJournal(
        evidence_dir / _JOURNAL_FILENAME, torch, binding=bundle_binding(evidence_dir), readonly=True
    ) as journal:
        report = recover_from_layer_calls(
            candidate, None, batch_size=metadata["batch_size"], _journal=journal
        )
        journal.finish()
        report.update(_evidence_report(journal, evidence_dir))
    report.update(
        provenance,
        source=None,
        checkpoint=str(output),
        oracle_staged_groups=0,
        oracle_peak_staged_state_bytes=0,
        oracle_host_layer_calls=0,
    )
    _publish(candidate, output, report, assets)
    return report


def _command(args):
    return recover_diffusion_model(
        args.model,
        args.output,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        stage_layers=args.stage_layers,
        evidence_dir=args.evidence_dir,
        restore_source_dtypes=args.restore_source_dtypes,
    )


def add_subcommands(commands):
    recover = commands.add_parser(
        "recover-diffusion", help="recover a native Diffusers component from chosen layer calls"
    )
    recover.add_argument(
        "--model", required=True, help="local native Diffusers component directory"
    )
    recover.add_argument(
        "--output", required=True, help="new native Diffusers checkpoint directory"
    )
    recover.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    recover.add_argument("--dtype", choices=("float32", "float64", "auto"), default="float32")
    recover.add_argument("--batch-size", type=int, default=128)
    recover.add_argument("--stage-layers", action="store_true")
    recover.add_argument("--restore-source-dtypes", action="store_true")
    recover.add_argument(
        "--evidence-dir", help="durable layer observation bundle to create or resume"
    )
    recover.set_defaults(func=_command)
    replay = commands.add_parser(
        "replay-diffusion", help="rebuild a native Diffusers component from sealed evidence"
    )
    replay.add_argument("--evidence-dir", required=True)
    replay.add_argument("--output", required=True)
    replay.set_defaults(func=lambda args: replay_diffusion_evidence(args.evidence_dir, args.output))
