"""Immutable scaffold assets and metadata for layer probe evidence."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from .io import file_sha256, write_json
from .weight_audit import _shards, _unique_json_object

_ASSET_SUFFIXES = {
    ".json",
    ".txt",
    ".model",
    ".vocab",
    ".merges",
    ".tiktoken",
    ".jinja",
    ".codes",
    ".spm",
    ".tokenizer",
}
_SCHEMA_VERSION = 1


def source_fingerprint(source):
    """Hash the local safetensors checkpoint and its config and tokenizer files."""
    source = Path(source)
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError(f"Expected a local Hugging Face checkpoint directory: {source}")
    files = set(_shards(source))
    for file in source.iterdir():
        if file.is_file() and (file.suffix in _ASSET_SUFFIXES or file.name == "model.safetensors"):
            files.add(file)
    templates = source / "chat_templates"
    if templates.is_dir():
        files.update(file for file in templates.rglob("*") if file.is_file())
    for file in files:
        if not file.is_file():
            raise ValueError(f"Expected a regular checkpoint file: {file}")
    return {file.relative_to(source).as_posix(): file_sha256(file) for file in sorted(files)}


def _asset_files(assets):
    if assets.is_symlink() or not assets.is_dir():
        raise ValueError(f"Expected a scaffold asset directory: {assets}")
    files = []
    for path in sorted(assets.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Expected regular scaffold assets: {path}")
        if path.is_file():
            files.append(path)
    if assets / "config.json" not in files:
        raise ValueError(f"Missing scaffold config in {assets}")
    return files


def _asset_hashes(assets, files):
    return {file.relative_to(assets).as_posix(): file_sha256(file) for file in files}


def _reject_constant(value):
    raise ValueError(f"invalid JSON constant {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"invalid JSON number {value}")
    return number


def open_bundle(path):
    """Validate an existing bundle and return its metadata and scaffold asset directory."""
    path = Path(path)
    manifest_path = path / "manifest.json"
    if path.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"Expected a layer evidence manifest in {path}")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read layer evidence manifest {manifest_path}: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.keys() != {"schema_version", "metadata", "assets"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != _SCHEMA_VERSION
        or not isinstance(manifest["metadata"], dict)
        or not isinstance(manifest["assets"], dict)
    ):
        raise ValueError(f"Invalid layer evidence manifest: {manifest_path}")
    assets = path / "assets"
    if _asset_hashes(assets, _asset_files(assets)) != manifest["assets"]:
        raise ValueError(f"Layer evidence scaffold assets changed: {assets}")
    return manifest["metadata"], assets


def bundle_binding(path):
    """Return the digest of the validated bundle's exact manifest bytes."""
    return file_sha256(Path(path) / "manifest.json")


def _sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def prepare_bundle(path, metadata, write_assets):
    """Create the bundle atomically or validate it against the requested metadata."""
    path = Path(path)
    if not isinstance(metadata, dict):
        raise ValueError("Layer evidence metadata must be a JSON object")
    expected = json.dumps(metadata, allow_nan=False, sort_keys=True)
    metadata = json.loads(expected)

    if not path.exists() and not path.is_symlink():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as temporary:
            staged = Path(temporary) / "bundle"
            assets = staged / "assets"
            assets.mkdir(parents=True)
            write_assets(assets)
            files = _asset_files(assets)
            for file in files:
                with file.open("rb") as handle:
                    os.fsync(handle.fileno())
            for directory, _, _ in os.walk(assets, topdown=False):
                _sync_directory(directory)
            write_json(
                staged / "manifest.json",
                {
                    "schema_version": _SCHEMA_VERSION,
                    "metadata": metadata,
                    "assets": _asset_hashes(assets, files),
                },
            )
            _sync_directory(staged)
            if not path.exists() and not path.is_symlink():
                try:
                    staged.rename(path)
                except OSError:
                    if not path.exists() and not path.is_symlink():
                        raise
                else:
                    _sync_directory(path.parent)

    actual, assets = open_bundle(path)
    if json.dumps(actual, allow_nan=False, sort_keys=True) != expected:
        raise ValueError(f"Layer evidence metadata does not match the requested collection: {path}")
    return actual, assets
