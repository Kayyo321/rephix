"""Atomic, finite artifact writes shared by experiment workflows."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def atomic_write_text(path: str | Path, text: str, *, newline: str | None = None) -> None:
    """Replace a file only after its complete new contents are safely written.

    The temporary lives beside its destination, so replacement remains atomic
    on the destination filesystem. Failed writes leave existing artifacts intact.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline=newline,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: str | Path, data) -> None:
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    atomic_write_text(path, payload)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_npz(path: str | Path, **arrays) -> None:
    """Write a pickle-free NumPy archive, preserving any prior complete artifact."""
    import numpy as np

    data = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.hasobject for value in data.values()):
        raise ValueError("object arrays are not permitted in experiment archives")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
