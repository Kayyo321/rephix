"""Shared safetensors checkpoint layout, bounded probes, and atomic publication."""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .weight_audit import _unique_json_object

_TILE_FLOATS = 1 << 22
_DTYPES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "U8": "u1", "F8_E4M3": "u1"}
_ASSETS = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


@dataclass(frozen=True)
class _Payload:
    shard: Path
    offset: int
    shape: tuple[int, ...]
    dtype: str

    def map(self, shape):
        return np.memmap(
            self.shard, mode="r", dtype=_DTYPES[self.dtype], offset=self.offset, shape=shape
        )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)


def _payload_layout(entries):
    # Validate every shard with safetensors before reading the payload offsets.
    by_shard = defaultdict(list)
    for name, entry in entries.items():
        by_shard[entry.shard].append(name)
    prefixes, payloads = {}, {}
    for path, names in by_shard.items():
        with path.open("rb") as shard:
            size = shard.read(8)
            header = shard.read(struct.unpack("<Q", size)[0])
        fields = json.loads(header, object_pairs_hook=_unique_json_object)
        prefix = size + header
        prefixes[path] = (prefix, path.stat().st_size)
        for name in names:
            entry = entries[name]
            offset = len(prefix) + fields[name]["data_offsets"][0]
            payloads[name] = _Payload(path, offset, entry.shape, entry.dtype)
    return prefixes, payloads


def _decode_dense(raw, dtype):
    if dtype == "BF16":
        values = (raw.astype(np.uint32) << 16).view(np.float32)
    else:
        values = raw.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("non-finite dense weights in checkpoint")
    return values


def _encode_dense(values, dtype):
    if dtype == "BF16":
        result = (values.view(np.uint32) >> 16).astype("<u2")
    else:
        with np.errstate(over="ignore"):
            result = values.astype(_DTYPES[dtype], copy=False)
    if not np.array_equal(_decode_dense(result, dtype), values):
        raise ValueError(f"float32 observations cannot be represented as {dtype}")
    return result


def _local_oracle(operators, device, torch):
    active = matrix = None

    def observe(name, start, stop, inputs):
        nonlocal active, matrix
        key = (name, start, stop)
        if key != active:
            values = operators[name].source_rows(start, stop)
            matrix = values if torch is None else torch.tensor(values, device=device)
            active = key
        if torch is None:
            # The CPU adapter specializes the affine operation to its unit-vector inputs.
            return matrix[:, np.argmax(inputs, axis=1)].T
        return (
            torch.nn.functional.linear(torch.as_tensor(inputs, device=device), matrix).cpu().numpy()
        )

    return observe


def _observed_tile(call_tensor, operator, start, stop, batch_size, *, dtype=np.float32):
    width, rows = operator.width, stop - start
    step = min(batch_size, max(1, _TILE_FLOATS // width))
    values = np.empty((rows, width), dtype=dtype)
    calls = 0
    for column in range(0, width, step):
        count = min(step, width - column)
        indices = np.arange(count)
        inputs = np.zeros((count, width), dtype=dtype)
        inputs[indices, column + indices] = 1
        response = np.asarray(call_tensor(operator.name, start, stop, inputs))
        if response.shape != (count, rows) or response.dtype != dtype:
            raise ValueError(f"{operator.name}: tensor oracle shape or dtype mismatch")
        if not np.isfinite(response).all():
            raise ValueError(f"{operator.name}: non-finite tensor oracle output")
        values[:, column : column + count] = response.T
        calls += 1
    return values, calls


@contextmanager
def _staged_checkpoint(source, output, prefixes):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rephix-safetensors-", dir=output.parent) as temporary:
        staged = Path(temporary) / "model"
        staged.mkdir()
        for path, (header, size) in prefixes.items():
            dest = staged / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as shard:
                shard.write(header)
                shard.truncate(size)
        yield staged
        for name in _ASSETS:
            asset = source / name
            if asset.is_file():
                shutil.copyfile(asset, staged / name)
        for template in (source / "chat_templates").glob("*.jinja"):
            dest = staged / "chat_templates" / template.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(template, dest)
        if output.exists() or output.is_symlink():
            raise ValueError("safetensors recovery output already exists")
        staged.rename(output)
