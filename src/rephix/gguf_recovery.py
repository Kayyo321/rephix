"""Rebuild supported GGUF tensors from chosen affine operator responses."""

from __future__ import annotations

import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np

from .gguf_quant import recover_blocks
from .io import write_json

_QUANTIZED = {"Q4_0", "Q5_0", "Q8_0", "Q4_K", "Q6_K"}
_SUPPORTED = _QUANTIZED | {"F32", "F16", "BF16"}
_TILE_FLOATS = 1 << 22


@dataclass(frozen=True)
class TensorLayout:
    name: str
    shape: tuple[int, ...]
    encoding: object
    offset: int
    rows: int
    width: int
    row_bytes: int

    @property
    def size(self):
        return self.rows * self.row_bytes


def read_gguf(path):
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError('Install GGUF support with: pip install -e ".[gguf]"') from exc

    if not Path(path).is_file():
        raise ValueError(f"GGUF source must be a local file: {path}")
    try:
        reader = gguf.GGUFReader(path)
    except (IndexError, KeyError, OverflowError, struct.error, ValueError) as exc:
        raise ValueError(f"invalid GGUF file {path}: {exc}") from exc
    if sys.byteorder != "little" or reader.endianess != gguf.GGUFEndian.LITTLE:
        raise ValueError("GGUF recovery requires a little-endian file and host")
    if int(reader.fields["GGUF.version"].contents()) not in {2, 3}:
        raise ValueError("GGUF recovery supports versions 2 and 3")
    split = reader.fields.get("split.count")
    if split is not None and split.contents() != 1:
        raise ValueError("GGUF recovery requires an unsplit checkpoint")
    if not 0 < reader.data_offset <= len(reader.data):
        raise ValueError("invalid GGUF tensor section offset")

    layouts = []
    names = set()
    for tensor in reader.tensors:
        name, encoding = tensor.name, tensor.tensor_type
        if name in names:
            raise ValueError(f"duplicate GGUF tensor {name!r}")
        names.add(name)
        if encoding.name not in _SUPPORTED:
            raise ValueError(f"{name}: unsupported GGUF encoding {encoding.name}")
        shape = tuple(map(int, tensor.shape))
        width = shape[0] if shape else 1
        rows = prod(shape[1:])
        block, block_bytes = gguf.GGML_QUANT_SIZES[encoding]
        offset = int(tensor.data_offset)
        if width < 1 or rows < 1 or width % block:
            raise ValueError(f"{name}: invalid GGUF tensor dimensions")
        layout = TensorLayout(
            name, shape, encoding, offset, rows, width, width // block * block_bytes
        )
        if (
            offset < reader.data_offset
            or offset % reader.alignment
            or layout.size != int(tensor.n_bytes)
            or offset + layout.size > len(reader.data)
        ):
            raise ValueError(f"{name}: invalid GGUF tensor extent")
        layouts.append(layout)
    end = reader.data_offset
    for layout in sorted(layouts, key=lambda item: item.offset):
        if layout.offset < end:
            raise ValueError(f"{layout.name}: overlapping GGUF tensor extents")
        end = layout.offset + layout.size
    if not layouts:
        raise ValueError("GGUF source contains no tensors")
    return gguf, reader, tuple(layouts)


def raw_rows(reader, layout, start, stop):
    left = layout.offset + start * layout.row_bytes
    right = layout.offset + stop * layout.row_bytes
    return reader.data[left:right].reshape(stop - start, layout.row_bytes)


def _local_oracle(gguf, reader, layouts, device, torch):
    by_name = {layout.name: layout for layout in layouts}
    active = matrix = None

    def observe(name, start, stop, inputs):
        nonlocal active, matrix
        key = (name, start, stop)
        if key != active:
            matrix = None
            layout = by_name[name]
            raw = raw_rows(reader, layout, start, stop).reshape(1, -1)
            decoded = gguf.dequantize(raw, layout.encoding).reshape(stop - start, layout.width)
            matrix = decoded if torch is None else torch.tensor(decoded, device=device)
            active = key
        if torch is None:
            return inputs @ matrix.T
        values = torch.nn.functional.linear(torch.as_tensor(inputs, device=device), matrix)
        return values.cpu().numpy()

    return observe


def _observed_tile(call_tensor, layout, start, stop, batch_size):
    values = np.empty((stop - start, layout.width), dtype=np.float32)
    step = min(batch_size, max(1, _TILE_FLOATS // layout.width))
    calls = 0
    for column in range(0, layout.width, step):
        count = min(step, layout.width - column)
        indices = np.arange(count)
        inputs = np.zeros((count, layout.width), dtype=np.float32)
        inputs[indices, indices + column] = 1
        response = np.asarray(call_tensor(layout.name, start, stop, inputs))
        if response.shape != (count, stop - start) or response.dtype != np.float32:
            raise ValueError(f"{layout.name}: GGUF oracle output shape or dtype mismatch")
        if not np.isfinite(response).all():
            raise ValueError(f"{layout.name}: non-finite GGUF oracle output")
        values[:, column : column + count] = response.T
        calls += 1
    return values, calls


def _write_tensors(gguf, reader, layouts, target, call_tensor, batch_size):
    by_encoding = {}
    calls = observed = 0
    target.write(memoryview(reader.data[: reader.data_offset]))
    target.truncate(len(reader.data))
    for layout in layouts:
        encoding = layout.encoding.name
        stats = by_encoding.setdefault(encoding, {"tensors": 0, "dequantized_values": 0})
        stats["tensors"] += 1
        stats["dequantized_values"] += layout.rows * layout.width
        tile_rows = max(1, _TILE_FLOATS // layout.width)
        target.seek(layout.offset)
        for start in range(0, layout.rows, tile_rows):
            stop = min(start + tile_rows, layout.rows)
            values, count = _observed_tile(call_tensor, layout, start, stop, batch_size)
            calls += count
            observed += values.size
            if encoding in _QUANTIZED:
                block_width, _ = gguf.GGML_QUANT_SIZES[layout.encoding]
                try:
                    packed, counts = recover_blocks(values.reshape(-1, block_width), encoding)
                except ValueError as exc:
                    raise ValueError(f"{layout.name}, output rows {start}:{stop}: {exc}") from exc
                for key, value in counts.items():
                    stats[key] = stats.get(key, 0) + value
            else:
                packed = gguf.quantize(values.reshape(1, -1), layout.encoding)
            if packed.nbytes != (stop - start) * layout.row_bytes:
                raise ValueError(f"{layout.name}: packed GGUF tensor shape mismatch")
            decoded = gguf.dequantize(packed.reshape(1, -1), layout.encoding).reshape(values.shape)
            if not np.array_equal(decoded, values):
                raise ValueError(f"{layout.name}: packed {encoding} tile does not reproduce probes")
            target.write(memoryview(np.ascontiguousarray(packed)).cast("B"))
    return calls, observed, by_encoding


def recover_gguf(source, output, *, batch_size=256, device="cpu", call_tensor=None):
    """Recover a GGUF file using metadata and exact float32 affine responses.

    The callback takes (tensor_name, row_start, row_stop, inputs) and returns
    float32 outputs of shape (batch, row_stop - row_start). The first GGML
    dimension is the input width, and all remaining dimensions form output rows.
    The default local adapter dequantizes the active source tile for that call.
    """
    source, output = Path(source), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("GGUF recovery output already exists")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("GGUF probe batch size must be a positive integer")
    if device not in {"cpu", "cuda"}:
        raise ValueError("GGUF probe device must be cpu or cuda")
    gguf, reader, layouts = read_gguf(source)
    torch = None
    if call_tensor is None and device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA GGUF probes require PyTorch") from exc
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
    if call_tensor is None:
        observe = _local_oracle(gguf, reader, layouts, device, torch)
    else:
        observe = call_tensor

    previous_precision = torch.get_float32_matmul_precision() if torch is not None else None
    if torch is not None:
        torch.set_float32_matmul_precision("highest")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".rephix-gguf-", dir=output.parent) as temp:
            checkpoint = Path(temp) / "checkpoint.gguf"
            with checkpoint.open("w+b") as target:
                calls, observed, by_encoding = _write_tensors(
                    gguf, reader, layouts, target, observe, batch_size
                )
                target.flush()
                os.fsync(target.fileno())
            os.link(checkpoint, output)
    finally:
        if torch is not None:
            torch.set_float32_matmul_precision(previous_precision)
    return {
        "schema_version": 1,
        "observation_mode": "chosen_inputs_to_dequantized_gguf_tensor_operators",
        "source": str(source),
        "checkpoint": str(output),
        "oracle_device": device if call_tensor is None else "external",
        "tensor_calls": calls,
        "observed_output_scalars": observed,
        "copied_structural_prefix_bytes": int(reader.data_offset),
        "by_encoding": by_encoding,
    }


def _recover_command(args):
    return recover_gguf(args.model, args.output, device=args.device, batch_size=args.batch_size)


def _audit_command(args):
    from .gguf_audit import audit_gguf

    if args.output and Path(args.output).resolve() in {
        Path(args.reference).resolve(),
        Path(args.candidate).resolve(),
    }:
        raise ValueError("GGUF audit report must be separate from both checkpoints")
    report = audit_gguf(args.reference, args.candidate)
    if args.output:
        write_json(Path(args.output), report)
    return report


def add_subcommands(commands):
    recover = commands.add_parser("recover-gguf", help="recover supported GGUF tensor encodings")
    recover.add_argument("--model", required=True, help="local GGUF checkpoint")
    recover.add_argument("--output", required=True, help="new GGUF checkpoint")
    recover.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    recover.add_argument("--batch-size", type=int, default=256)
    recover.set_defaults(func=_recover_command)

    audit = commands.add_parser("audit-gguf", help="compare packed and dequantized GGUF weights")
    audit.add_argument("reference")
    audit.add_argument("candidate")
    audit.add_argument("--output", help="optional JSON report path")
    audit.set_defaults(func=_audit_command)
