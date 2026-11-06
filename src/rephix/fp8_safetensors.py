"""Recover native E4M3FN safetensors checkpoints with blockwise scales."""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np

from .fp8_blocks import decode_fp8_blocks, infer_fp8_blocks
from .io import write_json
from .safetensors_payload import (
    _TILE_FLOATS,
    _decode_dense,
    _encode_dense,
    _local_oracle,
    _observed_tile,
    _Payload,
    _payload_layout,
    _read_json,
    _staged_checkpoint,
)
from .weight_audit import _inventory

_SCALE_SUFFIX = "_scale_inv"
_DENSE_DTYPES = {"BF16", "F16", "F32"}
_SCALE_DTYPES = {"BF16": "bfloat16", "F32": "float32"}
_OBSERVATION_DTYPES = {"float32": np.float32, "float64": np.float64}


@dataclass(frozen=True)
class _Operator:
    name: str
    rows: int
    width: int
    data: _Payload
    block: tuple[int, int]
    dtype: type
    scales: _Payload | None = None

    def packed_rows(self, start, stop):
        height, _ = self.block
        raw_codes = self.data.map(self.data.shape)[start:stop]
        raw_scales = self.scales.map(self.scales.shape)[start // height : _ceil(stop, height)]
        return raw_codes, raw_scales

    def source_rows(self, start, stop):
        if self.scales is None:
            raw = self.data.map((self.rows, self.width))[start:stop]
            return _decode_dense(raw, self.data.dtype).astype(self.dtype, copy=False)
        codes, raw_scales = self.packed_rows(start, stop)
        return _decode_tile(
            codes, _decode_dense(raw_scales, self.scales.dtype), self.block, self.dtype
        )


def _ceil(n, size):
    return (n + size - 1) // size


def _layout(source, dtype=np.float64):
    if sys.byteorder != "little" or not source.is_dir():
        raise ValueError(
            "FP8 recovery requires a local safetensors directory and little-endian host"
        )
    config = _read_json(source / "config.json")
    quant = config.get("quantization_config") if isinstance(config, dict) else None
    if (
        not isinstance(quant, dict)
        or quant.get("quant_method") != "fp8"
        or quant.get("fmt", "e4m3") != "e4m3"
        or quant.get("activation_scheme", "dynamic") != "dynamic"
    ):
        raise ValueError("FP8 recovery requires native dynamic E4M3 block quantization")
    block = quant.get("weight_block_size")
    if (
        not isinstance(block, list)
        or len(block) != 2
        or any(type(size) is not int or size < 1 for size in block)
    ):
        raise ValueError("FP8 weight_block_size must contain two positive integers")
    height, width = block
    prefixes, payloads = _payload_layout(_inventory(source))
    operators = {}
    paired = set()
    for name, data in payloads.items():
        if data.dtype != "F8_E4M3":
            continue
        if not name.endswith(".weight") or len(data.shape) != 2 or not prod(data.shape):
            raise ValueError(f"{name}: expected a nonempty rank-two FP8 weight")
        rows, columns = data.shape
        scale_name = name + _SCALE_SUFFIX
        scales = payloads.get(scale_name)
        if (
            scales is None
            or scales.dtype not in _SCALE_DTYPES
            or scales.shape != (_ceil(rows, height), _ceil(columns, width))
        ):
            raise ValueError(f"{name}: missing or incompatible FP8 scale tensor")
        paired.add(scale_name)
        operators[name] = _Operator(name, rows, columns, data, (height, width), dtype, scales)
    if not operators:
        raise ValueError("checkpoint contains no supported FP8 weight tensors")
    for name, data in payloads.items():
        if name in paired or name in operators:
            continue
        if name.endswith(_SCALE_SUFFIX) or data.dtype not in _DENSE_DTYPES or not prod(data.shape):
            raise ValueError(f"{name}: unsupported or unpaired FP8 checkpoint tensor")
        columns = data.shape[-1] if data.shape else 1
        operators[name] = _Operator(name, prod(data.shape) // columns, columns, data, (1, 1), dtype)
    return prefixes, operators


def _tiles(operator):
    height, _ = operator.block
    step = max(1, _TILE_FLOATS // (height * operator.width)) * height
    for start in range(0, operator.rows, step):
        yield start, min(operator.rows, start + step)


def _decode_tile(codes, scales, block, dtype):
    height, width = block
    rows, columns = codes.shape
    expanded = scales[np.arange(rows)[:, None] // height, np.arange(columns) // width]
    return decode_fp8_blocks(codes[..., None], expanded, dtype=dtype)[..., 0]


def _spans(size, block):
    complete = size // block * block
    if complete:
        yield slice(0, complete), block
    if complete < size:
        yield slice(complete, size), size - complete


def _infer_tile(values, block, scale_dtype, stats):
    rows, columns = values.shape
    height, width = block
    codes = np.empty(values.shape, dtype=np.uint8)
    scales = np.empty((_ceil(rows, height), _ceil(columns, width)), dtype=np.float32)
    for y, h in _spans(rows, height):
        for x, w in _spans(columns, width):
            region = values[y, x]
            n, m = region.shape[0] // h, region.shape[1] // w
            grouped = region.reshape(n, h, m, w).transpose(0, 2, 1, 3).reshape(n, m, h * w)
            factors, packed, counts = infer_fp8_blocks(grouped, scale_dtype=scale_dtype)
            scales[
                y.start // height : _ceil(y.stop, height), x.start // width : _ceil(x.stop, width)
            ] = factors
            codes[y, x] = packed.reshape(n, m, h, w).transpose(0, 2, 1, 3).reshape(region.shape)
            for key, value in counts.items():
                stats[key] += value
    return codes, scales


def _write_operator(source, destination, operator, observe, batch_size, stats):
    target = destination / operator.data.shard.relative_to(source)
    calls = 0
    with target.open("r+b") as weights:
        weights.seek(operator.data.offset)
        if operator.scales is None:
            for start, stop in _tiles(operator):
                values, count = _observed_tile(
                    observe, operator, start, stop, batch_size, dtype=operator.dtype
                )
                float32 = values.astype(np.float32, copy=False)
                if not np.array_equal(float32, values):
                    raise ValueError(
                        f"{operator.name}: dense tensor responses exceed FP32 precision"
                    )
                weights.write(memoryview(_encode_dense(float32, operator.data.dtype)).cast("B"))
                calls += count
            return calls
        scale_target = destination / operator.scales.shard.relative_to(source)
        scale_dtype = operator.scales.dtype
        with scale_target.open("r+b") as factors:
            factors.seek(operator.scales.offset)
            for start, stop in _tiles(operator):
                values, count = _observed_tile(
                    observe, operator, start, stop, batch_size, dtype=operator.dtype
                )
                try:
                    codes, scales = _infer_tile(
                        values, operator.block, _SCALE_DTYPES[scale_dtype], stats
                    )
                    if not np.array_equal(
                        _decode_tile(codes, scales, operator.block, operator.dtype), values
                    ):
                        raise ValueError("packed tile does not reproduce observations")
                    packed_scales = _encode_dense(scales, scale_dtype)
                except ValueError as exc:
                    raise ValueError(f"{operator.name}, rows {start}:{stop}: {exc}") from exc
                weights.write(memoryview(codes).cast("B"))
                factors.write(memoryview(packed_scales).cast("B"))
                calls += count
    return calls


def recover_fp8_safetensors(
    source, output, *, batch_size=256, device="cpu", call_tensor=None, observation_dtype="float64"
):
    """Rebuild an FP8 checkpoint from coordinate-preserving affine responses.

    External callbacks accept (name, row_start, row_stop, unit_inputs) and
    return (len(unit_inputs), row_stop - row_start) in the requested dtype.
    Float64 exactly represents products of finite FP32 scales and E4M3 values.
    """
    source, output = Path(source), Path(output)
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("FP8 output must be a new directory outside the source")
    if type(batch_size) is not int or batch_size < 1 or device not in {"cpu", "cuda"}:
        raise ValueError("invalid FP8 batch size or device")
    if observation_dtype not in _OBSERVATION_DTYPES:
        raise ValueError("FP8 observation dtype must be float32 or float64")
    prefixes, operators = _layout(source, _OBSERVATION_DTYPES[observation_dtype])
    torch = None
    if call_tensor is None and device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
    observe = _local_oracle(operators, device, torch) if call_tensor is None else call_tensor
    previous = torch.get_float32_matmul_precision() if torch is not None else None
    if torch is not None and observation_dtype == "float32":
        torch.set_float32_matmul_precision("highest")
    calls, stats = 0, defaultdict(int)
    try:
        with _staged_checkpoint(source, output, prefixes) as staged:
            for operator in operators.values():
                calls += _write_operator(source, staged, operator, observe, batch_size, stats)
    finally:
        if torch is not None and observation_dtype == "float32":
            torch.set_float32_matmul_precision(previous)
    return {
        "schema_version": 1,
        "observation_mode": "chosen_inputs_to_block_scaled_fp8_tensor_operators",
        "source": str(source),
        "checkpoint": str(output),
        "oracle_device": device if call_tensor is None else "external",
        "observation_dtype": observation_dtype,
        "tensor_calls": calls,
        "observed_output_scalars": sum(op.rows * op.width for op in operators.values()),
        "fp8": dict(stats),
    }


def audit_fp8_safetensors(reference, candidate):
    """Compare packed codes, stored scales, and numeric weights separately."""
    _, first = _layout(Path(reference))
    _, second = _layout(Path(candidate))
    if first.keys() != second.keys():
        raise ValueError("FP8 tensor inventories disagree")
    stats = defaultdict(int)
    for name, a in first.items():
        b = second[name]
        if (
            a.data.shape != b.data.shape
            or a.data.dtype != b.data.dtype
            or a.block != b.block
            or (a.scales is not None and a.scales.dtype != b.scales.dtype)
        ):
            raise ValueError(f"{name}: FP8 tensor layouts disagree")
        for start, stop in _tiles(a):
            if a.scales is None:
                values = a.source_rows(start, stop)
                stats["dense_values"] += values.size
                stats["exact_dense_values"] += int(
                    np.count_nonzero(values == b.source_rows(start, stop))
                )
                continue
            codes_a, scales_a = a.packed_rows(start, stop)
            codes_b, scales_b = b.packed_rows(start, stop)
            values_a = _decode_tile(
                codes_a, _decode_dense(scales_a, a.scales.dtype), a.block, np.float64
            )
            values_b = _decode_tile(
                codes_b, _decode_dense(scales_b, b.scales.dtype), b.block, np.float64
            )
            stats["fp8_codes"] += codes_a.size
            stats["exact_fp8_codes"] += int(np.count_nonzero(codes_a == codes_b))
            stats["fp8_scales"] += scales_a.size
            stats["exact_fp8_scales"] += int(np.count_nonzero(scales_a == scales_b))
            stats["exact_fp8_values"] += int(np.count_nonzero(values_a == values_b))
            with np.errstate(over="ignore", under="ignore"):
                rounded_a, rounded_b = values_a.astype(np.float32), values_b.astype(np.float32)
            stats["exact_fp32_values"] += int(np.count_nonzero(rounded_a == rounded_b))
    return {"schema_version": 1, "reference": str(reference), "candidate": str(candidate), **stats}


def _recover_command(args):
    return recover_fp8_safetensors(
        args.model,
        args.output,
        batch_size=args.batch_size,
        device=args.device,
        observation_dtype=args.observation_dtype,
    )


def _audit_command(args):
    if args.output and any(
        Path(args.output).resolve().is_relative_to(Path(path).resolve())
        for path in (args.reference, args.candidate)
    ):
        raise ValueError("FP8 audit report must be outside both checkpoints")
    report = audit_fp8_safetensors(args.reference, args.candidate)
    if args.output:
        write_json(args.output, report)
    return report


def add_subcommands(commands):
    recover = commands.add_parser(
        "recover-fp8-safetensors", help="recover block-scaled E4M3 weights"
    )
    recover.add_argument("--model", required=True)
    recover.add_argument("--output", required=True)
    recover.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    recover.add_argument("--batch-size", type=int, default=256)
    recover.add_argument("--observation-dtype", choices=_OBSERVATION_DTYPES, default="float64")
    recover.set_defaults(func=_recover_command)

    audit = commands.add_parser(
        "audit-fp8-safetensors", help="compare native FP8 weights and fields"
    )
    audit.add_argument("reference")
    audit.add_argument("candidate")
    audit.add_argument("--output")
    audit.set_defaults(func=_audit_command)
