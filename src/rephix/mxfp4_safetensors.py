"""Recover Hugging Face and original OpenAI GPT-OSS MXFP4 checkpoints."""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np

from .io import write_json
from .mxfp4 import infer_mxfp4_fields
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

_BLOCK = 32
_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, -0.0, -1, -2, -3, -4, -6, -8, -12], np.float32)


@dataclass(frozen=True)
class _Operator:
    name: str
    rows: int
    width: int
    data: _Payload
    scales: _Payload | None = None

    def source_rows(self, start, stop):
        if self.scales is None:
            return _decode_dense(
                self.data.map((self.rows, self.width))[start:stop], self.data.dtype
            )
        blocks, scales = self.packed_rows(start, stop)
        return decode_mxfp4_safetensors(blocks, scales).reshape(stop - start, self.width)

    def packed_rows(self, start, stop):
        groups = self.width // _BLOCK
        return (
            self.data.map((self.rows, groups, _BLOCK // 2))[start:stop],
            self.scales.map((self.rows, groups))[start:stop],
        )


def decode_mxfp4_safetensors(blocks, scales):
    """Decode native HF alternating nibbles into float32 groups of 32."""
    if (
        blocks.dtype != np.uint8
        or scales.dtype != np.uint8
        or blocks.shape != (*scales.shape, _BLOCK // 2)
        or np.any(scales == 255)
    ):
        raise ValueError("invalid native MXFP4 blocks or reserved scale byte")
    codes = np.empty((*scales.shape, _BLOCK), dtype=np.uint8)
    codes[..., ::2], codes[..., 1::2] = blocks & 15, blocks >> 4
    with np.errstate(over="ignore"):
        values = np.ldexp(_FP4[codes], scales.astype(np.int16)[..., None] - 128)
    if not np.isfinite(values).all():
        raise ValueError("native MXFP4 values exceed the finite float32 range")
    return values


def _checkpoint_inventory(source, layout):
    if layout == "hugging_face" or (source / "model.safetensors.index.json").exists():
        return _inventory(source)
    entries = {}
    for shard in sorted(source.glob("*.safetensors")):
        tensors = _inventory(shard)
        if entries.keys() & tensors.keys():
            raise ValueError("duplicate native MXFP4 tensors across original OpenAI shards")
        entries.update(tensors)
    if not entries:
        raise ValueError("original OpenAI checkpoint contains no safetensors weights")
    return entries


def _layout(source):
    if sys.byteorder != "little" or not source.is_dir():
        raise ValueError(
            "native MXFP4 requires a local checkpoint directory and little-endian host"
        )
    config = _read_json(source / "config.json")
    if not isinstance(config, dict):
        raise ValueError("native MXFP4 requires a GPT-OSS MXFP4 config")
    quant = config.get("quantization_config")
    if (
        config.get("model_type") == "gpt_oss"
        and isinstance(quant, dict)
        and quant.get("quant_method") == "mxfp4"
    ):
        layout = "hugging_face"
        expert_key = "num_local_experts"
        layer_prefix = "model.layers.{layer}.mlp.experts"
        proj_names = ("gate_up_proj", "down_proj")
        separator = "_"
    elif "model_type" not in config and quant is None and "num_experts" in config:
        layout = "openai_original"
        expert_key = "num_experts"
        layer_prefix = "block.{layer}.mlp"
        proj_names = ("mlp1_weight", "mlp2_weight")
        separator = "."
    else:
        raise ValueError("native MXFP4 requires a GPT-OSS MXFP4 config")
    layers, experts, hidden, inner = (
        config.get(key)
        for key in ("num_hidden_layers", expert_key, "hidden_size", "intermediate_size")
    )
    if any(type(n) is not int or n < 1 for n in (layers, experts, hidden, inner)):
        raise ValueError("invalid GPT-OSS dimensions")
    if hidden % _BLOCK or inner % _BLOCK:
        raise ValueError("GPT-OSS MXFP4 projection widths must be multiples of 32")

    prefixes, payloads = _payload_layout(_checkpoint_inventory(source, layout))

    operators = {}
    for layer in range(layers):
        for proj, outputs, width in (
            (proj_names[0], 2 * inner, hidden),
            (proj_names[1], hidden, inner),
        ):
            name = f"{layer_prefix.format(layer=layer)}.{proj}"
            blocks = payloads.pop(f"{name}{separator}blocks", None)
            scales = payloads.pop(f"{name}{separator}scales", None)
            shape = (experts, outputs, width // _BLOCK)
            if (
                blocks is None
                or scales is None
                or blocks.shape != (*shape, _BLOCK // 2)
                or scales.shape != shape
                or blocks.dtype != "U8"
                or scales.dtype != "U8"
            ):
                raise ValueError(f"{name}: invalid or missing native MXFP4 pair")
            operators[name] = _Operator(name, experts * outputs, width, blocks, scales)
    for name, data in payloads.items():
        if data.dtype not in {"F32", "F16", "BF16"} or not prod(data.shape) or name in operators:
            raise ValueError(f"{name}: unsupported native MXFP4 checkpoint payload")
        width = data.shape[-1] if data.shape else 1
        operators[name] = _Operator(name, prod(data.shape) // width, width, data)
    return layout, prefixes, operators


def _tiles(operator):
    tile_rows = max(1, _TILE_FLOATS // operator.width)
    for start in range(0, operator.rows, tile_rows):
        yield start, min(operator.rows, start + tile_rows)


def _write_operator(source, destination, operator, oracle, batch_size, stats):
    calls = 0
    payload = destination / operator.data.shard.relative_to(source)
    scales = operator.scales
    if scales is None:
        with payload.open("r+b") as data:
            data.seek(operator.data.offset)
            for start, stop in _tiles(operator):
                values, count = _observed_tile(oracle, operator, start, stop, batch_size)
                data.write(memoryview(_encode_dense(values, operator.data.dtype)).cast("B"))
                calls += count
        return calls

    scale_path = destination / scales.shard.relative_to(source)
    with payload.open("r+b") as data, scale_path.open("r+b") as scale_file:
        data.seek(operator.data.offset)
        scale_file.seek(scales.offset)
        for start, stop in _tiles(operator):
            values, count = _observed_tile(oracle, operator, start, stop, batch_size)
            groups = values.reshape(-1, _BLOCK)
            exponents, codes, counts = infer_mxfp4_fields(groups, max_exponent=254)
            blocks = codes[:, ::2] | (codes[:, 1::2] << 4)
            if not np.array_equal(decode_mxfp4_safetensors(blocks, exponents), groups):
                raise ValueError(
                    f"{operator.name}: packed native MXFP4 tile does not reproduce probes"
                )
            data.write(memoryview(np.ascontiguousarray(blocks)).cast("B"))
            scale_file.write(memoryview(np.ascontiguousarray(exponents)).cast("B"))
            for key, value in counts.items():
                stats[key] += value
            calls += count
    return calls


def recover_mxfp4_safetensors(source, output, *, batch_size=256, device="cpu", call_tensor=None):
    """Rebuild a native GPT-OSS MXFP4 directory from float32 affine responses.

    The optional callback takes (name, row_start, row_stop, unit_inputs), where
    expert and output axes are flattened in checkpoint order. The callback
    returns float32 values of shape (len(unit_inputs), row_stop - row_start).
    """
    source, output = Path(source), Path(output)
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("native MXFP4 output must be a new directory outside the source")
    if type(batch_size) is not int or batch_size < 1 or device not in {"cpu", "cuda"}:
        raise ValueError("invalid native MXFP4 probe batch size or device")
    layout, prefixes, operators = _layout(source)
    torch = None
    if call_tensor is None and device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
    oracle = _local_oracle(operators, device, torch) if call_tensor is None else call_tensor
    previous_precision = torch.get_float32_matmul_precision() if torch is not None else None
    if torch is not None:
        torch.set_float32_matmul_precision("highest")
    calls = 0
    stats = defaultdict(int)
    try:
        with _staged_checkpoint(source, output, prefixes) as staged:
            for operator in operators.values():
                calls += _write_operator(source, staged, operator, oracle, batch_size, stats)
    finally:
        if torch is not None:
            torch.set_float32_matmul_precision(previous_precision)
    return {
        "schema_version": 1,
        "observation_mode": "chosen_inputs_to_native_gpt_oss_tensor_operators",
        "checkpoint_layout": layout,
        "source": str(source),
        "checkpoint": str(output),
        "oracle_device": device if call_tensor is None else "external",
        "tensor_calls": calls,
        "observed_output_scalars": sum(op.rows * op.width for op in operators.values()),
        "dense_values": sum(op.rows * op.width for op in operators.values() if op.scales is None),
        "mxfp4": dict(stats),
    }


def audit_mxfp4_safetensors(reference, candidate):
    """Compare the exact effective weights, scale bytes and FP4 codes separately."""
    layout, _, first = _layout(Path(reference))
    other_layout, _, second = _layout(Path(candidate))
    if layout != other_layout or first.keys() != second.keys():
        raise ValueError("native MXFP4 tensor directories disagree")
    stats = defaultdict(int)
    for name, a in first.items():
        b = second[name]
        if (
            a.data.shape != b.data.shape
            or a.data.dtype != b.data.dtype
            or (a.scales is None) != (b.scales is None)
        ):
            raise ValueError(f"{name}: native MXFP4 tensor layouts disagree")
        for start, stop in _tiles(a):
            if a.scales is None:
                values = a.source_rows(start, stop)
                stats["dense_values"] += values.size
                stats["exact_dense_values"] += int(
                    np.count_nonzero(values == b.source_rows(start, stop))
                )
                continue
            raw_a, scales_a = a.packed_rows(start, stop)
            raw_b, scales_b = b.packed_rows(start, stop)
            same_scale = scales_a == scales_b
            xor = raw_a ^ raw_b
            exact_codes = np.count_nonzero((xor & 15) == 0) + np.count_nonzero((xor >> 4) == 0)
            exact_values = np.count_nonzero(
                decode_mxfp4_safetensors(raw_a, scales_a)
                == decode_mxfp4_safetensors(raw_b, scales_b)
            )
            stats["mxfp4_scales"] += scales_a.size
            stats["exact_mxfp4_scales"] += int(np.count_nonzero(same_scale))
            stats["mxfp4_codes"] += raw_a.size * 2
            stats["exact_mxfp4_codes"] += int(exact_codes)
            stats["mxfp4_values"] += raw_a.size * 2
            stats["exact_mxfp4_values"] += int(exact_values)
            stats["mxfp4_blocks"] += scales_a.size
            stats["exact_mxfp4_blocks"] += int(
                np.count_nonzero(same_scale & (xor == 0).all(axis=-1))
            )
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "checkpoint_layout": layout,
        **stats,
    }


def _recover_command(args):
    return recover_mxfp4_safetensors(
        args.model, args.output, device=args.device, batch_size=args.batch_size
    )


def _audit_command(args):
    if args.output and any(
        Path(args.output).resolve().is_relative_to(Path(path).resolve())
        for path in (args.reference, args.candidate)
    ):
        raise ValueError("native MXFP4 audit report must be outside both checkpoints")
    report = audit_mxfp4_safetensors(args.reference, args.candidate)
    if args.output:
        write_json(args.output, report)
    return report


def add_subcommands(commands):
    recover = commands.add_parser(
        "recover-mxfp4-safetensors", help="recover native GPT-OSS MXFP4 safetensors"
    )
    recover.add_argument("--model", required=True, help="local native MXFP4 checkpoint directory")
    recover.add_argument("--output", required=True, help="new checkpoint directory")
    recover.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    recover.add_argument("--batch-size", type=int, default=256)
    recover.set_defaults(func=_recover_command)

    audit = commands.add_parser(
        "audit-mxfp4-safetensors", help="compare native GPT-OSS MXFP4 checkpoints"
    )
    audit.add_argument("reference")
    audit.add_argument("candidate")
    audit.add_argument("--output", help="optional JSON report path")
    audit.set_defaults(func=_audit_command)
