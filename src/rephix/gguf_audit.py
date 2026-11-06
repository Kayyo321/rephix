"""Audit stored GGUF representations and independently dequantized values."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .weight_audit import DEFAULT_ATOL, DEFAULT_RTOL

_VALUES_PER_BATCH = 1 << 20
_BLOCK_FIELDS = {"Q4_0": 33, "Q5_0": 33, "Q8_0": 33, "Q4_K": 274, "Q6_K": 273, "MXFP4": 33}


@dataclass
class _Counts:
    tensors: int = 0
    payload_bytes: int = 0
    exact_payload_bytes: int = 0
    stored_fields: int = 0
    exact_stored_fields: int = 0
    quantized_blocks: int = 0
    exact_quantized_blocks: int = 0
    dequantized_values: int = 0
    exact_dequantized_values: int = 0
    within_tolerance_dequantized_values: int = 0
    max_abs_error: float = 0.0

    def report(self):
        report = vars(self).copy()
        for numerator, denominator in (
            ("exact_payload_bytes", "payload_bytes"),
            ("exact_stored_fields", "stored_fields"),
            ("exact_quantized_blocks", "quantized_blocks"),
            ("exact_dequantized_values", "dequantized_values"),
            ("within_tolerance_dequantized_values", "dequantized_values"),
        ):
            size = report[denominator]
            report[f"fraction_{numerator}"] = report[numerator] / size if size else None
        return report


def _dequantize(gguf, payload, encoding):
    # Feed contiguous bytes as one logical row. Blocks remain independent,
    # and gguf's grouped-row implementation avoids a dispatch per few blocks.
    return gguf.dequantize(payload.reshape(1, -1), encoding).reshape(-1)


def _matching_nibbles(changed):
    return int(np.count_nonzero((changed & 0x0F) == 0)) + int(
        np.count_nonzero((changed & 0xF0) == 0)
    )


def _matching_q5_0_codes(changed):
    # Q5_0 stores the high bit of each code in little-endian bit order.
    low = changed[:, 6:]
    low_equal = np.concatenate(((low & 0x0F) == 0, (low & 0xF0) == 0), axis=1)
    high_equal = np.unpackbits(changed[:, 2:6], axis=1, bitorder="little") == 0
    return int(np.count_nonzero(low_equal & high_equal))


def _matching_q4_k_fields(changed, equal_bytes):
    half_equal = equal_bytes[:, :4].reshape(-1, 2, 2).all(axis=2)
    scales, mins, upper = np.split(changed[:, 4:16], 3, axis=1)
    # The last four scale and min fields straddle two storage bytes each.
    upper_scales = ((scales & 0xC0) == 0) & ((upper & 0x0F) == 0)
    upper_mins = ((mins & 0xC0) == 0) & ((upper & 0xF0) == 0)
    return sum(
        int(np.count_nonzero(mask))
        for mask in (half_equal, (scales & 0x3F) == 0, (mins & 0x3F) == 0, upper_scales, upper_mins)
    ) + _matching_nibbles(changed[:, 16:])


def _matching_q6_k_fields(changed, equal_bytes):
    # Each 128-code half has separate four-bit and two-bit planes. Match both
    # pieces in logical code order, so a change in both counts only once.
    low = changed[:, :128].reshape(-1, 2, 1, 64)
    high = changed[:, 128:192].reshape(-1, 2, 1, 32)
    low_shift = np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    high_shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    low_equal = (((low >> low_shift) & 0x0F) == 0).reshape(-1, 256)
    high_equal = (((high >> high_shift) & 0x03) == 0).reshape(-1, 256)
    return (
        int(np.count_nonzero(low_equal & high_equal))
        + int(np.count_nonzero(equal_bytes[:, 192:208]))
        + int(np.count_nonzero(equal_bytes[:, 208:210].all(axis=1)))
    )


def _count_payload(counts, left, right, encoding):
    equal_bytes = left == right
    exact_blocks = int(np.count_nonzero(equal_bytes.all(axis=1)))
    same_payload = exact_blocks == len(left)
    counts.payload_bytes += left.nbytes
    counts.exact_payload_bytes += (
        left.nbytes if same_payload else int(np.count_nonzero(equal_bytes))
    )
    if encoding not in _BLOCK_FIELDS:
        counts.stored_fields += len(left)
        counts.exact_stored_fields += exact_blocks
        return same_payload

    counts.quantized_blocks += len(left)
    counts.exact_quantized_blocks += exact_blocks
    stored_fields = _BLOCK_FIELDS[encoding] * len(left)
    counts.stored_fields += stored_fields
    if same_payload:
        counts.exact_stored_fields += stored_fields
        return True

    changed = np.bitwise_xor(left, right)
    if encoding == "Q4_K":
        counts.exact_stored_fields += _matching_q4_k_fields(changed, equal_bytes)
    elif encoding == "Q6_K":
        counts.exact_stored_fields += _matching_q6_k_fields(changed, equal_bytes)
    elif encoding == "MXFP4":
        counts.exact_stored_fields += int(np.count_nonzero(equal_bytes[:, 0]))
        counts.exact_stored_fields += _matching_nibbles(changed[:, 1:])
    else:
        counts.exact_stored_fields += int(np.count_nonzero(equal_bytes[:, :2].all(axis=1)))
        if encoding == "Q8_0":
            counts.exact_stored_fields += int(np.count_nonzero(equal_bytes[:, 2:]))
        elif encoding == "Q5_0":
            counts.exact_stored_fields += _matching_q5_0_codes(changed)
        else:
            counts.exact_stored_fields += _matching_nibbles(changed[:, 2:])
    return False


def _count_values(gguf, counts, left, right, encoding, name, *, same_payload, atol, rtol):
    a = _dequantize(gguf, left, encoding)
    b = a if same_payload else _dequantize(gguf, right, encoding)
    if not np.isfinite(a).all() or (not same_payload and not np.isfinite(b).all()):
        raise ValueError(f"{name}: non-finite dequantized values in GGUF audit")
    counts.dequantized_values += a.size
    if same_payload or np.array_equal(a, b):
        counts.exact_dequantized_values += a.size
        counts.within_tolerance_dequantized_values += a.size
        return

    counts.exact_dequantized_values += int(np.count_nonzero(a == b))
    with np.errstate(over="ignore"):
        difference = np.abs(np.subtract(a, b, dtype=np.float64))
        tolerance = atol + rtol * np.abs(a.astype(np.float64, copy=False))
    if not np.isfinite(difference).all():
        raise ValueError(f"{name}: dequantized difference exceeds the finite audit range")
    counts.within_tolerance_dequantized_values += int(np.count_nonzero(difference <= tolerance))
    counts.max_abs_error = max(counts.max_abs_error, float(difference.max()))


def _total(by_encoding):
    sums = {}
    for field in fields(_Counts):
        values = [getattr(counts, field.name) for counts in by_encoding.values()]
        sums[field.name] = (
            max(values, default=0.0) if field.name == "max_abs_error" else sum(values)
        )
    return _Counts(**sums)


def audit_gguf(reference, candidate, *, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL):
    """Compare identical tensor layouts in bounded chunks without the recovery solver.

    Each dense value counts as one stored field. Quantized blocks count their
    individual scales, minimum offsets and logical codes. Stored equality is
    bitwise, while dequantized agreement compares numeric values.
    """
    from .gguf_recovery import raw_rows, read_gguf

    if any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError("GGUF audit tolerances must be finite and non-negative")
    reference, candidate = Path(reference), Path(candidate)
    gguf, left_reader, left_layout = read_gguf(reference)
    _, right_reader, right_layout = read_gguf(candidate)
    if len(left_layout) != len(right_layout) or left_reader.alignment != right_reader.alignment:
        raise ValueError("GGUF tensor layouts differ")
    left_start, right_start = int(left_reader.data_offset), int(right_reader.data_offset)
    prefix_exact = left_start == right_start and np.array_equal(
        left_reader.data[:left_start], right_reader.data[:right_start]
    )

    by_encoding = {}
    for left, right in zip(left_layout, right_layout, strict=True):
        if (
            left.name != right.name
            or left.shape != right.shape
            or left.encoding != right.encoding
            or left.offset - left_start != right.offset - right_start
        ):
            raise ValueError(f"GGUF tensor layouts differ at {left.name!r} and {right.name!r}")
        encoding = left.encoding.name
        block_size, block_bytes = gguf.GGML_QUANT_SIZES[left.encoding]
        if block_size != 1 and encoding not in _BLOCK_FIELDS:
            raise ValueError(f"{left.name}: no GGUF stored field audit for {encoding}")

        a = raw_rows(left_reader, left, 0, left.rows).reshape(-1, block_bytes)
        b = raw_rows(right_reader, right, 0, right.rows).reshape(-1, block_bytes)
        blocks_per_batch = _VALUES_PER_BATCH // block_size
        counts = by_encoding.setdefault(encoding, _Counts())
        counts.tensors += 1
        for start in range(0, len(a), blocks_per_batch):
            end = start + blocks_per_batch
            left_chunk, right_chunk = a[start:end], b[start:end]
            same_payload = _count_payload(counts, left_chunk, right_chunk, encoding)
            _count_values(
                gguf,
                counts,
                left_chunk,
                right_chunk,
                left.encoding,
                left.name,
                same_payload=same_payload,
                atol=atol,
                rtol=rtol,
            )

    return {
        "schema_version": 1,
        "reference": str(reference),
        "candidate": str(candidate),
        "metadata_prefix_exact": bool(prefix_exact),
        "atol": atol,
        "rtol": rtol,
        "totals": _total(by_encoding).report(),
        "by_encoding": {name: counts.report() for name, counts in sorted(by_encoding.items())},
    }
