"""Invert finite E4M3FN blocks with one positive dequantization scale."""

from __future__ import annotations

import numpy as np

_F32_TINY = np.finfo(np.float32).tiny
_WORK_VALUES = 1 << 20
_MAGNITUDE_FIELDS = np.arange(127, dtype=np.uint8)
_EXPONENTS = (_MAGNITUDE_FIELDS >> 3).astype(np.int16)
_MANTISSAS = _MAGNITUDE_FIELDS & 7
_MAGNITUDES = np.ldexp(
    np.where(_EXPONENTS == 0, _MANTISSAS, 8 + _MANTISSAS).astype(np.float32),
    np.maximum(_EXPONENTS, 1) - 10,
)
_MIDPOINTS = (_MAGNITUDES[:-1] + _MAGNITUDES[1:]) / 2
_POSITIVE = np.append(_MAGNITUDES, np.float32(np.nan))
_CODEBOOK = np.concatenate((_POSITIVE, -_POSITIVE))
_ANCHORS = _MAGNITUDES[:0:-1]


def decode_fp8_blocks(codes, scales, *, dtype=np.float32):
    """Decode flattened E4M3FN code rows with one FP32 scale per block."""
    codes, scales, dtype = np.asarray(codes), np.asarray(scales), np.dtype(dtype)
    if codes.dtype != np.uint8 or codes.ndim < 1 or codes.shape[:-1] != scales.shape:
        raise ValueError("FP8 codes must be uint8 with one scale per block")
    if (
        scales.dtype != np.float32
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or dtype not in (np.float32, np.float64)
    ):
        raise ValueError("FP8 decoding needs positive finite FP32 scales and FP32 or FP64 output")
    with np.errstate(over="ignore", under="ignore"):
        decoded = np.multiply(_CODEBOOK[codes], scales[..., None], dtype=dtype)
    if not np.isfinite(decoded).all():
        raise ValueError("non-finite decoded E4M3FN value")
    return decoded


def _distinct_magnitudes(values):
    ordered = np.sort(values, axis=1)
    distinct = ordered > 0
    distinct[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    rows, columns = distinct.nonzero()
    lengths = distinct.sum(axis=1)
    width = int(lengths.max(initial=0))
    if width >= len(_MAGNITUDES):
        raise ValueError("observations contain more distinct magnitudes than E4M3FN supports")
    indices = np.arange(len(rows)) - np.repeat(lengths.cumsum() - lengths, lengths)
    unique = np.zeros((len(values), width), dtype=values.dtype)
    unique[rows, indices] = ordered[rows, columns]
    return unique


def _scale_candidates(maximum, scale_dtype):
    with np.errstate(over="ignore", under="ignore"):
        nearest = (maximum.astype(np.float64)[:, None] / _ANCHORS).astype(np.float32)
    if scale_dtype == "bfloat16":
        bits = nearest.view(np.uint32)
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return rounded.view(np.float32)[..., None]
    if maximum.dtype == np.float64:
        return nearest[..., None]
    # For normal FP32 products, the inverse rounding interval lies within
    # one scale ULP of its exact quotient. Include the neighbors to handle
    # adjacent scales that round to the same observed maximum.
    with np.errstate(over="ignore", under="ignore"):
        return np.stack(
            (
                nearest,
                np.nextafter(nearest, np.float32(0)),
                np.nextafter(nearest, np.float32(np.inf)),
            ),
            axis=-1,
        )


def _nearest_codes(values, scales):
    ratios = values.astype(np.float64) / scales.astype(np.float64)[:, None]
    return np.searchsorted(_MIDPOINTS, ratios).astype(np.uint8)


def _infer_chunk(values, scale_dtype):
    if not np.isfinite(values).all():
        raise ValueError("FP8 observations must be finite")
    absolute = np.abs(values)
    if values.dtype == np.float32 and np.any((absolute != 0) & (absolute < _F32_TINY)):
        raise ValueError("FP32 FP8 inversion requires normal nonzero observations")
    unique = _distinct_magnitudes(absolute)
    maximum = absolute.max(axis=1)
    zero_blocks = maximum == 0
    candidates = _scale_candidates(maximum, scale_dtype)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        replayed = np.multiply(candidates, _ANCHORS[None, :, None], dtype=values.dtype)
    valid = (candidates > 0) & np.isfinite(candidates) & (replayed == maximum[:, None, None])
    if values.dtype == np.float32:
        valid &= candidates >= _F32_TINY
    valid[zero_blocks] = False
    rows, anchors, variants = valid.nonzero()
    scales = candidates[rows, anchors, variants]
    fits = np.empty(len(rows), dtype=bool)
    step = max(1, _WORK_VALUES // max(1, unique.shape[1]))
    for start in range(0, len(rows), step):
        stop = start + step
        observed = unique[rows[start:stop]]
        possible_scales = scales[start:stop]
        codes = _nearest_codes(observed, possible_scales)
        with np.errstate(over="ignore", under="ignore"):
            decoded = np.multiply(_MAGNITUDES[codes], possible_scales[:, None], dtype=values.dtype)
        fits[start:stop] = (decoded == observed).all(axis=1)
    rows, anchors, scales = rows[fits], anchors[fits], scales[fits]
    counts = np.bincount(rows, minlength=len(values))
    if np.any((counts == 0) & ~zero_blocks):
        raise ValueError("observations have no exact supported E4M3FN block encoding")

    selected_scales = np.ones(len(values), dtype=np.float32)
    saturated = 0
    code_aliases = np.zeros(len(values), dtype=bool)
    if len(rows):
        first = np.concatenate(([True], rows[1:] != rows[:-1]))
        last = np.concatenate((rows[:-1] != rows[1:], [True]))
        selected_scales[rows[first]] = scales[first]
        saturated = int(np.count_nonzero(anchors[first] == 0))
        code_aliases[rows[first]] = anchors[first] != anchors[last]
    codes = _nearest_codes(absolute, selected_scales)
    codes |= (np.signbit(values) & (codes != 0)).astype(np.uint8) << 7
    # Validate each returned code after inversion, separately from screening.
    if not np.array_equal(decode_fp8_blocks(codes, selected_scales, dtype=values.dtype), values):
        raise ValueError("recovered E4M3FN fields do not reproduce the observations")

    scale_aliases = zero_blocks | (counts > 1)
    zero_codes = (absolute == 0).sum(axis=1)
    zero_count = int(zero_codes.sum())
    scale_count = int(scale_aliases.sum())
    return (
        selected_scales,
        codes,
        {
            "blocks": len(values),
            "zero_blocks": int(zero_blocks.sum()),
            "saturated_blocks": saturated,
            "ambiguous_blocks": int(np.count_nonzero(scale_aliases | (zero_codes != 0))),
            "ambiguous_scale_blocks": scale_count,
            "ambiguous_code_scale_blocks": int(code_aliases.sum()),
            "ambiguous_zero_codes": zero_count,
            "ambiguous_packed_fields": (
                zero_count + scale_count + int((values.shape[1] - zero_codes[code_aliases]).sum())
            ),
        },
    )


def infer_fp8_blocks(values, *, scale_dtype="float32"):
    """Return FP32 scales, uint8 E4M3FN codes, and numeric ambiguity counts.

    The final axis contains one flattened block. FP64 observations capture
    exact products of finite positive FP32 scales and E4M3FN codes. The
    FP32 path supports normal nonzero observations and normal scales.
    BF16 scale candidates are restricted to the BF16 grid in either case.
    Prefer the largest feasible maximum code and use positive zero.
    """
    values = np.asarray(values)
    if values.ndim < 1 or not values.shape[-1] or values.dtype not in (np.float32, np.float64):
        raise ValueError("FP8 inversion needs nonempty FP32 or FP64 blocks")
    if scale_dtype not in {"float32", "bfloat16"}:
        raise ValueError("FP8 scale dtype must be float32 or bfloat16")
    width = values.shape[-1]
    flat = values.reshape(-1, width)
    scales = np.empty(len(flat), dtype=np.float32)
    codes = np.empty(flat.shape, dtype=np.uint8)
    totals = dict(
        blocks=0,
        zero_blocks=0,
        saturated_blocks=0,
        ambiguous_blocks=0,
        ambiguous_scale_blocks=0,
        ambiguous_code_scale_blocks=0,
        ambiguous_zero_codes=0,
        ambiguous_packed_fields=0,
    )
    step = min(512, max(1, _WORK_VALUES // width))
    for start in range(0, len(flat), step):
        stop = start + step
        scales[start:stop], codes[start:stop], counts = _infer_chunk(flat[start:stop], scale_dtype)
        for key, count in counts.items():
            totals[key] += count
    return scales.reshape(values.shape[:-1]), codes.reshape(values.shape), totals
