"""Invert GGML block quantization from exact dequantized values."""

from __future__ import annotations

import numpy as np

_BLOCK_SIZE = 32
_CHUNK_BLOCKS = 1 << 15
_HALF_UNIT_EXPONENT = 24
_HALF_MAX = int(np.finfo(np.float16).max)
_FORMATS = {"Q8_0": (128, 34), "Q4_0": (8, 18)}


def _scale_is_half(scale_units):
    values = np.ldexp(scale_units.astype(np.float64), -_HALF_UNIT_EXPONENT)
    with np.errstate(over="ignore"):
        return values.astype(np.float16).astype(np.float64) == values


def _recover_chunk(blocks, limit, q8):
    # All finite FP16 numbers are integer multiples of 2**-24. Their
    # products with these short integer codes are exact in FP32.
    with np.errstate(over="ignore"):
        units = np.ldexp(blocks.astype(np.float64), _HALF_UNIT_EXPONENT)
    largest = (_HALF_MAX * limit) << _HALF_UNIT_EXPONENT
    if (
        not np.isfinite(units).all()
        or np.any(np.abs(units) > largest)
        or np.any(units != np.trunc(units))
    ):
        raise ValueError("observations are outside the exact GGML block lattice")
    units = units.astype(np.int64)
    spacing = np.gcd.reduce(np.abs(units), axis=1)
    zero = spacing == 0
    primitive = units // np.maximum(spacing, 1)[:, None]
    pos = np.maximum(primitive.max(axis=1), 0)
    neg = np.maximum(-primitive.min(axis=1), 0)
    extent = np.maximum(pos, neg)

    # With code gcd m, |scale| is spacing/m and the codes are m times
    # the primitive row. The asymmetric code range limits m for each sign.
    pos_divisor = np.maximum(pos, 1)
    neg_divisor = np.maximum(neg, 1)
    m_positive = np.minimum(np.where(pos, (limit - 1) // pos_divisor, limit), limit // neg_divisor)
    m_negative = np.minimum(limit // pos_divisor, np.where(neg, (limit - 1) // neg_divisor, limit))
    upper = np.where(zero, 0, np.maximum(m_positive, m_negative))
    choices = np.zeros(len(blocks), dtype=np.int16)
    selected = np.zeros(len(blocks), dtype=np.int16)
    convention = zero.copy()

    eligible = np.flatnonzero(upper)
    for m in range(1, int(upper.max(initial=0)) + 1):
        eligible = eligible[
            (upper[eligible] >= m) & (~convention[eligible] | (choices[eligible] < 2))
        ]
        if not len(eligible):
            break
        rows = eligible[spacing[eligible] % m == 0]
        rows = rows[_scale_is_half(spacing[rows] // m)]
        positive = m <= m_positive[rows]
        negative = m <= m_negative[rows]
        choices[rows] += positive.astype(np.int16) + negative

        # The reference quantizers fix the sign and use an extremal code.
        # Outside that convention, prefer the smallest exact scale, then +d.
        if q8:
            preferred_positive = positive & (extent[rows] * m == limit - 1)
            preferred_negative = np.zeros(len(rows), dtype=bool)
        else:
            preferred_positive = positive & (neg[rows] * m == limit)
            preferred_negative = negative & (pos[rows] * m == limit)
        unselected = ~convention[rows]
        sign = np.where(preferred_negative, -1, np.where(positive, 1, -1))
        selected[rows[unselected]] = sign[unselected] * m
        convention[rows] |= preferred_positive | preferred_negative

    if np.any((selected == 0) & ~zero):
        raise ValueError("observations have no exact GGML block encoding")
    scale_units = spacing // np.maximum(np.abs(selected), 1)
    scale = np.ldexp(scale_units.astype(np.float64), -_HALF_UNIT_EXPONENT)
    scale[selected < 0] *= -1
    if not q8:
        # ggml's Q4_0 reference uses signed_max / -8; signed_max starts at +0.
        scale[zero] = -0.0
    d = scale.astype("<f2").view(np.uint8).reshape(-1, 2)
    codes = primitive * selected[:, None]
    if q8:
        qs = codes.astype(np.int8).view(np.uint8)
    else:
        nibbles = (codes + 8).astype(np.uint8)
        qs = nibbles[:, :16] | (nibbles[:, 16:] << 4)
    return np.concatenate((d, qs), axis=1), {
        "blocks": len(blocks),
        "ambiguous_blocks": int(np.count_nonzero(zero | (choices > 1))),
        "zero_blocks": int(np.count_nonzero(zero)),
        "reference_convention_blocks": int(np.count_nonzero(convention)),
    }


def recover_blocks(values, encoding):
    """Pack exact ``(..., 32)`` observations as little-endian Q8_0 or Q4_0.

    The packed shape has last dimension 34 or 18. The ambiguity count includes
    any distinct legal packed representation, even if a single one matches the
    GGML reference quantizer's nonzero extremum convention. The canonical zero
    encoding uses zero codes and the reference quantizer's signed zero scale.
    """
    encoding = getattr(encoding, "name", encoding)
    if not isinstance(encoding, str) or encoding not in _FORMATS:
        raise ValueError(f"unsupported GGML block encoding {encoding!r}")
    limit, width = _FORMATS[encoding]
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[-1] != _BLOCK_SIZE:
        raise ValueError("GGML block observations must have last dimension 32")
    if values.dtype.kind != "f" or values.dtype.itemsize > 8:
        raise ValueError("GGML block observations require real floating-point values")

    flat = values.reshape(-1, _BLOCK_SIZE)
    packed = np.empty((len(flat), width), dtype=np.uint8)
    stats = dict(blocks=0, ambiguous_blocks=0, zero_blocks=0, reference_convention_blocks=0)
    for start in range(0, len(flat), _CHUNK_BLOCKS):
        end = start + _CHUNK_BLOCKS
        packed[start:end], counts = _recover_chunk(flat[start:end], limit, encoding == "Q8_0")
        for key, count in counts.items():
            stats[key] += count
    return packed.reshape(*values.shape[:-1], width), stats
