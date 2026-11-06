"""Recover MXFP4 fields from exact numeric observations."""

from __future__ import annotations

import numpy as np

_BLOCK_VALUES = 32
_CHUNK_BLOCKS = 1 << 15
# E2M1 magnitudes doubled, mapped to their unsigned three-bit codes.
_MAGNITUDE_CODES = np.array([0, 1, 2, 3, 4, 255, 5, 255, 6, 255, 255, 255, 7], dtype=np.uint8)


def _infer_chunk(blocks, max_exponent):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        values = blocks.astype(np.float32, copy=False)
    if not np.isfinite(values).all() or (
        values is not blocks and not np.array_equal(blocks, values)
    ):
        raise ValueError("MXFP4 observations must be finite exact float32 values")

    maximum = np.abs(values).max(axis=1)
    zero = maximum == 0
    # The GGUF decoder uses 2**(e-128) times an integer code of magnitude
    # at most 12. A smaller exponent would need a code of magnitude >=16.
    reference_exponent = np.frexp(maximum)[1] + 124
    exponent = np.where(zero, 0, np.maximum(reference_exponent, 0))
    if np.any(exponent > max_exponent):
        raise ValueError("observations have no exact MXFP4 block encoding")
    with np.errstate(under="ignore"):
        magnitudes = np.ldexp(np.abs(values), (128 - exponent)[:, None])
    if np.any(magnitudes > 12):
        raise ValueError("observations have no exact MXFP4 block encoding")
    integers = magnitudes.astype(np.uint8)
    codes = _MAGNITUDE_CODES[integers]
    if np.any(integers != magnitudes) or np.any(codes == 255):
        raise ValueError("observations have no exact MXFP4 block encoding")

    # A feasible larger exponent would have codes whose doubles are
    # also legal at this smallest candidate. Conversely, halving even
    # legal codes is always legal, so the next exponent tests ambiguity.
    all_even = (np.bitwise_or.reduce(integers, axis=1) & 1) == 0
    ambiguous_scale = all_even & (exponent < max_exponent)
    zero_values = integers == 0
    zero_codes_per_block = zero_values.sum(axis=1)
    ambiguous_zero = zero_codes_per_block != 0
    zero_codes = int(zero_codes_per_block.sum())
    # Choosing a different exponent changes the scale field and every
    # nonzero code. Either zero code remains available at every exponent.
    ambiguous_fields = zero_codes + int((33 - zero_codes_per_block[ambiguous_scale]).sum())
    codes |= np.where(np.signbit(values) & ~zero_values, np.uint8(8), np.uint8(0))
    return (
        exponent.astype(np.uint8),
        codes,
        {
            "blocks": len(blocks),
            "ambiguous_blocks": int(np.count_nonzero(ambiguous_scale | ambiguous_zero)),
            "zero_blocks": int(np.count_nonzero(zero)),
            "reference_convention_blocks": int(
                np.count_nonzero(zero | (exponent == reference_exponent))
            ),
            "ambiguous_scale_blocks": int(np.count_nonzero(ambiguous_scale)),
            "ambiguous_zero_code_blocks": int(np.count_nonzero(ambiguous_zero)),
            "ambiguous_zero_codes": zero_codes,
            "ambiguous_packed_fields": ambiguous_fields,
        },
    )


def infer_mxfp4_fields(values, *, max_exponent=255):
    """Return scale bytes, logical E2M1 codes, and numeric ambiguity counts.

    The final axis holds the 32 exact observations in a block. The
    representative uses the smallest feasible exponent and positive zero.
    The GGUF decoder treats scale byte 255 numerically. Use max_exponent=254
    when the enclosing format reserves it as an E8M0 NaN.
    """
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[-1] != _BLOCK_VALUES:
        raise ValueError("MXFP4 block observations must have last dimension 32")
    if values.dtype.kind != "f" or values.dtype.itemsize > 8:
        raise ValueError("MXFP4 block observations require real floating-point values")
    if type(max_exponent) is not int or not 0 <= max_exponent <= 255:
        raise ValueError("MXFP4 maximum exponent must be an integer from 0 to 255")

    flat = values.reshape(-1, _BLOCK_VALUES)
    exponent = np.empty(len(flat), dtype=np.uint8)
    codes = np.empty(flat.shape, dtype=np.uint8)
    counts = dict(
        blocks=0,
        ambiguous_blocks=0,
        zero_blocks=0,
        reference_convention_blocks=0,
        ambiguous_scale_blocks=0,
        ambiguous_zero_code_blocks=0,
        ambiguous_zero_codes=0,
        ambiguous_packed_fields=0,
    )
    for start in range(0, len(flat), _CHUNK_BLOCKS):
        end = start + _CHUNK_BLOCKS
        exponent[start:end], codes[start:end], chunk = _infer_chunk(flat[start:end], max_exponent)
        for key, value in chunk.items():
            counts[key] += value
    return exponent.reshape(values.shape[:-1]), codes.reshape(values.shape), counts


def recover_mxfp4(values):
    """Return GGUF blocks with the two 16-code halves in the nibble planes."""
    exponent, codes, counts = infer_mxfp4_fields(values)
    nibbles = codes[..., :16] | (codes[..., 16:] << np.uint8(4))
    return np.concatenate((exponent[..., None], nibbles), axis=-1), counts
