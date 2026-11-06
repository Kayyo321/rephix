"""Invert the signed, two-level lattice used by GGML Q6_K."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .gguf_quant import _scale_is_half

_GROUPS = 16
_GROUP_SIZE = 16
_BLOCK_SIZE = _GROUPS * _GROUP_SIZE
_PACKED_SIZE = 210
_CHUNK_BLOCKS = 1 << 12
_MAX_PRODUCT = 128 * 32
_ANCHOR_SCORE = 1 << 16


@lru_cache(maxsize=1)
def _factor_tables():
    """For each code limit and product, keep the best multiplier and count."""
    best = np.zeros((2, 33, _MAX_PRODUCT + 1), dtype=np.int8)
    counts = np.zeros_like(best)
    for sign_index, max_scale in enumerate((127, 128)):
        for multiplier in range(1, 33):
            products = multiplier * np.arange(1, max_scale + 1)
            best[sign_index, multiplier:][:, products] = multiplier
            counts[sign_index, multiplier:][:, products] += 1
    return best, counts


def _code_limit(pos, neg):
    positive = np.where(pos, 31 // np.maximum(pos, 1), 32)
    negative = np.where(neg, 32 // np.maximum(neg, 1), 32)
    return np.minimum(positive, negative)


def _choose_groups(products, pos_limit, neg_limit, extent, prefer_positive, positive_scale):
    """Resolve local factors for a proposed signed block scale."""
    best, counts = _factor_tables()
    pos_index = (int(not positive_scale), pos_limit, products)
    neg_index = (int(positive_scale), neg_limit, products)
    positive, negative = best[pos_index], best[neg_index]
    use_positive = (positive > negative) | ((positive == negative) & prefer_positive)
    selected = np.where(use_positive, positive, -negative).astype(np.int16)
    zero_group = products == 0
    legal = ((selected != 0) | zero_group).all(axis=1)
    ambiguous = ((counts[pos_index] + counts[neg_index] > 1) | zero_group).any(axis=1)

    direction = 1 if positive_scale else -1
    # GGML chooses the block scale from the largest subgroup scale, whose
    # stored signed multiplier is -128. If possible, include that anchor.
    anchor_multiplier = products // 128
    anchor_limit = neg_limit if positive_scale else pos_limit
    can_anchor = (products != 0) & (products % 128 == 0) & (anchor_multiplier <= anchor_limit)
    anchor_codes = -direction * anchor_multiplier
    anchor_present = (can_anchor & (selected == anchor_codes)).any(axis=1)
    force_rows = np.flatnonzero(~anchor_present & can_anchor.any(axis=1))
    if len(force_rows):
        ordinary_preferred = (selected[force_rows] > 0) == prefer_positive[force_rows]
        anchor_preferred = (direction < 0) == prefer_positive[force_rows]
        cost = (
            (np.abs(selected[force_rows]) - anchor_multiplier[force_rows]) * extent[force_rows] * 32
            + ordinary_preferred.astype(np.int16)
            - anchor_preferred
        )
        columns = np.where(can_anchor[force_rows], cost, np.iinfo(np.int64).max).argmin(axis=1)
        selected[force_rows, columns] = anchor_codes[force_rows, columns]

    preferred = ((selected > 0) == prefer_positive) & ~zero_group
    score = (
        (np.abs(selected) * extent).sum(axis=1) * 32
        + preferred.sum(axis=1)
        + can_anchor.any(axis=1) * _ANCHOR_SCORE
    )
    return selected, legal, ambiguous, score


def _pack_q6k(d, scales, codes):
    unsigned = (codes + 32).astype(np.uint8).reshape(-1, 2, 4, 32)
    low = unsigned & 15
    high = unsigned >> 4
    ql = np.concatenate(
        (low[:, :, 0] | (low[:, :, 2] << 4), low[:, :, 1] | (low[:, :, 3] << 4)), axis=2
    )
    qh = high[:, :, 0] | (high[:, :, 1] << 2) | (high[:, :, 2] << 4) | (high[:, :, 3] << 6)
    return np.concatenate(
        (
            ql.reshape(-1, 128),
            qh.reshape(-1, 64),
            scales.astype(np.int8).view(np.uint8),
            d.astype("<f2").view(np.uint8).reshape(-1, 2),
        ),
        axis=1,
    )


def _recover_chunk(blocks):
    grouped = blocks.reshape(-1, _GROUPS, _GROUP_SIZE)
    # A finite FP16 scale is an integer multiple of 2**-24. Multiplying
    # its significand by the signed int8 scale and signed int6 code still
    # fits exactly in float32, so we can solve in integer units.
    with np.errstate(over="ignore"):
        units = np.ldexp(grouped.astype(np.float64), 24)
    largest = (65504 * _MAX_PRODUCT) << 24
    if (
        not np.isfinite(units).all()
        or np.any(np.abs(units) > largest)
        or np.any(units != np.trunc(units))
    ):
        raise ValueError("observations are outside the exact Q6_K block lattice")
    units = units.astype(np.int64)
    spacing = np.gcd.reduce(np.abs(units), axis=2)
    common = np.gcd.reduce(spacing, axis=1)
    zero = common == 0
    primitive = units // np.maximum(spacing, 1)[:, :, None]
    pos = np.maximum(primitive.max(axis=2), 0)
    neg = np.maximum(-primitive.min(axis=2), 0)
    extent = np.maximum(pos, neg)
    first_max = np.take_along_axis(
        primitive, np.abs(primitive).argmax(axis=2, keepdims=True), axis=2
    )[..., 0]
    prefer_positive = first_max < 0
    pos_limit, neg_limit = _code_limit(pos, neg), _code_limit(neg, pos)
    product = spacing // np.maximum(common, 1)[:, None]
    code_limit = np.maximum(pos_limit, neg_limit)
    max_product = 128 * code_limit
    upper = np.where(product, max_product // np.maximum(product, 1), _MAX_PRODUCT).min(axis=1)
    upper[zero] = 0

    best_score = np.full(len(blocks), -1, dtype=np.int64)
    max_score = _ANCHOR_SCORE + (code_limit * extent).sum(axis=1) * 32 + (spacing != 0).sum(axis=1)
    scale_units = np.zeros(len(blocks), dtype=np.int64)
    group_scales = np.zeros((len(blocks), _GROUPS), dtype=np.int16)
    multipliers = np.zeros_like(group_scales)
    representations = np.zeros(len(blocks), dtype=np.int8)
    eligible = np.flatnonzero(upper)
    for divisor in range(1, int(upper.max(initial=0)) + 1):
        eligible = eligible[
            (upper[eligible] >= divisor)
            & ((representations[eligible] < 2) | (best_score[eligible] < max_score[eligible]))
        ]
        if not len(eligible):
            break
        rows = eligible[common[eligible] % divisor == 0]
        rows = rows[_scale_is_half(common[rows] // divisor)]
        if not len(rows):
            continue
        products = product[rows] * divisor
        for positive_scale in (True, False):
            choice, legal, ambiguous, score = _choose_groups(
                products,
                pos_limit[rows],
                neg_limit[rows],
                extent[rows],
                prefer_positive[rows],
                positive_scale,
            )
            representations[rows] = np.minimum(
                2, representations[rows] + legal + (legal & ambiguous)
            )
            replace = legal & (score > best_score[rows])
            chosen_rows = rows[replace]
            direction = 1 if positive_scale else -1
            best_score[chosen_rows] = score[replace]
            scale_units[chosen_rows] = direction * (common[chosen_rows] // divisor)
            multipliers[chosen_rows] = choice[replace]
            group_scales[chosen_rows] = (
                direction
                * np.sign(choice[replace])
                * (products[replace] // np.maximum(np.abs(choice[replace]), 1))
            )
    if np.any((representations == 0) & ~zero):
        raise ValueError("observations have no exact Q6_K block encoding")

    d = np.ldexp(scale_units.astype(np.float64), -24)
    codes = primitive * multipliers[:, :, None]
    # A zero subgroup leaves its codes unobserved. Use the reference
    # zero-initialized code payload when writing such a subgroup.
    codes[spacing == 0] = -32
    return _pack_q6k(d, group_scales, codes), {
        "blocks": len(blocks),
        "ambiguous_blocks": int(np.count_nonzero(zero | (representations > 1))),
        "zero_blocks": int(np.count_nonzero(zero)),
        "reference_convention_blocks": int(np.count_nonzero(zero | (best_score >= _ANCHOR_SCORE))),
    }


def recover_q6k(values):
    """Recover exact Q6_K blocks, preferring the reference -128 scale anchor."""
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[-1] != _BLOCK_SIZE:
        raise ValueError("Q6_K block observations must have last dimension 256")
    if values.dtype.kind != "f" or values.dtype.itemsize > 8:
        raise ValueError("Q6_K block observations require real floating-point values")

    flat = values.reshape(-1, _BLOCK_SIZE)
    packed = np.empty((len(flat), _PACKED_SIZE), dtype=np.uint8)
    stats = dict(blocks=0, ambiguous_blocks=0, zero_blocks=0, reference_convention_blocks=0)
    for start in range(0, len(flat), _CHUNK_BLOCKS):
        end = start + _CHUNK_BLOCKS
        packed[start:end], counts = _recover_chunk(flat[start:end])
        for key, count in counts.items():
            stats[key] += count
    return packed.reshape(*values.shape[:-1], _PACKED_SIZE), stats
