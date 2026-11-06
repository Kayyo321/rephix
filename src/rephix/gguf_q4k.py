"""Invert Q4_K's shared half scales and affine subgroup lattices."""

from __future__ import annotations

import numpy as np

_GROUPS = 8
_GROUP_SIZE = 32
_BLOCK_SIZE = _GROUPS * _GROUP_SIZE
_PACKED_SIZE = 144
_SCALE_LIMIT = 63
_CODE_LIMIT = 15
_HALF_UNIT_EXPONENT = 24
_HALF_MAX_UNITS = 65504 << _HALF_UNIT_EXPONENT
_CHUNK_BLOCKS = 1 << 12


def _half(units):
    with np.errstate(over="ignore"):
        return np.ldexp(np.asarray(units, dtype=np.float64), -_HALF_UNIT_EXPONENT).astype("<f2")


def _valid_half(units):
    return np.ldexp(_half(units).astype(np.float64), _HALF_UNIT_EXPONENT) == units


def _fit_offsets(minimum, steps, spans, dmin):
    """Find a shared offset multiplier and local code origin for each group."""
    denominator = np.maximum(dmin[:, None], 1)
    code_origins = np.zeros(minimum.shape, dtype=np.int64)
    offsets = np.zeros_like(code_origins)
    saturated_origin = np.full_like(code_origins, -1)
    choices = np.zeros(minimum.shape, dtype=np.uint8)
    max_shift = int((_CODE_LIMIT - spans).max(initial=0))
    for shift in range(max_shift + 1):
        value = shift * steps - minimum
        multiplier = value // denominator
        valid = (
            (spans + shift <= _CODE_LIMIT)
            & ((steps > 0) | (shift == 0))
            & (value >= 0)
            & (value % denominator == 0)
            & (multiplier <= _SCALE_LIMIT)
            & ((dmin[:, None] > 0) | (value == 0))
        )
        first = valid & (choices == 0)
        code_origins[first] = shift
        offsets[first] = multiplier[first]
        saturated_origin[valid & (multiplier == _SCALE_LIMIT) & (saturated_origin < 0)] = shift
        choices += valid
    # A reference block has at least one maximum offset multiplier. Keep the
    # smallest code lifts elsewhere and, if necessary, change just one group.
    needs_max = (dmin > 0) & ~np.any(offsets == _SCALE_LIMIT, axis=1)
    eligible = needs_max & np.any(saturated_origin >= 0, axis=1)
    rows = np.flatnonzero(eligible)
    if rows.size:
        lifts = np.where(
            saturated_origin[rows] >= 0,
            saturated_origin[rows] - code_origins[rows],
            _CODE_LIMIT + 1,
        )
        group = lifts.argmin(axis=1)
        code_origins[rows, group] = saturated_origin[rows, group]
        offsets[rows, group] = _SCALE_LIMIT
    valid = np.all(choices > 0, axis=1)
    convention = (dmin == 0) | np.any(offsets == _SCALE_LIMIT, axis=1)
    ambiguous = np.any(choices > 1, axis=1)
    return offsets, code_origins, valid, convention, ambiguous


def _offset_candidates(minimum, steps, spans):
    """Enumerate reference-convention shared offsets for shifted groups."""
    for shift in range(1, int((_CODE_LIMIT - spans).max(initial=0)) + 1):
        for group in range(_GROUPS):
            offset = shift * steps[:, group] - minimum[:, group]
            candidate = offset // _SCALE_LIMIT
            valid = (
                (spans[:, group] + shift <= _CODE_LIMIT)
                & (steps[:, group] > 0)
                & (offset > 0)
                & (offset % _SCALE_LIMIT == 0)
                & _valid_half(candidate)
                & (offset >= -minimum.min(axis=1))
            )
            if np.any(valid):
                yield valid, candidate


def _shared_scales(gaps, spans, minimum):
    """Find possible global scales from a subgroup with a measured difference."""
    if not np.any(gaps):
        # Constant nonpositive groups can be represented using offsets alone.
        # Positive constants also constrain the scale if their offset is zero.
        positive = minimum[minimum > 0]
        return [0] if not positive.size else sorted(_half_divisors(int(positive[0]), 945))
    anchor = int(spans.argmax())
    gap = int(gaps[anchor])
    factors = range(1, _CODE_LIMIT // int(spans[anchor]) + 1)
    return sorted(
        {d for factor in factors if gap % factor == 0 for d in _half_divisors(gap // factor, 63)}
    )


def _half_divisors(units, max_multiplier):
    return {
        units // multiplier
        for multiplier in range(1, max_multiplier + 1)
        if units % multiplier == 0 and _valid_half(units // multiplier)
    }


def _group_options(centered, minimum, gap, d):
    """Map local subtractive offsets to all admissible scale and code origins."""
    options = {}
    for scale in range(_SCALE_LIMIT + 1):
        step = d * scale
        if gap and (step == 0 or gap % step or centered.max() > _CODE_LIMIT * step):
            continue
        indices = centered // max(step, 1)
        shifts = range(_CODE_LIMIT + 1 - int(indices.max())) if step else (0,)
        for shift in shifts:
            offset = shift * step - minimum
            if abs(offset) <= _HALF_MAX_UNITS * _SCALE_LIMIT:
                options.setdefault(offset, []).append((scale, shift, indices))
    return options


def _all_offset_scales(options):
    if not all(options):
        return set()
    # A group with no zero offset constrains dmin for every solution. Also
    # consider saturated offsets when no group has this requirement.
    nonzero = [group for group in options if 0 not in group]
    if nonzero:
        anchor = min(nonzero, key=len)
        return {d for offset in anchor for d in _half_divisors(offset, _SCALE_LIMIT)}
    return {0} | {
        offset // _SCALE_LIMIT
        for group in options
        for offset in group
        if offset and offset % _SCALE_LIMIT == 0 and _valid_half(offset // _SCALE_LIMIT)
    }


def _choose_groups(options, dmin):
    per_group = []
    for group in options:
        matches = [
            (s, m, shift, indices)
            for m in (range(_SCALE_LIMIT + 1) if dmin else (0,))
            for s, shift, indices in group.get(m * dmin, ())
        ]
        if not matches:
            return None
        # Nonconstant groups prefer the finest observed code lattice.
        # Constant groups prefer zero scale unless a code requires otherwise.
        matches.sort(key=lambda x: (x[2], x[0] if not np.any(x[3]) else -x[0]))
        per_group.append(matches)

    choices = [group[0] for group in per_group]
    if dmin and not any(choice[1] == _SCALE_LIMIT for choice in choices):
        # Prefer an available saturated offset without changing other groups.
        saturated = [
            (item[2] - choices[i][2], i, item)
            for i, group in enumerate(per_group)
            for item in group
            if item[1] == _SCALE_LIMIT
        ]
        if saturated:
            _, index, item = min(saturated, key=lambda x: x[:2])
            choices[index] = item
    return choices, any(len(group) > 1 for group in per_group)


def _fallback(centered, minimum, gaps, spans):
    best = None
    score = None
    solutions = 0
    ambiguous = False
    for d in _shared_scales(gaps, spans, minimum):
        options = [
            _group_options(row, int(low), int(gap), d)
            for row, low, gap in zip(centered, minimum, gaps, strict=True)
        ]
        for dmin in sorted(_all_offset_scales(options)):
            fitted = _choose_groups(options, dmin)
            if fitted is None:
                continue
            choices, local_ambiguity = fitted
            scales, offsets, origins, codes = zip(*choices, strict=True)
            convention = (
                (not d or max(scales) == _SCALE_LIMIT)
                and dmin >= 0
                and (not dmin or max(offsets) == _SCALE_LIMIT)
            )
            # Reference saturation is stronger evidence than a smaller code lift.
            rank = (not convention, sum(origins), -sum(scales), d, dmin)
            solutions += 1
            if score is None or rank < score:
                best = (
                    d,
                    dmin,
                    scales,
                    offsets,
                    np.stack(codes) + np.array(origins)[:, None],
                    convention,
                )
                score = rank
                ambiguous = local_ambiguity
    return None if best is None else (*best, ambiguous or solutions > 1)


def _fallback_block(centered, minimum, gaps, spans):
    fitted = _fallback(centered, minimum, gaps, spans)
    if fitted is not None:
        return fitted
    extent = centered.max(axis=1)
    fitted = _fallback(extent[:, None] - centered, -minimum - extent, gaps, spans)
    if fitted is not None:
        d, dmin, scales, offsets, codes, _, ambiguous = fitted
        return -d, -dmin, scales, offsets, codes, False, ambiguous
    raise ValueError("observations have no supported Q4_K encoding")


def _pack(d, dmin, scales, offsets, codes):
    scales, offsets, codes = (values.astype(np.uint8) for values in (scales, offsets, codes))
    scale_bytes = np.concatenate(
        (
            scales[:, :4] | ((scales[:, 4:] >> 4) << 6),
            offsets[:, :4] | ((offsets[:, 4:] >> 4) << 6),
            (scales[:, 4:] & 15) | ((offsets[:, 4:] & 15) << 4),
        ),
        axis=1,
    )
    code_bytes = (codes[:, ::2] | (codes[:, 1::2] << 4)).reshape(-1, 128)
    return np.concatenate(
        (
            _half(d).view(np.uint8).reshape(-1, 2),
            _half(dmin).view(np.uint8).reshape(-1, 2),
            scale_bytes,
            code_bytes,
        ),
        axis=1,
    )


def _recover_chunk(values):
    # A finite half is an integer multiple of 2**-24. For ordinary Q4_K
    # blocks the affine subtraction is also exact in this lattice.
    observations = values.reshape(-1, _GROUPS, _GROUP_SIZE)
    with np.errstate(over="ignore"):
        units = np.ldexp(observations.astype(np.float64), _HALF_UNIT_EXPONENT)
    if (
        not np.isfinite(units).all()
        or np.any(np.abs(units) > _HALF_MAX_UNITS * _SCALE_LIMIT * _CODE_LIMIT)
        or np.any(units != np.trunc(units))
    ):
        raise ValueError("observations are outside the Q4_K block lattice")
    units = units.astype(np.int64)
    minimum = units.min(axis=2)
    centered = units - minimum[..., None]
    gaps = np.gcd.reduce(centered, axis=2)
    primitive = centered // np.maximum(gaps, 1)[..., None]
    spans = primitive.max(axis=2)
    zero = np.all(units == 0, axis=(1, 2))
    max_step = gaps.max(axis=1)
    max_offset = np.maximum(-minimum.min(axis=1), 0)
    d, dmin = max_step // _SCALE_LIMIT, max_offset // _SCALE_LIMIT
    scales = gaps // np.maximum(d[:, None], 1)
    valid_steps = (
        (max_step % _SCALE_LIMIT == 0)
        & _valid_half(d)
        & np.all((gaps % np.maximum(d[:, None], 1) == 0) & (spans <= _CODE_LIMIT), axis=1)
    )
    offsets, origins, valid, convention, ambiguous = _fit_offsets(minimum, gaps, spans, dmin)
    codes = primitive + origins[..., None]
    fit = valid_steps & valid & convention & (max_offset % _SCALE_LIMIT == 0) & _valid_half(dmin)

    unresolved = np.flatnonzero(valid_steps & ~fit)
    for eligible, candidates in _offset_candidates(
        minimum[unresolved], gaps[unresolved], spans[unresolved]
    ):
        active = eligible & ~fit[unresolved]
        rows = unresolved[active]
        if not rows.size:
            continue
        next_dmin = candidates[active]
        new_offsets, new_origins, valid, reference, multiple = _fit_offsets(
            minimum[rows], gaps[rows], spans[rows], next_dmin
        )
        accept = valid & reference
        rows = rows[accept]
        fit[rows] = True
        dmin[rows] = next_dmin[accept]
        offsets[rows] = new_offsets[accept]
        codes[rows] = primitive[rows] + new_origins[accept, :, None]
        ambiguous[rows] = multiple[accept]

    for index in np.flatnonzero(~fit):
        (
            d[index],
            dmin[index],
            scales[index],
            offsets[index],
            codes[index],
            fit[index],
            ambiguous[index],
        ) = _fallback_block(centered[index], minimum[index], gaps[index], spans[index])
    reconstructed = (_half(d).astype(np.float32)[:, None] * scales.astype(np.float32))[
        ..., None
    ] * codes.astype(np.float32) - (
        _half(dmin).astype(np.float32)[:, None] * offsets.astype(np.float32)
    )[..., None]
    if not np.array_equal(reconstructed, observations):
        raise ValueError("Q4_K candidate does not reproduce the observed float32 values")
    ambiguous |= (dmin == 0) | np.any(scales == 0, axis=1)
    return _pack(d, dmin, scales, offsets, codes), {
        "blocks": len(values),
        "ambiguous_blocks": int(np.count_nonzero(ambiguous)),
        "zero_blocks": int(np.count_nonzero(zero)),
        "reference_convention_blocks": int(np.count_nonzero(fit)),
    }


def recover_q4k(values):
    """Pack observed Q4_K values, preferring the reference quantizer convention.

    The ambiguity count is a lower bound. It records degenerate blocks and
    alternatives found while fitting rather than certifying unique encodings.
    """
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[-1] != _BLOCK_SIZE:
        raise ValueError("Q4_K observations must have last dimension 256")
    if values.dtype.kind != "f" or values.dtype.itemsize > 8:
        raise ValueError("Q4_K observations require real floating-point values")
    flat = values.reshape(-1, _BLOCK_SIZE)
    packed = np.empty((len(flat), _PACKED_SIZE), dtype=np.uint8)
    stats = dict(blocks=0, ambiguous_blocks=0, zero_blocks=0, reference_convention_blocks=0)
    for start in range(0, len(flat), _CHUNK_BLOCKS):
        end = start + _CHUNK_BLOCKS
        packed[start:end], counts = _recover_chunk(flat[start:end])
        for key, count in counts.items():
            stats[key] += count
    return packed.reshape(*values.shape[:-1], _PACKED_SIZE), stats
