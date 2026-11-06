"""Recover native T5-family position bias through the attention operation."""

from __future__ import annotations

from dataclasses import dataclass

RELATIVE_BIAS_ATTENTION = {
    "transformers.models.t5.modeling_t5.T5Attention",
    "transformers.models.mt5.modeling_mt5.MT5Attention",
    "transformers.models.umt5.modeling_umt5.UMT5Attention",
}

_MAX_PROBE_KEYS = 1 << 16
_MAX_PROBE_OUTPUT_SCALARS = 1 << 22


@dataclass(frozen=True)
class RelativeBiasSpec:
    key_length: int
    query_positions: tuple[int, ...]


def relative_bias_spec(attention):
    import torch

    cls = type(attention)
    if (
        f"{cls.__module__}.{cls.__name__}" not in RELATIVE_BIAS_ATTENTION
        or not attention.has_relative_attention_bias
    ):
        raise ValueError(
            "attention-bias probes require native T5-family attention with a bias table"
        )

    buckets = attention.relative_attention_num_buckets
    distance = attention.relative_attention_max_distance
    table = attention.relative_attention_bias
    if any(type(value) is not int for value in (buckets, distance)):
        raise ValueError("relative attention bucket count and distance must be integers")
    per_direction = buckets if attention.is_decoder else buckets // 2
    if per_direction < 2 or distance <= per_direction // 2:
        raise ValueError("relative attention bucket configuration has invalid logarithmic bins")
    if (
        type(table) is not torch.nn.Embedding
        or table.max_norm is not None
        or table.num_embeddings != buckets
        or table.weight.shape != (buckets, table.embedding_dim)
        or table.embedding_dim < 1
    ):
        raise ValueError("relative attention bias table disagrees with its configuration")

    # One extra offset absorbs rounding at the logarithmic clipping point.
    outer = distance + 1
    positions = (outer,) if attention.is_decoder else (outer, 0)
    key_length = outer + 1
    if (
        key_length > _MAX_PROBE_KEYS
        or key_length * len(positions) * table.embedding_dim > _MAX_PROBE_OUTPUT_SCALARS
    ):
        raise ValueError("relative attention bias exceeds the supported probe size")
    return RelativeBiasSpec(key_length=key_length, query_positions=positions)


def recover_relative_bias(torch, attention, read, batch_size, *, bucket_device="cpu"):
    """Recover the rows seen by compute_bias and zero its unreachable rows."""
    spec = relative_bias_spec(attention)
    weight = attention.relative_attention_bias.weight
    bucket_count, heads = weight.shape
    positions = torch.tensor(spec.query_positions, dtype=torch.long, device=weight.device)
    keys = torch.arange(spec.key_length, device=weight.device)
    distances = (keys[None, :] - positions[:, None]).to(bucket_device)

    # Use the oracle's backend for logarithmic bucket boundaries. A difference
    # in rounding at an integer offset can move it into the adjacent bucket.
    bucket_map = attention._relative_position_bucket
    if type(attention).__name__ == "UMT5Attention":
        buckets = bucket_map(distances)
    else:
        buckets = bucket_map(
            distances,
            bidirectional=not attention.is_decoder,
            num_buckets=bucket_count,
            max_distance=attention.relative_attention_max_distance,
        )
    buckets = buckets.to(weight.device)
    if torch.any((buckets < 0) | (buckets >= bucket_count)):
        raise ValueError("relative attention probe has out-of-range bucket indices")
    per_direction = bucket_count if attention.is_decoder else bucket_count // 2
    if buckets[0, 0] != per_direction - 1 or (
        not attention.is_decoder and buckets[1, -1] != 2 * per_direction - 1
    ):
        raise ValueError("relative attention probe does not reach the outermost bucket")

    blocks = []
    for batch in positions.split(batch_size):
        observed = read(batch, (1, heads, len(batch), spec.key_length))
        blocks.append(observed[0].permute(1, 2, 0).reshape(-1, heads))
    observations = torch.cat(blocks, dim=0)
    bucket_indices = buckets.flatten()
    ordered_indices = bucket_indices.argsort(stable=True)
    reached, counts = bucket_indices[ordered_indices].unique_consecutive(return_counts=True)
    first = counts.cumsum(0) - counts
    recovered = torch.zeros_like(weight)
    recovered[reached] = observations[ordered_indices[first]]
    if not torch.equal(observations, recovered[bucket_indices]):
        raise ValueError("relative attention outputs disagree within an expected bucket")
    weight.copy_(recovered)

    unreachable = sorted(set(range(bucket_count)) - set(reached.tolist()))
    return {
        "unreachable_buckets": unreachable,
        "representative": "zero_unreachable_buckets",
        "affected_stored_elements": len(unreachable) * heads,
    }
