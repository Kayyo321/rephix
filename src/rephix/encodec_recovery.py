"""Chosen codebook lookups and explicit inactive registered-state observations."""

ENCODEC_CODEBOOK = "transformers.models.encodec.modeling_encodec.EncodecEuclideanCodebook"
ENCODEC_INACTIVE_BUFFERS = frozenset({"inited", "cluster_size", "embed_avg"})
ENCODEC_BUFFER_OPERATIONS = {
    "embed": "codebook_embed",
    **{name: f"inactive_state/{name}" for name in sorted(ENCODEC_INACTIVE_BUFFERS)},
}


def validate_encodec_codebook(module):
    import torch

    cls = type(module)
    if f"{cls.__module__}.{cls.__name__}" != ENCODEC_CODEBOOK:
        raise ValueError("codebook probes require a native Encodec Euclidean codebook")
    buffers = dict(module.named_buffers(recurse=False))
    if buffers.keys() != ENCODEC_BUFFER_OPERATIONS.keys():
        raise ValueError("unsupported native Encodec codebook state")
    embed = buffers["embed"]
    if (
        embed.ndim != 2
        or min(embed.shape) < 1
        or embed.shape[0] != module.codebook_size
        or buffers["embed_avg"].shape != embed.shape
        or buffers["cluster_size"].shape != (module.codebook_size,)
        or buffers["inited"].shape != (1,)
    ):
        raise ValueError("unsupported native Encodec codebook buffer shape")
    if any(
        value.layout != torch.strided or not value.is_floating_point() for value in buffers.values()
    ):
        raise ValueError("Encodec codebook probes require dense floating-point state")


def recover_encodec_buffer(torch, module, pending, read, batch_size, operation):
    """Recover selected state from chosen ID lookups, including explicit state reads."""
    validate_encodec_codebook(module)
    if pending.keys() != {"bias"} or operation not in ENCODEC_BUFFER_OPERATIONS.values():
        raise ValueError("unknown native Encodec buffer observation")
    value = pending["bias"]
    rows = value.reshape(value.shape[0], -1)
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        indices = torch.arange(start, stop, dtype=torch.long)
        response = read(indices, (stop - start, rows.shape[1]), operation=operation)
        rows[start:stop].copy_(response)


def observe_encodec_buffer(torch, module, inputs, operation):
    """Return native decode rows or an explicitly selected inactive state lookup.

    The inactive buffers never participate in native Transformers Encodec
    inference. Their registered-state interface is an additional access
    requirement and must be explicitly enabled by the recovery caller.
    """
    validate_encodec_codebook(module)
    if operation == "codebook_embed":
        return module.decode(inputs)
    prefix, separator, name = operation.partition("/")
    if prefix != "inactive_state" or not separator or name not in ENCODEC_INACTIVE_BUFFERS:
        raise ValueError("unknown Encodec inactive-state observation")
    values = getattr(module, name)
    table = values.reshape(values.shape[0], -1)
    return torch.nn.functional.embedding(inputs, table)
