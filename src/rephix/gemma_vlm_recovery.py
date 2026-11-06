"""Recover the native Gemma 3 image projection from chosen normalized inputs."""

GEMMA3_PROJECTOR = "transformers.models.gemma3.modeling_gemma3.Gemma3MultiModalProjector"


def validate_gemma3_projector(module, pending):
    weight = module.mm_input_projection_weight
    if (
        pending.keys() != {"mm_input_projection_weight"}
        or weight.ndim != 2
        or min(weight.shape) < 1
        or module.mm_soft_emb_norm.weight.shape != (weight.shape[0],)
        or module.patches_per_image < 1
    ):
        raise ValueError("unsupported Gemma 3 image projector layout")


def recover_gemma3_projector(torch, module, read, batch_size):
    """Recover the bias-free image projection one input coordinate at a time."""
    weight = module.mm_input_projection_weight
    in_features, out_features = weight.shape
    for start in range(0, in_features, batch_size):
        stop = min(start + batch_size, in_features)
        rows = torch.arange(stop - start, device=weight.device)
        inputs = weight.new_zeros((stop - start, in_features))
        inputs[rows, rows + start] = 1
        weight[start:stop].copy_(read(inputs, (stop - start, out_features)))


def observe_gemma3_projector(module, inputs):
    """Supply chosen features after normalization and run the native projection."""
    pixels = inputs.new_zeros((1, module.patches_per_image**2, inputs.shape[-1]))
    handle = module.mm_soft_emb_norm.register_forward_hook(
        lambda _module, _arguments, _output: inputs
    )
    try:
        return module(pixels)
    finally:
        handle.remove()
