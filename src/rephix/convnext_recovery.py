"""Recover ConvNeXT channel scales and native normalization parameters."""

from .native_operation import ProbeResult, capture_native_operation

CONVNEXT_LAYER_NORMS = {
    "transformers.models.convnext.modeling_convnext.ConvNextLayerNorm",
    "transformers.models.convnextv2.modeling_convnextv2.ConvNextV2LayerNorm",
}
CONVNEXT_LAYER_SCALE = "transformers.models.convnext.modeling_convnext.ConvNextLayer"
CONVNEXT_GRN = "transformers.models.convnextv2.modeling_convnextv2.ConvNextV2GRN"


def validate_convnext_norm(module, pending):
    if (
        pending.keys() != {"weight", "bias"}
        or module.weight.ndim != 1
        or module.weight.numel() < 2
        or module.weight.shape != module.bias.shape
        or module.weight.shape != module.normalized_shape
        or module.data_format not in {"channels_first", "channels_last"}
    ):
        raise ValueError("unsupported ConvNeXT LayerNorm layout")


def recover_convnext_norm(torch, module, pending, read):
    weight, bias = pending["weight"], pending["bias"]
    channels = weight.numel()
    channels_first = module.data_format == "channels_first"
    shape = (1, channels, 1, 1) if channels_first else (1, 1, 1, channels)
    amplitude = 2**18 if weight.dtype == torch.float64 else 32
    inputs = weight.new_full(shape, amplitude)
    inputs.flatten()[1::2] = -amplitude
    if channels_first and not isinstance(module, torch.nn.LayerNorm):
        # Older native channels-first LayerNorm normalizes in FP32 explicitly.
        centered = inputs.float() - inputs.float().mean(dim=1, keepdim=True)
        variance = centered.square().mean(dim=1, keepdim=True)
        unit = (centered / (variance + module.eps).sqrt()).to(weight.dtype)
    else:
        unit = torch.nn.functional.layer_norm(
            inputs.reshape(1, channels), module.normalized_shape, eps=module.eps
        )
    baseline = read(weight.new_zeros(shape), shape)
    response = read(inputs, shape)
    bias.copy_(baseline.flatten())
    weight.copy_((response.double() - baseline).flatten() / unit.flatten())


def validate_convnext_scale(module, pending):
    if pending.keys() != {"layer_scale_parameter"} or module.layer_scale_parameter.shape != (
        module.pwconv2.out_features,
    ):
        raise ValueError("unsupported ConvNeXT LayerScale layout")


def recover_convnext_scale(torch, module, pending, read):
    scale = pending["layer_scale_parameter"]
    shape = (1, scale.numel(), 1, 1)
    scale.copy_(read(scale.new_zeros(shape), shape, operation="layer_scale").flatten())


def observe_convnext_scale(module, inputs):
    # Run the native block with a unit pre-scale response and zero residual.
    handle = module.pwconv2.register_forward_hook(
        lambda _module, _args, output: output.new_ones(output.shape)
    )
    try:
        return module(inputs)
    finally:
        handle.remove()


def validate_convnext_grn(module, pending):
    weight = module.weight
    if (
        pending.keys() != {"weight", "bias"}
        or weight.ndim != 4
        or weight.shape[:3] != (1, 1, 1)
        or weight.shape != module.bias.shape
    ):
        raise ValueError("unsupported ConvNeXT global response normalization layout")


def recover_convnext_grn(torch, module, pending, read):
    weight, bias = pending["weight"], pending["bias"]
    shape = weight.shape
    # Equal channel magnitudes have unit response after rounding the 1e-6
    # denominator offset at these powers of two.
    amplitude = 2**34 if weight.dtype == torch.float64 else 32
    bias.copy_(read(weight.new_zeros(shape), shape))
    scaled = read(weight.new_full(shape, amplitude), shape, operation="grn_scale")
    weight.copy_(scaled / amplitude)


def observe_convnext_grn_scale(module, inputs):
    """Observe the native weighted response before the bias and residual add."""
    from torch import Tensor

    def intercept(func, args, kwargs):
        result = func(*args, **kwargs)
        if func is Tensor.mul and args[0] is module.weight:
            raise ProbeResult(result)
        return result

    return capture_native_operation(lambda: module(inputs), intercept)
