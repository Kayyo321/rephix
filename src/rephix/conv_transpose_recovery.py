"""Recover native transposed convolutions from isolated input impulses."""

from numbers import Integral

CONV_TRANSPOSE_TYPES = {
    "torch.nn.modules.conv.ConvTranspose1d",
    "torch.nn.modules.conv.ConvTranspose2d",
    "torch.nn.modules.conv.ConvTranspose3d",
}
PARAMETRIZED_CONV_TRANSPOSE1D = "torch.nn.utils.parametrize.ParametrizedConvTranspose1d"


def validate_conv_transpose(module):
    import torch

    qualified = f"{type(module).__module__}.{type(module).__name__}"
    if qualified not in CONV_TRANSPOSE_TYPES | {PARAMETRIZED_CONV_TRANSPOSE1D}:
        raise ValueError("transposed convolution probes require a supported native operator")
    rank = len(module.kernel_size)
    if module.padding_mode != "zeros" or rank not in (1, 2, 3):
        raise ValueError("transposed convolution probes require zero padding")
    for name, minimum in (
        ("kernel_size", 1),
        ("stride", 1),
        ("dilation", 1),
        ("padding", 0),
        ("output_padding", 0),
    ):
        dimensions = getattr(module, name)
        if (
            not isinstance(dimensions, tuple)
            or len(dimensions) != rank
            or any(
                not isinstance(value, Integral) or isinstance(value, bool) or value < minimum
                for value in dimensions
            )
        ):
            raise ValueError(f"unsupported transposed convolution {name}")
    if any(
        extra >= max(stride, dilation)
        for extra, stride, dilation in zip(module.output_padding, module.stride, module.dilation)
    ):
        raise ValueError(
            "transposed convolution output_padding must be smaller than stride or dilation"
        )
    if (
        min(module.in_channels, module.out_channels, module.groups) < 1
        or module.in_channels % module.groups
        or module.out_channels % module.groups
    ):
        raise ValueError("transposed convolution channels must be divisible by positive groups")
    shape = (module.in_channels, module.out_channels // module.groups, *module.kernel_size)
    if module.weight.shape != shape or (
        module.bias is not None and module.bias.shape != (module.out_channels,)
    ):
        raise ValueError("transposed convolution parameter shape disagrees with its configuration")
    parameters = (module.weight,) if module.bias is None else (module.weight, module.bias)
    if any(not value.is_floating_point() or value.layout != torch.strided for value in parameters):
        raise ValueError("transposed convolution probes require dense floating-point parameters")


def recover_conv_transpose(torch, module, pending, read, batch_size):
    """Expose a complete output kernel for each chosen input channel."""
    validate_conv_transpose(module)
    weight, bias = module.weight, module.bias
    rank = len(module.kernel_size)
    # The chosen input point leaves a complete, uncropped kernel in the output.
    anchors = tuple(
        (padding + stride - 1) // stride for padding, stride in zip(module.padding, module.stride)
    )
    input_sizes = tuple(2 * anchor + 1 for anchor in anchors)
    output_starts = tuple(
        anchor * stride - padding
        for anchor, stride, padding in zip(anchors, module.stride, module.padding)
    )
    output_sizes = tuple(
        (size - 1) * stride - 2 * padding + dilation * (kernel - 1) + extra + 1
        for size, stride, padding, dilation, kernel, extra in zip(
            input_sizes,
            module.stride,
            module.padding,
            module.dilation,
            module.kernel_size,
            module.output_padding,
        )
    )
    input_shape = (module.in_channels, *input_sizes)
    output_shape = (module.out_channels, *output_sizes)
    if bias is not None:
        zero = read(weight.new_zeros((1, *input_shape)), (1, *output_shape))
        baseline = zero[(0, slice(None), *output_starts)].double()
        if "bias" in pending:
            pending["bias"].copy_(baseline)
    if "weight" not in pending:
        return
    amplitude = 1 if bias is None else 32
    taps = torch.meshgrid(
        *(
            torch.arange(kernel, device=weight.device) * dilation + start
            for kernel, dilation, start in zip(module.kernel_size, module.dilation, output_starts)
        ),
        indexing="ij",
    )
    channels_per_group = module.in_channels // module.groups
    offsets = torch.arange(module.groups, device=weight.device) * channels_per_group
    for start in range(0, channels_per_group, batch_size):
        stop = min(start + batch_size, channels_per_group)
        rows = torch.arange(stop - start, device=weight.device)
        channels = torch.arange(start, stop, device=weight.device)
        chosen_channels = channels[:, None] + offsets
        inputs = weight.new_zeros((stop - start, *input_shape))
        inputs[(rows[:, None], chosen_channels, *anchors)] = amplitude
        values = read(inputs, (stop - start, *output_shape))[(slice(None), slice(None), *taps)]
        if bias is not None:
            values = (values.double() - baseline.reshape(1, -1, *((1,) * rank))) / amplitude
        values = values.reshape(
            stop - start, module.groups, module.out_channels // module.groups, *module.kernel_size
        )
        weight[chosen_channels] = values.to(weight.dtype)
