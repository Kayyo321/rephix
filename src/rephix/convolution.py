"""Recover Conv2d parameters using impulses in a complete receptive field."""

from __future__ import annotations

from numbers import Integral


def validate_conv2d(module):
    import torch

    padding = module.padding
    if (
        module.padding_mode != "zeros"
        or not isinstance(padding, tuple)
        or len(padding) != 2
        or any(
            not isinstance(size, Integral) or isinstance(size, bool) or size < 0 for size in padding
        )
    ):
        raise ValueError(
            "Conv2d probes require numeric nonnegative padding with padding_mode='zeros'"
        )
    dimensions = (module.kernel_size, module.stride, module.dilation)
    if any(
        len(pair) != 2
        or any(
            not isinstance(size, Integral) or isinstance(size, bool) or size < 1 for size in pair
        )
        for pair in dimensions
    ):
        raise ValueError("Conv2d probes require positive kernel size, stride, and dilation")
    if (
        module.groups < 1
        or min(module.in_channels, module.out_channels) < 1
        or module.in_channels % module.groups
        or module.out_channels % module.groups
    ):
        raise ValueError("Conv2d probes require positive channels divisible by groups")
    parameters = (module.weight,) if module.bias is None else (module.weight, module.bias)
    if any(param.layout != torch.strided or not param.is_floating_point() for param in parameters):
        raise ValueError("Conv2d probes require dense floating-point parameters")
    weight_shape = (module.out_channels, module.in_channels // module.groups, *module.kernel_size)
    if module.weight.shape != weight_shape or (
        module.bias is not None and module.bias.shape != (module.out_channels,)
    ):
        raise ValueError("Conv2d parameter shape disagrees with its configuration")


def _probe_axis(padding, stride, dilation, kernel_size):
    # Use the first output whose receptive field starts inside the input.
    output_index = (padding + stride - 1) // stride
    start = output_index * stride - padding
    extent = dilation * (kernel_size - 1) + 1
    input_size = start + extent
    output_size = (input_size + 2 * padding - extent) // stride + 1
    return output_index, start, input_size, output_size


def recover_conv2d(torch, module, pending, read, batch_size):
    validate_conv2d(module)
    weight = module.weight
    bias = module.bias
    kernel_h, kernel_w = module.kernel_size
    out_h, start_h, input_h, output_h = _probe_axis(
        module.padding[0], module.stride[0], module.dilation[0], kernel_h
    )
    out_w, start_w, input_w, output_w = _probe_axis(
        module.padding[1], module.stride[1], module.dilation[1], kernel_w
    )
    input_shape = (module.in_channels, input_h, input_w)
    output_shape = (module.out_channels, output_h, output_w)
    if bias is not None:
        baseline = read(weight.new_zeros((1, *input_shape)), (1, *output_shape))[0, :, out_h, out_w]
        baseline = baseline.to(dtype=torch.float64, copy=True)
        if "bias" in pending:
            bias.copy_(baseline)
    if "weight" not in pending:
        return

    # A power of two reduces cancellation when subtracting the observed bias.
    amplitude = 1 if bias is None else 32
    group_channels = weight.shape[1]
    group_offsets = torch.arange(module.groups, device=weight.device) * group_channels
    kernel_area = kernel_h * kernel_w
    coordinates = group_channels * kernel_area
    for start in range(0, coordinates, batch_size):
        end = min(start + batch_size, coordinates)
        count = end - start
        rows = torch.arange(count, device=weight.device)
        indices = rows + start
        channels = indices // kernel_area
        kernel_rows = indices % kernel_area // kernel_w
        kernel_cols = indices % kernel_w

        inputs = weight.new_zeros((count, *input_shape))
        # Outputs from each group observe only the impulse in that group.
        inputs[
            rows[:, None],
            channels[:, None] + group_offsets,
            (start_h + kernel_rows * module.dilation[0])[:, None],
            (start_w + kernel_cols * module.dilation[1])[:, None],
        ] = amplitude
        values = read(inputs, (count, *output_shape))[:, :, out_h, out_w]
        if bias is not None:
            values = (values.double() - baseline) / amplitude
        weight[:, channels, kernel_rows, kernel_cols] = values.T.to(weight.dtype)
