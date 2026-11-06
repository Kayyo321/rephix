"""Recover convolution parameters using impulses in a complete receptive field."""

from __future__ import annotations

from math import prod
from numbers import Integral


def validate_conv1d(module):
    _validate_convolution(module, 1)


def validate_conv2d(module):
    _validate_convolution(module, 2)


def validate_conv3d(module):
    _validate_convolution(module, 3)


def _numeric_padding(module):
    if module.padding == "valid":
        return (0,) * len(module.kernel_size)
    return module.padding


def _validate_convolution(module, rank):
    import torch

    operator = f"Conv{rank}d"
    padding = _numeric_padding(module)
    if (
        module.padding_mode != "zeros"
        or not isinstance(padding, tuple)
        or len(padding) != rank
        or any(
            not isinstance(size, Integral) or isinstance(size, bool) or size < 0 for size in padding
        )
    ):
        raise ValueError(
            f"{operator} probes require numeric nonnegative padding with padding_mode='zeros'"
        )
    dimensions = (module.kernel_size, module.stride, module.dilation)
    if any(
        len(pair) != rank
        or any(
            not isinstance(size, Integral) or isinstance(size, bool) or size < 1 for size in pair
        )
        for pair in dimensions
    ):
        raise ValueError(f"{operator} probes require positive kernel size, stride, and dilation")
    if (
        module.groups < 1
        or min(module.in_channels, module.out_channels) < 1
        or module.in_channels % module.groups
        or module.out_channels % module.groups
    ):
        raise ValueError(f"{operator} probes require positive channels divisible by groups")
    parameters = (module.weight,) if module.bias is None else (module.weight, module.bias)
    if any(param.layout != torch.strided or not param.is_floating_point() for param in parameters):
        raise ValueError(f"{operator} probes require dense floating-point parameters")
    weight_shape = (module.out_channels, module.in_channels // module.groups, *module.kernel_size)
    if module.weight.shape != weight_shape or (
        module.bias is not None and module.bias.shape != (module.out_channels,)
    ):
        raise ValueError(f"{operator} parameter shape disagrees with its configuration")


def _probe_axis(padding, stride, dilation, kernel_size):
    # Use the first output whose receptive field starts inside the input.
    output_index = (padding + stride - 1) // stride
    start = output_index * stride - padding
    extent = dilation * (kernel_size - 1) + 1
    input_size = start + extent
    output_size = (input_size + 2 * padding - extent) // stride + 1
    return output_index, start, input_size, output_size


def recover_conv1d(torch, module, pending, read, batch_size):
    validate_conv1d(module)
    _recover_convolution(torch, module, pending, read, batch_size)


def recover_conv2d(torch, module, pending, read, batch_size):
    validate_conv2d(module)
    _recover_convolution(torch, module, pending, read, batch_size)


def recover_conv3d(torch, module, pending, read, batch_size):
    validate_conv3d(module)
    _recover_convolution(torch, module, pending, read, batch_size)


def _recover_convolution(torch, module, pending, read, batch_size):
    weight = module.weight
    bias = module.bias
    probe_axes = (
        _probe_axis(padding, stride, dilation, kernel_size)
        for padding, stride, dilation, kernel_size in zip(
            _numeric_padding(module), module.stride, module.dilation, module.kernel_size
        )
    )
    output_indices, field_starts, input_sizes, output_sizes = zip(*probe_axes)
    input_shape = (module.in_channels, *input_sizes)
    output_shape = (module.out_channels, *output_sizes)
    output_slice = (slice(None), slice(None), *output_indices)
    if bias is not None:
        baseline = read(weight.new_zeros((1, *input_shape)), (1, *output_shape))[output_slice][0]
        baseline = baseline.to(dtype=torch.float64, copy=True)
        if "bias" in pending:
            bias.copy_(baseline)
    if "weight" not in pending:
        return

    # A power of two reduces cancellation when subtracting the observed bias.
    amplitude = 1 if bias is None else 32
    group_channels = weight.shape[1]
    group_offsets = torch.arange(module.groups, device=weight.device) * group_channels
    kernel_volume = prod(module.kernel_size)
    coordinates = group_channels * kernel_volume
    for start in range(0, coordinates, batch_size):
        end = min(start + batch_size, coordinates)
        count = end - start
        rows = torch.arange(count, device=weight.device)
        indices = rows + start
        channels = indices // kernel_volume
        taps = torch.unravel_index(indices % kernel_volume, module.kernel_size)
        spatial_indices = tuple(
            (field_start + tap * dilation)[:, None]
            for field_start, tap, dilation in zip(field_starts, taps, module.dilation)
        )

        inputs = weight.new_zeros((count, *input_shape))
        # Outputs from each group observe only the impulse in that group.
        input_indices = (
            rows[:, None],
            channels[:, None] + group_offsets,
            *spatial_indices,
        )
        inputs[input_indices] = amplitude
        values = read(inputs, (count, *output_shape))[output_slice]
        if bias is not None:
            values = (values.double() - baseline) / amplitude
        weight[(slice(None), channels, *taps)] = values.T.to(weight.dtype)
