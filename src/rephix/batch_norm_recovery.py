"""Recover native BatchNorm affine parameters and persistent statistics."""

import math
from unittest.mock import patch

from .native_operation import ProbeResult, capture_native_operation

BATCH_NORM_TYPES = {
    "torch.nn.modules.batchnorm.BatchNorm1d",
    "torch.nn.modules.batchnorm.BatchNorm2d",
    "torch.nn.modules.batchnorm.BatchNorm3d",
}
BATCH_NORM_BUFFER_OPERATIONS = {
    "running_mean": "batch_norm_running_mean",
    "running_var": "batch_norm_running_var",
    "num_batches_tracked": "batch_norm_counter",
}


def _shape(module):
    rank = {"BatchNorm1d": 1, "BatchNorm2d": 2, "BatchNorm3d": 3}[type(module).__name__]
    return (1, module.num_features, *((1,) * rank))


def validate_batch_norm(module):
    import torch

    qualified = f"{type(module).__module__}.{type(module).__name__}"
    if qualified not in BATCH_NORM_TYPES or module.num_features < 1:
        raise ValueError("BatchNorm probes require a native positive-channel layer")
    if not math.isfinite(module.eps) or module.eps <= 0:
        raise ValueError("BatchNorm probes require finite positive epsilon")
    if module.affine:
        for parameter in (module.weight, module.bias):
            if (
                parameter is None
                or parameter.shape != (module.num_features,)
                or not parameter.is_floating_point()
                or parameter.layout != torch.strided
            ):
                raise ValueError("unsupported BatchNorm affine parameter layout")
    elif module.weight is not None or module.bias is not None:
        raise ValueError("BatchNorm affine state disagrees with its configuration")
    if module.track_running_stats:
        mean, variance, counter = (
            module.running_mean,
            module.running_var,
            module.num_batches_tracked,
        )
        if (
            mean is None
            or variance is None
            or mean.shape != (module.num_features,)
            or variance.shape != mean.shape
            or not mean.is_floating_point()
            or variance.dtype != mean.dtype
            or counter is None
            or counter.shape != ()
            or counter.dtype != torch.int64
        ):
            raise ValueError("unsupported BatchNorm persistent statistic layout")
    elif any(
        value is not None
        for value in (module.running_mean, module.running_var, module.num_batches_tracked)
    ):
        raise ValueError("BatchNorm persistent state disagrees with its configuration")


def recover_batch_norm(torch, module, pending, read):
    """Recover the affine values through controlled native evaluation calls."""
    validate_batch_norm(module)
    if not module.affine or not pending.keys() <= {"weight", "bias"}:
        raise ValueError("BatchNorm affine probes require affine weight or bias")
    inputs = torch.zeros(_shape(module), dtype=torch.float64)
    for name in ("weight", "bias"):
        if name in pending:
            values = read(inputs, tuple(inputs.shape), operation=f"batch_norm_{name}")
            pending[name].copy_(values.reshape(module.num_features))


def recover_batch_norm_buffer(torch, module, pending, read, operation):
    """Recover one persistent statistic without evaluating a state-changing update."""
    validate_batch_norm(module)
    if not module.track_running_stats or pending.keys() != {"bias"}:
        raise ValueError("BatchNorm buffer probes require one tracked statistic")
    value = pending["bias"]
    if operation == "batch_norm_counter":
        value.copy_(read(torch.zeros(1, dtype=torch.int64), (), operation=operation))
        return
    if operation not in {"batch_norm_running_mean", "batch_norm_running_var"}:
        raise ValueError("unknown BatchNorm buffer observation")
    inputs = torch.zeros(_shape(module), dtype=torch.float64)
    observed = read(inputs, tuple(inputs.shape), operation=operation).reshape(module.num_features)
    if operation == "batch_norm_running_var":
        # A tiny chosen epsilon gives a finite response even at zero variance.
        observed = observed.reciprocal().square() - torch.finfo(torch.float64).tiny
        observed = observed.clamp_min(0)
        if not bool(torch.isfinite(observed).all()):
            raise ValueError("BatchNorm variance cannot be reconstructed at probe precision")
    value.copy_(observed)


def observe_batch_norm(torch, module, inputs, operation):
    """Drive a native BatchNorm call with chosen primitive arguments."""
    from torch.nn import functional as F

    validate_batch_norm(module)
    if operation == "batch_norm_counter":
        if not module.track_running_stats:
            raise ValueError("BatchNorm counter observation requires tracked statistics")

        def intercept_counter(func, args, kwargs):
            if func is torch.Tensor.add_ and args[0] is module.num_batches_tracked:
                # Choose a zero increment and a nonmutating native add. The
                # normal training forward has reached its counter update.
                raise ProbeResult(torch.add(args[0], inputs[0]))
            return func(*args, **kwargs)

        reference = module.running_mean
        pixels = torch.zeros(_shape(module), device=inputs.device, dtype=reference.dtype)
        with patch.object(module, "training", True):
            return capture_native_operation(lambda: module(pixels), intercept_counter)

    supported = {
        "batch_norm_weight",
        "batch_norm_bias",
        "batch_norm_running_mean",
        "batch_norm_running_var",
    }
    if operation not in supported:
        raise ValueError("unknown BatchNorm observation")

    def intercept(func, args, kwargs):
        if func is not F.batch_norm:
            return func(*args, **kwargs)
        original_mean, original_variance = args[1:3]
        original_weight = args[3] if len(args) > 3 else kwargs.get("weight")
        original_bias = args[4] if len(args) > 4 else kwargs.get("bias")
        shape = (module.num_features,)
        mean, variance = inputs.new_zeros(shape), inputs.new_ones(shape)
        weight, bias = inputs.new_ones(shape), inputs.new_zeros(shape)
        pixels, epsilon = torch.zeros_like(inputs), torch.finfo(torch.float64).tiny
        if operation == "batch_norm_weight":
            if original_weight is None:
                raise ValueError("BatchNorm source has no affine scale")
            weight, pixels = original_weight.to(inputs.dtype), torch.ones_like(inputs)
        elif operation == "batch_norm_bias":
            if original_bias is None:
                raise ValueError("BatchNorm source has no affine bias")
            bias = original_bias.to(inputs.dtype)
        elif operation == "batch_norm_running_mean":
            if original_mean is None:
                raise ValueError("BatchNorm source has no running mean")
            mean, weight = original_mean.to(inputs.dtype), -weight
        else:
            if original_variance is None:
                raise ValueError("BatchNorm source has no running variance")
            variance, pixels = original_variance.to(inputs.dtype), torch.ones_like(inputs)
            epsilon = torch.finfo(torch.float64).tiny
        result = F.batch_norm(pixels, mean, variance, weight, bias, training=False, eps=epsilon)
        raise ProbeResult(result)

    with patch.object(module, "training", False):
        return capture_native_operation(lambda: module(inputs), intercept)
