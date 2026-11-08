"""Recover stored weight-normalization coordinates through native primitives."""

from .convolution import recover_conv1d
from .native_operation import ProbeResult, capture_native_operation

WEIGHT_NORM_PARAMETRIZATIONS = "torch.nn.utils.parametrize.ParametrizationList"
WEIGHT_NORM_CONV1D = "torch.nn.utils.parametrize.ParametrizedConv1d"


def _weight_norm_spec(module):
    import torch

    if type(module).__name__ == "ParametrizationList":
        if len(module) != 1 or type(module[0]) is not torch.nn.utils.parametrizations._WeightNorm:
            raise ValueError("expected one native weight-normalization parametrization")
        return module.original0, module.original1, module[0].dim, ("original0", "original1")
    hooks = [
        hook
        for hook in module._forward_pre_hooks.values()
        if type(hook).__module__ == "torch.nn.utils.weight_norm"
        and type(hook).__name__ == "WeightNorm"
        and hook.name == "weight"
    ]
    if type(module) is torch.nn.Conv1d and len(hooks) == 1:
        return module.weight_g, module.weight_v, hooks[0].dim, ("weight_g", "weight_v")
    raise ValueError("expected native weight-normalized Conv1d parameters")


def validate_weight_norm(module, pending):
    scale, direction, dim, names = _weight_norm_spec(module)
    allowed = set(names)
    if type(module).__name__ != "ParametrizationList" and module.bias is not None:
        allowed.add("bias")
    if (
        pending.keys() != allowed
        or direction.ndim != 3
        or min(direction.shape) < 1
        or dim not in (-1, 0, 1, 2)
        or scale.dtype != direction.dtype
    ):
        raise ValueError("unsupported native weight-normalization parameter layout")
    expected = [1] * direction.ndim
    if dim >= 0:
        expected[dim] = direction.shape[dim]
    if scale.numel() != (direction.shape[dim] if dim >= 0 else 1):
        raise ValueError("weight-normalization scale shape disagrees with its dimension")
    if dim >= 0 and tuple(scale.shape) != tuple(expected):
        raise ValueError("weight-normalization scale must retain normalized dimensions")


def weight_norm_probe_dtype(torch, module):
    """Use a native FP32 primitive for low-precision stored parameters."""
    dtype = _weight_norm_spec(module)[1].dtype
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _native_weight_norm(torch, direction, scale, dim):
    direction, scale = direction.contiguous(), scale.contiguous()
    if dim in (0, direction.ndim - 1):
        return torch._weight_norm_interface(direction, scale, dim)
    return torch._weight_norm(direction, scale, dim), torch.norm_except_dim(direction, 2, dim)


def recover_weight_norm(torch, module, pending, read, batch_size):
    """Recover signed scale and direction, including when the stored scale is zero."""
    scale, direction, dim, _ = _weight_norm_spec(module)
    dtype = weight_norm_probe_dtype(torch, module)
    chosen = torch.zeros(direction.shape, dtype=dtype)
    index = [0] * direction.ndim
    if dim >= 0:
        index[dim] = slice(None)
    chosen[tuple(index)] = 1
    scale.copy_(read(chosen, scale.shape, operation="wn_scale"))
    norms = read(torch.zeros((1,), dtype=dtype), scale.shape, operation="wn_norms")
    if bool((norms <= 0).any()):
        raise ValueError("native weight normalization has a zero direction norm")
    direction.copy_(read(norms, direction.shape, operation="wn_direction"))
    if "bias" in pending:

        def read_bias(inputs, shape):
            return read(inputs.to(dtype), shape, operation="wn_bias")

        recover_conv1d(torch, module, {"bias": pending["bias"]}, read_bias, batch_size)


def observe_weight_norm(torch, module, inputs, operation):
    """Expose selected native primitive results with chosen direction or scale.

    FP16 and BF16 coordinates are exactly representable in FP32. The controlled
    primitive uses FP32 for these probes so its observed norm remains a valid
    scale argument without an intervening low-precision cast.
    """
    scale, direction, dim, _ = _weight_norm_spec(module)
    if operation == "wn_bias":
        return module(inputs.to(direction.dtype)).to(inputs.dtype)
    if operation not in {"wn_scale", "wn_norms", "wn_direction"}:
        return module(inputs)

    def intercept(func, args, kwargs):
        if func is torch._weight_norm:
            native_direction, native_scale, native_dim = args
            if native_dim != dim:
                raise ValueError("native weight-normalization dimension changed")
            native_direction = native_direction.to(dtype=inputs.dtype)
            native_scale = native_scale.to(dtype=inputs.dtype)
            if operation == "wn_scale":
                values, _ = _native_weight_norm(torch, inputs, native_scale, dim)
                index = [0] * values.ndim
                if dim >= 0:
                    index[dim] = slice(None)
                result = values[tuple(index)].reshape(scale.shape)
            elif operation == "wn_norms":
                _, result = _native_weight_norm(torch, native_direction, native_scale, dim)
            else:
                result, _ = _native_weight_norm(torch, native_direction, inputs, dim)
            raise ProbeResult(result)
        return func(*args, **kwargs)

    if type(module).__name__ == "ParametrizationList":
        call = module
    else:
        # The legacy pre-forward hook recomputes the effective kernel.
        shape = (1, module.in_channels, module.kernel_size[0])
        pixels = direction.new_zeros(shape)

        def call():
            return module(pixels)

    return capture_native_operation(call, intercept)
