"""Recover the affine parameters of a native GroupNorm layer."""

import math


def validate_group_norm(module):
    if not module.affine or module.weight is None or module.bias is None:
        raise ValueError("GroupNorm probes require an affine layer")
    if not math.isfinite(module.eps) or module.eps <= 0:
        raise ValueError("GroupNorm probes require finite positive epsilon")


def recover_group_norm(torch, module, pending, read):
    shape = (1, module.num_channels, 2)
    inputs = module.weight.new_zeros(shape)
    if "bias" in pending:
        pending["bias"].copy_(read(inputs, shape)[0, :, 0])
    if "weight" in pending:
        # Each group has zero mean and the same variance. The contrast of
        # opposite outputs cancels bias and reduces rounding in the scale.
        inputs[..., 0] = 32
        inputs[..., 1] = -32
        unit = torch.nn.functional.group_norm(inputs, module.num_groups, eps=module.eps).double()
        span = unit[0, :, 0] - unit[0, :, 1]
        if not bool(((span > 0) & torch.isfinite(span)).all()):
            raise ValueError("GroupNorm probe contrast vanished in the layer dtype")
        values = read(inputs, shape).double()
        contrast = values[0, :, 0] - values[0, :, 1]
        pending["weight"].copy_(contrast / span)
