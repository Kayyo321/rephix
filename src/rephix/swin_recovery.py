"""Chosen observations for native Swin relative positions and attention scales."""

from .native_operation import ProbeResult, capture_native_operation

SWIN_ATTENTION = "transformers.models.swin.modeling_swin.SwinSelfAttention"
SWINV2_ATTENTION = "transformers.models.swinv2.modeling_swinv2.Swinv2SelfAttention"


def validate_swin_bias(module, pending):
    shape = (
        (2 * module.window_size[0] - 1) * (2 * module.window_size[1] - 1),
        module.num_attention_heads,
    )
    if (
        pending.keys() != {"relative_position_bias_table"}
        or pending["relative_position_bias_table"].shape != shape
    ):
        raise ValueError("unsupported Swin relative position bias layout")


def recover_swin_bias(torch, module, pending, read, batch_size):
    table = pending["relative_position_bias_table"]
    for start in range(0, table.shape[0], batch_size):
        stop = min(start + batch_size, table.shape[0])
        indices = torch.arange(start, stop, dtype=torch.long)
        table[start:stop].copy_(
            read(indices, (stop - start, table.shape[1]), operation="relative_position_bias")
        )


def validate_swin_index(module):
    import torch

    index = module.relative_position_index
    window_tokens = module.window_size[0] * module.window_size[1]
    if (
        index.shape != (window_tokens, window_tokens)
        or index.dtype != torch.long
        or bool((index < 0).any())
        or bool((index >= module.relative_position_bias_table.shape[0]).any())
    ):
        raise ValueError("unsupported Swin relative position index layout")


def recover_swin_index(torch, module, pending, read):
    index = pending["bias"]
    index.copy_(
        read(
            torch.zeros(1, dtype=torch.long),
            tuple(index.shape),
            operation="relative_position_index",
        )
    )


def validate_swin_logit_scale(module, pending):
    if pending.keys() != {"logit_scale"} or pending["logit_scale"].shape != (
        module.num_attention_heads,
        1,
        1,
    ):
        raise ValueError("unsupported Swin V2 logit scale layout")


def recover_swin_logit_scale(torch, module, pending, read):
    scale = pending["logit_scale"]
    bound = scale.new_full((1,), torch.finfo(scale.dtype).max)
    scale.copy_(read(bound, tuple(scale.shape), operation="unclamped_logit_scale"))


def _attention_inputs(module):
    tokens = module.window_size[0] * module.window_size[1]
    return module.query.weight.new_zeros((1, tokens, module.all_head_size))


def observe_swin_bias(torch, module, inputs):
    """Choose row IDs for the native relative position table lookup."""

    def intercept(func, args, kwargs):
        if func is torch.Tensor.__getitem__ and args[0] is module.relative_position_bias_table:
            raise ProbeResult(func(args[0], inputs))
        return func(*args, **kwargs)

    return capture_native_operation(lambda: module(_attention_inputs(module)), intercept)


def observe_swin_index(torch, module, inputs):
    """Label table rows to observe the index used by the native lookup."""
    validate_swin_index(module)

    def intercept(func, args, kwargs):
        if func is torch.Tensor.__getitem__ and args[0] is module.relative_position_bias_table:
            labels = torch.arange(args[0].shape[0], device=inputs.device, dtype=torch.long)
            observed = func(labels, args[1])
            raise ProbeResult(observed.reshape(module.relative_position_index.shape))
        return func(*args, **kwargs)

    return capture_native_operation(lambda: module(_attention_inputs(module)), intercept)


def observe_swin_logit_scale(torch, module, inputs):
    """Choose a clamp bound that exposes the scale before exponentiation."""

    def intercept(func, args, kwargs):
        if func is torch.clamp and args[0] is module.logit_scale:
            raise ProbeResult(func(args[0], max=float(inputs[0])))
        return func(*args, **kwargs)

    return capture_native_operation(lambda: module(_attention_inputs(module)), intercept)
