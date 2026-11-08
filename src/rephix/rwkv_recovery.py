"""Recover native RWKV time mixtures and recurrent time parameters."""

from .native_operation import ProbeResult, capture_native_operation

RWKV_OPERATORS = {
    "transformers.models.rwkv.modeling_rwkv.RwkvSelfAttention": "attention",
    "transformers.models.rwkv.modeling_rwkv.RwkvFeedForward": "feed_forward",
}


def _kind(module):
    cls = type(module)
    name = f"{cls.__module__}.{cls.__name__}"
    if name not in RWKV_OPERATORS:
        raise ValueError("RWKV probes require a supported native time-mixing operator")
    return RWKV_OPERATORS[name]


def validate_rwkv_operator(module, pending):
    attention = _kind(module) == "attention"
    mixtures = {"time_mix_key", "time_mix_receptance"}
    if attention:
        mixtures.add("time_mix_value")
    expected = mixtures | ({"time_decay", "time_first"} if attention else set())
    if pending.keys() != expected:
        raise ValueError("unsupported native RWKV time parameter layout")
    if any(pending[name].shape != (1, 1, module.config.hidden_size) for name in mixtures):
        raise ValueError("RWKV time mixtures disagree with the hidden dimension")
    if attention and any(
        pending[name].shape != (module.attention_hidden_size,)
        for name in ("time_decay", "time_first")
    ):
        raise ValueError("RWKV time scalars disagree with the attention dimension")


def recover_rwkv_operator(torch, module, pending, read):
    """Recover standalone RWKV parameters from declared FP64 observations."""
    validate_rwkv_operator(module, pending)
    shape = (1, 1, module.config.hidden_size)
    for name, parameter in pending.items():
        if name.startswith("time_mix_"):
            response = read(torch.ones(shape, dtype=torch.float64), shape, operation=name)
        elif name == "time_first":
            response = read(
                torch.zeros(shape, dtype=torch.float64), tuple(parameter.shape), operation=name
            )
        else:
            rates = read(
                torch.zeros(shape, dtype=torch.float64),
                tuple(parameter.shape),
                operation="time_decay_exp",
            )
            if not bool(torch.isfinite(rates).all()) or bool((rates <= 0).any()):
                raise ValueError("RWKV time decay is outside the native FP64 exponential range")
            response = rates.log()
        parameter.copy_(response)


def observe_rwkv_operator(torch, module, inputs, operation):
    """Observe a native mixture, zero-key sum, or FP64 exponential.

    Unit one-token features have a zero shifted predecessor. Their native
    multiplication exposes each mixture before projection. Time decay uses a
    controlled FP64 exponential so lower precision coordinates remain resolvable
    after restoring their original storage dtype.
    """
    if operation.startswith("time_mix_"):
        parameter = getattr(module, operation)
    elif operation == "time_first":
        parameter = module.time_first
    elif operation == "time_decay_exp":
        parameter = module.time_decay
    else:
        raise ValueError("unsupported RWKV observation")

    def intercept(func, args, kwargs):
        if operation.startswith("time_mix_") and func is torch.Tensor.mul:
            if any(arg is parameter for arg in args):
                raise ProbeResult(func(*args, **kwargs).double())
        elif operation == "time_first" and func is torch.Tensor.add:
            if any(arg is parameter for arg in args):
                raise ProbeResult(func(*args, **kwargs).reshape(parameter.shape).double())
        elif operation == "time_decay_exp" and func is torch.exp and args[0] is parameter:
            raise ProbeResult(func(args[0].double()))
        return func(*args, **kwargs)

    hidden = inputs.to(dtype=module.key.weight.dtype)
    return capture_native_operation(lambda: module(hidden), intercept)
