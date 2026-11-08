"""Recover native LSTM affine branches before fused gate operations."""

from .native_operation import ProbeResult, capture_native_operation


def validate_lstm(module, pending):
    import torch

    if type(module) is not torch.nn.LSTM or pending.keys() != set(module._flat_weights_names):
        raise ValueError("LSTM probes require the complete native affine parameter layout")
    if min(module.input_size, module.hidden_size, module.num_layers) < 1:
        raise ValueError("LSTM probes require positive layer dimensions")
    for name, value in pending.items():
        rank = 1 if name.startswith("bias_") else 2
        if value.ndim != rank or min(value.shape) < 1:
            raise ValueError("unsupported native LSTM affine parameter shape")
        if not value.is_floating_point() or value.layout != torch.strided:
            raise ValueError("LSTM probes require dense floating-point parameters")


def recover_lstm(torch, module, pending, read, batch_size):
    """Choose inputs independently for every packed input/recurrent affine branch."""
    validate_lstm(module, pending)
    for name, parameter in pending.items():
        if name.startswith("bias_"):
            weight_name = "weight_" + name.removeprefix("bias_")
            inputs = parameter.new_zeros((1, pending[weight_name].shape[1]))
            parameter.copy_(read(inputs, (1, parameter.numel()), operation=f"gate_bias/{name}")[0])
            continue
        out_features, in_features = parameter.shape
        for start in range(0, in_features, batch_size):
            stop = min(start + batch_size, in_features)
            rows = torch.arange(stop - start, device=parameter.device)
            inputs = parameter.new_zeros((stop - start, in_features))
            inputs[rows, rows + start] = 1
            response = read(inputs, (stop - start, out_features), operation=f"gate_weight/{name}")
            parameter[:, start:stop].copy_(response.T)


def observe_lstm(torch, module, inputs, operation):
    """Observe a selected native gate-affine branch from fused LSTM arguments.

    This interface exposes branches before their nonlinearities and before
    input and recurrent biases are added together. The ordinary fused LSTM
    output alone cannot identify those two biases independently.
    """
    from torch.nn import functional as F

    kind, separator, name = operation.partition("/")
    if not separator or kind not in {"gate_weight", "gate_bias"}:
        raise ValueError("unknown native LSTM affine observation")
    if name not in module._flat_weights_names:
        raise ValueError("LSTM observation names an unknown native parameter")
    if (kind == "gate_bias") != name.startswith("bias_"):
        raise ValueError("LSTM observation disagrees with the native parameter kind")
    weight_name = "weight_" + name.removeprefix("bias_") if kind == "gate_bias" else name
    weight_index = module._flat_weights_names.index(weight_name)
    bias_index = module._flat_weights_names.index(name) if kind == "gate_bias" else None

    def intercept(func, args, kwargs):
        if func is torch._VF.lstm:
            weights = args[2]
            weight = weights[weight_index]
            bias = weights[bias_index] if bias_index is not None else None
            raise ProbeResult(F.linear(inputs, weight, bias))
        return func(*args, **kwargs)

    pixels = inputs.new_zeros((1, 1, module.input_size))
    return capture_native_operation(lambda: module(pixels), intercept)
