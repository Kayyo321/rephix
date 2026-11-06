"""Chosen native operation probes for dense GPT-OSS state."""

from __future__ import annotations

import copy

from .native_operation import ProbeResult, capture_native_operation

_MODELING = "transformers.models.gpt_oss.modeling_gpt_oss."
GPTOSS_OPERATORS = {
    _MODELING + "GptOssExperts": "gptoss_experts",
    _MODELING + "GptOssTopKRouter": "gptoss_router",
    _MODELING + "GptOssAttention": "gptoss_sinks",
}


def validate_gptoss_operator(module, pending, kind):
    """Check dimensions and full ownership before probing a fused module."""
    if kind == "gptoss_experts":
        count, hidden, intermediate = (
            module.num_experts,
            module.hidden_size,
            module.intermediate_size,
        )
        shapes = {
            "gate_up_proj": (count, hidden, 2 * intermediate),
            "gate_up_proj_bias": (count, 2 * intermediate),
            "down_proj": (count, intermediate, hidden),
            "down_proj_bias": (count, hidden),
        }
    elif kind == "gptoss_router":
        shapes = {"weight": (module.num_experts, module.hidden_dim), "bias": (module.num_experts,)}
    else:
        shapes = {"sinks": (module.config.num_attention_heads,)}
    if {name: tuple(parameter.shape) for name, parameter in pending.items()} != shapes:
        raise ValueError(f"unexpected or tied GPT-OSS parameters for {kind}")
    if len({parameter.dtype for parameter in pending.values()}) != 1:
        raise ValueError(f"GPT-OSS {kind} parameters must share a probe dtype")


def _recover_affine(torch, weight, bias, read, batch_size, operation, expert=None):
    suffix = "" if expert is None else f"/{expert}"
    input_width, output_width = weight.shape
    bias.copy_(
        read(weight.new_zeros((1, input_width)), (1, output_width), f"{operation}_bias{suffix}")[0]
    )
    for start in range(0, input_width, batch_size):
        stop = min(start + batch_size, input_width)
        rows = torch.arange(stop - start, device=weight.device)
        inputs = weight.new_zeros((stop - start, input_width))
        inputs[rows, rows + start] = 1
        weight[start:stop].copy_(
            read(inputs, (stop - start, output_width), f"{operation}_weight{suffix}")
        )


def recover_gptoss_operator(torch, module, pending, kind, read, batch_size):
    """Recover sink logits and native affine maps before lossy nonlinearities."""
    if kind == "gptoss_sinks":
        sinks = pending["sinks"]
        sinks.copy_(read(sinks.new_zeros((1,)), tuple(sinks.shape), "attention_sinks"))
    elif kind == "gptoss_router":
        _recover_affine(torch, pending["weight"].T, pending["bias"], read, batch_size, "router")
    else:
        for operation in ("gate_up", "down"):
            weight = pending[f"{operation}_proj"]
            bias = pending[f"{operation}_proj_bias"]
            for expert in range(module.num_experts):
                _recover_affine(
                    torch, weight[expert], bias[expert], read, batch_size, operation, expert
                )


def _observe_expert(torch, module, inputs, operation, expert):
    if not 0 <= expert < module.num_experts:
        raise ValueError("GPT-OSS expert probe index is out of range")
    down = operation in {"down_weight", "down_bias"}
    bias = operation in {"gate_up_bias", "down_bias"}
    # The first affine map needs one dummy row when the down input is supplied
    # independently at the second matmul.
    hidden = inputs.new_zeros((1, 1, module.hidden_size)) if down else inputs.unsqueeze(1)
    indices = torch.full((hidden.shape[0], 1), expert, dtype=torch.long, device=inputs.device)
    routing = inputs.new_zeros((hidden.shape[0], module.num_experts))
    routing[:, expert] = 1
    matmuls = 0
    unbiased = None

    def intercept(func, args, kwargs):
        nonlocal matmuls, unbiased
        if func is torch.Tensor.matmul:
            matmuls += 1
            if matmuls == (2 if down else 1):
                if down:
                    args = (inputs, args[1])
                unbiased = func(*args, **kwargs)
                if not bias:
                    raise ProbeResult(unbiased)
                return unbiased
        if func is torch.Tensor.add and unbiased is not None and args[0] is unbiased:
            raise ProbeResult(func(*args, **kwargs))
        return func(*args, **kwargs)

    # The native per-expert branch is also used during training. Force that
    # branch on CUDA to avoid executing every bank entry for each chosen ID.
    if inputs.device.type != "cpu" and not module.training:
        module = copy.copy(module)
        module.training = True
    return capture_native_operation(
        lambda: module(hidden, router_indices=indices, routing_weights=routing), intercept
    )


def _observe_router(torch, module, inputs, operation):
    def intercept(func, args, kwargs):
        if func is torch.nn.functional.linear:
            # The router passes its full affine output to top-k. Isolate the
            # matrix product so rounding after adding bias cannot hide a weight.
            if operation == "router_weight":
                args = args[:2]
            raise ProbeResult(func(*args, **kwargs))
        return func(*args, **kwargs)

    return capture_native_operation(lambda: module(inputs), intercept)


def _observe_sinks(torch, module, inputs):
    from transformers.models.gpt_oss.modeling_gpt_oss import eager_attention_forward

    heads, kv_heads = module.config.num_attention_heads, module.config.num_key_value_heads
    query = inputs.new_zeros((1, heads, 1, module.head_dim))
    key_value = inputs.new_zeros((1, kv_heads, 1, module.head_dim))

    def intercept(func, args, kwargs):
        if func is torch.cat:
            combined = func(*args, **kwargs)
            if combined.shape != (1, heads, 1, 2):
                raise ValueError("unexpected native GPT-OSS attention logit layout")
            raise ProbeResult(combined[0, :, 0, 1])
        return func(*args, **kwargs)

    return capture_native_operation(
        lambda: eager_attention_forward(module, query, key_value, key_value, None, module.scaling),
        intercept,
    )


def observe_gptoss_operator(torch, module, inputs, operation):
    """Invoke native operations with a controlled internal observation point."""
    if operation == "attention_sinks":
        return _observe_sinks(torch, module, inputs)
    if operation in {"router_weight", "router_bias"}:
        return _observe_router(torch, module, inputs, operation)
    name, _, expert = operation.partition("/")
    if (
        name in {"gate_up_weight", "gate_up_bias", "down_weight", "down_bias"}
        and expert.isdecimal()
    ):
        return _observe_expert(torch, module, inputs, name, int(expert))
    raise ValueError(f"unknown native GPT-OSS probe operation {operation!r}")
