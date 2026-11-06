"""Probe native router logits and the persistent correction used for expert selection."""

from __future__ import annotations

from .native_operation import ProbeResult, capture_native_operation

CORRECTED_ROUTER_MODULES = frozenset(
    {
        "transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3TopkRouter",
        "transformers.models.glm4_moe.modeling_glm4_moe.Glm4MoeTopkRouter",
    }
)
ROUTER_MODULES = CORRECTED_ROUTER_MODULES | {
    "transformers.models.deepseek_v2.modeling_deepseek_v2.DeepseekV2MoEGate",
}
ROUTER_CORRECTION_BUFFER = "e_score_correction_bias"


def recover_router_weight(torch, module, read, batch_size):
    """Recover the weight at the precision used by the native router."""
    weight = module.weight
    experts, hidden = weight.shape
    for start in range(0, hidden, batch_size):
        end = min(start + batch_size, hidden)
        rows = torch.arange(end - start, device=weight.device)
        inputs = weight.new_zeros((end - start, hidden), dtype=torch.float32)
        inputs[rows, rows + start] = 1
        logits = read(inputs, (end - start, experts), operation="router_logits")
        weight[:, start:end].copy_(logits.T)
    return weight.numel() if weight.dtype == torch.float64 else 0


def recover_router_correction(module, read):
    """Recover the correction as the native score addition at zero input."""
    correction = getattr(module, ROUTER_CORRECTION_BUFFER)
    shape = (1, correction.numel())
    correction.copy_(read(correction.new_zeros(shape), shape, operation="router_correction")[0])


def observe_router(torch, module, inputs, operation):
    """Tap the native operation's result before activation or expert selection."""
    if operation == "router_logits":
        primitive = torch.nn.functional.linear
        forward = module
        # V2's native forward requires a batch and sequence axis.
        inputs = inputs.unsqueeze(0)
    elif operation == "router_correction":
        primitive = torch.Tensor.add
        forward = module.get_topk_indices
    else:
        raise ValueError(f"unknown native router operation {operation!r}")

    def intercept(func, args, kwargs):
        result = func(*args, **kwargs)
        if func is primitive:
            # Selection discards coordinates and is unnecessary after the observation.
            raise ProbeResult(result)
        return result

    return capture_native_operation(lambda: forward(inputs), intercept)
