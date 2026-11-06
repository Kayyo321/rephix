"""Recover native grouped affine expert banks from chosen expert assignments."""

from __future__ import annotations

_JETMOE = "transformers.models.jetmoe.modeling_jetmoe."
_GRANITE_MOE = "transformers.models.granitemoe.modeling_granitemoe."
_GRANITE_MOE_SHARED = "transformers.models.granitemoeshared.modeling_granitemoeshared."

GROUPED_EXPERT_BANKS = {
    _JETMOE + "JetMoeParallelExperts",
    _GRANITE_MOE + "GraniteMoeParallelExperts",
    _GRANITE_MOE_SHARED + "GraniteMoeSharedParallelExperts",
}
GROUPED_EXPERT_BIASES = {_JETMOE + "JetMoeMoE", _JETMOE + "JetMoeMoA"}
GROUPED_EXPERT_RMS_NORMS = {
    _JETMOE + "JetMoeRMSNorm",
    _GRANITE_MOE + "GraniteMoeRMSNorm",
    _GRANITE_MOE_SHARED + "GraniteMoeSharedRMSNorm",
    "transformers.models.granite.modeling_granite.GraniteRMSNorm",
    "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm",
}


def _is_jet_attention(module):
    cls = type(module)
    return f"{cls.__module__}.{cls.__name__}" == _JETMOE + "JetMoeMoA"


def validate_grouped_experts(module, pending, kind):
    if kind == "grouped_expert_bank":
        shape = (module.num_experts, module.output_size, module.input_size)
        if pending.keys() != {"weight"} or module.weight.shape != shape or min(shape) <= 0:
            raise ValueError("unsupported grouped expert weight layout")
    elif pending.keys() != {"bias"} or module.bias.shape != (module.input_size,):
        raise ValueError("unsupported grouped expert bias layout")


def recover_grouped_experts(torch, module, read, batch_size, kind):
    if kind == "grouped_expert_bias":
        input_size = module.hidden_size if _is_jet_attention(module) else module.input_size
        bias = read(
            module.bias.new_zeros((1, input_size)), (1, module.input_size), operation="bias"
        )
        module.bias.copy_(bias[0])
        return

    weight = module.weight
    for expert in range(module.num_experts):
        for start in range(0, module.input_size, batch_size):
            stop = min(start + batch_size, module.input_size)
            rows = torch.arange(stop - start, device=weight.device)
            inputs = weight.new_zeros((stop - start, module.input_size))
            inputs[rows, start + rows] = 1
            shape = (stop - start, module.output_size)
            observed = read(inputs, shape, operation=f"expert_{expert}")
            weight[expert, :, start:stop].copy_(observed.T)


def observe_grouped_experts(torch, module, inputs, operation):
    if operation == "bias":
        if _is_jet_attention(module):
            # A single active expert with a unit gate maps zero to the reduce bias.
            indices = torch.arange(inputs.shape[0], device=inputs.device)
            sizes = [inputs.shape[0]] + [0] * (module.num_experts - 1)
            topology = (indices, indices, inputs.new_ones(inputs.shape[0]), sizes)
            return module.reduce(inputs[:, None, None, :], topology)[:, 0]
        return module(inputs[:, None, :])[0][:, 0]

    if not isinstance(operation, str) or not operation.startswith("expert_"):
        raise ValueError("grouped expert probe needs a selected expert")
    expert = int(operation.removeprefix("expert_"))
    if not 0 <= expert < module.num_experts:
        raise ValueError("grouped expert probe index is outside the bank")
    sizes = [0] * module.num_experts
    sizes[expert] = inputs.shape[0]
    return module(inputs, sizes)
