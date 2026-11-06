"""Recover the three flattened native DBRX expert projections."""

from .native_operation import ProbeResult, capture_native_operation

DBRX_EXPERT_GLU = "transformers.models.dbrx.modeling_dbrx.DbrxExpertGLU"
_PROJECTIONS = ("w1", "v1", "w2")


def validate_dbrx_experts(module, pending):
    shape = (module.moe_num_experts * module.ffn_hidden_size, module.hidden_size)
    if (
        min(module.moe_num_experts, module.ffn_hidden_size, module.hidden_size) <= 0
        or pending.keys() != set(_PROJECTIONS)
        or any(parameter.shape != shape for parameter in pending.values())
    ):
        raise ValueError("unexpected or tied DBRX expert projections")
    if len({parameter.dtype for parameter in pending.values()}) != 1:
        raise ValueError("DBRX expert projections must share a probe dtype")


def recover_dbrx_experts(torch, module, pending, read, batch_size):
    for name in _PROJECTIONS:
        bank = pending[name].view(
            module.moe_num_experts, module.ffn_hidden_size, module.hidden_size
        )
        if name != "w2":
            bank = bank.transpose(1, 2)
        for expert, matrix in enumerate(bank):
            width, output_width = matrix.shape
            for start in range(0, width, batch_size):
                stop = min(start + batch_size, width)
                rows = torch.arange(stop - start, device=matrix.device)
                inputs = matrix.new_zeros((stop - start, width))
                inputs[rows, rows + start] = 1
                observed = read(inputs, (stop - start, output_width), f"{name}/{expert}")
                matrix[start:stop].copy_(observed)


def observe_dbrx_experts(torch, experts, inputs, operation):
    name, _, expert = operation.partition("/")
    if name not in _PROJECTIONS or not expert.isdecimal() or int(expert) >= experts.moe_num_experts:
        raise ValueError(f"invalid DBRX expert probe operation {operation!r}")
    expert = int(expert)
    target_matmul = _PROJECTIONS.index(name) + 1
    if name == "w2":
        # The selected expert runs its two input projections on a single dummy
        # token before the down matmul receives the independent chosen input.
        hidden = inputs.new_zeros((1, 1, experts.mlp.hidden_size))
    else:
        hidden = inputs.unsqueeze(0)
    tokens = hidden.shape[1]
    top_experts = torch.full((tokens, 1), expert, dtype=torch.long, device=inputs.device)
    top_weights = inputs.new_ones((tokens, 1))
    routing_weights = inputs.new_zeros((tokens, experts.moe_num_experts))
    routing_weights[:, expert] = 1
    matmuls = 0

    def intercept(func, args, kwargs):
        nonlocal matmuls
        if func is torch.Tensor.matmul:
            matmuls += 1
            if matmuls == target_matmul:
                if name == "w2":
                    args = (inputs, args[1])
                raise ProbeResult(func(*args, **kwargs))
        return func(*args, **kwargs)

    return capture_native_operation(
        lambda: experts(hidden, routing_weights, top_weights, top_experts), intercept
    )
