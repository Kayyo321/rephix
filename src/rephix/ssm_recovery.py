"""Recover native Mamba skip gains and effective state transitions."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

_JAMBA_MIXER = "transformers.models.jamba.modeling_jamba.JambaMambaMixer"
SSM_MIXERS = {
    "transformers.models.mamba.modeling_mamba.MambaMixer",
    "transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaMixer",
    _JAMBA_MIXER,
}
_INITIAL_EXPONENTS = (-32, -16, 0, 16, 32)
_FULL_EXPONENTS = range(-128, 153, 8)


def validate_ssm_mixer(module):
    cls = type(module)
    if f"{cls.__module__}.{cls.__name__}" not in SSM_MIXERS:
        raise ValueError("state probes require a supported native Mamba mixer")
    if module.A_log.shape != (
        module.intermediate_size,
        module.ssm_state_size,
    ) or module.D.shape != (module.intermediate_size,):
        raise ValueError("Mamba state parameters disagree with the mixer configuration")


class _ProbeCache:
    def __init__(self, layer_idx, state):
        self.ssm_states = {layer_idx: state}

    def update_conv_state(self, layer_idx, new_conv_state, cache_position):
        return new_conv_state

    def update_ssm_state(self, layer_idx, new_ssm_state):
        self.ssm_states[layer_idx].copy_(new_ssm_state)


def _jamba_probe_cache(layer_idx, state):
    from transformers.models.jamba.modeling_jamba import HybridMambaAttentionDynamicCache

    # The native forward checks the cache type. Allocate only the mixer slot
    # rather than constructing cache tensors for the whole hybrid model.
    cache = object.__new__(HybridMambaAttentionDynamicCache)
    cache.ssm_states = {layer_idx: state}
    cache.conv_states = {}
    cache.has_previous_state = False
    return cache


def _inverse_softplus(steps):
    return steps + (-(-steps).expm1()).log()


def observe_ssm_mixer(torch, module, steps):
    """Run one controlled native recurrence step for each requested timestep."""
    validate_ssm_mixer(module)
    if module.D.device.type == "mps":
        raise ValueError("Mamba state probes need a float64 oracle; use cpu or cuda")
    steps = steps.to(device=module.D.device, dtype=torch.float64)
    batch = steps.numel()
    inner, state_size = module.intermediate_size, module.ssm_state_size
    conv = steps.new_ones((batch, inner, 1))
    projected = steps.new_ones((batch, 1, 2 * inner))
    selection = steps.new_zeros((batch, 1, module.time_step_rank + 2 * state_size))
    times = _inverse_softplus(steps)[:, None, None].expand(batch, 1, inner)
    cls = type(module)
    is_jamba = f"{cls.__module__}.{cls.__name__}" == _JAMBA_MIXER
    state = steps.new_ones((batch, inner, state_size))
    inputs = steps.new_zeros((batch, 1, module.hidden_size))

    # Unit scan input and gate expose D at the output projection input. Zero
    # input-dependent B/C leaves the seeded cache to measure only the decay.
    # The temporary child overrides are confined to the local oracle model.
    with ExitStack() as stack:
        for child, forward in (
            (module.in_proj, lambda _: projected),
            (module.conv1d, lambda _: conv),
            (module.act, lambda values: values),
            (module.x_proj, lambda _: selection),
            (module.dt_proj, lambda _: times),
            (module.out_proj, lambda values: values),
        ):
            stack.enter_context(patch.object(child, "forward", forward))
        if is_jamba:
            cache = _jamba_probe_cache(module.layer_idx, state)
            for norm in (module.dt_layernorm, module.b_layernorm, module.c_layernorm):
                stack.enter_context(patch.object(norm, "forward", lambda values: values))
            skip = module.slow_forward(inputs, cache_params=cache)
        else:
            cache = _ProbeCache(module.layer_idx, state)
            cache_position = torch.arange(module.conv_kernel_size, device=steps.device)
            skip = module.slow_forward(inputs, cache_params=cache, cache_position=cache_position)
    return torch.cat((skip, cache.ssm_states[module.layer_idx].transpose(1, 2)), dim=1)


def _rates(torch, steps, transitions):
    attenuation = -transitions.log()
    informative = (attenuation >= 2**-12) & (attenuation <= 16)
    best, indices = torch.where(informative, attenuation, 0).max(dim=0)
    return (best / steps[indices]).float(), best > 0


def _nearest_exp_boundary(torch, center, rates, missing):
    # CUDA exp can map the mathematical center to an adjacent rate, even when
    # dozens of consecutive storage values map to the observed rate. Search
    # towards that rate with the native exp rather than relying on log/exp
    # round trips or a fixed count of neighboring floats.
    increase = center.float().exp() < rates
    crossed = torch.where(increase, 128, -128).to(dtype=center.dtype)
    rejected = center
    for _ in range(64):
        middle = ((rejected.double() + crossed.double()) / 2).to(center.dtype)
        moving = missing & (middle != rejected) & (middle != crossed)
        if not bool(moving.any()):
            break
        observed = middle.float().exp()
        reaches_rate = torch.where(increase, observed >= rates, observed <= rates)
        crossed = torch.where(moving & reaches_rate, middle, crossed)
        rejected = torch.where(moving & ~reaches_rate, middle, rejected)
    return crossed


def _representative(torch, rates, dtype, rate_device):
    rates = rates.to(rate_device)
    # Native mixers cast A_log to float32 before evaluating exp, even when
    # their other parameters and the controlled recurrence are float64.
    coordinate_dtype = torch.float32 if dtype == torch.float64 else dtype
    center = rates.double().log().clamp(min=-128, max=128).to(coordinate_dtype)
    positive = torch.full_like(center, float("inf"))
    negative = -positive
    result = center.clone()
    found = center.float().exp() == rates
    low = high = center
    for _ in range(4):
        if bool(found.all()):
            break
        low = torch.nextafter(low, negative)
        high = torch.nextafter(high, positive)
        for neighbor in (low, high):
            match = ~found & (neighbor.float().exp() == rates)
            result[match] = neighbor[match]
            found |= match
    if not bool(found.all()):
        boundary = _nearest_exp_boundary(torch, center, rates, ~found)
        result = torch.where(found, result, boundary)
    if not bool((result.float().exp() == rates).all()):
        raise ValueError("Mamba effective rates have no verified representative in storage dtype")
    return result.to(device="cpu", dtype=dtype)


def recover_ssm_mixer(torch, module, pending, read, *, rate_device="cpu"):
    """Populate a mixer using float64 transition and skip observations."""
    validate_ssm_mixer(module)
    inner, state_size = module.intermediate_size, module.ssm_state_size

    def probe(exponents):
        steps = torch.tensor([2.0**exponent for exponent in exponents], dtype=torch.float64)
        data = read(steps, (len(steps), state_size + 1, inner))
        skip = data[:, 0]
        transitions = data[:, 1:].transpose(1, 2)
        if not torch.equal(skip, skip[:1].expand_as(skip)):
            raise ValueError("Mamba skip probe changed across timesteps")
        if torch.any((transitions < 0) | (transitions > 1)):
            raise ValueError("Mamba state probe returned an invalid decay")
        return steps, skip[0], transitions

    steps, skip, transitions = probe(_INITIAL_EXPONENTS)
    report = {
        "unidentified_parameter_elements": 0,
        "effective_rate_elements_recovered": 0,
        "saturated_effective_rate_elements": 0,
        "effective_rate_max_transition_error": 0.0,
        "representative": "nearest_log_reproducing_fp32_exp",
    }
    if "A_log" in pending:
        rates, resolved = _rates(torch, steps, transitions)
        if not bool(resolved.all()):
            more_steps, more_skip, more_transitions = probe(
                [value for value in _FULL_EXPONENTS if value not in _INITIAL_EXPONENTS]
            )
            if not torch.equal(skip, more_skip):
                raise ValueError("Mamba skip probe changed across timesteps")
            steps = torch.cat((steps, more_steps))
            transitions = torch.cat((transitions, more_transitions))
            rates, resolved = _rates(torch, steps, transitions)

        underflow = (transitions == 1).all(dim=0)
        overflow = (transitions == 0).all(dim=0)
        saturated = underflow | overflow
        if not bool((resolved | saturated).all()):
            raise ValueError("Mamba effective rates could not be resolved from the decay probes")
        rates[overflow] = float("inf")
        prediction = torch.exp(-steps[:, None, None] * rates.double())
        if not torch.allclose(prediction, transitions, atol=2e-15, rtol=2e-13):
            raise ValueError("Mamba effective rates do not reproduce the observed decay")
        raw = module.A_log
        raw.copy_(_representative(torch, rates, raw.dtype, rate_device))
        report.update(
            unidentified_parameter_elements=raw.numel(),
            effective_rate_elements_recovered=raw.numel(),
            saturated_effective_rate_elements=int(saturated.sum()),
            effective_rate_max_transition_error=float((prediction - transitions).abs().max()),
        )
    if "D" in pending:
        module.D.copy_(skip)
    return report
