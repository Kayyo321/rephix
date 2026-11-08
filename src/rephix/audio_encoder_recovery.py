"""Chosen observations for native waveform encoders and their attention gates."""

from unittest.mock import patch

from .native_operation import ProbeResult, capture_native_operation

AUDIO_MASK_MODELS = {
    "transformers.models.hubert.modeling_hubert.HubertModel",
    "transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model",
    "transformers.models.wavlm.modeling_wavlm.WavLMModel",
}
WAVLM_POSITION_GATE = "transformers.models.wavlm.modeling_wavlm.WavLMAttention"
AUDIO_ENCODER_TYPES = {"hubert", "wav2vec2", "wavlm"}


def audio_model_wrapper(config):
    from transformers import AutoModel, AutoModelForCTC

    prefixes = {"hubert": "Hubert", "wav2vec2": "Wav2Vec2", "wavlm": "WavLM"}
    prefix = prefixes[config.model_type]
    architectures = set(config.architectures or ())
    if architectures == {f"{prefix}Model"}:
        return AutoModel, False
    if architectures == {f"{prefix}ForCTC"}:
        return AutoModelForCTC, True
    raise ValueError("audio checkpoint must declare its native base-model or CTC architecture")


def initialize_missing_audio_masks(torch, target, missing_keys):
    """Initialize native optional mask vectors absent from stored checkpoint state.

    These zeros provide deterministic scaffold state for an unstored parameter.
    They are recorded separately from the original stored weight coordinates.
    """
    initialized, remaining = [], []
    for name in missing_keys:
        path, _, parameter_name = name.rpartition(".")
        try:
            module = target.get_submodule(path)
        except AttributeError:
            remaining.append(name)
            continue
        qualified = f"{type(module).__module__}.{type(module).__name__}"
        if parameter_name != "masked_spec_embed" or qualified not in AUDIO_MASK_MODELS:
            remaining.append(name)
            continue
        weight = dict(module.named_parameters(recurse=False)).get(parameter_name)
        validate_audio_mask_embedding(module, {parameter_name: weight})
        with torch.no_grad():
            weight.zero_()
        initialized.append(name)
    return initialized, remaining


def validate_audio_mask_embedding(module, pending):
    if pending.keys() != {"masked_spec_embed"} or pending["masked_spec_embed"].shape != (
        module.config.hidden_size,
    ):
        raise ValueError("unsupported native audio mask embedding layout")


def recover_audio_mask_embedding(module, pending, read):
    weight = pending["masked_spec_embed"]
    shape = (1, 1, weight.numel())
    weight.copy_(read(weight.new_zeros(shape), shape, operation="masked_spec").flatten())


def observe_audio_mask_embedding(torch, module, inputs):
    # Stored mask embeddings can remain present when SpecAugment is disabled.
    # Activate its native assignment only for this chosen observation.
    mask = torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.bool)
    with (
        patch.object(module.config, "apply_spec_augment", True),
        patch.object(module, "training", False),
    ):
        return module._mask_hidden_states(inputs.clone(), mask_time_indices=mask)


def validate_wavlm_position_gate(module, pending):
    if pending.keys() != {"gru_rel_pos_const"} or pending["gru_rel_pos_const"].shape != (
        1,
        module.num_heads,
        1,
        1,
    ):
        raise ValueError("unsupported WavLM relative position gate layout")


def recover_wavlm_position_gate(module, pending, read):
    weight = pending["gru_rel_pos_const"]
    shape = (1, 1, module.embed_dim)
    weight.copy_(read(weight.new_zeros(shape), weight.shape, operation="position_gate"))


def observe_wavlm_position_gate(torch, module, inputs):
    # The native gate multiplies its constant by sigmoid(sum(four features)).
    # Chosen features make that sigmoid exactly one at supported precisions.
    handle = module.gru_rel_pos_linear.register_forward_hook(
        lambda _layer, _arguments, values: values.new_full(values.shape, 64)
    )

    def intercept(func, args, kwargs):
        result = func(*args, **kwargs)
        if func is torch.Tensor.mul and any(arg is module.gru_rel_pos_const for arg in args):
            raise ProbeResult(result)
        return result

    try:
        position_bias = inputs.new_zeros((inputs.shape[0] * module.num_heads, 1, 1))
        return capture_native_operation(
            lambda: module(inputs, position_bias=position_bias), intercept
        )
    finally:
        handle.remove()
