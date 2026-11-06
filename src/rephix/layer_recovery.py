"""Recover a known model from chosen-input, coordinate-preserving layer calls."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path

from .convnext_recovery import (
    CONVNEXT_GRN,
    CONVNEXT_LAYER_NORMS,
    CONVNEXT_LAYER_SCALE,
    observe_convnext_grn_scale,
    observe_convnext_scale,
    recover_convnext_grn,
    recover_convnext_norm,
    recover_convnext_scale,
    validate_convnext_grn,
    validate_convnext_norm,
    validate_convnext_scale,
)
from .dbrx_experts import (
    DBRX_EXPERT_GLU,
    observe_dbrx_experts,
    recover_dbrx_experts,
    validate_dbrx_experts,
)
from .deepseek_experts import (
    CORRECTED_ROUTER_MODULES,
    ROUTER_CORRECTION_BUFFER,
    ROUTER_MODULES,
    observe_router,
    recover_router_correction,
    recover_router_weight,
)
from .encoder_recovery import ENCODER_MODEL_TYPES, encoder_auto_model
from .fused_expert_recovery import (
    GROUPED_EXPERT_BANKS,
    GROUPED_EXPERT_BIASES,
    GROUPED_EXPERT_RMS_NORMS,
    observe_grouped_experts,
    recover_grouped_experts,
    validate_grouped_experts,
)
from .gemma_recovery import GEMMA3_SCALED_EMBEDDING, GEMMA_OFFSET_RMS_NORMS, recover_gemma_rms
from .gemma_vlm_recovery import (
    GEMMA3_PROJECTOR,
    observe_gemma3_projector,
    recover_gemma3_projector,
    validate_gemma3_projector,
)
from .gptoss_recovery import (
    GPTOSS_OPERATORS,
    observe_gptoss_operator,
    recover_gptoss_operator,
    validate_gptoss_operator,
)
from .group_norm import recover_group_norm, validate_group_norm
from .hf import _deps, _device, _load_tokenizer
from .io import write_json
from .layer_staging import LayerStager
from .multimodal_operators import (
    CLIP_VISION_EMBEDDINGS,
    LLAVA_IMAGE_NEWLINE_MODELS,
    observe_clip_class_embedding,
    observe_llava_image_newline,
    recover_clip_class_embedding,
    recover_llava_image_newline,
    validate_clip_class_embedding,
    validate_llava_image_newline,
)
from .rms_probe import unit_float32_rms_amplitude
from .ssm_recovery import SSM_MIXERS, observe_ssm_mixer, recover_ssm_mixer, validate_ssm_mixer

_HF_RMS_NORMS = {
    "transformers.models.llama.modeling_llama.LlamaRMSNorm",
    "transformers.models.mistral.modeling_mistral.MistralRMSNorm",
    "transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm",
    "transformers.models.phi3.modeling_phi3.Phi3RMSNorm",
    "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm",
    "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm",
}
_DIRECT_HF_RMS_NORMS = {
    "transformers.models.deepseek_v2.modeling_deepseek_v2.DeepseekV2RMSNorm",
    "transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3RMSNorm",
    "transformers.models.falcon_mamba.modeling_falcon_mamba.FalconMambaRMSNorm",
    "transformers.models.glm4_moe.modeling_glm4_moe.Glm4MoeRMSNorm",
    "transformers.models.idefics3.modeling_idefics3.Idefics3RMSNorm",
    "transformers.models.jamba.modeling_jamba.JambaRMSNorm",
    "transformers.models.lfm2.modeling_lfm2.Lfm2RMSNorm",
    "transformers.models.mamba.modeling_mamba.MambaRMSNorm",
    "transformers.models.mt5.modeling_mt5.MT5LayerNorm",
    "transformers.models.olmoe.modeling_olmoe.OlmoeRMSNorm",
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRMSNorm",
    "transformers.models.smolvlm.modeling_smolvlm.SmolVLMRMSNorm",
    "transformers.models.t5.modeling_t5.T5LayerNorm",
    "transformers.models.umt5.modeling_umt5.UMT5LayerNorm",
} | GROUPED_EXPERT_RMS_NORMS
_POST_MULTIPLY_RMS_NORMS = {
    "transformers.models.gpt_oss.modeling_gpt_oss.GptOssRMSNorm",
    "transformers.models.olmo2.modeling_olmo2.Olmo2RMSNorm",
    "transformers.models.olmo3.modeling_olmo3.Olmo3RMSNorm",
}
_BART = "transformers.models.bart.modeling_bart."
_DEBERTA_MLM_HEAD = "transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2LMPredictionHead"
_FALCON_LINEAR = "transformers.models.falcon.modeling_falcon.FalconLinear"
_OPT_POSITIONS = "transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding"
_WHISPER_POSITIONS = "transformers.models.whisper.modeling_whisper.WhisperPositionalEmbedding"
_RELATIVE_BIAS_EMBEDDING = ".relative_attention_bias"
_VISION_EMBEDDINGS = {
    "transformers.models.vit.modeling_vit.ViTEmbeddings": ("cls_token",),
    "transformers.models.deit.modeling_deit.DeiTEmbeddings": ("cls_token", "distillation_token"),
}
_VISION_LANGUAGE_TYPES = {
    "gemma3",
    "idefics3",
    "lfm2-vl",
    "lfm2_vl",
    "llava",
    "llava_next",
    "llava_next_video",
    "llava_onevision",
    "paligemma",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "smolvlm",
}
_VISION_LANGUAGE_ASSETS = (
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
_PROBE_PLAN_VERSION = 1
_JOURNAL_FILENAME = "observations.sqlite3"
_BUFFER_KINDS = {"bart_logits_bias", "router_correction"}
_GPTOSS_KINDS = frozenset(GPTOSS_OPERATORS.values())


def _qualified_name(module):
    cls = type(module)
    return f"{cls.__module__}.{cls.__name__}"


def _operator(torch, module, *, direct):
    cls = type(module)
    qualified_name = _qualified_name(module)
    if cls is torch.nn.Linear or (direct and qualified_name == _FALCON_LINEAR):
        return "linear"
    if direct and qualified_name in SSM_MIXERS:
        return "ssm_mixer"
    if direct and qualified_name in GROUPED_EXPERT_BANKS:
        return "grouped_expert_bank"
    if direct and qualified_name in GROUPED_EXPERT_BIASES:
        return "grouped_expert_bias"
    if direct and qualified_name in ROUTER_MODULES:
        return "native_router"
    if direct and qualified_name in GPTOSS_OPERATORS:
        return GPTOSS_OPERATORS[qualified_name]
    if direct and qualified_name == DBRX_EXPERT_GLU:
        return "dbrx_expert_glu"
    if direct and qualified_name == _DEBERTA_MLM_HEAD:
        return "deberta_mlm_bias"
    if direct and qualified_name == CLIP_VISION_EMBEDDINGS:
        return "clip_class_embedding"
    if direct and qualified_name in LLAVA_IMAGE_NEWLINE_MODELS:
        return "llava_image_newline"
    if direct and qualified_name == GEMMA3_PROJECTOR:
        return "gemma3_projector"
    if direct and qualified_name in CONVNEXT_LAYER_NORMS:
        return "convnext_norm"
    if direct and qualified_name == CONVNEXT_LAYER_SCALE:
        return "convnext_layer_scale"
    if direct and qualified_name == CONVNEXT_GRN:
        return "convnext_grn"
    if direct and cls is torch.nn.GroupNorm:
        validate_group_norm(module)
        return "group_norm"
    if direct and cls is torch.nn.Conv1d:
        from .convolution import validate_conv1d

        validate_conv1d(module)
        return "spatial_conv1d"
    if cls is torch.nn.Conv2d:
        from .convolution import validate_conv2d

        validate_conv2d(module)
        return "conv2d"
    if direct and cls is torch.nn.Conv3d:
        from .convolution import validate_conv3d

        validate_conv3d(module)
        return "conv3d"
    if qualified_name == "transformers.pytorch_utils.Conv1D":
        return "conv1d"
    if cls is torch.nn.Embedding and module.max_norm is None:
        return "embedding"
    if (
        direct
        and qualified_name in {_WHISPER_POSITIONS, GEMMA3_SCALED_EMBEDDING}
        and module.max_norm is None
    ):
        return "embedding"
    if direct and qualified_name == _BART + "BartScaledWordEmbedding":
        return "scaled_embedding"
    if direct and qualified_name in {_BART + "BartLearnedPositionalEmbedding", _OPT_POSITIONS}:
        return "offset_embedding"
    if cls is torch.nn.LayerNorm and module.weight.numel() > 1:
        return "layer_norm"
    if qualified_name in _HF_RMS_NORMS or (direct and qualified_name in _DIRECT_HF_RMS_NORMS):
        return "hf_rms_norm"
    if direct and qualified_name in GEMMA_OFFSET_RMS_NORMS:
        return "gemma_rms_norm"
    if direct and qualified_name in _POST_MULTIPLY_RMS_NORMS:
        return "post_multiply_rms_norm"
    if cls is getattr(torch.nn, "RMSNorm", None):
        return "rms_norm"
    if qualified_name in _VISION_EMBEDDINGS and getattr(module, "mask_token", None) is None:
        return "vision_embeddings"
    raise ValueError(f"unsupported parameterized operator {qualified_name}")


def _plan(torch, model, *, direct=False, relative_bias_mode="embedding"):
    names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    buffers = model.state_dict().keys() - names
    bart_bias = direct and _qualified_name(model) == _BART + "BartForConditionalGeneration"
    correction_probes = []
    for path, module in model.named_modules() if direct else ():
        if _qualified_name(module) in CORRECTED_ROUTER_MODULES:
            name = f"{path}.{ROUTER_CORRECTION_BUFFER}" if path else ROUTER_CORRECTION_BUFFER
            correction = getattr(module, ROUTER_CORRECTION_BUFFER)
            if (
                name not in buffers
                or correction.shape != (module.weight.shape[0],)
                or not correction.is_floating_point()
                or correction.device.type != "cpu"
                or correction.layout != torch.strided
            ):
                raise ValueError(f"{path}: invalid persistent router correction")
            buffers.remove(name)
            correction_probes.append((path, module, {"bias": correction}, "router_correction"))
    unknown_buffers = sorted(buffers - ({"final_logits_bias"} if bart_bias else set()))
    if unknown_buffers:
        raise ValueError(f"persistent state has no layer probe: {', '.join(unknown_buffers[:5])}")

    plan = []
    seen = set()
    for path, module in model.named_modules():
        parameters = dict(module.named_parameters(recurse=False, remove_duplicate=False))
        pending = {name: value for name, value in parameters.items() if id(value) not in seen}
        # Native masked-LM heads can register the same bias on a wrapper and
        # its decoder. Probe the affine child, where zero input exposes it.
        if direct and type(module) is not torch.nn.Linear and "bias" in pending:
            if any(
                type(child) is torch.nn.Linear and child.bias is pending["bias"]
                for child in module.children()
            ):
                del pending["bias"]
        if not pending:
            continue
        kind = _operator(torch, module, direct=direct)
        if relative_bias_mode == "attention" and path.endswith(_RELATIVE_BIAS_EMBEDDING):
            from .relative_bias import relative_bias_spec

            path = path.removesuffix(_RELATIVE_BIAS_EMBEDDING)
            module = model.get_submodule(path)
            relative_bias_spec(module)
            kind = "relative_bias"
        if kind == "vision_embeddings":
            tokens = _VISION_EMBEDDINGS[_qualified_name(module)]
            allowed = set(tokens) | {"position_embeddings"}
            if pending.keys() != allowed:
                raise ValueError(f"{path}: unexpected or tied vision embedding parameters")
        elif kind == "ssm_mixer":
            allowed = {"A_log", "D"}
            if pending.keys() != allowed:
                raise ValueError(f"{path}: unexpected or tied state-space mixer parameters")
            validate_ssm_mixer(module)
        elif kind == "deberta_mlm_bias":
            allowed = {"bias"}
            if pending.keys() != allowed or pending["bias"].ndim != 1:
                raise ValueError(f"{path}: invalid standalone MLM bias")
        elif kind == "clip_class_embedding":
            validate_clip_class_embedding(module, pending)
            allowed = pending.keys()
        elif kind == "llava_image_newline":
            validate_llava_image_newline(module, pending)
            allowed = pending.keys()
        elif kind == "gemma3_projector":
            validate_gemma3_projector(module, pending)
            allowed = pending.keys()
        elif kind == "convnext_norm":
            validate_convnext_norm(module, pending)
            allowed = pending.keys()
        elif kind == "convnext_layer_scale":
            validate_convnext_scale(module, pending)
            allowed = pending.keys()
        elif kind == "convnext_grn":
            validate_convnext_grn(module, pending)
            allowed = pending.keys()
        elif kind in {"grouped_expert_bank", "grouped_expert_bias"}:
            validate_grouped_experts(module, pending, kind)
            allowed = pending.keys()
        elif kind == "native_router":
            allowed = {"weight"}
            if pending.keys() != allowed or pending["weight"].ndim != 2:
                raise ValueError(f"{path}: invalid native router weight")
        elif kind in _GPTOSS_KINDS:
            validate_gptoss_operator(module, pending, kind)
            allowed = pending.keys()
        elif kind == "dbrx_expert_glu":
            validate_dbrx_experts(module, pending)
            allowed = pending.keys()
        else:
            allowed = {"weight", "bias"}
        if not pending.keys() <= allowed:
            raise ValueError(f"{path}: unsupported parameters {sorted(pending.keys())}")
        for parameter in pending.values():
            if not parameter.is_floating_point() or parameter.layout != torch.strided:
                raise ValueError(f"{path}: layer probes require dense floating-point parameters")
            if parameter.device.type != "cpu":
                raise ValueError(f"{path}: recovery destination must be a materialized CPU model")
        plan.append((path, module, pending, kind))
        seen.update(map(id, pending.values()))
    if not seen:
        raise ValueError("model has no recoverable parameters")
    # The parent observation combines patch projection and position values.
    # Recover the child projection before subtracting it from that observation.
    plan.sort(key=lambda entry: entry[3] == "vision_embeddings")
    if bart_bias:
        bias = model.final_logits_bias
        if (
            bias.shape != (1, model.config.vocab_size)
            or bias.device.type != "cpu"
            or not bias.is_floating_point()
        ):
            raise ValueError("invalid BART final logit bias destination")
        plan.append(("final_logits_bias", model, {"bias": bias}, "bart_logits_bias"))
    plan.extend(correction_probes)
    return plan


def _affine(torch, module, pending, kind, read, batch_size):
    weight = module.weight
    columns = weight if kind == "conv1d" else weight.T
    in_features, out_features = columns.shape
    bias = module.bias
    if bias is not None:
        baseline = read(weight.new_zeros((1, in_features)), (1, out_features))[0]
        if "bias" in pending:
            bias.copy_(baseline)
    if "weight" not in pending:
        return
    # A power of two reduces cancellation from an affine offset. A biasless
    # primitive needs only unit inputs and never goes through subtraction.
    amplitude = 1 if bias is None else 32
    for start in range(0, in_features, batch_size):
        end = min(start + batch_size, in_features)
        rows = torch.arange(end - start, device=weight.device)
        inputs = weight.new_zeros((end - start, in_features))
        inputs[rows, rows + start] = amplitude
        values = read(inputs, (end - start, out_features))
        if bias is not None:
            values = (values.double() - baseline) / amplitude
        columns[start:end].copy_(values)


def _normalization(torch, module, pending, kind, read):
    from torch.nn import functional as F

    weight = module.weight
    shape = (1, *weight.shape)
    baseline = None
    if kind == "layer_norm":
        inputs = weight.new_ones(shape)
        inputs.flatten()[1::2] = -1
        unit = F.layer_norm(inputs, weight.shape, eps=module.eps)
        if module.bias is not None:
            baseline = read(weight.new_zeros(shape), shape)
            if "bias" in pending:
                module.bias.copy_(baseline[0])
    elif kind == "hf_rms_norm":
        inputs = weight.new_full(shape, 32)
        # These HF operators normalize in FP32 and round the multiplier to
        # input dtype before applying gamma.
        variance = torch.tensor(
            32**2 + module.variance_epsilon, dtype=torch.float32, device=weight.device
        )
        unit = (32 * torch.rsqrt(variance)).to(weight.dtype)
    else:
        inputs = weight.new_ones(shape)
        unit = F.rms_norm(inputs, weight.shape, eps=module.eps)
    if "weight" in pending:
        values = read(inputs, shape)
        if baseline is not None:
            values = values.double() - baseline
        weight.copy_((values.double() / unit)[0])


def _vision_embeddings(module, read):
    tokens = _VISION_EMBEDDINGS[_qualified_name(module)]
    positions = module.position_embeddings
    patches = module.patch_embeddings
    pixels = positions.new_zeros((1, patches.num_channels, *patches.image_size))
    combined = read(pixels, tuple(positions.shape))
    prefix = len(tokens)
    projected = patches(pixels)
    if projected.shape != combined[:, prefix:].shape:
        raise ValueError("vision patch grid does not match positional embedding rows")
    positions[:, prefix:].copy_(combined[:, prefix:] - projected)
    positions[:, :prefix].zero_()
    for index, token in enumerate(tokens):
        getattr(module, token).copy_(combined[:, index : index + 1])
    return {
        "special_tokens": list(tokens),
        "representative": "zero_special_position_rows",
        "free_dimensions": prefix * positions.shape[-1],
        "affected_stored_elements": 2 * prefix * positions.shape[-1],
    }


def _validate_probe_options(batch_size, relative_bias_mode):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("probe batch size must be a positive integer")
    if relative_bias_mode not in ("embedding", "attention"):
        raise ValueError("relative bias mode must be embedding or attention")


def recover_from_layer_calls(
    model,
    call_layer,
    *,
    batch_size=128,
    relative_bias_mode="embedding",
    relative_bias_bucket_device="cpu",
    ssm_rate_device="cpu",
    _journal=None,
):
    """Populate a CPU skeleton using only the call_layer(module_path, inputs) oracle.

    The oracle takes a CPU input tensor and returns the primitive's full output
    tensor in the original coordinate order and floating-point dtype. Model
    topology, parameter ties and normalization epsilon come from the skeleton.
    The callback can be served locally or by an instrumented external runtime.

    BART position probes take signed logical positions, with the reserved rows
    addressed by negative positions. Its final_logits_bias path takes one token
    ID and returns the root logits at zero input to the language-model head.
    OPT position probes also take signed logical positions. The Gemma3 lookup
    uses unscaled rows. Gemma-family offset RMSNorm uses FP32 input/output.
    OLMo 2 and 3 RMSNorm probes use at least FP32 input/output.
    The :clip_class probe exposes the CLIP class vector before adding its
    positional embedding. Gemma 3 projectors take chosen normalized features.
    The :image_newline probe exposes the token appended by LLaVA image packing.
    ConvNeXT :layer_scale probes use unit pre-scale features and a zero
    residual. :grn_scale exposes the weighted response before bias and residual.
    Grouped expert banks use paths ending in :expert_<index> for the
    corresponding affine map. Native router paths ending in :router_logits
    expose FP32 logits. :router_correction probes expose the additive
    correction at zero scores in the persistent buffer's dtype.
    Dense GPT-OSS exposes native affine products and biases at virtual
    :router_weight, :router_bias, :gate_up_weight/<index>,
    :gate_up_bias/<index>, :down_weight/<index> and :down_bias/<index>
    paths. :attention_sinks returns the sink logits before softmax.
    DBRX exposes expert products at :w1/<index>, :v1/<index>, and :w2/<index>.
    Whisper position probes take absolute row IDs, as for token embeddings.
    A standalone DeBERTa MLM bias probe returns logits at zero input to its
    tied vocabulary projection.
    In attention mode, T5-family bias probes take a vector of cache positions
    at the attention parent path and return its relative bias tensor. The bucket
    device selects the backend used by the oracle for position bucketing.
    For Mamba mixers, the oracle takes float64 step sizes and returns a tensor
    of shape (steps, 1 + state size, channels) in float64. The first row per
    step holds skip coefficients, and the remaining rows hold cache responses.
    The rate device selects the backend used by the oracle's float32 transform.
    """
    import torch

    _validate_probe_options(batch_size, relative_bias_mode)
    plan = _plan(torch, model, direct=True, relative_bias_mode=relative_bias_mode)
    calls = rows = scalars = recovered = recovered_buffers = buffer_probes = 0
    gauges = []
    relative_bias_gauges = []
    ssm_decays = []
    ssm_log_elements = 0
    gemma_rms_rounding = []
    router_rounding = []

    with torch.no_grad():
        for path, module, pending, kind in plan:
            if kind in _BUFFER_KINDS:
                recovered_buffers += pending["bias"].numel()
                buffer_probes += 1
            else:
                recovered += sum(parameter.numel() for parameter in pending.values())
            dtype = torch.float64 if kind == "ssm_mixer" else next(iter(pending.values())).dtype
            if kind in {"gemma_rms_norm", "native_router"}:
                dtype = torch.float32
            elif kind == "post_multiply_rms_norm" and dtype != torch.float64:
                dtype = torch.float32

            def read(inputs, shape, operation=None):
                nonlocal calls, rows, scalars
                probe_path = path if operation is None else f"{path}:{operation}"
                if _journal is not None:
                    values = _journal.observe(probe_path, inputs, shape, dtype, oracle=call_layer)
                else:
                    result = call_layer(probe_path, inputs)
                    if (
                        not isinstance(result, torch.Tensor)
                        or result.shape != shape
                        or result.dtype != dtype
                    ):
                        raise ValueError(
                            f"{probe_path}: oracle output shape or dtype disagrees with skeleton"
                        )
                    values = result.to("cpu")
                    if not torch.isfinite(values).all():
                        raise ValueError(f"{probe_path}: non-finite oracle output")
                calls += 1
                rows += inputs.shape[0]
                scalars += values.numel()
                return values

            if kind in {"linear", "conv1d"}:
                _affine(torch, module, pending, kind, read, batch_size)
            elif kind in {"grouped_expert_bank", "grouped_expert_bias"}:
                recover_grouped_experts(torch, module, read, batch_size, kind)
            elif kind == "native_router":
                ambiguous = recover_router_weight(torch, module, read, batch_size)
                if ambiguous:
                    router_rounding.append(
                        {"module": path, "ambiguous_parameter_elements": ambiguous}
                    )
            elif kind == "router_correction":
                recover_router_correction(module, read)
            elif kind in _GPTOSS_KINDS:
                recover_gptoss_operator(torch, module, pending, kind, read, batch_size)
            elif kind == "dbrx_expert_glu":
                recover_dbrx_experts(torch, module, pending, read, batch_size)
            elif kind == "ssm_mixer":
                decay = recover_ssm_mixer(torch, module, pending, read, rate_device=ssm_rate_device)
                ssm_decays.append({"module": path, **decay})
                ssm_log_elements += decay["unidentified_parameter_elements"]
            elif kind == "spatial_conv1d":
                from .convolution import recover_conv1d

                recover_conv1d(torch, module, pending, read, batch_size)
            elif kind == "conv2d":
                from .convolution import recover_conv2d

                recover_conv2d(torch, module, pending, read, batch_size)
            elif kind == "conv3d":
                from .convolution import recover_conv3d

                recover_conv3d(torch, module, pending, read, batch_size)
            elif kind in {"embedding", "scaled_embedding", "offset_embedding"}:
                weight = module.weight
                offset = module.offset if kind == "offset_embedding" else 0
                for start in range(0, module.num_embeddings, batch_size):
                    end = min(start + batch_size, module.num_embeddings)
                    inputs = torch.arange(start - offset, end - offset, device=weight.device)
                    values = read(inputs, (end - start, module.embedding_dim))
                    if kind == "scaled_embedding" and module.embed_scale != 1:
                        values = values.double() / module.embed_scale
                    weight[start:end].copy_(values)
            elif kind == "relative_bias":
                from .relative_bias import recover_relative_bias

                gauge = recover_relative_bias(
                    torch, module, read, batch_size, bucket_device=relative_bias_bucket_device
                )
                if gauge["affected_stored_elements"]:
                    relative_bias_gauges.append({"module": path, **gauge})
            elif kind == "bart_logits_bias":
                bias = module.final_logits_bias
                bias.copy_(read(torch.zeros(1, dtype=torch.long), tuple(bias.shape)))
            elif kind == "deberta_mlm_bias":
                bias = module.bias
                hidden = bias.new_zeros((1, module.dense.in_features))
                bias.copy_(read(hidden, (1, bias.numel()))[0])
            elif kind == "vision_embeddings":
                gauges.append({"module": path, **_vision_embeddings(module, read)})
            elif kind == "clip_class_embedding":
                recover_clip_class_embedding(module, pending, read)
            elif kind == "llava_image_newline":
                recover_llava_image_newline(module, pending, read)
            elif kind == "gemma3_projector":
                recover_gemma3_projector(torch, module, read, batch_size)
            elif kind == "convnext_norm":
                recover_convnext_norm(torch, module, pending, read)
            elif kind == "convnext_layer_scale":
                recover_convnext_scale(torch, module, pending, read)
            elif kind == "convnext_grn":
                recover_convnext_grn(torch, module, pending, read)
            elif kind == "group_norm":
                recover_group_norm(torch, module, pending, read)
            elif kind == "gemma_rms_norm":
                ambiguous = recover_gemma_rms(torch, module, read)
                if ambiguous:
                    gemma_rms_rounding.append(
                        {
                            "module": path,
                            "ambiguous_parameter_elements": ambiguous,
                        }
                    )
            elif kind == "post_multiply_rms_norm":
                weight = pending["weight"]
                shape = (1, *weight.shape)
                amplitude = unit_float32_rms_amplitude(module.variance_epsilon, weight.numel())
                inputs = weight.new_full(shape, amplitude, dtype=dtype)
                weight.copy_(read(inputs, shape)[0])
            else:
                _normalization(torch, module, pending, kind, read)

    return {
        "observation_mode": "chosen_inputs_to_individual_layers",
        "parameterized_modules_probed": len(plan) - buffer_probes,
        "unique_parameter_elements_recovered": recovered,
        "persistent_buffer_elements_recovered": recovered_buffers,
        "individually_identified_parameter_elements": recovered
        - sum(gauge["affected_stored_elements"] for gauge in gauges + relative_bias_gauges)
        - ssm_log_elements
        - sum(
            entry["ambiguous_parameter_elements"] for entry in gemma_rms_rounding + router_rounding
        ),
        "embedding_gauges": gauges,
        "relative_bias_mode": relative_bias_mode,
        "relative_bias_gauges": relative_bias_gauges,
        "state_space_decays": ssm_decays,
        "gemma_rms_rounding": gemma_rms_rounding,
        "router_rounding": router_rounding,
        "layer_calls": calls,
        "probe_input_rows": rows,
        "observed_output_scalars": scalars,
        "probe_batch_size": batch_size,
    }


def _observe_bart_logits_bias(torch, model, token):
    # The additive buffer is observable by setting the biasless LM head's
    # input to zero. This avoids reading the buffer or subtracting rounded logits.
    head = model.get_output_embeddings()
    handle = head.register_forward_pre_hook(lambda _layer, args: (torch.zeros_like(args[0]),))
    try:
        tokens = token.unsqueeze(0)
        return model(
            input_ids=tokens, decoder_input_ids=tokens, use_cache=False, return_dict=True
        ).logits[0]
    finally:
        handle.remove()


def _model_wrapper(config, causal, architecture=None):
    is_vision = config.model_type in {"convnext", "convnextv2", "deit", "vit"}
    is_speech = config.model_type in {"moonshine", "speech_to_text", "whisper"}
    is_encoder = (
        config.model_type in ENCODER_MODEL_TYPES
        and not config.is_decoder
        and not config.is_encoder_decoder
    )
    if architecture is not None and not is_encoder:
        raise ValueError("--architecture is supported only for native encoder-only models")
    wrapper = causal
    if is_vision:
        from transformers import AutoModelForImageClassification

        wrapper = AutoModelForImageClassification
    elif is_speech:
        from transformers import AutoModelForSpeechSeq2Seq

        wrapper = AutoModelForSpeechSeq2Seq
    elif config.model_type == "llava_next_video":
        # Older native auto-model registries omit this wrapper.
        from transformers import LlavaNextVideoForConditionalGeneration

        wrapper = LlavaNextVideoForConditionalGeneration
    elif config.model_type in _VISION_LANGUAGE_TYPES:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

        wrapper = AutoModelForImageTextToText
    elif config.is_encoder_decoder:
        from transformers import AutoModelForSeq2SeqLM

        wrapper = AutoModelForSeq2SeqLM
    elif is_encoder:
        wrapper = encoder_auto_model(config, architecture=architecture)
        if architecture is not None:
            config.architectures = [architecture]
    return wrapper, is_vision, is_speech, is_encoder


def _make_layer_skeleton(wrapper, config, dtype):
    if config.model_type == "llava_next_video":
        # The concrete native wrapper uses the PreTrainedModel constructor.
        return wrapper._from_config(config, torch_dtype=dtype).eval()
    return wrapper.from_config(config, trust_remote_code=False, torch_dtype=dtype).eval()


def _save_vision_language_assets(source, destination):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    if (
        getattr(processor, "image_processor", None) is None
        or getattr(processor, "tokenizer", None) is None
    ):
        raise ValueError("source does not provide a native image-text processor")
    for name in _VISION_LANGUAGE_ASSETS:
        path = source / name
        if path.is_file():
            shutil.copyfile(path, destination / name)
    for template in (source / "chat_templates").glob("*.jinja"):
        folder = destination / "chat_templates"
        folder.mkdir(exist_ok=True)
        shutil.copyfile(template, folder / template.name)


def _save_export_assets(tokenizer_class, config, source, destination, *, vision, speech, encoder):
    destination.mkdir(parents=True, exist_ok=True)
    if config.model_type in _VISION_LANGUAGE_TYPES:
        _save_vision_language_assets(source, destination)
        return
    if vision or speech:
        preprocessor = source / "preprocessor_config.json"
        shutil.copyfile(preprocessor, destination / preprocessor.name)
        processor = source / "processor_config.json"
        if speech and processor.is_file():
            shutil.copyfile(processor, destination / processor.name)
    if vision:
        return
    tokenizer_options = {}
    if config.model_type == "speech_to_text":
        tokenizer_options["use_fast"] = False
    if (
        config.model_type in {"t5", "mt5", "umt5"}
        and (source / "spiece.model").is_file()
        and not (source / "tokenizer.json").is_file()
    ):
        # Retain the native SentencePiece byte fallback for rare tokens.
        tokenizer_options["use_fast"] = False
    tokenizer = _load_tokenizer(
        tokenizer_class,
        source,
        model_config=config,
        local_files_only=True,
        trust_remote_code=False,
        **tokenizer_options,
    )
    tokenizer.save_pretrained(destination)
    if encoder and config.model_type == "albert":
        vocab = source / "spiece.model"
        if vocab.is_file() and not (destination / vocab.name).is_file():
            shutil.copyfile(vocab, destination / vocab.name)


def _normalize_config_model_type(config, destination):
    if config.model_type in {"lfm2-vl", "lfm2_vl"}:
        # Transformers 4.57 serializes the class spelling, while AutoConfig
        # looks up the registry spelling used by the published checkpoint.
        path = destination / "config.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        values["model_type"] = "lfm2_vl"
        write_json(path, values)


def _publish_checkpoint(candidate, output, report, write_assets):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rephix-recovery-", dir=output.parent) as temporary:
        staged = Path(temporary) / "checkpoint"
        candidate.save_pretrained(staged, safe_serialization=True)
        _normalize_config_model_type(candidate.config, staged)
        if write_assets is not None:
            write_assets(staged)
        write_json(staged / "layer-recovery.json", report)
        _check_output(output)
        os.rename(staged, output)


def _check_output(output):
    if output.exists() or output.is_symlink():
        raise ValueError("recovery output already exists; choose a new directory")


def _check_disjoint(left, right):
    left_path, right_path = left.resolve(), right.resolve()
    if (
        left_path == right_path
        or left_path in right_path.parents
        or right_path in left_path.parents
    ):
        raise ValueError("source, evidence and recovery output must use separate directories")


def _copy_export_assets(assets, destination):
    for asset in assets.rglob("*"):
        relative = asset.relative_to(assets)
        if asset.is_file() and relative.as_posix() not in {"config.json", "generation_config.json"}:
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(asset, output)


def _evidence_report(journal, evidence_dir):
    return {
        "evidence_dir": str(evidence_dir),
        "evidence_observations_cached": journal.reused_calls,
        "evidence_observations_collected": journal.fresh_calls,
    }


def export_recovered_model(
    source,
    output,
    *,
    device="cpu",
    dtype="float32",
    batch_size=128,
    include_tokenizer=True,
    relative_bias_mode="embedding",
    architecture=None,
    stage_layers=False,
    evidence_dir=None,
):
    """Run the local layer oracle and export a stand-alone Transformers model."""
    source, output = Path(source), Path(output)
    _validate_probe_options(batch_size, relative_bias_mode)
    if not (source / "config.json").is_file():
        raise ValueError("layer recovery requires a local Transformers directory with config.json")
    _check_output(output)
    if source.resolve() in output.resolve().parents:
        raise ValueError("recovery output must be outside the source model directory")
    if evidence_dir is not None:
        evidence_dir = Path(evidence_dir)
        _check_disjoint(source, evidence_dir)
        _check_disjoint(output, evidence_dir)

    torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer = _deps()
    device = _device(torch, device)
    config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("layer recovery requires a dense, nonquantized model")
    model_dtype = {"auto": "auto", "float32": torch.float32, "float64": torch.float64}.get(dtype)
    if model_dtype is None:
        raise ValueError("layer recovery dtype must be auto, float32, or float64")
    source_architectures = config.architectures
    auto_model, is_vision, is_speech, is_encoder = _model_wrapper(
        config, AutoModelForCausalLM, architecture
    )
    is_vision_language = config.model_type in _VISION_LANGUAGE_TYPES
    preprocessor = source / "preprocessor_config.json"
    if (
        include_tokenizer
        and (is_vision or is_speech or is_vision_language)
        and not preprocessor.is_file()
    ):
        raise ValueError("source has no preprocessor_config.json; use --without-preprocessor")
    target, loading = auto_model.from_pretrained(
        source,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=model_dtype,
        output_loading_info=True,
    )
    if any(loading[key] for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"source checkpoint did not fully load: {loading}")
    target.eval()
    stager = LayerStager(torch, device) if stage_layers and device != "cpu" else None
    if not stage_layers:
        target.to(device)
    candidate = _make_layer_skeleton(auto_model, copy.deepcopy(target.config), target.dtype)
    candidate.tie_weights()
    # Some loaders keep selected layers in float32 in half-precision checkpoints.
    # Mirror their dtype metadata while preserving the skeleton's parameter ties.
    for name, parameter in candidate.named_parameters():
        oracle_dtype = target.get_parameter(name).dtype
        if parameter.dtype != oracle_dtype:
            parameter.data = parameter.detach().to(oracle_dtype)

    def save_extras(destination):
        _save_export_assets(
            AutoTokenizer,
            config,
            source,
            destination,
            vision=is_vision,
            speech=is_speech,
            encoder=is_encoder,
        )

    provenance = {
        "model_type": target.config.model_type,
        "architecture": type(target).__name__,
        "oracle_device": device,
        "oracle_dtype": str(target.dtype),
        "oracle_placement": "staged" if stage_layers else "resident",
        "ignored_source_keys": sorted(loading["unexpected_keys"]),
        "image_processor_exported" if is_vision else "tokenizer_exported": include_tokenizer,
    }
    if is_speech:
        provenance["audio_feature_extractor_exported"] = include_tokenizer
    if is_vision_language:
        provenance["multimodal_processor_exported"] = include_tokenizer
    if is_encoder:
        provenance["source_architectures"] = source_architectures

    journal_context = nullcontext()
    write_assets = save_extras if include_tokenizer else None
    if evidence_dir is not None:
        from .layer_bundle import bundle_binding, prepare_bundle, source_fingerprint
        from .layer_journal import LayerJournal

        def save_scaffold(destination):
            scaffold = copy.deepcopy(target.config)
            scaffold.architectures = [type(target).__name__]
            scaffold.save_pretrained(destination)
            _normalize_config_model_type(scaffold, destination)
            if getattr(target, "generation_config", None) is not None:
                target.generation_config.save_pretrained(destination)
            if include_tokenizer:
                save_extras(destination)

        metadata = {
            "probe_plan_version": _PROBE_PLAN_VERSION,
            "source_files": source_fingerprint(source),
            "load_dtype": dtype,
            "batch_size": batch_size,
            "relative_bias_mode": relative_bias_mode,
            "parameter_dtype_overrides": {
                name: str(parameter.dtype)
                for name, parameter in candidate.named_parameters()
                if parameter.dtype != target.dtype
            },
            "provenance": provenance,
        }
        _, assets = prepare_bundle(evidence_dir, metadata, save_scaffold)
        journal_context = LayerJournal(
            evidence_dir / _JOURNAL_FILENAME, torch, binding=bundle_binding(evidence_dir)
        )

        def write_assets(destination):
            _copy_export_assets(assets, destination)

    host_layer_calls = 0

    def observe(path, inputs):
        nonlocal host_layer_calls
        path, _, operation = path.partition(":")
        if path == "final_logits_bias" and config.model_type == "bart":
            if stager is not None:
                # The virtual bias probe needs a full forward. Leave it on
                # CPU rather than staging the entire model for one observation.
                stager.clear()
                host_layer_calls += 1
                return _observe_bart_logits_bias(torch, target, inputs)
            return _observe_bart_logits_bias(torch, target, inputs.to(device))
        module = target.get_submodule(path)
        name = _qualified_name(module)
        if name in LLAVA_IMAGE_NEWLINE_MODELS:
            if stager is not None:
                stager.clear()
                host_layer_calls += 1
                return observe_llava_image_newline(torch, module, inputs)
            return observe_llava_image_newline(torch, module, inputs.to(device))
        if name == DBRX_EXPERT_GLU:
            parent = target.get_submodule(path.rpartition(".")[0])
            if stager is not None:
                (parent,) = stager.get(parent)
            return observe_dbrx_experts(torch, parent, inputs.to(device), operation)
        if name == _DEBERTA_MLM_HEAD:
            embeddings = target.get_input_embeddings()
            if stager is not None:
                module, embeddings = stager.get(module, embeddings)
            handle = module.LayerNorm.register_forward_hook(
                lambda _layer, _args, values: torch.zeros_like(values)
            )
            try:
                return module(inputs.to(device), embeddings)
            finally:
                handle.remove()
        if stager is not None:
            (module,) = stager.get(module)
        if name == CLIP_VISION_EMBEDDINGS:
            return observe_clip_class_embedding(torch, module, inputs.to(device))
        if name == GEMMA3_PROJECTOR:
            return observe_gemma3_projector(module, inputs.to(device))
        if name == CONVNEXT_LAYER_SCALE:
            return observe_convnext_scale(module, inputs.to(device))
        if name == CONVNEXT_GRN and operation == "grn_scale":
            return observe_convnext_grn_scale(module, inputs.to(device))
        if name in ROUTER_MODULES:
            return observe_router(torch, module, inputs.to(device), operation)
        if name in GROUPED_EXPERT_BANKS or name in GROUPED_EXPERT_BIASES:
            return observe_grouped_experts(torch, module, inputs.to(device), operation)
        if name in GPTOSS_OPERATORS:
            return observe_gptoss_operator(torch, module, inputs.to(device), operation)
        if name in SSM_MIXERS:
            return observe_ssm_mixer(torch, module, inputs.to(device))
        if name == _BART + "BartLearnedPositionalEmbedding":
            return module(inputs[None].to(device), past_key_values_length=int(inputs[0]))[0]
        if name == _OPT_POSITIONS:
            return module(None, position_ids=inputs.to(device))
        if name == GEMMA3_SCALED_EMBEDDING:
            return torch.nn.Embedding.forward(module, inputs.to(device))
        if name == _WHISPER_POSITIONS:
            return module(inputs[None].to(device), past_key_values_length=int(inputs[0]))
        if relative_bias_mode == "attention" and hasattr(module, "relative_attention_bias"):
            from .relative_bias import relative_bias_spec

            return module.compute_bias(
                inputs.numel(),
                relative_bias_spec(module).key_length,
                cache_position=inputs.to(device),
            )
        return module(inputs.to(device))

    try:
        with journal_context as journal:
            report = recover_from_layer_calls(
                candidate,
                observe,
                batch_size=batch_size,
                relative_bias_mode=relative_bias_mode,
                relative_bias_bucket_device=device,
                ssm_rate_device=device,
                _journal=journal,
            )
            if journal is not None:
                journal.finish()
                report.update(_evidence_report(journal, evidence_dir))
    finally:
        if stager is not None:
            stager.clear()
    if not is_vision:
        candidate.generation_config = target.generation_config
    report.update(
        provenance,
        source=str(source),
        checkpoint=str(output),
        oracle_staged_groups=stager.moves if stager is not None else 0,
        oracle_peak_staged_state_bytes=stager.peak_state_bytes if stager is not None else 0,
        oracle_host_layer_calls=(
            report.get("evidence_observations_collected", report["layer_calls"])
            if device == "cpu"
            else host_layer_calls
        ),
    )
    target = None

    _publish_checkpoint(candidate, output, report, write_assets)
    return report


def replay_layer_evidence(evidence_dir, output):
    """Build a standalone checkpoint from a complete local evidence bundle."""
    from .layer_bundle import bundle_binding, open_bundle
    from .layer_journal import LayerJournal

    evidence_dir, output = Path(evidence_dir), Path(output)
    _check_output(output)
    _check_disjoint(evidence_dir, output)
    metadata, assets = open_bundle(evidence_dir)
    if metadata.get("probe_plan_version") != _PROBE_PLAN_VERSION:
        raise ValueError("unsupported layer evidence probe plan version")
    torch, AutoConfig, causal, _ = _deps()
    from transformers import GenerationConfig

    with LayerJournal(
        evidence_dir / _JOURNAL_FILENAME, torch, binding=bundle_binding(evidence_dir), readonly=True
    ) as journal:
        provenance = metadata["provenance"]
        config = AutoConfig.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
        wrapper, _, _, _ = _model_wrapper(config, causal)
        dtypes = {
            str(dtype): dtype
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        }
        oracle_dtype = dtypes[provenance["oracle_dtype"]]
        candidate = _make_layer_skeleton(wrapper, config, oracle_dtype)
        candidate.tie_weights()
        if type(candidate).__name__ != provenance["architecture"]:
            raise ValueError("layer evidence scaffold architecture disagrees with capture")
        overrides = metadata["parameter_dtype_overrides"]
        for name, parameter in candidate.named_parameters():
            dtype = dtypes[overrides[name]] if name in overrides else oracle_dtype
            if parameter.dtype != dtype:
                parameter.data = parameter.detach().to(dtype)
        if (assets / "generation_config.json").is_file():
            candidate.generation_config = GenerationConfig.from_pretrained(
                assets, local_files_only=True
            )
        device = provenance["oracle_device"]
        relative_bias_mode = metadata["relative_bias_mode"]
        uses_bucket_backend = relative_bias_mode == "attention" and config.model_type in {
            "t5",
            "mt5",
            "umt5",
        }
        if uses_bucket_backend or config.model_type in {"mamba", "falcon_mamba", "jamba"}:
            _device(torch, device)
        report = recover_from_layer_calls(
            candidate,
            None,
            batch_size=metadata["batch_size"],
            relative_bias_mode=relative_bias_mode,
            relative_bias_bucket_device=device,
            ssm_rate_device=device,
            _journal=journal,
        )
        journal.finish()
        report.update(_evidence_report(journal, evidence_dir))
    report.update(
        provenance,
        source=None,
        checkpoint=str(output),
        oracle_staged_groups=0,
        oracle_peak_staged_state_bytes=0,
        oracle_host_layer_calls=0,
    )
    _publish_checkpoint(
        candidate, output, report, lambda staged: _copy_export_assets(assets, staged)
    )
    return report


def _command(args):
    return export_recovered_model(
        args.model,
        args.output,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        include_tokenizer=not args.without_tokenizer,
        relative_bias_mode=args.relative_bias_mode,
        architecture=args.architecture,
        stage_layers=args.stage_layers,
        evidence_dir=args.evidence_dir,
    )


def add_subcommands(commands):
    recover = commands.add_parser(
        "recover-layers", help="export a checkpoint using chosen inputs to known local layers"
    )
    recover.add_argument("--model", required=True, help="local Transformers safetensors directory")
    recover.add_argument("--output", required=True, help="new Transformers checkpoint directory")
    recover.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    recover.add_argument("--dtype", choices=("float32", "float64", "auto"), default="float32")
    recover.add_argument("--batch-size", type=int, default=128)
    recover.add_argument(
        "--evidence-dir", help="durable layer observation bundle to create or resume"
    )
    recover.add_argument(
        "--architecture", help="native encoder class to use instead of config.json"
    )
    recover.add_argument(
        "--stage-layers",
        action="store_true",
        help="keep the source on CPU and stage each probe layer",
    )
    recover.add_argument(
        "--relative-bias-mode",
        choices=("embedding", "attention"),
        default="embedding",
        help="observe T5-family lookup tables or only reachable attention biases",
    )
    recover.add_argument(
        "--without-tokenizer",
        "--without-preprocessor",
        dest="without_tokenizer",
        action="store_true",
    )
    recover.set_defaults(func=_command)
    replay = commands.add_parser("replay-layers", help="rebuild a checkpoint from sealed evidence")
    replay.add_argument("--evidence-dir", required=True, help="complete layer observation bundle")
    replay.add_argument("--output", required=True, help="new Transformers checkpoint directory")
    replay.set_defaults(func=lambda args: replay_layer_evidence(args.evidence_dir, args.output))
