"""Recover a known model from chosen-input, coordinate-preserving layer calls."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from pathlib import Path

from .hf import _deps, _device, _load_tokenizer
from .io import write_json

_HF_RMS_NORMS = {
    "transformers.models.llama.modeling_llama.LlamaRMSNorm",
    "transformers.models.mistral.modeling_mistral.MistralRMSNorm",
    "transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm",
    "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm",
}
_T5_RMS_NORMS = {
    "transformers.models.mt5.modeling_mt5.MT5LayerNorm",
    "transformers.models.t5.modeling_t5.T5LayerNorm",
    "transformers.models.umt5.modeling_umt5.UMT5LayerNorm",
}
_BART = "transformers.models.bart.modeling_bart."
_WHISPER_POSITIONS = "transformers.models.whisper.modeling_whisper.WhisperPositionalEmbedding"
_RELATIVE_BIAS_EMBEDDING = ".relative_attention_bias"
_VISION_EMBEDDINGS = {
    "transformers.models.vit.modeling_vit.ViTEmbeddings": ("cls_token",),
    "transformers.models.deit.modeling_deit.DeiTEmbeddings": ("cls_token", "distillation_token"),
}


def _qualified_name(module):
    cls = type(module)
    return f"{cls.__module__}.{cls.__name__}"


def _operator(torch, module, *, direct):
    cls = type(module)
    qualified_name = _qualified_name(module)
    if cls is torch.nn.Linear:
        return "linear"
    if direct and cls is torch.nn.Conv1d:
        from .convolution import validate_conv1d

        validate_conv1d(module)
        return "spatial_conv1d"
    if cls is torch.nn.Conv2d:
        from .convolution import validate_conv2d

        validate_conv2d(module)
        return "conv2d"
    if qualified_name == "transformers.pytorch_utils.Conv1D":
        return "conv1d"
    if cls is torch.nn.Embedding and module.max_norm is None:
        return "embedding"
    if direct and qualified_name == _WHISPER_POSITIONS and module.max_norm is None:
        return "embedding"
    if direct and qualified_name == _BART + "BartScaledWordEmbedding":
        return "scaled_embedding"
    if direct and qualified_name == _BART + "BartLearnedPositionalEmbedding":
        return "offset_embedding"
    if cls is torch.nn.LayerNorm and module.weight.numel() > 1:
        return "layer_norm"
    if qualified_name in _HF_RMS_NORMS or (direct and qualified_name in _T5_RMS_NORMS):
        return "hf_rms_norm"
    if cls is getattr(torch.nn, "RMSNorm", None):
        return "rms_norm"
    if qualified_name in _VISION_EMBEDDINGS and getattr(module, "mask_token", None) is None:
        return "vision_embeddings"
    raise ValueError(f"unsupported parameterized operator {qualified_name}")


def _plan(torch, model, *, direct=False, relative_bias_mode="embedding"):
    names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    buffers = model.state_dict().keys() - names
    bart_bias = direct and _qualified_name(model) == _BART + "BartForConditionalGeneration"
    unknown_buffers = sorted(buffers - ({"final_logits_bias"} if bart_bias else set()))
    if unknown_buffers:
        raise ValueError(f"persistent state has no layer probe: {', '.join(unknown_buffers[:5])}")

    plan = []
    seen = set()
    for path, module in model.named_modules():
        parameters = dict(module.named_parameters(recurse=False, remove_duplicate=False))
        pending = {name: value for name, value in parameters.items() if id(value) not in seen}
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


def recover_from_layer_calls(
    model,
    call_layer,
    *,
    batch_size=128,
    relative_bias_mode="embedding",
    relative_bias_bucket_device="cpu",
):
    """Populate a CPU skeleton using only the call_layer(module_path, inputs) oracle.

    The oracle takes a CPU input tensor and returns the primitive's full output
    tensor in the original coordinate order and floating-point dtype. Model
    topology, parameter ties and normalization epsilon come from the skeleton.
    The callback can be served locally or by an instrumented external runtime.

    BART position probes take signed logical positions, with the reserved rows
    addressed by negative positions. Its final_logits_bias path takes one token
    ID and returns the root logits at zero input to the language-model head.
    Whisper position probes take absolute row IDs, as for token embeddings.
    In attention mode, T5-family bias probes take a vector of cache positions
    at the attention parent path and return its relative bias tensor. The bucket
    device selects the backend used by the oracle for position bucketing.
    """
    import torch

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("probe batch size must be a positive integer")
    if relative_bias_mode not in ("embedding", "attention"):
        raise ValueError("relative bias mode must be embedding or attention")
    plan = _plan(torch, model, direct=True, relative_bias_mode=relative_bias_mode)
    calls = rows = scalars = recovered = recovered_buffers = 0
    gauges = []
    relative_bias_gauges = []

    with torch.no_grad():
        for path, module, pending, kind in plan:
            if kind == "bart_logits_bias":
                recovered_buffers += module.final_logits_bias.numel()
            else:
                recovered += sum(parameter.numel() for parameter in pending.values())
            dtype = next(iter(pending.values())).dtype

            def read(inputs, shape):
                nonlocal calls, rows, scalars
                result = call_layer(path, inputs)
                if (
                    not isinstance(result, torch.Tensor)
                    or result.shape != shape
                    or result.dtype != dtype
                ):
                    raise ValueError(
                        f"{path}: oracle output shape or dtype disagrees with skeleton"
                    )
                values = result.to("cpu")
                if not torch.isfinite(values).all():
                    raise ValueError(f"{path}: non-finite oracle output")
                calls += 1
                rows += inputs.shape[0]
                scalars += values.numel()
                return values

            if kind in {"linear", "conv1d"}:
                _affine(torch, module, pending, kind, read, batch_size)
            elif kind == "spatial_conv1d":
                from .convolution import recover_conv1d

                recover_conv1d(torch, module, pending, read, batch_size)
            elif kind == "conv2d":
                from .convolution import recover_conv2d

                recover_conv2d(torch, module, pending, read, batch_size)
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
            elif kind == "vision_embeddings":
                gauges.append({"module": path, **_vision_embeddings(module, read)})
            else:
                _normalization(torch, module, pending, kind, read)

    return {
        "observation_mode": "chosen_inputs_to_individual_layers",
        "parameterized_modules_probed": len(plan) - int(recovered_buffers > 0),
        "unique_parameter_elements_recovered": recovered,
        "persistent_buffer_elements_recovered": recovered_buffers,
        "individually_identified_parameter_elements": recovered
        - sum(gauge["affected_stored_elements"] for gauge in gauges + relative_bias_gauges),
        "embedding_gauges": gauges,
        "relative_bias_mode": relative_bias_mode,
        "relative_bias_gauges": relative_bias_gauges,
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


def export_recovered_model(
    source,
    output,
    *,
    device="cpu",
    dtype="float32",
    batch_size=128,
    include_tokenizer=True,
    relative_bias_mode="embedding",
):
    """Run the local layer oracle and export a stand-alone Transformers model."""
    source, output = Path(source), Path(output)
    if not (source / "config.json").is_file():
        raise ValueError("layer recovery requires a local Transformers directory with config.json")
    if output.exists() or output.is_symlink():
        raise ValueError("recovery output already exists; choose a new directory")
    if source.resolve() in output.resolve().parents:
        raise ValueError("recovery output must be outside the source model directory")

    torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer = _deps()
    device = _device(torch, device)
    config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("layer recovery requires a dense, nonquantized model")
    model_dtype = {"auto": "auto", "float32": torch.float32, "float64": torch.float64}.get(dtype)
    if model_dtype is None:
        raise ValueError("layer recovery dtype must be auto, float32, or float64")
    is_vision = config.model_type in {"vit", "deit"}
    is_speech = config.model_type == "whisper"
    preprocessor = source / "preprocessor_config.json"
    if include_tokenizer and (is_vision or is_speech) and not preprocessor.is_file():
        raise ValueError("source has no preprocessor_config.json; use --without-preprocessor")
    auto_model = AutoModelForCausalLM
    if is_vision:
        from transformers import AutoModelForImageClassification

        auto_model = AutoModelForImageClassification
    elif is_speech:
        from transformers import AutoModelForSpeechSeq2Seq

        auto_model = AutoModelForSpeechSeq2Seq
    elif config.is_encoder_decoder:
        from transformers import AutoModelForSeq2SeqLM

        auto_model = AutoModelForSeq2SeqLM
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
    target.eval().to(device)
    candidate = auto_model.from_config(
        copy.deepcopy(target.config), trust_remote_code=False, torch_dtype=target.dtype
    ).eval()
    candidate.tie_weights()
    # Some loaders keep selected layers in float32 in half-precision checkpoints.
    # Mirror their dtype metadata while preserving the skeleton's parameter ties.
    for name, parameter in candidate.named_parameters():
        oracle_dtype = target.get_parameter(name).dtype
        if parameter.dtype != oracle_dtype:
            parameter.data = parameter.detach().to(oracle_dtype)

    def observe(path, inputs):
        if path == "final_logits_bias" and config.model_type == "bart":
            return _observe_bart_logits_bias(torch, target, inputs.to(device))
        module = target.get_submodule(path)
        name = _qualified_name(module)
        if name == _BART + "BartLearnedPositionalEmbedding":
            return module(inputs[None].to(device), past_key_values_length=int(inputs[0]))[0]
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

    report = recover_from_layer_calls(
        candidate,
        observe,
        batch_size=batch_size,
        relative_bias_mode=relative_bias_mode,
        relative_bias_bucket_device=device,
    )
    if not is_vision:
        candidate.generation_config = target.generation_config
    report.update(
        model_type=target.config.model_type,
        architecture=type(target).__name__,
        source=str(source),
        checkpoint=str(output),
        oracle_device=device,
        oracle_dtype=str(target.dtype),
        ignored_source_keys=loading["unexpected_keys"],
    )
    report["image_processor_exported" if is_vision else "tokenizer_exported"] = include_tokenizer
    if is_speech:
        report["audio_feature_extractor_exported"] = include_tokenizer
    target = None

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rephix-recovery-", dir=output.parent) as temporary:
        staged = Path(temporary) / "checkpoint"
        candidate.save_pretrained(staged, safe_serialization=True)
        if include_tokenizer and (is_vision or is_speech):
            shutil.copyfile(preprocessor, staged / preprocessor.name)
            processor = source / "processor_config.json"
            if is_speech and processor.is_file():
                shutil.copyfile(processor, staged / processor.name)
        if include_tokenizer and not is_vision:
            tokenizer_options = {}
            if (
                config.model_type in {"t5", "mt5", "umt5"}
                and (source / "spiece.model").is_file()
                and not (source / "tokenizer.json").is_file()
            ):
                # Preserve the native SentencePiece byte fallback instead of
                # synthesizing a fast tokenizer that can change rare tokens.
                tokenizer_options["use_fast"] = False
            tokenizer = _load_tokenizer(
                AutoTokenizer,
                source,
                model_config=config,
                local_files_only=True,
                trust_remote_code=False,
                **tokenizer_options,
            )
            tokenizer.save_pretrained(staged)
        write_json(staged / "layer-recovery.json", report)
        if output.exists() or output.is_symlink():
            raise ValueError("recovery output already exists; choose a new directory")
        os.rename(staged, output)
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
