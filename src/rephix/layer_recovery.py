"""Recover a known model from chosen-input, coordinate-preserving layer calls."""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

from .hf import _deps, _device, _load_tokenizer
from .io import write_json

_HF_RMS_NORMS = {
    "transformers.models.llama.modeling_llama.LlamaRMSNorm",
    "transformers.models.mistral.modeling_mistral.MistralRMSNorm",
    "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm",
}


def _operator(torch, module):
    cls = type(module)
    qualified_name = f"{cls.__module__}.{cls.__name__}"
    if cls is torch.nn.Linear:
        return "linear"
    if qualified_name == "transformers.pytorch_utils.Conv1D":
        return "conv1d"
    if cls is torch.nn.Embedding and module.max_norm is None:
        return "embedding"
    if cls is torch.nn.LayerNorm and module.weight.numel() > 1:
        return "layer_norm"
    if qualified_name in _HF_RMS_NORMS:
        return "hf_rms_norm"
    if cls is getattr(torch.nn, "RMSNorm", None):
        return "rms_norm"
    raise ValueError(f"unsupported parameterized operator {qualified_name}")


def _plan(torch, model):
    names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    buffers = sorted(model.state_dict().keys() - names)
    if buffers:
        raise ValueError(f"persistent state has no layer probe: {', '.join(buffers[:5])}")

    plan = []
    seen = set()
    for path, module in model.named_modules():
        parameters = dict(module.named_parameters(recurse=False, remove_duplicate=False))
        pending = {name: value for name, value in parameters.items() if id(value) not in seen}
        if not pending:
            continue
        kind = _operator(torch, module)
        if not pending.keys() <= {"weight", "bias"}:
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


def recover_from_layer_calls(model, call_layer, *, batch_size=128):
    """Populate a CPU skeleton using only the call_layer(module_path, inputs) oracle.

    The oracle takes a CPU input tensor and returns the primitive's full output
    tensor in the original coordinate order and floating-point dtype. Model
    topology, parameter ties and normalization epsilon come from the skeleton.
    The callback can be served locally or by an instrumented external runtime.
    """
    import torch

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("probe batch size must be a positive integer")
    plan = _plan(torch, model)
    calls = rows = scalars = recovered = 0

    with torch.no_grad():
        for path, module, pending, kind in plan:
            recovered += sum(parameter.numel() for parameter in pending.values())

            def read(inputs, shape):
                nonlocal calls, rows, scalars
                result = call_layer(path, inputs)
                if (
                    not isinstance(result, torch.Tensor)
                    or result.shape != shape
                    or result.dtype != module.weight.dtype
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
            elif kind == "embedding":
                weight = module.weight
                for start in range(0, module.num_embeddings, batch_size):
                    end = min(start + batch_size, module.num_embeddings)
                    inputs = torch.arange(start, end, device=weight.device)
                    weight[start:end].copy_(read(inputs, (end - start, module.embedding_dim)))
            else:
                _normalization(torch, module, pending, kind, read)

    return {
        "observation_mode": "chosen_inputs_to_individual_layers",
        "parameterized_modules_probed": len(plan),
        "unique_parameter_elements_recovered": recovered,
        "layer_calls": calls,
        "probe_input_rows": rows,
        "observed_output_scalars": scalars,
        "probe_batch_size": batch_size,
    }


def export_recovered_model(
    source, output, *, device="cpu", dtype="float32", batch_size=128, include_tokenizer=True
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
    model_dtype = {"auto": "auto", "float32": torch.float32}.get(dtype)
    if model_dtype is None:
        raise ValueError("layer recovery dtype must be auto or float32")
    target, loading = AutoModelForCausalLM.from_pretrained(
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
    candidate = AutoModelForCausalLM.from_config(
        copy.deepcopy(target.config), trust_remote_code=False, torch_dtype=target.dtype
    ).eval()
    candidate.tie_weights()

    def observe(path, inputs):
        return target.get_submodule(path)(inputs.to(device))

    report = recover_from_layer_calls(candidate, observe, batch_size=batch_size)
    candidate.generation_config = target.generation_config
    report.update(
        model_type=target.config.model_type,
        architecture=type(target).__name__,
        source=str(source),
        checkpoint=str(output),
        oracle_device=device,
        oracle_dtype=str(target.dtype),
        ignored_source_keys=loading["unexpected_keys"],
        tokenizer_exported=include_tokenizer,
    )
    target = None

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rephix-recovery-", dir=output.parent) as temporary:
        staged = Path(temporary) / "checkpoint"
        candidate.save_pretrained(staged, safe_serialization=True)
        if include_tokenizer:
            tokenizer = _load_tokenizer(
                AutoTokenizer,
                source,
                model_config=config,
                local_files_only=True,
                trust_remote_code=False,
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
    )


def add_subcommands(commands):
    recover = commands.add_parser(
        "recover-layers", help="export a checkpoint using chosen inputs to known local layers"
    )
    recover.add_argument("--model", required=True, help="local Transformers safetensors directory")
    recover.add_argument("--output", required=True, help="new Transformers checkpoint directory")
    recover.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    recover.add_argument("--dtype", choices=("float32", "auto"), default="float32")
    recover.add_argument("--batch-size", type=int, default=128)
    recover.add_argument("--without-tokenizer", action="store_true")
    recover.set_defaults(func=_command)
