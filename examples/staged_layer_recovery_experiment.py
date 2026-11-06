"""Measure accelerator staging and independently audit exported causal models."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

from rephix.hf import _device, _generation_config, _load_tokenizer
from rephix.io import file_sha256, write_json
from rephix.layer_recovery import export_recovered_model
from rephix.weight_audit import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    _inventory,
    _open_shard,
    _selections,
    compare_checkpoints,
)

_PROMPTS = (
    "At dusk, the lighthouse keeper climbed the stairs and found",
    "Translate into French: The museum closes at six.\nFrench:",
    "The sum of 19 and 23 is",
    "def square(value):\n    return",
    "Why do maple leaves change color in autumn?\nAnswer:",
    "Complete this list of three ocean animals: dolphin, octopus,",
)


def _cuda_memory(torch, device):
    if device != "cuda":
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "free_bytes": free,
        "total_bytes": total,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _peak_host_rss():
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return None


def _recover(args):
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    device = _device(torch, args.device)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before = _cuda_memory(torch, device)
    started = time.monotonic()
    outcome = {}
    try:
        outcome["recovery"] = export_recovered_model(
            args.model,
            args.output,
            device=device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            include_tokenizer=not args.without_tokenizer,
            stage_layers=args.stage_layers,
        )
    except Exception as exc:
        outcome["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if device == "cuda":
            torch.cuda.synchronize()
        metrics = {
            "schema_version": 1,
            "source": str(args.model),
            "candidate": str(args.output),
            "stage_layers": args.stage_layers,
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "dtype": args.dtype,
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "threads": args.threads,
            "batch_size": args.batch_size,
            "cuda_before": before,
            "cuda_after": _cuda_memory(torch, device),
            "process_peak_rss_bytes": _peak_host_rss(),
            "elapsed_seconds": time.monotonic() - started,
            **outcome,
        }
        write_json(args.metrics, metrics)


def _hashes(path):
    suffixes = {".safetensors", ".json", ".model", ".txt", ".jinja"}
    return {
        file.name: file_sha256(file)
        for file in sorted(path.iterdir())
        if file.is_file() and file.suffix in suffixes
    }


def _all_finite_exact(torch, reference, candidate, aliases=None):
    refs, cands = _inventory(reference), _inventory(candidate)
    if not refs or (aliases is None and refs.keys() != cands.keys()):
        return None
    pairs = defaultdict(list)
    mapped = {}
    for name, a in sorted(refs.items()):
        target = name
        if name not in cands:
            options = cands.keys() & set(aliases.get(name, ()))
            if len(options) != 1:
                return None
            (target,) = options
        b = cands[target]
        if a.shape != b.shape or a.dtype != b.dtype or a.dtype not in {"BF16", "F16", "F32", "F64"}:
            return None
        mapped[name] = target
        pairs[a.shard, b.shard].append((name, target))
    if set(mapped.values()) != cands.keys():
        return None
    for (left, right), names in pairs.items():
        with _open_shard(left) as a, _open_shard(right) as b:
            for name, target in names:
                x, y = a.get_slice(name), b.get_slice(target)
                for selection in _selections(refs[name].shape):
                    reference_chunk = x[selection]
                    if not torch.equal(reference_chunk, y[selection]):
                        return None
                    if not bool(torch.isfinite(reference_chunk).all()):
                        raise ValueError(f"Non-finite reference values in {left} tensor {name!r}")
    elements = sum(entry.size for entry in refs.values())
    if not elements:
        return None
    report = {
        "method": "finite_exact_equality_by_stored_tensor",
        "scope": "raw stored safetensors tensors, including buffers, compared by name and shape",
        "reference_tensors": len(refs),
        "candidate_tensors": len(cands),
        "compared_tensors": len(refs),
        "reference_elements": elements,
        "candidate_elements": sum(entry.size for entry in cands.values()),
        "compared_elements": elements,
        "exactly_equal_elements": elements,
        "within_tolerance_elements": elements,
        "coverage": 1.0,
        "fraction_exactly_equal": 1.0,
        "fraction_within_tolerance": 1.0,
        "relative_l2": 0.0,
        "zero_filled_relative_l2": 0.0,
        "atol": DEFAULT_ATOL,
        "rtol": DEFAULT_RTOL,
        "missing_tensors": [],
        "extra_tensors": [],
        "shape_mismatches": {},
        "dtype_mismatches": {},
    }
    if aliases is not None:
        report["scope"] = (
            "original safetensors values with aliases verified on the loaded candidate"
        )
        report["mapped_names"] = {name: target for name, target in mapped.items() if name != target}
    return report


def _audit(torch, reference, candidate):
    exact = _all_finite_exact(torch, reference, candidate)
    if exact is not None:
        return exact
    result = compare_checkpoints(reference, candidate)
    result.pop("reference")
    result.pop("candidate")
    return {"method": "streaming_numeric_tolerance", **result}


def _capture(torch, path, device, dtype, max_new_tokens):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    options = {"local_files_only": True, "trust_remote_code": False}
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, use_safetensors=True, torch_dtype=dtype, **options
        )
        .eval()
        .to(device)
    )
    tokenizer = _load_tokenizer(AutoTokenizer, path, model_config=model.config, **options)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        list(_PROMPTS), padding=True, return_token_type_ids=False, return_tensors="pt"
    ).to(device)
    mask = inputs.attention_mask
    position_ids = (mask.cumsum(dim=-1) - 1).clamp(min=0)
    options = (
        {"logits_to_keep": 1}
        if "logits_to_keep" in inspect.signature(model.forward).parameters
        else {}
    )
    with torch.inference_mode():
        logits = model(**inputs, position_ids=position_ids, use_cache=False, **options).logits
        logits = logits[:, -1].double().cpu()
        config = _generation_config(model, tokenizer, max_new_tokens)
        config.use_cache = True
        generated = model.generate(**inputs, generation_config=config)
        completions = generated[:, inputs.input_ids.shape[1] :].tolist()
    tokens = [row[keep.bool()].tolist() for row, keep in zip(inputs.input_ids, mask)]
    registrations = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        registrations[id(parameter)].append(name)
    aliases = {name: names for names in registrations.values() if len(names) > 1 for name in names}
    return logits, completions, tokens, str(model.dtype), aliases


def _behavior(torch, baseline, recovered, max_new_tokens):
    a, completions_a, ids_a, dtype_a, _ = baseline
    b, completions_b, ids_b, dtype_b, _ = recovered
    if ids_a != ids_b or a.shape != b.shape:
        raise ValueError("the models disagree on held-out token coordinates")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("the held-out logits contain non-finite values")
    difference = a - b
    return {
        "prompt_count": len(ids_a),
        "prompt_token_ids_match": True,
        "logit_coordinates": a.numel(),
        "exact_logit_coordinates": int((a == b).sum()),
        "logit_rmse": float(difference.square().mean().sqrt()),
        "logit_max_abs_error": float(difference.abs().max()),
        "next_token_top1_matches": int((a.argmax(dim=1) == b.argmax(dim=1)).sum()),
        "exact_cached_greedy_matches": sum(x == y for x, y in zip(completions_a, completions_b)),
        "greedy_prompts": len(completions_a),
        "max_new_tokens": max_new_tokens,
        "reference_inference_dtype": dtype_a,
        "candidate_inference_dtype": dtype_b,
        "candidate_completion_ids": completions_b,
    }


def _evaluate(args):
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    device = _device(torch, args.device)
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    started = time.monotonic()
    reference_hashes = _hashes(args.reference)
    baseline = _capture(torch, args.reference, device, dtype, args.max_new_tokens)
    candidates = []
    for path in args.candidate:
        if device == "cuda":
            torch.cuda.empty_cache()
        recovered = _capture(torch, path, device, dtype, args.max_new_tokens)
        recovery = json.loads((path / "layer-recovery.json").read_text())
        recovery.pop("source", None)
        recovery.pop("checkpoint", None)
        raw_audit = _audit(torch, args.reference, path)
        alias_audit = None
        if raw_audit["missing_tensors"] and recovered[4]:
            alias_audit = _all_finite_exact(torch, args.reference, path, recovered[4])
        candidates.append(
            {
                "candidate": path.name,
                "sha256": _hashes(path),
                "recovery": recovery,
                "raw_checkpoint_audit": raw_audit,
                "exact_original_audit_with_verified_ties": alias_audit,
                "held_out_behavior": _behavior(torch, baseline, recovered, args.max_new_tokens),
            }
        )
    write_json(
        args.output,
        {
            "schema_version": 1,
            "model_id": args.model_id,
            "revision": args.revision,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "reference_sha256": reference_hashes,
            "held_out_prompts": list(_PROMPTS),
            "reference_prompt_token_ids": baseline[2],
            "reference_completion_ids": baseline[1],
            "candidates": candidates,
            "evaluation_seconds": time.monotonic() - started,
        },
    )


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recover = commands.add_parser("recover", help="export one candidate and measure memory")
    recover.add_argument("--model", required=True, type=Path)
    recover.add_argument("--output", required=True, type=Path)
    recover.add_argument("--metrics", required=True, type=Path)
    recover.add_argument("--stage-layers", action="store_true")
    recover.add_argument("--without-tokenizer", action="store_true")
    recover.add_argument("--batch-size", type=_positive_int, default=256)
    recover.set_defaults(func=_recover)

    evaluate = commands.add_parser("evaluate", help="compare exports against the original")
    evaluate.add_argument("--reference", required=True, type=Path)
    evaluate.add_argument("--candidate", required=True, type=Path, action="append")
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--model-id", required=True)
    evaluate.add_argument("--revision", required=True)
    evaluate.add_argument("--max-new-tokens", type=_positive_int, default=4)
    evaluate.set_defaults(func=_evaluate)

    for command in (recover, evaluate):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        command.add_argument("--dtype", choices=("auto", "float32", "float64"), default="auto")
        command.add_argument("--threads", type=_positive_int, default=4)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
