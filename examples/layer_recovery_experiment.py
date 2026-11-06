"""Audit an exported checkpoint and test a residual-sign symmetry on held-out prompts."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import re
import tempfile
import time
from pathlib import Path

from logit_subspace_experiment import TEST, prompts

from rephix.hf import _generation_config, _load_tokenizer
from rephix.io import file_sha256, write_json
from rephix.weight_audit import DEFAULT_ATOL, DEFAULT_RTOL, compare_checkpoints

_MATRICES = re.compile(
    r"(?:model\.embed_tokens|lm_head|model\.layers\.\d+\."
    r"(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj))\.weight"
)
_NORMS = re.compile(
    r"model\.(?:norm|layers\.\d+\.(?:input_layernorm|post_attention_layernorm))\.weight"
)


def _capture(model, tokenizer, texts, device, *, logit_dtype=None):
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows, ids = [], []
    forward = inspect.signature(model.forward).parameters
    last_logits = {"logits_to_keep": 1} if "logits_to_keep" in forward else {}
    auxiliary = {"output_router_logits": False} if "output_router_logits" in forward else {}
    with torch.inference_mode():
        for i in range(0, len(texts), 6):
            inputs = tokenizer(texts[i : i + 6], padding=True, return_tensors="pt").to(device)
            mask = inputs.attention_mask
            ids.extend(inputs.input_ids[mask.bool()].tolist())
            position_ids = (mask.cumsum(dim=-1) - 1).clamp(min=0)
            output = model(
                **inputs, position_ids=position_ids, use_cache=False, **last_logits, **auxiliary
            )
            rows.append(output.logits[:, -1].to(dtype=logit_dtype or torch.float32).cpu())
        selected = tokenizer(texts[::4], padding=True, return_tensors="pt").to(device)
        config = _generation_config(model, tokenizer, 8)
        completion = model.generate(**selected, generation_config=config, **auxiliary)
        continuations = completion[:, selected.input_ids.shape[1] :].tolist()
    return torch.cat(rows), continuations, ids


def _behavior(reference, other):
    a, completions, ids = reference
    b, other_completions, other_ids = other
    if ids != other_ids or a.shape != b.shape:
        raise ValueError("the models disagree on held-out token coordinates")
    a, b = a.double(), b.double()
    log_p, log_q = a.log_softmax(dim=1), b.log_softmax(dim=1)
    return {
        "prompt_count": len(a),
        "logit_rmse": float((a - b).square().mean().sqrt()),
        "logit_max_abs_error": float((a - b).abs().max()),
        "mean_kl_nats": max(0.0, float((log_p.exp() * (log_p - log_q)).sum(dim=1).mean())),
        "next_token_top1_matches": int((a.argmax(dim=1) == b.argmax(dim=1)).sum()),
        "exact_greedy_matches": sum(x == y for x, y in zip(completions, other_completions)),
        "greedy_prompts": len(completions),
        "max_new_tokens": 8,
    }


def _sign_symmetry(model, tokenizer, texts, device, reference):
    """Flip the residual coordinate system of a biasless Llama or Mistral."""
    import torch

    parameters = list(model.named_parameters())
    flipped = []
    for name, parameter in parameters:
        if _MATRICES.fullmatch(name) and parameter.ndim == 2:
            flipped.append(parameter)
        elif not (_NORMS.fullmatch(name) and parameter.ndim == 1):
            raise ValueError(f"the residual sign control does not cover {name}")
    total = sum(t.numel() for _, t in parameters)
    count_exact = count_close = count_overlap = total - sum(t.numel() for t in flipped)
    ref_norm = changed_norm = 0.0
    with torch.no_grad():
        for _, parameter in parameters:
            squared = sum(
                float(block.float().square().sum()) for block in parameter.flatten().split(1 << 22)
            )
            ref_norm += squared
            if parameter.ndim == 2:
                changed_norm += squared
                for block in parameter.flatten().split(1 << 22):
                    values = block.float().abs()
                    radius = DEFAULT_ATOL + DEFAULT_RTOL * values
                    count_exact += int((values == 0).sum())
                    count_close += int((2 * values <= radius).sum())
                    count_overlap += int((values <= radius).sum())
        for parameter in flipped:
            parameter.neg_()
        try:
            behavior = _behavior(reference, _capture(model, tokenizer, texts, device))
        finally:
            for parameter in flipped:
                parameter.neg_()

    return {
        "parameter_scope": "unique loaded parameter elements, after tying",
        "reference_elements": total,
        "atol": DEFAULT_ATOL,
        "rtol": DEFAULT_RTOL,
        "fraction_exactly_equal": count_exact / total,
        "fraction_within_tolerance": count_close / total,
        "relative_l2": 2 * (changed_norm / ref_norm) ** 0.5,
        "max_worst_case_tolerance_fraction_for_either_target": (1 + count_overlap / total) / 2,
        "behavior": behavior,
    }


def _hashes(path):
    return {
        file.name: file_sha256(file)
        for file in sorted(path.iterdir())
        if file.suffix in {".safetensors", ".json", ".model", ".txt", ".jinja"}
    }


def evaluate(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    started = time.monotonic()
    reference_hashes = _hashes(args.reference)
    candidate_hashes = _hashes(args.candidate)
    audit = audit_error = None
    try:
        audit = compare_checkpoints(args.reference, args.candidate)
        audit.pop("reference")
        audit.pop("candidate")
    except ValueError as exc:
        if not args.repack_reference:
            raise
        audit_error = str(exc)
    recovery_path = args.candidate / "layer-recovery.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovery.pop("source")
    recovery.pop("checkpoint")
    rows = prompts(TEST, 4, 7116)
    texts = [row["text"] for row in rows]

    def load(path):
        opts = {"trust_remote_code": False, "local_files_only": True}
        dtype = getattr(torch, recovery["oracle_dtype"].removeprefix("torch."))
        model = AutoModelForCausalLM.from_pretrained(
            path, use_safetensors=True, torch_dtype=dtype, **opts
        ).eval()
        tokenizer = _load_tokenizer(AutoTokenizer, path, model_config=model.config, **opts)
        return model.to(args.device), tokenizer

    target, tokenizer = load(args.reference)
    loaded_parameter_audit = None
    if args.repack_reference:
        with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
            target.save_pretrained(temporary, safe_serialization=True)
            loaded_parameter_audit = compare_checkpoints(temporary, args.candidate)
            loaded_parameter_audit.pop("reference")
            loaded_parameter_audit.pop("candidate")
    reference = _capture(target, tokenizer, texts, args.device)
    gauge = None
    if recovery["model_type"] in {"llama", "mistral"}:
        gauge = _sign_symmetry(target, tokenizer, texts, args.device, reference)
    del target, tokenizer
    if args.device == "cuda":
        torch.cuda.empty_cache()
    candidate, tokenizer = load(args.candidate)
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "reference_sha256": reference_hashes,
        "candidate_sha256": candidate_hashes,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "recovery": recovery,
        "raw_checkpoint_audit": audit,
        "raw_checkpoint_audit_error": audit_error,
        "repacked_reference_audit": loaded_parameter_audit,
        "held_out_prompts": rows,
        "recovered_behavior": _behavior(
            reference, _capture(candidate, tokenizer, texts, args.device)
        ),
        "function_preserving_sign_control": gauge,
        "evaluation_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--repack-reference",
        action="store_true",
        help="also audit the reference after loading and serializing it with current Transformers",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
