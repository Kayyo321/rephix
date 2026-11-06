"""Audit a recovered GGUF and compare standalone Transformers inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from rephix.gguf_audit import audit_gguf
from rephix.hf import _device, _generation_config, _load_tokenizer
from rephix.io import file_sha256, write_json

_PROMPTS = (
    "The city decided to build a new library because",
    "Translate into Spanish: The train arrives in ten minutes.\nSpanish:",
    "If a box holds 12 pencils, then 5 boxes hold",
    "def is_even(number):\n    return",
    "Why do some birds migrate in winter?\nAnswer:",
    "A balanced lunch might contain rice, vegetables, and",
)


def _capture(torch, path, device, max_new_tokens):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Give the loader only the GGUF, even if unrelated model files sit beside it.
    with tempfile.TemporaryDirectory(prefix="rephix-gguf-eval-") as temp:
        folder = Path(temp)
        (folder / path.name).symlink_to(path.resolve())
        options = {"gguf_file": path.name, "local_files_only": True, "trust_remote_code": False}
        model, loading = AutoModelForCausalLM.from_pretrained(
            folder, dtype=torch.float32, output_loading_info=True, **options
        )
        for key in ("missing_keys", "mismatched_keys", "error_msgs"):
            if loading.get(key):
                raise ValueError(f"standalone GGUF load returned {key}: {loading[key]}")
        tokenizer = _load_tokenizer(AutoTokenizer, folder, model_config=model.config, **options)
        model.eval().to(device)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            list(_PROMPTS), padding=True, return_token_type_ids=False, return_tensors="pt"
        ).to(device)
        mask = inputs.attention_mask
        positions = (mask.cumsum(dim=-1) - 1).clamp(min=0)
        forward_options = (
            {"logits_to_keep": 1}
            if "logits_to_keep" in inspect.signature(model.forward).parameters
            else {}
        )
        with torch.inference_mode():
            outputs = model(**inputs, position_ids=positions, use_cache=False, **forward_options)
            logits = outputs.logits[:, -1].float().cpu()
            generation = _generation_config(model, tokenizer, max_new_tokens)
            generation.use_cache = True
            completion = model.generate(**inputs, generation_config=generation)
            continuation_ids = completion[:, inputs.input_ids.shape[1] :].cpu().tolist()
        token_ids = [row[keep.bool()].tolist() for row, keep in zip(inputs.input_ids, mask)]
        return {
            "logits": logits,
            "logits_sha256": hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest(),
            "token_ids": token_ids,
            "continuation_ids": continuation_ids,
            "top1_token_ids": logits.argmax(dim=1).tolist(),
            "architecture": type(model).__name__,
            "tokenizer": type(tokenizer).__name__,
            "dtype": str(model.dtype),
            "unexpected_weight_keys": loading.get("unexpected_keys", []),
        }


def _behavior(torch, reference, candidate, max_new_tokens):
    left, right = reference.pop("logits"), candidate.pop("logits")
    if left.shape != right.shape:
        raise ValueError("standalone GGUF logits disagree on vocabulary dimensions")
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("standalone GGUF inference returned nonfinite logits")
    difference = left.double() - right.double()
    return {
        "prompt_count": len(_PROMPTS),
        "max_new_tokens": max_new_tokens,
        "prompt_token_ids_match": reference["token_ids"] == candidate["token_ids"],
        "logit_coordinates": left.numel(),
        "exact_logit_coordinates": int((left == right).sum()),
        "logit_max_abs_error": float(difference.abs().max()),
        "logit_rmse": float(difference.square().mean().sqrt()),
        "next_token_top1_matches": int((left.argmax(dim=1) == right.argmax(dim=1)).sum()),
        "exact_cached_greedy_matches": sum(
            a == b for a, b in zip(reference["continuation_ids"], candidate["continuation_ids"])
        ),
        "prompts": list(_PROMPTS),
        "reference": reference,
        "candidate": candidate,
    }


def _file_evidence(path):
    return {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _evaluate(args):
    import torch
    import transformers

    if args.output.resolve() in {
        args.reference.resolve(),
        args.candidate.resolve(),
        args.recovery_report.resolve(),
    }:
        raise ValueError(
            "experiment output must be separate from the checkpoints and recovery report"
        )
    if args.threads < 1 or args.max_new_tokens < 1:
        raise ValueError("threads and max new tokens must be positive")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    device = _device(torch, args.device)
    start = time.monotonic()

    recovery = json.loads(args.recovery_report.read_text(encoding="utf-8"))
    recovery.pop("source", None)
    recovery.pop("checkpoint", None)
    audit = audit_gguf(args.reference, args.candidate)
    audit.pop("reference", None)
    audit.pop("candidate", None)

    reference = _capture(torch, args.reference, device, args.max_new_tokens)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    candidate = _capture(torch, args.candidate, device, args.max_new_tokens)
    write_json(
        args.output,
        {
            "schema_version": 1,
            "model_id": args.model_id,
            "revision": args.revision,
            "reference_file": _file_evidence(args.reference),
            "candidate_file": _file_evidence(args.candidate),
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "transformers": transformers.__version__,
                "gguf": version("gguf"),
                "accelerate": version("accelerate"),
                "device": device,
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "threads": args.threads,
                "inference": "standalone GGUF dequantized by Transformers to FP32",
            },
            "recovery": recovery,
            "gguf_audit": audit,
            "held_out_behavior": _behavior(torch, reference, candidate, args.max_new_tokens),
            "evaluation_seconds": time.monotonic() - start,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--recovery-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    _evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
