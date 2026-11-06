"""Compare standalone FP8 checkpoints with Transformers or dense BF16 reloads."""

from __future__ import annotations

import argparse
import time
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from rephix.io import write_json
from rephix.weight_audit import _shards

_PROMPTS = (
    "Once upon a time, a fox found a",
    "The library closed early because",
    "If each box holds twelve pencils, five boxes hold",
    "def double(value):\n    return",
    "Why do some birds migrate in winter?\nAnswer:",
    "Translate into Spanish: The train arrives at noon.\nSpanish:",
)
_FIXED_IDS = (
    (2, 9, 16, 24),
    (31, 14, 286, 445),
    (380, 511, 700, 12),
    (451, 566, 804, 909),
    (4, 22, 7, 106),
    (166, 325, 701, 822),
)
_LOCAL = {"local_files_only": True, "trust_remote_code": False}
_MAX_NEW_TOKENS = 8
_SCALE_SUFFIX = "_scale_inv"


def _dense_reload(path, device):
    config = AutoConfig.from_pretrained(path, **_LOCAL)
    quant = getattr(config, "quantization_config", {})
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    # Transformers 4 reads the older names for YaRN metadata.
    rope = getattr(config, "rope_parameters", None)
    adapted_rope = bool(rope and getattr(config, "rope_scaling", None) is None)
    if adapted_rope:
        config.rope_scaling = rope
        config.rope_theta = rope["rope_theta"]
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)

    tensors = model.state_dict()
    weight_blocks = quant.get("weight_block_size")
    copied = set()
    skipped = []
    mtp_start = config.num_hidden_layers
    mtp_layers = getattr(config, "num_nextn_predict_layers", 0) or 0
    mtp_prefixes = tuple(
        f"model.layers.{layer}." for layer in range(mtp_start, mtp_start + mtp_layers)
    )

    with ExitStack() as stack:
        by_name = {}
        for shard in _shards(path):
            handle = stack.enter_context(safe_open(shard, framework="pt", device=device))
            for key in handle.keys():
                if key in by_name:
                    raise ValueError(f"Duplicate tensor: {key}")
                by_name[key] = handle

        for name, handle in sorted(by_name.items()):
            if name.endswith(_SCALE_SUFFIX):
                if name.removesuffix(_SCALE_SUFFIX) not in by_name:
                    raise ValueError(f"Scale without a weight: {name}")
                continue
            if name not in tensors:
                if not name.startswith(mtp_prefixes):
                    raise ValueError(f"Unexpected inference tensor: {name}")
                skipped.append(name)
                continue
            weight = handle.get_tensor(name)
            if weight.shape != tensors[name].shape:
                raise ValueError(f"Unexpected shape for {name}: {weight.shape}")
            if weight.dtype == torch.float8_e4m3fn:
                if weight_blocks is None:
                    raise ValueError(f"Missing block dimensions for {name}")
                scale_name = name + _SCALE_SUFFIX
                scale = by_name[scale_name].get_tensor(scale_name).float()
                rows, columns = weight_blocks
                if scale.shape != (
                    (weight.shape[0] + rows - 1) // rows,
                    (weight.shape[1] + columns - 1) // columns,
                ):
                    raise ValueError(f"Unexpected shape for {scale_name}: {scale.shape}")
                scale = scale.repeat_interleave(rows, dim=0).repeat_interleave(columns, dim=1)
                weight = weight.float() * scale[: weight.shape[0], : weight.shape[1]]
            tensors[name].copy_(weight)
            copied.add(name)

    if config.tie_word_embeddings and "model.embed_tokens.weight" in copied:
        copied.add("lm_head.weight")
    if missing := sorted(set(tensors) - copied):
        raise ValueError(f"Missing inference tensors: {missing}")
    return model.eval(), skipped, adapted_rope


@torch.inference_mode()
def _capture(path, mode, device, fixed_ids):
    started = time.monotonic()
    if mode == "native":
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map=device, dtype="auto", **_LOCAL
        ).eval()
        skipped, adapted_rope = [], False
    else:
        model, skipped, adapted_rope = _dense_reload(path, device)

    if fixed_ids:
        prompts = [(None, list(ids)) for ids in _FIXED_IDS]
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, **_LOCAL)
        prompts = [(prompt, tokenizer.encode(prompt)) for prompt in _PROMPTS]

    logits, observations = [], []
    for prompt, ids in prompts:
        tokens = torch.tensor([ids], device=device)
        mask = torch.ones_like(tokens)
        output = model(input_ids=tokens, attention_mask=mask, use_cache=True)
        row = output.logits[0, -1].float().cpu().numpy()
        if not np.isfinite(row).all():
            raise ValueError("Nonfinite next-token logits")
        continuation = model.generate(
            input_ids=tokens,
            attention_mask=mask,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )[0, len(ids) :].tolist()
        if len(continuation) != _MAX_NEW_TOKENS:
            raise ValueError("Model returned fewer continuation tokens than requested")
        logits.append(row)
        observations.append(
            {
                "prompt": prompt,
                "prompt_token_ids": ids,
                "greedy_continuation": continuation,
                "top_1_id": int(row.argmax()),
                "logits_sha256": sha256(row.tobytes()).hexdigest(),
            }
        )
    return np.stack(logits), {
        "rows": observations,
        "skipped_non_inference_tensors": skipped,
        "adapted_rope_parameters": adapted_rope,
        "elapsed_seconds": time.monotonic() - started,
    }


def _compare(reference, candidate):
    left_logits, left = reference
    right_logits, right = candidate
    if left_logits.shape != right_logits.shape:
        raise ValueError("Incompatible next-token logit shapes")
    differences = left_logits.astype(np.float64) - right_logits
    prompts = list(zip(left["rows"], right["rows"], strict=True))
    return {
        "prompt_count": len(prompts),
        "exact_prompt_token_ids": sum(
            a["prompt_token_ids"] == b["prompt_token_ids"] for a, b in prompts
        ),
        "exact_cached_greedy_matches": sum(
            a["greedy_continuation"] == b["greedy_continuation"] for a, b in prompts
        ),
        "next_token_top1_matches": sum(a["top_1_id"] == b["top_1_id"] for a, b in prompts),
        "next_token_logit_coordinates": int(differences.size),
        "next_token_exact_logit_coordinates": int(np.count_nonzero(differences == 0)),
        "next_token_max_abs_logit_error": float(np.max(np.abs(differences))),
        "next_token_logit_rmse": float(np.sqrt(np.mean(differences**2))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("reference", "candidate", "output"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--mode", choices=("native", "dense"), default="native")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--fixed-ids", action="store_true")
    parser.add_argument("--published-dense-reference", type=Path)
    args = parser.parse_args()
    if args.mode == "native" and (args.device != "cuda" or args.published_dense_reference):
        parser.error("native mode requires CUDA and cannot compare to a dense reference")
    inputs = (args.reference, args.candidate, args.published_dense_reference)
    if any(path and args.output.resolve().is_relative_to(path.resolve()) for path in inputs):
        parser.error("output must be outside the checkpoint directories")

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference, candidate = (
        _capture(path, args.mode, args.device, args.fixed_ids) for path in inputs[:2]
    )
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "device": args.device,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "max_new_tokens": _MAX_NEW_TOKENS,
        "metrics": _compare(reference, candidate),
        "reference": reference[1],
        "candidate": candidate[1],
    }
    if args.published_dense_reference:
        dense = _capture(args.published_dense_reference, "dense", args.device, args.fixed_ids)
        report["published_dense_reference"] = dense[1]
        report["reference_against_published_dense"] = _compare(dense, reference)
        report["candidate_against_published_dense"] = _compare(dense, candidate)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
