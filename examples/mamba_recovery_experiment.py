"""Audit a recovered Mamba checkpoint and compare its recurrent behavior."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

from rephix.io import file_sha256, write_json
from rephix.weight_audit import DEFAULT_ATOL, DEFAULT_RTOL, compare_checkpoints

_CACHE_SUFFIX_TOKENS = 6
_PROMPTS = (
    "The small town opened a public observatory on the hill. On its first night, visitors",
    "In Python, a function that counts the vowels in a string could start like this:\n\ndef",
    "The first three powers of two are 1, 2, and 4. The next three are",
    "Translate into English: Le train longe la rivière avant d'arriver au village.\nEnglish:",
    "The following ingredients are on the counter: potatoes, onions, oil, and salt. A simple recipe",
    "Mira keeps a yellow notebook in the top drawer. She put a postcard between its first two pages. "
    "Later, she moved the notebook to the bottom drawer without opening it. Her brother searched "
    "the top drawer and found a pencil, a key, and an empty envelope. If he wants to read the "
    "postcard, he should first look inside",
)


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode("utf-8")).hexdigest()


def _file_hashes(path):
    return {
        str(file.relative_to(path)): file_sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
        and not file.name.startswith(".")
        and file.suffix in {".safetensors", ".json", ".txt", ".model"}
        and ".cache" not in file.relative_to(path).parts
    }


def _step_logits(torch, model, ids, prefix_size):
    prefix = model(input_ids=ids[:, :prefix_size], use_cache=True, return_dict=True)
    cache, logits = prefix.cache_params, [prefix.logits]
    for index in range(prefix_size, ids.shape[1]):
        # Native Mamba starts decoding with the cache position after its kernel width.
        position = torch.tensor([model.config.conv_kernel + index - prefix_size], device=ids.device)
        result = model(
            input_ids=ids[:, index : index + 1],
            use_cache=True,
            cache_params=cache,
            cache_position=position,
            return_dict=True,
        )
        cache = result.cache_params
        logits.append(result.logits)
    return torch.cat(logits, dim=1)


def _model_outputs(torch, path, tokenizer, encodings, dtype, args):
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            path,
            use_safetensors=True,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
        )
        .eval()
        .to(args.device)
    )
    forward, cached, generated = [], [], []
    states = {"A_log": {}, "D": {}, "effective_decay_rate": {}}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with torch.inference_mode():
        for name, parameter in model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("A_log", "D"):
                states[leaf][name] = parameter.detach().cpu()
                if leaf == "A_log":
                    states["effective_decay_rate"][name] = parameter.float().exp().cpu()
        for encoding in encodings:
            ids = encoding.input_ids.to(args.device)
            mask = encoding.attention_mask.to(args.device)
            full = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            prefix_size = max(1, ids.shape[1] - _CACHE_SUFFIX_TOKENS)
            forward.append(full.logits[0].cpu())
            cached.append(_step_logits(torch, model, ids, prefix_size)[0].cpu())
            tokens = (
                model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=True,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=pad_id,
                )[0, ids.shape[1] :]
                .cpu()
                .tolist()
            )
            generated.append({"token_ids": tokens, "text": tokenizer.decode(tokens)})
    return forward, cached, generated, states


def _logits_metrics(torch, reference, candidate):
    tokens = elements = argmax = 0
    squared_error = maximum = 0.0
    for a, b in zip(reference, candidate, strict=True):
        if a.shape != b.shape or not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError("invalid held-out logits")
        difference = b.double() - a.double()
        tokens += a.shape[0]
        elements += a.numel()
        argmax += int((a.argmax(dim=-1) == b.argmax(dim=-1)).sum())
        squared_error += float(difference.square().sum())
        maximum = max(maximum, float(difference.abs().max()))
    if not elements:
        raise ValueError("held-out prompts yielded no logit positions")
    return {
        "positions": tokens,
        "argmax_matches": argmax,
        "logit_max_abs_error": maximum,
        "logit_rmse": (squared_error / elements) ** 0.5,
    }


def _state_metrics(torch, reference, candidate):
    if not reference or reference.keys() != candidate.keys():
        raise ValueError("state parameter names differ or are absent")
    elements = exact = within = 0
    maximum = 0.0
    for name, source in reference.items():
        recovered = candidate[name]
        if source.shape != recovered.shape or not (
            torch.isfinite(source).all() and torch.isfinite(recovered).all()
        ):
            raise ValueError(f"invalid state parameters at {name}")
        a, b = source.double(), recovered.double()
        error = (b - a).abs()
        elements += a.numel()
        exact += int((a == b).sum())
        within += int((error <= DEFAULT_ATOL + DEFAULT_RTOL * a.abs()).sum())
        maximum = max(maximum, float(error.max()))
    return {
        "elements": elements,
        "exactly_equal": exact,
        "within_tolerance": within,
        "max_abs_error": maximum,
    }


def _decode_suffix(logits):
    return [row[max(1, len(row) - _CACHE_SUFFIX_TOKENS) :] for row in logits if len(row) > 1]


def evaluate(args):
    import torch
    import transformers
    from transformers import AutoTokenizer

    start = time.monotonic()
    torch.set_num_threads(args.threads)
    paths = (args.reference, args.candidate)
    fit = json.loads((args.candidate / "layer-recovery.json").read_text(encoding="utf-8"))
    fit.pop("source", None)
    fit.pop("checkpoint", None)
    dtype = getattr(torch, args.inference_dtype)
    tokenizers = [
        AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        for path in paths
    ]
    encodings = [
        [tokenizer(text, return_tensors="pt") for text in _PROMPTS] for tokenizer in tokenizers
    ]
    for a, b in zip(*encodings, strict=True):
        if not (
            torch.equal(a.input_ids, b.input_ids)
            and torch.equal(a.attention_mask, b.attention_mask)
        ):
            raise ValueError("exported tokenizer changed the held-out encodings")

    a_forward, a_cached, a_gen, a_states = _model_outputs(
        torch,
        paths[0],
        tokenizers[0],
        encodings[0],
        dtype,
        args,
    )
    if args.device == "cuda":
        torch.cuda.empty_cache()
    b_forward, b_cached, b_gen, b_states = _model_outputs(
        torch,
        paths[1],
        tokenizers[1],
        encodings[1],
        dtype,
        args,
    )
    audit = compare_checkpoints(*paths)
    audit.pop("reference")
    audit.pop("candidate")
    state_audit = {name: _state_metrics(torch, a_states[name], b_states[name]) for name in a_states}
    if args.device == "cuda":
        cpu_rates = [
            {name: logs.float().exp() for name, logs in states["A_log"].items()}
            for states in (a_states, b_states)
        ]
        state_audit["cpu_effective_decay_rate"] = _state_metrics(torch, *cpu_rates)
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": args.device,
            "inference_dtype": str(dtype),
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "recovery": fit,
        "reference_sha256": _file_hashes(args.reference),
        "candidate_sha256": _file_hashes(args.candidate),
        "raw_checkpoint_audit": audit,
        "loaded_state_audit": state_audit,
        "held_out": {
            "prompt_sha256": _hash(_PROMPTS),
            "token_ids_sha256": _hash([row.input_ids[0].tolist() for row in encodings[0]]),
            "tokenizer_ids_and_masks_exact": True,
            "forward": _logits_metrics(torch, a_forward, b_forward),
            "cached_forward": _logits_metrics(torch, a_cached, b_cached),
            "cached_decode": _logits_metrics(
                torch, _decode_suffix(a_cached), _decode_suffix(b_cached)
            ),
            "reference_cache_consistency": _logits_metrics(torch, a_forward, a_cached),
            "candidate_cache_consistency": _logits_metrics(torch, b_forward, b_cached),
            "free_running": {
                "examples": len(_PROMPTS),
                "max_new_tokens": args.max_new_tokens,
                "cached": True,
                "exact_matches": sum(
                    a["token_ids"] == b["token_ids"] for a, b in zip(a_gen, b_gen, strict=True)
                ),
                "prompts": list(_PROMPTS),
                "reference": a_gen,
                "candidate": b_gen,
            },
        },
        "evaluation_seconds": time.monotonic() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--inference-dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
