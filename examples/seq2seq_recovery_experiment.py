"""Audit a recovered encoder-decoder checkpoint and compare held-out generation."""

import argparse
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

from rephix.io import file_sha256, write_json
from rephix.weight_audit import DEFAULT_ATOL, DEFAULT_RTOL, compare_checkpoints

_T5_PROMPTS = (
    (
        "translate English to German: The baker left a note beside the blue door.",
        "Der Bäcker hinterließ einen Zettel neben der blauen Tür.",
    ),
    (
        "translate English to French: After the rain, the children planted three trees.",
        "Après la pluie, les enfants ont planté trois arbres.",
    ),
    (
        "translate English to Romanian: Our train arrives shortly before noon.",
        "Trenul nostru sosește cu puțin înainte de prânz.",
    ),
    (
        "summarize: The town library will close on Tuesday for roof repairs. "
        "Readers can return books at the station kiosk while the library is closed. "
        "The library expects to reopen on Saturday.",
        "The town library will reopen Saturday after roof repairs.",
    ),
    (
        "summarize: A volunteer group collected broken bicycles in February. "
        "Mechanics repaired forty bicycles and donated them to students who live far from school.",
        "Volunteers repaired forty bicycles for students.",
    ),
)
_MULTILINGUAL_PROMPTS = (
    ("Le musée ouvre à <extra_id_0> le dimanche.", "<extra_id_0> dix heures <extra_id_1>"),
    ("El tren sale de la <extra_id_0> al amanecer.", "<extra_id_0> estación <extra_id_1>"),
    (
        "Die Kinder haben gestern im <extra_id_0> Fußball gespielt.",
        "<extra_id_0> Park <extra_id_1>",
    ),
    ("この町では毎年<extra_id_0>に祭りがある。", "<extra_id_0> 夏 <extra_id_1>"),
    ("فتحت المكتبة أبوابها في <extra_id_0> صباحًا.", "<extra_id_0> الساعة التاسعة <extra_id_1>"),
    (
        "새 책 표지에는 🪼와 𪚥가 그려져 있다. 책은 <extra_id_0>에 있다.",
        "<extra_id_0> 도서관 <extra_id_1>",
    ),
)
_BART_PROMPTS = (
    (
        "The harbor authority closed the eastern pier on Monday after inspectors found a "
        "damaged support beam. Ferries will leave from the western pier until crews finish "
        "repairs. The authority expects normal service to resume before the weekend.",
        "Ferries will use the western pier while crews repair the eastern pier.",
    ),
    (
        "Volunteers surveyed six neighborhood gardens during a hot July. The gardens that "
        "used drip hoses needed less water than those irrigated with sprinklers. The city "
        "will share the results with growers at a public workshop in September.",
        "The city will share a garden water study at a public workshop in September.",
    ),
    (
        "A fire broke out in an empty warehouse shortly after midnight. Firefighters stopped "
        "it from spreading to a nearby market. No injuries were reported, and investigators "
        "are still looking for the cause.",
        "Firefighters contained a warehouse fire. Investigators are looking for the cause.",
    ),
    (
        "The last train on the branch line normally leaves at eleven. Starting next month, "
        "the railway will run a later service on Friday and Saturday to connect with buses "
        "that serve two neighboring villages.",
        "The railway will add a late weekend train to connect with village buses.",
    ),
    (
        "The engineering class built a weather station from discarded electronics. The "
        "station measured wind and rainfall for three months, and the students published "
        "the observations for farmers in nearby communities.",
        "Students shared weather observations from their homemade station with local farmers.",
    ),
)
_PANELS = {
    "t5": ("translation_and_summarization", _T5_PROMPTS),
    "mt5": ("multilingual_masked_spans", _MULTILINGUAL_PROMPTS),
    "umt5": ("multilingual_masked_spans", _MULTILINGUAL_PROMPTS),
    "bart": ("summarization", _BART_PROMPTS),
}


def _json_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hashes(directory):
    return {
        path.name: file_sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in {".safetensors", ".json", ".model", ".txt"}
    }


def _audit(source, candidate):
    report = compare_checkpoints(source, candidate)
    report.pop("reference")
    report.pop("candidate")
    return report


def bart_original_coordinates(torch, source, candidate):
    """Check the original BART archive directly after resolving its storage names."""
    from safetensors import safe_open

    source, candidate = (path / "model.safetensors" for path in (source, candidate))
    if not source.is_file() or not candidate.is_file():
        return None  # The general raw and canonical audits also handle sharded files.
    tied_names = {"model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"}
    elements = exact = within = renamed = 0
    with (
        safe_open(source, framework="pt", device="cpu") as a,
        safe_open(candidate, framework="pt", device="cpu") as b,
    ):
        candidate_names = set(b.keys())
        for name in a.keys():
            mapped = (
                name if name in candidate_names or name.startswith("model.") else f"model.{name}"
            )
            if mapped not in candidate_names and mapped in tied_names:
                mapped = "model.shared.weight"
            ref, recovered = a.get_tensor(name), b.get_tensor(mapped)
            if ref.shape != recovered.shape:
                raise ValueError(f"BART tensor shape changed for {name}")
            renamed += int(mapped != name)
            elements += ref.numel()
            for x, y in zip(ref.flatten().split(1 << 20), recovered.flatten().split(1 << 20)):
                x, y = x.double(), y.double()
                exact += int((x == y).sum())
                within += int(torch.isclose(y, x, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL).sum())
    return {
        "renamed_tensors": renamed,
        "reference_elements": elements,
        "exactly_equal_elements": exact,
        "within_tolerance_elements": within,
        "atol": DEFAULT_ATOL,
        "rtol": DEFAULT_RTOL,
    }


def _tokenize(tokenizer, prompts):
    sources, targets = zip(*prompts)
    return (
        tokenizer(list(sources), padding=True, return_tensors="pt"),
        tokenizer(text_target=list(targets), padding=True, return_tensors="pt"),
    )


def _check_tokenization(torch, a, b):
    hashes = {}
    for label, source, candidate in zip(("source", "target"), a, b):
        for field in ("input_ids", "attention_mask"):
            if not torch.equal(source[field], candidate[field]):
                raise ValueError(f"exported tokenizer changed {label} {field}")
            hashes[f"{label}_{field}_sha256"] = _json_hash(source[field].tolist())
    return hashes


def _generate(model, input_ids, attention_mask, max_new_tokens):
    return (
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            min_length=0,
            early_stopping=False,
            length_penalty=1.0,
        )
        .cpu()
        .tolist()
    )


def _trim_at_eos(sequence, eos_ids):
    # BART starts its decoder with an EOS token, so the search begins at index one.
    return sequence[
        : next(
            (i + 1 for i, token in enumerate(sequence[1:], 1) if token in eos_ids), len(sequence)
        )
    ]


def _evaluate_behavior(torch, models, tokenizers, encodings, device, max_new_tokens):
    source, targets = encodings
    input_ids = source.input_ids.to(device)
    source_mask = source.attention_mask.to(device)
    valid = targets.attention_mask.to(device).bool()
    labels = targets.input_ids.to(device).masked_fill(~valid, -100)
    decoder_mask = torch.cat((torch.ones_like(valid[:, :1]), valid[:, :-1]), dim=1)
    logits, generated = [], []
    with torch.inference_mode():
        for model in models:
            decoder_ids = model.prepare_decoder_input_ids_from_labels(labels)
            scores = model(
                input_ids=input_ids,
                attention_mask=source_mask,
                decoder_input_ids=decoder_ids,
                decoder_attention_mask=decoder_mask,
                use_cache=False,
            ).logits
            logits.append(scores[valid].cpu().double())
            del scores
            generated.append(_generate(model, input_ids, source_mask, max_new_tokens))
    a, b = logits
    difference = b - a
    eos = models[0].generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    a_ids, b_ids = ([_trim_at_eos(sequence, eos_ids) for sequence in batch] for batch in generated)
    return {
        "teacher_forced": {
            "examples": len(input_ids),
            "target_tokens": len(a),
            "vocab_size": a.shape[-1],
            "source_padding_tokens": int((source_mask == 0).sum()),
            "target_padding_tokens": int((~valid).sum()),
            "argmax_matches": int((a.argmax(-1) == b.argmax(-1)).sum()),
            "logit_max_abs_error": float(difference.abs().max()),
            "logit_rmse": float(difference.square().mean().sqrt()),
        },
        "free_running": {
            "examples": len(a_ids),
            "cached": True,
            "max_new_tokens": max_new_tokens,
            "exact_matches": sum(x == y for x, y in zip(a_ids, b_ids)),
            "reference_ids_sha256": _json_hash(a_ids),
            "candidate_ids_sha256": _json_hash(b_ids),
            "reference_text": tokenizers[0].batch_decode(a_ids, skip_special_tokens=True),
            "candidate_text": tokenizers[1].batch_decode(b_ids, skip_special_tokens=True),
        },
    }


def evaluate(args):
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    fit = json.loads((args.candidate / "layer-recovery.json").read_text(encoding="utf-8"))
    fit.pop("source")
    fit.pop("checkpoint")
    dtype = getattr(torch, fit["oracle_dtype"].removeprefix("torch."))
    options = {"local_files_only": True, "trust_remote_code": False}

    def load(path):
        return AutoModelForSeq2SeqLM.from_pretrained(
            path, use_safetensors=True, torch_dtype=dtype, **options
        ).eval()

    reference = load(args.reference)
    panel, prompts = _PANELS[reference.config.model_type]
    raw_audit = _audit(args.reference, args.candidate)
    with tempfile.TemporaryDirectory(prefix="rephix-audit-", dir=args.candidate.parent) as tmp:
        reference.save_pretrained(tmp, safe_serialization=True)
        canonical_audit = _audit(tmp, args.candidate)
    candidate = load(args.candidate)
    use_fast = not (
        reference.config.model_type in {"t5", "mt5", "umt5"}
        and not (args.reference / "tokenizer.json").is_file()
        and (args.reference / "spiece.model").is_file()
    )
    tokenizers = [
        AutoTokenizer.from_pretrained(p, use_fast=use_fast, **options)
        for p in (args.reference, args.candidate)
    ]
    if tokenizers[0].is_fast != tokenizers[1].is_fast:
        raise ValueError("exported tokenizer changed the implementation used for tokenization")
    tokenizer_mode = "fast" if tokenizers[0].is_fast else "slow"
    a, b = [_tokenize(tokenizer, prompts) for tokenizer in tokenizers]
    tokenization = _check_tokenization(torch, a, b)
    models = [model.to(args.device) for model in (reference, candidate)]
    behavior = _evaluate_behavior(torch, models, tokenizers, a, args.device, args.max_new_tokens)
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "upstream_bin_sha256": args.upstream_bin_sha256,
        "reference_sha256": _file_hashes(args.reference),
        "candidate_sha256": _file_hashes(args.candidate),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": args.device,
            "dtype": str(dtype),
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "recovery": fit,
        "raw_checkpoint_audit": raw_audit,
        "canonical_checkpoint_audit": canonical_audit,
        "bart_original_coordinates": (
            bart_original_coordinates(torch, args.reference, args.candidate)
            if reference.config.model_type == "bart"
            else None
        ),
        "held_out_panel": {
            "name": panel,
            "fixtures_sha256": _json_hash(prompts),
            "tokenizer_mode": tokenizer_mode,
            **tokenization,
        },
        "held_out_behavior": behavior,
        "evaluation_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--upstream-bin-sha256")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
