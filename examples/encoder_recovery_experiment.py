"""Audit encoder weights and compare native task predictions on held-out text."""

import argparse
import hashlib
import json
import platform
import time
from contextlib import ExitStack
from pathlib import Path

from rephix.io import file_sha256, write_json
from rephix.weight_audit import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    _inventory,
    _open_shard,
    compare_checkpoints,
)

_MASKED = (
    ("The children watched a {mask} swim past the dock.", "It vanished beneath the waves."),
    ("The baker removed the {mask} from the oven.", "A clock hung above the door."),
    ("She turned on the {mask} because the room was dark.", "She could finally read the note."),
    ("Paris is the {mask} of France.", "The choir rehearsed in a small auditorium."),
    ("I put the leftovers in the {mask} to keep them cold.", "We ate them the next morning."),
    ("The {mask} delivered the package before noon.", "The seedlings grew beside the fence."),
)
_SENTIMENT = (
    "The performances made a familiar story feel surprising and alive.",
    "A tedious plot, muddy sound, and a final act that never earns its twists.",
    "I expected to dislike the film, but its quiet ending won me over.",
    "The sets are beautiful; unfortunately the characters have nothing interesting to say.",
    "A wonderfully strange little comedy that I would happily watch again.",
    "Even at ninety minutes it feels twice as long.",
)
_QA = (
    ("When did the museum reopen?", "The museum closed in March and reopened in early September."),
    (
        "Where is the spare key?",
        "The spare key is beneath the blue flowerpot beside the back door.",
    ),
    ("How many runners finished?", "Nine runners entered the race. Seven finished before dusk."),
    ("Who repaired the clock?", "Inez repaired the clock while Omar painted the station door."),
    ("What instrument did the band play?", "The train stopped briefly at the mountain station."),
    ("What did the orchard sell?", "In October the orchard sold pears and pressed apple cider."),
)


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode("utf-8")).hexdigest()


def _file_hashes(path):
    return {
        str(file.relative_to(path)): file_sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
        and file.suffix in {".safetensors", ".json", ".txt", ".model"}
        and ".cache" not in file.relative_to(path).parts
    }


def _task(architecture):
    for suffix, task in (
        ("ForMaskedLM", "masked_language_modeling"),
        ("ForPreTraining", "bert_pretraining"),
        ("ForSequenceClassification", "sequence_classification"),
        ("ForQuestionAnswering", "extractive_question_answering"),
    ):
        if architecture.endswith(suffix):
            return task
    raise ValueError(f"No held-out experiment for {architecture}")


def _encode(tokenizer, task):
    options = dict(padding=True, truncation=True, max_length=256, return_tensors="pt")
    if task in {"masked_language_modeling", "bert_pretraining"}:
        if tokenizer.mask_token is None:
            raise ValueError("The tokenizer has no mask token")
        sentences = [a.format(mask=tokenizer.mask_token) for a, _ in _MASKED]
        pairs = [b for _, b in _MASKED] if task == "bert_pretraining" else None
        prompts = list(zip(sentences, pairs, strict=True)) if pairs else sentences
        return prompts, tokenizer(sentences, text_pair=pairs, **options)
    if task == "sequence_classification":
        return _SENTIMENT, tokenizer(list(_SENTIMENT), **options)
    questions, contexts = zip(*_QA, strict=True)
    return _QA, tokenizer(list(questions), list(contexts), return_offsets_mapping=True, **options)


def _qa_prediction(torch, encoded, row, start, end):
    context = _QA[row][1]
    positions = torch.arange(start.numel())
    offsets = encoded.offset_mapping[row].tolist()
    context_tokens = torch.tensor([seq == 1 for seq in encoded.sequence_ids(row)])
    length = positions[None, :] - positions[:, None] + 1
    valid = context_tokens[:, None] & context_tokens[None, :] & (length > 0) & (length <= 20)
    scores = (start[:, None] + end[None, :]).masked_fill(~valid, -torch.inf)
    start_index, end_index = divmod(int(scores.argmax()), end.numel())
    if start[0] + end[0] >= scores[start_index, end_index]:
        return {"answer": "", "span": None}
    char_start, char_end = offsets[start_index][0], offsets[end_index][1]
    return {"answer": context[char_start:char_end], "span": [char_start, char_end]}


def _outputs(torch, model, tokenizer, encoded, task, device):
    inputs = {key: tensor.to(device) for key, tensor in encoded.items() if key != "offset_mapping"}
    with torch.inference_mode():
        outputs = model.to(device)(**inputs, return_dict=True)
        if task in {"masked_language_modeling", "bert_pretraining"}:
            masks = inputs["input_ids"] == tokenizer.mask_token_id
            if not (masks.sum(dim=1) == 1).all():
                raise ValueError("Expected exactly one mask per example")
            predictions = (
                outputs.prediction_logits if task == "bert_pretraining" else outputs.logits
            )
            logits = {"masked_lm": predictions[masks].cpu()}
            top_ids = logits["masked_lm"].topk(5, dim=-1).indices.tolist()
            decisions = {"masked_top5_ids": top_ids}
            if task == "bert_pretraining":
                logits["next_sentence"] = outputs.seq_relationship_logits.cpu()
                decisions["next_sentence_labels"] = logits["next_sentence"].argmax(-1).tolist()
        elif task == "sequence_classification":
            logits = {"classification": outputs.logits.cpu()}
            ids = logits["classification"].argmax(-1).tolist()
            decisions = {"class_ids": ids, "labels": [model.config.id2label[i] for i in ids]}
        else:
            start, end = outputs.start_logits.cpu(), outputs.end_logits.cpu()
            logits = {"start": start, "end": end}
            decisions = {
                "answers": [
                    _qa_prediction(torch, encoded, row, a, b)
                    for row, (a, b) in enumerate(zip(start, end, strict=True))
                ]
            }
    return logits, decisions


def _logit_metrics(torch, a, b):
    if a.shape != b.shape or not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("Invalid held-out logits")
    delta = (b.double() - a.double()).abs()
    return {
        "examples": len(a),
        "coordinates": a.numel(),
        "argmax_matches": int((a.argmax(dim=-1) == b.argmax(dim=-1)).sum()),
        "exactly_equal": int((a == b).sum()),
        "max_abs_error": float(delta.max()),
        "rmse": float(delta.square().mean().sqrt()),
    }


def _parameter_aliases(model):
    by_id = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        by_id.setdefault(id(parameter), []).append(name)
    return {name: names for names in by_id.values() for name in names if len(names) > 1}


def _canonical_tensors(model):
    tensors = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for name in model.state_dict().keys() & buffers.keys():
        tensors[name] = buffers[name]
    return tensors


def _tensor_metrics(torch, pairs, *, outside_tolerance=None):
    elements = exact = within = tensors = 0
    maximum = 0.0
    for name, source, candidate in pairs:
        if source.shape != candidate.shape:
            raise ValueError(f"Shape mismatch for {name}")
        tensors += 1
        elements += source.numel()
        before = within
        source_chunks = source.detach().reshape(-1).split(1 << 20)
        candidate_chunks = candidate.detach().reshape(-1).split(1 << 20)
        for a, b in zip(source_chunks, candidate_chunks, strict=True):
            if not a.numel():
                continue
            a, b = a.double(), b.double()
            delta = (b - a).abs()
            if not torch.isfinite(delta).all():
                raise ValueError(f"Nonfinite values in {name}")
            exact += int((a == b).sum())
            within += int((delta <= DEFAULT_ATOL + DEFAULT_RTOL * a.abs()).sum())
            maximum = max(maximum, float(delta.max()))
        missed = source.numel() - (within - before)
        if outside_tolerance is not None and missed:
            outside_tolerance[name] = missed
    return {
        "tensors": tensors,
        "elements": elements,
        "exactly_equal": exact,
        "within_tolerance": within,
        "max_abs_error": maximum,
    }


def _original_audit(torch, reference, candidate, model):
    originals, saved = _inventory(reference), _inventory(candidate)
    aliases, buffers = _parameter_aliases(model), dict(model.named_buffers())
    mapping, unused, nonpersistent = {}, {}, {}
    outside_tolerance = {}
    for name, entry in sorted(originals.items()):
        renamed = name
        if name.endswith(".LayerNorm.gamma"):
            renamed = name.removesuffix(".gamma") + ".weight"
        elif name.endswith(".LayerNorm.beta"):
            renamed = name.removesuffix(".beta") + ".bias"
        if renamed in saved:
            mapping[name] = renamed
        elif equivalent := sorted(saved.keys() & set(aliases.get(renamed, ()))):
            mapping[name] = equivalent[0]
        elif renamed in buffers:
            nonpersistent[name] = renamed
        else:
            unused[name] = entry.size

    with ExitStack() as stack:
        source_shards = {
            path: stack.enter_context(_open_shard(path))
            for path in {t.shard for t in originals.values()}
        }
        candidate_shards = {
            path: stack.enter_context(_open_shard(path))
            for path in {t.shard for t in saved.values()}
        }

        def read_source(name):
            return source_shards[originals[name].shard].get_tensor(name)

        stored = _tensor_metrics(
            torch,
            (
                (name, read_source(name), candidate_shards[saved[target].shard].get_tensor(target))
                for name, target in mapping.items()
            ),
            outside_tolerance=outside_tolerance,
        )
        derived = _tensor_metrics(
            torch,
            ((name, read_source(name), buffers[target]) for name, target in nonpersistent.items()),
        )
    return {
        "reference_elements": sum(entry.size for entry in originals.values()),
        "mapped_stored_values": stored,
        "mapped_values_outside_tolerance_by_source_tensor": outside_tolerance,
        "mapped_names": {name: target for name, target in mapping.items() if name != target},
        "nonpersistent_buffer_names": nonpersistent,
        "derived_runtime_buffer_values": derived,
        "unused_source_tensors": unused,
    }


def evaluate(args):
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer, PreTrainedModel

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    paths = args.reference, args.candidate
    fit = json.loads((args.candidate / "layer-recovery.json").read_text(encoding="utf-8"))
    fit.pop("source", None)
    fit.pop("checkpoint", None)
    dtype = getattr(torch, args.inference_dtype)
    architecture = fit["architecture"]
    task = _task(architecture)
    options = dict(local_files_only=True, trust_remote_code=False)
    configs = [AutoConfig.from_pretrained(path, **options) for path in paths]
    model_cls = getattr(transformers, architecture, None)
    if (
        not isinstance(model_cls, type)
        or not issubclass(model_cls, PreTrainedModel)
        or any(model_cls.config_class is not type(config) for config in configs)
    ):
        raise ValueError(f"Unsupported native architecture {architecture}")
    models = [
        model_cls.from_pretrained(
            path, local_files_only=True, use_safetensors=True, torch_dtype=dtype
        ).eval()
        for path in paths
    ]
    tokenizers = [AutoTokenizer.from_pretrained(path, **options) for path in paths]
    prompts, source_ids = _encode(tokenizers[0], task)
    candidate_prompts, candidate_ids = _encode(tokenizers[1], task)
    if (
        prompts != candidate_prompts
        or source_ids.keys() != candidate_ids.keys()
        or any(not torch.equal(value, candidate_ids[key]) for key, value in source_ids.items())
    ):
        raise ValueError("The exported tokenizer changed the held-out encodings")
    raw = compare_checkpoints(*paths)
    raw.pop("reference")
    raw.pop("candidate")
    original = _original_audit(torch, *paths, models[1])
    source_tensors, candidate_tensors = [_canonical_tensors(model) for model in models]
    if source_tensors.keys() != candidate_tensors.keys():
        raise ValueError("Native source and candidate state keys differ")
    canonical = _tensor_metrics(
        torch, ((name, value, candidate_tensors[name]) for name, value in source_tensors.items())
    )
    del source_tensors, candidate_tensors
    actual = []
    for model, tokenizer, encoded in zip(
        models, tokenizers, (source_ids, candidate_ids), strict=True
    ):
        actual.append(_outputs(torch, model, tokenizer, encoded, task, args.device))
        model.to("cpu")
    (source_logits, source_choices), (candidate_logits, candidate_choices) = actual
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "task": task,
        "source_declared_architectures": configs[0].architectures,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": args.device,
            "inference_dtype": str(dtype),
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "recovery": fit,
        "reference_sha256": _file_hashes(paths[0]),
        "candidate_sha256": _file_hashes(paths[1]),
        "raw_checkpoint_audit": raw,
        "direct_original_archive_audit": original,
        "canonical_runtime_audit": canonical,
        "held_out": {
            "prompts": prompts,
            "prompt_sha256": _hash(prompts),
            "tokenizer_inputs_exact": True,
            "tokenizer_input_sha256": _hash(
                {key: value.tolist() for key, value in source_ids.items()}
            ),
            "logits": {
                name: _logit_metrics(torch, value, candidate_logits[name])
                for name, value in source_logits.items()
            },
            "reference_decisions": source_choices,
            "candidate_decisions": candidate_choices,
            "decisions_exact": source_choices == candidate_choices,
        },
        "evaluation_seconds": time.monotonic() - started,
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
    parser.add_argument("--inference-dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
