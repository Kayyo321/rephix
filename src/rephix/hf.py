"""Optional local Transformers experiments, with explicit evidence boundaries.

Training from prompt/output pairs is behavioral reconstruction. It is not a
procedure for uniquely recovering a transformer's original parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from .evidence import Observation, read_observations, write_observations


def _deps():
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install optional dependencies with: pip install -e ".[hf]"') from exc
    return torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _device(torch, name):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return name


def _seed(torch, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokenizer_fingerprint(tokenizer):
    payload = {
        "vocab": sorted(tokenizer.get_vocab().items()),
        "special_tokens": tokenizer.special_tokens_map,
        "chat_template": tokenizer.chat_template,
        "backend": tokenizer.backend_tokenizer.to_str()
        if hasattr(tokenizer, "backend_tokenizer")
        else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _render_prompt(tokenizer, prompt):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
    return tokenizer.encode(prompt, add_special_tokens=True)


def _completion_text(record):
    if "completion_text" in record.metadata:
        text = record.metadata["completion_text"]
        if not isinstance(text, str):
            raise ValueError(f"{record.id}: metadata.completion_text must be a string")
        return text
    if record.thinking:
        raise ValueError(
            f"{record.id}: separate thinking requires metadata.completion_text with the exact "
            "serialized visible target, or include the visible text in output"
        )
    return record.output


def _generation_config(model, tokenizer, max_new_tokens):
    """Neutral greedy decoding, independent of downloaded generation_config.json."""
    from transformers import GenerationConfig

    config = GenerationConfig.from_model_config(model.config)
    special_ids = {"bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id"}
    for key, value in GenerationConfig().to_dict().items():
        if key not in special_ids and not key.startswith("_"):
            setattr(config, key, value)
    config.do_sample = False
    config.num_beams = 1
    config.num_return_sequences = 1
    config.max_new_tokens = max_new_tokens
    if config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id
    if config.pad_token_id is None:
        config.pad_token_id = (
            config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
        )
    # Transformers >=4.50 can merge model.generation_config back into default-
    # valued fields even when a config is supplied. Replace that fallback too.
    model.generation_config = config
    return config


def _check_disjoint(*datasets):
    from .evidence import assert_disjoint

    assert_disjoint(*datasets)


def _write_report(path, report):
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _positive(value, label):
    if value <= 0:
        raise ValueError(f"{label} must be positive")


def collect(args):
    """Collect raw greedy continuations and optional teacher-forced top-k evidence."""
    _positive(args.max_new_tokens, "max-new-tokens")
    if args.top_k < 0:
        raise ValueError("top-k must be nonnegative")
    prompts = read_observations(args.prompts)
    if not prompts:
        raise ValueError("Prompt dataset is empty")
    torch, _, AutoModel, AutoTokenizer = _deps()
    _seed(torch, args.seed)
    device = _device(torch, args.device)
    options = {"revision": args.revision, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    model = AutoModel.from_pretrained(args.model, **options).to(device).eval()
    fingerprint = _tokenizer_fingerprint(tokenizer)
    generation_config = _generation_config(model, tokenizer, args.max_new_tokens)
    context_limit = getattr(model.config, "max_position_embeddings", None)
    records = []
    for record in prompts:
        prefix = _render_prompt(tokenizer, record.prompt)
        if not prefix:
            raise ValueError(f"{record.id}: prompt encoded to no tokens")
        if context_limit and len(prefix) + args.max_new_tokens > context_limit:
            raise ValueError(
                f"{record.id}: prompt and generation exceed model context limit {context_limit}"
            )
        inputs = torch.tensor([prefix], device=device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=generation_config,
            )
            completion = generated[0, len(prefix) :].tolist()
            top_rows = None
            if args.top_k and completion:
                # Token j was predicted at prefix_length + j - 1.
                full = torch.tensor([prefix + completion], device=device)
                logits = model(input_ids=full, attention_mask=torch.ones_like(full)).logits[0]
                logp = (
                    logits[len(prefix) - 1 : len(prefix) + len(completion) - 1]
                    .float()
                    .log_softmax(-1)
                )
                values, indices = logp.topk(min(args.top_k, logp.shape[-1]), dim=-1)
                top_rows = [
                    [
                        {"token_id": int(token), "logprob": float(value)}
                        for token, value in zip(ids, vals)
                    ]
                    for ids, vals in zip(indices.cpu().tolist(), values.cpu().tolist())
                ]
        metadata = dict(record.metadata)
        metadata.update(
            {
                "source_model": str(args.model),
                "source_revision": args.revision,
                "source_commit": getattr(model.config, "_commit_hash", None),
                "tokenizer_fingerprint": fingerprint,
                "generation": {
                    "do_sample": False,
                    "seed": args.seed,
                    "max_new_tokens": args.max_new_tokens,
                },
                "top_k": args.top_k,
                "prompt_rendering": "chat_template" if tokenizer.chat_template else "plain",
                "effective_generation_config": generation_config.to_dict(),
                "logprob_semantics": "raw teacher-forced full-vocabulary log_softmax, before generation processing",
                "model_dtype": str(next(model.parameters()).dtype),
                "execution_device": device,
                "model_config_sha256": hashlib.sha256(
                    model.config.to_json_string(use_diff=False).encode()
                ).hexdigest(),
                "library_versions": {
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                },
            }
        )
        metadata.pop("completion_text", None)
        records.append(
            Observation(
                id=record.id,
                prompt=record.prompt,
                output=tokenizer.decode(completion, skip_special_tokens=False),
                metadata=metadata,
                prompt_token_ids=prefix,
                completion_token_ids=completion,
                top_logprobs=top_rows,
            )
        )
    write_observations(args.output, records)
    return {
        "observations": len(records),
        "output": str(args.output),
        "model": str(args.model),
        "tokenizer_fingerprint": fingerprint,
        "top_k": args.top_k,
        "completion_tokens": sum(len(r.completion_token_ids) for r in records),
    }


def _prepare(records, tokenizer, max_length, vocab_size):
    _positive(max_length, "max-seq-length")
    fingerprint = _tokenizer_fingerprint(tokenizer)
    prepared = []
    truncated = 0
    for record in records:
        if record.prompt_token_ids is not None or record.completion_token_ids is not None:
            if record.prompt_token_ids is None or record.completion_token_ids is None:
                raise ValueError(f"{record.id}: both prompt and completion token ids are required")
            if record.metadata.get("tokenizer_fingerprint") != fingerprint:
                raise ValueError(f"{record.id}: tokenizer fingerprint does not match the evidence")
            prefix, completion = list(record.prompt_token_ids), list(record.completion_token_ids)
            if prefix != _render_prompt(tokenizer, record.prompt):
                raise ValueError(f"{record.id}: stored prompt ids do not match the prompt/template")
            decoded = tokenizer.decode(completion, skip_special_tokens=False)
            # Exact serialized ids can carry the reasoning+answer target while
            # output/thinking remain separately readable fields.
            target_text = (
                decoded
                if record.thinking and "completion_text" not in record.metadata
                else _completion_text(record)
            )
            if decoded != target_text:
                raise ValueError(
                    f"{record.id}: stored completion ids do not match the visible target"
                )
        else:
            target_text = _completion_text(record)
            if record.top_logprobs is not None:
                raise ValueError(f"{record.id}: top-logprob evidence requires exact token ids")
            prefix = _render_prompt(tokenizer, record.prompt)
            completion = tokenizer.encode(target_text, add_special_tokens=False)
        if not prefix or not completion:
            raise ValueError(f"{record.id}: nonempty prompt and completion tokens required")
        if any(
            isinstance(t, bool) or not isinstance(t, int) or t < 0 or t >= vocab_size
            for t in prefix + completion
        ):
            raise ValueError(f"{record.id}: token id is outside model vocabulary")
        available = max_length - len(prefix)
        if available <= 0:
            raise ValueError(f"{record.id}: prompt leaves no continuation room in max-seq-length")
        if len(completion) > available:
            truncated += 1
        rows = record.top_logprobs
        if rows is not None:
            if len(rows) != len(completion):
                raise ValueError(f"{record.id}: top-logprob rows must align with completion tokens")
            if any(item["token_id"] >= vocab_size for row in rows for item in row):
                raise ValueError(f"{record.id}: top-logprob token id is outside model vocabulary")
        prepared.append(
            {
                "id": record.id,
                "prefix": prefix,
                "completion": completion[:available],
                "full_completion": completion,
                "top_logprobs": rows[:available] if rows is not None else None,
            }
        )
    return prepared, truncated


def _bucketed_loss(logits, rows):
    """Cross entropy on observed top-k tokens plus one residual-mass bucket.

    Missing vocabulary entries are never assigned zero teacher probability.
    """
    import torch

    if len(rows) != logits.shape[0]:
        raise ValueError("Top-logprob rows do not align with logits")
    losses = []
    for step, row in zip(logits.float(), rows):
        if not row:
            raise ValueError("Top-logprob rows must be nonempty")
        ids = [entry["token_id"] for entry in row]
        values = [entry["logprob"] for entry in row]
        if len(set(ids)) != len(ids) or any(i < 0 or i >= step.shape[-1] for i in ids):
            raise ValueError("Top-logprob ids must be unique and in vocabulary")
        if any(not math.isfinite(v) or v > 0 for v in values):
            raise ValueError("Top-logprobs must be finite and nonpositive")
        indices = torch.tensor(ids, device=step.device)
        probabilities = torch.tensor(values, device=step.device).exp()
        mass = probabilities.sum()
        if mass.item() > 1.00001:
            raise ValueError("Top-logprob mass exceeds one")
        if len(ids) == step.shape[-1] and abs(mass.item() - 1) > 1e-5:
            raise ValueError("Full-vocabulary top-logprob mass must sum to one")
        # Account for float32 rounding when a full-vocabulary row sums just above one.
        probabilities = probabilities / mass.clamp(min=1.0)
        tail_probability = (1 - probabilities.sum()).clamp(min=0)
        normalizer = torch.logsumexp(step, dim=-1)
        loss = -(probabilities * (step[indices] - normalizer)).sum()
        if len(ids) < step.shape[-1] and tail_probability.item() > 0:
            tail_logits = step.clone()
            tail_logits[indices] = -torch.inf
            loss = loss - tail_probability * (torch.logsumexp(tail_logits, dim=-1) - normalizer)
        losses.append(loss)
    if not losses:
        raise ValueError("No top-logprob rows")
    return torch.stack(losses).mean()


def _forward(torch, model, record, device):
    sequence = record["prefix"] + record["completion"]
    tokens = torch.tensor([sequence], device=device)
    logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits[0]
    start = len(record["prefix"]) - 1
    logits = logits[start : start + len(record["completion"])].float()
    targets = torch.tensor(record["completion"], device=device)
    return logits, targets


def _metrics(torch, model, records, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    bucket_loss = 0.0
    bucket_tokens = 0
    with torch.no_grad():
        for record in records:
            logits, targets = _forward(torch, model, record, device)
            n = len(targets)
            total_loss += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            count += n
            if record["top_logprobs"] is not None:
                bucket_loss += _bucketed_loss(logits, record["top_logprobs"]).item() * n
                bucket_tokens += n
    report = {
        "continuation_cross_entropy": total_loss / count,
        "teacher_token_agreement": correct / count,
        "evaluated_tokens": count,
        "examples": len(records),
    }
    if bucket_tokens:
        report["bucketed_cross_entropy"] = bucket_loss / bucket_tokens
        report["bucketed_tokens"] = bucket_tokens
    return report


def _dataset_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evidence_keys(records):
    """Persist leakage checks without putting original prompt text in the report."""
    from .evidence import normalized_prompt

    keys = set()
    for record in records:
        identities = [("id", record.id), ("prompt", normalized_prompt(record.prompt))]
        if record.metadata.get("group_id") is not None:
            identities.append(("group", record.metadata["group_id"]))
        for identity in identities:
            keys.add(hashlib.sha256(json.dumps(identity).encode()).hexdigest())
    return sorted(keys)


def fit(args):
    """Fit a candidate; teacher weights are read only for explicit pretrained init."""
    _positive(args.steps, "steps")
    _positive(args.eval_every, "eval-every")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not 0 <= args.topk_weight <= 1:
        raise ValueError("topk-weight must be between zero and one")
    train_records, validation_records = (
        read_observations(args.train),
        read_observations(args.validation),
    )
    if not train_records or not validation_records:
        raise ValueError("Training and validation datasets must both be nonempty")
    _check_disjoint(train_records, validation_records)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be absent or empty; choose a new run directory")
    torch, AutoConfig, AutoModel, AutoTokenizer = _deps()
    _seed(torch, args.seed)
    device = _device(torch, args.device)
    options = {"revision": args.revision, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, **options)
    config = AutoConfig.from_pretrained(args.teacher, **options)
    context_limit = getattr(config, "max_position_embeddings", None)
    if context_limit and args.max_seq_length > context_limit:
        raise ValueError(f"max-seq-length exceeds model context limit {context_limit}")
    train, train_truncated = _prepare(
        train_records, tokenizer, args.max_seq_length, config.vocab_size
    )
    validation, validation_truncated = _prepare(
        validation_records, tokenizer, args.max_seq_length, config.vocab_size
    )
    if args.init == "random":
        model = AutoModel.from_config(config, trust_remote_code=False)
    elif args.init == "pretrained":
        model = AutoModel.from_pretrained(args.teacher, **options)
    else:
        raise ValueError("init must be random or pretrained")
    model = model.to(device)
    model.config.use_cache = False
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    baseline = _metrics(torch, model, validation, device)
    best = dict(baseline)
    best_step = 0
    model.save_pretrained(output, safe_serialization=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    order = list(range(len(train)))
    selection_key = (
        "bucketed_cross_entropy"
        if all(r["top_logprobs"] is not None for r in validation) and args.topk_weight == 1
        else "continuation_cross_entropy"
    )
    for step in range(1, args.steps + 1):
        if (step - 1) % len(order) == 0:
            random.shuffle(order)
        record = train[order[(step - 1) % len(order)]]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, targets = _forward(torch, model, record, device)
        ce = torch.nn.functional.cross_entropy(logits, targets)
        loss = ce
        if record["top_logprobs"] is not None and args.topk_weight:
            loss = (1 - args.topk_weight) * ce + args.topk_weight * _bucketed_loss(
                logits, record["top_logprobs"]
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = _metrics(torch, model, validation, device)
            history.append({"step": step, "training_loss": loss.item(), "validation": metrics})
            if metrics[selection_key] < best[selection_key]:
                best, best_step = metrics, step
                model.save_pretrained(output, safe_serialization=True)
    report = {
        "initialization": args.init,
        "teacher_weights_used_as_initialization": args.init == "pretrained",
        "source_model": str(args.teacher),
        "source_revision": args.revision,
        "execution_device": device,
        "model_dtype": str(next(model.parameters()).dtype),
        "library_versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "tokenizer_fingerprint": _tokenizer_fingerprint(tokenizer),
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "topk_weight": args.topk_weight,
        "max_seq_length": args.max_seq_length,
        "checkpoint_selection_metric": selection_key,
        "best_step": best_step,
        "baseline_validation": baseline,
        "best_validation": best,
        "history": history,
        "train_sha256": _dataset_hash(args.train),
        "validation_sha256": _dataset_hash(args.validation),
        "train_examples": len(train),
        "seen_evidence_keys": _evidence_keys(train_records + validation_records),
        "validation_examples": len(validation),
        "truncated_train_examples": train_truncated,
        "truncated_validation_examples": validation_truncated,
        "interpretation": "Behavioral reconstruction; prompt/output evidence does not identify unique original weights.",
    }
    return _write_report(output / "metrics.json", report)


def evaluate(args):
    """Score held-out continuations; generation is an additional optional metric."""
    records = read_observations(args.data)
    if not records:
        raise ValueError("Evaluation dataset is empty")
    torch, _, AutoModel, AutoTokenizer = _deps()
    device = _device(torch, args.device)
    options = {"revision": args.revision, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    model = AutoModel.from_pretrained(args.model, **options).to(device).eval()
    if getattr(model.config, "max_position_embeddings", args.max_seq_length) < args.max_seq_length:
        raise ValueError("max-seq-length exceeds model context limit")
    prepared, truncated = _prepare(records, tokenizer, args.max_seq_length, model.config.vocab_size)
    # Candidate reports retain split keys so reformatted files cannot disguise
    # reuse of training or checkpoint-selection examples as held-out evaluation.
    run_file = Path(args.model) / "metrics.json"
    if run_file.is_file():
        run = json.loads(run_file.read_text(encoding="utf-8"))
        if _dataset_hash(args.data) in {run.get("train_sha256"), run.get("validation_sha256")}:
            raise ValueError(
                "Evaluation reuses the training/validation file; provide a held-out test split"
            )
        if set(_evidence_keys(records)).intersection(run.get("seen_evidence_keys", [])):
            raise ValueError(
                "Evaluation prompt/id/group overlaps training or validation; provide a held-out test split"
            )
    report = _metrics(torch, model, prepared, device)
    report.update(
        {
            "model": str(args.model),
            "data_sha256": _dataset_hash(args.data),
            "truncated_examples": truncated,
            "max_seq_length": args.max_seq_length,
            "execution_device": device,
            "model_dtype": str(next(model.parameters()).dtype),
        }
    )
    if args.generate:
        _positive(args.max_new_tokens, "max-new-tokens")
        exact = 0
        covered = 0
        context_limit = getattr(model.config, "max_position_embeddings", None)
        generation_config = _generation_config(model, tokenizer, args.max_new_tokens)
        with torch.no_grad():
            for record in prepared:
                if context_limit and len(record["prefix"]) + args.max_new_tokens > context_limit:
                    raise ValueError(f"{record['id']}: generation exceeds context limit")
                inputs = torch.tensor([record["prefix"]], device=device)
                generated = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    generation_config=generation_config,
                )
                continuation = generated[0, len(record["prefix"]) :].tolist()
                exact += continuation == record["full_completion"]
                covered += len(record["full_completion"]) <= args.max_new_tokens
        report["generation_exact_match"] = exact / len(prepared)
        report["generation_budget_covers_examples"] = covered
        report["generation_max_new_tokens"] = args.max_new_tokens
        report["effective_generation_config"] = generation_config.to_dict()
    return _write_report(args.output, report)


def compare_weights(args):
    """Compare aligned parameter tensors; permutations/gauges are not aligned here."""
    torch, _, AutoModel, _ = _deps()
    device = _device(torch, args.device)
    # Models are loaded separately: CPU copies are kept for the truth reference.
    truth_model = AutoModel.from_pretrained(args.truth, trust_remote_code=False)
    truth = {name: value.detach().cpu() for name, value in truth_model.named_parameters()}
    del truth_model
    candidate = AutoModel.from_pretrained(args.candidate, trust_remote_code=False).to(device)
    parameters = dict(candidate.named_parameters())
    if set(truth) != set(parameters) or any(
        truth[name].shape != parameters[name].shape for name in truth
    ):
        raise ValueError("Models must have identical parameter names and shapes")
    squared_error = truth_norm = candidate_norm = dot = 0.0
    count = 0
    with torch.no_grad():
        for name, reference in truth.items():
            # Accumulate on CPU in float64, one tensor at a time.
            a = reference.double()
            b = parameters[name].detach().cpu().double()
            squared_error += ((a - b) ** 2).sum().item()
            truth_norm += (a**2).sum().item()
            candidate_norm += (b**2).sum().item()
            dot += (a * b).sum().item()
            count += a.numel()
    relative_l2 = (
        math.sqrt(squared_error / truth_norm)
        if truth_norm
        else (0.0 if squared_error == 0 else None)
    )
    cosine = dot / math.sqrt(truth_norm * candidate_norm) if truth_norm and candidate_norm else None
    report = {
        "relative_l2": relative_l2,
        "cosine_similarity": cosine,
        "parameters": count,
        "truth": str(args.truth),
        "candidate": str(args.candidate),
        "interpretation": "Coordinate-wise comparison only; no neuron permutation or symmetry alignment.",
    }
    return _write_report(args.output, report)


def add_subcommands(subparsers):
    def device(parser):
        parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    parser = subparsers.add_parser(
        "collect", help="Collect local teacher continuations and optional top-k logprobs"
    )
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    device(parser)
    parser.set_defaults(func=collect)

    parser = subparsers.add_parser(
        "fit", help="Fit a candidate from evidence; random initialization by default"
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument(
        "--teacher",
        required=True,
        help="Source config/tokenizer; weights only read with --init pretrained",
    )
    parser.add_argument("--revision")
    parser.add_argument("--init", choices=["random", "pretrained"], default="random")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--topk-weight",
        type=float,
        default=1.0,
        help="Mix bucketed distillation with token CE (0..1)",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    device(parser)
    parser.set_defaults(func=fit)

    parser = subparsers.add_parser(
        "evaluate", help="Measure candidate agreement on held-out evidence"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    device(parser)
    parser.set_defaults(func=evaluate)

    parser = subparsers.add_parser(
        "compare-weights", help="Compare aligned truth/candidate parameter coordinates"
    )
    parser.add_argument("--truth", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    device(parser)
    parser.set_defaults(func=compare_weights)
