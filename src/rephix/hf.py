"""Optional local Transformers experiments, with explicit evidence boundaries.

Training from prompt/output pairs is behavioral reconstruction. It is not a
procedure for uniquely recovering a transformer's original parameters.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from . import __version__
from .evidence import Observation, read_observations, validate_observation
from .io import atomic_write_text, write_json


def _deps():
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install optional dependencies with: pip install -e ".[hf]"') from exc
    return torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _load_tokenizer(AutoTokenizer, source, *, model_config=None, **options):
    """Preserve Qwen tokenization when Transformers misidentifies a local tokenizer.

    Transformers 4.57.3–4.57.x considers local configs saved by those versions
    Mistral-like even when model_type is qwen2. Explicit False only for a known
    Qwen architecture avoids that warning and never changes its stored regex.
    Other architectures retain the library's normal behavior.
    """
    model_type = getattr(model_config, "model_type", None)
    config_file = Path(source) / "config.json"
    if model_type is None and config_file.is_file():
        model_type = json.loads(config_file.read_text(encoding="utf-8")).get("model_type")
    if model_type in {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"}:
        options.setdefault("fix_mistral_regex", False)
    return AutoTokenizer.from_pretrained(source, **options)


# Process-wide record of accelerator use, attached to command reports by the CLI.
ACCELERATOR = {}
_MPS_FALLBACK = re.compile(r"operator '([^']+)' is not currently supported on the MPS backend")


def _monitor_mps():
    """Record every operator PyTorch runs on the CPU instead of MPS.

    Without PYTORCH_ENABLE_MPS_FALLBACK=1 an unsupported operator raises, so a
    successful run executed every operator on MPS. With it, PyTorch warns once per
    operator; those warnings are captured and reported, never silently dropped.
    """
    if ACCELERATOR:
        return
    import warnings

    enabled = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    ACCELERATOR.update(
        {"device": "mps", "cpu_fallback_enabled": enabled, "cpu_fallback_operators": []}
    )
    previous = warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        match = _MPS_FALLBACK.search(str(message))
        if match and match.group(1) not in ACCELERATOR["cpu_fallback_operators"]:
            ACCELERATOR["cpu_fallback_operators"].append(match.group(1))
        previous(message, category, filename, lineno, file, line)

    warnings.showwarning = show
    warnings.filterwarnings("always", message=".*not currently supported on the MPS backend.*")
    if enabled:
        print(
            "rephix: PYTORCH_ENABLE_MPS_FALLBACK=1: unsupported MPS operators will run on the "
            "CPU and are listed under accelerator.cpu_fallback_operators",
            file=sys.stderr,
        )


def _device(torch, name, monitor=True):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if monitor:
                _monitor_mps()
            return "mps"
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            built = bool(mps is not None and mps.is_built())
            raise ValueError(
                "MPS was requested but is unavailable ("
                + (
                    "this PyTorch build supports MPS, but no Metal device is usable: macOS "
                    "12.3+ on Apple silicon with a non-virtualized GPU is required"
                    if built
                    else "this PyTorch build has no MPS support"
                )
                + "). Use --device cpu."
            )
        if monitor:
            _monitor_mps()
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
        write_json(path, report)
    return report


def _positive(value, label):
    if value <= 0:
        raise ValueError(f"{label} must be positive")


def _completion_logits(torch, model, prefix, completion, device):
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        # The last observed token is a target, so the final input position
        # needed is the one immediately before it. Skip projecting the prefix.
        tokens = torch.tensor([(prefix + completion)[:-1]], device=device)
        logits = model(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), logits_to_keep=len(completion)
        ).logits[0]
    else:
        tokens = torch.tensor([prefix + completion], device=device)
        all_logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits[0]
        start = len(prefix) - 1
        logits = all_logits[start : start + len(completion)]
    return logits.float()


def _collection_digest(payload):
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect(args):
    """Collect raw greedy continuations and optional teacher-forced top-k evidence."""
    _positive(args.max_new_tokens, "max-new-tokens")
    if args.top_k < 0:
        raise ValueError("top-k must be nonnegative")
    prompts = read_observations(args.prompts)
    output = Path(args.output)
    journal = output.with_suffix(output.suffix + ".collection.json")
    resume = getattr(args, "resume", False)
    if not resume and (output.exists() or journal.exists()):
        raise ValueError("Collection output exists; choose a new output or use --resume")
    if resume and not journal.is_file():
        raise ValueError("Cannot resume collection without its .collection.json manifest")
    torch, _, AutoModel, AutoTokenizer = _deps()
    _seed(torch, args.seed)
    device = _device(torch, args.device)
    options = {"revision": args.revision, "trust_remote_code": False}
    model_source = _resolve_model_path(args.model)
    tokenizer = _load_tokenizer(AutoTokenizer, model_source, **options)
    model = AutoModel.from_pretrained(model_source, **options).to(device).eval()
    fingerprint = _tokenizer_fingerprint(tokenizer)
    generation_config = _generation_config(model, tokenizer, args.max_new_tokens)
    context_limit = getattr(model.config, "max_position_embeddings", None)
    settings = {
        "schema_version": 1,
        "prompts_sha256": _dataset_hash(args.prompts),
        "source_model": str(args.model),
        "source_revision": args.revision,
        "source_commit": getattr(model.config, "_commit_hash", None),
        "source_weights_sha256": _source_weights_fingerprint(model_source),
        "source_config_sha256": _config_fingerprint(model.config),
        "tokenizer_fingerprint": fingerprint,
        "generation_config": generation_config.to_dict(),
        "top_k": args.top_k,
        "seed": args.seed,
    }
    settings_fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    environment = {
        "execution_device": device,
        "model_dtype": str(next(model.parameters()).dtype),
        "torch_num_threads": torch.get_num_threads(),
        "torch": str(torch.__version__),
        "transformers": __import__("transformers").__version__,
        "logits_to_keep": "logits_to_keep" in inspect.signature(model.forward).parameters,
    }
    records = []
    payload = ""
    if resume:
        existing = json.loads(journal.read_text(encoding="utf-8"))
        if existing.get("settings") != settings:
            raise ValueError(
                "Collection resume settings, source weights, or prompt evidence changed"
            )
        checksummed = "environment" in existing or "evidence_sha256" in existing
        if checksummed:
            if existing.get("environment") != environment:
                raise ValueError("Collection resume execution environment changed")
        if output.is_file():
            payload = output.read_bytes().decode("utf-8")
            records = read_observations(output)
        if checksummed:
            count = existing.get("completed_observations")
            digest = existing.get("evidence_sha256")
            if type(count) is not int or count < 0 or not isinstance(digest, str):
                raise ValueError("Invalid collection checksum manifest")
            pending = existing.get("pending_evidence_sha256")
            actual = (len(records), _collection_digest(payload))
            if actual != (count, digest) and actual != (count + 1, pending):
                raise ValueError("Collection evidence checksum does not match the resume manifest")
        if len(records) > len(prompts):
            raise ValueError("Collection output contains more observations than the prompt source")
        for observed, prompt in zip(records, prompts):
            if observed.id != prompt.id or observed.prompt != prompt.prompt:
                raise ValueError("Collection output is not a prefix of the original prompt source")
            if observed.metadata.get("collection_fingerprint") != settings_fingerprint:
                raise ValueError("Collection output provenance does not match the resume manifest")

    def write_journal(count, digest, *, pending=None, status="collecting"):
        report = {
            "settings": settings,
            "environment": environment,
            "completed_observations": count,
            "evidence_sha256": digest,
            "status": status,
        }
        if pending is not None:
            report["pending_evidence_sha256"] = pending
        _write_report(journal, report)

    digest = _collection_digest(payload)
    write_journal(len(records), digest)
    for record in prompts[len(records) :]:
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
                logp = _completion_logits(torch, model, prefix, completion, device).log_softmax(-1)
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
                "collection_fingerprint": settings_fingerprint,
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
                    "rephix": __version__,
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                },
            }
        )
        metadata.pop("completion_text", None)
        observed = Observation(
            id=record.id,
            prompt=record.prompt,
            output=tokenizer.decode(completion, skip_special_tokens=False),
            metadata=metadata,
            prompt_token_ids=prefix,
            completion_token_ids=completion,
            top_logprobs=top_rows,
        )
        validate_observation(observed)
        separator = "\n" if payload and not payload.endswith("\n") else ""
        next_payload = payload + separator + json.dumps(observed.to_dict(), allow_nan=False) + "\n"
        next_digest = _collection_digest(next_payload)
        # Publish both valid checksums before replacing the output. Resume can
        # recognize either side of the atomic replacement.
        write_journal(len(records), digest, pending=next_digest)
        atomic_write_text(output, next_payload, newline="")
        records.append(observed)
        payload, digest = next_payload, next_digest
    write_journal(len(records), digest, status="complete")
    return {
        "observations": len(records),
        "resumed": resume,
        "collection_manifest": str(journal),
        "output": str(args.output),
        "model": str(args.model),
        "tokenizer_fingerprint": fingerprint,
        "top_k": args.top_k,
        "evidence_sha256": digest,
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
    if not rows:
        raise ValueError("No top-logprob rows")
    vocab_size = logits.shape[-1]
    width = max(map(len, rows))
    token_ids, weights, tails, mask_ids = [], [], [], []
    for row in rows:
        if not row:
            raise ValueError("Top-logprob rows must be nonempty")
        ids = [entry["token_id"] for entry in row]
        values = [entry["logprob"] for entry in row]
        if len(set(ids)) != len(ids) or any(i < 0 or i >= vocab_size for i in ids):
            raise ValueError("Top-logprob ids must be unique and in vocabulary")
        if any(not math.isfinite(v) or v > 0 for v in values):
            raise ValueError("Top-logprobs must be finite and nonpositive")
        probabilities = [math.exp(value) for value in values]
        mass = math.fsum(probabilities)
        if mass > 1.00001:
            raise ValueError("Top-logprob mass exceeds one")
        if len(ids) == vocab_size and abs(mass - 1) > 1e-5:
            raise ValueError("Full-vocabulary top-logprob mass must sum to one")
        probabilities = [probability / max(mass, 1.0) for probability in probabilities]
        tail = max(1 - mass, 0.0) if len(ids) < vocab_size else 0.0
        padding = width - len(ids)
        token_ids.append(ids + [0] * padding)
        weights.append(probabilities + [0.0] * padding)
        tails.append(tail)
        mask_ids.append(ids + [vocab_size] * padding if tail else [vocab_size] * width)
    logits = logits.float()
    indices = torch.tensor(token_ids, device=logits.device)
    probabilities = logits.new_tensor(weights)
    normalizer = logits.logsumexp(dim=-1)
    losses = -(probabilities * (logits.gather(1, indices) - normalizer[:, None])).sum(dim=-1)
    if any(tails):
        # Ragged padding scatters into an extra column that never enters the
        # tail. Rows with no teacher tail keep finite logits for backward.
        tail_logits = torch.nn.functional.pad(logits, (0, 1))
        tail_logits.scatter_(1, torch.tensor(mask_ids, device=logits.device), -torch.inf)
        tail_logp = tail_logits[:, :-1].logsumexp(dim=-1) - normalizer
        losses = losses - logits.new_tensor(tails) * tail_logp
    return losses.mean()


def _forward(torch, model, record, device):
    logits = _completion_logits(torch, model, record["prefix"], record["completion"], device)
    targets = torch.tensor(record["completion"], device=device)
    return logits, targets


def _metrics(torch, model, records, device, topk_weight=1.0):
    """Token-weighted completion metrics; recorded tokens need not be teacher argmax."""
    model.eval()
    total_loss = objective_loss = 0.0
    correct = count = bucket_tokens = observed_top_correct = 0
    bucket_loss = 0.0
    with torch.no_grad():
        for record in records:
            logits, targets = _forward(torch, model, record, device)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced nonfinite evaluation logits")
            n = len(targets)
            ce = torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            total_loss += ce
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            count += n
            objective = ce
            if record["top_logprobs"] is not None:
                bucket = _bucketed_loss(logits, record["top_logprobs"]).item() * n
                bucket_loss += bucket
                bucket_tokens += n
                observed = torch.tensor(
                    [
                        max(row, key=lambda item: item["logprob"])["token_id"]
                        for row in record["top_logprobs"]
                    ],
                    device=logits.device,
                )
                observed_top_correct += (logits.argmax(dim=-1) == observed).sum().item()
                objective = (1 - topk_weight) * ce + topk_weight * bucket
            objective_loss += objective
    if not count:
        raise ValueError("No supervised continuation tokens")
    report = {
        "continuation_cross_entropy": total_loss / count,
        "recorded_token_agreement": correct / count,
        # Backward-compatible alias. For sampled external observations this is
        # agreement with the sampled token, not necessarily teacher argmax.
        "teacher_token_agreement": correct / count,
        "reconstruction_objective": objective_loss / count,
        "objective_topk_weight": topk_weight,
        "evaluated_tokens": count,
        "examples": len(records),
    }
    if bucket_tokens:
        report["bucketed_cross_entropy"] = bucket_loss / bucket_tokens
        report["bucketed_tokens"] = bucket_tokens
        report["observed_top_token_agreement"] = observed_top_correct / bucket_tokens
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


def _config_fingerprint(config):
    payload = config.to_dict()
    # Name and incidental loading metadata do not alter the architecture.
    payload.pop("_name_or_path", None)
    payload.pop("transformers_version", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _source_weights_fingerprint(source):
    """Detect changed local pretrained priors without reading weights for random init."""
    root = Path(source)
    if not root.is_dir():
        return None
    files = sorted([*root.glob("*.safetensors"), *root.glob("*.bin")])
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(torch, payload, path):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save_best(model, tokenizer, output, step):
    """An immutable directory is complete before any manifest points to it."""
    checkpoints = output / ".checkpoints"
    checkpoints.mkdir(exist_ok=True)
    name = f"best-{step:08d}-{uuid.uuid4().hex[:12]}"
    staging = checkpoints / f".{name}.tmp"
    destination = checkpoints / name
    try:
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return str(destination.relative_to(output))


def _publish_best(output, relative_path):
    """Publish a pointer atomically; root files preserve ordinary HF compatibility."""
    checkpoint = output / relative_path
    if not checkpoint.is_dir():
        raise ValueError("The durable best checkpoint is missing")
    _write_report(output / "best_checkpoint.json", {"path": relative_path})
    for source in checkpoint.iterdir():
        if not source.is_file():
            continue
        destination = output / source.name
        temporary = output / f".{source.name}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def _resolve_model_path(source):
    """Resolve a Rephix best-checkpoint manifest, otherwise use a normal model path/id."""
    root = Path(source)
    manifest = root / "best_checkpoint.json"
    if not manifest.is_file():
        return str(source)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    relative = Path(data["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid best-checkpoint manifest path")
    checkpoint = root / relative
    if not checkpoint.is_dir():
        raise ValueError("The best-checkpoint manifest points to a missing directory")
    return str(checkpoint)


def _set_trainable(torch, model, mode):
    untied = False
    if mode == "head":
        head = model.get_output_embeddings()
        inputs = model.get_input_embeddings()
        if head is None:
            raise ValueError("This architecture has no trainable output head")
        # Tied embeddings would otherwise update the supposedly frozen backbone.
        if inputs is not None and getattr(head, "weight", None) is getattr(inputs, "weight", None):
            head.weight = torch.nn.Parameter(head.weight.detach().clone())
            model.config.tie_word_embeddings = False
            untied = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in head.parameters():
            parameter.requires_grad_(True)
    elif mode != "all":
        raise ValueError("trainable must be all or head")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters")
    return trainable, untied


def fit(args):
    """Fit or resume a candidate, persisting latest optimizer state separately from best weights.

    ``steps`` is the total desired optimizer-step count, including completed steps.
    A crash can lose work after the last durable save, but cannot partially replace it.
    """
    _positive(args.steps, "steps")
    _positive(args.eval_every, "eval-every")
    accumulation = getattr(args, "gradient_accumulation_steps", 1)
    save_every = getattr(args, "save_every", 10)
    _positive(accumulation, "gradient-accumulation-steps")
    _positive(save_every, "save-every")
    resume = getattr(args, "resume", False)
    trainable_mode = getattr(args, "trainable", "all")
    student_source = getattr(args, "student_config", None)
    student_revision = getattr(args, "student_revision", None)
    if student_revision and not student_source:
        raise ValueError("student-revision requires student-config")
    if student_source and args.init != "random":
        raise ValueError("student-config requires random initialization")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.topk_weight) or not 0 <= args.topk_weight <= 1:
        raise ValueError("topk-weight must be between zero and one")
    train_records = read_observations(args.train)
    validation_records = read_observations(args.validation)
    _check_disjoint(train_records, validation_records)
    output = Path(args.output)
    state_file = output / "training_state.pt"
    if resume:
        if not state_file.is_file():
            raise ValueError("Cannot resume: output/training_state.pt does not exist")
    elif output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory must be absent or empty; use --resume for an existing run"
        )
    torch, AutoConfig, AutoModel, AutoTokenizer = _deps()
    _seed(torch, args.seed)
    device = _device(torch, args.device)
    options = {"revision": args.revision, "trust_remote_code": False}
    tokenizer = _load_tokenizer(AutoTokenizer, args.teacher, **options)
    reference_config = AutoConfig.from_pretrained(args.teacher, **options)
    config = (
        AutoConfig.from_pretrained(
            student_source, revision=student_revision, trust_remote_code=False
        )
        if student_source
        else reference_config
    )
    if config.vocab_size != reference_config.vocab_size:
        raise ValueError(
            "Student vocabulary size must match the reference model/tokenizer vocabulary"
        )
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(config, name, None) != getattr(reference_config, name, None):
            raise ValueError(f"Student {name} must match the reference model")
    context_limit = getattr(config, "max_position_embeddings", None)
    if context_limit and args.max_seq_length > context_limit:
        raise ValueError(f"max-seq-length exceeds student context limit {context_limit}")
    train, train_truncated = _prepare(
        train_records, tokenizer, args.max_seq_length, config.vocab_size
    )
    validation, validation_truncated = _prepare(
        validation_records, tokenizer, args.max_seq_length, config.vocab_size
    )
    selection_key = getattr(args, "selection_metric", "reconstruction_objective")
    if selection_key not in {"reconstruction_objective", "continuation_cross_entropy"}:
        raise ValueError("Unsupported checkpoint selection metric")
    versions = {
        "rephix": __version__,
        "torch": str(torch.__version__),
        "transformers": __import__("transformers").__version__,
    }
    run_config = {
        "schema_version": 1,
        "source_model": str(args.teacher),
        "source_revision": args.revision,
        "source_commit": getattr(reference_config, "_commit_hash", None),
        "source_config_sha256": _config_fingerprint(reference_config),
        "source_weights_sha256": _source_weights_fingerprint(args.teacher)
        if args.init == "pretrained"
        else None,
        "student_config": str(student_source) if student_source else None,
        "student_revision": student_revision,
        "student_commit": getattr(config, "_commit_hash", None),
        "student_config_sha256": _config_fingerprint(config),
        "initialization": args.init,
        "trainable": trainable_mode,
        "tokenizer_fingerprint": _tokenizer_fingerprint(tokenizer),
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "topk_weight": args.topk_weight,
        "max_seq_length": args.max_seq_length,
        "eval_every": args.eval_every,
        "gradient_accumulation_steps": accumulation,
        "execution_device": device,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "library_versions": versions,
        "train_sha256": _dataset_hash(args.train),
        "validation_sha256": _dataset_hash(args.validation),
        "seen_evidence_keys": _evidence_keys(train_records + validation_records),
    }
    if selection_key != "reconstruction_objective":
        run_config["selection_metric"] = selection_key
    state = None
    if resume:
        try:
            state = torch.load(state_file, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("Cannot safely load the durable training state") from exc
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("Unsupported training-state schema")
        if state.get("run_config") != run_config:
            previous = state.get("run_config", {})
            changed = sorted(
                key
                for key in set(previous) | set(run_config)
                if previous.get(key) != run_config.get(key)
            )
            raise ValueError(
                f"Resume configuration differs from the original run: {', '.join(changed)}"
            )
        if args.steps < state["completed_steps"]:
            raise ValueError("steps cannot be less than the number of completed steps")
    if args.init == "random" or resume:
        model = AutoModel.from_config(config, trust_remote_code=False)
    elif args.init == "pretrained":
        weight_options = dict(options)
        if getattr(reference_config, "_commit_hash", None):
            weight_options["revision"] = reference_config._commit_hash
        model = AutoModel.from_pretrained(args.teacher, **weight_options)
    else:
        raise ValueError("init must be random or pretrained")
    _, untied = _set_trainable(torch, model, trainable_mode)
    model = model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    if state is None:
        baseline = _metrics(torch, model, validation, device, args.topk_weight)
        best, best_step = dict(baseline), 0
        history, order, sample_index, completed = [], list(range(len(train))), 0, 0
        best_checkpoint = _save_best(model, tokenizer, output, 0)
    else:
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        baseline, best = state["baseline_validation"], state["best_validation"]
        best_step, best_checkpoint = state["best_step"], state["best_checkpoint"]
        history, order = state["history"], state["order"]
        sample_index, completed = state["sample_index"], state["completed_steps"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if device == "mps" and state.get("mps_rng") is not None:
            torch.mps.set_rng_state(state["mps_rng"])
        del state

    def report():
        return {
            **run_config,
            "teacher_weights_used_as_initialization": args.init == "pretrained",
            "frozen_backbone_uses_pretrained_prior": trainable_mode == "head"
            and args.init == "pretrained",
            "untied_output_embeddings": untied,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "model_dtype": str(next(model.parameters()).dtype),
            "steps": args.steps,
            "completed_steps": completed,
            "save_every": save_every,
            "resumed": resume,
            "checkpoint_selection_metric": selection_key,
            "best_step": best_step,
            "best_checkpoint": str(output / best_checkpoint),
            "best_checkpoint_relative": best_checkpoint,
            "latest_training_state": str(state_file),
            "baseline_validation": baseline,
            "best_validation": best,
            "history": history,
            "train_examples": len(train),
            "validation_examples": len(validation),
            "truncated_train_examples": train_truncated,
            "truncated_validation_examples": validation_truncated,
            "supervised_train_tokens": sum(len(record["completion"]) for record in train),
            "interpretation": "Behavioral reconstruction; prompt/output evidence does not identify unique original weights.",
        }

    def persist():
        payload = {
            "schema_version": 1,
            "run_config": run_config,
            "completed_steps": completed,
            "sample_index": sample_index,
            "order": order,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
            "baseline_validation": baseline,
            "best_validation": best,
            "best_step": best_step,
            "best_checkpoint": best_checkpoint,
            "history": history,
        }
        _atomic_torch_save(torch, payload, state_file)
        _write_report(output / "training_config.json", run_config)
        _write_report(output / "metrics.json", report())
        # The prior checkpoint is retained until the atomic state commit succeeds.
        _publish_best(output, best_checkpoint)
        for checkpoint in (output / ".checkpoints").glob("best-*"):
            if str(checkpoint.relative_to(output)) != best_checkpoint:
                shutil.rmtree(checkpoint)

    persist()
    for step in range(completed + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_loss = 0.0
        for _ in range(accumulation):
            if sample_index % len(order) == 0:
                random.shuffle(order)
            record = train[order[sample_index % len(order)]]
            sample_index += 1
            logits, targets = _forward(torch, model, record, device)
            ce = torch.nn.functional.cross_entropy(logits, targets)
            loss = ce
            if record["top_logprobs"] is not None and args.topk_weight:
                loss = (1 - args.topk_weight) * ce + args.topk_weight * _bucketed_loss(
                    logits, record["top_logprobs"]
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Training loss became nonfinite; resume from the last durable state"
                )
            training_loss += loss.item() / accumulation
            (loss / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        completed = step
        evaluated = step % args.eval_every == 0 or step == args.steps
        if evaluated:
            metrics = _metrics(torch, model, validation, device, args.topk_weight)
            history.append({"step": step, "training_loss": training_loss, "validation": metrics})
            if metrics[selection_key] < best[selection_key]:
                best, best_step = metrics, step
                best_checkpoint = _save_best(model, tokenizer, output, step)
        if evaluated or step % save_every == 0:
            persist()
    return report()


def evaluate(args):
    """Score held-out continuations; generation is an additional optional metric."""
    generation_details = getattr(args, "generation_details", False)
    if generation_details and not args.generate:
        raise ValueError("generation-details requires generate")
    records = read_observations(args.data)
    if not records:
        raise ValueError("Evaluation dataset is empty")
    torch, _, AutoModel, AutoTokenizer = _deps()
    device = _device(torch, args.device)
    options = {"revision": args.revision, "trust_remote_code": False}
    model_source = _resolve_model_path(args.model)
    tokenizer = _load_tokenizer(AutoTokenizer, model_source, **options)
    model = AutoModel.from_pretrained(model_source, **options).to(device).eval()
    if getattr(model.config, "max_position_embeddings", args.max_seq_length) < args.max_seq_length:
        raise ValueError("max-seq-length exceeds model context limit")
    prepared, truncated = _prepare(records, tokenizer, args.max_seq_length, model.config.vocab_size)
    # Candidate reports retain split keys so reformatted files cannot disguise
    # reuse of training or checkpoint-selection examples as held-out evaluation.
    run_file = Path(args.model) / "metrics.json"
    run = {}
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
    explicit_weight = getattr(args, "topk_weight", None)
    objective_weight = (
        explicit_weight if explicit_weight is not None else run.get("topk_weight", 1.0)
    )
    if (
        not isinstance(objective_weight, (float, int))
        or not math.isfinite(objective_weight)
        or not 0 <= objective_weight <= 1
    ):
        raise ValueError("Evaluation topk-weight must be finite and between zero and one")
    report = _metrics(torch, model, prepared, device, objective_weight)
    report["objective_topk_weight_source"] = (
        "explicit"
        if explicit_weight is not None
        else "training_run"
        if "topk_weight" in run
        else "default"
    )
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
        first_token_correct = matched_prefix_tokens = reference_tokens = 0
        covered = 0
        details = []
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
                reference = record["full_completion"]
                exact += continuation == reference
                covered += len(reference) <= args.max_new_tokens
                first_token_correct += continuation[:1] == reference[:1]
                reference_tokens += len(reference)
                prefix_match = 0
                for generated_token, reference_token in zip(continuation, reference):
                    if generated_token != reference_token:
                        break
                    prefix_match += 1
                matched_prefix_tokens += prefix_match
                if generation_details:
                    details.append(
                        {
                            "id": record["id"],
                            "candidate_token_ids": continuation,
                            "candidate_text": tokenizer.decode(
                                continuation, skip_special_tokens=False
                            ),
                            "matched_prefix_tokens": prefix_match,
                            "reference_tokens": len(reference),
                            "exact_match": continuation == reference,
                        }
                    )
        report["generation_exact_match"] = exact / len(prepared)
        report["generation_first_token_agreement"] = first_token_correct / len(prepared)
        report["generation_reference_prefix_fraction"] = matched_prefix_tokens / reference_tokens
        report["generation_budget_covers_examples"] = covered
        report["generation_max_new_tokens"] = args.max_new_tokens
        report["effective_generation_config"] = generation_config.to_dict()
        if generation_details:
            report["generation_details"] = details
    return _write_report(args.output, report)


def compare_weights(args):
    """Compare aligned parameter tensors; permutations/gauges are not aligned here."""
    torch, _, AutoModel, _ = _deps()
    device = _device(torch, args.device)
    # Models are loaded separately: CPU copies are kept for the truth reference.
    truth_model = AutoModel.from_pretrained(
        _resolve_model_path(args.truth), trust_remote_code=False
    )
    truth = {name: value.detach().cpu() for name, value in truth_model.named_parameters()}
    del truth_model
    candidate = AutoModel.from_pretrained(
        _resolve_model_path(args.candidate), trust_remote_code=False
    ).to(device)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a durable partial collection with unchanged settings",
    )
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
    parser.add_argument(
        "--selection-metric",
        choices=["reconstruction_objective", "continuation_cross_entropy"],
        default="reconstruction_objective",
        help="Validation score used to select the best checkpoint",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Durable latest-state interval; validation checkpoints are always saved",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue this output directory to the total --steps target",
    )
    parser.add_argument(
        "--trainable",
        choices=["all", "head"],
        default="all",
        help="Optimize all parameters or only an untied output head",
    )
    parser.add_argument(
        "--student-config",
        help="Optional smaller architecture config for random initialization; vocabulary/special IDs must match",
    )
    parser.add_argument(
        "--student-revision", help="Revision for a separate student-config repository"
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
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
    parser.add_argument(
        "--generation-details",
        action="store_true",
        help="Include per-prompt generated continuations",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--topk-weight",
        type=float,
        help="Evaluation objective mixture; defaults to saved training value, otherwise 1",
    )
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
