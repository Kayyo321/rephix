"""Choose informative new prompts without consulting their target completions.

Scores use candidate models, never a reference teacher. Exclusion joins prompts,
ids, and explicit groups transitively so a related pool entry cannot evade a
reserved evaluation set by changing its id.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evidence import (
    Observation,
    normalized_prompt,
    read_observations,
    validate_observation,
    validate_observations,
    write_observations,
)
from .io import write_json


def _probabilities(logits) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("candidate logits must be a nonempty finite vector")
    with np.errstate(over="ignore"):
        shifted = values - values.max()
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def entropy(logits) -> float:
    """Shannon entropy in nats of a full candidate next-token distribution."""
    probabilities = _probabilities(logits)
    positive = probabilities > 0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


def jensen_shannon(first_logits, second_logits) -> float:
    """Symmetric Jensen-Shannon divergence in nats, bounded by log(2)."""
    first, second = _probabilities(first_logits), _probabilities(second_logits)
    if first.shape != second.shape:
        raise ValueError("candidate distributions must use the same vocabulary")
    mixture = (first + second) / 2

    def divergence(distribution):
        positive = distribution > 0
        return np.sum(distribution[positive] * np.log(distribution[positive] / mixture[positive]))

    value = float((divergence(first) + divergence(second)) / 2)
    return min(math.log(2), max(0.0, value))


def _identities(record: Observation):
    keys = [("id", record.id), ("prompt", normalized_prompt(record.prompt))]
    if record.metadata.get("group_id") is not None:
        keys.append(("group", record.metadata["group_id"]))
    return keys


def _eligible_groups(
    pool: Iterable[Observation], excluded: Iterable[Observation] = ()
) -> list[list[Observation]]:
    pool = validate_observations(pool)
    excluded = list(excluded)
    for record in excluded:
        validate_observation(record)
    # Duplicate ids across the pool/exclusions are expected evidence of overlap.
    # Each input dataset has already been validated independently by its reader.
    records = pool + excluded
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, record in enumerate(records):
        for key in _identities(record):
            if key in seen:
                parents[root(index)] = root(seen[key])
            seen[key] = index
    forbidden = {root(index) for index in range(len(pool), len(records))}
    groups = {}
    for index, record in enumerate(pool):
        component = root(index)
        if component not in forbidden:
            groups.setdefault(component, []).append(record)
    result = [sorted(group, key=lambda record: record.id) for group in groups.values()]
    return sorted(result, key=lambda group: tuple(record.id for record in group))


@dataclass(frozen=True)
class QueryChoice:
    record: Observation
    score: float
    group_size: int


def select_records(
    pool: Iterable[Observation],
    *,
    excluded: Iterable[Observation] = (),
    budget: int,
    strategy: str = "random",
    seed: int = 42,
    scores: dict[str, float] | None = None,
) -> list[QueryChoice]:
    """Select at most one prompt per independent group, using no target labels.

    For model strategies, ``scores`` is indexed by observation id. The CLI
    computes these scores from prompt tokens alone. Random group selection is
    uniform over groups and is independent of how many rows a group contains.
    """
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if strategy not in {"random", "entropy", "disagreement"}:
        raise ValueError("strategy must be random, entropy, or disagreement")
    groups = _eligible_groups(pool, excluded)
    if not groups:
        raise ValueError("no independent query groups remain after exclusions")
    choices = []
    generator = random.Random(seed)
    for group in groups:
        if strategy == "random":
            record = generator.choice(group)
            score = generator.random()
        else:
            if scores is None or any(record.id not in scores for record in group):
                raise ValueError("a candidate score is required for every eligible prompt")
            if any(not math.isfinite(scores[record.id]) for record in group):
                raise ValueError("candidate scores must be finite")
            record = min(group, key=lambda row: (-scores[row.id], row.id))
            score = float(scores[record.id])
        choices.append(QueryChoice(record, score, len(group)))
    choices.sort(key=lambda choice: (-choice.score, choice.record.id))
    return choices[:budget]


def _candidate_scores(groups, args) -> tuple[dict[str, float], dict]:
    from .hf import (
        _deps,
        _device,
        _load_tokenizer,
        _render_prompt,
        _resolve_model_path,
        _seed,
        _tokenizer_fingerprint,
    )

    torch, _, AutoModel, AutoTokenizer = _deps()
    _seed(torch, args.seed)
    device = _device(torch, args.device)
    options = {"revision": args.revision, "trust_remote_code": False}
    model_path = _resolve_model_path(args.model)
    tokenizer = _load_tokenizer(AutoTokenizer, model_path, **options)
    fingerprint = _tokenizer_fingerprint(tokenizer)
    candidate = AutoModel.from_pretrained(model_path, **options).to(device).eval()
    reference = None
    reference_path = None
    if args.strategy == "disagreement":
        reference_path = _resolve_model_path(args.reference_candidate)
        other_tokenizer = _load_tokenizer(AutoTokenizer, reference_path, **options)
        if _tokenizer_fingerprint(other_tokenizer) != fingerprint:
            raise ValueError("disagreement candidates must use identical tokenizers and templates")
        reference = AutoModel.from_pretrained(reference_path, **options).to(device).eval()
        if candidate.config.vocab_size != reference.config.vocab_size:
            raise ValueError("disagreement candidates must use the same model vocabulary")

    def logits(model, prefix):
        limit = getattr(model.config, "max_position_embeddings", args.max_seq_length)
        if len(prefix) > limit:
            raise ValueError(f"prompt exceeds candidate context limit {limit}")
        inputs = torch.tensor([prefix], device=device)
        result = model(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False
        ).logits[0, -1]
        # Copy the final vector so a CPU NumPy view does not retain the whole
        # prompt-by-vocabulary logits allocation while scoring the other model.
        return result.float().cpu().numpy().copy()

    scores = {}
    with torch.no_grad():
        for group in groups:
            for record in group:
                prefix = _render_prompt(tokenizer, record.prompt)
                if not prefix or len(prefix) > args.max_seq_length:
                    raise ValueError(
                        f"{record.id}: prompt is empty or exceeds max-seq-length; "
                        "query selection does not silently truncate prompts"
                    )
                first = logits(candidate, prefix)
                scores[record.id] = (
                    entropy(first)
                    if reference is None
                    else jensen_shannon(first, logits(reference, prefix))
                )
    return scores, {
        "model": str(args.model),
        "resolved_model_path": str(model_path),
        "reference_candidate": str(args.reference_candidate) if reference is not None else None,
        "resolved_reference_candidate": str(reference_path) if reference is not None else None,
        "revision": args.revision,
        "source_commit": getattr(candidate.config, "_commit_hash", None),
        "tokenizer_fingerprint": fingerprint,
        "execution_device": device,
        "model_dtype": str(next(candidate.parameters()).dtype),
        "library_versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "scoring_context": "next token at the end of the rendered prompt; no completions read",
    }


_COLLECTOR_METADATA = {
    "completion_text",
    "source_model",
    "source_revision",
    "source_commit",
    "tokenizer_fingerprint",
    "generation",
    "top_k",
    "prompt_rendering",
    "effective_generation_config",
    "logprob_semantics",
    "model_dtype",
    "execution_device",
    "model_config_sha256",
    "library_versions",
    "query_selection",
}


def select_queries(args):
    if args.max_seq_length <= 0:
        raise ValueError("max-seq-length must be positive")
    if type(args.budget) is not int or args.budget <= 0:
        raise ValueError("budget must be a positive integer")
    if args.strategy != "random" and not args.model:
        raise ValueError("entropy and disagreement strategies require --model")
    if args.strategy == "disagreement" and not args.reference_candidate:
        raise ValueError("disagreement requires --reference-candidate")
    output = Path(args.output)
    report_path = Path(str(output) + ".report.json")
    if output.exists() or report_path.exists():
        raise ValueError("query output/report already exists; choose a new output path")
    pool = read_observations(args.pool)
    excluded = []
    excluded_sources = []
    for path in args.exclude:
        excluded.extend(read_observations(path))
        excluded_sources.append(
            {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        )
    groups = _eligible_groups(pool, excluded)
    if not groups:
        raise ValueError("no independent query groups remain after exclusions")
    scores, model_report = (None, {})
    if args.strategy != "random":
        scores, model_report = _candidate_scores(groups, args)
    choices = select_records(
        pool,
        excluded=excluded,
        budget=args.budget,
        strategy=args.strategy,
        seed=args.seed,
        scores=scores,
    )
    pool_hash = hashlib.sha256(Path(args.pool).read_bytes()).hexdigest()
    score_name = {
        "random": "seeded_random_group_priority",
        "entropy": "next_token_entropy_nats",
        "disagreement": "next_token_jensen_shannon_nats",
    }[args.strategy]
    selected = []
    for choice in choices:
        metadata = {
            key: value
            for key, value in choice.record.metadata.items()
            if key not in _COLLECTOR_METADATA
        }
        metadata["query_selection"] = {
            "strategy": args.strategy,
            "score_name": score_name,
            "score": choice.score,
            "seed": args.seed,
            "pool_sha256": pool_hash,
        }
        selected.append(
            Observation(
                id=choice.record.id,
                prompt=choice.record.prompt,
                output="",
                metadata=metadata,
            )
        )
    report = {
        "schema_version": 1,
        "strategy": args.strategy,
        "score_name": score_name,
        "seed": args.seed,
        "budget": args.budget,
        "selected_prompts": len(selected),
        "eligible_prompts": sum(map(len, groups)),
        "eligible_groups": len(groups),
        "excluded_pool_prompts": len(pool) - sum(map(len, groups)),
        "pool": str(args.pool),
        "pool_sha256": pool_hash,
        "excluded_sources": excluded_sources,
        "max_seq_length": args.max_seq_length,
        "group_policy": "one prompt per connected id/normalized-prompt/group_id component",
        "selection": [
            {"id": choice.record.id, "score": choice.score, "group_size": choice.group_size}
            for choice in choices
        ],
        "candidate_scores": scores,
        "output": str(output),
        "report": str(report_path),
        "interpretation": "Candidate uncertainty/disagreement is a query heuristic, not recovery accuracy.",
        **model_report,
    }
    write_observations(output, selected)
    write_json(report_path, report)
    return report


def add_subcommands(subparsers):
    parser = subparsers.add_parser(
        "select-queries", help="Choose new prompts using candidate uncertainty or disagreement"
    )
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument(
        "--strategy", choices=["random", "entropy", "disagreement"], default="random"
    )
    parser.add_argument(
        "--model", help="Candidate surrogate; never consulted for the random strategy"
    )
    parser.add_argument("--reference-candidate", help="Second candidate surrogate for disagreement")
    parser.add_argument("--revision")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Reserved or previously queried evidence; repeatable",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.set_defaults(func=select_queries)
