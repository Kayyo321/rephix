"""Create Mode orchestration: durable run directory, phases, resume checks and reports.

Phases: prepare -> hypotheses -> structure -> identify -> fit -> evaluate -> export.
Each completed phase writes its result atomically and records the result's SHA-256 in
``state.json``. Resume verifies the base content fingerprint, the evidence inputs and
the frozen configuration, re-verifies completed phase outputs and continues from the
first incomplete phase (the fit phase continues from its own checkpoint).
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..io import atomic_write_text, file_sha256, write_json
from . import constraints, evaluate, identify, structure
from . import records as rec
from . import splits as sp
from .base import available_memory, load_base, resolve_contract
from .data import NEUTRAL_PROMPTS
from .modeling import (
    generate,
    kl_to_reference,
    lora_frames,
    random_frames,
    region_shapes,
)
from .serialize import Example, Serializer

RUN_SCHEMA = "rephix-create-run/1"
PHASES = ("prepare", "hypotheses", "structure", "identify", "fit", "evaluate", "export")
METHODS = ("structured", "random-regions", "random-subspace", "lora", "lora-matched", "sft")
BEHAVIORAL_ONLY = ("lora", "lora-matched", "sft")

PROFILES = {
    "cpu-smoke": {
        "device": "cpu",
        "autocast": None,
        "max_tokens": 96,
        "context_tokens": 24,
        "steps": 12,
        "batch_size": 2,
        "retain_batch_size": 2,
        "rank": 4,
        "regions": 3,
        "screen": 3,
        "identify_top": 2,
        "identify_steps": 6,
        "eval_every": 6,
        "save_every": 3,
        "stats_examples": 8,
        "retain_new_tokens": 24,
        "base_new_tokens": 24,
        "generation_count": 3,
        "generation_new_tokens": 16,
        "hypothesis_records": 16,
    },
    "rtx5080-16gb": {
        "device": "cuda",
        "autocast": "bfloat16",
        "max_tokens": 384,
        "context_tokens": 96,
        "steps": 400,
        "batch_size": 8,
        "retain_batch_size": 4,
        "rank": 16,
        "regions": 12,
        "screen": 12,
        "identify_top": 3,
        "identify_steps": 150,
        "eval_every": 50,
        "save_every": 25,
        "stats_examples": 192,
        "retain_new_tokens": 96,
        "base_new_tokens": 96,
        "generation_count": 24,
        "generation_new_tokens": 96,
        "hypothesis_records": 96,
    },
}
METHOD_LR = {
    "structured": 2e-3,
    "random-regions": 2e-3,
    "random-subspace": 2e-3,
    "lora": 5e-4,
    "lora-matched": 5e-4,
    "sft": 2e-5,
}


@dataclass
class Context:
    torch: object
    model: object
    tokenizer: object
    device: str
    serializer: Serializer
    autocast: object
    records: list
    split: dict
    parts: dict
    train_examples: list = field(default_factory=list)
    validation_examples: list = field(default_factory=list)
    test_examples: list = field(default_factory=list)
    retain_examples: list = field(default_factory=list)
    retain_eval_examples: list = field(default_factory=list)
    instruction_ids: list | None = None
    base_model: object = None


class Progress:
    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.started = time.monotonic()
        self.stage = ""

    def __call__(self, stage, message):
        self.stage = stage
        index = PHASES.index(stage) + 1 if stage in PHASES else "-"
        memory = ""
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            memory = f" | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        print(
            f"[create {index}/{len(PHASES)} {stage}] {message} | elapsed "
            f"{time.monotonic() - self.started:.0f}s{memory}",
            file=self.stream,
            flush=True,
        )


# ----------------------------------------------------------------------------- locking


@contextlib.contextmanager
def run_lock(run_dir: Path):
    path = run_dir / ".lock"
    handle = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError(f"{run_dir} is in use by another Rephix process") from None
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError(f"{run_dir} is in use by another Rephix process") from None
        yield
    finally:
        handle.close()


# ----------------------------------------------------------------------------- config


def _versions():
    versions = {"rephix": __version__, "python": platform.python_version()}
    for name in ("torch", "transformers", "safetensors", "numpy"):
        module = sys.modules.get(name)
        if module is None:
            try:
                module = __import__(name)
            except ImportError:
                continue
        versions[name] = getattr(module, "__version__", None)
    return versions


def _input_fingerprints(paths):
    result = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix in rec.CORPUS_SUFFIXES
            )
            result.append(
                {
                    "path": str(path.resolve()),
                    "files": {str(f.relative_to(path)): file_sha256(f) for f in files},
                }
            )
        else:
            result.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    return result


def estimate_resources(contract, cfg) -> dict:
    """Coarse upper estimates. Measured peaks are recorded separately after the run."""
    params = sum(int(np.prod(t["shape"])) for t in contract["tensor_schema"])
    working = 4 if cfg["working_dtype"] == "float32" else 2
    hidden = contract["config"]["hidden_size"]
    layers = contract["config"]["num_hidden_layers"]
    inter = contract["config"].get("intermediate_size") or 4 * hidden
    vocab = contract["config"]["vocab_size"]
    tokens = cfg["batch_size"] * cfg["max_tokens"]
    weights = params * working
    # Activations kept for backward through frozen layers (bf16 autocast ~ 2 bytes).
    activations = tokens * layers * (14 * hidden + 3 * inter) * 2
    logits = tokens * vocab * 4 * 4  # logits, log-softmax, gradient, reference
    linear = sum(
        int(np.prod(t["shape"]))
        for t in contract["tensor_schema"]
        if ".layers." in t["name"] and t["name"].endswith("proj.weight")
    )
    if cfg["method"] == "sft":
        optimizer = params * 4 * 3
    else:
        optimizer = cfg["regions"] * cfg["rank"] * (hidden + inter) * 4 * 4
    statistics = linear * 4  # evidence gradients for every candidate region
    dense = lambda d: d * d * 4 if d <= 2048 else d * 4  # noqa: E731
    kfac = layers * (3 * dense(hidden) + dense(inter) + 3 * dense(hidden) + 2 * dense(inter))
    peak_device = (
        weights
        + activations
        + logits
        + optimizer
        + (statistics + kfac if cfg["method"] in ("structured", "random-subspace") else 0)
    )
    return {
        "parameters": params,
        "working_dtype": cfg["working_dtype"],
        "estimated_peak_device_bytes": int(peak_device * 1.2),
        "estimated_peak_host_bytes": int(params * 4 * 3 + 2 * 2**30),
        "components_bytes": {
            "weights": weights,
            "activations": activations,
            "logits": logits,
            "optimizer": optimizer,
            "structure_statistics": statistics + kfac,
        },
        "note": "Coarse estimate (20% margin). Measured peaks are reported after each phase.",
    }


def build_config(args, contract) -> dict:
    profile_name = args.profile
    if profile_name == "auto":
        import importlib.util

        cuda = False
        if importlib.util.find_spec("torch") is not None:
            import torch

            cuda = torch.cuda.is_available()
        profile_name = "rtx5080-16gb" if cuda else "cpu-smoke"
    profile = dict(PROFILES[profile_name])
    params = sum(int(np.prod(t["shape"])) for t in contract["tensor_schema"])
    device = args.device if args.device != "auto" else profile["device"]
    if args.device == "auto" and device == "cuda":
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    working = args.dtype or ("float32" if params <= 1_200_000_000 else "bfloat16")
    if contract["config"]["vocab_size"] > 100_000 and profile_name == "rtx5080-16gb":
        profile["batch_size"] = 4
    for key in ("steps", "rank", "regions", "batch_size", "max_tokens", "eval_every"):
        value = getattr(args, key, None)
        if value is not None:
            profile[key] = value
    method = args.method
    cfg = {
        **profile,
        "profile": profile_name,
        "device": device,
        "working_dtype": working,
        "method": method,
        "seed": args.seed,
        "lr": args.lr if args.lr is not None else METHOD_LR[method],
        "retain_weight": args.retain_weight,
        "prior_weight": args.prior_weight,
        "instruction_weight": args.instruction_weight if args.prompt else 0.0,
        "train_frames": not args.fixed_subspace,
        "epsilon": 0.02,
        "identify": method == "structured" and not args.no_identify,
        "generation_temperature": 0.8,
        "max_seconds": None,
    }
    if device == "cpu":
        cfg["autocast"] = None
    if device == "mps":
        cfg["autocast"] = None
    return cfg


def create_run(args) -> Path:
    run_dir = Path(args.output)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(
            f"{run_dir} already exists and is not empty; resume it with "
            f"`rephix create-resume {run_dir}` or choose a new --output"
        )
    if not args.evidence and not args.corpus:
        raise ValueError("supply --evidence (unknown-model observations) or --corpus (writing)")
    if args.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    contract = resolve_contract(args.base, args.base_revision)
    for path in [*(args.evidence or []), *(args.corpus or [])]:
        if Path(path).resolve().is_relative_to(Path(contract["path"])):
            raise ValueError("evidence must not live inside the base model directory")
    cfg = build_config(args, contract)
    run_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "schema": RUN_SCHEMA,
        "created_by": _versions(),
        "base": contract,
        "inputs": {
            "evidence": _input_fingerprints(args.evidence or []),
            "corpus": _input_fingerprints(args.corpus or []),
            "section_chars": args.section_chars,
        },
        "instruction": args.prompt,
        "config": cfg,
        "resources": estimate_resources(contract, cfg),
        "export_path": str(Path(args.export).resolve())
        if args.export
        else str((run_dir / "export").resolve()),
    }
    write_json(run_dir / "run.json", run)
    write_json(run_dir / "state.json", {"phases": {}, "status": "created"})
    return run_dir


def load_run(run_dir: Path) -> dict:
    path = run_dir / "run.json"
    if not path.is_file():
        raise ValueError(f"{run_dir} is not a Create Mode run (no run.json)")
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("schema") != RUN_SCHEMA:
        raise ValueError(f"{run_dir}: unsupported run schema {run.get('schema')!r}")
    return run


def verify_inputs(run) -> None:
    for kind in ("evidence", "corpus"):
        for entry in run.get("inputs", {}).get(kind, []):
            path = Path(entry["path"])
            if not path.exists():
                raise ValueError(f"{kind} input {path} is missing; resume needs the same inputs")
            current = _input_fingerprints([path])[0]
            if current != entry:
                raise ValueError(
                    f"{kind} input {path} changed since the run started; refusing to resume "
                    "with different evidence (start a new run)"
                )


def plan_summary(run) -> dict:
    cfg = run["config"]
    method = cfg["method"]
    claim_level = (
        "behavioral reconstruction and mechanistic synthesis in the base; partial source "
        "recovery only if the region-class test explains the change; full source recovery "
        "is never claimed"
        if method == "structured"
        else "behavioral fitting baseline only; no structural or source-recovery claim"
        if method in BEHAVIORAL_ONLY
        else "ablation of the structural stage; behavioral comparison only"
    )
    resources = run["resources"]
    memory = available_memory()
    return {
        "base": {
            "path": run["base"]["path"],
            "origin": run["base"]["origin"],
            "architecture": run["base"]["architecture"],
            "content_sha256": run["base"]["content_sha256"],
            "support_evidence": run["base"]["support_evidence"],
        },
        "evidence_inputs": [e["path"] for e in run["inputs"]["evidence"]],
        "corpus_inputs": [e["path"] for e in run["inputs"]["corpus"]],
        "instruction": run["instruction"],
        "method": method,
        "profile": cfg["profile"],
        "device": cfg["device"],
        "working_dtype": cfg["working_dtype"],
        "steps": cfg["steps"],
        "regions": cfg["regions"],
        "rank": cfg["rank"],
        "claim_level": claim_level,
        "estimated_peak_device_gib": round(resources["estimated_peak_device_bytes"] / 2**30, 2),
        "estimated_peak_host_gib": round(resources["estimated_peak_host_bytes"] / 2**30, 2),
        "system_ram_available_gib": round((memory["system_ram_available_bytes"] or 0) / 2**30, 2),
        "export_path": run["export_path"],
    }


# ----------------------------------------------------------------------------- context


def _records(run, run_dir):
    records = []
    for entry in run["inputs"]["evidence"]:
        records.extend(rec.read_evidence(entry["path"]))
    corpus = [entry["path"] for entry in run["inputs"]["corpus"]]
    if corpus:
        records.extend(rec.read_corpus(corpus, section_chars=run["inputs"]["section_chars"]))
    rec.validate_unique(records)
    return records


def _autocast(torch, cfg, device):
    if cfg.get("autocast") and device == "cuda":
        dtype = getattr(torch, cfg["autocast"])
        return lambda: torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext


def build_context(run, run_dir, progress):
    from ..hf import _deps, _device

    torch = _deps()[0]
    cfg = run["config"]
    device = _device(torch, cfg["device"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model, tokenizer = load_base(run["base"], device=device, dtype=cfg["working_dtype"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    serializer = Serializer(
        tokenizer, max_tokens=cfg["max_tokens"], context_tokens=cfg["context_tokens"]
    )
    records = _records(run, run_dir)
    split_file = run_dir / "evidence" / "splits.json"
    if split_file.is_file():
        split = json.loads(split_file.read_text(encoding="utf-8"))
        if split["fingerprint"] != rec.evidence_fingerprint(records):
            raise ValueError("evidence changed after the splits were frozen; refusing to resume")
    else:
        split = sp.assign_splits(records, seed=cfg["seed"])
        split["fingerprint"] = rec.evidence_fingerprint(records)
    parts = sp.partition(records, split)
    ctx = Context(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        device=device,
        serializer=serializer,
        autocast=_autocast(torch, cfg, device),
        records=records,
        split=split,
        parts=parts,
    )
    ctx.train_examples = serializer.encode_all(parts["train"])
    ctx.validation_examples = serializer.encode_all(parts["validation"])
    ctx.test_examples = serializer.encode_all(parts["test"])
    if run["instruction"]:
        ctx.instruction_ids = tokenizer(run["instruction"] + "\n\n", add_special_tokens=False)[
            "input_ids"
        ]
    retain_file = run_dir / "evidence" / "retain.json"
    if retain_file.is_file():
        retain = json.loads(retain_file.read_text(encoding="utf-8"))
        ctx.retain_examples = [
            Example(f"retain{i}", ids, [1] * len(ids), "retain")
            for i, ids in enumerate(retain["train"])
        ]
        ctx.retain_eval_examples = [
            Example(f"retain-eval{i}", ids, [1] * len(ids), "retain")
            for i, ids in enumerate(retain["evaluation"])
        ]
    return ctx


# ----------------------------------------------------------------------------- phases


def phase_prepare(ctx, run, run_dir, progress):
    cfg = run["config"]
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    records = ctx.records
    manifest = rec.capability_manifest(records)
    write_json(evidence_dir / "splits.json", ctx.split)
    payload = "".join(
        json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for r in records
    )
    atomic_write_text(evidence_dir / "records.jsonl", payload)
    progress(
        "prepare",
        f"{len(records)} records, splits {ctx.split['counts']}, groups {ctx.split['group_counts']}",
    )
    # Retain contexts: the base's own sampled continuations of neutral prompts.
    prefixes = [ctx.serializer.prompt_text_ids(p) for p in NEUTRAL_PROMPTS]
    texts = generate(
        ctx.torch,
        ctx.model,
        ctx.tokenizer,
        prefixes,
        max_new_tokens=cfg["retain_new_tokens"],
        device=ctx.device,
        seed=cfg["seed"] + 101,
        temperature=1.0,
    )
    sequences = []
    for prefix, text in zip(prefixes, texts):
        ids = prefix + ctx.tokenizer(text, add_special_tokens=False)["input_ids"]
        sequences.append(ids[: cfg["max_tokens"]])
    held = max(4, len(sequences) // 4)
    retain = {
        "train": sequences[:-held],
        "evaluation": sequences[-held:],
        "source": "base sampled continuations (temperature 1.0) of fixed neutral prompts",
    }
    write_json(evidence_dir / "retain.json", retain)
    ctx.retain_examples = [
        Example(f"retain{i}", ids, [1] * len(ids), "retain")
        for i, ids in enumerate(retain["train"])
    ]
    ctx.retain_eval_examples = [
        Example(f"retain-eval{i}", ids, [1] * len(ids), "retain")
        for i, ids in enumerate(retain["evaluation"])
    ]
    # Base continuations of the same contexts, for hypothesis tests.
    chosen = {"train": [], "validation": []}
    limit = cfg["hypothesis_records"]
    for name in chosen:
        for record in ctx.parts[name]:
            prefix = _hypothesis_prefix(ctx, record)
            if prefix:
                chosen[name].append((record, prefix))
            if len(chosen[name]) >= (limit if name == "train" else max(8, limit // 2)):
                break
    base_texts = {}
    for name, items in chosen.items():
        outputs = generate(
            ctx.torch,
            ctx.model,
            ctx.tokenizer,
            [p for _, p in items],
            max_new_tokens=cfg["base_new_tokens"],
            device=ctx.device,
            seed=cfg["seed"] + 202,
            temperature=cfg["generation_temperature"],
        )
        for (record, _), text in zip(items, outputs):
            base_texts[record.id] = text
        progress("prepare", f"base continuations for {name}: {len(items)}")
    write_json(evidence_dir / "base_continuations.json", base_texts)
    result = {
        "capabilities": manifest,
        "serialization": ctx.serializer.manifest(),
        "examples": {
            "train": len(ctx.train_examples),
            "validation": len(ctx.validation_examples),
            "test": len(ctx.test_examples),
            "train_target_tokens": sum(e.target_tokens for e in ctx.train_examples),
            "validation_target_tokens": sum(e.target_tokens for e in ctx.validation_examples),
            "test_target_tokens": sum(e.target_tokens for e in ctx.test_examples),
            "truncated": sum(
                e.truncated
                for e in ctx.train_examples + ctx.validation_examples + ctx.test_examples
            ),
        },
        "splits": {k: v for k, v in ctx.split.items() if k not in ("assignment", "group_of")},
        "retain_contexts": {"train": len(retain["train"]), "evaluation": len(retain["evaluation"])},
        "frozen_before_fitting": True,
    }
    if not ctx.train_examples or not ctx.validation_examples or not ctx.test_examples:
        raise ValueError("every split needs at least one example after serialization")
    return result


def _hypothesis_prefix(ctx, record):
    if record.kind == "exchange":
        return ctx.serializer.generation_prefix(record)
    head = ctx.tokenizer(record.text, add_special_tokens=False)["input_ids"][:16]
    prefix = ctx.serializer.generation_prefix(record) or []
    return prefix + head


def phase_hypotheses(ctx, run, run_dir, progress):
    base_texts = json.loads((run_dir / "evidence" / "base_continuations.json").read_text())
    words_budget = run["config"]["base_new_tokens"]
    result = {}
    diffs = {}
    for name in ("train", "validation"):
        evidence_rows, base_rows, groups = [], [], []
        for record in ctx.parts[name]:
            if record.id not in base_texts:
                continue
            target = record.target_text()
            if record.kind == "passage":
                head = ctx.tokenizer(record.text, add_special_tokens=False)["input_ids"]
                target = ctx.tokenizer.decode(head[16 : 16 + words_budget])
            tool_calls = sum(len(m.tool_calls or []) for m in record.assistant_messages())
            reasoning = any(m.reasoning for m in record.assistant_messages())
            if record.kind == "exchange":
                from .serialize import render_assistant

                target = render_assistant(record.assistant_messages()[-1])
            evidence_rows.append(
                constraints.features(target, tool_calls=tool_calls, reasoning=reasoning)
            )
            from .serialize import parse_tool_calls

            base_text = base_texts[record.id]
            calls, _ = parse_tool_calls(base_text)
            base_rows.append(
                constraints.features(
                    base_text, tool_calls=len(calls), reasoning="<think>" in base_text
                )
            )
            groups.append(ctx.split["group_of"][record.id])
        diffs[name] = (
            constraints.standardized_differences(
                evidence_rows, base_rows, groups, seed=run["config"]["seed"]
            ),
            len(set(groups)),
        )
    tested = constraints.hypotheses(
        diffs["train"][0],
        diffs["validation"][0],
        groups_train=diffs["train"][1],
        groups_validation=diffs["validation"][1],
    )
    result["style_and_behavior"] = tested
    evidence_texts = [r.target_text() for r in ctx.parts["train"]]
    base_list = [base_texts[r.id] for r in ctx.parts["train"] if r.id in base_texts]
    result["content_terms"] = {
        "terms": constraints.content_terms(evidence_texts, base_list),
        "note": "Distinctive topic words: content, not style. Reported to separate topic from "
        "voice; not used as a style hypothesis.",
    }
    result["instruction"] = constraints.map_instruction(run["instruction"])
    supported = [h["feature"] for h in tested["hypotheses"] if h["status"] == "supported"]
    progress("hypotheses", f"{len(supported)} supported feature hypotheses: {supported[:8]}")
    return result


def _select_for_method(ctx, cfg, scores, contract):
    method = cfg["method"]
    rng = np.random.default_rng(cfg["seed"] + 7)
    all_regions = list(contract["regions"])
    if method in ("structured", "random-subspace"):
        regions = structure.select_regions(scores, cfg["regions"])
    elif method in ("random-regions", "lora-matched"):
        shuffled = list(rng.permutation(all_regions))
        regions, per_layer = [], {}
        for region in shuffled:
            layer = region.split(".")[0]
            if per_layer.get(layer, 0) >= 2:
                continue
            regions.append(region)
            per_layer[layer] = per_layer.get(layer, 0) + 1
            if len(regions) == cfg["regions"]:
                break
    elif method == "lora":
        regions = all_regions
    else:
        regions = []
    return regions


def phase_structure(ctx, run, run_dir, progress):
    torch = ctx.torch
    cfg = run["config"]
    contract = run["base"]
    regions = list(contract["regions"])
    method = cfg["method"]
    result = {"method": method}
    frames_out = {}
    scores = None
    if method in ("structured", "random-subspace"):
        stats_examples = _sample(ctx.train_examples, cfg["stats_examples"], cfg["seed"])
        evidence_contexts = _sample(
            ctx.train_examples, max(8, len(ctx.retain_examples)), cfg["seed"] + 5
        )
        sites, outputs = structure.collect_curvature(
            torch,
            ctx.model,
            regions,
            ctx.retain_examples + evidence_contexts,
            ctx.tokenizer,
            ctx.device,
            seed=cfg["seed"],
            progress=lambda i, n: (
                progress("structure", f"curvature {i}/{n} contexts") if i == n else None
            ),
        )
        gradients = structure.evidence_gradients(
            torch,
            ctx.model,
            regions,
            stats_examples,
            ctx.tokenizer,
            ctx.device,
            progress=lambda i, n: (
                progress("structure", f"evidence gradients {i}/{n} examples") if i == n else None
            ),
        )
        scores = structure.score_regions(
            torch, regions, gradients, sites, outputs, rank=cfg["rank"], damping=0.1
        )
        del gradients, sites, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        selected = _select_for_method(ctx, cfg, scores, contract)
        ranked = sorted(scores, key=lambda r: scores[r]["score"], reverse=True)
        screen = ranked[: cfg["screen"]]
        rng = np.random.default_rng(cfg["seed"] + 11)
        others = [r for r in ranked[cfg["screen"] :]]
        random_controls = (
            list(rng.choice(others, size=min(cfg["screen"], len(others)), replace=False))
            if others
            else []
        )

        def edits_kl(examples):
            return _kl_hooks(ctx, examples)

        trial = structure.trial_edits(
            torch,
            ctx.model,
            ctx.tokenizer,
            ctx.device,
            scores,
            screen + random_controls,
            ctx.validation_examples,
            ctx.retain_eval_examples + evidence_contexts[:8],
            epsilon=cfg["epsilon"],
            seed=cfg["seed"],
            edits_kl=edits_kl,
            progress=lambda i, n: (
                progress("structure", f"trial edits {i}/{n}") if i == n or i % 6 == 0 else None
            ),
        )
        top_rows = [r for r in trial["rows"] if r["region"] in screen]
        random_rows = [r for r in trial["rows"] if r["region"] in random_controls]
        gains_top = [r["validation_gain"] for r in top_rows]
        gains_random = [r["validation_gain"] for r in random_rows]
        summary = {
            "screened_top_regions": screen,
            "random_control_regions": random_controls,
            "mean_validation_gain_top": float(np.mean(gains_top)) if gains_top else None,
            "mean_validation_gain_random_regions": float(np.mean(gains_random))
            if gains_random
            else None,
            "mean_validation_gain_random_direction_in_top": float(
                np.mean([r["random_direction_validation_gain"] for r in top_rows])
            )
            if top_rows
            else None,
            "spearman_score_vs_validation_gain": structure.spearman(
                [r["score"] for r in trial["rows"]], [r["validation_gain"] for r in trial["rows"]]
            ),
        }
        if gains_top and gains_random:
            # One-sided permutation test: are top-region gains larger than random-region gains?
            observed = np.mean(gains_top) - np.mean(gains_random)
            pooled = np.array(gains_top + gains_random)
            perm_rng = np.random.default_rng(cfg["seed"] + 13)
            exceed = 0
            for _ in range(5000):
                perm = perm_rng.permutation(pooled)
                exceed += (
                    perm[: len(gains_top)].mean() - perm[len(gains_top) :].mean()
                ) >= observed
            summary["permutation_p_value"] = float((exceed + 1) / 5001)
            summary["selected_beats_random"] = bool(
                observed > 0 and summary["permutation_p_value"] < 0.05
            )
        result.update(
            {
                "scores": structure.public_scores(scores),
                "ranked": ranked,
                "selected_regions": selected,
                "trial_edits": trial,
                "trial_summary": summary,
                "feature_descriptions": structure.describe_directions(
                    torch, ctx.model, ctx.tokenizer, scores, selected
                ),
                "curvature": "Kronecker-factored model Fisher with labels sampled from the "
                "base, on retain plus training-evidence contexts; dense factors up to 2048 "
                "features, diagonal beyond",
            }
        )
        progress(
            "structure",
            f"selected {selected}; trial edits: top {summary['mean_validation_gain_top']}"
            f" vs random {summary['mean_validation_gain_random_regions']}",
        )
        if method == "structured":
            for region in set(selected) | set(screen[: cfg["identify_top"]]):
                P, Q = scores[region]["_frame"]
                frames_out[region] = (P.cpu().numpy(), Q.cpu().numpy())
        else:
            frames_out = random_frames(
                np.random.default_rng(cfg["seed"] + 17),
                selected,
                region_shapes(ctx.model, selected),
                cfg["rank"],
            )
    else:
        selected = _select_for_method(ctx, cfg, None, contract)
        result["selected_regions"] = selected if method != "lora" else "all"
        result["note"] = "no structural inference for this method"
        shapes = region_shapes(ctx.model, selected)
        rng = np.random.default_rng(cfg["seed"] + 17)
        if method == "random-regions":
            frames_out = random_frames(rng, selected, shapes, cfg["rank"])
        elif method in ("lora", "lora-matched"):
            rank = 8 if method == "lora" else cfg["rank"]
            frames_out = lora_frames(rng, selected, shapes, rank)
    np.savez_compressed(
        run_dir / "structure-frames.npz",
        **{f"{r}|P": f[0] for r, f in frames_out.items()},
        **{f"{r}|Q": f[1] for r, f in frames_out.items()},
    )
    result["frames_file_sha256"] = file_sha256(run_dir / "structure-frames.npz")
    return result


def _kl_hooks(ctx, examples):
    """KL(base || current hooked model) using the hook-free base as reference."""
    from .modeling import collate, pad_id

    torch = ctx.torch
    padding = pad_id(ctx.tokenizer)
    total, tokens = 0.0, 0
    model = ctx.model
    with torch.inference_mode():
        for start in range(0, len(examples), 4):
            ids, attention, _ = collate(torch, examples[start : start + 4], padding, ctx.device)
            current = torch.log_softmax(
                model(input_ids=ids, attention_mask=attention).logits.float(), -1
            )
            # Temporarily disable every hook registered on submodules (trial edits).
            saved = []
            for module in model.modules():
                if module._forward_hooks:
                    saved.append((module, dict(module._forward_hooks)))
                    module._forward_hooks.clear()
            try:
                reference = torch.log_softmax(
                    model(input_ids=ids, attention_mask=attention).logits.float(), -1
                )
            finally:
                for module, saved_hooks in saved:
                    module._forward_hooks.update(saved_hooks)
            kl = (reference.exp() * (reference - current)).sum(-1)
            mask = attention.bool()
            total += float(kl[mask].sum().item())
            tokens += int(mask.sum().item())
    return total / max(tokens, 1)


def _sample(examples, limit, seed):
    if len(examples) <= limit:
        return list(examples)
    order = np.random.default_rng(seed).permutation(len(examples))[:limit]
    return [examples[i] for i in sorted(order)]


def _frames(run_dir):
    data = np.load(run_dir / "structure-frames.npz")
    frames = {}
    for key in data.files:
        region, side = key.split("|")
        frames.setdefault(region, [None, None])[0 if side == "P" else 1] = data[key]
    return {r: (p, q) for r, (p, q) in frames.items()}


def _fit_spec(cfg, name, regions, *, seed, steps, method=None):
    method = method or cfg["method"]
    init = "lora" if method in ("lora", "lora-matched") else "zero_core"
    rank = 8 if method == "lora" else cfg["rank"]
    return {
        "name": name,
        "method": method,
        "regions": sorted(regions),
        "rank": rank,
        "init": init,
        "train_frames": cfg["train_frames"] if init == "zero_core" else True,
        "seed": seed,
        "steps": steps,
        "lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "retain_batch_size": cfg["retain_batch_size"],
        "retain_weight": cfg["retain_weight"],
        "prior_weight": cfg["prior_weight"],
        "instruction_weight": cfg["instruction_weight"],
        "eval_every": min(cfg["eval_every"], steps),
        "save_every": cfg["save_every"],
    }


def phase_identify(ctx, run, run_dir, progress, structure_result):
    from .fit import fit_edits

    cfg = run["config"]
    result = {}
    if cfg["identify"]:
        frames = _frames(run_dir)
        top = structure_result["ranked"][: cfg["identify_top"]]
        selected = structure_result["selected_regions"]

        def fit_one(name, regions, seed):
            spec = _fit_spec(
                cfg, name, regions, seed=seed, steps=cfg["identify_steps"], method="structured"
            )
            spec["instruction_weight"] = 0.0
            edits, report = fit_edits(
                ctx,
                spec,
                {r: frames[r] for r in regions},
                checkpoint_dir=run_dir / "identify" / name.replace(".", "_"),
            )
            progress("identify", f"{name}: validation NLL {report['best_validation_nll']:.4f}")
            return edits, report

        region_test, _ = identify.region_class_test(
            ctx, fit_one, top, selected, ctx.split["group_of"]
        )
        result["region_class_test"] = region_test
    else:
        result["region_class_test"] = {
            "status": "not_run",
            "reason": f"method {cfg['method']} or --no-identify",
        }
    result["declared_class_head"] = identify.declared_class_head(ctx, ctx.parts["train"])
    result["expand_and_cluster"] = {
        "status": "not_applicable",
        "reason": "Expand-and-Cluster needs real-valued outputs of a small numerical network "
        "on controlled inputs; text evidence does not satisfy its observation assumptions. "
        "Use `rephix expand-cluster` for numerical-output evidence.",
    }
    return result


def _fit_main(ctx, run, run_dir, progress, max_seconds=None):
    from .fit import fit_edits, fit_full

    cfg = run["config"]
    if cfg["method"] == "sft":
        spec = _fit_spec(cfg, "main", [], seed=cfg["seed"], steps=cfg["steps"])
        report = fit_full(
            ctx,
            spec,
            checkpoint_dir=run_dir / "fit" / "main",
            progress=lambda s, n, e: progress(
                "fit", f"step {s}/{n} validation NLL {e['validation_nll']:.4f}"
            ),
            max_seconds=max_seconds,
        )
        return None, report
    frames = _frames(run_dir)
    structure_result = json.loads((run_dir / "phases" / "structure.json").read_text())
    selected = structure_result["selected_regions"]
    if selected == "all":
        selected = list(run["base"]["regions"])
    spec = _fit_spec(cfg, "main", selected, seed=cfg["seed"], steps=cfg["steps"])
    edits, report = fit_edits(
        ctx,
        spec,
        {r: frames[r] for r in selected},
        checkpoint_dir=run_dir / "fit" / "main",
        progress=lambda s, n, e: progress(
            "fit",
            f"step {s}/{n} validation NLL {e['validation_nll']:.4f} "
            f"retain KL {e.get('retain_kl', float('nan')):.4f} checkpoint {run_dir / 'fit' / 'main'}",
        ),
        max_seconds=max_seconds,
    )
    return edits, report


def phase_evaluate(
    ctx,
    run,
    run_dir,
    progress,
    edits,
    fit_report,
    hypotheses_result,
    structure_result,
    identification,
):
    cfg = run["config"]
    group_of = ctx.split["group_of"]
    result = {
        "fit": {k: v for k, v in fit_report.items() if k != "history"},
        "fit_history": fit_report["history"],
    }
    if edits is None:
        # Full fine-tune: the base side of every comparison uses a fresh base load.
        ctx.base_model, _ = load_base(run["base"], device=ctx.device, dtype=cfg["working_dtype"])
    result["validation_nll"] = evaluate.nll_comparison(
        ctx, ctx.validation_examples, group_of, edits=edits, seed=1
    )
    progress(
        "evaluate",
        f"validation NLL base {result['validation_nll']['base_nll']:.4f} -> "
        f"created {result['validation_nll']['created_nll']:.4f}",
    )
    result["test_nll"] = evaluate.nll_comparison(
        ctx, ctx.test_examples, group_of, edits=edits, seed=2
    )
    result["test_nll"]["locked"] = "computed once, after model selection on validation"
    progress(
        "evaluate",
        f"test NLL base {result['test_nll']['base_nll']:.4f} -> created "
        f"{result['test_nll']['created_nll']:.4f}",
    )
    if edits is not None:
        result["restoration_validation"] = evaluate.restoration(
            ctx, edits, ctx.validation_examples, group_of
        )
        result["retain_kl_evaluation"] = kl_to_reference(
            ctx.torch, ctx.model, edits, ctx.retain_eval_examples, ctx.tokenizer, ctx.device
        )
    supported = evaluate.supported_style_features(hypotheses_result["style_and_behavior"])
    if not supported:
        supported = [
            f for f in constraints.FEATURES if f not in ("tool_call_count", "reasoning_present")
        ]
        result["style_features_used"] = "all (no supported hypotheses; exploratory)"
    else:
        result["style_features_used"] = supported
    generation, _, _ = evaluate.free_running(
        ctx,
        edits,
        ctx.parts["test"],
        train_records=ctx.parts["train"],
        supported_features=supported,
        count=cfg["generation_count"],
        max_new_tokens=cfg["generation_new_tokens"],
        temperature=cfg["generation_temperature"],
        seed=cfg["seed"] + 303,
    )
    result["generation"] = generation
    progress("evaluate", "free-running generation measured")
    result["fact_cloze"] = evaluate.fact_cloze(ctx, edits)
    result["agent"] = evaluate.agent_metrics(
        ctx,
        edits,
        ctx.parts["test"],
        ctx.parts["train"],
        max_new_tokens=cfg["generation_new_tokens"],
    )
    result["instruction"] = evaluate.instruction_following(
        hypotheses_result.get("instruction"), generation
    )
    result["claims"] = evaluate.select_claims(
        method=cfg["method"],
        evaluation=result,
        structure=structure_result,
        identification=identification,
    )
    return result


def _write_phase(run_dir, name, result):
    write_json(run_dir / "phases" / f"{name}.json", result)
    state = json.loads((run_dir / "state.json").read_text())
    state["phases"][name] = {
        "status": "complete",
        "output_sha256": file_sha256(run_dir / "phases" / f"{name}.json"),
        "completed_at": time.time(),
    }
    write_json(run_dir / "state.json", state)


def _phase_done(run_dir, name):
    state = json.loads((run_dir / "state.json").read_text())
    entry = state["phases"].get(name)
    path = run_dir / "phases" / f"{name}.json"
    if not entry:
        return False
    if not path.is_file() or file_sha256(path) != entry["output_sha256"]:
        raise ValueError(
            f"{path}: completed phase output is missing or was modified; refusing to resume on "
            "corrupted state"
        )
    return True


def _load_phase(run_dir, name):
    return json.loads((run_dir / "phases" / f"{name}.json").read_text())


def execute(run_dir: Path, *, max_seconds=None, stop_after=None, progress=None, export_to=None):
    run_dir = Path(run_dir)
    progress = progress or Progress()
    run = load_run(run_dir)
    verify_inputs(run)
    (run_dir / "phases").mkdir(exist_ok=True)
    started = time.monotonic()
    with run_lock(run_dir):
        state = json.loads((run_dir / "state.json").read_text())
        state["status"] = "running"
        write_json(run_dir / "state.json", state)
        progress("prepare", "loading base and evidence")
        ctx = build_context(run, run_dir, progress)
        torch = ctx.torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        timings = state.get("timings", {})

        def finish(name, result, t0):
            timings[name] = round(time.monotonic() - t0, 2)
            _write_phase(run_dir, name, result)
            s = json.loads((run_dir / "state.json").read_text())
            s["timings"] = timings
            write_json(run_dir / "state.json", s)

        def stop(name):
            return stop_after == name

        def budget():
            if max_seconds is None:
                return None
            return max(1.0, max_seconds - (time.monotonic() - started))

        for name in ("prepare", "hypotheses", "structure"):
            if not _phase_done(run_dir, name):
                t0 = time.monotonic()
                progress(name, "starting")
                function = {
                    "prepare": phase_prepare,
                    "hypotheses": phase_hypotheses,
                    "structure": phase_structure,
                }[name]
                finish(name, function(ctx, run, run_dir, progress), t0)
            if stop(name):
                return _paused(run_dir, f"stopped after {name} as requested")
        structure_result = _load_phase(run_dir, "structure")
        if not _phase_done(run_dir, "identify"):
            t0 = time.monotonic()
            progress("identify", "starting")
            finish("identify", phase_identify(ctx, run, run_dir, progress, structure_result), t0)
        if stop("identify"):
            return _paused(run_dir, "stopped after identify as requested")
        identification = _load_phase(run_dir, "identify")
        from .fit import Budget

        t0 = time.monotonic()
        progress("fit", f"method {run['config']['method']}, {run['config']['steps']} total steps")
        try:
            edits, fit_report = _fit_main(ctx, run, run_dir, progress, max_seconds=budget())
        except Budget:
            return _paused(run_dir, "time budget reached during fit; resume to continue")
        if not _phase_done(run_dir, "fit"):
            finish("fit", {k: v for k, v in fit_report.items()}, t0)
        if stop("fit"):
            return _paused(run_dir, "stopped after fit as requested")
        hypotheses_result = _load_phase(run_dir, "hypotheses")
        if not _phase_done(run_dir, "evaluate"):
            t0 = time.monotonic()
            progress("evaluate", "starting (validation, then the locked test split)")
            evaluation = phase_evaluate(
                ctx,
                run,
                run_dir,
                progress,
                edits,
                fit_report,
                hypotheses_result,
                structure_result,
                identification,
            )
            evaluation["resources_measured"] = _measured(torch)
            finish("evaluate", evaluation, t0)
        evaluation = _load_phase(run_dir, "evaluate")
        results = assemble_results(run, run_dir)
        report = render_report(results)
        write_json(run_dir / "results.json", results)
        atomic_write_text(run_dir / "REPORT.md", report)
        if stop("evaluate"):
            return _paused(run_dir, "stopped after evaluate as requested")
        if not _phase_done(run_dir, "export"):
            t0 = time.monotonic()
            from .export import export

            destination = Path(export_to or run["export_path"])
            progress("export", f"merging and verifying into {destination}")
            provenance = provenance_record(run, run_dir)
            verification = export(
                ctx,
                edits,
                run["base"],
                destination,
                provenance=provenance,
                report_md=report,
                results=results,
                device=ctx.device,
                full_model=edits is None,
            )
            finish("export", {"path": str(destination), "verification": verification}, t0)
            results = assemble_results(run, run_dir)
            write_json(run_dir / "results.json", results)
            atomic_write_text(run_dir / "REPORT.md", render_report(results))
        state = json.loads((run_dir / "state.json").read_text())
        state["status"] = "complete"
        write_json(run_dir / "state.json", state)
        results = assemble_results(run, run_dir)
        write_json(run_dir / "results.json", results)
        atomic_write_text(run_dir / "REPORT.md", render_report(results))
        progress("export", "complete")
        return {
            "status": "complete",
            "run": str(run_dir),
            "export": _load_phase(run_dir, "export")["path"],
            "claims": evaluation["claims"],
        }


def _paused(run_dir, reason):
    state = json.loads((run_dir / "state.json").read_text())
    state["status"] = "paused"
    state["reason"] = reason
    write_json(run_dir / "state.json", state)
    return {
        "status": "paused",
        "reason": reason,
        "run": str(run_dir),
        "resume": f"rephix create-resume {run_dir}",
    }


def _measured(torch):
    measured = {"peak_host_rss_bytes": None, "peak_device_bytes": None}
    try:
        import resource  # POSIX only

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        measured["peak_host_rss_bytes"] = int(rss * (1 if sys.platform == "darwin" else 1024))
    except (ImportError, AttributeError, OSError):
        pass
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        measured["peak_device_bytes"] = int(torch.cuda.max_memory_allocated())
        measured["device_name"] = torch.cuda.get_device_name(0)
    return measured


def provenance_record(run, run_dir):
    return {
        "schema": "rephix-create-provenance/1",
        "rephix_version": __version__,
        "versions": _versions(),
        "base": {
            k: run["base"][k]
            for k in (
                "origin",
                "path",
                "model_type",
                "architecture",
                "content_sha256",
                "config_sha256",
                "tensor_schema_sha256",
                "tokenizer_files",
            )
        },
        "evidence_inputs": run["inputs"],
        "evidence_fingerprint": json.loads((run_dir / "evidence" / "splits.json").read_text())[
            "fingerprint"
        ],
        "instruction": run["instruction"],
        "instruction_use": "context distillation from the base during fitting only; the "
        "export contains no prompt and needs none",
        "config": run["config"],
        "transformation": "theta_created = theta_base + Delta, Delta confined to the "
        "listed regions, merged into standalone weights",
        "run_directory": str(run_dir.resolve()),
    }


def assemble_results(run, run_dir):
    state = json.loads((run_dir / "state.json").read_text())
    phases = {
        name: _load_phase(run_dir, name)
        for name in PHASES
        if (run_dir / "phases" / f"{name}.json").is_file()
    }
    evaluation = phases.get("evaluate", {})
    return {
        "schema": "rephix-create-results/1",
        "rephix_version": __version__,
        "run_status": state.get("status"),
        "plan": plan_summary(run),
        "claims": evaluation.get("claims"),
        "evidence": phases.get("prepare"),
        "hypotheses": phases.get("hypotheses"),
        "structure": _compact_structure(phases.get("structure")),
        "identification": phases.get("identify"),
        "evaluation": {k: v for k, v in evaluation.items() if k != "claims"},
        "export": phases.get("export"),
        "timings_seconds": state.get("timings"),
        "limitations": LIMITATIONS,
    }


def _compact_structure(result):
    if not result or "scores" not in result:
        return result
    compact = dict(result)
    ranked = result["ranked"]
    compact["scores_top"] = {r: result["scores"][r] for r in ranked[:24]}
    compact.pop("scores")
    return compact


LIMITATIONS = [
    "Create Mode synthesizes behavior inside the selected base. It does not recover the "
    "parameters of an unknown source model from text; fitting sampled text is sequence-level "
    "distillation, and the structural stage decides where and how the base is edited.",
    "Structural findings (region scores, subspaces, logit-lens readings, restoration effects) "
    "describe the constructed base model, not the inaccessible source's circuitry.",
    "A partial-recovery status, when present, is conditional on the declared hypothesis class "
    "and cannot detect function-preserving reparameterizations of the source.",
    "Style statistics are coarse and finite; agreement on held-out text does not mean the model "
    "writes like the source on every future input.",
]


def render_report(results) -> str:
    plan = results["plan"]
    claims = results.get("claims") or {}
    evaluation = results.get("evaluation") or {}
    lines = [
        "# Rephix Create Mode report",
        "",
        f"Rephix {results['rephix_version']} · run status: **{results['run_status']}** · "
        f"method: `{plan['method']}`",
        "",
        "## Scientific claim status",
        "",
        "| Claim | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for key, label in (
        ("behavioral_reconstruction", "Behavioral reconstruction"),
        ("mechanistic_synthesis_in_base", "Mechanistic synthesis in the base"),
        ("partial_source_parameter_recovery", "Partial source-parameter recovery"),
        ("full_source_parameter_recovery", "Full source-parameter recovery"),
    ):
        entry = claims.get(key) or {}
        lines.append(
            f"| {label} | `{entry.get('status')}` | {'; '.join(entry.get('evidence', []))} |"
        )
    lines += ["", "Process completion is not a scientific claim.", "", "## Inputs", ""]
    lines.append(
        f"* Base: `{plan['base']['architecture']}` at `{plan['base']['path']}` "
        f"(content {plan['base']['content_sha256'][:16]}…)"
    )
    lines.append(
        f"* Evidence: {plan['evidence_inputs'] or '—'}; corpus: {plan['corpus_inputs'] or '—'}"
    )
    lines.append(
        f"* Instruction: {plan['instruction']!r}" if plan["instruction"] else "* Instruction: none"
    )
    prepare = results.get("evidence") or {}
    if prepare:
        tiers = prepare["capabilities"]["tiers"]
        lines.append("* Capability tiers present: " + ", ".join(k for k, v in tiers.items() if v))
        lines.append(f"* Examples: {prepare['examples']}")
    lines += ["", "## Measurements", ""]
    for split in ("validation_nll", "test_nll"):
        entry = evaluation.get(split)
        if entry:
            imp = entry["improvement_nats_per_token"]
            lines.append(
                f"* {split.replace('_', ' ')}: base {entry['base_nll']:.4f} → created "
                f"{entry['created_nll']:.4f} nats/token; improvement {imp['mean']:.4f} "
                f"[{imp['low']:.4f}, {imp['high']:.4f}] over {imp['groups']} groups"
            )
    generation = evaluation.get("generation") or {}
    for label in ("base", "created"):
        entry = generation.get(label)
        if entry:
            sd = (entry.get("style_distance_held_out") or {}).get("distance")
            lines.append(
                f"* {label} generation: style distance {sd if sd is None else round(sd, 3)}, "
                f"base-model NLL {entry['base_model_nll_of_generation']}, 8-gram overlap with "
                f"training {entry['ngram8_overlap_with_training_mean']}, longest copied run "
                f"{entry['longest_copied_words_max']} words"
            )
    cloze = evaluation.get("fact_cloze")
    if cloze:
        lines.append(
            f"* Fact cloze accuracy: base {cloze['base_accuracy']:.2f} → created "
            f"{cloze['created_accuracy']:.2f} ({cloze['items']} items)"
        )
    if evaluation.get("retain_kl_evaluation") is not None:
        lines.append(
            f"* Retain KL(base‖created) on held-out neutral contexts: "
            f"{evaluation['retain_kl_evaluation']:.5f} nats/token"
        )
    structure_result = results.get("structure") or {}
    if structure_result.get("trial_summary"):
        lines += ["", "## Structural inference (base model)", ""]
        lines.append(f"* Selected regions: {structure_result['selected_regions']}")
        lines.append(f"* Trial-edit summary: {json.dumps(structure_result['trial_summary'])}")
    identification = results.get("identification") or {}
    if identification:
        lines += ["", "## Identification", ""]
        for key, value in identification.items():
            lines.append(f"* {key}: `{(value or {}).get('status')}`")
    export_info = results.get("export")
    if export_info:
        lines += [
            "",
            "## Export",
            "",
            f"* Path: `{export_info['path']}`",
            f"* Verification passed: {export_info['verification']['passed']}",
        ]
    lines += ["", "## Limitations", ""] + [f"* {item}" for item in LIMITATIONS]
    return "\n".join(lines) + "\n"


def inspect_run(run_dir, verify=False):
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    state = json.loads((run_dir / "state.json").read_text())
    result = {
        "run": str(run_dir),
        "status": state.get("status"),
        "reason": state.get("reason"),
        "phases": {
            name: (state["phases"].get(name) or {}).get("status", "pending") for name in PHASES
        },
        "timings_seconds": state.get("timings"),
    }
    try:
        result["plan"] = plan_summary(run)
    except (KeyError, TypeError):
        result["plan"] = None
        result["plan_error"] = "run.json lacks the resolved base or configuration"
    checkpoint = run_dir / "fit" / "main" / "checkpoint.pt"
    result["fit_checkpoint"] = str(checkpoint) if checkpoint.is_file() else None
    if verify:
        problems = []
        for name in PHASES:
            try:
                _phase_done(run_dir, name)
            except ValueError as exc:
                problems.append(str(exc))
        try:
            verify_inputs(run)
        except ValueError as exc:
            problems.append(str(exc))
        result["verification"] = {"passed": not problems, "problems": problems}
    if (run_dir / "results.json").is_file():
        result["claims"] = json.loads((run_dir / "results.json").read_text()).get("claims")
    return result


def remove_export_if_incomplete(path):
    path = Path(path)
    if path.exists() and not (path / "EXPORT_COMPLETE").is_file():
        shutil.rmtree(path)
