"""Local, untrained Qwen fixtures and a reproducible offline integration trial.

The fixture is deliberately synthetic. Its measurements test the experiment
machinery and must not be interpreted as pretrained Qwen reconstruction results.
"""

from __future__ import annotations

import json
import time
from argparse import Namespace
from pathlib import Path

from .evidence import Observation, read_observations, split_observations, write_observations
from .io import atomic_write_text, file_sha256, write_json

_COLORS = ("red", "blue", "green", "yellow", "orange", "purple", "white", "black")
_OBJECTS = ("cat", "dog", "bird", "tree", "river", "stone", "sun", "moon")
_VERBS = ("find", "count", "name", "compare", "describe", "repeat", "choose", "reverse")
_ADJECTIVES = ("small", "large", "bright", "dark", "warm", "cold", "quick", "quiet")
_NUMBERS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "and")


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _fresh_directory(path):
    directory = Path(path).resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("output directory must be absent or empty")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _prompt_records(count=96):
    records = []
    for index in range(count):
        prompt = " ".join(
            (
                _VERBS[index % 8],
                _ADJECTIVES[(index // 8) % 8],
                _COLORS[(index // 64) % 8],
                _OBJECTS[(index * 5 + index // 8) % 8],
                str((index * 7 + index // 16) % 16),
            )
        )
        records.append(
            Observation(
                id=f"tiny-{index:03d}",
                prompt=prompt,
                output="",
                metadata={
                    "group_id": f"tiny-group-{index:03d}",
                    "source": "synthetic_untrained_qwen_fixture",
                },
            )
        )
    return records


def init_tiny(args):
    """Create a deterministic Qwen2 model and tokenizer with no network access."""
    hidden_size = args.hidden_size
    layers = args.layers
    _positive_integer(hidden_size, "hidden-size")
    _positive_integer(layers, "layers")
    if hidden_size % 8:
        raise ValueError("hidden-size must be divisible by eight for the fixture's attention heads")
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    from .hf import _deps, _seed, _tokenizer_fingerprint

    torch, _, _, _ = _deps()
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    output = _fresh_directory(args.output)
    model_path = output / "model"
    _seed(torch, args.seed)
    words = (
        ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "user", "assistant", ":"]
        + list(_COLORS)
        + list(_OBJECTS)
        + list(_VERBS)
        + list(_ADJECTIVES)
        + list(_NUMBERS)
        + [str(number) for number in range(16)]
    )
    vocabulary = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] + ' : ' + message['content'] + ' ' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant : ' }}{% endif %}"
    )
    config = Qwen2Config(
        vocab_size=len(vocabulary),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(model_path, safe_serialization=True)
    tokenizer.save_pretrained(model_path)
    records = _prompt_records()
    write_observations(output / "prompts.jsonl", records[:64])
    write_observations(output / "pool.jsonl", records)
    report = {
        "fixture": "locally_initialized_untrained_qwen2",
        "model": str(model_path),
        "seed": args.seed,
        "hidden_size": hidden_size,
        "layers": layers,
        "vocabulary_size": len(vocabulary),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "tokenizer_fingerprint": _tokenizer_fingerprint(tokenizer),
        "prompts": str(output / "prompts.jsonl"),
        "prompt_count": 64,
        "pool": str(output / "pool.jsonl"),
        "pool_count": len(records),
        "pool_policy": "64 fixture prompts plus 32 additional independent prompts",
        "network_access_required": False,
        "library_versions": {
            "torch": str(torch.__version__),
            "transformers": __import__("transformers").__version__,
        },
        "interpretation": "Synthetic untrained reference for pipeline checks; no pretrained Qwen weights.",
    }
    write_json(output / "fixture.json", report)
    return report


def offline_trial(args):
    """Run all local experiment tracks, choosing settings on validation only."""
    _positive_integer(args.steps, "steps")
    threads = getattr(args, "threads", 1)
    _positive_integer(threads, "threads")
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    teacher_seed = getattr(args, "teacher_seed", 7)
    if type(teacher_seed) is not int or teacher_seed < 0:
        raise ValueError("teacher-seed must be a nonnegative integer")
    from . import experiments, hf, projection, query
    from .alignment import symmetry_demo

    torch, _, _, _ = hf._deps()
    output = _fresh_directory(args.output)
    started = time.monotonic()
    original_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    stage = "initializing_fixture"

    def progress(status="running", **details):
        write_json(
            output / "trial-status.json",
            {"status": status, "stage": stage, "seed": args.seed, **details},
        )

    try:
        progress()
        fixture = init_tiny(
            Namespace(output=str(output / "fixture"), seed=teacher_seed, hidden_size=32, layers=2)
        )
        prompts = read_observations(fixture["prompts"])[:48]
        prompt_path = output / "collected-prompts.jsonl"
        evidence_path = output / "observations.jsonl"
        write_observations(prompt_path, prompts)
        stage = "collecting_reference_evidence"
        progress()
        collection = hf.collect(
            Namespace(
                prompts=str(prompt_path),
                model=fixture["model"],
                revision=None,
                output=str(evidence_path),
                max_new_tokens=4,
                top_k=8,
                seed=args.seed,
                device="cpu",
                resume=False,
            )
        )
        stage = "splitting_independent_prompt_groups"
        progress()
        splits = split_observations(read_observations(evidence_path), seed=args.seed)
        split_paths = {}
        split_manifest = {
            "seed": args.seed,
            "source_sha256": file_sha256(evidence_path),
            "splits": {},
        }
        for name, records in splits.items():
            path = output / "splits" / f"{name}.jsonl"
            write_observations(path, records)
            split_paths[name] = str(path)
            split_manifest["splits"][name] = {
                "path": str(path),
                "observations": len(records),
                "sha256": file_sha256(path),
            }
        write_json(output / "splits" / "manifest.json", split_manifest)
        stage = "fitting_equal_budget_candidates"
        progress()
        sweep_config = {
            "schema_version": 1,
            "teacher": fixture["model"],
            "train": split_paths["train"],
            "validation": split_paths["validation"],
            "device": "cpu",
            "defaults": {
                "steps": args.steps,
                "learning_rate": 0.003,
                "max_seq_length": 64,
                "eval_every": min(8, args.steps),
                "init": "random",
                "trainable": "all",
            },
            "grid": {"seed": [args.seed], "topk_weight": [0.0, 1.0]},
        }
        config_path = output / "sweep-config.json"
        sweep_path = output / "sweep"
        write_json(config_path, sweep_config)
        sweep_report = experiments.sweep(
            Namespace(config=str(config_path), output=str(sweep_path), resume=False)
        )
        comparison = experiments.create_report([str(sweep_path)], output / "comparison", plot=True)
        manifest = json.loads((sweep_path / "sweep.json").read_text(encoding="utf-8"))
        candidates = []
        for run in manifest["runs"]:
            path = sweep_path / run["id"]
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            candidates.append(
                (metrics["best_validation"]["continuation_cross_entropy"], run["id"], path, metrics)
            )
        score, run_id, winner, winner_metrics = min(candidates, key=lambda item: (item[0], item[1]))
        selection = {
            "selected_run": run_id,
            "model": str(winner),
            "selection_metric": "best_checkpoint_validation_continuation_cross_entropy",
            "selection_direction": "minimize",
            "selection_score": score,
            "topk_weight": winner_metrics["topk_weight"],
            "test_used_for_selection": False,
            "candidate_runs": len(candidates),
            "interpretation": "Common validation continuation loss chooses between the two saved checkpoints; test metrics are computed once afterwards.",
        }
        write_json(output / "selection.json", selection)
        stage = "evaluating_selected_candidate_on_untouched_test"
        progress()
        test_report = hf.evaluate(
            Namespace(
                data=split_paths["test"],
                model=str(winner),
                revision=None,
                output=str(output / "test.json"),
                max_seq_length=64,
                device="cpu",
                generate=True,
                max_new_tokens=4,
            )
        )
        stage = "recovering_observable_output_projection"
        progress()
        component = projection.run(
            Namespace(
                model=fixture["model"],
                revision=None,
                prompts=fixture["prompts"],
                output=str(output / "projection"),
                max_seq_length=64,
                max_samples=512,
                max_memory_mb=32,
                vocab_limit=32,
                token_ids=None,
                test_fraction=0.25,
                bias_mode="auto",
                ridge=0.0,
                seed=args.seed,
                device="cpu",
            )
        )
        stage = "selecting_new_queries_without_reference_labels"
        progress()
        query_report = query.select_queries(
            Namespace(
                pool=fixture["pool"],
                output=str(output / "next-queries.jsonl"),
                budget=8,
                strategy="entropy",
                model=str(winner),
                reference_candidate=None,
                revision=None,
                exclude=list(split_paths.values()),
                seed=args.seed,
                device="cpu",
                max_seq_length=64,
            )
        )
        stage = "demonstrating_supported_parameter_symmetries"
        progress()
        symmetry = symmetry_demo(seed=args.seed)
        write_json(output / "symmetry.json", symmetry)
        report = {
            "experiment": "offline_untrained_qwen_pipeline_trial",
            "status": "complete",
            "output": str(output),
            "seed": args.seed,
            "teacher_seed": teacher_seed,
            "steps_per_candidate": args.steps,
            "threads": threads,
            "network_access_required": False,
            "fixture": fixture,
            "collection": collection,
            "split_manifest": split_manifest,
            "sweep": sweep_report,
            "comparison": comparison,
            "selection": selection,
            "test_evaluations": 1,
            "selected_candidate_test": test_report,
            "projection": component,
            "query_selection": query_report,
            "symmetry_diagnostics": symmetry["alignment"]["diagnostics"],
            "elapsed_seconds": time.monotonic() - started,
            "interpretation": "Pipeline validation against a tiny untrained local teacher. Component recovery uses observed hidden activations. Neither result establishes pretrained Qwen reconstruction accuracy.",
        }
        write_json(output / "trial.json", report)
        summary = [
            "# Offline integration trial",
            "",
            "The reference is a locally initialized, untrained Qwen2 model with a 64-token vocabulary. These are pipeline checks, not pretrained Qwen recovery results.",
            "",
            f"- Candidate optimization budget: {args.steps} steps for each of two random-initialized candidates.",
            "- Compared objectives: recorded-token cross entropy and top-k residual-bucket distillation.",
            f"- Selection: validation continuation cross entropy; chosen top-k weight {selection['topk_weight']}.",
            f"- Untouched test recorded-token agreement: {test_report['teacher_token_agreement']:.6f}.",
            f"- Untouched test greedy exact match: {test_report['generation_exact_match']:.6f}.",
            f"- Projection relative parameter error with observed hidden features: {component['parameter_relative_error']:.6g}.",
            f"- Projection design rank: {component['diagnostics']['design_rank']} of {component['hidden_features']}.",
            f"- New candidate-entropy queries selected: {query_report['selected_prompts']}.",
            "",
            "See `trial.json` for provenance, `comparison/comparison.md` for validation comparisons, and `test.json` for the single final test evaluation.",
            "",
        ]
        atomic_write_text(output / "trial.md", "\n".join(summary))
        stage = "complete"
        progress("complete", elapsed_seconds=report["elapsed_seconds"])
        return report
    except BaseException as exc:
        progress(
            "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        torch.set_num_threads(original_threads)


def add_subcommands(subparsers):
    parser = subparsers.add_parser(
        "init-tiny", help="Create a tiny untrained local Qwen2 fixture without downloads"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.set_defaults(func=init_tiny)
    parser = subparsers.add_parser(
        "offline-trial",
        help="Run a complete local experiment pipeline with a synthetic tiny teacher",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.set_defaults(func=offline_trial)
