"""Command line entry points. Optional model libraries load only when needed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import __version__
from .evidence import leakage_groups, read_observations, split_observations, write_observations
from .io import write_json


def _write_json(path, data):
    write_json(path, data)


def _demo(args):
    from .linear import demo

    result = demo(
        seed=args.seed, samples=args.samples, features=args.features, classes=args.classes
    )
    if args.output:
        _write_json(args.output, result)
    return result


def _recover(args):
    import numpy as np

    from .linear import recover

    data = np.load(args.input, allow_pickle=False)
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError("input must be an NPZ archive with features and observations arrays")
    with data:
        if not {"features", "observations"}.issubset(data.files):
            raise ValueError("input NPZ must contain features and observations arrays")
        result = recover(data["features"], data["observations"], kind=args.kind, ridge=args.ridge)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    return {"output": str(args.output), **result.diagnostics}


def _validate(args):
    records = read_observations(args.input)
    return {
        "path": str(args.input),
        "observations": len(records),
        "independent_prompt_groups": len(leakage_groups(records)),
        "with_observed_thinking": sum(item.thinking is not None for item in records),
        "with_token_ids": sum(item.completion_token_ids is not None for item in records),
        "with_top_logprobs": sum(item.top_logprobs is not None for item in records),
    }


def _split(args):
    records = read_observations(args.input)
    output = Path(args.output_dir)
    if any(
        (output / name).exists()
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")
    ):
        raise ValueError("split output files already exist; choose a new output directory")
    splits = split_observations(
        records,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    manifest = {
        "source": str(Path(args.input).resolve()),
        "source_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "seed": args.seed,
        "requested_validation_fraction": args.validation_fraction,
        "requested_test_fraction": args.test_fraction,
        "grouping": "connected components of normalized prompt and metadata.group_id",
        "splits": {},
    }
    for name, items in splits.items():
        path = output / f"{name}.jsonl"
        write_observations(path, items)
        manifest["splits"][name] = {
            "path": str(path),
            "observations": len(items),
            "independent_prompt_groups": len(leakage_groups(items)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    _write_json(output / "manifest.json", manifest)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rephix", description="Experimental weight recovery and behavioral reconstruction"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--threads", type=int, help="bound Torch CPU threads for model experiments")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser(
        "demo", help="recover planted affine weights and evaluate held-out features"
    )
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--samples", type=int, default=256)
    demo.add_argument("--features", type=int, default=8)
    demo.add_argument("--classes", type=int, default=5)
    demo.add_argument("--output", help="also save the JSON report here")
    demo.set_defaults(func=_demo)

    recover = commands.add_parser(
        "recover-linear", help="fit an observable affine layer from a NumPy archive"
    )
    recover.add_argument("input", help="NPZ with features and observations matrices")
    recover.add_argument(
        "--kind", choices=["logits", "log_probs", "probabilities"], default="logits"
    )
    recover.add_argument("--ridge", type=float, default=0.0)
    recover.add_argument("--output", required=True, help="recovered NPZ path")
    recover.set_defaults(func=_recover)

    validate = commands.add_parser(
        "validate", help="validate evidence JSONL and report available signals"
    )
    validate.add_argument("input")
    validate.set_defaults(func=_validate)

    split = commands.add_parser("split", help="isolate train, validation and test prompt groups")
    split.add_argument("input")
    split.add_argument("--output-dir", required=True)
    split.add_argument("--validation-fraction", type=float, default=0.2)
    split.add_argument("--test-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(func=_split)

    from . import alignment, doctor, experiments, hf, projection, query, tiny

    for module in (hf, projection, alignment, query, experiments, doctor, tiny):
        module.add_subcommands(commands)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.threads is not None:
            if args.threads < 1:
                raise ValueError("threads must be a positive integer")
            import importlib.util

            if importlib.util.find_spec("torch") is not None:
                import torch

                torch.set_num_threads(args.threads)
        result = args.func(args)
        if result is not None:
            print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"rephix: {exc}", file=sys.stderr)
        return 1
    return 0
