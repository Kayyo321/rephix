"""Prepare weight-free trace inputs and independently evaluate a recovered decoder."""

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
from layer_recovery_experiment import _behavior, _capture, _hashes
from logit_subspace_experiment import TEST, prompts

from rephix.hf import _load_tokenizer
from rephix.io import file_sha256, write_json
from rephix.weight_audit import compare_checkpoints

_METADATA = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
)


def prepare(args):
    args.work.mkdir(parents=True, exist_ok=True)
    config_path = args.work / "config"
    config_path.mkdir()
    for name in _METADATA:
        source = args.source / name
        if source.is_file():
            shutil.copyfile(source, config_path / name)
    config = json.loads((config_path / "config.json").read_text(encoding="utf-8"))
    classes = config["vocab_size"]
    rng = np.random.default_rng(7115)
    ids = rng.permutation(classes)
    ids = np.concatenate((ids, rng.integers(classes, size=(-classes % 16)))).reshape(-1, 16)
    np.save(args.work / "train.npy", ids.astype(np.int32), allow_pickle=False)
    validation = rng.integers(classes, size=(32, 16), dtype=np.int32)
    np.save(args.work / "validation.npy", validation, allow_pickle=False)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    started = time.monotonic()
    work = args.work
    candidate = work / "recovered"
    report = _read(candidate / "trace-recovery.json")
    manifest = _read(work / "traces" / "manifest.json")
    if report["dtype"] != manifest["dtype"]:
        raise ValueError("capture and recovery use different execution precision")
    opts = {"local_files_only": True, "trust_remote_code": False}
    dtype = getattr(torch, report["dtype"])
    prompts_spec = prompts(TEST, 4, 7116)
    texts = [row["text"] for row in prompts_spec]

    def capture(path):
        model = (
            AutoModelForCausalLM.from_pretrained(
                path, use_safetensors=True, torch_dtype=dtype, **opts
            )
            .eval()
            .to(args.device)
        )
        tokenizer = _load_tokenizer(AutoTokenizer, path, model_config=model.config, **opts)
        return _capture(model, tokenizer, texts, args.device, logit_dtype=torch.float64)

    audit = compare_checkpoints(args.source, candidate)
    audit.pop("reference")
    audit.pop("candidate")
    reference, reconstructed = capture(args.source), capture(candidate)
    operators = report["operators"]
    report.pop("checkpoint")
    report.pop("operators")
    rank_checks = [op for op in operators if "condition_number" in op]
    validation_checks = [op for op in operators if "validation_rmse" in op]
    summary = {
        "operator_count": len(operators),
        "full_design_rank_count": sum(
            op["design_rank"] == op["design_width"] for op in rank_checks
        ),
        "rank_checked_count": len(rank_checks),
        "embedding_tables": [op for op in operators if op["kind"] == "embedding"],
        "worst_condition": max(rank_checks, key=lambda op: op["condition_number"]),
        "worst_validation": max(validation_checks, key=lambda op: op["validation_rmse"]),
        "worst_runtime_validation": max(
            validation_checks, key=lambda op: op["validation_runtime_rmse"]
        ),
    }
    report["summary"] = summary
    manifest.pop("layout")
    manifest.pop("train_counts")
    manifest.pop("validation_counts")
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "reference_sha256": _hashes(args.source),
        "candidate_sha256": _hashes(candidate),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "contexts_sha256": {
            split: file_sha256(work / f"{split}.npy") for split in ("train", "validation")
        },
        "trace_bytes": sum(p.stat().st_size for p in (work / "traces").rglob("*.npy")),
        "capture": manifest,
        "recovery": report,
        "raw_checkpoint_audit": audit,
        "held_out_prompts": prompts_spec,
        "recovered_behavior": _behavior(reference, reconstructed),
        "evaluation_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    prep = stages.add_parser("prepare")
    ev = stages.add_parser("evaluate")
    for stage in (prep, ev):
        stage.add_argument("--source", type=Path, required=True)
        stage.add_argument("--work", type=Path, required=True)
    ev.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ev.add_argument("--threads", type=int, default=4)
    ev.add_argument("--model-id", required=True)
    ev.add_argument("--revision", required=True)
    ev.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
