"""Run one local dense endpoint head recovery experiment."""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from logit_subspace_experiment import TEST, prompts

from rephix.endpoint_head import _forward_kwargs
from rephix.hf import _load_tokenizer
from rephix.io import file_sha256, write_json
from rephix.weight_audit import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    _inventory,
    _open_shard,
    compare_checkpoints,
)

ANCHOR = 0
BATCH = 16
DELTA_STD = 0.02
GATE = 1e-3
MODEL_OPTIONS = {"local_files_only": True, "trust_remote_code": False}


def load_model(torch, auto_model, path, precision="float32"):
    return auto_model.from_pretrained(
        path, dtype=getattr(torch, precision), use_safetensors=True, **MODEL_OPTIONS
    ).eval()


def source_info(path):
    weights = path / "model.safetensors"
    return {
        "path": str(path),
        "model_safetensors_sha256": file_sha256(weights) if weights.is_file() else None,
    }


def make_contexts(tokenizer, base, fit_multiplier, validation_rows, seed):
    head = base.get_output_embeddings()
    train_rows = fit_multiplier * (head.in_features + int(head.bias is not None))
    capacity = base.get_input_embeddings().num_embeddings
    context_limit = getattr(base.config, "max_position_embeddings", None)
    vocab = sorted(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids))
    valid = np.asarray([token_id for token_id in vocab if 0 <= token_id < capacity], dtype=np.int64)
    if len(valid) < 2:
        raise ValueError("need at least two nonspecial tokenizer IDs")
    max_length = min(context_limit, 12) if context_limit is not None else 12
    min_length = min(4, max_length)
    natural = prompts(TEST, 4, seed + 1)
    final = [tuple(tokenizer.encode(row["text"], add_special_tokens=False)) for row in natural]
    seen = set(final)
    if len(seen) != len(final) or any(
        not row
        or min(row) < 0
        or max(row) >= capacity
        or (context_limit is not None and len(row) > context_limit)
        for row in final
    ):
        raise ValueError("natural prompts must encode to distinct valid contexts")

    def synthetic(count, rng):
        contexts = []
        for _ in range(count * 20):
            length = rng.integers(min_length, max_length + 1)
            row = tuple(rng.choice(valid, size=length))
            if row not in seen:
                seen.add(row)
                contexts.append(row)
                if len(contexts) == count:
                    return contexts
        raise ValueError("could not sample enough unique synthetic contexts")

    train_seed, val_seed = np.random.SeedSequence(seed).spawn(2)
    return natural, {
        "train": synthetic(train_rows, np.random.default_rng(train_seed)),
        "validation": synthetic(validation_rows, np.random.default_rng(val_seed)),
        "final": final,
    }


def allocate_capture(directory, contexts, classes, *, store_logits, dtype):
    directory.mkdir()
    lengths = np.asarray([len(row) for row in contexts], dtype=np.int32)
    tokens = np.zeros((len(contexts), int(lengths.max())), dtype=np.int32)
    for index, row in enumerate(contexts):
        tokens[index, : len(row)] = row
    np.save(directory / "input_ids.npy", tokens, allow_pickle=False)
    np.save(directory / "lengths.npy", lengths, allow_pickle=False)
    shape = len(contexts), classes
    scores = np.lib.format.open_memmap(
        directory / "scores.npy", mode="w+", dtype=dtype, shape=shape
    )
    logits = (
        np.lib.format.open_memmap(directory / "logits.npy", mode="w+", dtype=dtype, shape=shape)
        if store_logits
        else None
    )
    return lengths, tokens, scores, logits


def capture(
    torch,
    model,
    contexts,
    directory,
    device,
    *,
    store_logits=False,
    precision="float32",
    raw_scores=False,
):
    classes = model.get_output_embeddings().out_features
    lengths, tokens, scores, logits = allocate_capture(
        directory, contexts, classes, store_logits=store_logits, dtype=np.dtype(precision)
    )
    kwargs = _forward_kwargs(model)
    with torch.inference_mode():
        for length in np.unique(lengths):
            members = np.flatnonzero(lengths == length)
            for start in range(0, len(members), BATCH):
                rows = members[start : start + BATCH]
                ids = torch.as_tensor(tokens[rows, :length], dtype=torch.long, device=device)
                values = model(input_ids=ids, use_cache=False, **kwargs).logits[:, -1]
                if values.dtype != getattr(torch, precision):
                    raise ValueError(f"endpoint forward did not emit {precision} logits")
                scores[rows] = (values if raw_scores else values.log_softmax(dim=-1)).cpu().numpy()
                if logits is not None:
                    logits[rows] = values.cpu().numpy()
    scores.flush()
    if logits is not None:
        logits.flush()
    return {
        "queries": len(contexts),
        "input_tokens": int(lengths.sum()),
        "dense_scores": len(contexts) * classes,
    }


def plant(torch, model, directory, args):
    head = model.get_output_embeddings()
    head_name = next(name for name, layer in model.named_modules() if layer is head)
    base_tied = head.weight is model.get_input_embeddings().weight
    if base_tied:
        head.weight = torch.nn.Parameter(head.weight.detach().clone())
    model.config.tie_word_embeddings = False
    model.save_pretrained(directory / "base_fp32", safe_serialization=True)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    row_count = head.out_features
    if args.fit_mode == "anchored":
        changed = torch.arange(1, row_count)
    elif args.fit_mode == "base_median" and not args.plant_common_shift:
        chosen = torch.randperm(row_count - 1, generator=rng)[: max(1, row_count // 5) - 1]
        changed = torch.cat((torch.zeros(1, dtype=torch.long), chosen + 1))
    else:
        changed = torch.arange(row_count)
    with torch.no_grad():
        for parameter in (head.weight, head.bias):
            if parameter is None:
                continue
            if args.plant_common_shift:
                parameter.add_(torch.randn(parameter.shape[1:], generator=rng) * DELTA_STD)
                continue
            for indices in changed.split(512):
                parameter[indices] += (
                    torch.randn(parameter[indices].shape, generator=rng) * DELTA_STD
                )
    model.save_pretrained(directory / "planted_reference", safe_serialization=True)
    return head_name, base_tied, head.bias is not None, len(changed)


def run_estimator(args, candidate):
    argv = [
        sys.executable,
        "-m",
        "rephix",
        "--threads",
        str(args.threads),
        "recover-endpoint-head",
        "--base",
        str(args.base),
        "--train",
        str(args.output / "train"),
        "--validation",
        str(args.output / "validation"),
        "--output",
        str(candidate),
        "--mode",
        args.fit_mode,
        "--dtype",
        args.backbone_precision,
        "--device",
        args.device,
        "--batch-size",
        str(BATCH),
        "--max-validation-rmse",
        str(GATE),
    ]
    if args.fit_mode == "anchored":
        argv.extend(("--anchor-id", str(ANCHOR)))
    env = {
        **os.environ,
        **dict.fromkeys(
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"), str(args.threads)
        ),
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    started = time.perf_counter()
    with (args.output / "estimator.stdout.txt").open("w") as stdout:
        with (args.output / "estimator.stderr.txt").open("w") as stderr:
            status = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr, check=False)
    return {
        "command": argv,
        "returncode": status.returncode,
        "seconds": time.perf_counter() - started,
        "candidate_present": candidate.exists(),
        "stderr": (args.output / "estimator.stderr.txt").read_text(encoding="utf-8").strip(),
    }


class ErrorMetrics:
    def __init__(self):
        self.count = 0
        self.exact = 0
        self.within = 0
        self.error_squared = 0.0
        self.reference_squared = 0.0
        self.max_abs_error = 0.0

    def add(self, reference, estimate):
        error = np.abs(estimate - reference)
        self.count += reference.size
        self.exact += int(np.count_nonzero(error == 0))
        self.within += int(
            np.count_nonzero(error <= DEFAULT_ATOL + DEFAULT_RTOL * np.abs(reference))
        )
        self.error_squared += float(np.vdot(error, error))
        self.reference_squared += float(np.vdot(reference, reference))
        self.max_abs_error = max(self.max_abs_error, float(error.max(initial=0.0)))

    def result(self):
        return {
            "elements": self.count,
            "exactly_equal_elements": self.exact,
            "within_tolerance_elements": self.within,
            "fraction_exactly_equal": self.exact / self.count if self.count else None,
            "fraction_within_tolerance": self.within / self.count if self.count else None,
            "relative_l2": (
                (self.error_squared / self.reference_squared) ** 0.5
                if self.reference_squared
                else (0.0 if self.count and not self.error_squared else None)
            ),
            "rmse": (self.error_squared / self.count) ** 0.5 if self.count else None,
            "max_abs_error": self.max_abs_error if self.count else None,
        }


def compare_heads(reference_path, baseline_path, candidate_path, names):
    metrics = {label: (ErrorMetrics(), ErrorMetrics()) for label in ("recovered", "base_only")}
    checkpoints = [_inventory(path) for path in (reference_path, baseline_path, candidate_path)]
    anchor_equal = {}
    for name in names:
        entries = [inventory[name] for inventory in checkpoints]
        if not all(entry.shape == entries[0].shape for entry in entries):
            raise ValueError(f"checkpoint shape mismatch for {name}")
        with ExitStack() as stack:
            sources = [
                stack.enter_context(_open_shard(entry.shard)).get_slice(name) for entry in entries
            ]
            for start in range(0, entries[0].shape[0], 512):
                selection = slice(start, start + 512)
                reference, baseline, candidate = [
                    source[selection].double().numpy() for source in sources
                ]
                if start <= ANCHOR < start + len(reference):
                    index = ANCHOR - start
                    anchor_equal[name] = {
                        "reference_matches_base": bool(
                            np.array_equal(reference[index], baseline[index])
                        ),
                        "recovered_matches_base": bool(
                            np.array_equal(candidate[index], baseline[index])
                        ),
                    }
                changed = reference != baseline
                for label, other in (("recovered", candidate), ("base_only", baseline)):
                    full, modified = metrics[label]
                    full.add(reference, other)
                    modified.add(reference[changed], other[changed])
    return {
        "reference_changed_definition": "reference numeric coordinate differs exactly from base",
        "atol": DEFAULT_ATOL,
        "rtol": DEFAULT_RTOL,
        "anchor_coordinates": anchor_equal,
        **{
            label: {"all": full.result(), "changed": modified.result()}
            for label, (full, modified) in metrics.items()
        },
    }


def compare_predictions(reference, candidate):
    result = {}
    for name in ("logits", "scores"):
        a = np.load(reference / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        b = np.load(candidate / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if a.shape != b.shape:
            raise ValueError("final score matrices have different shapes")
        metrics = ErrorMetrics()
        agreements = 0
        for start in range(0, len(a), BATCH):
            a_rows = a[start : start + BATCH].astype(np.float64)
            b_rows = b[start : start + BATCH].astype(np.float64)
            metrics.add(a_rows, b_rows)
            agreements += int(np.count_nonzero(a_rows.argmax(axis=1) == b_rows.argmax(axis=1)))
        result["log_probs" if name == "scores" else name] = {
            **metrics.result(),
            "top1_matches": agreements,
            "top1_agreement": agreements / len(a),
        }
    return result


def run(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    args.base = args.base.resolve()
    args.output = args.output.resolve()
    if args.target is not None:
        args.target = args.target.resolve()
    for source in (args.base, args.target):
        if source is not None and (args.output == source or source in args.output.parents):
            raise ValueError("experiment output must be outside source checkpoints")
    args.output.mkdir(parents=True)
    tokenizer = _load_tokenizer(AutoTokenizer, args.base, **MODEL_OPTIONS)
    base = load_model(torch, AutoModelForCausalLM, args.base, args.backbone_precision)
    natural, contexts = make_contexts(
        tokenizer, base, args.fit_multiplier, args.validation_rows, args.seed
    )
    write_json(args.output / "final_prompts.json", natural)
    candidate = args.output / "recovered"
    negative = args.target is not None
    target_model = source_info(args.target) if negative else {"planted_from_base": True}
    report = {
        "schema_version": 1,
        "case": "different_backbone"
        if negative
        else "rank_deficient_backbone"
        if args.expect_rank_deficient
        else "planted_private_head",
        "base_model": source_info(args.base),
        "target_model": target_model,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "model_dtype": args.backbone_precision,
            "evidence_dtype": args.endpoint_precision,
            "torch_threads": args.threads,
        },
        "configuration": {
            "seed": args.seed,
            "fit_multiplier": args.fit_multiplier,
            "validation_rows": args.validation_rows,
            "batch_size": BATCH,
            "anchor_id": ANCHOR if args.fit_mode == "anchored" else None,
            "fit_mode": args.fit_mode,
            "plant_common_shift": args.plant_common_shift,
            "max_validation_contrast_rmse": GATE,
            "plant_delta_std": None if negative else DELTA_STD,
        },
    }

    if negative:
        target_tokenizer = _load_tokenizer(AutoTokenizer, args.target, **MODEL_OPTIONS)
        if tokenizer.get_vocab() != target_tokenizer.get_vocab():
            raise ValueError("negative control requires identical token-to-ID coordinates")
        if any(
            tuple(target_tokenizer.encode(item["text"], add_special_tokens=False)) != row
            for item, row in zip(natural, contexts["final"])
        ):
            raise ValueError("negative control must tokenize the final prompts identically")
        endpoint = load_model(torch, AutoModelForCausalLM, args.target, args.backbone_precision)
        if (
            base.config.model_type != endpoint.config.model_type
            or base.get_output_embeddings().weight.shape
            != endpoint.get_output_embeddings().weight.shape
            or base.get_input_embeddings().num_embeddings
            != endpoint.get_input_embeddings().num_embeddings
        ):
            raise ValueError("negative control requires matching model family and token dimensions")
        report["tokenizer_coordinates_match"] = True
        del base
    else:
        capture(
            torch,
            base.to(args.device),
            contexts["final"],
            args.output / "base_final",
            args.device,
            store_logits=True,
            precision=args.backbone_precision,
        )
        base.to("cpu")
        head_name, base_tied, has_bias, changed_rows = plant(torch, base, args.output, args)
        report["base_output_was_tied"] = base_tied
        report["configuration"]["planted_changed_rows"] = changed_rows
        endpoint = base
        del base

    endpoint.to(args.device)
    hook = None
    if args.endpoint_precision != args.backbone_precision:
        endpoint.get_output_embeddings().double()
        hook = endpoint.get_output_embeddings().register_forward_pre_hook(
            lambda module, inputs: (inputs[0].double(),)
        )
    try:
        report["target_query_counts"] = {
            split: capture(
                torch,
                endpoint,
                rows,
                args.output / split,
                args.device,
                store_logits=split == "final",
                precision=args.endpoint_precision,
                raw_scores=args.fit_mode == "raw_logits" and split != "final",
            )
            for split, rows in contexts.items()
        }
    finally:
        if hook is not None:
            hook.remove()
    del endpoint
    if args.device == "cuda":
        torch.cuda.empty_cache()
    result = run_estimator(args, candidate)
    report["estimator"] = result

    if negative:
        match = re.search(r"held-out contrast RMSE ([\d.eE+-]+) exceeds", result["stderr"])
        if match is not None:
            result["rejected_validation_contrast_rmse"] = float(match.group(1))
        passed = result["returncode"] != 0 and not result["candidate_present"] and match is not None
    elif args.expect_rank_deficient:
        match = re.search(
            r"training design is rank deficient: rank (\d+) of (\d+)", result["stderr"]
        )
        if match is not None:
            result["rejected_design_rank"], result["parameters_per_class"] = map(
                int, match.groups()
            )
        passed = result["returncode"] != 0 and not result["candidate_present"] and match is not None
    else:
        passed = result["returncode"] == 0 and result["candidate_present"]
        if passed:
            report["fit"] = json.loads(
                (candidate / "endpoint-head.json").read_text(encoding="utf-8")
            )
            recovered = load_model(
                torch, AutoModelForCausalLM, candidate, args.backbone_precision
            ).to(args.device)
            capture(
                torch,
                recovered,
                contexts["final"],
                args.output / "recovered_final",
                args.device,
                store_logits=True,
                precision=args.backbone_precision,
            )
            del recovered
            reference = args.output / "planted_reference"
            baseline = args.output / "base_fp32"
            report["full_parameters"] = {
                "recovered": compare_checkpoints(reference, candidate),
                "base_only": compare_checkpoints(reference, baseline),
            }
            names = [f"{head_name}.weight"]
            if has_bias:
                names.append(f"{head_name}.bias")
            report["head_parameters"] = compare_heads(reference, baseline, candidate, names)
            report["final_behavior"] = {
                "recovered": compare_predictions(
                    args.output / "final", args.output / "recovered_final"
                ),
                "base_only": compare_predictions(args.output / "final", args.output / "base_final"),
            }

    report["control_passed"] = passed
    report["seconds"] = time.perf_counter() - started
    write_json(args.output / "report.json", report)
    if not passed:
        raise RuntimeError(f"control failed; see {args.output / 'report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="local public base checkpoint")
    parser.add_argument("--target", type=Path, help="local different-backbone negative control")
    parser.add_argument("--output", required=True, type=Path, help="new scratch output directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=7115)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fit-multiplier", type=int, default=3)
    parser.add_argument("--validation-rows", type=int, default=128)
    parser.add_argument("--endpoint-precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--backbone-precision", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--fit-mode", choices=("anchored", "raw_logits", "base_median"), default="anchored"
    )
    parser.add_argument("--plant-common-shift", action="store_true")
    parser.add_argument("--expect-rank-deficient", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or min(args.threads, args.fit_multiplier, args.validation_rows) < 1:
        parser.error("seed must be nonnegative and counts must be positive")
    if args.target is not None and (
        args.endpoint_precision != "float32"
        or args.fit_mode != "anchored"
        or args.expect_rank_deficient
    ):
        parser.error("different-backbone control requires standard FP32 endpoint scores")
    if args.backbone_precision == "float64" and args.endpoint_precision != "float64":
        parser.error("FP64 backbone requires FP64 endpoint scores")
    if args.plant_common_shift and (args.fit_mode != "base_median" or args.target is not None):
        parser.error("the common-shift control requires a planted base_median trial")
    run(args)


if __name__ == "__main__":
    main()
