"""Identify an open model's output projection from observable head inputs.

This deliberately white-box experiment uses final hidden activations, an access
level that prompt/output collection does not provide. Reference parameters are
used only after the least-squares solve, to evaluate recovery accuracy.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from .evidence import assert_disjoint, leakage_groups, read_observations
from .io import atomic_save_npz, file_sha256, write_json
from .linear import RecoveryResult, recover


def recover_projection(features, logits, *, bias_mode="known-zero", ridge=0.0):
    """Fit ``features @ weights + bias`` without reading reference parameters.

    ``known-zero`` supplies the architecture's absent bias as a constraint;
    ``fit`` estimates an affine intercept. Neither mode infers hidden features.
    """
    if bias_mode == "fit":
        return recover(features, logits, kind="logits", ridge=ridge)
    if bias_mode != "known-zero":
        raise ValueError("bias_mode must be fit or known-zero")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or not all(x.shape) or not all(y.shape):
        raise ValueError("features and logits must be nonempty matrices")
    if x.shape[0] != y.shape[0] or y.shape[1] < 2:
        raise ValueError("features/logits must share sample count and have at least two outputs")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and logits must be finite")
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)):
        raise ValueError("ridge must be a finite nonnegative number")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be a finite nonnegative number")
    singular = np.linalg.svd(x, compute_uv=False)
    tolerance = max(x.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    full_rank = rank == x.shape[1]
    design, targets = x, y
    if ridge:
        design = np.vstack((x, np.sqrt(ridge) * np.eye(x.shape[1])))
        targets = np.vstack((y, np.zeros((x.shape[1], y.shape[1]))))
    weights = np.linalg.lstsq(design, targets, rcond=None)[0]
    return RecoveryResult(
        weights,
        np.zeros(y.shape[1], dtype=np.float64),
        {
            "observation_kind": "logits",
            "parameter_gauge": "full_known_zero_bias",
            "bias_mode": bias_mode,
            "samples": int(x.shape[0]),
            "features": int(x.shape[1]),
            "classes": int(y.shape[1]),
            "design_rank": rank,
            "parameters_per_class": int(x.shape[1]),
            "identifiable_in_reported_gauge": full_rank,
            "condition_number": float(singular[0] / singular[-1]) if full_rank else None,
            "rank_tolerance": float(tolerance),
            "ridge": float(ridge),
            "training_target_rmse": float(np.sqrt(np.mean((x @ weights - y) ** 2))),
            "interpretation": (
                "Unique selected output weights for exact observed features/logits and known zero bias."
                if full_rank
                else "Rank deficient: unobserved feature directions prevent unique parameter recovery."
            ),
        },
    )


def projection_demo(
    *,
    seed=42,
    feature_sizes=(8, 16, 32),
    repeats=3,
    noise_levels=(0.0, 0.01, 0.1),
    test_samples=256,
):
    """Measure planted projection recovery across observation budgets and noise.

    Features are independent Gaussian draws and the zero bias is known. Every
    budget reuses the same nested training features, model, and test set within
    a replicate. Observation noise is added only to training logits; held-out
    error is measured against the clean planted function.
    """
    feature_sizes = tuple(feature_sizes)
    noise_levels = tuple(noise_levels)
    if not feature_sizes or any(type(size) is not int or size < 2 for size in feature_sizes):
        raise ValueError("feature_sizes must contain integers >= 2")
    if type(repeats) is not int or repeats < 1 or type(test_samples) is not int or test_samples < 1:
        raise ValueError("repeats and test_samples must be positive integers")
    if not noise_levels or any(not math.isfinite(noise) or noise < 0 for noise in noise_levels):
        raise ValueError("noise_levels must contain finite nonnegative values")
    rng = np.random.default_rng(seed)
    ratios = (0.25, 0.5, 1.0, 2.0, 4.0)
    outcomes = {}
    for width in feature_sizes:
        budgets = sorted({max(1, round(width * ratio)) for ratio in ratios})
        for _ in range(repeats):
            weights = rng.normal(scale=1 / np.sqrt(width), size=(width, 5))
            x = rng.normal(size=(max(budgets), width))
            test_x = rng.normal(size=(test_samples, width))
            clean_y = x @ weights
            test_y = test_x @ weights
            standard_noise = rng.normal(size=clean_y.shape)
            for noise in noise_levels:
                observed = clean_y + noise * standard_noise
                for budget in budgets:
                    result = recover_projection(x[:budget], observed[:budget])
                    predictions = result.predict(test_x)
                    key = (width, budget, float(noise))
                    outcomes.setdefault(key, []).append(
                        {
                            "parameter_relative_error": float(
                                np.linalg.norm(result.weights - weights) / np.linalg.norm(weights)
                            ),
                            "heldout_logit_rmse": float(
                                np.sqrt(np.mean((predictions - test_y) ** 2))
                            ),
                            "training_target_rmse": result.diagnostics["training_target_rmse"],
                            "design_rank": result.diagnostics["design_rank"],
                            "condition_number": result.diagnostics["condition_number"],
                        }
                    )
    rows = []
    for (width, budget, noise), values in sorted(outcomes.items()):
        row = {
            "features": width,
            "classes": 5,
            "train_samples": budget,
            "samples_per_feature": budget / width,
            "observation_noise_std": noise,
            "replicates": repeats,
            "design_rank_min": min(value["design_rank"] for value in values),
            "design_rank_max": max(value["design_rank"] for value in values),
            "identifiable_all_replicates": all(value["design_rank"] == width for value in values),
        }
        for name in ("parameter_relative_error", "heldout_logit_rmse", "training_target_rmse"):
            measurements = [value[name] for value in values]
            row[f"{name}_mean"] = float(np.mean(measurements))
            row[f"{name}_std"] = float(np.std(measurements, ddof=1)) if repeats > 1 else 0.0
        conditions = [
            value["condition_number"] for value in values if value["condition_number"] is not None
        ]
        row["condition_number_mean"] = float(np.mean(conditions)) if conditions else None
        rows.append(row)
    return {
        "experiment": "synthetic_known_feature_projection_budget_noise",
        "seed": seed,
        "feature_sizes": list(feature_sizes),
        "replicates": repeats,
        "noise_levels": list(noise_levels),
        "test_samples_per_replicate": test_samples,
        "known_bias": "zero",
        "observation_type": "full_selected_output_logits",
        "heldout_targets": "clean planted logits on independent Gaussian features",
        "aggregation": "mean and sample standard deviation across independently planted models",
        "rows": rows,
        "interpretation": "Controlled Gaussian affine-layer identification; these curves do not measure Qwen or text-only reconstruction.",
    }


def plot_projection_demo(report, path):
    """Export a standalone Matplotlib figure for ``projection_demo`` results."""
    try:
        import matplotlib
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError("Install rephix[analysis] to plot component recovery results") from exc
    features = report["feature_sizes"]
    figure = Figure(figsize=(4 * len(features), 6.5), layout="constrained")
    axes = figure.subplots(2, len(features), squeeze=False, sharex=True, sharey="row")
    for column, width in enumerate(features):
        for noise in report["noise_levels"]:
            rows = sorted(
                [
                    row
                    for row in report["rows"]
                    if row["features"] == width and row["observation_noise_std"] == noise
                ],
                key=lambda row: row["train_samples"],
            )
            x = [row["samples_per_feature"] for row in rows]
            for row_index, metric in enumerate(
                ("parameter_relative_error_mean", "heldout_logit_rmse_mean")
            ):
                axes[row_index, column].plot(
                    x,
                    [max(row[metric], 1e-15) for row in rows],
                    marker="o",
                    label=f"noise std {noise:g}",
                )
        axes[0, column].set_title(f"{width} known features")
        for axis in axes[:, column]:
            axis.axvline(1, color="0.5", linestyle="--", linewidth=1)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.grid(True, alpha=0.2)
        axes[1, column].set_xlabel("Training observations / features")
    axes[0, 0].set_ylabel("Parameter relative L2 error")
    axes[1, 0].set_ylabel("Clean held-out logit RMSE")
    axes[0, 0].legend(fontsize="small")
    figure.suptitle(
        f"Known-feature projection recovery; {report['replicates']} planted models per point"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.hashsalt": "rephix-projection-demo"}):
        figure.savefig(destination, metadata={"Date": None, "Creator": "rephix.projection_demo"})
    return str(destination)


def _split_prompts(records, test_fraction, seed):
    if not math.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError("test-fraction must be between zero and one")
    groups = leakage_groups(records)
    if len(groups) < 2:
        raise ValueError("projection recovery needs at least two independent prompt groups")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * test_fraction)))
    test = [record for group in groups[:count] for record in group]
    train = [record for group in groups[count:] for record in group]
    assert_disjoint(train, test)
    return train, test


def _selected_ids(vocab_size, vocab_limit, token_ids):
    if token_ids:
        try:
            selected = [int(value.strip()) for value in token_ids.split(",")]
        except (TypeError, ValueError) as exc:
            raise ValueError("token-ids must be comma-separated integer vocabulary IDs") from exc
        if len(selected) != len(set(selected)):
            raise ValueError("token-ids must be unique")
        if any(value < 0 or value >= vocab_size for value in selected):
            raise ValueError("token-ids contains an ID outside the model vocabulary")
    else:
        if vocab_limit < 0:
            raise ValueError("vocab-limit must be nonnegative (zero selects the entire vocabulary)")
        count = min(vocab_size, vocab_limit) if vocab_limit else vocab_size
        selected = np.linspace(0, vocab_size - 1, num=count, dtype=np.int64).tolist()
    if len(selected) < 2:
        raise ValueError("projection recovery needs at least two selected vocabulary IDs")
    return selected


def _feature_novelty(train_features, test_features, observed_logits, predictions):
    """Separate repeated exact features from new features within a prompt split."""
    # Inputs are saved float32 observations, so byte equality has an unambiguous
    # meaning here. This diagnoses shared chat prefixes without changing splits.
    known = {row.tobytes() for row in train_features}
    test_keys = [row.tobytes() for row in test_features]
    novel = np.array([key not in known for key in test_keys])
    count = int(novel.sum())
    return {
        "matching_training_rows": int(len(novel) - count),
        "novel_test_rows": count,
        "unique_test_feature_rows": len(set(test_keys)),
        "novel_only_logit_rmse": (
            float(np.sqrt(np.mean((predictions[novel] - observed_logits[novel]) ** 2)))
            if count
            else None
        ),
        "novel_only_selected_vocabulary_top1_agreement": (
            float(np.mean(predictions[novel].argmax(-1) == observed_logits[novel].argmax(-1)))
            if count
            else None
        ),
    }


def _collect_features(torch, model, tokenizer, records, selected, max_length, cap, device, seed):
    """Capture actual projection inputs, paired with the model's emitted logits."""
    from .hf import _render_prompt

    head = model.get_output_embeddings()
    capture = []

    def before_head(module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            raise ValueError("output projection does not receive a tensor as its first input")
        capture.append(inputs[0].detach())

    hook = head.register_forward_pre_hook(before_head)
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    truncated = 0
    try:
        with torch.no_grad():
            for record in records:
                tokens = _render_prompt(tokenizer, record.prompt)
                if not tokens:
                    raise ValueError(f"{record.id}: prompt encoded to zero tokens")
                truncated += len(tokens) > max_length
                tokens = tokens[:max_length]
                inputs = torch.tensor([tokens], device=device)
                capture.clear()
                output = model(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False
                )
                if len(capture) != 1:
                    raise ValueError(
                        "expected exactly one output projection call per model forward"
                    )
                hidden = capture[0]
                if hidden.ndim != 3 or hidden.shape[:2] != output.logits.shape[:2]:
                    raise ValueError("output projection inputs do not align with model logits")
                x = hidden[0].float().cpu().numpy()
                y = output.logits[0, :, selected].float().cpu().numpy()
                if y.shape != (len(tokens), len(selected)):
                    raise ValueError("model logits have an unsupported shape")
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError("model returned nonfinite hidden features or logits")
                for position, (features, logits) in enumerate(zip(x, y)):
                    seen += 1
                    item = (features.copy(), logits.copy(), record.id, position)
                    if len(reservoir) < cap:
                        reservoir.append(item)
                    else:
                        slot = rng.randrange(seen)
                        if slot < cap:
                            reservoir[slot] = item
                del output, hidden
                capture.clear()
    finally:
        hook.remove()
    if not reservoir:
        raise ValueError("no projection samples were collected")
    return {
        "features": np.stack([item[0] for item in reservoir]),
        "logits": np.stack([item[1] for item in reservoir]),
        "record_ids": np.array([item[2] for item in reservoir]),
        "positions": np.array([item[3] for item in reservoir], dtype=np.int64),
        "available_positions": seen,
        "truncated_prompts": int(truncated),
    }


def run(args):
    from .hf import _deps, _device, _seed, _tokenizer_fingerprint

    for name in ("max_seq_length", "max_samples", "max_memory_mb"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_samples < 4:
        raise ValueError("max-samples must be at least four for train/test sampling")
    if not math.isfinite(args.ridge) or args.ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be absent or empty")
    records = read_observations(args.prompts)
    train_records, test_records = _split_prompts(records, args.test_fraction, args.seed)
    torch, _, AutoModel, AutoTokenizer = _deps()
    _seed(torch, args.seed)
    device = _device(torch, args.device)
    options = {"trust_remote_code": False, "revision": args.revision}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    model = AutoModel.from_pretrained(args.model, **options).to(device).eval()
    head = model.get_output_embeddings()
    if not isinstance(head, torch.nn.Linear) or head.weight.ndim != 2:
        raise ValueError("only a standard Linear output projection is supported")
    classes, features = head.weight.shape
    selected = _selected_ids(classes, args.vocab_limit, args.token_ids)
    context = getattr(model.config, "max_position_embeddings", args.max_seq_length)
    if args.max_seq_length > context:
        raise ValueError("max-seq-length exceeds the model context limit")
    bias_mode = args.bias_mode
    has_bias = head.bias is not None
    if bias_mode == "auto":
        bias_mode = "fit" if has_bias else "known-zero"
    if bias_mode == "known-zero" and has_bias:
        raise ValueError("known-zero bias mode requires an architecture with no output bias")
    # Bound stored observations plus float64 solver copies and selected truth.
    estimated_bytes = (
        args.max_samples * (features + len(selected)) * 12 + features * len(selected) * 24
    )
    if estimated_bytes > args.max_memory_mb * 1024**2:
        raise ValueError(
            "projection arrays exceed max-memory-mb; reduce samples or selected vocabulary"
        )
    test_cap = min(args.max_samples - 1, max(1, round(args.max_samples * args.test_fraction)))
    train = _collect_features(
        torch,
        model,
        tokenizer,
        train_records,
        selected,
        args.max_seq_length,
        args.max_samples - test_cap,
        device,
        args.seed,
    )
    test = _collect_features(
        torch,
        model,
        tokenizer,
        test_records,
        selected,
        args.max_seq_length,
        test_cap,
        device,
        args.seed + 1,
    )
    # These two arrays alone enter the estimator. Reference weights are fetched
    # only afterwards, for the explicit recovery-accuracy measurement.
    recovered = recover_projection(
        train["features"], train["logits"], bias_mode=bias_mode, ridge=args.ridge
    )
    truth_weights = head.weight.detach()[selected].float().cpu().numpy().T.astype(np.float64)
    truth_bias = (
        head.bias.detach()[selected].float().cpu().numpy().astype(np.float64)
        if has_bias
        else np.zeros(len(selected))
    )
    reference_prediction = test["features"].astype(np.float64) @ truth_weights + truth_bias
    if not np.allclose(reference_prediction, test["logits"], rtol=1e-4, atol=1e-4):
        raise ValueError(
            "model output logits include a transform beyond the captured affine projection"
        )
    predictions = recovered.predict(test["features"])
    truth_parameters = np.vstack((truth_weights, truth_bias))
    recovered_parameters = np.vstack((recovered.weights, recovered.bias))
    parameter_error = float(np.linalg.norm(recovered_parameters - truth_parameters))
    truth_norm = float(np.linalg.norm(truth_parameters))
    report = {
        "experiment": "open_model_output_projection_recovery",
        "access": "white_box_observed_projection_inputs_and_logits",
        "uses_hidden_activations": True,
        "uses_reference_weights_in_estimator": False,
        "source_model": str(args.model),
        "source_revision": args.revision,
        "source_commit": getattr(model.config, "_commit_hash", None),
        "prompt_file_sha256": file_sha256(args.prompts),
        "tokenizer_fingerprint": _tokenizer_fingerprint(tokenizer),
        "seed": args.seed,
        "device": device,
        "model_dtype": str(head.weight.dtype),
        "bias_mode": bias_mode,
        "selected_token_ids": selected,
        "total_vocabulary_size": int(classes),
        "selected_vocabulary_size": len(selected),
        "hidden_features": int(features),
        "train_prompt_ids": [record.id for record in train_records],
        "test_prompt_ids": [record.id for record in test_records],
        "train_samples": int(len(train["features"])),
        "test_samples": int(len(test["features"])),
        "available_train_positions": train["available_positions"],
        "available_test_positions": test["available_positions"],
        "truncated_train_prompts": train["truncated_prompts"],
        "truncated_test_prompts": test["truncated_prompts"],
        "parameter_relative_error": parameter_error / truth_norm if truth_norm else None,
        "parameter_l2_error": parameter_error,
        "heldout_logit_rmse": float(np.sqrt(np.mean((predictions - test["logits"]) ** 2))),
        "heldout_selected_vocabulary_top1_agreement": float(
            np.mean(predictions.argmax(-1) == test["logits"].argmax(-1))
        ),
        "reference_projection_sanity_rmse": float(
            np.sqrt(np.mean((reference_prediction - test["logits"]) ** 2))
        ),
        "heldout_feature_novelty": _feature_novelty(
            train["features"], test["features"], test["logits"], predictions
        ),
        "diagnostics": recovered.diagnostics,
        "interpretation": "Recovery of selected output-projection columns with observed hidden features; not text-only recovery of a transformer.",
    }
    output.mkdir(parents=True, exist_ok=True)
    recovered.save(output / "recovered.npz")
    atomic_save_npz(
        output / "reference.npz",
        weights=truth_weights,
        bias=truth_bias,
        token_ids=np.asarray(selected),
    )
    atomic_save_npz(
        output / "observations.npz",
        train_features=train["features"],
        train_logits=train["logits"],
        train_record_ids=train["record_ids"],
        train_positions=train["positions"],
        test_features=test["features"],
        test_logits=test["logits"],
        test_record_ids=test["record_ids"],
        test_positions=test["positions"],
    )
    write_json(output / "report.json", report)
    return report


def add_subcommands(subparsers):
    parser = subparsers.add_parser(
        "recover-projection",
        help="Recover selected open-model output weights from observed hidden activations",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--max-memory-mb", type=int, default=512)
    parser.add_argument(
        "--vocab-limit",
        type=int,
        default=64,
        help="Evenly spaced vocabulary columns; zero selects all",
    )
    parser.add_argument(
        "--token-ids", help="Comma-separated vocabulary IDs, overriding vocab-limit"
    )
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--bias-mode", choices=["auto", "fit", "known-zero"], default="auto")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.set_defaults(func=run)
