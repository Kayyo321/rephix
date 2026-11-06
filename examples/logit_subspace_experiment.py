"""Run the dense log-probability output-subspace experiment in three stages."""

import argparse
import inspect
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

from rephix.io import atomic_save_npz, file_sha256, write_json
from rephix.logit_subspace import fit_logit_subspace, log_odds

TRAIN = (
    "Write an inventory of parts for a {a} made out of {b} next to a {c}.",
    "Find the next element in the sequence {n}, {m}, {t}, {u},",
    "def {a}(items, {b}):\n    return {{value: index for index, value in enumerate(items) if",
    "Traduce al inglés esta frase: Mi {a} encuentra {n} pequeñas figuras de {b} en",
    "Explique pourquoi la couleur d'un {a} change quand on ajoute {m} litres de",
    "为了完成第{m}次任务，在纸上写出{t}个有关科学和",
    "جاء المسافر بعد {n} أيام ليجد أن المدينة القديمة",
    'Here is a partial HTTP response:\nHTTP/1.1 {n} {a}\nContent-Type: "{b}"\nX-{c}:',
    "The report from sector {m} says that {a} increased by {t}% over {u} weeks after",
    "Στο άρθρο {n} περιγράφεται μια σύγκρουση μεταξύ {a} και {b}. Η αιτία είναι",
    "Give the next line of a poem whose last line is 'my {a} {b} in the {c} light'.",
    "If the rule says ({m}*x + {u}) % {n} == {t}, then the smallest positive solution is",
)
TEST = (
    "Write a four-word complaint from the perspective of a {a} misplaced under a {b}.",
    "A {a} travels at {m} knots for {t} minutes. Convert that distance to meters.",
    "SELECT department, COUNT(*) FROM {a}_{b} GROUP BY department HAVING COUNT(*) > {m} ORDER BY",
    "Ein {a} kostet {m} Euro. Wie viel kosten {t} Stück nach einem Rabatt von {n} Prozent?",
    "次の言葉を比較して、違いを日本語で説明してください：{a}、{b}、{c}",
    "Rename the commit 'make {a} run {n} times before loading {b}' using a conventional commit.",
)
WORDS = (
    "abacus",
    "asteroid",
    "avocado",
    "balcony",
    "beetle",
    "bicycle",
    "canyon",
    "carpet",
    "chess",
    "coral",
    "coyote",
    "diagram",
    "dolphin",
    "enamel",
    "entropy",
    "flamingo",
    "glacier",
    "guitar",
    "hammer",
    "horizon",
    "igloo",
    "isotope",
    "jasmine",
    "jigsaw",
    "kilogram",
    "kiosk",
    "lemon",
    "locomotive",
    "marble",
    "medal",
    "nectarine",
    "neuron",
    "orchid",
    "otter",
    "pavilion",
    "pepper",
    "quail",
    "quartz",
    "relay",
    "ribbon",
    "seagull",
    "syrup",
    "tapestry",
    "tractor",
    "ukulele",
    "utopia",
    "vanilla",
    "volcano",
    "wagon",
    "walnut",
    "xylophone",
    "yogurt",
    "zeppelin",
    "zircon",
)
CUTOFF = 1e-6
CONTROLS = (1e-4, 1e-5, CUTOFF, 1e-7, 1e-8)
SELECTED = 1153
ANCHORS = 1000
TOP_K = 20
SHORT_ROWS = 128
ROUNDING_DECIMALS = 3


def prompts(templates, count, seed):
    rng = random.Random(seed)
    rows = []
    for family, template in enumerate(templates):
        seen = set()
        for _ in range(count * 50):
            a, b, c = rng.sample(WORDS, 3)
            n, m, t, u = rng.sample(range(3, 700), 4)
            rendered = template.format(a=a, b=b, c=c, n=n, m=m, t=t, u=u)
            if rendered not in seen:
                seen.add(rendered)
                rows.append({"family": family, "text": rendered})
                if len(seen) == count:
                    break
        else:
            raise ValueError(f"could not make {count} unique prompts for family {family}")
    rng.shuffle(rows)
    return rows


def capture(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    args.run.mkdir(parents=True)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float32)
        .eval()
        .to(args.device)
    )
    params = inspect.signature(model.forward).parameters
    last_logits = {"logits_to_keep": 1} if "logits_to_keep" in params else {}
    if "position_ids" not in params and args.batch_size > 1:
        raise ValueError("this model needs batch size 1 to avoid implicit padded position IDs")
    fixed_ids = np.random.default_rng(args.seed).choice(len(tokenizer), SELECTED, replace=False)
    ids = torch.tensor(fixed_ids, device=args.device)
    splits = {
        "train": prompts(TRAIN, 128, args.seed),
        "test": prompts(TEST, 32, args.seed + 1),
    }
    if {x["text"] for x in splits["train"]} & {x["text"] for x in splits["test"]}:
        raise ValueError("duplicate text across predeclared splits")
    write_json(args.run / "prompts.json", splits)
    counts = {}
    with torch.inference_mode():
        for split, rows in splits.items():
            observed, top_ids = [], []
            total_tokens = 0
            for i in range(0, len(rows), args.batch_size):
                texts = [row["text"] for row in rows[i : i + args.batch_size]]
                tokens = tokenizer(
                    texts, return_tensors="pt", padding=True, add_special_tokens=False
                ).to(args.device)
                mask = tokens["attention_mask"]
                total_tokens += int(mask.sum().item())
                position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
                positions = {"position_ids": position_ids} if "position_ids" in params else {}
                logits = model(**tokens, **positions, **last_logits, use_cache=False).logits[:, -1]
                distribution = logits.log_softmax(dim=1)
                observed.append(distribution.index_select(1, ids).cpu().numpy())
                top_ids.append(distribution.topk(TOP_K, dim=1).indices.cpu().numpy())
            atomic_save_npz(
                args.run / f"{split}.npz",
                logprobs=np.concatenate(observed),
                top_ids=np.concatenate(top_ids),
                fixed_ids=fixed_ids,
            )
            counts[split] = {
                "queries": len(rows),
                "families": len({row["family"] for row in rows}),
                "input_tokens": total_tokens,
                "fixed_scores": len(rows) * SELECTED,
            }
    weights = args.model / "model.safetensors"
    metadata = {
        "schema_version": 1,
        "model": args.model_id,
        "revision": args.revision,
        "model_sha256": file_sha256(weights),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "peak_allocated_cuda_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        ),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "precision": "float32",
        "seed": args.seed,
        "capture_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss_bytes(),
        "counts": counts,
        "files": file_hashes(args.run, ("prompts.json", "train.npz", "test.npz")),
    }
    write_json(args.run / "capture.json", metadata)


def file_hashes(run, names):
    return {name: file_sha256(run / name) for name in names}


def peak_rss_bytes():
    if sys.platform != "linux":
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def fit(args):
    started = time.perf_counter()
    args.run.mkdir(parents=True)
    with np.load(args.observations, allow_pickle=False) as train:
        dense = train["logprobs"]
        ids = train["fixed_ids"]
    for name, data in (
        ("dense", dense),
        ("short", dense[:SHORT_ROWS]),
        ("rounded", np.round(dense, ROUNDING_DECIMALS)),
    ):
        result = fit_logit_subspace(data, relative_cutoff=CUTOFF)
        atomic_save_npz(
            args.run / f"{name}-fit.npz",
            origin=result.origin,
            basis=result.basis,
            singular=result.singular_values,
            training_rows=np.asarray(result.training_rows),
            fixed_ids=ids,
        )
    write_json(
        args.run / "fit.json",
        {
            "fit_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss_bytes(),
            "relative_cutoff": CUTOFF,
            "threshold_sensitivity": CONTROLS,
            "rounded_decimals": ROUNDING_DECIMALS,
            "training_rows_short_control": SHORT_ROWS,
            "training_evidence_sha256": file_sha256(args.observations),
            "files": file_hashes(args.run, ("dense-fit.npz", "short-fit.npz", "rounded-fit.npz")),
            "command": args.command,
        },
    )


def test_fit(args):
    import torch
    from transformers import AutoModelForCausalLM

    from rephix.logit_subspace import LogitSubspace

    started = time.perf_counter()
    capture_metadata = json.loads((args.run / "capture.json").read_text())
    fit_metadata = json.loads((args.fit_dir / "fit.json").read_text())
    for path, record in ((args.run, capture_metadata), (args.fit_dir, fit_metadata)):
        if record["files"] != file_hashes(path, record["files"]):
            raise ValueError("artifact integrity check failed")
    if fit_metadata["training_evidence_sha256"] != capture_metadata["files"]["train.npz"]:
        raise ValueError("fit did not use the frozen training evidence")
    if file_sha256(args.model / "model.safetensors") != capture_metadata["model_sha256"]:
        raise ValueError("evaluation must use the exact reference weights queried")
    with np.load(args.run / "test.npz", allow_pickle=False) as test:
        observed, ids, top_ids = test["logprobs"], test["fixed_ids"], test["top_ids"]
    with np.load(args.run / "train.npz", allow_pickle=False) as train:
        train_top_ids = train["top_ids"]
    odds = log_odds(observed)
    anchors = np.arange(ANCHORS)
    # Only the evaluation command reads the ground-truth output weights.
    torch.set_num_threads(2)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32
    )
    weights = model.get_output_embeddings().weight.detach()[ids].double().numpy()
    ref_basis, ref_singular, _ = np.linalg.svd(weights[1:] - weights[0], full_matrices=False)
    ref_width = int(model.config.hidden_size)
    ref_rank = int(np.count_nonzero(ref_singular > CUTOFF * ref_singular[0]))
    ref_basis = ref_basis[:, :ref_rank]
    measurements = {}
    for name in ("dense", "short", "rounded"):
        with np.load(args.fit_dir / f"{name}-fit.npz", allow_pickle=False) as fitted:
            if not np.array_equal(fitted["fixed_ids"], ids):
                raise ValueError("fit and test token coordinates differ")
            result = LogitSubspace(
                fitted["origin"],
                fitted["basis"],
                fitted["singular"],
                fit_metadata["relative_cutoff"],
                int(fitted["training_rows"].item()),
            )
        residual = ref_basis - result.basis @ (result.basis.T @ ref_basis)
        extra = result.basis - ref_basis @ (ref_basis.T @ result.basis)
        normalized_projection_error = np.sqrt((np.sum(residual**2) + np.sum(extra**2)) / ref_rank)
        anchored_rmse, selected_top1 = None, None
        if result.rank <= ANCHORS:
            inferred = result.predict(observed[:, : ANCHORS + 1], anchors)[:, ANCHORS:]
            unseen = odds[:, ANCHORS:]
            anchored_rmse = rmse(inferred - unseen)
            selected_top1 = float((inferred.argmax(axis=1) == unseen.argmax(axis=1)).mean())
        singular = result.singular_values
        measurements[name] = {
            "estimated_rank": result.rank,
            "rank_ceiling": result.rank_ceiling,
            "relative_rank_sensitivity": {
                str(threshold): min(
                    int(np.count_nonzero(singular > threshold * singular[0])), result.rank_ceiling
                )
                for threshold in fit_metadata["threshold_sensitivity"]
            },
            "last_retained_singular_ratio": (
                float(singular[result.rank - 1] / singular[0]) if result.rank else None
            ),
            "next_singular_ratio": (
                float(singular[result.rank] / singular[0])
                if singular[0] and result.rank < len(singular)
                else None
            ),
            "relative_projection_operator_error": float(normalized_projection_error),
            "held_out_projection_log_odds_rmse": rmse(result.project(observed) - odds),
            "held_out_anchor_prediction_rmse": anchored_rmse,
            "held_out_mean_baseline_rmse": rmse(odds[:, ANCHORS:] - result.origin[ANCHORS:]),
            "held_out_target_columns_top1": selected_top1,
        }

    def support(top):
        overlap = np.isin(top, ids).sum(axis=1)
        return {
            "rows_with_reference_token": int(np.any(top == ids[0], axis=1).sum()),
            "rows_with_all_fixed_columns": int((overlap == SELECTED).sum()),
            "mean_observed_fixed_coordinates": float(overlap.mean()),
            "max_observed_fixed_coordinates": int(overlap.max()),
        }

    dense_metrics = measurements["dense"]
    criteria = {
        "recovered_rank_matches_configured_width": dense_metrics["estimated_rank"] == ref_width,
        "normalized_projection_error_below_1e-3": (
            dense_metrics["relative_projection_operator_error"] < 1e-3
        ),
        "heldout_anchor_rmse_below_1e-3_nats": (
            dense_metrics["held_out_anchor_prediction_rmse"] is not None
            and dense_metrics["held_out_anchor_prediction_rmse"] < 1e-3
        ),
        "underbudget_projection_rmse_above_0p1_nats": (
            measurements["short"]["held_out_projection_log_odds_rmse"] > 0.1
        ),
    }
    report = {
        "schema_version": 1,
        "track": "parameter_and_structure_recovery",
        "method": "fixed_coordinate_logprob_affine_subspace",
        "capture": capture_metadata,
        "fit": fit_metadata,
        "evaluation_seconds": time.perf_counter() - started,
        "evaluation_peak_process_rss_bytes": peak_rss_bytes(),
        "ground_truth_width_evaluator_only": ref_width,
        "ground_truth_contrast_rank_evaluator_only": ref_rank,
        "recovered_original_weights": None,
        "fixed_coordinates": SELECTED,
        "anchor_coordinates_including_reference": ANCHORS + 1,
        "held_out_target_coordinates": SELECTED - ANCHORS - 1,
        "top_k_support_control": {
            "k": TOP_K,
            "train": support(train_top_ids),
            "test": support(top_ids),
        },
        "measurements": measurements,
        "acceptance_checks": criteria,
        "commands": [
            *json.loads((args.run / "commands.json").read_text()),
            fit_metadata["command"],
            args.command,
        ],
    }
    write_json(args.run / "report.json", report)
    print(json.dumps(measurements, indent=2))


def rmse(error):
    return float(np.sqrt(np.mean(error**2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(required=True, dest="stage")
    capture_args = stages.add_parser("capture")
    capture_args.add_argument("--model", type=Path, required=True)
    capture_args.add_argument("--model-id", required=True)
    capture_args.add_argument("--revision", required=True)
    capture_args.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    capture_args.add_argument("--batch-size", type=int, default=32)
    capture_args.add_argument("--seed", type=int, default=7115)
    capture_args.set_defaults(action=capture)
    fit_args = stages.add_parser("fit")
    fit_args.add_argument("--observations", type=Path, required=True)
    fit_args.set_defaults(action=fit)
    test_args = stages.add_parser("test")
    test_args.add_argument("--model", type=Path, required=True)
    test_args.add_argument("--fit-dir", type=Path, required=True)
    test_args.set_defaults(action=test_fit)
    for subparser in (capture_args, fit_args, test_args):
        subparser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "capture" and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    args.command = [sys.executable, *sys.argv]
    args.action(args)
    if args.stage == "capture":
        write_json(args.run / "commands.json", [args.command])


if __name__ == "__main__":
    main()
