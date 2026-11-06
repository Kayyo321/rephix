#!/usr/bin/env python3
"""Collect, calibrate, and test a blinded open-set behavior fingerprint."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

from rephix.identification import (
    ABSOLUTE,
    CONTRASTIVE,
    BehaviorObservation,
    FingerprintProfile,
    calibrate_fingerprints,
    identify,
)
from rephix.io import file_sha256, write_json

MODELS = {
    "qwen-instruct": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "family": "Qwen",
        "architecture": "qwen2",
    },
    "smollm2-135m-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "family": "SmolLM2",
        "architecture": "llama",
    },
    "gpt2": {
        "repo": "openai-community/gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "family": "GPT-2",
        "architecture": "gpt2",
    },
    "qwen2.5-0.5b-base": {
        "repo": "Qwen/Qwen2.5-0.5B",
        "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "family": "Qwen",
        "architecture": "qwen2",
    },
    "smollm2-135m-base": {
        "repo": "HuggingFaceTB/SmolLM2-135M",
        "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "family": "SmolLM2",
        "architecture": "llama",
    },
    "distilgpt2": {
        "repo": "distilbert/distilgpt2",
        "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
        "family": "GPT-2",
        "architecture": "gpt2",
    },
    "pythia-70m": {
        "repo": "EleutherAI/pythia-70m",
        "revision": "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
        "family": "Pythia",
        "architecture": "gpt_neox",
    },
    "qwen2-0.5b-instruct": {
        "repo": "Qwen/Qwen2-0.5B-Instruct",
        "revision": "c540970f9e29518b1d8f06ab8b24cba66ad77b6d",
        "family": "Qwen",
        "architecture": "qwen2",
    },
    "smollm2-360m-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "revision": "a10cc1512eabd3dde888204e902eca88bddb4951",
        "family": "SmolLM2",
        "architecture": "llama",
    },
    "gpt2-medium": {
        "repo": "openai-community/gpt2-medium",
        "revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "family": "GPT-2",
        "architecture": "gpt2",
    },
    "gpt-neo-125m": {
        "repo": "EleutherAI/gpt-neo-125m",
        "revision": "21def0189f5705e2521767faed922f1f15e7d7db",
        "family": "GPT-Neo",
        "architecture": "gpt_neo",
    },
}
MAX_NEW_TOKENS = 24
ALPHA = 0.20
SYSTEM = {
    "development": "Open the reply with 'Notebook entry:' and express the answer as short fragments.",
    "holdout": "Keep the response concise. Begin with 'Lab note:' and use no bullet points.",
}
WRAPPER = {
    "development": "Task transcript\nQuestion:\n{}\nAssistant's reply:\n",
    "holdout": "A person asked the assistant:\n<<<\n{}\n>>>\nThe assistant replied:",
}
POPULATION = (
    "Same three enrolled checkpoints, exact fixed probes, same rendering, 24 new token cap, "
    "temperature 0.70, top-p 0.90, seeds 4100-4111. Calibration units are whole panels. "
    "Held-out sampling seeds test this population; checkpoint and deployment shifts are "
    "outside it."
)


def plan_runs(phase):
    rows = []

    def add(model, cohort, *, seed=0, temperature=0, top_p=1.0, render="default"):
        rows.append(
            {
                "id": f"r{len(rows):03d}",
                "model": model,
                "cohort": cohort,
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "render": render,
            }
        )

    enrollment = ("qwen-instruct", "smollm2-135m-instruct", "gpt2")
    if phase == "development":
        variants = ("qwen2.5-0.5b-base", "smollm2-135m-base", "distilgpt2")
        unfamiliar = "pythia-70m"
        start = 6200
    else:
        variants = ("qwen2-0.5b-instruct", "smollm2-360m-instruct", "gpt2-medium")
        unfamiliar = "gpt-neo-125m"
        start = 8300
    for model in enrollment:
        add(model, "enrollment")
        for seed in range(4100, 4112):
            add(model, "calibration", seed=seed, temperature=0.7, top_p=0.9)
        for seed in range(start, start + 8):
            add(model, "matched_sampling", seed=seed, temperature=0.7, top_p=0.9)
        add(model, "greedy_repeat")
        for seed in range(start, start + 3):
            add(model, "high_temperature", seed=seed, temperature=1.15, top_p=0.98)
        add(model, "system_instruction", render="system")
        add(model, "agent_wrapper", render="wrapper")
    for model in variants:
        add(model, "held_out_variant")
        for seed in range(start, start + 3):
            add(model, "held_out_variant", seed=seed, temperature=0.7, top_p=0.9)
    add(unfamiliar, "unfamiliar_family")
    for seed in range(start, start + 8):
        add(unfamiliar, "unfamiliar_family", seed=seed, temperature=0.7, top_p=0.9)
    return rows


def rendered_prompt(tokenizer, text, mode, phase):
    if mode == "wrapper":
        return WRAPPER[phase].format(text)
    messages = [{"role": "user", "content": text}]
    if mode == "system":
        messages.insert(0, {"role": "system", "content": SYSTEM[phase]})
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM[phase]}\n\n{text}" if mode == "system" else text


def collect(args, probes, plan):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    observed = {}
    run_seconds = {}
    started = time.perf_counter()
    for model_name in dict.fromkeys(row["model"] for row in plan):
        path = args.qwen if model_name == "qwen-instruct" else args.models / model_name
        model_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = (
            AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float32,
            )
            .to(args.device)
            .eval()
        )
        batches = {
            mode: tokenizer(
                [rendered_prompt(tokenizer, text, mode, args.phase) for text in probes.values()],
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            ).to(args.device)
            for mode in ("default", "system", "wrapper")
        }
        with torch.inference_mode():
            for row in plan:
                if row["model"] != model_name:
                    continue
                torch.manual_seed(row["seed"])
                do_sample = row["temperature"] > 0
                generation = GenerationConfig(
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=do_sample,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **(
                        {"temperature": row["temperature"], "top_p": row["top_p"], "top_k": 0}
                        if do_sample
                        else {}
                    ),
                )
                batch = batches[row["render"]]
                output = model.generate(**batch, generation_config=generation)
                decoded = tokenizer.batch_decode(
                    output[:, batch["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                observed[row["id"]] = dict(zip(probes, decoded, strict=True))
        run_seconds[model_name] = time.perf_counter() - model_started
        print(f"{model_name}: {run_seconds[model_name]:.1f} s", file=sys.stderr, flush=True)
        del model, tokenizer, output, batch, batches
        if args.device == "cuda":
            torch.cuda.empty_cache()
    return observed, {
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "dtype": "float32",
        "threads": args.threads,
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "seconds": time.perf_counter() - started,
        "seconds_by_model": run_seconds,
    }


def measure(probes, plan, observed, output, method, locked_profile=None):
    references = {}
    calibration = defaultdict(list)
    for row in plan:
        family = MODELS[row["model"]]["family"]
        if row["cohort"] == "enrollment":
            references[family] = observed[row["id"]]
        if row["cohort"] == "calibration":
            calibration[family].append(observed[row["id"]])
    profile = calibrate_fingerprints(
        probes, references, calibration, population=POPULATION, alpha=ALPHA, method=method
    )
    if locked_profile is not None and profile.to_dict() != locked_profile.to_dict():
        raise ValueError("repeated enrollment/calibration did not reproduce the locked profile")
    write_json(output / "profile.json", profile.to_dict())
    # Freeze the profile before any held-out responses enter the scoring function.
    profile_sha256 = file_sha256(output / "profile.json")

    def observations(rows):
        return [
            BehaviorObservation(probe, text, segment)
            for segment, panel in rows
            for probe, text in panel.items()
        ]

    # Identify each panel by observed text and a fixed probe ID only. The
    # evaluator joins target labels only after the identifier returns.
    predictions = {
        row["id"]: identify(observations([("single", observed[row["id"]])]), profile)
        for row in plan
        if row["cohort"] not in {"calibration", "enrollment"}
    }
    write_json(output / "predictions.json", predictions)

    scores = defaultdict(
        lambda: dict(
            panels=0,
            accepted=0,
            wrong_family=0,
            compatible_truth=0,
            correct=0,
            abstained=0,
            raw_top_family_correct=0,
        )
    )
    matrix = []
    for row in plan:
        if row["id"] not in predictions:
            continue
        model_family = MODELS[row["model"]]["family"]
        truth = model_family if model_family in references else "unknown"
        report = predictions[row["id"]]["lineage"]
        segment = report["segments"][0]
        guess = report["outcome"]
        first = segment["hypotheses"][0]["family"]
        tally = scores[row["cohort"]]
        tally["panels"] += 1
        tally["accepted"] += guess != "unknown"
        tally["wrong_family"] += guess not in {"unknown", truth}
        tally["compatible_truth"] += truth in segment["compatible_families"]
        tally["correct"] += guess == truth
        tally["abstained"] += guess == "unknown"
        tally["raw_top_family_correct"] += first == truth
        matrix.append(
            {
                **row,
                "truth": truth,
                "prediction": guess,
                "raw_top_family": first,
                "compatible": segment["compatible_families"],
                "ranks": {
                    item["family"]: item["compatibility_p"] for item in segment["hypotheses"]
                },
            }
        )

    greedy = {row["model"]: observed[row["id"]] for row in plan if row["cohort"] == "greedy_repeat"}
    unfamiliar = next(observed[row["id"]] for row in plan if row["cohort"] == "unfamiliar_family")
    qwen, gpt2 = greedy["qwen-instruct"], greedy["gpt2"]
    cases = {
        "known_backend_change": [("period-1", qwen), ("period-2", gpt2)],
        "unfamiliar_backend_change": [("period-1", qwen), ("period-2", unfamiliar)],
        "unsegmented_mixture": [
            ("period-1", {p: (qwen if i % 2 == 0 else gpt2)[p] for i, p in enumerate(probes)})
        ],
    }
    routing = {
        case: {
            "reported": result["lineage"]["outcome"],
            "segments": [segment["outcome"] for segment in result["lineage"]["segments"]],
        }
        for case, rows in cases.items()
        for result in [identify(observations(rows), profile)]
    }
    return {
        "profile_sha256": profile_sha256,
        "cohorts": scores,
        "routing": routing,
        "predictions": matrix,
        "fit": {
            "enrollment": len(references),
            "calibration_panels_per_lineage": {key: len(v) for key, v in calibration.items()},
            "alpha": ALPHA,
            "reference_probes": len(probes),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--qwen", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--phase", choices=["development", "holdout"], default="holdout")
    parser.add_argument(
        "--locked-profile", type=Path, help="Frozen contrastive development profile"
    )
    args = parser.parse_args()
    if args.phase == "holdout" and args.locked_profile is None:
        parser.error("--phase holdout requires the locked contrastive development profile")
    if args.threads < 1:
        parser.error("--threads must be positive")

    probes_path = Path(__file__).with_name("identification-probes.json")
    probes = json.loads(probes_path.read_text())
    plan = plan_runs(args.phase)
    method = ABSOLUTE if args.phase == "development" else CONTRASTIVE
    locked_profile = (
        FingerprintProfile.from_dict(json.loads(args.locked_profile.read_text()))
        if args.locked_profile is not None
        else None
    )
    if locked_profile is not None and (
        locked_profile.method != method or locked_profile.probes != probes
    ):
        parser.error("locked profile method and probe text must match the experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    models = {key: MODELS[key] for key in dict.fromkeys(row["model"] for row in plan)}
    model_files = {}
    for key in models:
        root = args.qwen if key == "qwen-instruct" else args.models / key
        model_files[key] = {
            file.name: file_sha256(file) for file in sorted(root.iterdir()) if file.is_file()
        }
    write_json(
        args.output / "evaluator-plan.json",
        {
            "models": models,
            "model_files_sha256": model_files,
            "runs": plan,
            "probes_sha256": file_sha256(probes_path),
            "max_new_tokens": MAX_NEW_TOKENS,
            "calibration_population": POPULATION,
            "alpha": ALPHA,
            "system": SYSTEM[args.phase],
            "wrapper": WRAPPER[args.phase],
            "phase": args.phase,
            "method": method,
            "locked_profile_sha256": file_sha256(args.locked_profile)
            if args.locked_profile is not None
            else None,
        },
    )
    observed, hardware = collect(args, probes, plan)
    observed_path = args.output / "observed.json"
    write_json(observed_path, observed)
    summary = {
        "schema_version": 1,
        "phase": args.phase,
        "method": method,
        **measure(probes, plan, observed, args.output, method, locked_profile),
        "observed_sha256": file_sha256(observed_path),
        "evaluator_plan_sha256": file_sha256(args.output / "evaluator-plan.json"),
        "probes_sha256": file_sha256(probes_path),
        "reference_weight_sha256": {
            key: files["model.safetensors"] for key, files in model_files.items()
        },
        "queries": len(plan) * len(probes),
        "hardware_and_runtime": hardware,
    }
    if args.phase == "development":
        summary["contrastive_reanalysis"] = measure(
            probes, plan, observed, args.output / "contrastive", CONTRASTIVE
        )
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary["cohorts"], indent=2))
    print(json.dumps(summary["routing"], indent=2))


if __name__ == "__main__":
    main()
