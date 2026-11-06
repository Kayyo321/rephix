#!/usr/bin/env python3
"""Measured cross-family text sessions and deterministic process-crash comparison."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from rephix.io import file_sha256, write_json
from rephix.session import Session
from rephix.session_store import object_hash
from rephix.session_targets import target_from_config

MODELS = {
    "qwen": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "smol": ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
}


def _run(logs, label, command, *, expected=0):
    started = time.monotonic()
    stdout, stderr = (logs / f"{label}.{suffix}" for suffix in ("json", "stderr"))
    with stdout.open("w") as output, stderr.open("w") as errors:
        with subprocess.Popen(command, stdout=output, stderr=errors) as process:
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
    if process.returncode != expected:
        raise RuntimeError(f"{label} returned {process.returncode}. See {stderr}")
    return {
        "argv": command,
        "status": process.returncode,
        "seconds": time.monotonic() - started,
        "peak_rss_kib": usage.ru_maxrss,
    }


def _compact(metrics):
    return {key: metrics[key] for key in ("teacher_forced", "free_running", "task_success")}


def _extract(directory, *, include_final):
    with Session(directory) as session:
        report = session.inspect(verify=True, history=True)
        partition_usage = {}
        canonical = {}
        for evidence_id, ref in session.state["evidence"].items():
            part = ref["partition"]
            obs = session.journal.evidence(evidence_id, ref["sha256"])
            usage = partition_usage.setdefault(
                part,
                {
                    "queries": 0,
                    "native_generated_tokens": 0,
                    "completion_bytes": 0,
                    "eligible_at_cap": 0,
                    "below_cap": 0,
                },
            )
            usage["queries"] += 1
            usage["native_generated_tokens"] += obs["usage"].get("output_tokens", 0)
            byte_count = len(obs["completion"].encode())
            cap = report["round_config"]["byte_cap"]
            usage["completion_bytes"] += byte_count
            usage["eligible_at_cap"] += min(byte_count, cap)
            usage["below_cap"] += byte_count < cap
            canonical[ref["prompt_id"]] = {
                key: obs[key] for key in ("completion", "status", "events", "usage")
            }
        checkpoints = [
            item["detail"] for item in report["journal"] if item["action"] == "training_checkpoint"
        ]
        rounds = []
        for entry in report["history"]:
            item = {
                "number": entry["number"],
                "selected": entry["selected"],
                "dataset_sha256": entry["dataset"]["sha256"],
                "member_prompt_ids": [row["prompt_id"] for row in entry["dataset"]["members"]],
                "validation": {
                    name: _compact(value["metrics"]) for name, value in entry["evaluations"].items()
                },
                "checkpoint_sha256": {
                    name: value["sha256"] for name, value in entry["evaluations"].items()
                },
            }
            rounds.append(item)
        result = {
            "settings": report["round_config"],
            "priors": report["priors"],
            "candidates": report["candidates"],
            "active_queries": report["query_count"],
            "epoch_events": report["observed_epochs"],
            "checkpoints": checkpoints,
            "rounds": rounds,
            "usage": partition_usage,
            "response_and_event_hash": object_hash(canonical),
            "journal_actions": [entry["action"] for entry in report["journal"]],
            "best_sha256": report["best"]["sha256"],
            "best_round": report["best"]["round"],
            "best_candidate": report["best"]["candidate"],
        }
        if include_final:
            final = report["finals"]["1"]
            result["final"] = {
                "metrics": _compact(final["metrics"]),
                "examples": final["metrics"]["per_prompt"][:3],
                "manifest_sha256": final["prompt_manifest_sha256"],
                "locked_sha256": final["locked"]["sha256"],
                "evidence_sha256": object_hash(final["evidence"]),
            }
        return result


def crash(directory, target_path, stage):
    import torch

    torch.set_num_threads(2)
    target = target_from_config(json.loads(target_path.read_text()))
    with Session(directory) as session:
        session.run(
            target,
            max_rounds=3,
            max_queries=51,
            checkpoint_updates=24,
            hook=lambda point: os.kill(os.getpid(), signal.SIGKILL) if point == stage else None,
        )
    raise RuntimeError("crash hook not reached")


def experiment(args):
    import torch
    import transformers

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = {
        "schema_version": 1,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "argv": [sys.executable, *sys.argv],
        "cpu": platform.processor(),
        "models": {},
        "arms": {},
    }
    if args.device == "cuda":
        results["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
        }
    cli = [sys.executable, "-m", "rephix", "--threads", "2", "session"]
    arms = [
        ("qwen-active-17", "qwen", "active", 17, "cumulative", True),
        ("qwen-uninterrupted-17", "qwen", "active", 17, "cumulative", False),
        ("qwen-random-17", "qwen", "random", 17, "cumulative", False),
        ("qwen-active-42", "qwen", "active", 42, "cumulative", False),
        ("qwen-random-42", "qwen", "random", 42, "cumulative", False),
        ("qwen-recent-17", "qwen", "active", 17, "recent", False),
        ("smol-active-17", "smol", "active", 17, "cumulative", False),
        ("gpt2-active-17", "gpt2", "active", 17, "cumulative", False),
    ]
    for label, model, policy, seed, replay, interrupt in arms[:2] if args.resume_only else arms:
        print(label, flush=True)
        folder = root / label
        folder.mkdir()
        directory = folder / "session"
        spec_path = folder / "spec.json"
        target_path = folder / "target.json"
        model_path = getattr(args, model).resolve()
        model_name, revision = MODELS[model]
        if model not in results["models"]:
            results["models"][model] = {
                "repository": model_name,
                "revision": revision,
                "precision": "float32",
                "safetensors_sha256": file_sha256(model_path / "model.safetensors"),
            }
        calls = [
            _run(
                folder,
                "plan",
                [
                    sys.executable,
                    str(Path(__file__).with_name("session_prompt_plan.py")),
                    "--mode",
                    "text",
                    "--policy",
                    policy,
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--replay",
                    replay,
                    "--output",
                    str(spec_path),
                ],
            )
        ]
        write_json(
            target_path,
            {
                "kind": "hf",
                "model": str(model_path),
                "revision": revision,
                "receipts": str(folder / "receipts.sqlite"),
                "epoch": "text-reference-1",
                "max_new_tokens": 24,
                "device": args.device,
                "dtype": "float32",
            },
        )
        calls.append(
            _run(folder, "create", [*cli, "create", str(directory), "--spec", str(spec_path)])
        )
        run = [
            *cli,
            "resume",
            str(directory),
            "--target",
            str(target_path),
            "--max-rounds",
            "3",
            "--max-queries",
            "51",
            "--checkpoint-updates",
            "24",
        ]
        if interrupt:
            for stage in (
                "after_submit",
                "after_snapshot",
                "after_checkpoint",
                "before_checkpoint_commit",
            ):
                calls.append(
                    _run(
                        folder,
                        stage,
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "_crash",
                            str(directory),
                            str(target_path),
                            stage,
                        ],
                        expected=-signal.SIGKILL,
                    )
                )
            calls.append(_run(folder, "checkpoint-stop", [*run, "--stop-after-checkpoint"]))
            calls.append(_run(folder, "pause", [*cli, "pause", str(directory)]))
        calls.append(_run(folder, "resume", run))
        do_final = label in {"qwen-active-17", "smol-active-17", "gpt2-active-17"}
        if do_final:
            calls.append(
                _run(
                    folder,
                    "evaluate",
                    [
                        *cli,
                        "evaluate",
                        str(directory),
                        "--target",
                        str(target_path),
                        "--max-queries",
                        "66",
                    ],
                )
            )
            exported = folder / "export"
            calls.append(
                _run(
                    folder,
                    "export",
                    [
                        *cli,
                        "export",
                        str(directory),
                        "--cohort",
                        "1",
                        "--output",
                        str(exported),
                    ],
                )
            )
            calls.append(
                _run(
                    folder,
                    "generate",
                    [
                        *cli,
                        "generate",
                        str(exported),
                        "--max-bytes",
                        "24",
                        "--prompt",
                        "Explain why some leaves change color in autumn.",
                    ],
                )
            )
        report = _extract(directory, include_final=do_final)
        results["arms"][label] = report | {"commands": calls, "spec_sha256": file_sha256(spec_path)}
        write_json(root / "results.json", results)

    resumed = results["arms"]["qwen-active-17"]
    uninterrupted = results["arms"]["qwen-uninterrupted-17"]
    parity = {
        "per_round_checkpoint_bytes": (
            [row["checkpoint_sha256"] for row in resumed["rounds"]]
            == [row["checkpoint_sha256"] for row in uninterrupted["rounds"]]
        ),
        "per_round_validation": (
            [row["validation"] for row in resumed["rounds"]]
            == [row["validation"] for row in uninterrupted["rounds"]]
        ),
        "acquired_prompt_order": (
            [row["member_prompt_ids"] for row in resumed["rounds"]]
            == [row["member_prompt_ids"] for row in uninterrupted["rounds"]]
        ),
        "all_recovered_checkpoint_sha256": resumed["best_sha256"] == uninterrupted["best_sha256"],
    }
    if not all(parity.values()):
        raise RuntimeError(f"resumed experiment diverged: {parity}")
    results["process_restart_parity"] = parity
    write_json(root / "results.json", results)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_crash":
        crash(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        parser.add_argument(
            "--resume-only", action="store_true", help="run just the Qwen crash comparison"
        )
        for model in MODELS:
            parser.add_argument(f"--{model}", type=Path, required=True)
        experiment(parser.parse_args())
