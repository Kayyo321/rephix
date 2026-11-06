#!/usr/bin/env python3
"""Validation-only Qwen schedule and architecture ablation with a fresh final cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run_text_sessions import MODELS, _extract, _run

from rephix.io import file_sha256, write_json

FRESH_TEST = {
    "logical-entailment": [
        "Explain whether 'some cats sleep' follows from 'all cats sleep' and the existence of cats.",
        "Does seeing wet pavement prove it rained? Explain your reasoning in a short paragraph.",
        "What follows if every bus stops here and vehicle 3 is a bus? Explain the inference.",
    ],
    "etymology": [
        "Explain the origin of the word 'robot' in a few clear sentences.",
        "Where does the name 'quarantine' come from? Explain the connection.",
        "Explain the origin of the English word 'alphabet' and what its parts meant.",
    ],
    "fictional-scene": [
        "Write a tiny scene in which a librarian finds a seashell inside a returned book.",
        "Write a short scene about a train conductor who notices an unexpected snowflake in June.",
        "Write a tiny scene about a clockmaker who hears a tune from a silent clock.",
    ],
    "coordinate-movement": [
        "Explain how to find the new position after moving 3 east and 2 south from (1, 4).",
        "On a grid, start at (2, 5), travel north 4 and west 2, and explain the endpoint.",
        "A rover starts at (4, 1) and travels west 3 then north 5. Describe the resulting coordinates.",
    ],
    "instrument-construction": [
        "Explain how to make a simple string telephone and why it can transmit sound.",
        "Describe how a pinhole camera forms an image using ordinary household materials.",
        "Explain how to build a simple sundial and what determines its orientation.",
    ],
}


def experiment(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    final_prompts = [
        {"id": f"fresh-{group}-{i}", "group": f"fresh-{group}", "partition": "test", "prompt": text}
        for group, prompts in FRESH_TEST.items()
        for i, text in enumerate(prompts)
    ]
    fresh = root / "fresh-final-prompts.json"
    write_json(fresh, final_prompts)
    cli = [sys.executable, "-m", "rephix", "--threads", "2", "session"]
    report = {
        "schema_version": 1,
        "argv": [sys.executable, *sys.argv],
        "final_prompt_sha256": file_sha256(fresh),
        "arms": {},
    }
    winner = None
    for name, seed, learning_rate, updates, architectures in (
        ("lr-only-17", 17, "0.001", 96, ["gru"]),
        ("short-only-17", 17, "0.003", 48, ["gru"]),
        ("joint-17", 17, "0.001", 48, ["gru"]),
        ("joint-42", 42, "0.001", 48, ["gru"]),
        ("architecture-17", 17, "0.001", 48, ["gru", "conv"]),
    ):
        print(name, flush=True)
        arm = root / name
        arm.mkdir()
        target = arm / "target.json"
        spec = arm / "spec.json"
        directory = arm / "session"
        commands = [
            _run(
                arm,
                "plan",
                [
                    sys.executable,
                    str(Path(__file__).with_name("session_prompt_plan.py")),
                    "--mode",
                    "text",
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--learning-rate",
                    learning_rate,
                    "--updates",
                    str(updates),
                    "--policy",
                    "random" if len(architectures) > 1 else "active",
                    "--output",
                    str(spec),
                    *(argument for kind in architectures for argument in ("--architecture", kind)),
                ],
            )
        ]
        plan = json.loads(spec.read_text())
        plan["prompts"] = [
            row for row in plan["prompts"] if row["partition"] != "test"
        ] + final_prompts
        write_json(spec, plan)
        write_json(
            target,
            {
                "kind": "hf",
                "model": str(args.qwen.resolve()),
                "revision": MODELS["qwen"][1],
                "receipts": str(arm / "receipts.sqlite"),
                "epoch": "text-reference-1",
                "max_new_tokens": 24,
                "device": args.device,
                "dtype": "float32",
            },
        )
        commands.append(_run(arm, "create", [*cli, "create", str(directory), "--spec", str(spec)]))
        for round_number in (1, 3):
            commands.append(
                _run(
                    arm,
                    f"resume-{round_number}",
                    [
                        *cli,
                        "resume",
                        str(directory),
                        "--target",
                        str(target),
                        "--max-rounds",
                        str(round_number),
                        "--max-queries",
                        "51",
                        "--checkpoint-updates",
                        "24",
                    ],
                )
            )
        result = _extract(directory, include_final=False)
        report["arms"][name] = result | {"commands": commands, "spec_sha256": file_sha256(spec)}
        best_metrics = result["rounds"][result["best_round"] - 1]["validation"][
            result["best_candidate"]
        ]
        score = best_metrics["teacher_forced"]["cross_entropy"]
        if winner is None or score < winner[0]:
            winner = (score, name)
        write_json(root / "results.json", report)
    score, name = winner
    lock = {
        "arm": name,
        "validation_byte_ce": score,
        "checkpoint_sha256": report["arms"][name]["best_sha256"],
        "prompt_sha256": report["final_prompt_sha256"],
    }
    write_json(root / "selection-lock.json", lock)
    arm = root / name
    command = _run(
        arm,
        "fresh-final",
        [
            *cli,
            "evaluate",
            str(arm / "session"),
            "--target",
            str(arm / "target.json"),
            "--max-queries",
            "66",
        ],
    )
    final = _extract(arm / "session", include_final=True)["final"]
    if final["locked_sha256"] != lock["checkpoint_sha256"]:
        raise RuntimeError("final used a different checkpoint from the pre-test lock")
    report["final"] = final | {"command": command, "selection_lock": lock}
    write_json(root / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    experiment(parser.parse_args())
