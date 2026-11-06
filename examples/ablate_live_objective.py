#!/usr/bin/env python3
"""Replay two live training snapshots without querying or accessing final observations."""

from __future__ import annotations

import argparse
import platform
import resource
import sys
import time
from pathlib import Path

import torch

from rephix.io import write_json
from rephix.session import Session
from rephix.session_candidate import ByteCandidate
from rephix.session_evaluation import evaluate
from rephix.session_store import digest, object_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose an unused output path")
    torch.set_num_threads(2)
    with Session(args.session) as session:
        state = session.state
        source = state["history"]
        if state["scope"] != "agent_combined" or len(source) != 2:
            raise ValueError("supply the measured two-round controlled live session")
        validated = session._rows("validation")
        validation = [
            (
                state["prompts"][ref["prompt_id"]],
                session.journal.evidence(ref["id"], ref["sha256"]),
            )
            for ref in validated
        ]
        config = state["candidates"]["byte-gru"] | {"device": "cpu"}
        settings = state["round_config"]
        examples = [session._examples(entry["dataset"]["members"]) for entry in source]
        report = {
            "schema_version": 1,
            "argv": [sys.executable, *sys.argv],
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "validation_evidence_sha256": object_hash(validated),
            "round_dataset_sha256": [entry["dataset"]["sha256"] for entry in source],
            "new_target_queries": 0,
            "candidate": config,
            "arms": {},
        }
        for objective in ("byte_ce", "segment_balanced_ce"):
            print(objective, flush=True)
            candidate = ByteCandidate(
                config if objective == "byte_ce" else config | {"objective": objective}
            )
            rounds = []
            for rows, entry in zip(examples, source, strict=True):
                started = time.monotonic()
                training = candidate.train(rows, settings["updates"], settings["byte_cap"])
                metrics = evaluate(candidate, validation, settings["byte_cap"])
                rounds.append(
                    {
                        "round": entry["number"],
                        "training": training,
                        "seconds_including_validation": time.monotonic() - started,
                        "checkpoint_sha256": digest(candidate.checkpoint()),
                        "validation": {
                            key: value for key, value in metrics.items() if key != "per_prompt"
                        },
                    }
                )
            report["arms"][objective] = rounds
        report["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(args.output, report)


if __name__ == "__main__":
    main()
