"""Exercise a durable live reconstruction session against two local model backends."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import selectors
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import closing, contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from session_prompt_plan import PLAN_SEED, session_spec

from rephix.io import file_sha256, write_json
from rephix.session import Session
from rephix.session_targets import HTTPTarget

QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PYTHIA_REVISION = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
OUTPUT_TOKENS = 24
CHECKPOINT_UPDATES = 16


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def crash_client(args):
    def terminate(point):
        if point == "after_submit":
            os.kill(os.getpid(), signal.SIGKILL)

    with Session(args.session) as session:
        session.run(
            HTTPTarget(**read_json(args.target)["connection"]), max_queries=1, hook=terminate
        )
    raise RuntimeError("the fault injection did not run")


class Harness:
    def __init__(self, output, threads):
        self.output = output
        self.threads = threads
        self.commands = []
        self.env = os.environ | {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
        }

    def cli(self, *parts):
        return [sys.executable, "-m", "rephix", "--threads", str(self.threads), *map(str, parts)]

    def run(self, name, command, *, expected=0):
        started = time.monotonic()
        stdout = self.output / f"{name}.json"
        stderr = self.output / f"{name}.stderr"
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            process = subprocess.run(command, stdout=out, stderr=err, env=self.env, check=False)
        self.commands.append(
            {
                "label": name,
                "argv": command,
                "exit": process.returncode,
                "seconds": time.monotonic() - started,
            }
        )
        if process.returncode != expected:
            raise RuntimeError(f"{name} exited {process.returncode}. See {stderr}")
        return read_json(stdout) if expected == 0 else None

    @contextmanager
    def server(self, label, config, port=0):
        command = self.cli("session-target-serve", "--config", config, "--port", port)
        started = time.monotonic()
        with (self.output / f"{label}.stderr").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command, env=self.env, stdout=subprocess.PIPE, stderr=stderr, text=True
            )
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    if not selector.select(timeout=60):
                        raise RuntimeError(f"{label} did not report readiness")
                ready = json.loads(process.stdout.readline())
                write_json(self.output / f"{label}.json", ready)
                yield ready["url"]
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
                self.commands.append(
                    {
                        "label": label,
                        "argv": command,
                        "exit": process.returncode,
                        "seconds": time.monotonic() - started,
                    }
                )

    def pause_at_checkpoint(self, directory, client, query_limit):
        command = self.cli(
            "session",
            "resume",
            directory,
            "--target",
            client,
            "--max-rounds",
            1,
            "--max-queries",
            query_limit,
            "--checkpoint-updates",
            CHECKPOINT_UPDATES,
        )
        started = time.monotonic()
        stdout = self.output / "graceful-pause.json"
        stderr = self.output / "graceful-pause.stderr"
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            process = subprocess.Popen(command, env=self.env, stdout=out, stderr=err)
            try:
                with closing(sqlite3.connect(directory / "session.sqlite")) as db:
                    while process.poll() is None and time.monotonic() - started < 900:
                        state = json.loads(db.execute("SELECT state FROM current").fetchone()[0])
                        if state["phase"] == "training" and state["recovery"]:
                            process.send_signal(signal.SIGTERM)
                            break
                        time.sleep(0.05)
                    else:
                        raise RuntimeError(
                            f"training did not reach a durable checkpoint. See {stderr}"
                        )
                status = process.wait(timeout=120)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        self.commands.append(
            {
                "label": "graceful-pause",
                "argv": command,
                "exit": status,
                "signal_sent_after_training_checkpoint": "SIGTERM",
                "seconds": time.monotonic() - started,
            }
        )
        if status:
            raise RuntimeError(f"graceful pause exited {status}. See {stderr}")
        stopped = read_json(stdout)
        if stopped["stop"] != "paused" or stopped["phase"] != "training":
            raise RuntimeError("process did not stop inside the training stage")
        return stopped


def request_for(row, request_id):
    return {
        "request_id": request_id,
        "session_id": "sandbox-operator",
        "messages": [{"role": "user", "content": row["prompt"]}],
    } | ({"task": row["task"]} if "task" in row else {})


def receipt(path, request):
    with closing(sqlite3.connect(path)) as db:
        rows = db.execute(
            "SELECT response, response_digest FROM receipts WHERE session_id=? AND request_id=?",
            (request["session_id"], request["request_id"]),
        ).fetchall()
    if len(rows) != 1:
        raise RuntimeError("expected exactly one durable receipt for the request")
    response, digest = rows[0]
    return json.loads(response), digest


def target_config(model, revision, epoch, receipts):
    return {
        "kind": "sandbox",
        "model": str(model),
        "revision": revision,
        "receipts": str(receipts),
        "device": "cpu",
        "dtype": "float32",
        "max_new_tokens": OUTPUT_TOKENS,
        "prompt_mode": "auto",
        "expose_probabilities": 4,
        "epoch": epoch,
    }


def report_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key not in ("per_prompt", "failed_prompt_ids")
    }


def scenario(args):
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True)
    harness = Harness(output, args.threads)
    use_cli = harness.cli
    execute = harness.run
    session_dir = output / "session"
    receipt_db = output / "server-receipts.sqlite"
    profile_path = args.identity_profile.resolve()
    spec = session_spec(
        argparse.Namespace(
            mode="tool",
            plan_seed=PLAN_SEED,
            identity_profile=profile_path,
            policy="active",
            replay="cumulative",
            seed=args.seed,
            acquire=args.acquire,
            updates=args.updates,
            byte_cap=args.byte_cap,
            width=args.width,
            context=1024,
            batch=args.batch,
            learning_rate=0.003,
            device="cpu",
            architecture=["gru"],
        )
    )
    spec_file = output / "spec.json"
    write_json(spec_file, spec)
    server1, server2 = output / "server-a.json", output / "server-b.json"
    write_json(
        server1, target_config(args.qwen.resolve(), QWEN_REVISION, "deployment-a", receipt_db)
    )
    write_json(
        server2, target_config(args.pythia.resolve(), PYTHIA_REVISION, "deployment-b", receipt_db)
    )
    models = [
        {
            "epoch": "deployment-a",
            "repo": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": QWEN_REVISION,
            "path": args.qwen.resolve(),
        },
        {
            "epoch": "deployment-b",
            "repo": "EleutherAI/pythia-70m",
            "revision": PYTHIA_REVISION,
            "path": args.pythia.resolve(),
        },
    ]
    for model in models:
        model["model_sha256"] = file_sha256(model["path"] / "model.safetensors")
        model["tokenizer_sha256"] = file_sha256(model["path"] / "tokenizer.json")
        model["path"] = str(model["path"])
    write_json(output / "evaluator-only-models.json", models)
    execute("create", use_cli("session", "create", session_dir, "--spec", spec_file))
    client = output / "client.json"
    fault_config = output / "fault-client.json"
    requests = output / "requests"
    requests.mkdir()
    tool_pool = [row for row in spec["prompts"] if row["partition"] == "pool" and "task" in row]
    validation = [row for row in spec["prompts"] if row["partition"] == "validation"]
    final_rows = [row for row in spec["prompts"] if row["partition"] == "test"]
    probes = [row for row in spec["prompts"] if "probe_id" in row]

    def passive(row, label):
        path = requests / f"{label}.json"
        write_json(path, request_for(row, label))
        return execute(
            label, use_cli("session-target-interact", "--config", client, "--request", path)
        )

    def inspect(label):
        return execute(label, use_cli("session", "inspect", session_dir, "--verify", "--history"))

    def observe(label):
        return execute(
            label, use_cli("session", "observe", session_dir, "--target", client, "--feed", "live")
        )

    def resume(label, *budget):
        return execute(
            label,
            use_cli(
                "session",
                "resume",
                session_dir,
                "--target",
                client,
                "--checkpoint-updates",
                CHECKPOINT_UPDATES,
                *budget,
            ),
        )

    with harness.server("target-process-a", server1) as url:
        port = urlsplit(url).port
        write_json(client, {"kind": "http", "url": url, "timeout": 60})
        write_json(fault_config, {"connection": {"url": url, "timeout": 60}})
        pool_receipt = passive(tool_pool[0], "passive-pool-tool")
        passive(validation[0], "passive-quarantine-validation")
        for row in probes:
            passive(row, f"passive-a-{row['probe_id']}")
        imported = observe("observe-a")
        repeated = observe("observe-a-repeat")
        counts = imported["evidence_counts"]
        if counts != repeated["evidence_counts"] or (
            counts["pool"] != 1 + len(probes)
            or counts["passive_unassigned"] != 1
            or counts["validation"] != 0
        ):
            raise RuntimeError("passive receipt import or quarantine did not behave as expected")
        if [
            event["source"] for event in pool_receipt["events"] if event["kind"] == "tool_result"
        ] != ["environment", "environment"]:
            raise RuntimeError("expected two separately attributed tool results")
        execute(
            "crash-client",
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "crash-client",
                "--session",
                str(session_dir),
                "--target",
                str(fault_config),
            ],
            expected=-signal.SIGKILL,
        )
        crashed = inspect("inspect-after-kill")
        pending = crashed["pending"]["request"]
        saved, saved_sha = receipt(receipt_db, pending)
        reconciled = resume("resume-reconcile")
        if reconciled["pending"] is not None or reconciled["query_count"] != 1:
            raise RuntimeError("the first request did not reconcile within its existing budget")
        if (saved, saved_sha) != receipt(receipt_db, pending):
            raise RuntimeError("the original receipt changed after reconciliation")
        paused = harness.pause_at_checkpoint(session_dir, client, len(validation) + args.acquire)
        before = inspect("inspect-paused")
        if before["recovery"] != paused["recovery"]:
            raise RuntimeError("the graceful checkpoint changed after process shutdown")
        first = resume("resume-round-one", "--max-rounds", 1)

    with harness.server("target-process-b", server2, port=port) as url:
        if HTTPTarget(url=url).reconcile(pending) != saved:
            raise RuntimeError("the first receipt could not be reconciled after server restart")
        for row in probes:
            passive(row, f"passive-b-{row['probe_id']}")
        observe("observe-b")
        second = resume(
            "resume-round-two",
            "--max-rounds",
            2,
            "--max-queries",
            len(validation) + 2 * args.acquire,
        )
        final = execute(
            "final",
            use_cli(
                "session",
                "evaluate",
                session_dir,
                "--target",
                client,
                "--max-queries",
                second["query_count"] + len(final_rows),
            ),
        )
        inspected = inspect("inspect-final")
        evidence = execute("evidence", use_cli("session", "evidence", session_dir))["evidence"]
        export = execute(
            "export",
            use_cli("session", "export", session_dir, "--cohort", 1, "--output", output / "export"),
        )

    with socket.socket() as test:
        test.settimeout(2)
        if test.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError("target listener still active at standalone inference")
    standalone_path = requests / "standalone.json"
    write_json(standalone_path, request_for(final_rows[0], "standalone"))
    standalone = execute(
        "standalone",
        use_cli(
            "session",
            "generate",
            output / "export",
            "--request",
            standalone_path,
            "--max-bytes",
            args.byte_cap,
            "--device",
            "cpu",
        ),
    )
    observed_saved = [
        item for item in evidence if item["observation"]["request_id"] == pending["request_id"]
    ]
    if len(observed_saved) != 1 or observed_saved[0]["observation"] != saved:
        raise RuntimeError("recovered committed observation differs from the server's receipt")
    if (
        len(final["evidence"]) != len(final_rows)
        or export["checkpoint_sha256"] != final["locked"]["sha256"]
    ):
        raise RuntimeError("reserved evidence or locked export did not verify")
    if any(item["partition"] == "test" and item["source"] != "active" for item in evidence):
        raise RuntimeError("unexpected passive evidence in final test")
    epochs = [item["epoch"] for item in inspected["observed_epochs"]]
    if epochs != ["deployment-a", "deployment-b"]:
        raise RuntimeError(f"unexpected backend epoch history: {epochs}")
    journal = inspected["journal"]
    lock = next(entry["sequence"] for entry in journal if entry["action"] == "final_locked")
    test_reservations = [
        entry["sequence"]
        for entry in journal
        if entry["action"] == "request_reserved" and entry["detail"]["partition"] == "test"
    ]
    if len(test_reservations) != len(final_rows) or min(test_reservations) <= lock:
        raise RuntimeError("final request was made before the candidate lock")
    memberships = {
        item["id"] for round_ in second["history"] for item in round_["dataset"]["members"]
    }
    if any(item["evidence_id"] in memberships for item in evidence if item["partition"] != "pool"):
        raise RuntimeError("reserved or unassigned evidence leaked into training")

    import numpy
    import torch
    import transformers

    report = {
        "schema_version": 1,
        "milestone": "C",
        "status": "demonstrated_on_controlled_loopback_agent",
        "invocation": [sys.executable, *sys.argv],
        "scope": "combined model decisions and fixed tool wrapper",
        "identity_profile_sha256": file_sha256(profile_path),
        "identity_scope": "text passthrough diagnostics; greedy CPU is a deployment shift from calibration",
        "target_details_evaluator_only": models,
        "experiment": {
            "epochs": epochs,
            "acquisition": spec["round"],
            "candidate": spec["candidates"],
            "query_count": inspected["query_count"],
            "passive_counts": {
                "pool_before_training": counts["pool"],
                "quarantined": counts["passive_unassigned"],
            },
            "final_target_epoch": epochs[-1],
            "validation_target_epoch": epochs[0],
            "final_lock_sequence": lock,
            "first_final_request_sequence": min(test_reservations),
            "training_checkpoints": [
                entry["detail"] for entry in journal if entry["action"] == "training_checkpoint"
            ],
            "event_counts": {
                kind: sum(
                    event["kind"] == kind
                    for item in evidence
                    for event in item["observation"]["events"]
                )
                for kind in sorted(
                    {event["kind"] for item in evidence for event in item["observation"]["events"]}
                )
            },
            "usage": {
                key: sum(
                    item["observation"]["usage"].get(key, int(key == "model_calls"))
                    for item in evidence
                )
                for key in (
                    "model_calls",
                    "prompt_tokens",
                    "output_tokens",
                    "decision_tokens",
                    "decision_prompt_tokens",
                )
            },
        },
        "recovery": {
            "collection_exit": -signal.SIGKILL,
            "pending_request_id": pending["request_id"],
            "target_receipt_sha256": saved_sha,
            "reconciled_observation_sha256": observed_saved[0]["sha256"],
            "receipt_survived_target_restart": True,
            "graceful_signal": "SIGTERM",
            "training_checkpoint_at_pause": paused["recovery"],
        },
        "identity_round_one": first["identity"],
        "identity_round_two": second["identity"],
        "validation": [
            {
                "round": item["number"],
                "dataset_sha256": item["dataset"]["sha256"],
                "observations": len(item["dataset"]["members"]),
                "metrics": report_metrics(item["evaluations"]["byte-gru"]["metrics"]),
            }
            for item in second["history"]
        ],
        "final": {
            "locked_candidate_round": final["locked"]["round"],
            "locked_checkpoint_sha256": final["locked"]["sha256"],
            "prompt_manifest_sha256": final["prompt_manifest_sha256"],
            "metrics": final["metrics"],
        },
        "standalone": {"target_listener_closed": True, "export": export, "generation": standalone},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "threads": args.threads,
            "device": "cpu",
            "peak_child_rss_bytes_linux": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            * 1024,
            "wall_seconds": time.monotonic() - started,
        },
        "artifacts": {
            name: file_sha256(output / name)
            for name in (
                "spec.json",
                "evidence.json",
                "inspect-final.json",
                "final.json",
                "server-receipts.sqlite",
                "export/weights.safetensors",
            )
        },
        "commands": harness.commands,
    }
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {"report": str(output / "report.json"), "metrics": report_metrics(final["metrics"])},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    for name in ("output", "qwen", "pythia", "identity-profile"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--updates", type=int, default=64)
    run.add_argument("--acquire", type=int, default=12)
    run.add_argument("--byte-cap", type=int, default=64)
    run.add_argument("--width", type=int, default=32)
    run.add_argument("--batch", type=int, default=16)
    run.set_defaults(func=scenario)
    fault = commands.add_parser("crash-client")
    fault.add_argument("--session", type=Path, required=True)
    fault.add_argument("--target", type=Path, required=True)
    fault.set_defaults(func=crash_client)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
