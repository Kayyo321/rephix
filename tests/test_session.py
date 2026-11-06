import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from rephix.session import Session
from rephix.session_targets import _ReceiptTarget

try:
    import torch
except ImportError:
    torch = None


def _row(key, group, partition, text):
    return {"id": key, "group": group, "partition": partition, "prompt": text}


def _spec():
    return {
        "schema_version": 1,
        "prompts": [
            _row("one", "pool-1", "pool", "Who?"),
            _row("two", "pool-2", "pool", "What?"),
            _row("three", "pool-3", "pool", "When?"),
            _row("val", "val-1", "validation", "Where?"),
            _row("test", "test-1", "test", "Why?"),
        ],
        "candidates": {
            "bytes": {"width": 8, "batch": 4, "context": 16, "seed": 13},
            "conv": {"architecture": "conv", "width": 8, "batch": 4, "context": 16},
        },
        "round": {"acquire": 1, "updates": 2, "byte_cap": 8, "policy": "active"},
    }


class Echo(_ReceiptTarget):
    def _run(self, request):
        events = [
            {"sequence": 0, "source": "user", "kind": "message", **request["messages"][0]},
            {
                "sequence": 1,
                "source": "model",
                "kind": "message",
                "role": "assistant",
                "content": "Hello",
            },
        ]
        return self._result(
            request, events, "Hello", usage={"finish_reason": "stop"}, capabilities={}
        )


class UnknownRemote:
    retry_missing = False

    def __init__(self):
        self.calls = 0

    def submit(self, request):
        self.calls += 1
        return None

    def reconcile(self, request):
        return None


class SessionJournalTests(unittest.TestCase):
    def test_query_ceiling_survives_restart_without_flags(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            path = folder / "session"
            spec = _spec()
            spec["prompts"].append(_row("val2", "val-2", "validation", "Why now?"))
            target = Echo(receipts=folder / "receipts.sqlite", epoch="x")
            with Session(path, spec) as session:
                report = session.run(target, max_queries=1)
                self.assertEqual(report["remaining_active_queries"], 0)
            with Session(path) as session:
                report = session.run(target)
                self.assertEqual((report["query_count"], report["stop"]), (1, "query_budget"))
                report = session.run(target, max_queries=2)
                self.assertEqual((report["query_count"], report["stop"]), (2, "query_budget"))

    def test_validation_and_test_groups_reserved_and_passive_held_out_quarantined(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            bad = _spec()
            bad["prompts"][0]["group"] = "test-1"
            with self.assertRaisesRegex(ValueError, "crosses"):
                Session(folder / "invalid", bad)
            self.assertFalse((folder / "invalid").exists())
            target = Echo(receipts=folder / "target.sqlite", epoch="old")
            for index, text in enumerate(("Who?", "Why?", "Other")):
                target.submit(
                    {
                        "session_id": "existing",
                        "request_id": str(index),
                        "mode": "passive",
                        "messages": [{"role": "user", "content": text}],
                    }
                )
            with Session(folder / "session", _spec()) as session:
                report = session.observe(target, "example", 2)
                self.assertEqual(report["evidence_counts"]["pool"], 1)
                self.assertEqual(report["evidence_counts"]["passive_unassigned"], 1)
                self.assertEqual(report["evidence_counts"]["test"], 0)
                self.assertEqual(report["passive_cursors"]["example"], 2)
            with Session(folder / "session") as session:
                report = session.observe(target, "example", 10)
                self.assertEqual(report["evidence_counts"]["passive_unassigned"], 2)
                admitted = session.add_prompts([_row("other", "new-pool", "pool", "Other")])
                self.assertEqual(admitted["evidence_counts"]["pool"], 2)
                self.assertEqual(admitted["evidence_counts"]["passive_unassigned"], 1)
                self.assertTrue(session.inspect(verify=True)["integrity_verified"])
            with closing(sqlite3.connect(folder / "session" / "session.sqlite")) as db, db:
                db.execute(
                    "UPDATE evidence SET payload=? WHERE id LIKE 'passive:%'",
                    (b'{"edited": true}',),
                )
            with Session(folder / "session") as session:
                with self.assertRaisesRegex(ValueError, "observation integrity"):
                    session.inspect(verify=True)

    def test_no_remote_resubmission_of_uncertain_request(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "session"
            target = UnknownRemote()
            with Session(path, _spec()) as session:
                first = session.run(target, max_queries=1)
                self.assertEqual(first["stop"], "uncertain_request")
                self.assertEqual(target.calls, 1)
            with Session(path) as session:
                again = session.run(target, max_queries=1)
                self.assertEqual(target.calls, 1)
                self.assertEqual(again["pending"], first["pending"])
                self.assertEqual(again["query_count"], 1)
                resolved = session.abandon("operator could not verify remote receipt")
                self.assertIsNone(resolved["pending"])
                self.assertEqual(len(resolved["abandoned"]), 1)


@unittest.skipIf(torch is None, "optional Torch dependency unavailable")
class SessionCandidateTests(unittest.TestCase):
    def test_restart_matches_uninterrupted_and_final_is_locked_before_dispatch(self):
        import safetensors.torch

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            reports = []
            for interrupted in (True, False):
                branch = folder / str(interrupted)
                branch.mkdir()
                path = branch / "session"
                target = Echo(receipts=branch / "target.sqlite", epoch="x")
                with Session(path, _spec()) as session:
                    if interrupted:
                        first = session.run(
                            target, max_rounds=2, checkpoint_updates=1, stop_after_checkpoint=True
                        )
                        self.assertEqual(first["stop"], "checkpoint")
                        self.assertEqual(first["phase"], "training")
                        self.assertTrue(session.pause()["pause_requested"])
                with Session(path) as session:
                    report = session.run(target, max_rounds=2, checkpoint_updates=1)
                    self.assertEqual(report["round"], 2)
                    self.assertEqual(report["evidence_counts"]["test"], 0)
                    reports.append(report)
                    seen = []

                    def check_lock(point):
                        if point == "after_final_lock":
                            seen.append(point)
                            inspect = session.inspect(verify=True)
                            self.assertEqual(inspect["evidence_counts"]["test"], 0)
                            self.assertEqual(
                                inspect["finals"]["1"]["locked"]["sha256"], report["best"]["sha256"]
                            )

                    final = session.final(target, hook=check_lock)
                    self.assertEqual(seen, ["after_final_lock"])
                    self.assertEqual(final, session.final(target))
                    export = branch / "export"
                    manifest = session.export(export, cohort=1)
                    self.assertFalse(manifest["target_access"])
                    self.assertTrue(safetensors.torch.load_file(export / "weights.safetensors"))
                    fresh = _row("test-next", "new-test", "test", "How?")
                    session.add_prompts([fresh], fresh_test=True)
                    self.assertEqual(session.state["test_cohort"], 2)
                    self.assertEqual(len(session.inspect()["finals"]), 1)
                    session.add_candidate(
                        "balanced",
                        {"width": 8, "context": 16, "batch": 4, "objective": "segment_balanced_ce"},
                    )
                    refit = session.run(None, refit=True, checkpoint_updates=1)
                    self.assertEqual(refit["query_count"], report["query_count"] + 1)
                    self.assertTrue(refit["history"][-1]["refit"])
                    self.assertEqual(refit["recovery"]["balanced"]["step"], 2)
            self.assertEqual(reports[0]["best"]["sha256"], reports[1]["best"]["sha256"])
            self.assertEqual(reports[0]["best"]["metrics"], reports[1]["best"]["metrics"])

    def test_export_rejects_altered_weights(self):
        from rephix.session_candidate import ByteCandidate, candidate_config, load_export

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "export"
            candidate = ByteCandidate(candidate_config({"width": 8, "context": 16}))
            candidate.export(output)
            self.assertEqual(len(load_export(output).generate(["hi"], 3)), 1)
            manifest = output / "manifest.json"
            original = manifest.read_text()
            manifest.write_text(original.replace('"context": 16', '"context": 32'))
            with self.assertRaisesRegex(ValueError, "manifest integrity"):
                load_export(output)
            manifest.write_text(original)
            weights = output / "weights.safetensors"
            with weights.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_export(output)
