import json
import sqlite3
import threading
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from rephix.session_evaluation import evaluate, examples_for_observation
from rephix.session_targets import (
    HTTPTarget,
    SandboxAgentTarget,
    make_observation,
    sandbox_execute,
    serve_target,
    training_segments,
    validate_observation,
)


class FakeBackend(SandboxAgentTarget):
    def __init__(self, receipts, epoch="opaque-1"):
        super().__init__(model="fixture", receipts=receipts, epoch=epoch)
        self.calls = 0

    def _choose(self, messages, choices):
        self.calls += 1
        prompt = messages[-1]["content"]
        desired = "subtract" if "Choose tool 1" in prompt else "double"
        choice = next(line[0] for line in prompt.splitlines() if f'"name":"{desired}"' in line)
        return choices.index(choice), len(prompt)

    def _generate(self, messages, events):
        return "8", {"output_tokens": 1, "finish_reason": "stop"}


class TargetSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "receipts.sqlite"
        self.backend = FakeBackend(self.path)

    def request(self, request_id="r1", mode="active"):
        return {
            "request_id": request_id,
            "session_id": "example",
            "mode": mode,
            "messages": [{"role": "user", "content": "Subtract 3 from 7 and double."}],
            "task": {"a": 7, "b": 3},
        }

    def test_model_actions_and_environment_results_have_separate_supervision(self):
        request = self.request()
        response = self.backend.submit(request)
        validate_observation(response, request)
        self.assertEqual(response["completion"], "8")
        self.assertEqual(response["usage"]["finish_reason"], "stop")
        self.assertEqual(response["capabilities"]["reasoning"], "unavailable")
        events = response["events"]
        self.assertEqual(
            [
                (event["name"], event["result"])
                for event in events
                if event["kind"] == "tool_result"
            ],
            [("subtract", 4), ("double", 8)],
        )
        segments = training_segments(response)
        self.assertEqual(len(segments), 3)
        self.assertEqual(
            json.loads(segments[0][1]), {"name": "subtract", "arguments": {"a": 7, "b": 3}}
        )
        self.assertIn('"result":4', segments[1][0])
        self.assertNotIn('"result"', segments[1][1])
        self.assertEqual(segments[2][1], "8")
        with self.assertRaisesRegex(ValueError, "bounded integers"):
            sandbox_execute("double", {"value": True})
        next(event for event in events if event["kind"] == "tool_result")["source"] = "model"
        with self.assertRaisesRegex(ValueError, "environment input"):
            validate_observation(response, request)

    def test_receipts_preserve_request_identity_and_backend_epoch_across_restarts(self):
        request = self.request()
        self.assertIsNone(self.backend.reconcile(request))
        original = self.backend.submit(request)
        next_backend = FakeBackend(self.path, epoch="opaque-2")
        self.assertEqual(next_backend.submit(request), original)
        self.assertEqual(next_backend.calls, 0)
        self.assertEqual(next_backend.submit(self.request("r2"))["target_epoch"], "opaque-2")
        changed_request = {**request, "task": {"a": 0, "b": 0}}
        with self.assertRaisesRegex(ValueError, "reused"):
            next_backend.reconcile(changed_request)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE receipts SET response = 'damaged' WHERE request_id = 'r1'")
        with self.assertRaisesRegex(ValueError, "checksum"):
            next_backend.reconcile(request)

    def test_http_active_passive_and_reconcile(self):
        server = serve_target(self.backend)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server_thread.join)
        self.addCleanup(server.shutdown)
        client = HTTPTarget(url=f"http://127.0.0.1:{server.server_port}")
        active = self.request()
        self.assertFalse(client.retry_missing)
        self.assertIsNone(client.reconcile(active))
        self.assertEqual(client.submit(active), client.reconcile(active))
        first = client.submit(self.request("p1", "passive"))
        client.submit(self.request("r2"))
        second = client.submit(self.request("p2", "passive"))
        observed = list(client.iter_passive())
        self.assertEqual(observed, [(2, first), (4, second)])
        self.assertEqual(list(client.iter_passive(after=2)), [(4, second)])
        self.assertEqual(self.backend.calls, 8)

    def test_remote_timeout_recovers_committed_receipt_without_resubmitting(self):
        running = threading.Event()
        release = threading.Event()
        committed = threading.Event()
        original = self.backend.submit

        def submit(request):
            running.set()
            release.wait(2)
            receipt = original(request)
            committed.set()
            return receipt

        self.backend.submit = submit
        server = serve_target(self.backend)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server_thread.join)
        self.addCleanup(server.shutdown)
        self.addCleanup(release.set)
        url = f"http://127.0.0.1:{server.server_port}"
        client = HTTPTarget(url=url, timeout=0.1)
        request = self.request()
        with self.assertRaises(TimeoutError):
            client.submit(request)
        self.assertTrue(running.is_set())
        # No receipt yet is ambiguous while the original request is in flight.
        reliable_client = HTTPTarget(url=url)
        self.assertIsNone(reliable_client.reconcile(request))
        self.assertFalse(reliable_client.retry_missing)
        release.set()
        self.assertTrue(committed.wait(2))
        recovered = reliable_client.reconcile(request)
        self.assertEqual(recovered["completion"], "8")
        self.assertEqual(self.backend.calls, 2)

    def test_closed_loop_evaluation_uses_the_candidates_own_first_tool_result(self):
        request = self.request()
        target = self.backend.submit(request)
        target["usage"]["finish_reason"] = "length"

        class Candidate:
            def teacher_forced(self, examples, byte_cap):
                return {"positions": len(examples)}

            def choose(self, context, outputs):
                stage = "double" if '"result":10' in context else "add"
                return next(
                    i for i, output in enumerate(outputs) if json.loads(output)["name"] == stage
                )

            def generate(self, contexts, byte_cap):
                return ["20" if '"result":20' in context else "8" for context in contexts]

        expected = {
            "tools": [
                {"name": "subtract", "arguments": {"a": 7, "b": 3}},
                {"name": "double", "arguments": {"value": 4}},
            ],
            "result": 8,
        }
        row = {
            "id": "validation-1",
            "prompt": request["messages"][0]["content"],
            "task": request["task"],
            "expect": expected,
        }
        report = evaluate(Candidate(), [(row, target)], 32)
        self.assertEqual(
            [ex["truncated"] for ex in examples_for_observation(target)], [False, False, True]
        )
        self.assertEqual(report["teacher_forced"]["positions"], 3)
        self.assertEqual(report["tools"]["selection"]["rate"], 0.5)
        self.assertEqual(report["tools"]["arguments"]["rate"], 0.5)
        self.assertEqual(report["tools"]["multi_step_tool_exact"]["rate"], 0.0)
        self.assertEqual(report["task_success"]["candidate"]["rate"], 0.0)
        self.assertEqual(report["task_success"]["target"]["rate"], 1.0)
        self.assertEqual(
            report["free_running"]["completion_exact_at_common_byte_cap"]["compared"], 0
        )
        self.assertEqual(report["free_running"]["censored_target_visible_prefix"]["rate"], 0.0)
        self.assertEqual(report["per_prompt"][0]["candidate_tool_results"], [10, 20])
        self.assertEqual(report["per_prompt"][0]["candidate_completion"], "20")
        self.assertEqual(report["failed_prompt_ids"], ["validation-1"])

        # A tool transcript without a matching executable environment provides
        # teacher-forced evidence but cannot measure a standalone tool rollout.
        target["capabilities"].pop("sandbox")
        external = evaluate(Candidate(), [(row, target)], 32)
        self.assertEqual(external["free_running"]["recorded_context_only"], 1)
        self.assertIsNone(external["tools"]["selection"]["rate"])

    def test_unknown_tokenizer_evidence_and_tool_only_response(self):
        request = self.request()
        response = make_observation(
            request,
            target_epoch="unknown-host",
            events=[
                {
                    "sequence": 0,
                    "kind": "message",
                    "source": "user",
                    "role": "user",
                    "content": request["messages"][0]["content"],
                },
                {
                    "sequence": 1,
                    "kind": "logprobs",
                    "source": "model",
                    "tokens": ["opaque text"],
                    "top_logprobs": [[{"token": "opaque text", "logprob": -0.3}]],
                },
                {
                    "sequence": 2,
                    "kind": "tool_call",
                    "source": "model",
                    "call_id": "external",
                    "name": "subtract",
                    "arguments": '{"a":7,"b":3}',
                },
            ],
            completion="",
            capabilities={"token_probabilities": True, "reasoning": "unavailable"},
            usage={},
        )
        validate_observation(response, request)
        self.assertEqual(len(training_segments(response)), 1)
        response["events"][1].update(
            token_ids=[3], top_logprobs=[[{"token_id": 3, "logprob": -0.3}]]
        )
        with self.assertRaisesRegex(ValueError, "unverified alignment"):
            validate_observation(response, request)
        response["events"][1]["alignment"] = "unverified"
        validate_observation(response, request)
