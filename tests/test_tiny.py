"""Complete local trial checks using only generated, untrained checkpoints."""

import importlib.util
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rephix.evidence import assert_disjoint, read_observations
from rephix.tiny import init_tiny, offline_trial

HAS_HF = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_HF, "install rephix[hf] for the offline trial")
class TinyFixtureTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)

    def fixture(self, destination, seed=7):
        return init_tiny(Namespace(output=str(destination), seed=seed, hidden_size=32, layers=2))

    def test_fixture_is_local_deterministic_and_uses_known_prompt_tokens(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            first = self.fixture(self.root / "first")
            second = self.fixture(self.root / "second")
            a = AutoModelForCausalLM.from_pretrained(first["model"], local_files_only=True)
            b = AutoModelForCausalLM.from_pretrained(second["model"], local_files_only=True)
            self.assertEqual(set(a.state_dict()), set(b.state_dict()))
            for key, value in a.state_dict().items():
                self.assertTrue(torch.equal(value, b.state_dict()[key]), key)
            tokenizer = AutoTokenizer.from_pretrained(first["model"], local_files_only=True)
        self.assertEqual(first["vocabulary_size"], 64)
        self.assertEqual(first["tokenizer_fingerprint"], second["tokenizer_fingerprint"])
        prompts = read_observations(first["prompts"])
        pool = read_observations(first["pool"])
        self.assertEqual(len(prompts), 64)
        self.assertEqual(len(pool), 96)
        self.assertEqual(len({record.prompt for record in pool}), 96)
        self.assertEqual(len({record.metadata["group_id"] for record in pool}), 96)
        self.assertEqual(prompts, pool[:64])
        for record in pool:
            self.assertNotIn(tokenizer.unk_token_id, tokenizer.encode(record.prompt))
        self.assertFalse(first["network_access_required"])

    def test_invalid_architecture_and_nonempty_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            init_tiny(
                Namespace(output=str(self.root / "invalid"), seed=7, hidden_size=30, layers=2)
            )
        self.assertFalse((self.root / "invalid").exists())
        output = self.root / "occupied"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("existing contents", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.fixture(output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing contents")

    def test_complete_trial_does_not_use_network_and_evaluates_test_once(self):
        from rephix import hf

        output = self.root / "trial"
        with (
            patch.dict(
                os.environ,
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "MPLCONFIGDIR": str(self.root / "matplotlib"),
                },
            ),
            patch(
                "requests.sessions.Session.request", side_effect=AssertionError("network attempted")
            ),
            patch("rephix.hf.evaluate", wraps=hf.evaluate) as evaluate,
        ):
            report = offline_trial(
                Namespace(output=str(output), steps=2, seed=42, teacher_seed=7, threads=1)
            )
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(Path(evaluate.call_args.args[0].data).name, "test.jsonl")
        self.assertEqual(report["test_evaluations"], 1)
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["network_access_required"])
        self.assertFalse(report["selection"]["test_used_for_selection"])
        self.assertEqual(report["selection"]["candidate_runs"], 2)
        self.assertEqual(report["sweep"]["completed_runs"], 2)
        json.dumps(report, allow_nan=False)
        self.assertEqual(json.loads((output / "trial.json").read_text()), report)
        self.assertEqual(
            json.loads((output / "trial-status.json").read_text())["status"], "complete"
        )
        split_rows = [
            read_observations(output / "splits" / f"{name}.jsonl")
            for name in ("train", "validation", "test")
        ]
        assert_disjoint(*split_rows)
        self.assertEqual(sum(map(len, split_rows)), 48)
        selected = read_observations(output / "next-queries.jsonl")
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(not record.output for record in selected))
        assert_disjoint(selected, [record for split in split_rows for record in split])
        self.assertTrue(report["projection"]["uses_hidden_activations"])
        self.assertFalse(report["projection"]["uses_reference_weights_in_estimator"])
        self.assertEqual(report["projection"]["diagnostics"]["design_rank"], 32)
        self.assertLess(report["projection"]["parameter_relative_error"], 1e-4)
        self.assertTrue((output / "comparison" / "comparison.csv").is_file())
        self.assertTrue((output / "trial.md").is_file())

    def test_failed_stage_is_recorded_without_overwriting_trial_on_rerun(self):
        output = self.root / "failed"
        with patch("rephix.hf.collect", side_effect=RuntimeError("deliberate collection failure")):
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                offline_trial(Namespace(output=str(output), steps=2, seed=42, threads=1))
        status = json.loads((output / "trial-status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["stage"], "collecting_reference_evidence")
        with self.assertRaisesRegex(ValueError, "empty"):
            offline_trial(Namespace(output=str(output), steps=2, seed=42, threads=1))


if __name__ == "__main__":
    unittest.main()
