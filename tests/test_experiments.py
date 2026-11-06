"""Sweep durability, evidence identity, and report semantics."""

import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rephix import experiments
from rephix.evidence import Observation, write_observations


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write_observations(self.root / "train.jsonl", [Observation("a", "red", "yes")])
        write_observations(self.root / "validation.jsonl", [Observation("b", "blue", "yes")])
        self.config = self.root / "config.json"
        self.output = self.root / "runs"
        self.payload = {
            "schema_version": 1,
            "teacher": "owner/model",
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "device": "cpu",
            "revision": "a" * 40,
            "defaults": {"steps": 2, "eval_every": 1},
            "grid": {"seed": [1, 2], "topk_weight": [0.0, 1.0]},
        }
        self.save_config()

    def save_config(self):
        self.config.write_text(json.dumps(self.payload), encoding="utf-8")

    def args(self, resume=False):
        return Namespace(config=str(self.config), output=str(self.output), resume=resume)

    @staticmethod
    def fake_fit(args):
        directory = Path(args.output)
        directory.mkdir(parents=True, exist_ok=True)
        metrics = {
            "initialization": args.init,
            "seed": args.seed,
            "steps": args.steps,
            "best_step": 1,
            "learning_rate": args.learning_rate,
            "topk_weight": args.topk_weight,
            "max_seq_length": args.max_seq_length,
            "train_examples": 1,
            "execution_device": "cpu",
            "checkpoint_selection_metric": "continuation_cross_entropy",
            "baseline_validation": {
                "continuation_cross_entropy": 2.0,
                "teacher_token_agreement": 0.1,
            },
            "best_validation": {"continuation_cross_entropy": 1.0, "teacher_token_agreement": 0.5},
            "history": [
                {
                    "step": 1,
                    "validation": {
                        "continuation_cross_entropy": 1.0,
                        "teacher_token_agreement": 0.5,
                    },
                }
            ],
        }
        (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return metrics

    def test_grid_is_deterministic_and_resume_skips_verified_completed_runs(self):
        with patch("rephix.hf.fit", side_effect=self.fake_fit) as fit:
            first = experiments.sweep(self.args())
            self.assertEqual(fit.call_count, 4)
            second = experiments.sweep(self.args(resume=True))
            self.assertEqual(fit.call_count, 4)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        manifest = json.loads((self.output / "sweep.json").read_text())
        self.assertEqual(len({run["id"] for run in manifest["runs"]}), 4)
        self.assertTrue(all(run["status"] == "completed" for run in manifest["runs"]))
        report = json.loads((self.output / "report" / "comparison.json").read_text())
        self.assertEqual(len(report["runs"]), 4)
        self.assertTrue(all(run["observed_completion_tokens"] is None for run in report["runs"]))
        self.assertNotIn("winner", report)

    def test_changed_evidence_is_rejected_before_resume(self):
        with patch("rephix.hf.fit", side_effect=self.fake_fit):
            experiments.sweep(self.args())
        write_observations(self.root / "train.jsonl", [Observation("a", "red", "different")])
        with patch("rephix.hf.fit") as fit, self.assertRaisesRegex(ValueError, "changed"):
            experiments.sweep(self.args(resume=True))
        fit.assert_not_called()

    def test_changed_local_model_is_rejected_before_resume(self):
        teacher = self.root / "teacher"
        teacher.mkdir()
        (teacher / "config.json").write_text('{"hidden_size":8}')
        self.payload["teacher"] = "./teacher"
        self.save_config()
        with patch("rephix.hf.fit", side_effect=self.fake_fit):
            experiments.sweep(self.args())
        (teacher / "config.json").write_text('{"hidden_size":16}')
        with self.assertRaisesRegex(ValueError, "changed"):
            experiments.sweep(self.args(resume=True))

    def test_modified_manifest_run_settings_are_rejected(self):
        with patch("rephix.hf.fit", side_effect=self.fake_fit):
            experiments.sweep(self.args())
        path = self.output / "sweep.json"
        manifest = json.loads(path.read_text())
        manifest["runs"][0]["parameters"]["learning_rate"] = 0.25
        path.write_text(json.dumps(manifest))
        with patch("rephix.hf.fit") as fit, self.assertRaisesRegex(ValueError, "inconsistent"):
            experiments.sweep(self.args(resume=True))
        fit.assert_not_called()

    def test_malformed_sweep_manifest_is_rejected(self):
        self.output.mkdir()
        path = self.output / "sweep.json"
        for payload in [42, {"runs": [None]}]:
            with self.subTest(payload=payload):
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "manifest"):
                    experiments.sweep(self.args(resume=True))

    def test_interrupted_run_uses_durable_training_state_on_resume(self):
        def interrupted(args):
            directory = Path(args.output)
            directory.mkdir(parents=True)
            (directory / "training_state.pt").write_bytes(b"marker for mocked fit")
            raise KeyboardInterrupt("simulated interruption")

        with patch("rephix.hf.fit", side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
            experiments.sweep(self.args())
        manifest = json.loads((self.output / "sweep.json").read_text())
        self.assertEqual(manifest["runs"][0]["status"], "interrupted")
        with patch("rephix.hf.fit", side_effect=self.fake_fit) as fit:
            experiments.sweep(self.args(resume=True))
        self.assertTrue(fit.call_args_list[0].args[0].resume)
        self.assertTrue(all(not call.args[0].resume for call in fit.call_args_list[1:]))

    def test_tampered_completed_metrics_are_rejected(self):
        with patch("rephix.hf.fit", side_effect=self.fake_fit):
            experiments.sweep(self.args())
        path = next(self.output.glob("run-*/metrics.json"))
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            experiments.sweep(self.args(resume=True))
        with self.assertRaisesRegex(ValueError, "match sweep"):
            experiments.create_report([str(self.output)], self.root / "report")

    def test_leaking_splits_are_rejected_before_model_load(self):
        write_observations(self.root / "validation.jsonl", [Observation("b", " red ", "no")])
        with patch("rephix.hf.fit") as fit, self.assertRaisesRegex(ValueError, "leakage"):
            experiments.sweep(self.args())
        fit.assert_not_called()

    def test_strict_schema_rejects_unknown_keys_and_duplicate_grid_points(self):
        for patch_value in [
            {"test": "test.jsonl"},
            {"grid": {"seed": [1, 1]}},
            {"grid": {"seed": []}},
            {"defaults": {"typo": 1}},
            {"defaults": {"learning_rate": float("nan")}},
            {"defaults": {"steps": True}},
            {"device": {}},
        ]:
            with self.subTest(patch_value=patch_value):
                self.config.write_text(json.dumps({**self.payload, **patch_value}))
                with self.assertRaises(ValueError):
                    experiments.load_sweep_config(self.config)

    def test_duplicate_json_keys_are_rejected(self):
        self.config.write_text('{"schema_version":1,"schema_version":2}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            experiments.load_sweep_config(self.config)

    def test_mutable_remote_references_are_rejected(self):
        self.payload["revision"] = "main"
        self.save_config()
        with patch("rephix.hf.fit") as fit, self.assertRaisesRegex(ValueError, "immutable"):
            experiments.sweep(self.args())
        fit.assert_not_called()

    def test_evidence_summary_distinguishes_missing_token_counts(self):
        record = Observation(
            "a",
            "red",
            "yes",
            prompt_token_ids=[1],
            completion_token_ids=[2],
            top_logprobs=[[{"token_id": 2, "logprob": -0.5}]],
        )
        summary = experiments.evidence_summary([record])
        self.assertEqual(summary["observed_completion_tokens"], 1)
        self.assertEqual(summary["top_logprob_token_coverage"], 1.0)
        mixed = experiments.evidence_summary([record, Observation("b", "blue", "yes")])
        self.assertIsNone(mixed["observed_completion_tokens"])
        self.assertEqual(mixed["examples_without_token_ids"], 1)

    def test_report_nonfinite_metrics_are_rejected(self):
        self.output.mkdir()
        args = Namespace(
            output=str(self.output),
            init="random",
            seed=1,
            steps=2,
            learning_rate=0.001,
            topk_weight=0,
            max_seq_length=32,
        )
        metrics = self.fake_fit(args)
        metrics["best_validation"]["continuation_cross_entropy"] = float("nan")
        (self.output / "metrics.json").write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "finite"):
            experiments.create_report([str(self.output)], self.root / "report")

    def test_malformed_report_shapes_are_rejected(self):
        self.output.mkdir()
        for value in [
            42,
            [],
            {"baseline_validation": {}, "best_validation": []},
            {"baseline_validation": {}, "best_validation": {}, "history": [3]},
        ]:
            with self.subTest(value=value):
                (self.output / "metrics.json").write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    experiments.create_report([str(self.output)], self.root / "report")

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "install rephix[analysis] for SVG reports",
    )
    def test_plot_report_produces_standalone_svg(self):
        args = Namespace(
            output=str(self.output),
            init="random",
            seed=1,
            steps=2,
            learning_rate=0.001,
            topk_weight=0,
            max_seq_length=32,
        )
        self.fake_fit(args)
        report = experiments.create_report([str(self.output)], self.root / "report", plot=True)
        self.assertEqual(report["plot"]["status"], "created")
        svg = Path(report["plot"]["path"])
        self.assertEqual(ET.parse(svg).getroot().tag, "{http://www.w3.org/2000/svg}svg")


HAS_HF = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_HF, "install rephix[hf] for offline sweep integration")
class TinySweepIntegrationTests(unittest.TestCase):
    def test_two_local_fits_produce_comparison_artifacts(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            teacher = root / "teacher"
            vocabulary = {"[PAD]": 0, "[UNK]": 1, "red": 2, "blue": 3, "yes": 4, "no": 5}
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
            )
            tokenizer.save_pretrained(teacher)
            model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=6,
                    n_embd=8,
                    n_layer=1,
                    n_head=1,
                    n_positions=32,
                    bos_token_id=None,
                    eos_token_id=None,
                )
            )
            model.save_pretrained(teacher)
            write_observations(root / "train.jsonl", [Observation("a", "red", "yes")])
            write_observations(root / "validation.jsonl", [Observation("b", "blue", "yes")])
            config = root / "sweep.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "teacher": "./teacher",
                        "train": "train.jsonl",
                        "validation": "validation.jsonl",
                        "device": "cpu",
                        "defaults": {"steps": 1, "max_seq_length": 16, "eval_every": 1},
                        "grid": {"seed": [7, 8], "topk_weight": [0]},
                    }
                )
            )
            result = experiments.sweep(
                Namespace(config=str(config), output=str(root / "runs"), resume=False)
            )
            self.assertEqual(result["completed_runs"], 2)
            self.assertTrue((root / "runs" / "report" / "comparison.csv").is_file())
            self.assertTrue((root / "runs" / "report" / "comparison.md").is_file())


if __name__ == "__main__":
    unittest.main()
