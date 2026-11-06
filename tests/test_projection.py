"""Identification tests for explicitly observed output-projection inputs."""

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from rephix.evidence import Observation, write_observations
from rephix.projection import (
    _feature_novelty,
    _selected_ids,
    _split_prompts,
    projection_demo,
    recover_projection,
    run,
)

HAS_HF = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


class ProjectionSolverTests(unittest.TestCase):
    def test_novel_feature_metrics_separate_exactly_repeated_prefixes(self):
        train = np.array([[1, 2], [2, 3]], dtype=np.float32)
        test = np.array([[1, 2], [4, 5], [4, 5]], dtype=np.float32)
        observed = np.array([[3, 4], [7, 8], [7, 8]], dtype=np.float32)
        predicted = np.array([[3, 4], [8, 7], [8, 7]], dtype=np.float32)
        report = _feature_novelty(train, test, observed, predicted)
        self.assertEqual(report["matching_training_rows"], 1)
        self.assertEqual(report["novel_test_rows"], 2)
        self.assertEqual(report["unique_test_feature_rows"], 2)
        self.assertEqual(report["novel_only_logit_rmse"], 1)
        self.assertEqual(report["novel_only_selected_vocabulary_top1_agreement"], 0)

    def test_budget_demo_distinguishes_fit_from_identification_and_noise(self):
        report = projection_demo(feature_sizes=(4,), repeats=2, test_samples=64)
        json.dumps(report, allow_nan=False)
        for row in report["rows"]:
            expected_rank = min(row["train_samples"], 4)
            self.assertEqual(row["design_rank_min"], expected_rank)
            self.assertEqual(row["identifiable_all_replicates"], row["train_samples"] >= 4)
            if row["observation_noise_std"] == 0:
                self.assertLess(row["training_target_rmse_mean"], 1e-12)
                if row["train_samples"] >= 4:
                    self.assertLess(row["parameter_relative_error_mean"], 1e-12)
                else:
                    self.assertGreater(row["heldout_logit_rmse_mean"], 0.1)

    def test_known_zero_bias_recovers_unseen_logits(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(96, 8))
        weights = rng.normal(size=(8, 5))
        result = recover_projection(x[:64], x[:64] @ weights)
        np.testing.assert_allclose(result.weights, weights, atol=1e-12)
        np.testing.assert_allclose(result.predict(x[64:]), x[64:] @ weights, atol=1e-12)
        self.assertTrue(result.diagnostics["identifiable_in_reported_gauge"])
        self.assertEqual(result.diagnostics["parameters_per_class"], 8)

    def test_affine_bias_is_estimated_when_not_known_zero(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(96, 8))
        weights = rng.normal(size=(8, 5))
        bias = rng.normal(size=5)
        result = recover_projection(x[:64], x[:64] @ weights + bias, bias_mode="fit")
        np.testing.assert_allclose(result.weights, weights, atol=1e-12)
        np.testing.assert_allclose(result.bias, bias, atol=1e-12)
        np.testing.assert_allclose(result.predict(x[64:]), x[64:] @ weights + bias, atol=1e-12)

    def test_rank_deficient_observations_remain_unidentifiable_with_ridge(self):
        x = np.ones((8, 5))
        result = recover_projection(x, np.ones((8, 3)), ridge=0.1)
        self.assertFalse(result.diagnostics["identifiable_in_reported_gauge"])
        self.assertEqual(result.diagnostics["design_rank"], 1)
        self.assertIsNone(result.diagnostics["condition_number"])

    def test_invalid_solver_inputs_are_rejected(self):
        for features, logits, kwargs in [
            (np.zeros((2, 3)), np.zeros((3, 2)), {}),
            (np.zeros((2, 3)), np.zeros((2, 2)), {"ridge": -1}),
            (np.zeros((2, 3)), np.zeros((2, 2)), {"bias_mode": "inferred"}),
            (np.full((2, 3), np.nan), np.zeros((2, 2)), {}),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                recover_projection(features, logits, **kwargs)

    def test_prompt_groups_are_isolated_before_positions_are_collected(self):
        records = [
            Observation("1", "repeat me", "", metadata={"group_id": "one"}),
            Observation("2", "repeat   me", "", metadata={"group_id": "two"}),
            Observation("3", "another variant", "", metadata={"group_id": "two"}),
            Observation("4", "independent", ""),
        ]
        train, test = _split_prompts(records, 0.25, 42)
        self.assertEqual(sorted([len(train), len(test)]), [1, 3])
        with self.assertRaisesRegex(ValueError, "two independent"):
            _split_prompts(records[:3], 0.25, 42)

    def test_selected_vocabulary_is_explicit_and_unique(self):
        self.assertEqual(_selected_ids(100, 3, None), [0, 49, 99])
        self.assertEqual(_selected_ids(4, 0, None), [0, 1, 2, 3])
        self.assertEqual(_selected_ids(100, 3, "1,7,42"), [1, 7, 42])
        for value in ("1,1", "1,999", "x,3", "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _selected_ids(100, 3, value)


@unittest.skipUnless(HAS_HF, "install rephix[hf] for offline projection integration")
class OfflineProjectionIntegrationTests(unittest.TestCase):
    def test_actual_tiny_qwen_projection_is_recovered_on_heldout_prompt_groups(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            torch.set_num_threads(1)
            torch.manual_seed(31)
            vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}
            vocabulary.update({f"w{i}": i + 3 for i in range(61)})
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
            )
            tokenizer.save_pretrained(source)
            model = Qwen2ForCausalLM(
                Qwen2Config(
                    vocab_size=64,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=128,
                    eos_token_id=2,
                    pad_token_id=0,
                )
            )
            model.save_pretrained(source)
            rng = np.random.default_rng(8)
            prompts = [
                Observation(
                    f"prompt-{i}", " ".join(f"w{x}" for x in rng.integers(0, 61, size=20)), ""
                )
                for i in range(12)
            ]
            prompt_path = root / "prompts.jsonl"
            write_observations(prompt_path, prompts)
            args = Namespace(
                model=str(source),
                prompts=str(prompt_path),
                output=str(root / "recovered"),
                revision=None,
                max_seq_length=24,
                max_samples=192,
                max_memory_mb=16,
                vocab_limit=8,
                token_ids=None,
                test_fraction=0.25,
                bias_mode="auto",
                ridge=0.0,
                seed=42,
                device="cpu",
            )
            report = run(args)
            json.dumps(report, allow_nan=False)
            self.assertEqual(report["bias_mode"], "known-zero")
            self.assertTrue(report["uses_hidden_activations"])
            self.assertFalse(report["uses_reference_weights_in_estimator"])
            self.assertTrue(report["diagnostics"]["identifiable_in_reported_gauge"])
            self.assertLess(report["parameter_relative_error"], 1e-5)
            self.assertLess(report["heldout_logit_rmse"], 1e-6)
            self.assertEqual(report["heldout_selected_vocabulary_top1_agreement"], 1.0)
            self.assertLessEqual(report["train_samples"] + report["test_samples"], 192)
            self.assertTrue(set(report["train_prompt_ids"]).isdisjoint(report["test_prompt_ids"]))
            for filename in ("observations.npz", "reference.npz", "recovered.npz"):
                with np.load(root / "recovered" / filename, allow_pickle=False) as archive:
                    for key in archive.files:
                        archive[key]
            args.output = str(root / "recovered")
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                run(args)
            args.output = str(root / "too-large")
            args.max_memory_mb = 1
            args.max_samples = 100000
            with self.assertRaisesRegex(ValueError, "max-memory-mb"):
                run(args)


if __name__ == "__main__":
    unittest.main()
