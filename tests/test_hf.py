"""Offline tests for the optional local-model workflow.

The integration fixture creates a tiny Qwen2 model and tokenizer on disk.  No
test contacts Hugging Face or downloads a checkpoint.
"""

import importlib.util
import json
import math
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from rephix import hf
from rephix.evidence import Observation, read_observations, write_observations

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_HF = HAS_TORCH and importlib.util.find_spec("transformers") is not None


class SplitIntegrityTests(unittest.TestCase):
    def test_prompt_whitespace_cannot_leak_across_splits(self):
        train = [Observation(id="a", prompt="  hello   world  ", output="yes")]
        validation = [Observation(id="b", prompt="hello world", output="no")]
        with self.assertRaises(ValueError):
            hf._check_disjoint(train, validation)

    def test_group_cannot_leak_across_splits(self):
        train = [Observation(id="a", prompt="red", output="yes", metadata={"group_id": "colors"})]
        validation = [
            Observation(id="b", prompt="blue", output="no", metadata={"group_id": "colors"})
        ]
        with self.assertRaises(ValueError):
            hf._check_disjoint(train, validation)

    def test_distinct_prompts_are_accepted(self):
        hf._check_disjoint(
            [Observation(id="a", prompt="red", output="yes")],
            [Observation(id="b", prompt="blue", output="no")],
        )


@unittest.skipUnless(HAS_TORCH, "install rephix[hf] for torch loss tests")
class BucketedLossTests(unittest.TestCase):
    def test_tail_bucket_has_correct_loss_and_gradient(self):
        import torch

        logits = torch.tensor([[math.log(0.1), math.log(0.2), math.log(0.7)]], requires_grad=True)
        rows = [
            [{"token_id": 2, "logprob": math.log(0.5)}, {"token_id": 1, "logprob": math.log(0.3)}]
        ]
        loss = hf._bucketed_loss(logits, rows)
        expected = -(0.5 * math.log(0.7) + 0.3 * math.log(0.2) + 0.2 * math.log(0.1))
        self.assertAlmostEqual(loss.item(), expected, places=5)
        loss.backward()
        torch.testing.assert_close(
            logits.grad, torch.tensor([[-0.1, -0.1, 0.2]]), atol=1e-6, rtol=1e-5
        )

    def test_full_vocabulary_has_no_nan_from_empty_tail(self):
        import torch

        logits = torch.tensor([[math.log(0.2), math.log(0.3), math.log(0.5)]], requires_grad=True)
        rows = [
            [
                {"token_id": token, "logprob": math.log(probability)}
                for token, probability in enumerate((0.2, 0.3, 0.5))
            ]
        ]
        loss = hf._bucketed_loss(logits, rows)
        self.assertTrue(math.isfinite(loss.item()))
        self.assertAlmostEqual(
            loss.item(), -sum(p * math.log(p) for p in (0.2, 0.3, 0.5)), places=5
        )
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_full_vocabulary_missing_probability_mass_is_rejected(self):
        import torch

        rows = [
            [{"token_id": 0, "logprob": math.log(0.2)}, {"token_id": 1, "logprob": math.log(0.3)}]
        ]
        with self.assertRaisesRegex(ValueError, "sum to one"):
            hf._bucketed_loss(torch.zeros(1, 2), rows)


@unittest.skipUnless(HAS_HF, "install rephix[hf] for offline Qwen integration")
class LocalQwenIntegrationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.teacher = self.root / "teacher"
        self.teacher.mkdir()
        torch.set_num_threads(1)
        torch.manual_seed(7)
        vocabulary = {
            "[PAD]": 0,
            "[BOS]": 1,
            "[EOS]": 2,
            "[UNK]": 3,
            "user": 4,
            "assistant": 5,
            "hello": 6,
            "world": 7,
            "red": 8,
            "blue": 9,
            "one": 10,
            "two": 11,
            "green": 12,
            "yes": 13,
            "no": 14,
            ":": 15,
        }
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            unk_token="[UNK]",
        )
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] + ': ' + message['content'] + ' ' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
        )
        tokenizer.save_pretrained(self.teacher)
        config = Qwen2Config(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
        model = Qwen2ForCausalLM(config)
        # Downloaded generation settings must not leak beam/repetition choices
        # into a protocol that compares neutral greedy raw distributions.
        model.generation_config.num_beams = 3
        model.generation_config.repetition_penalty = 1.7
        model.save_pretrained(self.teacher)

    def assert_json_finite(self, value):
        """Reports must be serializable with standards-compliant finite numbers."""
        json.dumps(value, allow_nan=False)

    def collect_fixture(self):
        prompts = self.root / "prompts.jsonl"
        observations = self.root / "observations.jsonl"
        write_observations(
            prompts,
            [
                Observation(id="red", prompt="red", output=""),
                Observation(id="blue", prompt="blue", output=""),
                Observation(id="green", prompt="green", output=""),
            ],
        )
        report = hf.collect(
            Namespace(
                prompts=str(prompts),
                model=str(self.teacher),
                revision=None,
                output=str(observations),
                device="cpu",
                max_new_tokens=4,
                top_k=3,
                seed=42,
            )
        )
        self.assert_json_finite(report)
        records = read_observations(observations)
        self.assertEqual(len(records), 3)
        for record in records:
            self.assertTrue(record.prompt_token_ids)
            self.assertTrue(record.completion_token_ids)
            self.assertEqual(len(record.top_logprobs), len(record.completion_token_ids))
            self.assertTrue(all(len(row) == 3 for row in record.top_logprobs))
            self.assertEqual(record.metadata["effective_generation_config"]["num_beams"], 1)
            self.assertEqual(
                record.metadata["effective_generation_config"]["repetition_penalty"], 1.0
            )
            self.assertIn("raw teacher-forced", record.metadata["logprob_semantics"])
        return records

    def test_exact_ids_preserve_separately_recorded_reasoning_targets(self):
        from transformers import AutoTokenizer

        record = self.collect_fixture()[0]
        record.thinking = "separately displayed observed reasoning"
        record.output = "separately displayed final answer"
        tokenizer = AutoTokenizer.from_pretrained(self.teacher)
        prepared, _ = hf._prepare([record], tokenizer, 64, 16)
        self.assertEqual(prepared[0]["completion"], record.completion_token_ids)
        record.metadata["completion_text"] = "inconsistent serialized target"
        with self.assertRaisesRegex(ValueError, "visible target"):
            hf._prepare([record], tokenizer, 64, 16)

    def test_collect_fit_evaluate_and_weight_comparison(self):
        records = self.collect_fixture()
        train = self.root / "train.jsonl"
        validation = self.root / "validation.jsonl"
        heldout = self.root / "heldout.jsonl"
        candidate = self.root / "candidate"
        write_observations(train, records[:1])
        write_observations(validation, records[1:2])
        write_observations(heldout, records[2:])
        fit_report = hf.fit(
            Namespace(
                train=str(train),
                validation=str(validation),
                teacher=str(self.teacher),
                revision=None,
                init="random",
                output=str(candidate),
                steps=2,
                learning_rate=0.001,
                max_seq_length=64,
                device="cpu",
                seed=42,
                topk_weight=1.0,
                eval_every=1,
            )
        )
        self.assert_json_finite(fit_report)
        self.assertTrue((candidate / "config.json").is_file())
        self.assertTrue((candidate / "tokenizer.json").is_file())
        eval_report = hf.evaluate(
            Namespace(
                data=str(heldout),
                model=str(candidate),
                revision=None,
                output=None,
                max_seq_length=64,
                device="cpu",
                generate=True,
                max_new_tokens=4,
            )
        )
        self.assert_json_finite(eval_report)
        self.assertGreater(eval_report["evaluated_tokens"], 0)
        self.assertGreaterEqual(eval_report["teacher_token_agreement"], 0.0)
        self.assertLessEqual(eval_report["teacher_token_agreement"], 1.0)
        self.assertGreaterEqual(eval_report["generation_exact_match"], 0.0)
        self.assertLessEqual(eval_report["generation_exact_match"], 1.0)
        comparison = hf.compare_weights(
            Namespace(
                truth=str(self.teacher),
                candidate=str(self.teacher),
                device="cpu",
                output=None,
            )
        )
        self.assert_json_finite(comparison)
        self.assertAlmostEqual(comparison["relative_l2"], 0.0, places=7)
        self.assertAlmostEqual(comparison["cosine_similarity"], 1.0, places=7)

        for seen_file in (train, validation):
            with self.subTest(seen_file=seen_file.name), self.assertRaises(ValueError):
                hf.evaluate(
                    Namespace(
                        data=str(seen_file),
                        model=str(candidate),
                        revision=None,
                        output=None,
                        max_seq_length=64,
                        device="cpu",
                        generate=False,
                        max_new_tokens=4,
                    )
                )
        # Changing the record ID changes the file hash but must not evade the
        # prompt-level overlap check stored with a fitted checkpoint.
        copied_train = self.root / "copied-train.jsonl"
        write_observations(copied_train, [replace(records[0], id="renamed-training-record")])
        self.assertNotEqual(train.read_bytes(), copied_train.read_bytes())
        with self.assertRaises(ValueError):
            hf.evaluate(
                Namespace(
                    data=str(copied_train),
                    model=str(candidate),
                    revision=None,
                    output=None,
                    max_seq_length=64,
                    device="cpu",
                    generate=False,
                    max_new_tokens=4,
                )
            )

    def test_stored_token_fingerprint_mismatch_is_rejected(self):
        records = self.collect_fixture()
        # The collector's source vocabulary fingerprint binds cached token IDs
        # to their actual token meanings, even when the numeric IDs are valid.
        metadata = records[0].metadata
        keys = [key for key in metadata if "fingerprint" in key]
        self.assertTrue(keys, "collector must record a tokenizer fingerprint")
        for key in keys:
            metadata[key] = "corrupted-fingerprint"
        train = self.root / "train.jsonl"
        validation = self.root / "validation.jsonl"
        write_observations(train, records[:1])
        write_observations(validation, records[1:2])
        with self.assertRaises(ValueError):
            hf.fit(
                Namespace(
                    train=str(train),
                    validation=str(validation),
                    teacher=str(self.teacher),
                    revision=None,
                    init="random",
                    output=str(self.root / "bad-candidate"),
                    steps=1,
                    learning_rate=0.001,
                    max_seq_length=64,
                    device="cpu",
                    seed=42,
                    topk_weight=1.0,
                    eval_every=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
