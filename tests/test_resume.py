"""Offline training-state integrity, cost-reduced candidates, and metric contracts."""

import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rephix import hf
from rephix.evidence import Observation, write_observations

HAS_HF = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_HF, "install rephix[hf] for offline resumable-training tests")
class ResumeTests(unittest.TestCase):
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
        vocabulary = {
            "[PAD]": 0,
            "[BOS]": 1,
            "[EOS]": 2,
            "[UNK]": 3,
            "red": 4,
            "blue": 5,
            "green": 6,
            "yellow": 7,
            "yes": 8,
            "no": 9,
            "one": 10,
            "two": 11,
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
        tokenizer.save_pretrained(self.teacher)
        torch.set_num_threads(1)
        torch.manual_seed(11)
        config = Qwen2Config(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.2,
            tie_word_embeddings=True,
        )
        model = Qwen2ForCausalLM(config)
        model.save_pretrained(self.teacher)
        self.train = self.root / "train.jsonl"
        self.validation = self.root / "validation.jsonl"
        write_observations(
            self.train,
            [
                Observation(id="red", prompt="red", output="yes one yes"),
                Observation(id="blue", prompt="blue", output="no two no"),
                Observation(id="green", prompt="green", output="yes two yes"),
            ],
        )
        write_observations(
            self.validation, [Observation(id="yellow", prompt="yellow", output="no one no")]
        )

    def args(self, output, **overrides):
        values = dict(
            train=str(self.train),
            validation=str(self.validation),
            teacher=str(self.teacher),
            revision=None,
            init="random",
            output=str(self.root / output),
            steps=4,
            learning_rate=0.001,
            max_seq_length=32,
            device="cpu",
            seed=31,
            topk_weight=0.4,
            eval_every=1,
            save_every=1,
            trainable="all",
            student_config=None,
            gradient_accumulation_steps=2,
            resume=False,
        )
        values.update(overrides)
        return Namespace(**values)

    def state(self, output):
        import torch

        return torch.load(
            self.root / output / "training_state.pt", map_location="cpu", weights_only=True
        )

    def test_resumed_optimizer_and_dropout_rng_match_uninterrupted_training(self):
        import torch

        uninterrupted = hf.fit(self.args("full"))
        first = hf.fit(self.args("resumed", steps=2))
        self.assertEqual(first["completed_steps"], 2)
        resumed = hf.fit(self.args("resumed", resume=True))
        left, right = self.state("full"), self.state("resumed")
        self.assertEqual(right["completed_steps"], 4)
        self.assertEqual(left["order"], right["order"])
        self.assertEqual(left["sample_index"], right["sample_index"])
        self.assertEqual(uninterrupted["best_validation"], resumed["best_validation"])
        for name in left["model"]:
            with self.subTest(parameter=name):
                self.assertTrue(torch.equal(left["model"][name], right["model"][name]))
        self.assertEqual(left["optimizer"]["param_groups"], right["optimizer"]["param_groups"])
        self.assertEqual(resumed["checkpoint_selection_metric"], "reconstruction_objective")
        same_target = hf.fit(self.args("resumed", resume=True))
        self.assertEqual(same_target["completed_steps"], 4)

    def test_resume_rejects_changed_configuration_and_evidence(self):
        hf.fit(self.args("strict", steps=1))
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            hf.fit(self.args("strict", resume=True, learning_rate=0.002))
        with self.assertRaisesRegex(ValueError, "less than"):
            hf.fit(self.args("strict", resume=True, steps=0.5))
        write_observations(
            self.train, [Observation(id="changed", prompt="changed", output="different")]
        )
        with self.assertRaisesRegex(ValueError, "train_sha256"):
            hf.fit(self.args("strict", resume=True))

    def test_head_only_updates_output_projection_and_unties_embeddings(self):
        import torch
        from transformers import AutoModelForCausalLM

        original = AutoModelForCausalLM.from_pretrained(self.teacher)
        before = {name: value.detach().clone() for name, value in original.state_dict().items()}
        report = hf.fit(self.args("head", init="pretrained", trainable="head", steps=2))
        latest = self.state("head")["model"]
        changed = []
        for name in before:
            if not torch.equal(before[name], latest[name]):
                changed.append(name)
        self.assertEqual(changed, ["lm_head.weight"])
        self.assertTrue(report["untied_output_embeddings"])
        self.assertTrue(report["frozen_backbone_uses_pretrained_prior"])
        self.assertLess(report["trainable_parameters"], report["total_parameters"])
        hf.fit(self.args("head", init="pretrained", trainable="head", steps=3, resume=True))

    def test_smaller_random_student_has_compatible_vocabulary(self):
        from transformers import Qwen2Config

        student = self.root / "student-config"
        config = Qwen2Config.from_pretrained(self.teacher)
        config.hidden_size = 8
        config.intermediate_size = 16
        config.save_pretrained(student)
        report = hf.fit(self.args("small", student_config=str(student), steps=1))
        self.assertFalse(report["teacher_weights_used_as_initialization"])
        self.assertEqual(self.state("small")["model"]["model.embed_tokens.weight"].shape[1], 8)
        with self.assertRaisesRegex(ValueError, "random initialization"):
            hf.fit(self.args("invalid", student_config=str(student), init="pretrained"))
        config.vocab_size += 1
        config.save_pretrained(student)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            hf.fit(self.args("invalid", student_config=str(student)))

    def test_failed_state_write_leaves_last_complete_state_resumable(self):
        import torch

        original = hf._atomic_torch_save
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated interrupted disk write")
            return original(*args, **kwargs)

        with patch.object(hf, "_atomic_torch_save", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "interrupted"):
                hf.fit(self.args("interrupted", steps=2))
        self.assertEqual(self.state("interrupted")["completed_steps"], 0)
        resumed = hf.fit(self.args("interrupted", steps=2, resume=True))
        full = hf.fit(self.args("reference", steps=2))
        self.assertEqual(resumed["best_validation"], full["best_validation"])
        for name, tensor in self.state("reference")["model"].items():
            self.assertTrue(torch.equal(tensor, self.state("interrupted")["model"][name]))

    def test_partial_torch_serialization_does_not_replace_existing_state(self):
        import torch

        path = self.root / "state.pt"
        hf._atomic_torch_save(torch, {"completed": 1}, path)
        original = path.read_bytes()

        def partial_write(payload, stream):
            stream.write(b"partial")
            raise OSError("disk full")

        with patch.object(torch, "save", side_effect=partial_write):
            with self.assertRaisesRegex(OSError, "disk full"):
                hf._atomic_torch_save(torch, {"completed": 2}, path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob(".state.pt.*")), [])

    def test_token_and_observed_top_token_agreement_are_distinct(self):
        import torch

        model = torch.nn.Linear(1, 1)
        records = [{"top_logprobs": [[{"token_id": 2, "logprob": -0.1}]]}]
        logits = torch.tensor([[0.0, 0.0, 2.0]])
        target = torch.tensor([1])
        with patch.object(hf, "_forward", return_value=(logits, target)):
            report = hf._metrics(torch, model, records, "cpu", topk_weight=0.4)
        self.assertEqual(report["recorded_token_agreement"], 0.0)
        self.assertEqual(report["observed_top_token_agreement"], 1.0)
        expected = (
            0.6 * report["continuation_cross_entropy"] + 0.4 * report["bucketed_cross_entropy"]
        )
        self.assertAlmostEqual(report["reconstruction_objective"], expected)


if __name__ == "__main__":
    unittest.main()
