"""Label-free query scoring and group-aware exclusion checks."""

import importlib.util
import json
import math
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rephix.evidence import Observation, read_observations, write_observations
from rephix.query import (
    _candidate_scores,
    _eligible_groups,
    entropy,
    jensen_shannon,
    select_queries,
    select_records,
)


class QuerySelectionTests(unittest.TestCase):
    def record(self, ident, prompt=None, group=None):
        return Observation(
            id=ident,
            prompt=prompt or f"prompt {ident}",
            output="target should be ignored",
            metadata={"group_id": group} if group else {},
        )

    def test_entropy_and_jensen_shannon_math(self):
        self.assertAlmostEqual(entropy(np.zeros(4)), math.log(4))
        self.assertAlmostEqual(entropy([0, -1000]), 0)
        first, second = np.log([0.2, 0.8]), np.log([0.7, 0.3])
        self.assertAlmostEqual(jensen_shannon(first, first), 0)
        self.assertAlmostEqual(jensen_shannon(first, second), jensen_shannon(second, first))
        self.assertAlmostEqual(jensen_shannon([0, -1000], [-1000, 0]), math.log(2))
        self.assertGreater(jensen_shannon(first, second), 0)
        for bad in [[], [float("nan")], [[1, 2]]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                entropy(bad)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            jensen_shannon([1, 2], [1, 2, 3])

    def test_random_is_reproducible_order_independent_and_label_free(self):
        pool = [self.record(str(index)) for index in range(12)]
        first = select_records(pool, budget=4, seed=7)
        changed = [
            replace(record, output="a completely different answer", thinking="annotation")
            for record in reversed(pool)
        ]
        second = select_records(changed, budget=4, seed=7)
        self.assertEqual(
            [choice.record.id for choice in first], [choice.record.id for choice in second]
        )
        self.assertEqual([choice.score for choice in first], [choice.score for choice in second])

    def test_group_exclusion_is_transitive_across_pool_and_reserved_data(self):
        pool = [
            self.record("a", "same prompt", "family-a"),
            self.record("b", "same  prompt", "family-b"),
            self.record("c", "other prompt", "family-b"),
            self.record("d", "a new independent prompt", "family-c"),
        ]
        reserved = [self.record("reserved", "test prompt", "family-a")]
        self.assertEqual(
            [[record.id for record in group] for group in _eligible_groups(pool, reserved)], [["d"]]
        )
        self.assertEqual(select_records(pool, excluded=reserved, budget=20)[0].record.id, "d")

    def test_id_overlap_is_excluded_even_if_prompt_changed(self):
        pool = [self.record("same-id", "new prompt"), self.record("keep")]
        excluded = [self.record("same-id", "old prompt")]
        self.assertEqual(select_records(pool, excluded=excluded, budget=2)[0].record.id, "keep")

    def test_model_selection_chooses_best_prompt_once_per_group(self):
        pool = [self.record("a", group="one"), self.record("b", group="one"), self.record("c")]
        choices = select_records(
            pool, budget=5, strategy="entropy", scores={"a": 0.1, "b": 0.9, "c": 0.8}
        )
        self.assertEqual([choice.record.id for choice in choices], ["b", "c"])
        self.assertEqual(choices[0].group_size, 2)
        for scores in [None, {"a": 0.1}, {"a": 0.1, "b": float("nan"), "c": 0.8}]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                select_records(pool, budget=1, strategy="entropy", scores=scores)

    def test_invalid_budget_and_empty_eligible_pool_are_rejected(self):
        record = self.record("a")
        for budget in [0, -1, True, 1.5]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                select_records([record], budget=budget)
        with self.assertRaisesRegex(ValueError, "no independent"):
            select_records([record], excluded=[record], budget=1)

    def test_random_cli_strips_targets_and_preserves_context_without_model_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            pool = folder / "pool.jsonl"
            output = folder / "selected.jsonl"
            rows = [
                Observation(
                    id=str(index),
                    prompt=f"prompt {index}",
                    output="old answer",
                    thinking="old reasoning",
                    metadata={
                        "context": {"domain": "science"},
                        "group_id": str(index),
                        "completion_text": "old reasoning and answer",
                        "source_model": "old teacher",
                    },
                    prompt_token_ids=[1],
                    completion_token_ids=[2],
                    top_logprobs=[[{"token_id": 2, "logprob": -0.2}]],
                )
                for index in range(3)
            ]
            write_observations(pool, rows)
            args = Namespace(
                pool=str(pool),
                output=str(output),
                budget=2,
                strategy="random",
                seed=8,
                exclude=[],
                max_seq_length=64,
                model=None,
                reference_candidate=None,
                revision=None,
                device="cpu",
            )
            with patch(
                "rephix.query._candidate_scores", side_effect=AssertionError("must not load models")
            ):
                report = select_queries(args)
            json.dumps(report, allow_nan=False)
            self.assertEqual(report["selected_prompts"], 2)
            selected = read_observations(output)
            for record in selected:
                self.assertEqual(record.output, "")
                self.assertIsNone(record.thinking)
                self.assertIsNone(record.completion_token_ids)
                self.assertIsNone(record.top_logprobs)
                self.assertEqual(record.metadata["context"], {"domain": "science"})
                self.assertNotIn("completion_text", record.metadata)
                self.assertNotIn("source_model", record.metadata)
                self.assertIn("query_selection", record.metadata)
            self.assertTrue(Path(str(output) + ".report.json").is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                select_queries(args)


HAS_HF = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_HF, "install rephix[hf] for offline query scoring")
class OfflineCandidateScoringTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_path = self.root / "candidate"
        self.model_path.mkdir()
        torch.set_num_threads(1)
        torch.manual_seed(4)
        vocabulary = {
            "[PAD]": 0,
            "[EOS]": 1,
            "[UNK]": 2,
            "hello": 3,
            "world": 4,
            "blue": 5,
            "red": 6,
        }
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]"
        )
        tokenizer.save_pretrained(self.model_path)
        config = Qwen2Config(
            vocab_size=len(vocabulary),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            eos_token_id=1,
            pad_token_id=0,
        )
        Qwen2ForCausalLM(config).save_pretrained(self.model_path)

    def args(self, strategy="entropy"):
        return Namespace(
            model=str(self.model_path),
            reference_candidate=str(self.model_path),
            revision=None,
            device="cpu",
            seed=42,
            max_seq_length=64,
            strategy=strategy,
        )

    def test_entropy_uses_only_rendered_prompt_not_any_target_fields(self):
        first = Observation(id="a", prompt="hello world", output="blue", thinking="reasoning")
        second = replace(
            first,
            id="b",
            output="red red red",
            thinking=None,
            metadata={"completion_text": "an annotation which must not be tokenized"},
        )
        scores, report = _candidate_scores([[first, second]], self.args())
        self.assertAlmostEqual(scores["a"], scores["b"], places=12)
        self.assertGreaterEqual(scores["a"], 0)
        self.assertLessEqual(scores["a"], math.log(7))
        self.assertEqual(report["execution_device"], "cpu")

    def test_identical_candidates_have_zero_disagreement(self):
        scores, _ = _candidate_scores(
            [[Observation(id="a", prompt="hello", output="")]], self.args("disagreement")
        )
        self.assertAlmostEqual(scores["a"], 0, places=12)

    def test_selection_does_not_silently_truncate_prompt(self):
        args = self.args()
        args.max_seq_length = 1
        with self.assertRaisesRegex(ValueError, "does not silently truncate"):
            _candidate_scores([[Observation(id="a", prompt="hello world", output="")]], args)

    def test_disagreement_rejects_different_tokenizer_template(self):
        import shutil

        from transformers import AutoTokenizer

        other = self.root / "different-template"
        shutil.copytree(self.model_path, other)
        tokenizer = AutoTokenizer.from_pretrained(other)
        tokenizer.chat_template = "{{ messages[0]['content'] + ' assistant' }}"
        tokenizer.save_pretrained(other)
        args = self.args("disagreement")
        args.reference_candidate = str(other)
        with self.assertRaisesRegex(ValueError, "identical tokenizers"):
            _candidate_scores([[Observation(id="a", prompt="hello", output="")]], args)

    def test_candidate_run_directory_resolves_best_checkpoint_pointer(self):
        wrapped = self.root / "run"
        wrapped.mkdir()
        # The pointer is relative and stays within the candidate run directory.
        import shutil

        shutil.copytree(self.model_path, wrapped / "snapshots" / "best")
        (wrapped / "best_checkpoint.json").write_text('{"path":"snapshots/best"}')
        record = Observation(id="a", prompt="hello", output="")
        original, _ = _candidate_scores([[record]], self.args())
        args = self.args()
        args.model = str(wrapped)
        actual, report = _candidate_scores([[record]], args)
        self.assertEqual(actual, original)
        self.assertEqual(Path(report["resolved_model_path"]), wrapped / "snapshots" / "best")


if __name__ == "__main__":
    unittest.main()
