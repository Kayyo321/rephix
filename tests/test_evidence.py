import json
import tempfile
import unittest
from pathlib import Path

from rephix.evidence import (
    Observation,
    assert_disjoint,
    leakage_groups,
    read_observations,
    split_observations,
    validate_observation,
    write_observations,
)


class EvidenceTests(unittest.TestCase):
    def record(self, ident="1", prompt="hello", **kwargs):
        return Observation(id=ident, prompt=prompt, output="answer", **kwargs)

    def test_roundtrip_preserves_context_and_observed_thinking(self):
        record = self.record(
            thinking="an observed explanation",
            metadata={"source": "local", "context": ["one"]},
            prompt_token_ids=[1, 2],
            completion_token_ids=[3],
            top_logprobs=[[{"token_id": 3, "logprob": -0.2}]],
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evidence.jsonl"
            write_observations(path, [record])
            self.assertEqual(read_observations(path), [record])

    def test_malformed_inputs_report_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jsonl"
            path.write_text(json.dumps({"id": "one", "prompt": "p", "output": "o"}) + "\n[]\n")
            with self.assertRaisesRegex(ValueError, r":2:"):
                read_observations(path)

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                write_observations(
                    Path(folder) / "bad", [self.record(), self.record(prompt="other")]
                )

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jsonl"
            path.write_text('{"id":"one","prompt":"p","output":"first","output":"second"}\n')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                read_observations(path)

    def test_unpaired_token_ids_are_rejected(self):
        for kwargs in [dict(prompt_token_ids=[1]), dict(completion_token_ids=[2])]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "together"):
                validate_observation(self.record(**kwargs))

    def test_topk_alignment_mass_and_types_are_checked(self):
        bad = [
            dict(prompt_token_ids=[0], completion_token_ids=[1], top_logprobs=[]),
            dict(
                prompt_token_ids=[0],
                completion_token_ids=[1],
                top_logprobs=[[{"token_id": 1, "logprob": 0}, {"token_id": 2, "logprob": 0}]],
            ),
            dict(
                prompt_token_ids=[0],
                completion_token_ids=[1],
                top_logprobs=[[{"token_id": 1, "logprob": float("nan")}]],
            ),
            dict(prompt_token_ids=[0], completion_token_ids=[True]),
            dict(metadata={"nonfinite": float("inf")}),
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_observation(self.record(**kwargs))

    def test_group_connections_are_transitive(self):
        records = [
            self.record("1", "same prompt", metadata={"group_id": "g"}),
            self.record("2", "same\t prompt", metadata={"group_id": "h"}),
            self.record("3", "different prompt", metadata={"group_id": "h"}),
        ]
        self.assertEqual(len(leakage_groups(records)), 1)

    def test_splits_are_reproducible_disjoint_and_complete(self):
        records = [self.record(str(i), f"prompt {i // 2}") for i in range(20)]
        splits = split_observations(records)
        self.assertEqual(splits, split_observations(records))
        assert_disjoint(*splits.values())
        self.assertEqual(
            sorted(r.id for rows in splits.values() for r in rows), sorted(r.id for r in records)
        )
        self.assertTrue(all(splits.values()))

    def test_disjoint_rejects_id_prompt_and_group_leakage(self):
        for second in [
            self.record(prompt="another"),
            self.record("2", "hello "),
            self.record("2", "another", metadata={"group_id": "g"}),
        ]:
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "leakage"):
                assert_disjoint([self.record(metadata={"group_id": "g"})], [second])

    def test_small_splits_and_invalid_fractions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "three"):
            split_observations([self.record()])
        for fraction in [-1, 0, 0.9, float("nan")]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split_observations([self.record()], validation_fraction=fraction)


if __name__ == "__main__":
    unittest.main()
