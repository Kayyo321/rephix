import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rephix.text_identifiability import (
    finite_query_budget,
    greedy_temperature,
    hidden_unit_permutation,
    in_class_identification,
)
from rephix.text_recovery import (
    STATUS_DETERMINISTIC,
    STATUS_INSUFFICIENT,
    STATUS_MISSPECIFIED,
    STATUS_RECOVERED,
    CommandOracle,
    NgramFeatures,
    RecoveredTextModel,
    fit_multinomial,
    recover,
    span_basis,
)

VOCAB = list("abcdef")


def _target(directory, theta, order, **extra):
    np.savez(directory / "target.npz", theta=theta)
    config = {
        "kind": "ngram",
        "vocabulary": VOCAB,
        "order": order,
        "parameters": str(directory / "target.npz"),
        "seed": 5,
        **extra,
    }
    (directory / "server.json").write_text(json.dumps(config), encoding="utf-8")
    return [
        sys.executable,
        "-m",
        "rephix",
        "text-target-serve",
        "--config",
        str(directory / "server.json"),
    ]


def _canonical(theta, basis):
    centre = np.eye(len(VOCAB)) - 1 / len(VOCAB)
    return centre @ theta @ basis


class IdentifiabilityTests(unittest.TestCase):
    def test_counterexamples_are_indistinguishable_and_in_class_control_identifies(self):
        self.assertTrue(hidden_unit_permutation(3)["indistinguishable"])
        self.assertGreater(hidden_unit_permutation(3)["relative_parameter_distance"], 0.1)
        greedy = greedy_temperature(3)
        self.assertTrue(greedy["greedy_outputs_identical"])
        self.assertGreater(greedy["mean_total_variation_between_laws"], 0.05)
        budget = finite_query_budget(3)
        self.assertTrue(budget["indistinguishable_on_queries"])
        self.assertGreater(budget["total_variation_on_trigger"], 0.1)
        self.assertTrue(in_class_identification(3)["identified"])

    def test_mle_recovers_exact_canonical_parameters_from_exact_counts(self):
        rng = np.random.default_rng(0)
        features = NgramFeatures(VOCAB, 2)
        prompts = features.design_contexts(60, rng)
        design = features.features(prompts)
        basis, report = span_basis(design, features.span_dimension())
        self.assertEqual(report["rank"], features.span_dimension())
        theta = rng.normal(size=(6, features.width))
        logits = design @ theta.T
        probabilities = np.exp(logits - logits.max(1, keepdims=True))
        probabilities /= probabilities.sum(1, keepdims=True)
        # Expected counts are an exact law: the MLE must reproduce canonical parameters.
        gamma, info = fit_multinomial(design @ basis, 1e6 * probabilities, ridge=1e-9)
        self.assertTrue(info["converged"])
        np.testing.assert_allclose(gamma, _canonical(theta, basis), atol=1e-6)

    def test_fisher_information_matches_kronecker_definition(self):
        from rephix.text_recovery import _softmax, fisher_information

        rng = np.random.default_rng(1)
        z, gamma = rng.normal(size=(30, 4)), rng.normal(size=(3, 4))
        counts = rng.integers(1, 50, size=(30, 3)).astype(float)
        p = _softmax(z @ gamma.T)
        expected = sum(
            n * np.kron(np.diag(q) - np.outer(q, q), np.outer(x, x))
            for n, q, x in zip(counts.sum(1), p, z)
        )
        np.testing.assert_allclose(fisher_information(z, counts, gamma), expected, atol=1e-9)


class RecoveryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.features = {"kind": "ngram", "vocabulary": VOCAB, "order": 1, "prompt_length": 3}

    def run_recover(self, argv, name="export", **options):
        settings = dict(
            features_spec=self.features,
            target_argv=argv,
            output=self.directory / name,
            transcript_path=self.directory / name / "transcript.jsonl",
            contexts=40,
            samples_per_context=3000,
            validation_contexts=4,
            seed=1,
            bootstrap=39,
            alpha=0.05,
        )
        settings.update(options)
        return recover(**settings)

    def test_in_class_target_is_recovered_and_runs_without_target(self):
        theta = np.random.default_rng(2).normal(size=(6, 7))
        report = self.run_recover(_target(self.directory, theta, 1))
        self.assertEqual(report["status"], STATUS_RECOVERED, report["reasons"])
        with np.load(self.directory / "export" / "parameters.npz") as data:
            gamma, basis, se = data["gamma"], data["basis"], data["gamma_standard_error"]
        error = np.linalg.norm(gamma - _canonical(theta, basis)) / np.linalg.norm(
            _canonical(theta, basis)
        )
        self.assertLess(error, 3 * report["estimated_relative_standard_error"])
        self.assertLess(np.abs((gamma - _canonical(theta, basis)) / se).max(), 5)
        model = RecoveredTextModel(self.directory / "export")
        reloaded = model.log_probabilities(["abc"])
        self.assertAlmostEqual(float(np.exp(reloaded).sum()), 1.0, places=12)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "rephix",
                "text-model",
                "logprobs",
                "--model",
                str(self.directory / "export"),
                "--prompt",
                "abc",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        values = json.loads(completed.stdout)["log_probabilities"]
        np.testing.assert_allclose([values[c] for c in VOCAB], reloaded[0], atol=1e-12)

    def test_out_of_class_target_is_reported_as_approximation(self):
        theta = 1.5 * np.random.default_rng(3).normal(size=(6, 13))
        report = self.run_recover(_target(self.directory, theta, 2))
        self.assertEqual(report["status"], STATUS_MISSPECIFIED)

    def test_greedy_outputs_are_not_treated_as_parameters(self):
        theta = np.random.default_rng(4).normal(size=(6, 7))
        report = self.run_recover(
            _target(self.directory, theta, 1, greedy=True), samples_per_context=20
        )
        self.assertEqual(report["status"], STATUS_DETERMINISTIC)

    def test_too_few_contexts_are_insufficient(self):
        theta = np.random.default_rng(5).normal(size=(6, 7))
        report = self.run_recover(
            _target(self.directory, theta, 1), contexts=3, validation_contexts=0
        )
        self.assertEqual(report["status"], STATUS_INSUFFICIENT)

    def test_resume_reuses_transcript_and_rejects_tampering_or_new_plans(self):
        theta = np.random.default_rng(6).normal(size=(6, 7))
        argv = _target(self.directory, theta, 1)
        self.run_recover(argv, samples_per_context=200, bootstrap=19)
        transcript = self.directory / "export" / "transcript.jsonl"
        lines = transcript.read_text(encoding="utf-8").splitlines()
        transcript.write_text("\n".join(lines[:4]) + '\n{"torn', encoding="utf-8")
        report = self.run_recover(argv, samples_per_context=200, bootstrap=19)
        self.assertEqual(report["budget"]["resumed_contexts"], 3)
        self.assertEqual(len(transcript.read_text(encoding="utf-8").splitlines()), len(lines))
        tampered = transcript.read_text(encoding="utf-8").replace('"a"', '"b"', 1)
        transcript.write_text(tampered, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_recover(argv, samples_per_context=200, bootstrap=19)
        with self.assertRaisesRegex(ValueError, "different query plan"):
            self.run_recover(argv, samples_per_context=201, bootstrap=19)

    def test_bootstrap_must_be_able_to_reject(self):
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            self.run_recover(["unused"], bootstrap=19, alpha=0.01)

    def test_oracle_rejects_any_signal_beyond_strings(self):
        script = self.directory / "leaky.py"
        script.write_text(
            "import json,sys\n"
            "print(json.dumps({'protocol':'rephix-text-oracle/1'}),flush=True)\n"
            "for line in sys.stdin:\n"
            "    print(json.dumps({'outputs':['a'],'logits':[0.0]}),flush=True)\n",
            encoding="utf-8",
        )
        oracle = CommandOracle([sys.executable, str(script)])
        try:
            with self.assertRaisesRegex(RuntimeError, "only an outputs list"):
                oracle.sample("abc", 1)
        finally:
            oracle.close()

    def test_undeclared_output_string_is_a_contract_error(self):
        script = self.directory / "odd.py"
        script.write_text(
            "import json,sys\n"
            "print(json.dumps({'protocol':'rephix-text-oracle/1'}),flush=True)\n"
            "for line in sys.stdin:\n"
            "    n=json.loads(line)['n']\n"
            "    print(json.dumps({'outputs':['zz']*n}),flush=True)\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "not exactly one public class"):
            self.run_recover([sys.executable, str(script)], samples_per_context=5, bootstrap=19)


if __name__ == "__main__":
    unittest.main()
