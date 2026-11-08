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
        "seed": 6,
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

    def test_layernorm_float32_features_use_their_true_span(self):
        # LayerNorm with bias puts features on a hyperplane; float32 rounding must not
        # be mistaken for an extra identified direction.
        rng = np.random.default_rng(3)
        d, contexts = 24, 400
        raw = rng.normal(size=(contexts, d))
        normed = (raw - raw.mean(1, keepdims=True)) / raw.std(1, keepdims=True)
        gain, shift = rng.uniform(0.5, 1.5, d), rng.normal(size=d)
        hidden = (normed * gain + shift).astype(np.float32).astype(np.float64)
        design = np.concatenate([hidden, np.ones((contexts, 1))], 1)
        exact, _ = span_basis(design, tolerance=1e-9)
        basis, report = span_basis(design, tolerance=1e-5)
        self.assertEqual(report["rank"], d)
        self.assertGreater(exact.shape[1], d)  # the old tolerance kept a rounding direction
        theta = rng.normal(size=(4, d + 1)) * 0.3
        from rephix.text_recovery import _softmax

        counts = np.stack([rng.multinomial(4000, q) for q in _softmax(design @ theta.T)]).astype(
            float
        )
        gamma, _ = fit_multinomial(design @ basis, counts)
        truth = (np.eye(4) - 0.25) @ theta @ basis
        self.assertLess(np.linalg.norm(gamma - truth) / np.linalg.norm(truth), 0.05)


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
            bootstrap=79,
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
        self.run_recover(argv, samples_per_context=200, bootstrap=79)
        transcript = self.directory / "export" / "transcript.jsonl"
        lines = transcript.read_text(encoding="utf-8").splitlines()
        transcript.write_text("\n".join(lines[:4]) + '\n{"torn', encoding="utf-8")
        report = self.run_recover(argv, samples_per_context=200, bootstrap=79)
        self.assertEqual(report["budget"]["resumed_contexts"], 3)
        self.assertEqual(len(transcript.read_text(encoding="utf-8").splitlines()), len(lines))
        tampered = transcript.read_text(encoding="utf-8").replace('"a"', '"b"', 1)
        transcript.write_text(tampered, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_recover(argv, samples_per_context=200, bootstrap=79)
        with self.assertRaisesRegex(ValueError, "different query plan"):
            self.run_recover(argv, samples_per_context=201, bootstrap=79)

    def test_bootstrap_must_be_able_to_reject(self):
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            self.run_recover(["unused"], bootstrap=99, alpha=0.01)

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

    def test_under_dispersed_outputs_are_not_accepted_as_samples(self):
        # Rounded expected counts leak the law through strings but are not samples.
        theta = np.random.default_rng(8).normal(size=(6, 7))
        np.save(self.directory / "theta.npy", theta)
        script = self.directory / "rounded.py"
        script.write_text(
            "import json,sys,numpy as np\n"
            f"theta=np.load(r'{self.directory / 'theta.npy'}')\n"
            "v='abcdef'\n"
            "print(json.dumps({'protocol':'rephix-text-oracle/1'}),flush=True)\n"
            "for line in sys.stdin:\n"
            "    r=json.loads(line); x=np.zeros(7); x[v.index(r['prompt'][-1])]=1; x[-1]=1\n"
            "    l=theta@x; p=np.exp(l-l.max()); p/=p.sum()\n"
            "    c=np.floor(p*r['n']).astype(int); c[np.argmax(p)]+=r['n']-c.sum()\n"
            "    print(json.dumps({'outputs':[v[i] for i in range(6) for _ in range(c[i])]}),"
            "flush=True)\n",
            encoding="utf-8",
        )
        report = self.run_recover([sys.executable, str(script)])
        self.assertEqual(report["status"], STATUS_INSUFFICIENT)
        self.assertIn("under-dispersed", " ".join(report["reasons"]))

    def test_departure_on_unqueried_inputs_is_a_documented_blind_spot(self):
        # The guarantee is conditional on class membership: a target that differs only
        # on a never-queried suffix cannot be detected from these observations.
        theta = np.random.default_rng(9).normal(size=(6, 7))
        argv = _target(
            self.directory, theta, 1, trigger={"suffix": "fedcba", "class": "a", "delta": 8.0}
        )
        report = self.run_recover(argv)
        self.assertEqual(report["status"], STATUS_RECOVERED)
        self.assertIn("cannot prove", report["guarantee"])

    def test_oracle_timeout_and_non_object_replies_fail_cleanly(self):
        quiet = self.directory / "quiet.py"
        quiet.write_text(
            "import json,sys,time\n"
            "print(json.dumps({'protocol':'rephix-text-oracle/1'}),flush=True)\n"
            "time.sleep(30)\n",
            encoding="utf-8",
        )
        oracle = CommandOracle([sys.executable, str(quiet)], timeout=1)
        with self.assertRaisesRegex(RuntimeError, "did not reply"):
            oracle.sample("abc", 1)
        listy = self.directory / "listy.py"
        listy.write_text(
            "import json,sys\n"
            "print(json.dumps({'protocol':'rephix-text-oracle/1'}),flush=True)\n"
            "for line in sys.stdin:\n    print('[1]',flush=True)\n",
            encoding="utf-8",
        )
        oracle = CommandOracle([sys.executable, str(listy)])
        try:
            with self.assertRaisesRegex(RuntimeError, "JSON object"):
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
            self.run_recover([sys.executable, str(script)], samples_per_context=5, bootstrap=79)


if __name__ == "__main__":
    unittest.main()
