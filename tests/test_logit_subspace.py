import unittest

import numpy as np

from rephix.logit_subspace import fit_logit_subspace, log_odds


class LogitSubspaceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(981)
        width, columns = 4, 12
        weights = rng.normal(size=(width, columns))
        bias = rng.normal(size=columns)
        # Log softmax contributes an unknown row-dependent scalar to each token.
        logits = rng.normal(size=(44, width)) @ weights + bias
        self.logprobs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)

    def test_affine_span_and_disjoint_anchor_prediction(self):
        recovered = fit_logit_subspace(self.logprobs[:31])
        self.assertEqual(recovered.rank, 4)
        self.assertEqual(recovered.rank_ceiling, 11)
        held_out = self.logprobs[31:]
        expected = log_odds(held_out)
        np.testing.assert_allclose(recovered.project(held_out), expected, atol=1e-12)
        anchors = np.array([8, 1, 3, 5, 7])
        predicted = recovered.predict(held_out[:, np.r_[0, anchors + 1]], anchors)
        unseen = np.setdiff1d(np.arange(expected.shape[1]), anchors)
        np.testing.assert_allclose(predicted[:, unseen], expected[:, unseen], atol=1e-12)

    def test_missing_evidence_and_underdetermined_anchors_are_rejected(self):
        values = self.logprobs.copy()
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "dense finite"):
            fit_logit_subspace(values)
        recovered = fit_logit_subspace(self.logprobs)
        with self.assertRaisesRegex(ValueError, "do not span"):
            recovered.predict(self.logprobs[:, :4], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "unique fitted contrast"):
            recovered.predict(self.logprobs[:, :4], [0, 0, 1])

    def test_sample_count_limits_observable_rank(self):
        recovered = fit_logit_subspace(self.logprobs[:3])
        self.assertEqual(recovered.rank, 2)
        self.assertEqual(recovered.rank_ceiling, 2)
        error = recovered.project(self.logprobs[3:]) - log_odds(self.logprobs[3:])
        self.assertGreater(np.linalg.norm(error), 0.1)
