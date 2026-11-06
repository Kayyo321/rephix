"""Focused parity tests for the vectorized top-k distillation objective."""

from __future__ import annotations

import importlib.util
import math
import unittest

from rephix.hf import _bucketed_loss as bucketed_loss


def _reference(logits, rows):
    """Original per-row formula, retained here as a small numerical oracle."""
    import torch

    losses = []
    for step, row in zip(logits.float(), rows):
        ids = [entry["token_id"] for entry in row]
        probabilities = torch.tensor(
            [entry["logprob"] for entry in row], dtype=torch.float32, device=step.device
        ).exp()
        mass = probabilities.sum()
        probabilities = probabilities / mass.clamp(min=1.0)
        tail_probability = (1 - probabilities.sum()).clamp(min=0)
        normalizer = torch.logsumexp(step, dim=-1)
        loss = -(probabilities * (step[ids] - normalizer)).sum()
        if len(ids) < step.shape[-1] and tail_probability.item() > 0:
            tail_logits = step.clone()
            tail_logits[ids] = -torch.inf
            loss = loss - tail_probability * (torch.logsumexp(tail_logits, dim=-1) - normalizer)
        losses.append(loss)
    return torch.stack(losses).mean()


@unittest.skipUnless(importlib.util.find_spec("torch"), "install rephix[hf] for torch tests")
class BucketedLossTests(unittest.TestCase):
    def test_ragged_rows_match_reference_loss_and_gradients(self):
        import torch

        torch.manual_seed(19)
        rows = [
            [
                {"token_id": 3, "logprob": math.log(0.42)},
                {"token_id": 1, "logprob": math.log(0.23)},
            ],
            [
                {"token_id": 0, "logprob": math.log(0.1)},
                {"token_id": 1, "logprob": math.log(0.2)},
                {"token_id": 2, "logprob": math.log(0.3)},
                {"token_id": 3, "logprob": math.log(0.4)},
            ],
            [
                {"token_id": 4, "logprob": math.log(0.7)},
            ],
        ]
        actual_logits = torch.randn(3, 5, requires_grad=True)
        reference_logits = actual_logits.detach().clone().requires_grad_(True)
        actual = bucketed_loss(actual_logits, rows)
        expected = _reference(reference_logits, rows)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual_logits.grad, reference_logits.grad, atol=2e-6, rtol=2e-6)

    def test_incomplete_row_with_zero_tail_mass_has_finite_zero_tail_gradient(self):
        import torch

        logits = torch.tensor([[0.2, -0.1, 0.7]], requires_grad=True)
        rows = [[{"token_id": 2, "logprob": 0.0}]]
        loss = bucketed_loss(logits, rows)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        torch.testing.assert_close(
            logits.grad, torch.softmax(logits.detach(), dim=-1) - torch.tensor([[0.0, 0.0, 1.0]])
        )

    def test_invalid_rows_are_rejected(self):
        import torch

        logits = torch.zeros(1, 3)
        invalid = [
            ([[]], "nonempty"),
            ([[{"token_id": 0, "logprob": 0.0}, {"token_id": 0, "logprob": -1.0}]], "unique"),
            ([[{"token_id": 3, "logprob": -1.0}]], "vocabulary"),
            ([[{"token_id": 0, "logprob": 0.1}]], "nonpositive"),
            ([[{"token_id": 0, "logprob": float("nan")}]], "finite"),
            ([[{"token_id": 0, "logprob": 0.0}, {"token_id": 1, "logprob": 0.0}]], "exceeds one"),
        ]
        for rows, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                bucketed_loss(logits, rows)


if __name__ == "__main__":
    unittest.main()
