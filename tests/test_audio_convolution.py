"""Exercise Conv1d recovery against native PyTorch audio operators."""

import importlib.util
import unittest

from rephix.convolution import recover_conv1d, validate_conv1d

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs optional PyTorch dependency")
class Conv1dRecoveryTests(unittest.TestCase):
    def test_strided_grouped_dilated_operator(self):
        import torch

        for bias in (True, False):
            with self.subTest(bias=bias):
                options = dict(
                    in_channels=6,
                    out_channels=9,
                    kernel_size=3,
                    stride=2,
                    padding=3,
                    dilation=2,
                    groups=3,
                    bias=bias,
                )
                source = torch.nn.Conv1d(**options).double()
                candidate = torch.nn.Conv1d(**options).double()
                batches = []

                def read(inputs, shape):
                    self.assertEqual(inputs.shape, (shape[0], 6, 6))
                    self.assertEqual(shape, (shape[0], 9, 4))
                    expected_nnz = 0 if bias and not batches else shape[0] * source.groups
                    self.assertEqual(torch.count_nonzero(inputs).item(), expected_nnz)
                    batches.append(shape[0])
                    result = source(inputs)
                    self.assertEqual(result.shape, shape)
                    return result

                with torch.no_grad():
                    recover_conv1d(
                        torch, candidate, dict(candidate.named_parameters()), read, batch_size=4
                    )
                    torch.testing.assert_close(candidate.weight, source.weight, atol=1e-15, rtol=0)
                    if bias:
                        torch.testing.assert_close(candidate.bias, source.bias, atol=0, rtol=0)
                    samples = torch.randn(2, 6, 23, dtype=torch.float64)
                    torch.testing.assert_close(
                        candidate(samples), source(samples), atol=1e-14, rtol=0
                    )
                self.assertEqual(batches, [1, 4, 2] if bias else [4, 2])

    def test_skips_already_recovered_weight(self):
        import torch

        source = torch.nn.Conv1d(3, 6, kernel_size=1, stride=3, groups=3)
        candidate = torch.nn.Conv1d(3, 6, kernel_size=1, stride=3, groups=3)
        weight = candidate.weight.detach().clone()
        calls = []

        def read(inputs, shape):
            calls.append(shape)
            return source(inputs)

        with torch.no_grad():
            recover_conv1d(torch, candidate, {"bias": candidate.bias}, read, batch_size=5)
        self.assertEqual(calls, [(1, 6, 1)])
        self.assertTrue(torch.equal(candidate.weight, weight))
        self.assertTrue(torch.equal(candidate.bias, source.bias))

    def test_unsupported_padding_fails_before_the_probe(self):
        import torch

        for options in ({"padding": -1}, {"padding_mode": "circular"}):
            with (
                self.subTest(**options),
                self.assertRaisesRegex(ValueError, "Conv1d probes require numeric nonnegative"),
            ):
                validate_conv1d(torch.nn.Conv1d(1, 1, 3, **options))
