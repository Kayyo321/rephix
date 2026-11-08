"""Verify complete transposed-convolution kernels and native normalized storage."""

import importlib.util
import unittest

from rephix.conv_transpose_recovery import recover_conv_transpose, validate_conv_transpose
from rephix.weight_norm_recovery import (
    observe_weight_norm,
    recover_weight_norm,
    weight_norm_probe_dtype,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs PyTorch")
class ConvTransposeRecoveryTests(unittest.TestCase):
    def test_grouped_kernels_with_padding_dilation_and_output_padding(self):
        import torch

        for rank in (1, 2, 3):
            for bias in (False, True):
                with self.subTest(rank=rank, bias=bias), torch.no_grad():
                    cls = getattr(torch.nn, f"ConvTranspose{rank}d")
                    options = dict(
                        in_channels=4,
                        out_channels=6,
                        groups=2,
                        kernel_size=(3, 2, 4)[:rank],
                        stride=(2, 3, 2)[:rank],
                        dilation=(3, 2, 1)[:rank],
                        padding=(5, 4, 3)[:rank],
                        output_padding=(2, 1, 1)[:rank],
                        bias=bias,
                        dtype=torch.float64,
                    )
                    source, target = cls(**options), cls(**options)
                    source.weight.data = source.weight.transpose(0, 1).contiguous().transpose(0, 1)
                    calls = []

                    def read(inputs, shape):
                        values = source(inputs)
                        self.assertEqual(values.shape, shape)
                        calls.append(len(inputs))
                        return values

                    recover_conv_transpose(
                        torch, target, dict(target.named_parameters()), read, batch_size=1
                    )
                    torch.testing.assert_close(
                        target.weight, source.weight, rtol=0, atol=1e-15 if bias else 0
                    )
                    if bias:
                        torch.testing.assert_close(target.bias, source.bias, rtol=0, atol=0)
                    self.assertEqual(calls, [1, 1, 1] if bias else [1, 1])
                    inputs = torch.randn((2, 4, *((9,) * rank)), dtype=torch.float64)
                    torch.testing.assert_close(
                        target(inputs), source(inputs), rtol=1e-13, atol=1e-13
                    )

    def test_pending_parameters_remain_separate(self):
        import torch

        for omitted in ("weight", "bias"):
            with self.subTest(omitted=omitted), torch.no_grad():
                source = torch.nn.ConvTranspose2d(2, 4, (3, 2), stride=2).double()
                target = torch.nn.ConvTranspose2d(2, 4, (3, 2), stride=2).double()
                saved = getattr(target, omitted).clone()
                pending = {
                    name: value for name, value in target.named_parameters() if name != omitted
                }
                calls = []

                def read(inputs, shape):
                    calls.append(len(inputs))
                    return source(inputs)

                recover_conv_transpose(torch, target, pending, read, batch_size=5)
                torch.testing.assert_close(getattr(target, omitted), saved, rtol=0, atol=0)
                for name, value in pending.items():
                    torch.testing.assert_close(value, getattr(source, name), rtol=0, atol=1e-15)
                self.assertEqual(calls, [1] if omitted == "weight" else [1, 2])

    def test_invalid_output_padding_and_sparse_state_are_rejected(self):
        import torch

        layer = torch.nn.ConvTranspose1d(2, 4, 3, stride=2, dilation=3, output_padding=3)
        with self.assertRaisesRegex(ValueError, "output_padding"):
            validate_conv_transpose(layer)
        layer = torch.nn.ConvTranspose2d(2, 4, 3)
        layer.weight = torch.nn.Parameter(layer.weight.to_sparse())
        with self.assertRaisesRegex(ValueError, "dense floating-point"):
            validate_conv_transpose(layer)

    def test_legacy_and_current_weight_normalized_transpose_storage(self):
        import torch

        for legacy in (False, True):
            with self.subTest(legacy=legacy), torch.no_grad():
                apply = (
                    torch.nn.utils.weight_norm
                    if legacy
                    else torch.nn.utils.parametrizations.weight_norm
                )
                source = apply(
                    torch.nn.ConvTranspose1d(4, 6, 5, stride=3, groups=2, padding=4).double(), dim=0
                )
                target = apply(
                    torch.nn.ConvTranspose1d(4, 6, 5, stride=3, groups=2, padding=4).double(), dim=0
                )
                if legacy:
                    source.weight_g[0] = 0
                    source.weight_g[1].neg_()
                    oracle, candidate = source, target
                else:
                    oracle, candidate = (
                        source.parametrizations.weight,
                        target.parametrizations.weight,
                    )
                    oracle.original0[0] = 0
                    oracle.original0[1].neg_()
                    recover_conv_transpose(
                        torch,
                        target,
                        {"bias": target.bias},
                        lambda x, shape: source(x),
                        batch_size=3,
                    )

                def read(inputs, shape, operation):
                    values = observe_weight_norm(torch, oracle, inputs, operation)
                    self.assertEqual(values.shape, shape)
                    self.assertEqual(values.dtype, weight_norm_probe_dtype(torch, candidate))
                    return values

                recover_weight_norm(
                    torch,
                    candidate,
                    dict(candidate.named_parameters(recurse=False)),
                    read,
                    batch_size=3,
                )
                for actual, expected in zip(candidate.parameters(), oracle.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(target.bias, source.bias, rtol=0, atol=0)
                samples = torch.randn(2, 4, 13, dtype=torch.float64)
                torch.testing.assert_close(target(samples), source(samples), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
