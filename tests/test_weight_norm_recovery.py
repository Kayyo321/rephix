"""Verify original weight-normalization coordinates through native operations."""

import importlib.util
import unittest

from rephix.weight_norm_recovery import (
    observe_weight_norm,
    recover_weight_norm,
    validate_weight_norm,
    weight_norm_probe_dtype,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs PyTorch")
class WeightNormRecoveryTests(unittest.TestCase):
    def test_native_parametrization_recovers_signed_zero_scales_and_directions(self):
        import torch

        for dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
            for dim in (-1, 0, 1, 2):
                with self.subTest(dtype=dtype, dim=dim), torch.no_grad():
                    conv = torch.nn.Conv1d(6, 4, 7, groups=2, dtype=dtype)
                    torch.nn.utils.parametrizations.weight_norm(conv, dim=dim)
                    source = conv.parametrizations.weight
                    source.original0.flatten()[::3] = 0
                    source.original0.flatten()[1::3].neg_()
                    source.original1.data = (
                        source.original1.transpose(0, 1).contiguous().transpose(0, 1)
                    )
                    saved = [value.clone() for value in source.parameters()]
                    target = torch.nn.Conv1d(6, 4, 7, groups=2, dtype=dtype)
                    torch.nn.utils.parametrizations.weight_norm(target, dim=dim)
                    target = target.parametrizations.weight
                    pending = dict(target.named_parameters(recurse=False))
                    validate_weight_norm(target, pending)
                    operations = []
                    expected_dtype = weight_norm_probe_dtype(torch, target)

                    def read(inputs, shape, operation):
                        output = observe_weight_norm(torch, source, inputs, operation)
                        self.assertEqual(output.shape, shape)
                        self.assertEqual(output.dtype, expected_dtype)
                        operations.append(operation)
                        return output

                    recover_weight_norm(torch, target, pending, read, batch_size=3)
                    self.assertEqual(operations, ["wn_scale", "wn_norms", "wn_direction"])
                    for actual, expected in zip(target.parameters(), saved):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for actual, expected in zip(source.parameters(), saved):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_legacy_convolution_recovers_bias_and_preserves_hooks(self):
        import torch

        for dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype), torch.no_grad():
                source = torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(4, 6, 5, groups=2, padding=3, dtype=dtype), dim=2
                )
                source.weight_g[..., 0] = 0
                source.weight_g[..., 1].neg_()
                target = torch.nn.utils.weight_norm(
                    torch.nn.Conv1d(4, 6, 5, groups=2, padding=3, dtype=dtype), dim=2
                )
                pending = dict(target.named_parameters(recurse=False))
                validate_weight_norm(target, pending)
                saved = [value.clone() for value in source.parameters()]
                hooks = dict(source._forward_pre_hooks)

                def read(inputs, shape, operation):
                    output = observe_weight_norm(torch, source, inputs, operation)
                    self.assertEqual(output.shape, shape)
                    self.assertEqual(output.dtype, weight_norm_probe_dtype(torch, target))
                    return output

                recover_weight_norm(torch, target, pending, read, batch_size=3)
                self.assertEqual(source._forward_pre_hooks, hooks)
                for actual, expected in zip(target.parameters(), saved):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_direction_norm_is_rejected(self):
        import torch

        source = torch.nn.Conv1d(2, 4, 3)
        torch.nn.utils.parametrizations.weight_norm(source, dim=2)
        source = source.parametrizations.weight
        with torch.no_grad():
            source.original1[..., 1] = 0

            def read(inputs, shape, operation):
                return observe_weight_norm(torch, source, inputs, operation)

            with self.assertRaisesRegex(ValueError, "zero direction norm"):
                recover_weight_norm(
                    torch, source, dict(source.named_parameters(recurse=False)), read, batch_size=3
                )


if __name__ == "__main__":
    unittest.main()
