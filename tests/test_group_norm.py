"""Exercise affine GroupNorm recovery from native layer outputs."""

import importlib.util
import unittest

from rephix.group_norm import recover_group_norm, validate_group_norm


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs PyTorch")
class GroupNormRecoveryTests(unittest.TestCase):
    def test_grouped_affine_probe_recovers_scale_and_bias(self):
        import torch

        for dtype, atol in ((torch.float32, 2e-7), (torch.float64, 1e-15), (torch.bfloat16, 0)):
            for groups in (1, 2, 8):
                with self.subTest(dtype=dtype, groups=groups), torch.no_grad():
                    source = torch.nn.GroupNorm(groups, 8, dtype=dtype)
                    target = torch.nn.GroupNorm(groups, 8, dtype=dtype)
                    source.weight.copy_(torch.arange(-4, 4) / 4)
                    source.bias.copy_(torch.arange(8) / 8)
                    calls = []

                    def read(inputs, shape):
                        self.assertEqual(inputs.shape, (1, 8, 2))
                        observed = source(inputs)
                        self.assertEqual(observed.shape, shape)
                        calls.append(inputs.clone())
                        return observed

                    validate_group_norm(target)
                    recover_group_norm(torch, target, dict(target.named_parameters()), read)
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(calls[0].count_nonzero().item(), 0)
                    torch.testing.assert_close(
                        calls[1].mean(dim=-1), torch.zeros(1, 8, dtype=dtype)
                    )
                    torch.testing.assert_close(target.bias, source.bias, atol=0, rtol=0)
                    torch.testing.assert_close(target.weight, source.weight, atol=atol, rtol=0)
                    held_out = torch.arange(40, dtype=dtype).reshape(1, 8, 5) / 8
                    torch.testing.assert_close(target(held_out), source(held_out))

    def test_invalid_operator_is_rejected(self):
        import torch

        with self.assertRaisesRegex(ValueError, "affine"):
            validate_group_norm(torch.nn.GroupNorm(2, 4, affine=False))
        for epsilon in (0, -1, float("inf"), float("nan")):
            with self.subTest(epsilon=epsilon), self.assertRaisesRegex(ValueError, "epsilon"):
                validate_group_norm(torch.nn.GroupNorm(2, 4, eps=epsilon))
