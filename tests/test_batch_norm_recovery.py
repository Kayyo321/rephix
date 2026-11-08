"""Exercise original BatchNorm affine and persistent-state observations."""

import importlib.util
import unittest

from rephix.batch_norm_recovery import (
    BATCH_NORM_BUFFER_OPERATIONS,
    observe_batch_norm,
    recover_batch_norm,
    recover_batch_norm_buffer,
    validate_batch_norm,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs PyTorch")
class BatchNormRecoveryTests(unittest.TestCase):
    def test_complete_dispatch_recovers_a_buffer_only_layer(self):
        import torch

        from rephix.layer_recovery import recover_from_layer_calls

        with torch.no_grad():
            source = torch.nn.BatchNorm2d(4, affine=False, dtype=torch.float64).train()
            target = torch.nn.BatchNorm2d(4, affine=False, dtype=torch.float64).eval()
            source.running_mean.copy_(torch.tensor([-2, 0, 0.25, 16]))
            source.running_var.copy_(torch.tensor([0, 0.25, 2, 64]))
            source.num_batches_tracked.fill_(2**62 + 7)
            saved = {name: value.clone() for name, value in source.state_dict().items()}

            def read(path, inputs):
                name, separator, operation = path.partition(":")
                self.assertEqual(separator, ":")
                self.assertEqual(name, "")
                return observe_batch_norm(torch, source, inputs, operation)

            report = recover_from_layer_calls(target, read)
            self.assertEqual(report["unique_parameter_elements_recovered"], 0)
            self.assertEqual(report["persistent_buffer_elements_recovered"], 9)
            self.assertEqual(report["layer_calls"], 3)
            self.assertTrue(source.training)
            for name, value in source.state_dict().items():
                torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
                atol = 1e-13 if name == "running_var" else 0
                torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=atol)

    def test_affine_and_persistent_state_recover_without_mutation(self):
        import torch

        for cls in (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d):
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                for affine in (False, True):
                    with (
                        self.subTest(cls=cls.__name__, dtype=dtype, affine=affine),
                        torch.no_grad(),
                    ):
                        source = cls(4, affine=affine, momentum=None, dtype=dtype).train()
                        target = cls(4, affine=affine, dtype=dtype).eval()
                        if affine:
                            source.weight.copy_(torch.tensor([-2, 0, 0.5, 3]))
                            source.bias.copy_(torch.tensor([0.125, -1, 8, 0]))
                        source.running_mean.copy_(torch.tensor([-2, 0, 0.25, 16]))
                        source.running_var.copy_(torch.tensor([0, 0.25, 2, 64]))
                        source.num_batches_tracked.fill_(2**62 + 7)
                        saved = {name: value.clone() for name, value in source.state_dict().items()}
                        calls = []

                        def read(inputs, shape, operation):
                            value = observe_batch_norm(torch, source, inputs, operation)
                            self.assertEqual(value.shape, shape)
                            expected = (
                                torch.int64 if operation == "batch_norm_counter" else torch.float64
                            )
                            self.assertEqual(value.dtype, expected)
                            calls.append(operation)
                            return value

                        validate_batch_norm(target)
                        if affine:
                            recover_batch_norm(torch, target, dict(target.named_parameters()), read)
                        for name, operation in BATCH_NORM_BUFFER_OPERATIONS.items():
                            recover_batch_norm_buffer(
                                torch, target, {"bias": getattr(target, name)}, read, operation
                            )
                        self.assertTrue(source.training)
                        for name, value in source.state_dict().items():
                            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
                        for name, value in target.state_dict().items():
                            atol = 1e-13 if name == "running_var" and dtype == torch.float64 else 0
                            torch.testing.assert_close(value, saved[name], rtol=0, atol=atol)
                        self.assertEqual(len(calls), 5 if affine else 3)
                        source.eval()
                        shape = (
                            2,
                            4,
                            *(
                                (3,)
                                * {
                                    torch.nn.BatchNorm1d: 1,
                                    torch.nn.BatchNorm2d: 2,
                                    torch.nn.BatchNorm3d: 3,
                                }[cls]
                            ),
                        )
                        pixels = torch.randn(shape, dtype=dtype)
                        torch.testing.assert_close(target(pixels), source(pixels))

    def test_untracked_affine_normalization_recovers_parameters(self):
        import torch

        with torch.no_grad():
            source = torch.nn.BatchNorm2d(3, track_running_stats=False).train()
            target = torch.nn.BatchNorm2d(3, track_running_stats=False).eval()
            source.weight.copy_(torch.tensor([-0.25, 0, 16]))
            source.bias.copy_(torch.tensor([8, -0.5, 0.125]))

            def read(inputs, shape, operation):
                result = observe_batch_norm(torch, source, inputs, operation)
                self.assertEqual(result.shape, shape)
                return result

            recover_batch_norm(torch, target, dict(target.named_parameters()), read)
            torch.testing.assert_close(target.weight, source.weight, rtol=0, atol=0)
            torch.testing.assert_close(target.bias, source.bias, rtol=0, atol=0)
            self.assertTrue(source.training)
            self.assertIsNone(source.running_mean)
            pixels = torch.randn(2, 3, 4, 4)
            torch.testing.assert_close(target(pixels), source(pixels))

    def test_invalid_layout_and_observation_are_rejected(self):
        import torch

        for epsilon in (0, -1, float("inf"), float("nan")):
            with self.subTest(epsilon=epsilon), self.assertRaisesRegex(ValueError, "epsilon"):
                validate_batch_norm(torch.nn.BatchNorm1d(3, eps=epsilon))
        with self.assertRaisesRegex(ValueError, "native"):
            validate_batch_norm(torch.nn.GroupNorm(1, 3))
        layer = torch.nn.BatchNorm1d(3)
        with self.assertRaisesRegex(ValueError, "unknown"):
            observe_batch_norm(torch, layer, torch.zeros(1, 3, 1, dtype=torch.float64), "wrong")
