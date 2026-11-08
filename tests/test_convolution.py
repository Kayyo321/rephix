"""Exercise Conv2d probes against native PyTorch operators."""

import importlib.util
import unittest

from rephix.convolution import recover_conv1d, recover_conv2d, recover_conv3d, validate_conv2d

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs optional PyTorch dependency")
class Conv2dRecoveryTests(unittest.TestCase):
    def test_grouped_conv2d_recovers_weights_and_native_outputs(self):
        for bias in (True, False):
            with self.subTest(bias=bias):
                self._check_grouped_conv2d(bias)

    def _check_grouped_conv2d(self, bias):
        import torch

        kwargs = dict(
            in_channels=6,
            out_channels=9,
            kernel_size=(3, 2),
            stride=(2, 3),
            dilation=(2, 3),
            padding=(3, 4),
            groups=3,
            bias=bias,
        )
        source = torch.nn.Conv2d(**kwargs).double()
        destination = torch.nn.Conv2d(**kwargs).double().to(memory_format=torch.channels_last)
        batches = []

        def read(inputs, shape):
            result = source(inputs)
            self.assertEqual(inputs.shape, (shape[0], 6, 6, 6))
            self.assertEqual(result.shape, shape)
            self.assertEqual(shape, (shape[0], 9, 4, 4))
            self.assertEqual(
                torch.count_nonzero(inputs).item(),
                0 if bias and not batches else shape[0] * source.groups,
            )
            batches.append(shape[0])
            return result

        with torch.no_grad():
            recover_conv2d(
                torch, destination, dict(destination.named_parameters()), read, batch_size=5
            )
            torch.testing.assert_close(destination.weight, source.weight, atol=1e-15, rtol=0)
            if bias:
                torch.testing.assert_close(destination.bias, source.bias, atol=0, rtol=0)
            inputs = torch.randn(2, 6, 15, 21, dtype=torch.float64)
            torch.testing.assert_close(destination(inputs), source(inputs), atol=1e-14, rtol=0)
        self.assertEqual(batches, [1, 5, 5, 2] if bias else [5, 5, 2])

    def test_conv2d_only_writes_pending_parameters(self):
        for shared_parameter in ("weight", "bias"):
            with self.subTest(shared_parameter=shared_parameter):
                self._check_pending_parameters(shared_parameter)

    def _check_pending_parameters(self, shared_parameter):
        import torch

        source = torch.nn.Conv2d(2, 4, (1, 3), groups=2)
        destination = torch.nn.Conv2d(2, 4, (1, 3), groups=2)
        owner = torch.nn.Conv2d(2, 4, (1, 3), groups=2)
        shared = getattr(owner, shared_parameter)
        setattr(destination, shared_parameter, shared)
        shared_before = shared.detach().clone()
        pending = dict(destination.named_parameters())
        del pending[shared_parameter]
        batches = []

        def read(inputs, shape):
            batches.append(shape[0])
            return source(inputs)

        with torch.no_grad():
            recover_conv2d(torch, destination, pending, read, batch_size=2)

        self.assertIs(getattr(destination, shared_parameter), shared)
        torch.testing.assert_close(shared, shared_before, atol=0, rtol=0)
        for name, parameter in pending.items():
            torch.testing.assert_close(parameter, getattr(source, name), atol=1e-7, rtol=0)
        self.assertEqual(batches, [1] if shared_parameter == "weight" else [1, 2, 1])

    def test_conv2d_rejects_unsupported_padding(self):
        import torch

        for kwargs in (
            {"padding": -1},
            {"padding_mode": "reflect"},
        ):
            with (
                self.subTest(**kwargs),
                self.assertRaisesRegex(ValueError, "numeric nonnegative padding"),
            ):
                validate_conv2d(torch.nn.Conv2d(1, 1, 3, **kwargs))

    def test_same_padding_recovers_even_kernels_and_dilation(self):
        import torch

        for rank, recover in ((1, recover_conv1d), (2, recover_conv2d), (3, recover_conv3d)):
            for bias in (False, True):
                with self.subTest(rank=rank, bias=bias), torch.no_grad():
                    options = dict(
                        in_channels=4,
                        out_channels=6,
                        kernel_size=(2, 3, 4)[:rank],
                        dilation=(3, 2, 1)[:rank],
                        groups=2,
                        padding="same",
                        bias=bias,
                        dtype=torch.float64,
                    )
                    cls = getattr(torch.nn, f"Conv{rank}d")
                    source, target = cls(**options), cls(**options)

                    def read(inputs, shape):
                        output = source(inputs)
                        self.assertEqual(output.shape, shape)
                        self.assertEqual(tuple(output.shape[2:]), tuple(inputs.shape[2:]))
                        return output

                    recover(torch, target, dict(target.named_parameters()), read, batch_size=7)
                    atol = 1e-15 if bias else 0
                    torch.testing.assert_close(target.weight, source.weight, atol=atol, rtol=0)
                    if bias:
                        torch.testing.assert_close(target.bias, source.bias, atol=0, rtol=0)
                    pixels = torch.randn((2, 4, *(9,) * rank), dtype=torch.float64)
                    torch.testing.assert_close(
                        target(pixels), source(pixels), atol=1e-13, rtol=1e-13
                    )

    def test_conv2d_rejects_sparse_weights(self):
        import torch

        module = torch.nn.Conv2d(1, 1, 3)
        module.weight = torch.nn.Parameter(module.weight.to_sparse())
        with self.assertRaisesRegex(ValueError, "dense floating-point parameters"):
            validate_conv2d(module)
