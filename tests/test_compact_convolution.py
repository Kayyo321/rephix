"""Verify compact convolution responses preserve recovery and probe inputs."""

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rephix.convolution import (
    COMPACT_CONVOLUTION_OPERATION,
    observe_compact_convolution,
    recover_conv1d,
    recover_conv2d,
    recover_conv3d,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs PyTorch")
class CompactConvolutionTests(unittest.TestCase):
    def _recover_pair(self, rank, geometry, bias, dtype):
        import torch

        options = dict(in_channels=4, out_channels=6, groups=2, bias=bias, dtype=dtype)
        if geometry == "padded":
            options.update(
                kernel_size=(3, 2, 2)[:rank],
                stride=(2, 3, 2)[:rank],
                padding=(3, 4, 2)[:rank],
                dilation=(2, 3, 1)[:rank],
            )
        elif geometry == "same":
            options.update(
                kernel_size=(2, 3, 4)[:rank],
                dilation=(3, 2, 1)[:rank],
                padding="same",
            )
        else:
            options.update(
                kernel_size=(1, 2, 3)[:rank],
                stride=(3, 1, 2)[:rank],
                dilation=(1, 2, 1)[:rank],
                padding="valid",
            )
        cls = getattr(torch.nn, f"Conv{rank}d")
        source, full, compact = (cls(**options) for _ in range(3))
        if rank == 2:
            compact.to(memory_format=torch.channels_last)
        elif rank == 3:
            compact.to(memory_format=torch.channels_last_3d)
        recover = {1: recover_conv1d, 2: recover_conv2d, 3: recover_conv3d}[rank]
        full_inputs, compact_inputs = [], []
        full_scalars = compact_scalars = 0

        def read_full(inputs, shape):
            nonlocal full_scalars
            full_inputs.append(inputs.clone())
            result = source(inputs)
            self.assertEqual(result.shape, shape)
            full_scalars += math.prod(shape)
            return result

        def read_compact(inputs, shape, operation):
            nonlocal compact_scalars
            self.assertEqual(operation, COMPACT_CONVOLUTION_OPERATION)
            compact_inputs.append(inputs.clone())
            result = observe_compact_convolution(torch, source, inputs, operation)
            self.assertEqual(result.shape, shape)
            self.assertEqual(shape, (inputs.shape[0], source.out_channels))
            compact_scalars += math.prod(shape)
            return result

        with torch.no_grad():
            recover(torch, full, dict(full.named_parameters()), read_full, batch_size=5)
            recover(
                torch,
                compact,
                dict(compact.named_parameters()),
                read_compact,
                batch_size=5,
                compact=True,
            )
            self.assertEqual(len(full_inputs), len(compact_inputs))
            for left, right in zip(full_inputs, compact_inputs, strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            for name, value in full.state_dict().items():
                torch.testing.assert_close(compact.state_dict()[name], value, rtol=0, atol=0)
            tolerance = 1e-6 if dtype == torch.float32 else 1e-13
            pixels = torch.randn((2, 4, *(15,) * rank), dtype=dtype)
            torch.testing.assert_close(
                compact(pixels), source(pixels), rtol=tolerance, atol=tolerance
            )
        if geometry == "valid":
            self.assertEqual(compact_scalars, full_scalars)
        else:
            self.assertLess(compact_scalars, full_scalars)
        return full, compact

    def test_all_ranks_preserve_weights_with_smaller_responses(self):
        import torch

        for rank in (1, 2, 3):
            for geometry in ("padded", "same", "valid"):
                for bias in (False, True):
                    for dtype in (torch.float32, torch.float64):
                        with self.subTest(rank=rank, geometry=geometry, bias=bias, dtype=dtype):
                            self._recover_pair(rank, geometry, bias, dtype)

    def test_compact_bias_only_keeps_shared_weight_unchanged(self):
        import torch

        source = torch.nn.Conv2d(2, 4, 3, padding=1).double()
        target = torch.nn.Conv2d(2, 4, 3, padding=1).double()
        previous = target.weight.detach().clone()
        calls = []

        def read(inputs, shape, operation):
            calls.append((shape, operation))
            return observe_compact_convolution(torch, source, inputs, operation)

        with torch.no_grad():
            recover_conv2d(torch, target, {"bias": target.bias}, read, 5, compact=True)
        self.assertEqual(calls, [((1, 4), COMPACT_CONVOLUTION_OPERATION)])
        torch.testing.assert_close(target.weight, previous, rtol=0, atol=0)
        torch.testing.assert_close(target.bias, source.bias, rtol=0, atol=0)

    def test_unknown_compact_operation_fails_before_native_forward(self):
        import torch

        source = torch.nn.Conv1d(1, 1, 3)
        with patch.object(source, "forward", side_effect=AssertionError("unexpected probe")):
            with self.assertRaisesRegex(ValueError, "unknown compact"):
                observe_compact_convolution(torch, source, torch.zeros(1, 1, 3), "other")

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "needs safetensors")
    def test_compact_and_full_exports_have_identical_weight_bytes(self):
        import torch
        from safetensors.torch import save_file

        full, compact = self._recover_pair(2, "same", True, torch.float64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(
                {name: value.contiguous() for name, value in full.state_dict().items()},
                root / "full.safetensors",
            )
            save_file(
                {name: value.contiguous() for name, value in compact.state_dict().items()},
                root / "compact.safetensors",
            )
            self.assertEqual(
                (root / "full.safetensors").read_bytes(),
                (root / "compact.safetensors").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
