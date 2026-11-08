"""Verify metadata-only restoration of mixed and tied parameter precision."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from rephix.storage_dtypes import apply_parameter_dtypes, source_parameter_dtypes

HAS_HF = all(importlib.util.find_spec(name) for name in ("torch", "safetensors"))


@unittest.skipUnless(HAS_HF, "needs PyTorch and safetensors")
class StorageDtypeTests(unittest.TestCase):
    def _model(self):
        import torch

        model = torch.nn.Module()
        model.embedding = torch.nn.Embedding(7, 4, dtype=torch.float64)
        model.head = torch.nn.Linear(4, 7, bias=False, dtype=torch.float64)
        model.head.weight = model.embedding.weight
        model.projection = torch.nn.Linear(4, 3, dtype=torch.float64)
        return model

    def test_mixed_precision_legacy_names_and_tied_alias(self):
        import torch
        from safetensors.torch import save_file

        model = self._model()
        identities = {name: id(value) for name, value in model.named_parameters()}
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "model.safetensors"
            save_file(
                {
                    "head.weight": torch.randn(7, 4, dtype=torch.bfloat16),
                    "legacy.weight": torch.randn(3, 4, dtype=torch.float16),
                    "projection.bias": torch.randn(3, dtype=torch.float32),
                },
                source,
            )
            dtypes = source_parameter_dtypes(
                torch, source, model, {"projection.weight": "legacy.weight"}
            )
            self.assertEqual(
                dtypes,
                {
                    "embedding.weight": "torch.bfloat16",
                    "projection.weight": "torch.float16",
                    "projection.bias": "torch.float32",
                },
            )
            expected = {
                name: parameter.to(getattr(torch, dtypes[name].removeprefix("torch.")))
                for name, parameter in model.named_parameters()
            }
            apply_parameter_dtypes(torch, model, dtypes)
            self.assertIs(model.head.weight, model.embedding.weight)
            for name, parameter in model.named_parameters():
                self.assertEqual(id(parameter), identities[name])
                torch.testing.assert_close(parameter, expected[name], atol=0, rtol=0)

    def test_conflicting_tied_source_dtypes_are_rejected(self):
        import torch
        from safetensors.torch import save_file

        model = self._model()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "model.safetensors"
            save_file(
                {
                    "embedding.weight": torch.ones(7, 4, dtype=torch.float32),
                    "head.weight": torch.ones(7, 4, dtype=torch.float16),
                },
                source,
            )
            with self.assertRaisesRegex(ValueError, "tied source parameters.*conflicting"):
                source_parameter_dtypes(torch, source, model, {})

    def test_invalid_restoration_is_validated_before_mutation(self):
        import torch

        model = self._model()
        cases = (
            {"embedding.weight": "torch.float32", "head.weight": "torch.float16"},
            {"projection.weight": "torch.float32", "projection.bias": "torch.int64"},
            {"projection.weight": "torch.float32", "missing": "torch.float64"},
        )
        for dtypes in cases:
            with self.subTest(dtypes=dtypes):
                with self.assertRaises(ValueError):
                    apply_parameter_dtypes(torch, model, dtypes)
                self.assertTrue(all(p.dtype == torch.float64 for p in model.parameters()))
                self.assertIs(model.head.weight, model.embedding.weight)

    def test_missing_shape_and_unsupported_metadata_are_rejected(self):
        import torch
        from safetensors.torch import save_file

        model = torch.nn.Linear(2, 3, bias=False)
        for values, expected in (
            ({"other": torch.zeros(3, 2)}, "metadata is missing"),
            ({"weight": torch.zeros(2, 3)}, "shape disagrees"),
            ({"weight": torch.zeros(3, 2, dtype=torch.int64)}, "unsupported source"),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "model.safetensors"
                save_file(values, source)
                with self.assertRaisesRegex(ValueError, expected):
                    source_parameter_dtypes(torch, source, model, {})

    def test_explicit_unstored_parameters_are_excluded(self):
        import torch
        from safetensors.torch import save_file

        model = torch.nn.Linear(2, 3)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "model.safetensors"
            save_file({"weight": torch.ones(3, 2, dtype=torch.float16)}, source)
            self.assertEqual(
                source_parameter_dtypes(torch, source, model, {}, unstored_names=("bias",)),
                {"weight": "torch.float16"},
            )


if __name__ == "__main__":
    unittest.main()
