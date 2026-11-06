import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rephix import weight_audit

HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _save(path, **weights):
    from safetensors.torch import save_file

    save_file(weights, path)
    return path


def _write_index(root, weight_map):
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


def _shard(root, *weights):
    root.mkdir()
    weight_map = {}
    for index, tensors in enumerate(weights):
        filename = f"model-{index + 1:05d}-of-{len(weights):05d}.safetensors"
        _save(root / filename, **tensors)
        weight_map.update({name: filename for name in tensors})
    _write_index(root, weight_map)
    return root


class WeightAuditInputTests(unittest.TestCase):
    def test_invalid_tolerances_and_unsupported_file_path(self):
        for name in ("atol", "rtol"):
            for tolerance in (-1, float("nan"), float("inf")):
                with (
                    self.subTest(name=name, tolerance=tolerance),
                    self.assertRaisesRegex(ValueError, f"{name} must be finite and nonnegative"),
                ):
                    weight_audit.compare_checkpoints("unused", "unused", **{name: tolerance})
        with tempfile.TemporaryDirectory() as folder:
            unsupported = Path(folder) / "weights.bin"
            unsupported.touch()
            with self.assertRaisesRegex(ValueError, "Expected a local safetensors file"):
                weight_audit.compare_checkpoints(unsupported, unsupported)


@unittest.skipUnless(HAS_SAFETENSORS and HAS_TORCH, "install rephix[hf] for checkpoint audit tests")
class WeightAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_same_values_across_different_sharding_and_dtypes(self):
        torch = self.torch
        weights = {
            "wide": torch.arange(20).reshape(2, 2, 5).to(torch.bfloat16),
            "short": torch.arange(12).reshape(4, 3),
            "scalar": torch.tensor(-0.25, dtype=torch.float16),
            "empty": torch.empty((0, 2)),
        }
        reference = _shard(
            self.root / "reference",
            {"wide": weights["wide"], "short": weights["short"]},
            {"scalar": weights["scalar"], "empty": weights["empty"]},
        )
        candidate = _shard(
            self.root / "candidate",
            {"wide": weights["wide"].float(), "scalar": weights["scalar"]},
            {"short": weights["short"]},
            {"empty": weights["empty"]},
        )
        with patch.object(weight_audit, "_CHUNK_ELEMENTS", 3):
            report = weight_audit.compare_checkpoints(reference, candidate, atol=0, rtol=0)
        self.assertEqual(report["relative_l2"], 0)
        self.assertEqual(report["zero_filled_relative_l2"], 0)
        self.assertEqual(report["fraction_exactly_equal"], 1)
        self.assertEqual(report["fraction_within_tolerance"], 1)
        self.assertEqual(report["coverage"], 1)
        self.assertEqual(report["reference_elements"], 33)
        self.assertEqual(report["candidate_elements"], 33)
        self.assertEqual(report["reference_tensors"], 4)
        self.assertEqual(report["compared_tensors"], 4)
        self.assertEqual(
            report["dtype_mismatches"], {"wide": {"reference": "BF16", "candidate": "F32"}}
        )

    def test_missing_mismatched_and_extra_tensors_use_full_reference_denominator(self):
        torch = self.torch
        directory = self.root / "reference"
        directory.mkdir()
        _save(
            directory / "model.safetensors",
            match=torch.tensor([3.0, 4.0]),
            missing=torch.tensor([0.0, 12.0]),
            shape=torch.tensor([0.0, 0.0, 5.0]),
        )
        candidate = _save(
            self.root / "candidate.safetensors",
            match=torch.tensor([3.0, 4.5]),
            shape=torch.zeros((3, 1), dtype=torch.float16),
            extra=torch.tensor([1234.0]),
        )
        report = weight_audit.compare_checkpoints(directory, candidate, atol=0.5, rtol=0)
        self.assertIsNone(report["relative_l2"])
        self.assertAlmostEqual(
            report["zero_filled_relative_l2"], math.sqrt(169.25 / 194), places=12
        )
        self.assertAlmostEqual(report["fraction_exactly_equal"], 1 / 7, places=12)
        self.assertAlmostEqual(report["fraction_within_tolerance"], 2 / 7, places=12)
        self.assertAlmostEqual(report["coverage"], 2 / 7, places=12)
        self.assertEqual(report["missing_tensors"], ["missing"])
        self.assertEqual(report["extra_tensors"], ["extra"])
        self.assertEqual(
            report["shape_mismatches"], {"shape": {"reference": [3], "candidate": [3, 1]}}
        )
        self.assertEqual(
            report["dtype_mismatches"], {"shape": {"reference": "F32", "candidate": "F16"}}
        )

    def test_relative_l2_and_relative_tolerance_are_stable_at_extreme_scales(self):
        torch = self.torch
        for scale in (1e-280, 1e280):
            with self.subTest(scale=scale):
                reference = _save(
                    self.root / "reference.safetensors",
                    x=torch.tensor([scale, -2 * scale], dtype=torch.float64),
                )
                candidate = _save(
                    self.root / "candidate.safetensors",
                    x=torch.tensor([0.875 * scale, -scale], dtype=torch.float64),
                )
                with patch.object(weight_audit, "_CHUNK_ELEMENTS", 1):
                    report = weight_audit.compare_checkpoints(
                        reference, candidate, atol=0, rtol=0.2
                    )
                self.assertAlmostEqual(
                    report["relative_l2"], math.hypot(0.125, 1) / math.sqrt(5), places=12
                )
                self.assertEqual(report["fraction_within_tolerance"], 0.5)
                self.assertEqual(report["fraction_exactly_equal"], 0)

    def test_zero_reference_norm(self):
        reference = _save(self.root / "reference.safetensors", x=self.torch.zeros(1))
        candidate = _save(self.root / "candidate.safetensors", x=self.torch.ones(1))
        self.assertEqual(weight_audit.compare_checkpoints(reference, reference)["relative_l2"], 0)
        self.assertIsNone(weight_audit.compare_checkpoints(reference, candidate)["relative_l2"])

    def test_large_integer_buffers_are_not_rounded_before_exact_comparison(self):
        reference = _save(self.root / "reference.safetensors", x=self.torch.tensor([2**60]))
        candidate = _save(self.root / "candidate.safetensors", x=self.torch.tensor([2**60 + 1]))
        if np.finfo(np.longdouble).nmant < 63:
            with self.assertRaisesRegex(ValueError, "cannot compare large integers exactly"):
                weight_audit.compare_checkpoints(reference, candidate)
            return
        report = weight_audit.compare_checkpoints(reference, candidate, atol=0, rtol=0)
        self.assertEqual(report["fraction_exactly_equal"], 0)
        self.assertEqual(report["fraction_within_tolerance"], 0)
        self.assertEqual(report["relative_l2"], 2.0**-60)

    def test_nonfinite_values_are_rejected_even_outside_the_scores(self):
        torch = self.torch
        for side in ("reference", "extra_candidate", "incompatible_candidate"):
            with self.subTest(side=side):
                reference = _save(
                    self.root / "reference.safetensors",
                    weight=torch.tensor([float("nan")]) if side == "reference" else torch.ones(1),
                )
                candidate_weights = {"weight": torch.ones(1)}
                if side == "extra_candidate":
                    candidate_weights["unused"] = torch.tensor([float("inf")])
                if side == "incompatible_candidate":
                    candidate_weights["weight"] = torch.tensor([0.0, float("inf")])
                candidate = _save(self.root / "candidate.safetensors", **candidate_weights)
                with self.assertRaisesRegex(ValueError, r"Non-finite values in .* tensor"):
                    weight_audit.compare_checkpoints(reference, candidate)

    def test_malformed_checkpoint_file(self):
        invalid = self.root / "weights.safetensors"
        invalid.write_bytes(b"malformed checkpoint")
        with self.assertRaisesRegex(ValueError, "Cannot read safetensors shard"):
            weight_audit.compare_checkpoints(invalid, invalid)

    def test_invalid_sharded_index_paths_and_contents(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        actual = _save(checkpoint / "actual.safetensors", actual=self.torch.ones(1))
        for path in ("../outside.safetensors", str(actual), r"C:\outside.safetensors", "file.bin"):
            with self.subTest(path=path):
                _write_index(checkpoint, {"actual": path})
                with self.assertRaisesRegex(ValueError, "invalid shard path"):
                    weight_audit.compare_checkpoints(checkpoint, actual)
        _write_index(checkpoint, {"actual": "missing.safetensors"})
        with self.assertRaisesRegex(ValueError, "Cannot read safetensors shard"):
            weight_audit.compare_checkpoints(checkpoint, actual)
        _write_index(checkpoint, {"stale": actual.name})
        with self.assertRaisesRegex(
            ValueError, r"Missing tensors: .*stale.* Unindexed tensors: .*actual"
        ):
            weight_audit.compare_checkpoints(checkpoint, actual)
        (checkpoint / "model.safetensors.index.json").write_text(
            '{"weight_map":{"actual":"first.safetensors","actual":"actual.safetensors"}}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "duplicate key 'actual'"):
            weight_audit.compare_checkpoints(checkpoint, actual)

    def test_relative_symlink_shards_can_point_to_local_blobs(self):
        checkpoint = self.root / "snapshot"
        checkpoint.mkdir()
        original = _save(self.root / "original.safetensors", tensor=self.torch.tensor([1.0]))
        shard = checkpoint / "link.safetensors"
        try:
            shard.symlink_to(f"../{original.name}")
        except OSError as exc:
            self.skipTest(f"Cannot create a symlink on this platform: {exc}")
        _write_index(checkpoint, {"tensor": shard.name})
        self.assertEqual(weight_audit.compare_checkpoints(checkpoint, original)["relative_l2"], 0)
