"""Failure injection for scientific artifact replacement and finite reports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rephix.io import atomic_save_npz, atomic_write_text, write_json


class AtomicArtifactTests(unittest.TestCase):
    def test_partial_numpy_serialization_preserves_complete_previous_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / "recovered.npz"
            original = np.arange(6).reshape(2, 3)
            atomic_save_npz(destination, weights=original)
            before = destination.read_bytes()

            def partial_write(stream, **arrays):
                stream.write(b"partial ZIP data")
                raise OSError("simulated disk full during NumPy serialization")

            with patch("numpy.savez_compressed", side_effect=partial_write):
                with self.assertRaisesRegex(OSError, "disk full"):
                    atomic_save_npz(destination, weights=np.zeros((2, 3)))
            self.assertEqual(destination.read_bytes(), before)
            with np.load(destination, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["weights"], original)
            self.assertEqual(list(root.iterdir()), [destination])

    def test_object_and_nested_object_arrays_cannot_replace_pickle_free_archive(self):
        invalid = [
            np.array([{"value": 1}], dtype=object),
            np.array([(1, {"value": 2})], dtype=[("number", "i4"), ("value", object)]),
        ]
        for values in invalid:
            with self.subTest(dtype=values.dtype), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                destination = root / "recovered.npz"
                atomic_save_npz(destination, weights=np.eye(2))
                before = destination.read_bytes()
                with self.assertRaisesRegex(ValueError, "object arrays"):
                    atomic_save_npz(destination, weights=values)
                self.assertEqual(destination.read_bytes(), before)
                self.assertEqual(list(root.iterdir()), [destination])

    def test_nonfinite_json_is_rejected_before_the_previous_report_changes(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=invalid), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                destination = root / "report.json"
                write_json(destination, {"complete": True, "loss": 0.5})
                before = destination.read_bytes()
                with self.assertRaises(ValueError):
                    write_json(destination, {"history": [{"loss": invalid}]})
                self.assertEqual(destination.read_bytes(), before)
                self.assertEqual(json.loads(destination.read_text())["loss"], 0.5)
                self.assertEqual(list(root.iterdir()), [destination])

    def test_failed_replace_cleans_temporary_numpy_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / "recovered.npz"
            atomic_save_npz(destination, weights=np.eye(2))
            before = destination.read_bytes()
            with patch("rephix.io.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    atomic_save_npz(destination, weights=np.zeros((2, 2)))
            self.assertEqual(destination.read_bytes(), before)
            self.assertEqual(list(root.iterdir()), [destination])

    def test_failed_first_text_replace_leaves_no_destination_or_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / "report.json"
            with patch("rephix.io.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    atomic_write_text(destination, "complete new report")
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
