"""Evidence replacement must preserve the last complete data on write failure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rephix.evidence import Observation, read_observations, write_observations


class AtomicEvidenceTests(unittest.TestCase):
    def test_failed_atomic_replace_keeps_previous_complete_evidence_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / "evidence.jsonl"
            original = Observation(id="old", prompt="old prompt", output="old answer")
            write_observations(destination, [original])
            before = destination.read_bytes()
            replacement = Observation(id="new", prompt="new prompt", output="new answer")
            with patch("rephix.io.os.replace", side_effect=OSError("simulated replace failure")):
                with self.assertRaisesRegex(OSError, "simulated replace failure"):
                    write_observations(destination, [replacement])
            self.assertEqual(destination.read_bytes(), before)
            self.assertEqual(read_observations(destination), [original])
            self.assertEqual(list(root.iterdir()), [destination])

    def test_invalid_replacement_is_rejected_before_previous_evidence_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "evidence.jsonl"
            original = Observation(id="old", prompt="old prompt", output="old answer")
            write_observations(destination, [original])
            before = destination.read_bytes()
            with self.assertRaises(ValueError):
                write_observations(destination, [Observation(id="", prompt="new", output="new")])
            self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
