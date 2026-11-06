"""Exercise the public command boundary and recoverable user errors."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rephix.cli import main
from rephix.evidence import Observation, read_observations, write_observations


class CoreCommandTests(unittest.TestCase):
    def call(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_demo_writes_the_same_report_it_prints(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "reports" / "demo.json"
            status, stdout, stderr = self.call(["demo", "--output", str(output)])
            self.assertEqual(status, 0, stderr)
            self.assertEqual(json.loads(stdout), json.loads(output.read_text()))
            self.assertLess(json.loads(stdout)["full_logits"]["parameter_relative_error"], 1e-12)

    def test_validate_and_split_create_complete_isolated_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "evidence.jsonl", root / "splits"
            records = [
                Observation(id=str(i), prompt=f"prompt {i // 2}", output="answer")
                for i in range(16)
            ]
            write_observations(source, records)
            status, stdout, stderr = self.call(["validate", str(source)])
            self.assertEqual(status, 0, stderr)
            self.assertEqual(json.loads(stdout)["independent_prompt_groups"], 8)
            status, stdout, stderr = self.call(["split", str(source), "--output-dir", str(output)])
            self.assertEqual(status, 0, stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest, json.loads(stdout))
            ids = []
            prompt_owners = {}
            for split_name in ("train", "validation", "test"):
                rows = read_observations(output / f"{split_name}.jsonl")
                self.assertEqual(manifest["splits"][split_name]["observations"], len(rows))
                for row in rows:
                    ids.append(row.id)
                    self.assertIn(prompt_owners.setdefault(row.prompt, split_name), [split_name])
            self.assertEqual(sorted(ids), sorted(record.id for record in records))

    def test_existing_split_destination_cannot_overwrite_source_or_create_partial_files(self):
        for collision in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                source = output / collision
                write_observations(
                    source,
                    [
                        Observation(id=str(i), prompt=f"prompt {i}", output="answer")
                        for i in range(6)
                    ],
                )
                original = source.read_bytes()
                status, _, stderr = self.call(["split", str(source), "--output-dir", str(output)])
                self.assertEqual(status, 1)
                self.assertNotIn("Traceback", stderr)
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(list(output.iterdir()), [source])

    def test_numpy_array_instead_of_archive_is_a_normal_user_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "wrong-format.npy"
            output = root / "recovered.npz"
            np.save(source, np.zeros((4, 2)))
            status, _, stderr = self.call(
                [
                    "recover-linear",
                    str(source),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 1)
            self.assertTrue(stderr.startswith("rephix:"))
            self.assertNotIn("Traceback", stderr)
            self.assertFalse(output.exists())

    def test_archive_missing_observation_arrays_is_a_normal_user_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "wrong-keys.npz"
            output = root / "recovered.npz"
            np.savez(source, values=np.ones((3, 2)))
            status, _, stderr = self.call(
                [
                    "recover-linear",
                    str(source),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 1)
            self.assertIn("features", stderr)
            self.assertFalse(output.exists())

    def test_invalid_demo_request_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "bad-demo.json"
            status, _, stderr = self.call(["demo", "--classes", "1", "--output", str(output)])
            self.assertEqual(status, 1)
            self.assertNotIn("Traceback", stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
