"""Verify lossless journal codecs, legacy resume, and bounded corrupt replay."""

import hashlib
import importlib.util
import json
import sqlite3
import tempfile
import threading
import tracemalloc
import unittest
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rephix.layer_journal import LayerJournal

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _tensor_bytes(torch, values):
    # Copy into independent flat storage, including singleton strided views.
    flat = torch.empty(values.numel(), dtype=values.dtype)
    flat.copy_(values.reshape(-1))
    return flat.view(torch.uint8).numpy().tobytes()


def _legacy_digest(metadata, payload):
    digest = hashlib.sha256(json.dumps(metadata, separators=(",", ":")).encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


@unittest.skipUnless(HAS_TORCH, "needs PyTorch")
class LayerJournalCompressionTests(unittest.TestCase):
    def test_adaptive_roundtrip_preserves_bits_shapes_and_byte_counts(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.sqlite3"
            probes = []
            for dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
                torch.int64,
            ):
                base = torch.arange(100, dtype=dtype)
                probes.extend(
                    [
                        (base[7].reshape(()), base[11].reshape(())),
                        (
                            base.as_strided((1, 1), (32, 16), storage_offset=7),
                            base.as_strided((1, 1), (28, 28), storage_offset=11),
                        ),
                        (base[::2], torch.zeros((64, 64), dtype=dtype).transpose(0, 1)),
                        (base[:0], torch.empty((0, 3), dtype=dtype)),
                    ]
                )
            # Negative zero must survive without a numerical conversion.
            probes.append((torch.tensor([0.0]), torch.full((2048,), -0.0)))
            expected_bytes = sum(value.numel() * value.element_size() for _, value in probes)
            with LayerJournal(path, torch, binding="bits") as journal:
                self.assertEqual(journal.schema_version, 2)
                for index, (inputs, response) in enumerate(probes):
                    actual = journal.observe(
                        str(index), inputs, response.shape, response.dtype, lambda *_: response
                    )
                    self.assertEqual(_tensor_bytes(torch, actual), _tensor_bytes(torch, response))
                self.assertEqual(journal.fresh_calls, len(probes))
                self.assertEqual(journal.decoded_payload_bytes, expected_bytes)
                self.assertLess(journal.stored_payload_bytes, expected_bytes)
                stored_bytes = journal.stored_payload_bytes
                journal.finish()
            with sqlite3.connect(path) as db:
                self.assertEqual(
                    {row[0] for row in db.execute("SELECT DISTINCT encoding FROM responses")},
                    {"raw", "zlib"},
                )
                self.assertEqual(
                    db.execute("SELECT SUM(LENGTH(payload)) FROM responses").fetchone()[0],
                    stored_bytes,
                )
            with LayerJournal(path, torch, binding="bits", readonly=True) as journal:
                for index, (inputs, response) in enumerate(probes):
                    actual = journal.observe(str(index), inputs, response.shape, response.dtype)
                    self.assertEqual(_tensor_bytes(torch, actual), _tensor_bytes(torch, response))
                self.assertEqual(journal.fresh_calls, 0)
                self.assertEqual(journal.reused_calls, len(probes))
                self.assertEqual(journal.decoded_payload_bytes, expected_bytes)
                self.assertEqual(journal.stored_payload_bytes, stored_bytes)
                journal.finish()

    def test_compression_can_be_disabled_for_a_new_journal(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.sqlite3"
            response = torch.zeros(4096)
            with LayerJournal(path, torch, binding="raw", compression=False) as journal:
                journal.observe(
                    "layer", response, response.shape, response.dtype, lambda *_: response
                )
                self.assertEqual(journal.stored_payload_bytes, response.numel() * 4)
                self.assertEqual(journal.decoded_payload_bytes, journal.stored_payload_bytes)
                journal.finish()
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute("SELECT encoding FROM responses").fetchone(), ("raw",))
                self.assertEqual(
                    db.execute("SELECT schema_version FROM journal_metadata").fetchone(), (2,)
                )
            with self.assertRaisesRegex(ValueError, "boolean"):
                LayerJournal(path, torch, binding="raw", compression="false")

    def test_schema1_unsealed_capture_resumes_without_migration(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            inputs, response = torch.tensor([7.0]), torch.zeros(1024, dtype=torch.bfloat16)
            input_digest = _legacy_digest(
                ("old", str(inputs.dtype), tuple(inputs.shape)), _tensor_bytes(torch, inputs)
            )
            metadata = (0, "old", input_digest, str(response.dtype), "[1024]")
            payload = _tensor_bytes(torch, response)
            with sqlite3.connect(path) as db:
                db.execute(
                    "CREATE TABLE journal_metadata "
                    "(id INTEGER PRIMARY KEY, schema_version INTEGER, binding TEXT, total_calls INTEGER)"
                )
                db.execute("INSERT INTO journal_metadata VALUES (1, 1, 'legacy', NULL)")
                db.execute(
                    "CREATE TABLE responses (position INTEGER PRIMARY KEY, path TEXT, "
                    "input_digest TEXT, dtype TEXT, shape TEXT, payload BLOB, checksum TEXT)"
                )
                db.execute(
                    "INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (*metadata, payload, _legacy_digest(metadata, payload)),
                )
            calls = []

            def oracle(path, _inputs):
                calls.append(path)
                return response

            with LayerJournal(path, torch, binding="legacy") as journal:
                self.assertEqual(journal.schema_version, 1)
                for layer in ("old", "new"):
                    actual = journal.observe(layer, inputs, response.shape, response.dtype, oracle)
                    self.assertEqual(_tensor_bytes(torch, actual), payload)
                self.assertEqual(calls, ["new"])
                self.assertEqual((journal.fresh_calls, journal.reused_calls), (1, 1))
                self.assertEqual(journal.decoded_payload_bytes, 2 * len(payload))
                self.assertEqual(journal.stored_payload_bytes, 2 * len(payload))
                journal.finish()
            with sqlite3.connect(path) as db:
                self.assertNotIn(
                    "encoding", {row[1] for row in db.execute("PRAGMA table_info(responses)")}
                )
                self.assertEqual(
                    db.execute(
                        "SELECT schema_version, total_calls FROM journal_metadata"
                    ).fetchone(),
                    (1, 2),
                )
                self.assertEqual(
                    [
                        row[0]
                        for row in db.execute("SELECT payload FROM responses ORDER BY position")
                    ],
                    [payload, payload],
                )
            with LayerJournal(path, torch, binding="legacy", readonly=True) as journal:
                for layer in ("old", "new"):
                    actual = journal.observe(layer, inputs, response.shape, response.dtype)
                    self.assertEqual(_tensor_bytes(torch, actual), payload)
                journal.finish()

    def _capture_one(self, path, values):
        import torch

        inputs = torch.tensor([1.0])
        with LayerJournal(path, torch, binding="corruption") as journal:
            journal.observe("layer", inputs, values.shape, values.dtype, lambda *_: values)
            journal.finish()
        return inputs

    def test_rejects_corrupt_compressed_streams_and_encoding_changes(self):
        import torch

        cases = {
            "truncated": lambda payload: ("zlib", payload[:-1]),
            "trailing": lambda payload: ("zlib", payload + b"trailing"),
            "concatenated": lambda payload: ("zlib", payload + zlib.compress(b"extra")),
            "oversized": lambda _payload: ("zlib", zlib.compress(bytes(16385))),
            "undersized": lambda _payload: ("zlib", zlib.compress(bytes(16383))),
            "encoding": lambda payload: ("raw", payload),
            "invalid_stream": lambda _payload: ("zlib", bytes(100)),
        }
        for case, mutate in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "observations.sqlite3"
                values = torch.zeros(4096)
                inputs = self._capture_one(path, values)
                with sqlite3.connect(path) as db:
                    payload = db.execute("SELECT payload FROM responses").fetchone()[0]
                    db.execute("UPDATE responses SET encoding = ?, payload = ?", mutate(payload))
                with LayerJournal(path, torch, binding="corruption", readonly=True) as journal:
                    with self.assertRaisesRegex(ValueError, "corrupt|length mismatch"):
                        journal.observe("layer", inputs, values.shape, values.dtype)
                    self.assertEqual((journal.fresh_calls, journal.reused_calls), (0, 0))
                    self.assertEqual(journal.decoded_payload_bytes, 0)
                    self.assertEqual(journal.stored_payload_bytes, 0)

    def test_checksum_binds_original_bytes_and_encoding(self):
        import torch

        # Re-encoding the same bytes is accepted; changing the codec without
        # recomputing its bound checksum must fail even with valid decoded bytes.
        for mutate_codec in (False, True):
            with (
                self.subTest(mutate_codec=mutate_codec),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "observations.sqlite3"
                values = torch.zeros(4096)
                inputs = self._capture_one(path, values)
                with sqlite3.connect(path) as db:
                    if mutate_codec:
                        db.execute(
                            "UPDATE responses SET encoding = 'raw', payload = ?",
                            (_tensor_bytes(torch, values),),
                        )
                    else:
                        db.execute(
                            "UPDATE responses SET payload = ?",
                            (zlib.compress(_tensor_bytes(torch, values), level=9),),
                        )
                with LayerJournal(path, torch, binding="corruption", readonly=True) as journal:
                    if mutate_codec:
                        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                            journal.observe("layer", inputs, values.shape, values.dtype)
                    else:
                        journal.observe("layer", inputs, values.shape, values.dtype)
                        journal.finish()

    def test_decompression_allocation_is_bounded_by_expected_shape(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.sqlite3"
            values = torch.tensor(0.0, dtype=torch.float64)
            inputs = self._capture_one(path, values)
            payload = zlib.compress(bytes(32 * 1024 * 1024))
            with sqlite3.connect(path) as db:
                db.execute("UPDATE responses SET encoding = 'zlib', payload = ?", (payload,))
            with LayerJournal(path, torch, binding="corruption", readonly=True) as journal:
                tracemalloc.start()
                try:
                    with self.assertRaisesRegex(ValueError, "corrupt|length mismatch"):
                        journal.observe("layer", inputs, values.shape, values.dtype)
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertLess(peak, 2 * 1024 * 1024)

    def test_failed_capture_keeps_committed_prefix_and_counters(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.sqlite3"
            values, inputs = torch.zeros(4096), torch.tensor([1.0])

            def interrupted(*_):
                raise RuntimeError("injected interruption")

            with LayerJournal(path, torch, binding="resume") as journal:
                journal.observe("first", inputs, values.shape, values.dtype, lambda *_: values)
                byte_counts = (journal.decoded_payload_bytes, journal.stored_payload_bytes)
                with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                    journal.observe("second", inputs, values.shape, values.dtype, interrupted)
                self.assertEqual((journal.fresh_calls, journal.reused_calls), (1, 0))
                self.assertEqual(
                    (journal.decoded_payload_bytes, journal.stored_payload_bytes), byte_counts
                )
            with LayerJournal(path, torch, binding="resume") as journal:
                journal.observe("first", inputs, values.shape, values.dtype, interrupted)
                journal.observe("second", inputs, values.shape, values.dtype, lambda *_: values)
                self.assertEqual((journal.fresh_calls, journal.reused_calls), (1, 1))
                self.assertEqual(journal.decoded_payload_bytes, 2 * byte_counts[0])
                self.assertEqual(journal.stored_payload_bytes, 2 * byte_counts[1])
                journal.finish()

    def test_concurrent_writers_share_one_committed_response(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.sqlite3"
            barrier = threading.Barrier(2)
            calls = []
            values, inputs = torch.zeros(4096), torch.tensor([1.0])

            def capture():
                with LayerJournal(path, torch, binding="concurrent") as journal:
                    barrier.wait(timeout=10)

                    def oracle(*_):
                        calls.append(1)
                        return values

                    journal.observe("layer", inputs, values.shape, values.dtype, oracle)
                    journal.finish()
                    return journal.fresh_calls, journal.reused_calls, journal.stored_payload_bytes

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: capture(), range(2)))
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                sorted((fresh, reused) for fresh, reused, _ in results), [(0, 1), (1, 0)]
            )
            self.assertEqual(results[0][2], results[1][2])


if __name__ == "__main__":
    unittest.main()
