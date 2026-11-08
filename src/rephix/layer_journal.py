"""Durable capture and ordered replay of chosen-layer observations."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

_SCHEMA_VERSION = 1


def _digest(metadata, payload):
    digest = hashlib.sha256(json.dumps(metadata, separators=(",", ":")).encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


class LayerJournal:
    """Persist validated responses and replay the same probe sequence in order.

    A capture retries any call whose response was lost before SQLite committed it.
    """

    def __init__(self, path, torch, *, binding, readonly=False):
        self.path = Path(path)
        self.torch = torch
        self.binding = binding
        self.readonly = readonly
        self.reused_calls = 0
        self.fresh_calls = 0
        self._finished = False

    def __enter__(self):
        if sys.byteorder != "little":
            raise ValueError("layer journals require a little-endian host")
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError(f"expected a regular layer journal file: {self.path}")
        uri = self.path.resolve().as_uri() + ("?mode=ro" if self.readonly else "?mode=rwc")
        try:
            self._db = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=60)
        except sqlite3.Error as exc:
            raise ValueError(f"cannot open layer journal {self.path}: {exc}") from exc
        try:
            if self.readonly:
                with self._sql_errors():
                    self._db.execute("BEGIN")
                    if self._total_calls() is None:
                        raise ValueError("layer journal capture is incomplete")
            else:
                with self._sql_errors():
                    self._db.execute("PRAGMA synchronous = FULL")
                with self._write_transaction():
                    self._initialize()
        except BaseException:
            self._db.close()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._db.close()

    @contextmanager
    def _sql_errors(self):
        try:
            yield
        except sqlite3.Error as exc:
            raise ValueError(f"invalid or unavailable layer journal {self.path}: {exc}") from exc

    @contextmanager
    def _write_transaction(self):
        with self._sql_errors(), self._db:
            # Keep the write reservation while calling the oracle. Concurrent
            # writers at the same position will reuse the committed response.
            self._db.execute("BEGIN IMMEDIATE")
            yield

    def _initialize(self):
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS journal_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                schema_version INTEGER NOT NULL,
                binding TEXT NOT NULL,
                total_calls INTEGER CHECK (total_calls >= 0)
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                position INTEGER PRIMARY KEY CHECK (position >= 0),
                path TEXT NOT NULL,
                input_digest TEXT NOT NULL,
                dtype TEXT NOT NULL,
                shape TEXT NOT NULL,
                payload BLOB NOT NULL,
                checksum TEXT NOT NULL
            )
        """)
        self._db.execute(
            """
            INSERT INTO journal_metadata (id, schema_version, binding)
            SELECT 1, ?, ? WHERE NOT EXISTS (SELECT 1 FROM responses)
            ON CONFLICT (id) DO NOTHING
            """,
            (_SCHEMA_VERSION, self.binding),
        )
        self._total_calls()

    def _total_calls(self):
        row = self._db.execute(
            "SELECT schema_version, binding, total_calls FROM journal_metadata WHERE id = 1"
        ).fetchone()
        if row is None or row[0] != _SCHEMA_VERSION:
            raise ValueError("missing or unsupported layer journal metadata")
        if row[1] != self.binding:
            raise ValueError("layer journal binding does not match the requested probe plan")
        return row[2]

    def _payload(self, tensor):
        flat = tensor.detach().cpu().contiguous().reshape(-1)
        # Singleton views can count as contiguous while retaining a nonunit
        # stride. Normalize that stride before reinterpreting elements as bytes.
        flat = flat.as_strided(flat.shape, (1,))
        return memoryview(flat.view(self.torch.uint8).numpy())

    def _check_values(self, values, path, shape, dtype):
        if not isinstance(values, self.torch.Tensor) or tuple(values.shape) != shape:
            raise ValueError(f"{path}: layer response has unexpected shape or type")
        if values.dtype != dtype or values.layout != self.torch.strided:
            raise ValueError(f"{path}: layer response has unexpected dtype or layout")
        values = values.detach().cpu()
        if not self.torch.isfinite(values).all().item():
            raise ValueError(f"{path}: layer response contains non-finite values")
        return values

    def _read(self, position, path, input_digest, shape, dtype, oracle, inputs):
        row = self._db.execute(
            "SELECT path, input_digest, dtype, shape, payload, checksum "
            "FROM responses WHERE position = ?",
            (position,),
        ).fetchone()
        encoded_shape = json.dumps(shape, separators=(",", ":"))
        metadata = (position, path, input_digest, str(dtype), encoded_shape)
        if row is not None:
            stored_path, stored_input_digest, stored_dtype, stored_shape, payload, checksum = row
            if (stored_path, stored_input_digest) != (path, input_digest):
                raise ValueError(f"{path}: layer journal input mismatch at call {position}")
            if (stored_dtype, stored_shape) != (str(dtype), encoded_shape):
                raise ValueError(
                    f"{path}: layer journal output metadata mismatch at call {position}"
                )
            if not isinstance(payload, bytes) or _digest(metadata, payload) != checksum:
                raise ValueError(f"{path}: layer journal checksum mismatch at call {position}")
            values = self.torch.empty(0, dtype=dtype, device="cpu")
            if len(payload) != math.prod(shape) * values.element_size():
                raise ValueError(
                    f"{path}: layer journal payload length mismatch at call {position}"
                )
            if payload:
                values = self.torch.frombuffer(bytearray(payload), dtype=dtype)
            return self._check_values(values.reshape(shape), path, shape, dtype), False

        if self.readonly or oracle is None or self._total_calls() is not None:
            raise ValueError(f"{path}: missing layer journal response at call {position}")
        later = self._db.execute(
            "SELECT 1 FROM responses WHERE position > ? LIMIT 1", (position,)
        ).fetchone()
        if later is not None:
            raise ValueError(f"{path}: layer journal has a missing response before a later call")
        values = self._check_values(oracle(path, inputs), path, shape, dtype)
        payload = self._payload(values)
        self._db.execute(
            "INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
            (*metadata, payload, _digest(metadata, payload)),
        )
        return values, True

    def observe(self, path, inputs, expected_shape, expected_dtype, oracle=None):
        if self._finished:
            raise ValueError("layer journal traversal has finished")
        position = self.reused_calls + self.fresh_calls
        input_digest = _digest(
            (path, str(inputs.dtype), tuple(inputs.shape)), self._payload(inputs)
        )
        transaction = self._sql_errors() if self.readonly else self._write_transaction()
        with transaction:
            values, fresh = self._read(
                position, path, input_digest, tuple(expected_shape), expected_dtype, oracle, inputs
            )
        if fresh:
            self.fresh_calls += 1
        else:
            self.reused_calls += 1
        return values

    def finish(self):
        """Seal a completed capture or verify that replay consumed all responses."""
        count = self.reused_calls + self.fresh_calls
        transaction = self._sql_errors() if self.readonly else self._write_transaction()
        with transaction:
            stored = self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            total = self._total_calls()
            if stored != count or (total is not None and total != count):
                raise ValueError(
                    f"layer journal response count mismatch: consumed {count}, stored {stored}, "
                    f"sealed {total}"
                )
            if total is None:
                self._db.execute(
                    "UPDATE journal_metadata SET total_calls = ? WHERE id = 1", (count,)
                )
        self._finished = True
