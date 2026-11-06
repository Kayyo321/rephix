"""Transactional reconstruction journal and content-addressed recovery artifacts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: bytes):
    return hashlib.sha256(value).hexdigest()


def object_hash(value):
    return digest(encode(value))


def _database(path):
    connection = sqlite3.connect(path, timeout=10)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


class Journal:
    VERSION = 1

    def __init__(self, directory, *, create=False):
        self.directory = Path(directory)
        self.path = self.directory / "session.sqlite"
        if create:
            self.directory.mkdir(parents=True, exist_ok=False)
        elif not self.path.is_file():
            raise ValueError(f"no reconstruction session at {directory}")
        self.db = _database(self.path)
        if create:
            self.db.executescript(
                """
                CREATE TABLE current (state BLOB, checksum TEXT, tail TEXT);
                CREATE TABLE history (seq INTEGER PRIMARY KEY, entry BLOB, checksum TEXT);
                CREATE TABLE evidence (id TEXT PRIMARY KEY, payload BLOB, checksum TEXT);
                CREATE TABLE artifacts (checksum TEXT PRIMARY KEY, payload BLOB);
                CREATE TABLE controls (name TEXT PRIMARY KEY, value INTEGER);
                INSERT INTO controls VALUES ('pause', 0);
                """
            )
            self.db.execute(f"PRAGMA user_version={self.VERSION}")
            self.db.commit()
        elif self.db.execute("PRAGMA user_version").fetchone()[0] != self.VERSION:
            raise ValueError("unsupported reconstruction session schema")

    def close(self):
        self.db.close()

    def size_bytes(self):
        files = (self.path, self.path.with_name(self.path.name + "-wal"))
        return sum(path.stat().st_size for path in files if path.exists())

    @contextmanager
    def runner(self):
        # Separate lock database permits a concurrent pause signal and read-only inspection.
        lock = sqlite3.connect(self.directory / "runner.sqlite", timeout=0)
        try:
            lock.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            lock.close()
            raise ValueError(
                "another reconstruction process holds the session writer lock"
            ) from exc
        try:
            yield
        finally:
            lock.close()

    def pause(self, flag):
        with self.db:
            self.db.execute("UPDATE controls SET value=? WHERE name='pause'", (int(flag),))

    def pause_requested(self):
        return bool(self.db.execute("SELECT value FROM controls WHERE name='pause'").fetchone()[0])

    def load(self):
        row = self.db.execute("SELECT state, checksum, tail FROM current").fetchone()
        if row is None or digest(row[0]) != row[1]:
            raise ValueError("session manifest is missing or its checksum has changed")
        return json.loads(row[0])

    def commit(self, state, action, detail=None, *, evidence=None, artifact=None, retain=None):
        data = encode(state)
        state_hash = digest(data)
        self.db.execute("BEGIN IMMEDIATE")
        with self.db:
            previous = self.db.execute(
                "SELECT seq, checksum FROM history ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            sequence, parent = (previous[0] + 1, previous[1]) if previous else (1, None)
            entry = {
                "schema_version": self.VERSION,
                "sequence": sequence,
                "parent": parent,
                "manifest_sha256": state_hash,
                "action": action,
                "detail": detail or {},
            }
            if evidence is not None:
                evidence_id, value = evidence
                payload = encode(value)
                checksum = digest(payload)
                self.db.execute(
                    "INSERT INTO evidence VALUES (?, ?, ?)", (evidence_id, payload, checksum)
                )
                entry["evidence"] = {"id": evidence_id, "sha256": checksum}
            if artifact is not None:
                checksum = digest(artifact)
                self.db.execute(
                    "INSERT OR IGNORE INTO artifacts VALUES (?, ?)", (checksum, artifact)
                )
                entry["artifact_sha256"] = checksum
            if retain is not None:
                for (checksum,) in self.db.execute("SELECT checksum FROM artifacts").fetchall():
                    if checksum not in retain:
                        self.db.execute("DELETE FROM artifacts WHERE checksum=?", (checksum,))
            entry_data = encode(entry)
            entry_hash = digest(entry_data)
            self.db.execute(
                "INSERT INTO history VALUES (?, ?, ?)", (sequence, entry_data, entry_hash)
            )
            self.db.execute("DELETE FROM current")
            self.db.execute("INSERT INTO current VALUES (?, ?, ?)", (data, state_hash, entry_hash))

    def evidence(self, evidence_id, expected):
        row = self.db.execute(
            "SELECT payload, checksum FROM evidence WHERE id=?", (evidence_id,)
        ).fetchone()
        if row is None or row[1] != expected or digest(row[0]) != expected:
            raise ValueError(f"observation integrity check failed for {evidence_id}")
        return json.loads(row[0])

    def artifact(self, checksum):
        row = self.db.execute(
            "SELECT payload FROM artifacts WHERE checksum=?", (checksum,)
        ).fetchone()
        if row is None or digest(row[0]) != checksum:
            raise ValueError(f"checkpoint integrity check failed for {checksum}")
        return row[0]

    def verify(self, artifacts):
        self.db.execute("BEGIN")
        try:
            return self._verify(artifacts)
        finally:
            self.db.rollback()

    def _verify(self, artifacts):
        state = self.load()
        manifest_hash, tail = self.db.execute("SELECT checksum, tail FROM current").fetchone()
        history = []
        parent = None
        references = set()
        for sequence, payload, checksum in self.db.execute(
            "SELECT seq, entry, checksum FROM history ORDER BY seq"
        ):
            entry = json.loads(payload)
            if (
                sequence != len(history) + 1
                or entry["sequence"] != sequence
                or entry["schema_version"] != self.VERSION
                or entry["parent"] != parent
                or digest(payload) != checksum
            ):
                raise ValueError(f"journal chain integrity check failed at entry {sequence}")
            if "evidence" in entry:
                reference = entry["evidence"]
                self.evidence(reference["id"], reference["sha256"])
                references.add(reference["id"])
            history.append(entry)
            parent = checksum
        if not history or parent != tail or history[-1]["manifest_sha256"] != manifest_hash:
            raise ValueError("journal tail does not match the session manifest")
        if len(references) != self.db.execute("SELECT count(*) FROM evidence").fetchone()[0]:
            raise ValueError("unreferenced observations exist in the evidence store")
        for checksum in artifacts(state):
            self.artifact(checksum)
        return state, history
