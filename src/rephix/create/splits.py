"""Leakage groups and frozen train/validation/test splits for Create Mode evidence.

Whole documents, conversation families, explicit groups, exact duplicates and
near-duplicates stay in one split. A single-document corpus is split into contiguous
section blocks. The assignment is frozen before fitting or selection.
"""

from __future__ import annotations

import hashlib
import random
import re

from .records import CreateRecord, normalize_for_grouping, sha256_text

NEAR_DUPLICATE_JACCARD = 0.7
_WORD = re.compile(r"\w+", re.UNICODE)


def _shingles(text: str, size: int = 5) -> set[str]:
    words = _WORD.findall(normalize_for_grouping(text))
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def _minhash(shingles: set[str], permutations: int = 64) -> list[int]:
    if not shingles:
        return [0] * permutations
    hashed = [
        int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in shingles
    ]
    mask = (1 << 61) - 1
    result = []
    for seed in range(permutations):
        a = (seed * 0x9E3779B97F4A7C15 + 1) & mask or 1
        b = (seed * 0xC2B2AE3D27D4EB4F + 7) & mask
        result.append(min((a * h + b) & mask for h in hashed))
    return result


def _grouping_text(record: CreateRecord) -> str:
    """Passages: their text. Exchanges: every message, so templated answers to different
    questions are not merged, while repeated or paraphrased conversations are."""
    if record.kind == "passage":
        return record.target_text()
    parts = []
    for message in record.messages:
        parts.append(message.role)
        for value in (message.content, message.reasoning):
            if value:
                parts.append(value)
        for call in message.tool_calls or []:
            parts.append(
                call["name"]
                + " "
                + " ".join(f"{k} {v}" for k, v in sorted(call["arguments"].items()))
            )
    return "\n".join(parts)


class _Union:
    def __init__(self, size):
        self.parent = list(range(size))

    def root(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, left, right):
        a, b = self.root(left), self.root(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def leakage_groups(
    records: list[CreateRecord], *, block_sections: int | None = None
) -> list[list[int]]:
    """Return groups of record indices that must share a split.

    ``block_sections`` splits a single long document into contiguous blocks of that
    many sections instead of keeping the whole document together.
    """
    union = _Union(len(records))
    keys = {}
    single_document = len({r.document_id for r in records if r.kind == "passage"}) == 1 and all(
        r.kind == "passage" for r in records
    )
    for index, record in enumerate(records):
        identities = [("text", sha256_text(normalize_for_grouping(_grouping_text(record))))]
        if record.group_id:
            identities.append(("group", record.group_id))
        if record.conversation_id:
            identities.append(("conversation", record.conversation_id))
        if record.document_id:
            if single_document and block_sections:
                block = (record.position or 0) // block_sections
                identities.append(("document-block", f"{record.document_id}:{block}"))
            else:
                identities.append(("document", record.document_id))
        context = record.context_key()
        if context is not None and record.kind == "exchange":
            # Repeated samples of one context are one family.
            identities.append(("context", context))
        for identity in identities:
            if identity in keys:
                union.join(index, keys[identity])
            else:
                keys[identity] = index
    # Near-duplicates by MinHash banding on word 5-grams, verified by exact Jaccard.
    shingles = [_shingles(_grouping_text(record)) for record in records]
    signatures = [_minhash(s) for s in shingles]
    buckets = {}
    for index, signature in enumerate(signatures):
        if not shingles[index]:
            continue
        for band in range(16):
            key = (band, tuple(signature[band * 4 : band * 4 + 4]))
            buckets.setdefault(key, []).append(index)
    checked = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if (a, b) in checked or union.root(a) == union.root(b):
                    continue
                checked.add((a, b))
                inter = len(shingles[a] & shingles[b])
                if inter / max(1, len(shingles[a] | shingles[b])) >= NEAR_DUPLICATE_JACCARD:
                    union.join(a, b)
    groups = {}
    for index in range(len(records)):
        groups.setdefault(union.root(index), []).append(index)
    return sorted(groups.values(), key=lambda members: members[0])


def assign_splits(
    records: list[CreateRecord],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> dict:
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("validation and test fractions must be in (0, 1)")
    if validation_fraction + test_fraction >= 0.9:
        raise ValueError("validation and test fractions must leave training evidence")
    single_document = all(r.kind == "passage" for r in records) and (
        len({r.document_id for r in records}) == 1
    )
    block = None
    if single_document:
        sections = len(records)
        # About forty contiguous blocks, at least one section each.
        block = max(1, sections // 40)
    groups = leakage_groups(records, block_sections=block)
    if len(groups) < 3:
        raise ValueError(
            f"evidence forms only {len(groups)} independent group(s); at least three are "
            "needed for train/validation/test. Supply more documents, conversations or text."
        )
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    total = sum(len(g) for g in groups)
    validation, test, train = [], [], []
    taken_v = taken_t = 0
    for position in order:
        members = groups[position]
        if not validation or taken_v + len(members) <= validation_fraction * total:
            validation.append(position)
            taken_v += len(members)
        elif not test or taken_t + len(members) <= test_fraction * total:
            test.append(position)
            taken_t += len(members)
        else:
            train.append(position)
    if not train:
        # Keep at least one training group.
        train.append((validation if len(validation) > 1 else test).pop())
    if not train or not validation or not test:
        raise ValueError("evidence groups are too unbalanced for train/validation/test")
    assignment = {}
    group_ids = {}
    for name, positions in (("train", train), ("validation", validation), ("test", test)):
        for position in positions:
            group_name = f"g{position:05d}"
            for index in groups[position]:
                assignment[records[index].id] = name
                group_ids[records[index].id] = group_name
    split = {
        "schema": "rephix-create-splits/1",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "grouping": (
            "connected components of document"
            + (" block" if block else "")
            + ", conversation, group_id, repeated context, normalized duplicate and "
            "near-duplicate (MinHash word 5-gram Jaccard >= 0.7)"
        ),
        "single_document_block_sections": block,
        "groups": len(groups),
        "assignment": assignment,
        "group_of": group_ids,
        "counts": {
            name: sum(1 for value in assignment.values() if value == name)
            for name in ("train", "validation", "test")
        },
        "group_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
    }
    verify_disjoint(records, split)
    return split


def verify_disjoint(records: list[CreateRecord], split: dict) -> None:
    owner = {}
    for record in records:
        name = split["assignment"].get(record.id)
        if name is None:
            raise ValueError(f"record {record.id} has no frozen split")
        keys = [("group", split["group_of"][record.id])]
        if record.document_id and not split.get("single_document_block_sections"):
            keys.append(("document", record.document_id))
        if record.conversation_id:
            keys.append(("conversation", record.conversation_id))
        keys.append(("text", sha256_text(normalize_for_grouping(_grouping_text(record)))))
        for key in keys:
            if key in owner and owner[key] != name:
                raise ValueError(f"leakage across splits: {key[0]} {key[1]}")
            owner[key] = name


def partition(records: list[CreateRecord], split: dict) -> dict[str, list[CreateRecord]]:
    result = {"train": [], "validation": [], "test": []}
    for record in records:
        result[split["assignment"][record.id]].append(record)
    return result
