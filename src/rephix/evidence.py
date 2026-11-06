"""Portable observations and prompt-group isolation for reconstruction experiments.

Thinking is optional *observed text*. It is never assumed to expose activations.
Unknown context belongs in metadata and is not silently used as a model input.
"""

from __future__ import annotations

import json
import math
import random
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Observation:
    id: str
    prompt: str
    output: str
    thinking: str | None = None
    metadata: dict = field(default_factory=dict)
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    top_logprobs: list[list[dict]] | None = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


def validate_observation(record: Observation) -> None:
    if not isinstance(record.id, str) or not record.id.strip():
        raise ValueError("observation id must be a nonempty string")
    if not isinstance(record.prompt, str) or not record.prompt.strip():
        raise ValueError(f"{record.id}: prompt must be a nonempty string")
    if not isinstance(record.output, str):
        raise ValueError(f"{record.id}: output must be a string (empty for an unqueried prompt)")
    if record.thinking is not None and not isinstance(record.thinking, str):
        raise ValueError(f"{record.id}: thinking must be observed text or null")
    if not isinstance(record.metadata, dict):
        raise ValueError(f"{record.id}: metadata must be an object")
    group = record.metadata.get("group_id")
    if group is not None and (not isinstance(group, str) or not group.strip()):
        raise ValueError(f"{record.id}: metadata.group_id must be a nonempty string")
    for name in ("prompt_token_ids", "completion_token_ids"):
        values = getattr(record, name)
        if values is not None and (
            not isinstance(values, list)
            or any(type(token) is not int or token < 0 for token in values)
            or (name == "prompt_token_ids" and not values)
        ):
            raise ValueError(f"{record.id}: {name} must contain nonnegative integer token ids")
    if (record.prompt_token_ids is None) != (record.completion_token_ids is None):
        raise ValueError(f"{record.id}: prompt and completion token ids must be provided together")
    if record.top_logprobs is not None:
        if not isinstance(record.top_logprobs, list):
            raise ValueError(f"{record.id}: top_logprobs must be a list of token distributions")
        if record.completion_token_ids is None or len(record.top_logprobs) != len(
            record.completion_token_ids
        ):
            raise ValueError(f"{record.id}: top_logprobs must align with completion_token_ids")
        for entries in record.top_logprobs:
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"{record.id}: every observed top-k distribution must be nonempty")
            seen = set()
            mass = 0.0
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {"token_id", "logprob"}:
                    raise ValueError(f"{record.id}: top-k entries need token_id and logprob")
                token = entry["token_id"]
                prob = entry["logprob"]
                if type(token) is not int or token < 0 or token in seen:
                    raise ValueError(
                        f"{record.id}: top-k token ids must be unique nonnegative ints"
                    )
                if (
                    isinstance(prob, bool)
                    or not isinstance(prob, (int, float))
                    or not math.isfinite(prob)
                    or prob > 1e-7
                ):
                    raise ValueError(f"{record.id}: logprob must be finite and <= 0")
                seen.add(token)
                mass += math.exp(prob)
            if mass > 1.00001:
                raise ValueError(f"{record.id}: top-k probability mass exceeds one")
    try:
        json.dumps(record.to_dict(), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record.id}: observation must contain finite JSON data") from exc


def validate_observations(records: Iterable[Observation]) -> list[Observation]:
    result = list(records)
    seen = set()
    for record in result:
        validate_observation(record)
        if record.id in seen:
            raise ValueError(f"duplicate observation id: {record.id}")
        seen.add(record.id)
    if not result:
        raise ValueError("dataset contains no observations")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_observations(path: str | Path) -> list[Observation]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line, object_pairs_hook=_unique_object)
                if not isinstance(data, dict):
                    raise ValueError("expected a JSON object")
                record = Observation(**data)
                validate_observation(record)
                records.append(record)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return validate_observations(records)


def write_observations(path: str | Path, records: Iterable[Observation]) -> None:
    records = validate_observations(records)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Construct first, so validation/serialization failures cannot truncate the destination.
    payload = "".join(json.dumps(record.to_dict(), allow_nan=False) + "\n" for record in records)
    destination.write_text(payload, encoding="utf-8")


def normalized_prompt(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFC", prompt).split())


def leakage_groups(records: Iterable[Observation]) -> list[list[Observation]]:
    """Union repeated prompts and explicit groups, including transitive connections."""
    records = validate_observations(records)
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    keys = {}
    for index, record in enumerate(records):
        identities = [("prompt", normalized_prompt(record.prompt))]
        if record.metadata.get("group_id") is not None:
            identities.append(("group", record.metadata["group_id"]))
        for identity in identities:
            if identity in keys:
                parents[root(index)] = root(keys[identity])
            keys[identity] = index
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(root(index), []).append(record)
    return list(groups.values())


def assert_disjoint(*datasets: Iterable[Observation]) -> None:
    """Reject repeated ids, normalized prompts, or group_id across dataset boundaries."""
    seen = {}
    for split_index, dataset in enumerate(datasets):
        for record in validate_observations(dataset):
            keys = [("id", record.id), ("prompt", normalized_prompt(record.prompt))]
            if record.metadata.get("group_id") is not None:
                keys.append(("group", record.metadata["group_id"]))
            for key in keys:
                if key in seen and seen[key] != split_index:
                    raise ValueError(f"dataset leakage across splits: {key[0]} {key[1]!r}")
                seen[key] = split_index


def split_observations(
    records: Iterable[Observation],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[Observation]]:
    if not (
        0 < validation_fraction < 1
        and 0 < test_fraction < 1
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError("validation/test fractions must be positive and sum to less than one")
    groups = leakage_groups(records)
    if len(groups) < 3:
        raise ValueError("need at least three independent prompt groups for train/validation/test")
    random.Random(seed).shuffle(groups)
    validation_count = max(1, round(len(groups) * validation_fraction))
    test_count = max(1, round(len(groups) * test_fraction))
    # Keep a train group even for tiny datasets or fractions near one.
    while validation_count + test_count >= len(groups):
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        else:
            test_count -= 1
    boundaries = {
        "validation": groups[:validation_count],
        "test": groups[validation_count : validation_count + test_count],
        "train": groups[validation_count + test_count :],
    }
    splits = {
        name: [record for group in items for record in group] for name, items in boundaries.items()
    }
    assert_disjoint(*splits.values())
    return splits
