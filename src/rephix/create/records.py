"""Create Mode evidence schema: exchanges and passages, with explicit missing context.

The legacy :class:`rephix.evidence.Observation` requires a nonempty prompt. Create Mode
also accepts output-only passages and corpus documents, so it uses its own schema
instead of weakening that invariant. Nothing here fabricates a prompt: an absent
context is recorded as ``missing`` (unknown) or ``none`` (known document start).

Visible reasoning is observed text. Tool results describe the environment and are never
training targets. Tool calls are parsed as data and are never executed.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "rephix-create-evidence/1"
ROLES = ("system", "user", "assistant", "tool")
CONTEXT_STATUSES = ("known", "none", "missing")
CORPUS_SUFFIXES = (".txt", ".md", ".markdown", ".rst", ".tex")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_for_grouping(text: str) -> str:
    """Unicode-normalized, whitespace-collapsed, casefolded. Used for grouping only."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass
class Message:
    role: str
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        data = {"role": self.role}
        for key in ("content", "reasoning", "tool_calls", "tool_call_id", "name"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass
class CreateRecord:
    id: str
    kind: str  # "exchange" or "passage"
    messages: list[Message] = field(default_factory=list)
    text: str | None = None
    context: str | None = None
    context_status: str = "missing"
    joiner: str = ""  # exact text between a known context and the passage
    document_id: str | None = None
    position: int | None = None
    conversation_id: str | None = None
    group_id: str | None = None
    decoding: dict | None = None
    source_signals: dict | None = None
    synthesized: bool = False
    metadata: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {"id": self.id, "kind": self.kind}
        if self.kind == "exchange":
            data["messages"] = [message.to_dict() for message in self.messages]
        else:
            data["text"] = self.text
            data["context_status"] = self.context_status
            if self.context is not None:
                data["context"] = self.context
            if self.joiner:
                data["joiner"] = self.joiner
        for key in (
            "document_id",
            "position",
            "conversation_id",
            "group_id",
            "decoding",
            "source_signals",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.synthesized:
            data["synthesized"] = True
        if self.metadata:
            data["metadata"] = self.metadata
        data["provenance"] = self.provenance
        return data

    # Observable text the model is fitted to produce.
    def target_text(self) -> str:
        if self.kind == "passage":
            return self.text or ""
        parts = []
        for message in self.messages:
            if message.role == "assistant":
                if message.reasoning:
                    parts.append(message.reasoning)
                if message.content:
                    parts.append(message.content)
        return "\n\n".join(parts)

    def assistant_messages(self) -> list[Message]:
        return [message for message in self.messages if message.role == "assistant"]

    def context_key(self) -> str | None:
        """Identity of the known context preceding the final assistant turn."""
        if self.kind == "passage":
            if self.context_status == "missing":
                return None
            return sha256_text("passage\0" + (self.context or "") + "\0" + self.joiner)
        last = max(
            (i for i, message in enumerate(self.messages) if message.role == "assistant"),
            default=None,
        )
        if last is None or last == 0:
            return None
        prefix = [message.to_dict() for message in self.messages[:last]]
        return sha256_text(canonical_json(prefix))


def _require(condition, record_id, message):
    if not condition:
        raise ValueError(f"{record_id}: {message}")


def _parse_tool_calls(value, record_id):
    if value is None:
        return None
    _require(isinstance(value, list) and value, record_id, "tool_calls must be a nonempty list")
    calls = []
    for call in value:
        _require(isinstance(call, dict), record_id, "each tool call must be an object")
        # Accept the common nested {"function": {"name", "arguments"}} shape as data.
        if "function" in call and isinstance(call["function"], dict):
            call = {**call["function"], **({"id": call["id"]} if "id" in call else {})}
        name = call.get("name")
        _require(isinstance(name, str) and name.strip(), record_id, "tool call needs a name")
        arguments = call.get("arguments", {})
        parsed_from_string = False
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
                parsed_from_string = True
            except json.JSONDecodeError:
                # Preserve the exact string; it is evidence even if not valid JSON.
                arguments = {"__unparsed__": arguments}
        _require(isinstance(arguments, dict), record_id, "tool arguments must be an object")
        entry = {"name": name, "arguments": arguments}
        if "id" in call:
            entry["id"] = str(call["id"])
        if parsed_from_string:
            entry["arguments_were_string"] = True
        calls.append(entry)
    return calls


def _parse_message(data, record_id):
    _require(isinstance(data, dict), record_id, "each message must be an object")
    unknown = set(data) - {"role", "content", "reasoning", "tool_calls", "tool_call_id", "name"}
    _require(not unknown, record_id, f"unknown message fields {sorted(unknown)}")
    role = data.get("role")
    _require(role in ROLES, record_id, f"message role must be one of {ROLES}")
    content = data.get("content")
    _require(content is None or isinstance(content, str), record_id, "content must be text")
    reasoning = data.get("reasoning")
    _require(reasoning is None or isinstance(reasoning, str), record_id, "reasoning must be text")
    _require(
        role == "assistant" or reasoning is None,
        record_id,
        "visible reasoning belongs to assistant messages",
    )
    tool_calls = _parse_tool_calls(data.get("tool_calls"), record_id)
    _require(
        role == "assistant" or tool_calls is None,
        record_id,
        "tool calls belong to assistant messages",
    )
    _require(
        content is not None or tool_calls is not None or reasoning is not None,
        record_id,
        "message needs content, reasoning or tool calls",
    )
    return Message(
        role=role,
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
    )


_RECORD_FIELDS = {
    "id",
    "kind",
    "messages",
    "text",
    "context",
    "context_status",
    "joiner",
    "document_id",
    "position",
    "conversation_id",
    "group_id",
    "decoding",
    "source_signals",
    "synthesized",
    "metadata",
}
_LEGACY_FIELDS = {
    "id",
    "prompt",
    "output",
    "thinking",
    "metadata",
    "prompt_token_ids",
    "completion_token_ids",
    "top_logprobs",
}


def _from_legacy(data, provenance):
    """Convert a legacy Observation (nonempty prompt) without changing its meaning."""
    from ..evidence import Observation, validate_observation

    observation = Observation(**data)
    validate_observation(observation)
    _require(observation.output != "", observation.id, "legacy record has no observed output")
    signals = None
    if observation.completion_token_ids is not None:
        signals = {
            "source_prompt_token_ids": observation.prompt_token_ids,
            "source_completion_token_ids": observation.completion_token_ids,
            "source_tokenizer": observation.metadata.get("tokenizer"),
        }
        if observation.top_logprobs is not None:
            signals["source_top_logprobs"] = observation.top_logprobs
    metadata = dict(observation.metadata)
    group = metadata.pop("group_id", None)
    return CreateRecord(
        id=observation.id,
        kind="exchange",
        messages=[
            Message(role="user", content=observation.prompt),
            Message(role="assistant", content=observation.output, reasoning=observation.thinking),
        ],
        group_id=group,
        source_signals=signals,
        metadata=metadata,
        provenance={**provenance, "converted_from": "rephix-observation"},
    )


def parse_record(data: dict, provenance: dict) -> CreateRecord:
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    if "kind" not in data and "prompt" in data and set(data) <= _LEGACY_FIELDS:
        return _from_legacy(data, provenance)
    record_id = data.get("id")
    _require(isinstance(record_id, str) and record_id.strip(), "record", "id must be text")
    unknown = set(data) - _RECORD_FIELDS
    _require(not unknown, record_id, f"unknown fields {sorted(unknown)}")
    kind = data.get("kind")
    _require(kind in ("exchange", "passage"), record_id, "kind must be exchange or passage")
    metadata = data.get("metadata", {})
    _require(isinstance(metadata, dict), record_id, "metadata must be an object")
    for key in ("document_id", "conversation_id", "group_id"):
        value = data.get(key)
        _require(
            value is None or (isinstance(value, str) and value.strip()),
            record_id,
            f"{key} must be nonempty text",
        )
    position = data.get("position")
    _require(
        position is None or (type(position) is int and position >= 0),
        record_id,
        "position must be a nonnegative integer",
    )
    decoding = data.get("decoding")
    _require(decoding is None or isinstance(decoding, dict), record_id, "decoding: object")
    signals = data.get("source_signals")
    _require(signals is None or isinstance(signals, dict), record_id, "source_signals: object")
    synthesized = data.get("synthesized", False)
    _require(isinstance(synthesized, bool), record_id, "synthesized must be a boolean")
    record = CreateRecord(
        id=record_id,
        kind=kind,
        document_id=data.get("document_id"),
        position=position,
        conversation_id=data.get("conversation_id"),
        group_id=data.get("group_id"),
        decoding=decoding,
        source_signals=signals,
        synthesized=synthesized,
        metadata=metadata,
        provenance=provenance,
    )
    if kind == "exchange":
        _require("text" not in data and "context" not in data, record_id, "use messages")
        messages = data.get("messages")
        _require(isinstance(messages, list) and messages, record_id, "messages: nonempty list")
        record.messages = [_parse_message(item, record_id) for item in messages]
        _require(
            any(message.role == "assistant" for message in record.messages),
            record_id,
            "an exchange needs at least one observed assistant message",
        )
        # Tool results must answer a preceding call when ids are given.
        call_ids = set()
        for message in record.messages:
            for call in message.tool_calls or []:
                if "id" in call:
                    call_ids.add(call["id"])
            if message.role == "tool" and message.tool_call_id is not None and call_ids:
                _require(
                    message.tool_call_id in call_ids,
                    record_id,
                    f"tool result {message.tool_call_id!r} answers no preceding call",
                )
    else:
        _require("messages" not in data, record_id, "passages have text, not messages")
        text = data.get("text")
        _require(isinstance(text, str) and text.strip(), record_id, "text must be nonempty")
        record.text = text
        context = data.get("context")
        status = data.get("context_status")
        if status is None:
            status = "known" if context is not None else "missing"
        _require(status in CONTEXT_STATUSES, record_id, f"context_status in {CONTEXT_STATUSES}")
        _require(
            (status == "known") == (context is not None),
            record_id,
            "context is given exactly when context_status is known",
        )
        _require(context is None or isinstance(context, str), record_id, "context must be text")
        record.context = context
        record.context_status = status
        joiner = data.get("joiner", "")
        _require(isinstance(joiner, str), record_id, "joiner must be text")
        _require(not joiner or status == "known", record_id, "joiner needs a known context")
        record.joiner = joiner
    try:
        json.dumps(record.to_dict(), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record_id}: record must contain finite JSON data") from exc
    return record


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_evidence(path: str | Path) -> list[CreateRecord]:
    """Read exchange/passage JSONL (or legacy observation JSONL), keeping raw provenance."""
    path = Path(path)
    records = []
    raw = path.read_bytes()
    for number, line in enumerate(raw.decode("utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        provenance = {
            "file": path.name,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "line": number,
            "raw_sha256": sha256_text(line),
            "origin": "supplied",
        }
        try:
            data = json.loads(line, object_pairs_hook=_unique_object)
            records.append(parse_record(data, provenance))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: no evidence records")
    return records


_HEADING = re.compile(r"^(#{1,6}\s|chapter\s+[0-9ivxlc]+\b|[0-9]+(\.[0-9]+)*\s+[A-Z])", re.I)


def split_sections(text: str, target_chars: int = 1500) -> list[str]:
    """Split a document into sections at paragraph boundaries, preferring headings."""
    paragraphs = [block.strip("\n") for block in re.split(r"\n\s*\n", text) if block.strip()]
    sections, current, size = [], [], 0
    for paragraph in paragraphs:
        starts_heading = bool(_HEADING.match(paragraph.strip()))
        if current and (size >= target_chars or (starts_heading and size >= target_chars // 3)):
            sections.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph)
    if current:
        if sections and size < target_chars // 4:
            sections[-1] = sections[-1] + "\n\n" + "\n\n".join(current)
        else:
            sections.append("\n\n".join(current))
    return sections


def read_corpus(paths, *, section_chars: int = 1500) -> list[CreateRecord]:
    """Read plain-text documents as output-only passages. No prompt is invented."""
    files = []
    for item in paths:
        item = Path(item)
        if item.is_dir():
            files.extend(
                sorted(p for p in item.rglob("*") if p.is_file() and p.suffix in CORPUS_SUFFIXES)
            )
        elif item.is_file():
            files.append(item)
        else:
            raise ValueError(f"corpus path does not exist: {item}")
    if not files:
        raise ValueError("corpus contains no text files")
    records = []
    for file in files:
        raw = file.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{file}: corpus files must be UTF-8 text") from exc
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        digest = hashlib.sha256(raw).hexdigest()
        document_id = f"doc-{digest[:16]}"
        sections = split_sections(text, section_chars)
        if not sections:
            continue
        for index, section in enumerate(sections):
            records.append(
                CreateRecord(
                    id=f"{document_id}-s{index:05d}",
                    kind="passage",
                    text=section,
                    # The preceding sections are known text of the same document.
                    context_status="none" if index == 0 else "known",
                    context=None if index == 0 else "\n\n".join(sections[:index])[-4000:],
                    joiner="" if index == 0 else "\n\n",
                    document_id=document_id,
                    position=index,
                    metadata={
                        "document_end": index == len(sections) - 1,
                        "file": file.name,
                    },
                    provenance={
                        "file": file.name,
                        "file_sha256": digest,
                        "section": index,
                        "origin": "supplied-corpus",
                        "raw_sha256": sha256_text(section),
                    },
                )
            )
    if not records:
        raise ValueError("corpus contains no text")
    return records


def validate_unique(records: list[CreateRecord]) -> None:
    seen = set()
    for record in records:
        if record.id in seen:
            raise ValueError(f"duplicate evidence id: {record.id}")
        seen.add(record.id)


def capability_manifest(records: list[CreateRecord], *, queryable_source: bool = False) -> dict:
    """Which observables are actually present. Nothing is assumed about the source."""
    exchanges = [r for r in records if r.kind == "exchange"]
    passages = [r for r in records if r.kind == "passage"]
    contexts = {}
    for record in records:
        key = record.context_key()
        if key is not None:
            contexts.setdefault(key, []).append(record)
    repeated = {key: items for key, items in contexts.items() if len(items) > 1}
    distinct_repeat_outputs = [
        len({item.target_text() for item in items}) for items in repeated.values()
    ]
    tool_calls = sum(
        1 for r in exchanges for m in r.messages if m.role == "assistant" and m.tool_calls
    )
    tool_results = sum(1 for r in exchanges for m in r.messages if m.role == "tool")
    reasoning = sum(1 for r in exchanges for m in r.messages if m.reasoning)
    with_probabilities = sum(
        1
        for r in records
        if r.source_signals
        and any("logprob" in key for key in r.source_signals)
        and any(r.source_signals.get(key) for key in r.source_signals if "logprob" in key)
    )
    with_source_ids = sum(
        1
        for r in records
        if r.source_signals and r.source_signals.get("source_completion_token_ids") is not None
    )
    internal = sum(
        1 for r in records if r.source_signals and r.source_signals.get("internal_traces")
    )
    decodings = sorted({canonical_json(r.decoding) for r in records if r.decoding is not None})
    tiers = {
        "corpus": bool(passages),
        "passive_exchange": bool(exchanges),
        "structured_agent": bool(tool_calls or tool_results or reasoning),
        "repeated_samples": bool(repeated),
        "source_token_probabilities": bool(with_probabilities),
        "queryable_source": bool(queryable_source),
        "internal_traces": bool(internal),
    }
    return {
        "schema": "rephix-create-capabilities/1",
        "tiers": tiers,
        "counts": {
            "records": len(records),
            "exchanges": len(exchanges),
            "passages": len(passages),
            "passages_missing_context": sum(p.context_status == "missing" for p in passages),
            "documents": len({p.document_id for p in passages if p.document_id}),
            "assistant_tool_call_turns": tool_calls,
            "tool_results": tool_results,
            "assistant_turns_with_visible_reasoning": reasoning,
            "repeated_contexts": len(repeated),
            "repeated_contexts_with_one_distinct_output": sum(
                1 for n in distinct_repeat_outputs if n == 1
            ),
            "with_source_probabilities": with_probabilities,
            "with_source_token_ids": with_source_ids,
            "synthesized_records": sum(r.synthesized for r in records),
            "target_characters": sum(len(r.target_text()) for r in records),
        },
        "declared_decoding": [json.loads(item) for item in decodings],
        "notes": [
            "Visible reasoning is observed text, not activations or the causal computation.",
            "Tool results are environment observations; they are context, never targets.",
            "Source token ids are never used as base token ids without a validated mapping.",
        ]
        + (
            [
                "Every repeated context returned one distinct output: consistent with greedy "
                "decoding or a deterministic wrapper, but that does not prove greedy decoding."
            ]
            if repeated and all(n == 1 for n in distinct_repeat_outputs)
            else []
        ),
    }


def evidence_fingerprint(records: list[CreateRecord]) -> str:
    payload = "\n".join(canonical_json(record.to_dict()) for record in records)
    return sha256_text(payload)
