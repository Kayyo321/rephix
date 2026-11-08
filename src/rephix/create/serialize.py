"""Declared serialization of Create Mode evidence into base-token training examples.

Conventions are recorded in the run manifest; none is guessed per record:

* ``chat-template/1``: exchanges rendered with the base tokenizer's own chat template.
* ``rephix-plain-dialogue/1``: exchanges for bases without a chat template.
* ``document-continuation/1``: passages; BOS only at a known document start, EOS only at
  a known document end, context tokens conditioned on but never trained on.

Assistant content, visible reasoning (``<think>...</think>``) and tool calls
(``<tool_call>{json}</tool_call>``) are targets. Tool results, user and system text are
context only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .records import CreateRecord

REASONING_OPEN = "<think>\n"
REASONING_CLOSE = "\n</think>\n\n"
TOOL_OPEN = "<tool_call>\n"
TOOL_CLOSE = "\n</tool_call>"


@dataclass
class Example:
    record_id: str
    input_ids: list[int]
    target_mask: list[int]
    kind: str
    truncated: bool = False

    @property
    def target_tokens(self) -> int:
        return sum(self.target_mask)


def render_assistant(message) -> str:
    parts = []
    if message.reasoning:
        parts.append(REASONING_OPEN + message.reasoning + REASONING_CLOSE)
    if message.content:
        parts.append(message.content)
    for call in message.tool_calls or []:
        payload = {"name": call["name"], "arguments": call["arguments"]}
        block = TOOL_OPEN + json.dumps(payload, ensure_ascii=False, sort_keys=True) + TOOL_CLOSE
        parts.append(("\n" if parts else "") + block)
    return "".join(parts)


def parse_tool_calls(text: str) -> tuple[list[dict], int]:
    """Parse ``<tool_call>`` blocks from generated text. Returns calls and malformed count."""
    calls, malformed = [], 0
    start = 0
    while True:
        open_at = text.find(TOOL_OPEN.strip(), start)
        if open_at < 0:
            break
        close_at = text.find(TOOL_CLOSE.strip(), open_at)
        if close_at < 0:
            malformed += 1
            break
        body = text[open_at + len(TOOL_OPEN.strip()) : close_at].strip()
        try:
            payload = json.loads(body)
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("name"), str)
                or not isinstance(payload.get("arguments", {}), dict)
            ):
                raise ValueError
            calls.append({"name": payload["name"], "arguments": payload.get("arguments", {})})
        except ValueError:
            malformed += 1
        start = close_at + len(TOOL_CLOSE.strip())
    return calls, malformed


def _template_messages(messages):
    rendered = []
    for message in messages:
        if message.role == "assistant":
            rendered.append({"role": "assistant", "content": render_assistant(message)})
        else:
            rendered.append({"role": message.role, "content": message.content or ""})
    return rendered


class Serializer:
    def __init__(self, tokenizer, *, max_tokens: int = 384, context_tokens: int = 96):
        if max_tokens < 32:
            raise ValueError("max_tokens must be at least 32")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.context_tokens = min(context_tokens, max_tokens // 2)
        self.chat = bool(getattr(tokenizer, "chat_template", None))
        self.adds_bos = bool(
            tokenizer.bos_token_id is not None
            and tokenizer("x", add_special_tokens=True)["input_ids"][:1] == [tokenizer.bos_token_id]
        )
        self.notes = []
        if self.chat:
            probe = tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=False
            )
            if "system" in probe:
                self.notes.append(
                    "the base chat template inserts its own default system message when none "
                    "is supplied; this is the base's runtime convention, not observed context"
                )

    def manifest(self) -> dict:
        return {
            "exchange_convention": "chat-template/1" if self.chat else "rephix-plain-dialogue/1",
            "passage_convention": "document-continuation/1",
            "reasoning_wrapper": [REASONING_OPEN, REASONING_CLOSE],
            "tool_call_wrapper": [TOOL_OPEN, TOOL_CLOSE],
            "max_tokens": self.max_tokens,
            "context_tokens": self.context_tokens,
            "bos_at_document_start": self.adds_bos,
            "eos_at_document_end": self.tokenizer.eos_token is not None,
            "loss": "assistant or passage tokens only; context, user, system and tool "
            "results are never targets",
            "notes": self.notes,
        }

    # ------------------------------------------------------------------ exchanges
    def _chat_segments(self, messages):
        """Full rendered text plus [start, end) character spans of assistant targets."""
        tokenizer = self.tokenizer
        rendered = _template_messages(messages)
        full = tokenizer.apply_chat_template(rendered, tokenize=False)
        spans = []
        for index, message in enumerate(messages):
            if message.role != "assistant":
                continue
            prefix = (
                tokenizer.apply_chat_template(
                    rendered[:index], tokenize=False, add_generation_prompt=True
                )
                if index
                else tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=True)
            )
            upto = tokenizer.apply_chat_template(rendered[: index + 1], tokenize=False)
            if not upto.startswith(prefix) or not full.startswith(upto):
                raise ValueError(
                    "the base chat template is not prefix-stable for this conversation; "
                    "cannot locate assistant targets exactly"
                )
            spans.append((len(prefix), len(upto)))
        return full, spans

    def _plain_segments(self, messages):
        text, spans = "", []
        for message in messages:
            if message.role == "assistant":
                text += "Assistant: "
                start = len(text)
                text += render_assistant(message)
                spans.append((start, len(text)))
                text += "\n\n"
            else:
                label = {"system": "System", "user": "User", "tool": "Tool"}[message.role]
                text += f"{label}: {message.content or ''}\n\n"
        return text, spans

    def _encode_spans(self, text, spans, *, append_eos=False):
        tokenizer = self.tokenizer
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        mask = []
        for start, end in encoded["offset_mapping"]:
            inside = any(start >= s and end <= e and end > start for s, e in spans)
            mask.append(1 if inside else 0)
        if append_eos and tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
            mask.append(1)
        return ids, mask

    def encode_exchange(self, record: CreateRecord) -> list[Example]:
        messages = record.messages
        if self.chat:
            try:
                text, spans = self._chat_segments(messages)
                ids, mask = self._encode_spans(text, spans)
            except Exception as exc:  # template errors are surfaced with the record id
                raise ValueError(f"{record.id}: {exc}") from exc
        else:
            text, spans = self._plain_segments(messages)
            ids, mask = self._encode_spans(text, spans, append_eos=True)
        if not any(mask):
            raise ValueError(f"{record.id}: no assistant target tokens after serialization")
        truncated = False
        if len(ids) > self.max_tokens:
            # Keep the end (the latest assistant turn); drop the oldest context.
            ids, mask = ids[-self.max_tokens :], mask[-self.max_tokens :]
            truncated = True
            if not any(mask):
                raise ValueError(f"{record.id}: truncation removed every target token")
        return [Example(record.id, ids, mask, "exchange", truncated)]

    # ------------------------------------------------------------------ passages
    def encode_passage(self, record: CreateRecord) -> list[Example]:
        tokenizer = self.tokenizer
        prefix_ids = []
        if record.context_status == "none" and self.adds_bos:
            prefix_ids = [tokenizer.bos_token_id]
        elif record.context_status == "known" and record.context:
            context = tokenizer(record.context, add_special_tokens=False)["input_ids"]
            prefix_ids = context[-self.context_tokens :]
            if record.joiner:
                # The exact declared text between context and passage (corpus: a blank line).
                prefix_ids = (
                    prefix_ids + tokenizer(record.joiner, add_special_tokens=False)["input_ids"]
                )
        target = tokenizer(record.text, add_special_tokens=False)["input_ids"]
        if record.metadata.get("document_end") and tokenizer.eos_token_id is not None:
            target = target + [tokenizer.eos_token_id]
        examples = []
        window = self.max_tokens - self.context_tokens
        position = 0
        while position < len(target):
            chunk = target[position : position + window]
            if position == 0:
                context = prefix_ids[-self.context_tokens :] if prefix_ids else []
            else:
                context = target[max(0, position - self.context_tokens) : position]
            ids = context + chunk
            mask = [0] * len(context) + [1] * len(chunk)
            examples.append(
                Example(
                    f"{record.id}#w{len(examples)}",
                    ids,
                    mask,
                    "passage",
                    truncated=False,
                )
            )
            position += window
        return examples

    def encode(self, record: CreateRecord) -> list[Example]:
        if record.kind == "exchange":
            return self.encode_exchange(record)
        return self.encode_passage(record)

    def encode_all(self, records) -> list[Example]:
        examples = []
        for record in records:
            examples.extend(self.encode(record))
        return examples

    # ------------------------------------------------------------------ generation
    def generation_prefix(self, record: CreateRecord) -> list[int] | None:
        """Token ids of the known context before the final assistant output, if any."""
        tokenizer = self.tokenizer
        if record.kind == "passage":
            if record.context_status == "missing":
                return None
            if record.context_status == "none":
                return [tokenizer.bos_token_id] if self.adds_bos else None
            context = tokenizer(record.context, add_special_tokens=False)["input_ids"]
            joiner = (
                tokenizer(record.joiner, add_special_tokens=False)["input_ids"]
                if record.joiner
                else []
            )
            return context[-self.context_tokens :] + joiner
        last = max(i for i, m in enumerate(record.messages) if m.role == "assistant")
        if last == 0:
            return None
        return self.prompt_ids(record.messages[:last])

    def prompt_ids(self, messages) -> list[int]:
        tokenizer = self.tokenizer
        if self.chat:
            text = tokenizer.apply_chat_template(
                _template_messages(messages), tokenize=False, add_generation_prompt=True
            )
        else:
            text, _ = self._plain_segments(messages)
            text += "Assistant: "
        return tokenizer(text, add_special_tokens=False)["input_ids"][-self.max_tokens :]

    def prompt_text_ids(self, text: str) -> list[int]:
        """Generation-time prompt for a plain document start (no chat wrapping)."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if self.adds_bos:
            ids = [self.tokenizer.bos_token_id] + ids
        return ids
