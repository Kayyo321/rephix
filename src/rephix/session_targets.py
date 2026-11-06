"""Observable target adapters and a replay-safe controlled agent.

The wire format separates model events, orchestration, and environment input.
A backend may keep its identity and all model internals private.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

_SCHEMA = 1
_MAX_BODY = 8 * 1024 * 1024
_MODES = ("active", "passive")
_SOURCES = ("user", "model", "orchestrator", "environment", "transport")


def _json(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def request_messages(request: dict) -> list[dict]:
    messages = request.get("messages")
    if messages is None and isinstance(request.get("prompt"), str):
        messages = [{"role": "user", "content": request["prompt"]}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("target request needs a nonempty messages array")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ("system", "developer", "user", "assistant", "tool")
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("target messages need a supported role and string content")
    return messages


def validate_request(request: dict) -> None:
    if not isinstance(request, dict) or not all(
        _nonempty(request.get(field)) for field in ("request_id", "session_id")
    ):
        raise ValueError("target request needs request_id and session_id")
    if request.get("mode", "active") not in _MODES:
        raise ValueError("target request mode must be active or passive")
    request_messages(request)
    _json(request).encode()


def request_sha256(request: dict) -> str:
    """Hash the exact validated wire request independently of key order."""
    validate_request(request)
    return _digest(request)


def validate_observation(observation: dict, request: dict | None = None) -> None:
    if (
        not isinstance(observation, dict)
        or type(observation.get("schema_version")) is not int
        or observation["schema_version"] != _SCHEMA
    ):
        raise ValueError("unsupported target observation schema")
    for field in ("request_id", "session_id", "target_epoch"):
        if not _nonempty(observation.get(field)):
            raise ValueError(f"observation needs {field}")
    if request is not None:
        validate_request(request)
        if (
            any(observation[key] != request[key] for key in ("request_id", "session_id"))
            or observation.get("mode") != request.get("mode", "active")
            or observation.get("request_sha256") != _digest(request)
        ):
            raise ValueError("target observation does not match its request")
    if observation.get("mode") not in _MODES or observation.get("status") not in ("ok", "error"):
        raise ValueError("observation needs a supported mode and status")
    if not isinstance(observation.get("completion"), str):
        raise ValueError("observation completion must be a string")
    if "task" in observation and (
        not isinstance(observation["task"], dict)
        or (request is not None and _digest(observation["task"]) != _digest(request.get("task")))
    ):
        raise ValueError("observation task does not match its request")
    if not isinstance(observation.get("capabilities"), dict) or not isinstance(
        observation.get("usage"), dict
    ):
        raise ValueError("observation needs capabilities and usage objects")
    if not re.fullmatch(r"[a-f0-9]{64}", str(observation.get("request_sha256", ""))):
        raise ValueError("observation needs a request SHA-256")
    events = observation.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("observation needs ordered events")
    calls = {}
    completed_calls = set()
    generated_messages = []
    for sequence, event in enumerate(events):
        if (
            not isinstance(event, dict)
            or type(event.get("sequence")) is not int
            or event["sequence"] != sequence
            or event.get("source") not in _SOURCES
        ):
            raise ValueError("observation events must have contiguous sequences and known sources")
        kind, source = event.get("kind"), event["source"]
        if kind == "message":
            role = event.get("role")
            if not _nonempty(role) or not isinstance(event.get("content"), str):
                raise ValueError("message events need a role and content")
            if source == "model":
                if role != "assistant":
                    raise ValueError("generated message must have assistant role")
                generated_messages.append(event["content"])
        elif kind == "tool_call":
            call_id = event.get("call_id")
            if (
                not _nonempty(call_id)
                or call_id in calls
                or not _nonempty(event.get("name"))
                or not isinstance(event.get("arguments"), (dict, str))
                or source not in {"model", "orchestrator"}
            ):
                raise ValueError("invalid or duplicate tool call")
            calls[call_id] = event["name"]
        elif kind == "tool_result":
            call_id = event.get("call_id")
            if (
                not _nonempty(call_id)
                or call_id not in calls
                or call_id in completed_calls
                or event.get("name") != calls[call_id]
                or "result" not in event
                or source != "environment"
            ):
                raise ValueError("tool result must be environment input linked to an earlier call")
            completed_calls.add(call_id)
        elif kind in ("output_delta", "reasoning"):
            if source != "model" or not isinstance(event.get("text"), str):
                raise ValueError(f"{kind} events need model-exposed text")
        elif kind == "error":
            if not isinstance(event.get("message"), str):
                raise ValueError("error events need a message")
        elif kind == "logprobs":
            _validate_logprobs(event)
        elif kind != "metadata":
            raise ValueError(f"unsupported observation event kind {kind!r}")
    if observation["status"] == "ok":
        completion = observation["completion"]
        if generated_messages and generated_messages[-1] != completion:
            raise ValueError("final model message disagrees with completion")
        if not generated_messages and (
            completion
            or not any(e["kind"] == "tool_call" and e["source"] == "model" for e in events)
        ):
            raise ValueError("successful observation needs a model message or a model tool call")
    if observation["status"] == "error" and not any(event["kind"] == "error" for event in events):
        raise ValueError("failed observation needs an error event")
    _json(observation).encode()


def _validate_logprobs(event: dict) -> None:
    if event["source"] != "model":
        raise ValueError("logprobs must be exposed by the model integration")
    aligned = _nonempty(event.get("tokenizer_fingerprint"))
    numeric = aligned or "token_ids" in event
    token_field = "token_ids" if numeric else "tokens"
    # Unknown hosted tokenizers can expose opaque token strings, which are
    # retained as evidence and never treated as IDs in a candidate's vocabulary.
    if not aligned and numeric and event.get("alignment") != "unverified":
        raise ValueError(
            "numeric token IDs without a tokenizer fingerprint need unverified alignment"
        )
    tokens = event.get(token_field)
    distributions = event.get("top_logprobs")
    if (
        not isinstance(tokens, list)
        or not isinstance(distributions, list)
        or len(tokens) != len(distributions)
    ):
        raise ValueError("logprob rows must align with their emitted tokens")

    def valid_token(token):
        return type(token) is int and token >= 0 if numeric else isinstance(token, str)

    if not all(map(valid_token, tokens)):
        raise ValueError("invalid emitted tokens in logprob evidence")
    for row in distributions:
        if not isinstance(row, list) or not row:
            raise ValueError("top logprob rows must be nonempty")
        ids = set()
        mass = 0.0
        for entry in row:
            if not isinstance(entry, dict):
                raise ValueError("invalid top logprob entry")
            token = entry.get("token_id" if numeric else "token")
            logprob = entry.get("logprob")
            if (
                not valid_token(token)
                or (numeric and token in ids)
                or type(logprob) not in (float, int)
                or not math.isfinite(logprob)
                or logprob > 1e-7
            ):
                raise ValueError("invalid top logprob entry")
            ids.add(token)
            mass += math.exp(logprob)
        if mass > 1.00001:
            raise ValueError("top logprob mass exceeds one")


def make_observation(
    request: dict,
    *,
    target_epoch: str,
    events: list[dict],
    completion: str,
    capabilities: dict,
    usage: dict,
    status: str = "ok",
    **fields,
) -> dict:
    """Construct a validated response for a target integration."""
    digest = request_sha256(request)
    observation = {
        "schema_version": _SCHEMA,
        "request_id": request["request_id"],
        "session_id": request["session_id"],
        "request_sha256": digest,
        "target_epoch": target_epoch,
        "mode": request.get("mode", "active"),
        "status": status,
        "events": events,
        "completion": completion,
        "capabilities": capabilities,
        "usage": usage,
        **fields,
    }
    validate_observation(observation, request)
    return observation


def _event(events: list[dict], kind: str, source: str, **fields) -> None:
    events.append({"sequence": len(events), "kind": kind, "source": source, **fields})


def _message_events(messages: list[dict]) -> list[dict]:
    events = []
    for message in messages:
        role = message["role"]
        source = {"user": "user", "tool": "environment"}.get(role, "orchestrator")
        _event(events, "message", source, role=role, content=message["content"])
    return events


def tool_action(event: dict) -> str:
    """Serialize a tool decision without its routing ID or returned result."""
    return _json({"name": event["name"], "arguments": event["arguments"]})


def model_context(events: Iterable[dict]) -> str:
    """Serialize only observable context before a model turn."""
    entries = []
    for event in events:
        kind = event["kind"]
        if kind == "message":
            entries.append({"role": event["role"], "content": event["content"]})
        elif kind == "tool_call":
            entries.append(
                {
                    "role": "assistant",
                    "call_id": event["call_id"],
                    "tool_call": json.loads(tool_action(event)),
                }
            )
        elif kind == "tool_result":
            entries.append(
                {
                    "role": "tool",
                    "call_id": event["call_id"],
                    "name": event["name"],
                    "result": event["result"],
                }
            )
    if len(entries) == 1 and entries[0]["role"] == "user":
        return entries[0]["content"]
    return "\n".join(map(_json, entries))


def training_segments(observation: dict) -> list[tuple[str, str]]:
    """Use tool results as context, then supervise generated calls and messages."""
    validate_observation(observation)
    segments = []
    prior = []
    for event in observation["events"]:
        if event["source"] == "model":
            if event["kind"] == "tool_call":
                segments.append((model_context(prior), tool_action(event)))
            elif event["kind"] == "message":
                segments.append((model_context(prior), event["content"]))
        prior.append(event)
    return segments


class Target(Protocol):
    retry_missing: bool

    def submit(self, request: dict) -> dict: ...

    def reconcile(self, request: dict) -> dict | None: ...

    def iter_passive(self, after: int = 0) -> Iterable[tuple[int, dict]]: ...


class _ReceiptTarget:
    """Durable request and passive receipts for pure, repeatable sandbox actions."""

    retry_missing = True

    def __init__(self, *, receipts: str | Path, epoch: str):
        if not _nonempty(epoch):
            raise ValueError("target epoch must be an opaque nonempty string")
        self.receipts = Path(receipts)
        self.epoch = epoch
        self.receipts.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, _SCHEMA):
                raise ValueError(f"unsupported target receipt database version {version}")
            db.execute(
                "CREATE TABLE IF NOT EXISTS receipts ("
                "cursor INTEGER PRIMARY KEY, session_id TEXT NOT NULL, request_id TEXT NOT NULL, "
                "mode TEXT NOT NULL, request_digest TEXT NOT NULL, response TEXT NOT NULL, "
                "response_digest TEXT NOT NULL, UNIQUE (session_id, request_id))"
            )
            db.execute(f"PRAGMA user_version = {_SCHEMA}")

    def _connect(self):
        db = sqlite3.connect(self.receipts, timeout=60)
        db.execute("PRAGMA synchronous = FULL")
        return db

    @staticmethod
    def _decode(response: str, digest: str, request: dict | None = None) -> dict:
        if hashlib.sha256(response.encode()).hexdigest() != digest:
            raise ValueError("target receipt checksum mismatch")
        data = json.loads(response)
        validate_observation(data, request)
        return data

    def _reconcile(self, db, request: dict) -> dict | None:
        row = db.execute(
            "SELECT request_digest, response, response_digest FROM receipts "
            "WHERE session_id = ? AND request_id = ?",
            (request["session_id"], request["request_id"]),
        ).fetchone()
        if row is None:
            return None
        if row[0] != _digest(request):
            raise ValueError("target request ID reused with different contents")
        return self._decode(row[1], row[2], request)

    def reconcile(self, request: dict) -> dict | None:
        validate_request(request)
        with closing(self._connect()) as db:
            return self._reconcile(db, request)

    def submit(self, request: dict) -> dict:
        validate_request(request)
        with closing(self._connect()) as db, db:
            # Serialize request execution and receipt creation. A process crash rolls
            # this transaction back, and these local tools have no side effects to repeat.
            db.execute("BEGIN IMMEDIATE")
            old = self._reconcile(db, request)
            if old is not None:
                return old
            response = self._run(request)
            validate_observation(response, request)
            db.execute(
                "INSERT INTO receipts (session_id, request_id, mode, request_digest, response, "
                "response_digest) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    request["session_id"],
                    request["request_id"],
                    response["mode"],
                    _digest(request),
                    _json(response),
                    _digest(response),
                ),
            )
            return response

    def iter_passive(self, after: int = 0) -> Iterable[tuple[int, dict]]:
        """Yield increasing receipt cursors; active rows can leave gaps."""
        if type(after) is not int or after < 0:
            raise ValueError("passive cursor must be a nonnegative integer")
        with closing(self._connect()) as db:
            for cursor, response, digest in db.execute(
                "SELECT cursor, response, response_digest FROM receipts "
                "WHERE mode = 'passive' AND cursor > ? ORDER BY cursor",
                (after,),
            ):
                yield cursor, self._decode(response, digest)

    def _run(self, request: dict) -> dict:
        raise NotImplementedError

    def _result(self, request: dict, events: list[dict], completion: str, **fields) -> dict:
        return make_observation(
            request, target_epoch=self.epoch, events=events, completion=completion, **fields
        )


class LocalHFTarget(_ReceiptTarget):
    """Greedy local causal-LM target, with independent tokenizer and visible capabilities."""

    def __init__(
        self,
        *,
        model: str,
        receipts: str | Path,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "float32",
        max_new_tokens: int = 24,
        prompt_mode: str = "auto",
        expose_probabilities: int = 0,
        epoch: str = "local-1",
    ):
        if not _nonempty(str(model)):
            raise ValueError("local model must be a path or cache identifier")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if type(expose_probabilities) is not int or expose_probabilities < 0:
            raise ValueError("expose_probabilities must be nonnegative")
        if prompt_mode not in {"auto", "chat", "text"}:
            raise ValueError("prompt_mode must be auto, chat or text")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16 or bfloat16")
        super().__init__(receipts=receipts, epoch=epoch)
        self.model_source = str(model)
        self.revision = revision
        self.device_name = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.prompt_mode = prompt_mode
        self.expose_probabilities = expose_probabilities
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            from .hf import (
                _deps,
                _device,
                _generation_config,
                _load_tokenizer,
                _tokenizer_fingerprint,
            )

            torch, _, AutoModel, AutoTokenizer = _deps()
            device = _device(torch, self.device_name)
            options = {
                "revision": self.revision,
                "local_files_only": True,
                "trust_remote_code": False,
            }
            tokenizer = _load_tokenizer(AutoTokenizer, self.model_source, **options)
            model = (
                AutoModel.from_pretrained(
                    self.model_source,
                    torch_dtype=getattr(torch, self.dtype),
                    use_safetensors=True,
                    **options,
                )
                .to(device)
                .eval()
            )
            config = _generation_config(model, tokenizer, self.max_new_tokens)
            fingerprint = _tokenizer_fingerprint(tokenizer) if self.expose_probabilities else None
            self._loaded = torch, model, tokenizer, device, config, fingerprint
        return self._loaded

    def _encode(self, messages):
        torch, model, tokenizer, device, _, _ = self._load()
        chat = self.prompt_mode == "chat" or (
            self.prompt_mode == "auto" and tokenizer.chat_template
        )
        if chat:
            if not tokenizer.chat_template:
                raise ValueError("chat mode needs an available local chat template")
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        elif len(messages) == 1 and messages[0]["role"] == "user":
            ids = tokenizer.encode(messages[0]["content"], add_special_tokens=True)
        else:
            transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            ids = tokenizer.encode(f"{transcript}\nassistant:", add_special_tokens=True)
        if not ids:
            start = tokenizer.bos_token_id
            if start is None:
                start = tokenizer.eos_token_id
            if start is None:
                raise ValueError("target requires a nonempty encoded context")
            ids = [start]
        limit = getattr(model.config, "max_position_embeddings", None)
        if type(limit) is int and len(ids) + self.max_new_tokens > limit:
            raise ValueError(f"target context and generation exceed context limit of {limit}")
        return torch.tensor([ids], dtype=torch.long, device=device)

    def _generate(self, messages: list[dict], events: list[dict]):
        from transformers import TextStreamer

        torch, model, tokenizer, _, config, fingerprint = self._load()
        ids = self._encode(messages)

        class CaptureStreamer(TextStreamer):
            def on_finalized_text(self, text: str, stream_end: bool = False):
                if text:
                    _event(events, "output_delta", "model", text=text)

        streamer = CaptureStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        with torch.inference_mode():
            output = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                generation_config=config,
                streamer=streamer,
                return_dict_in_generate=True,
                output_scores=bool(self.expose_probabilities),
            )
        emitted = output.sequences[0, ids.shape[1] :].tolist()
        completion = tokenizer.decode(
            emitted, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if self.expose_probabilities:
            rows = []
            for score in output.scores:
                logprob = score[0].float().log_softmax(dim=-1)
                values, indices = logprob.topk(min(len(logprob), self.expose_probabilities))
                rows.append(
                    [
                        {"token_id": index, "logprob": value}
                        for index, value in zip(indices.tolist(), values.tolist(), strict=True)
                        if math.isfinite(value)
                    ]
                )
            _event(
                events,
                "logprobs",
                "model",
                token_ids=emitted,
                top_logprobs=rows,
                tokenizer_fingerprint=fingerprint,
                space="generation_scores",
            )
        eos = config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        finish_reason = "stop" if emitted and emitted[-1] in eos_ids else "length"
        return completion, {
            "prompt_tokens": int(ids.shape[1]),
            "output_tokens": len(emitted),
            "finish_reason": finish_reason,
        }

    def _capabilities(self) -> dict:
        return {
            "scope": "model",
            "messages": True,
            "streamed_output": True,
            "tool_events": False,
            "token_probabilities": bool(self.expose_probabilities),
            "reasoning": "unavailable",
            "internals": "unavailable",
            "identity_metadata": "withheld",
        }

    def _run(self, request: dict) -> dict:
        messages = request_messages(request)
        events = _message_events(messages)
        completion, usage = self._generate(messages, events)
        _event(events, "message", "model", role="assistant", content=completion)
        return self._result(
            request, events, completion, capabilities=self._capabilities(), usage=usage
        )

    def _choose(self, messages: list[dict], choices: tuple[str, ...]) -> tuple[int, int]:
        """Select a single constrained action by the model's next-token logit."""
        torch, model, tokenizer, _, _, _ = self._load()
        ids = self._encode(messages)
        encoded = [tokenizer.encode(choice, add_special_tokens=False) for choice in choices]
        if any(len(tokens) != 1 for tokens in encoded) or len(
            {tokens[0] for tokens in encoded}
        ) != len(choices):
            raise ValueError("sandbox action labels must be distinct single tokens")
        choice_ids = [tokens[0] for tokens in encoded]
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1]
            choice = int(logits[choice_ids].argmax().item())
        return choice, int(ids.shape[1])


def sandbox_options(a: int, b: int, previous: int | None = None) -> list[dict]:
    """The disclosed action space for the controlled, side-effect-free tools."""
    if previous is None:
        names = ("add", "multiply", "subtract")
        arguments = {"a": a, "b": b}
        stage = "first"
    else:
        names = ("double", "negate", "format")
        arguments = {"value": previous}
        stage = "second"
    order = sorted(names, key=lambda name: _digest([a, b, stage, name]))
    return [{"name": name, "arguments": dict(arguments)} for name in order]


def sandbox_execute(name: str, arguments: dict) -> int | str:
    """Execute the sandbox's allowlisted pure integer operations."""
    binary = {
        "add": lambda a, b: a + b,
        "multiply": lambda a, b: a * b,
        "subtract": lambda a, b: a - b,
    }
    unary = {"double": lambda value: 2 * value, "negate": lambda value: -value, "format": str}
    if not isinstance(arguments, dict) or any(
        type(value) is not int or abs(value) > 100_000_000 for value in arguments.values()
    ):
        raise ValueError("sandbox tool arguments must be bounded integers")
    if name in binary and set(arguments) == {"a", "b"}:
        return binary[name](arguments["a"], arguments["b"])
    if name in unary and set(arguments) == {"value"}:
        return unary[name](arguments["value"])
    raise ValueError("unknown sandbox tool or invalid arguments")


_SANDBOX_LABELS = ("A", "B", "C")


def sandbox_rollout(request: dict, choose, complete) -> tuple[list[dict], str]:
    """Run two observable model decisions and one final model response.

    choose receives the events so far and the three executable actions, and
    returns the chosen index. complete receives the updated observable events.
    The same wrapper can run against an independently trained candidate.
    """
    validate_request(request)
    task = request.get("task")
    if (
        not isinstance(task, dict)
        or set(task) != {"a", "b"}
        or any(type(value) is not int or abs(value) > 10_000 for value in task.values())
    ):
        raise ValueError(
            "sandbox requests need task.a and task.b integers between -10000 and 10000"
        )
    events = _message_events(request_messages(request))
    previous = None
    for turn in (1, 2):
        options = sandbox_options(task["a"], task["b"], previous)
        choices = "\n".join(
            f"{label}. {tool_action(option)}"
            for label, option in zip(_SANDBOX_LABELS, options, strict=True)
        )
        _event(
            events,
            "message",
            "orchestrator",
            role="user",
            content=f"Choose tool {turn} for the original request. Reply A, B, or C.\n{choices}",
        )
        index = choose(events, options)
        if type(index) is not int or not 0 <= index < len(options):
            raise ValueError("sandbox model returned an invalid action index")
        action = options[index]
        call_id = f"call-{turn}"
        _event(
            events,
            "tool_call",
            "model",
            call_id=call_id,
            selected_option=_SANDBOX_LABELS[index],
            **action,
        )
        previous = sandbox_execute(action["name"], action["arguments"])
        _event(
            events,
            "tool_result",
            "environment",
            call_id=call_id,
            name=action["name"],
            result=previous,
        )
    _event(
        events,
        "message",
        "orchestrator",
        role="user",
        content="Use the sandbox results to answer the original request. Give only the final answer.",
    )
    completion = complete(events)
    _event(events, "message", "model", role="assistant", content=completion)
    return events, completion


def _sandbox_messages(events: list[dict]) -> list[dict]:
    """The controlled wrapper's disclosed translation into model messages."""
    messages = []
    for event in events:
        if event["kind"] == "message":
            messages.append({"role": event["role"], "content": event["content"]})
        elif event["kind"] == "tool_call":
            messages.append({"role": "assistant", "content": tool_action(event)})
        elif event["kind"] == "tool_result":
            messages.append(
                {"role": "user", "content": f"Sandbox result: {_json(event['result'])}"}
            )
    return messages


class SandboxAgentTarget(LocalHFTarget):
    """An HF model controls sandbox tool decisions, arguments and final text."""

    def _run(self, request: dict) -> dict:
        if "task" not in request:
            return super()._run(request)
        decision_inputs = []
        usage = {}

        def choose(events, _options):
            index, tokens = self._choose(_sandbox_messages(events), _SANDBOX_LABELS)
            decision_inputs.append(tokens)
            return index

        def complete(events):
            completion, final_usage = self._generate(_sandbox_messages(events), events)
            usage.update(final_usage)
            return completion

        events, completion = sandbox_rollout(request, choose, complete)
        usage.update(
            {
                "model_calls": 3,
                "decision_tokens": len(decision_inputs),
                "decision_prompt_tokens": sum(decision_inputs),
            }
        )
        capabilities = {
            **self._capabilities(),
            "scope": "model_and_agent_wrapper",
            "tool_events": True,
            "sandbox": "bounded_integer_tools_v1",
            "action_constraint": "one model-selected label per turn; tools and operands supplied",
            "wrapper": "two tool turns and a final message, without memory or retrieval",
            "token_probability_scope": "final_message"
            if self.expose_probabilities
            else "unavailable",
        }
        return self._result(
            request,
            events,
            completion,
            capabilities=capabilities,
            usage=usage,
            task=request["task"],
        )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _read_wire(handle, maximum=_MAX_BODY):
    data = handle.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("target HTTP payload exceeds size limit")
    return json.loads(data, object_pairs_hook=_unique_wire_keys)


def _unique_wire_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate wire key {key!r}")
        obj[key] = value
    return obj


class HTTPTarget:
    """A client for an authorized agent exposing the observable Rephix protocol."""

    retry_missing = False

    def __init__(self, *, url: str, timeout: float = 60, token_env: str | None = None):
        endpoint = urlsplit(url)
        if (
            not endpoint.hostname
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
            or endpoint.scheme not in {"https", "http"}
            or (
                endpoint.scheme == "http"
                and endpoint.hostname not in {"127.0.0.1", "::1", "localhost"}
            )
        ):
            raise ValueError(
                "target endpoint must use HTTPS or loopback HTTP without URL credentials"
            )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("target HTTP timeout must be positive and finite")
        self.url = endpoint.geturl().rstrip("/")
        self.timeout = timeout
        self.token_env = token_env
        handlers = [_NoRedirect()]
        if endpoint.hostname in {"127.0.0.1", "::1", "localhost"}:
            handlers.append(ProxyHandler({}))
        self._opener = build_opener(*handlers)

    def _call(self, path: str, value: dict | None = None):
        headers = {"Accept": "application/json"}
        payload = None
        if value is not None:
            headers["Content-Type"] = "application/json"
            payload = _json(value).encode()
            if len(payload) > _MAX_BODY:
                raise ValueError("target HTTP request exceeds size limit")
        if self.token_env:
            token = os.environ.get(self.token_env)
            if not token:
                raise ValueError(f"target API token environment variable {self.token_env} is unset")
            headers["Authorization"] = f"Bearer {token}"
        request = Request(f"{self.url}{path}", data=payload, headers=headers)
        with self._opener.open(request, timeout=self.timeout) as response:
            return _read_wire(response)

    def submit(self, request: dict) -> dict:
        validate_request(request)
        path = "/v1/interactions" if request.get("mode") == "passive" else "/v1/requests"
        response = self._call(path, request)
        validate_observation(response, request)
        return response

    def reconcile(self, request: dict) -> dict | None:
        validate_request(request)
        try:
            response = self._call("/v1/reconcile", request)
        except HTTPError as exc:
            if exc.code == 404:
                exc.close()
                return None
            raise
        validate_observation(response, request)
        return response

    def iter_passive(self, after: int = 0) -> Iterable[tuple[int, dict]]:
        if type(after) is not int or after < 0:
            raise ValueError("passive cursor must be a nonnegative integer")
        while True:
            page = self._call(f"/v1/passive?{urlencode({'after': after})}")
            if not isinstance(page, dict) or not isinstance(page.get("entries"), list):
                raise ValueError("invalid passive target response")
            if not page["entries"]:
                return
            for entry in page["entries"]:
                if (
                    not isinstance(entry, dict)
                    or type(entry.get("cursor")) is not int
                    or entry["cursor"] <= after
                ):
                    raise ValueError("passive target cursors must be strictly increasing")
                observation = entry.get("observation")
                validate_observation(observation)
                if observation["mode"] != "passive":
                    raise ValueError("active response appeared in passive observation stream")
                after = entry["cursor"]
                yield after, observation


def target_from_config(config: dict) -> Target:
    if not isinstance(config, dict):
        raise ValueError("target config must be a JSON object")
    kind = config.get("kind")
    if kind not in ("hf", "sandbox", "http"):
        raise ValueError("target kind must be hf, sandbox or http")
    target_class = {"hf": LocalHFTarget, "sandbox": SandboxAgentTarget, "http": HTTPTarget}[kind]
    return target_class(**{key: value for key, value in config.items() if key != "kind"})


def serve_target(target: Target, *, port: int = 0) -> ThreadingHTTPServer:
    """Bind a controlled target and its passive mirror to the local loopback."""

    class Handler(BaseHTTPRequestHandler):
        timeout = 60

        def log_message(self, *args):
            # The receipt store captures provenance, without printing private prompts.
            pass

        def _reply(self, status, value):
            payload = _json(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The committed receipt remains available if a client disconnected.

        def do_POST(self):
            if self.path not in ("/v1/requests", "/v1/interactions", "/v1/reconcile"):
                self._reply(404, {"error": "unknown target route"})
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= _MAX_BODY:
                    raise ValueError("missing or oversized request body")
                data = json.loads(self.rfile.read(size), object_pairs_hook=_unique_wire_keys)
                validate_request(data)
                if self.path == "/v1/reconcile":
                    response = target.reconcile(data)
                    if response is None:
                        self._reply(404, {"error": "no committed receipt"})
                        return
                else:
                    mode = "passive" if self.path == "/v1/interactions" else "active"
                    if data.get("mode", "active") != mode:
                        raise ValueError("request mode does not match target route")
                    response = target.submit(data)
                self._reply(200, response)
            except (TypeError, ValueError) as exc:
                self._reply(400, {"error": str(exc)})

        def do_GET(self):
            route = urlsplit(self.path)
            if route.path != "/v1/passive":
                self._reply(404, {"error": "unknown target route"})
                return
            try:
                after = int(parse_qs(route.query).get("after", ["0"])[0])
                entries = []
                bytes_used = 0
                for cursor, observation in target.iter_passive(after=after):
                    entry = {"cursor": cursor, "observation": observation}
                    entry_size = len(_json(entry).encode()) + 1
                    if entry_size > _MAX_BODY - 64:
                        raise ValueError("passive observation exceeds the HTTP response limit")
                    if bytes_used + entry_size > _MAX_BODY - 64 or len(entries) >= 100:
                        break
                    entries.append(entry)
                    bytes_used += entry_size
                self._reply(200, {"entries": entries})
            except (TypeError, ValueError) as exc:
                self._reply(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def _read_config(path):
    with Path(path).open("rb") as handle:
        return _read_wire(handle)


def _serve_cli(args):
    target = target_from_config(_read_config(args.config))
    if isinstance(target, HTTPTarget):
        raise ValueError("serve requires a local target")
    with serve_target(target, port=args.port) as server:
        print(_json({"url": f"http://127.0.0.1:{server.server_port}", "ready": True}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def _interact_cli(args):
    target = target_from_config(_read_config(args.config))
    request = _read_config(args.request)
    if not isinstance(request, dict):
        raise ValueError("passive request must be a JSON object")
    return target.submit(request | {"mode": "passive"})


def add_subcommands(commands):
    serve = commands.add_parser(
        "session-target-serve",
        help="serve a local target and passive observation mirror on loopback",
    )
    serve.add_argument("--config", required=True)
    serve.add_argument(
        "--port", type=int, default=0, help="default: choose an available local port"
    )
    serve.set_defaults(func=_serve_cli)

    interact = commands.add_parser(
        "session-target-interact", help="interact with the controlled target outside a session"
    )
    interact.add_argument("--config", required=True)
    interact.add_argument("--request", required=True)
    interact.set_defaults(func=_interact_cli)
