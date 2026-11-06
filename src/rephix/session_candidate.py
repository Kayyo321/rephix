"""Tokenizer-independent candidate training and standalone inference."""

from __future__ import annotations

import io
import math
from bisect import bisect_right
from collections.abc import Sequence
from itertools import accumulate, islice
from pathlib import Path

from .io import file_sha256, write_json
from .session_store import object_hash

PAD, BOS, EOS, VOCAB = 256, 257, 258, 259
DEFAULT_CANDIDATE = {
    "architecture": "gru",
    "width": 64,
    "context": 128,
    "batch": 64,
    "learning_rate": 0.003,
    "seed": 17,
    "device": "cpu",
}


def candidate_config(options: dict) -> dict:
    allowed = DEFAULT_CANDIDATE.keys() | {"objective"}
    if set(options) - allowed:
        raise ValueError(f"unknown candidate options: {sorted(set(options) - allowed)}")
    config = DEFAULT_CANDIDATE | options
    if config["architecture"] not in {"gru", "conv"}:
        raise ValueError("candidate architecture must be gru or conv")
    if config.get("objective", "byte_ce") not in {"byte_ce", "segment_balanced_ce"}:
        raise ValueError("candidate objective must be byte_ce or segment_balanced_ce")
    for name, minimum, maximum in (("width", 8, 512), ("context", 16, 1024), ("batch", 1, 4096)):
        if type(config[name]) is not int or not minimum <= config[name] <= maximum:
            raise ValueError(f"candidate {name} must be an integer in [{minimum}, {maximum}]")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**63:
        raise ValueError("candidate seed must be a nonnegative 63-bit integer")
    if (
        not isinstance(config["learning_rate"], (int, float))
        or not 0 < config["learning_rate"] <= 1
    ):
        raise ValueError("candidate learning_rate must be in (0, 1]")
    if config["device"] not in {"cpu", "cuda", "mps"}:
        raise ValueError("candidate device must be cpu, cuda or mps")
    return config


def _model(config):
    import torch
    from torch import nn

    class ByteModel(nn.Module):
        def __init__(self):
            super().__init__()
            width = config["width"]
            self.embedding = nn.Embedding(VOCAB, width, padding_idx=PAD)
            self.recurrent = config["architecture"] == "gru"
            if self.recurrent:
                self.layers = nn.GRU(width, width, batch_first=True)
            else:
                dilations = [2**power for power in range(math.ceil(math.log2(config["context"])))]
                self.layers = nn.ModuleList(
                    nn.Conv1d(width, width, 2, dilation=dilation) for dilation in dilations
                )
            self.readout = nn.Linear(width, VOCAB)

        def forward(self, tokens):
            encoded = self.embedding(tokens)
            if self.recurrent:
                encoded, _ = self.layers(encoded)
                return self.readout(encoded[:, -1])
            encoded = encoded.transpose(1, 2)
            for layer in self.layers:
                encoded = encoded + torch.nn.functional.gelu(
                    layer(torch.nn.functional.pad(encoded, (layer.dilation[0], 0)))
                )
            return self.readout(encoded[:, :, -1])

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        return ByteModel().to(config["device"])


def _context(tokens, size):
    if len(tokens) <= size:
        return [PAD] * (size - len(tokens)) + tokens
    anchor = size // 2
    return tokens[:anchor] + tokens[-(size - anchor) :]


def _prefix(text):
    return [BOS, *text.encode("utf-8")]


def _sequence(example, byte_cap):
    prefix = _prefix(example["input"])
    completion = list(example["output"].encode("utf-8"))
    # A capped continuation remains right-censored.
    targets = completion[:byte_cap]
    if len(completion) < byte_cap and not example.get("truncated", False):
        targets.append(EOS)
    return prefix + targets, len(prefix)


class ByteCandidate:
    """Fixed-supervision-budget autoregressive UTF-8 byte learner."""

    def __init__(self, config, state: bytes | None = None):
        import torch

        self.config = candidate_config(config)
        self.model = _model(self.config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        # The scheduler state is recorded even for the constant-rate policy.
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        self.rng = torch.Generator(device="cpu").manual_seed(self.config["seed"])
        self.step = 0
        if state is not None:
            saved = torch.load(io.BytesIO(state), map_location="cpu", weights_only=True)
            if saved["schema_version"] != 1 or saved["config"] != self.config:
                raise ValueError("incompatible candidate recovery checkpoint")
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.scheduler.load_state_dict(saved["scheduler"])
            self.rng.set_state(saved["sampler_rng"])
            self.step = saved["step"]

    def checkpoint(self) -> bytes:
        import torch

        output = io.BytesIO()
        torch.save(
            {
                "schema_version": 1,
                "config": self.config,
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "sampler_rng": self.rng.get_state(),
            },
            output,
        )
        return output.getvalue()

    def train(self, examples: Sequence[dict], steps: int, byte_cap: int) -> dict:
        import torch

        sequences = [_sequence(example, byte_cap) for example in examples]
        offsets = list(accumulate(len(sequence) - start for sequence, start in sequences))
        positions = offsets[-1] if offsets else 0
        if positions == 0:
            raise ValueError("dataset has no exposed model-generated bytes or end markers")
        batch_size = self.config["batch"]
        balanced = self.config.get("objective") == "segment_balanced_ce"
        nonempty = [index for index, (tokens, start) in enumerate(sequences) if len(tokens) > start]
        losses = []
        self.model.train()
        for _ in range(steps):
            if balanced:
                choices = torch.randint(len(nonempty), (batch_size,), generator=self.rng).tolist()
                draws = torch.rand(batch_size, generator=self.rng, dtype=torch.float64).tolist()
                samples = []
                for choice, draw in zip(choices, draws):
                    row = nonempty[choice]
                    first = offsets[row - 1] if row else 0
                    samples.append(first + int(draw * (offsets[row] - first)))
            else:
                samples = torch.randint(positions, (batch_size,), generator=self.rng).tolist()
            inputs, labels = [], []
            for choice in samples:
                row = bisect_right(offsets, choice)
                sequence, start = sequences[row]
                position = start + choice - (offsets[row - 1] if row else 0)
                inputs.append(_context(sequence[:position], self.config["context"]))
                labels.append(sequence[position])
            predictions = self.model(self._tensor(inputs))
            loss = torch.nn.functional.cross_entropy(predictions, self._tensor(labels))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            losses.append(float(loss.detach()))
        return {
            "updates": steps,
            "supervised_positions": steps * batch_size,
            "eligible_positions": positions,
            "training_loss": sum(losses) / len(losses) if losses else None,
        }

    def _tensor(self, value):
        import torch

        return torch.tensor(value, dtype=torch.long, device=self.config["device"])

    def teacher_forced(self, examples: Sequence[dict], byte_cap: int) -> dict:
        import torch

        self.model.eval()

        def positions():
            for example in examples:
                sequence, start = _sequence(example, byte_cap)
                for position in range(start, len(sequence)):
                    yield _context(sequence[:position], self.config["context"]), sequence[position]

        stream = positions()
        count, correct, loss = 0, 0, 0.0
        with torch.inference_mode():
            while rows := list(islice(stream, 256)):
                inputs, labels = zip(*rows)
                targets = self._tensor(labels)
                logits = self.model(self._tensor(inputs))
                loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum"))
                correct += int((logits.argmax(dim=-1) == targets).sum())
                count += len(rows)
        return {
            "positions": count,
            "matches": correct,
            "agreement": correct / count if count else None,
            "cross_entropy": loss / count if count else None,
        }

    def uncertainty(self, inputs: Sequence[str]) -> list[float]:
        import torch

        self.model.eval()
        with torch.inference_mode():
            tokens = [_context(_prefix(text), self.config["context"]) for text in inputs]
            log_probs = self.model(self._tensor(tokens)).log_softmax(dim=-1)
            return (-log_probs.exp() * log_probs).sum(dim=-1).tolist()

    def choose(self, context: str, outputs: Sequence[str]) -> int:
        losses = [
            self.teacher_forced([{"input": context, "output": output}], 4096)["cross_entropy"]
            for output in outputs
        ]
        return min(range(len(losses)), key=losses.__getitem__)

    def generate(self, inputs: Sequence[str], max_bytes: int) -> list[str]:
        if not inputs:
            raise ValueError("generation needs at least one input")
        return [
            text
            for start in range(0, len(inputs), 128)
            for text in _generate(self.model, self.config, inputs[start : start + 128], max_bytes)
        ]

    def export(self, destination: Path, *, provenance=None) -> dict:
        from safetensors.torch import save_file

        destination.mkdir(parents=True, exist_ok=False)
        weights = destination / "weights.safetensors"
        config = self.config | {"device": "cpu"}
        report = {
            "schema_version": 2,
            "codec": "utf8-bytes-v1",
            "candidate": config,
            "training_step": self.step,
            "initialization": "random",
            "runtime_dependencies": ["rephix", "torch", "safetensors"],
            "response_cache": False,
            "retrieval": False,
            "target_access": False,
            "provenance": provenance or {},
        }
        save_file(
            {key: value.cpu() for key, value in self.model.state_dict().items()},
            weights,
            metadata={"inference_manifest_sha256": object_hash(report)},
        )
        report["weights_sha256"] = file_sha256(weights)
        write_json(destination / "manifest.json", report)
        return report


def _generate(model, config, inputs: Sequence[str], max_bytes: int) -> list[str]:
    import torch

    if not inputs or not 1 <= max_bytes <= 4096:
        raise ValueError("generation needs inputs and a positive byte cap up to 4096")
    prefixes = [_prefix(value) for value in inputs]
    continuations = [bytearray() for _ in inputs]
    done = [False] * len(inputs)
    model.eval()
    with torch.inference_mode():
        for _ in range(max_bytes):
            rows = [_context(prefix, config["context"]) for prefix in prefixes]
            logits = model(torch.tensor(rows, dtype=torch.long, device=config["device"]))
            logits[:, PAD : BOS + 1] = -torch.inf
            for index, value in enumerate(logits.argmax(dim=-1).tolist()):
                if not done[index] and value != EOS:
                    prefixes[index].append(value)
                    continuations[index].append(value)
                else:
                    done[index] = True
            if all(done):
                break
    return [value.decode("utf-8", errors="replace") for value in continuations]


def load_export(directory: Path, device="cpu") -> ByteCandidate:
    import json

    from safetensors import safe_open
    from safetensors.torch import load_file

    with (directory / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    weights = directory / "weights.safetensors"
    if (
        manifest.get("schema_version") not in (1, 2)
        or manifest.get("codec") != "utf8-bytes-v1"
        or manifest.get("weights_sha256") != file_sha256(weights)
    ):
        raise ValueError("candidate export manifest or weights integrity mismatch")
    with safe_open(weights, framework="pt") as tensors:
        bound_manifest = (tensors.metadata() or {}).get("inference_manifest_sha256")
    if manifest["schema_version"] == 2 or bound_manifest is not None:
        expected = object_hash(
            {key: value for key, value in manifest.items() if key != "weights_sha256"}
        )
        if bound_manifest != expected:
            raise ValueError("candidate export inference manifest integrity mismatch")
    config = candidate_config(manifest["candidate"] | {"device": device})
    candidate = ByteCandidate(config)
    candidate.model.load_state_dict(load_file(weights))
    candidate.step = manifest["training_step"]
    return candidate


def run_export(directory: Path, inputs: Sequence[str], max_bytes: int, device="cpu") -> list[str]:
    return load_export(directory, device).generate(inputs, max_bytes)
