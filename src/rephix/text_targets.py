"""String-only oracle processes for controlled text-recovery experiments.

The server reads private target files and writes only sampled strings. It runs in
its own process so the recovery process never holds target parameters in memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .text_recovery import PROTOCOL, NgramFeatures, _softmax


class _NgramTarget:
    def __init__(self, config):
        self.features = NgramFeatures(
            config["vocabulary"], config["order"], config.get("separator", "")
        )
        with np.load(config["parameters"], allow_pickle=False) as data:
            self.theta = np.asarray(data["theta"], dtype=np.float64)
            # Optional nonlinear term places the target outside the log-linear class.
            self.mlp = (
                (np.asarray(data["w_in"]), np.asarray(data["w_out"]))
                if "w_in" in data.files
                else None
            )
        if self.theta.shape != (len(self.features.vocabulary), self.features.width):
            raise ValueError("ngram target theta must be (vocabulary, order*vocabulary+1)")
        trigger = config.get("trigger")
        self.trigger = None
        if trigger is not None:
            self.trigger = (
                trigger["suffix"],
                self.features.vocabulary.index(trigger["class"]),
                float(trigger["delta"]),
            )
        self.classes = self.features.vocabulary

    def logits(self, prompt):
        features = self.features.features([prompt])
        values = (features @ self.theta.T)[0]
        if self.mlp is not None:
            values = values + (np.tanh(features @ self.mlp[0]) @ self.mlp[1])[0]
        if self.trigger is not None and prompt.endswith(self.trigger[0]):
            values[self.trigger[1]] += self.trigger[2]
        return values


class _HfTarget:
    def __init__(self, config):
        from .hf import _deps, _device, _load_tokenizer

        torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer = _deps()
        self.torch = torch
        path = config["model"]
        model_config = AutoConfig.from_pretrained(path)
        self.tokenizer = _load_tokenizer(AutoTokenizer, path, model_config=model_config)
        self.device = _device(torch, config.get("device", "cpu"))
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                path, dtype=getattr(torch, config.get("dtype", "float32"))
            )
            .to(self.device)
            .eval()
        )
        self.head = None
        if config.get("head"):
            with np.load(config["head"], allow_pickle=False) as data:
                self.head = (np.asarray(data["weight"]), np.asarray(data["bias"]))
            self.classes = list(config["labels"])
        else:
            self.classes = None

    def logits(self, prompt):
        torch = self.torch
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        with torch.inference_mode():
            if self.head is not None:
                hidden = self.model.base_model(input_ids=ids).last_hidden_state[0, -1]
                hidden = hidden.double().cpu().numpy()
                return self.head[0] @ hidden + self.head[1]
            return self.model(input_ids=ids).logits[0, -1].double().cpu().numpy()

    def render(self, index):
        if self.classes is not None:
            return self.classes[index]
        return self.tokenizer.decode([int(index)])


def serve(config, stdin=None, stdout=None):
    if stdin is None:
        sys.stdin.reconfigure(encoding="utf-8")
        stdin = sys.stdin
    if stdout is None:
        sys.stdout.reconfigure(encoding="utf-8")
        stdout = sys.stdout
    kind = config.get("kind")
    target = _NgramTarget(config) if kind == "ngram" else _HfTarget(config)
    temperature = float(config.get("temperature", 1.0))
    greedy = bool(config.get("greedy", False))
    seed = int(config.get("seed", 0))
    requests = {}
    stdout.write(json.dumps({"protocol": PROTOCOL}) + "\n")
    stdout.flush()
    for line in stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        prompt, n = request["prompt"], int(request["n"])
        logits = target.logits(prompt)
        # A stream per (seed, prompt, repeat) keeps contexts independent and makes a
        # restarted server reproduce, not reuse, its random streams.
        import hashlib

        digest = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:8], "big")
        repeat = requests.get(prompt, 0)
        requests[prompt] = repeat + 1
        rng = np.random.default_rng([seed, digest, repeat])
        if greedy:
            choices = np.full(n, int(np.argmax(logits)))
        else:
            p = _softmax((logits / temperature)[None, :])[0]
            choices = rng.choice(len(p), size=n, p=p)
        if kind == "ngram":
            outputs = [target.classes[int(c)] for c in choices]
        else:
            outputs = [target.render(int(c)) for c in choices]
        stdout.write(json.dumps({"outputs": outputs}, ensure_ascii=False) + "\n")
        stdout.flush()
    return 0


def _serve_command(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    serve(config)
    return None


def add_subcommands(commands):
    command = commands.add_parser(
        "text-target-serve",
        help="serve a local controlled target as a string-only JSONL oracle on stdin/stdout",
    )
    command.add_argument("--config", required=True, help="private target configuration JSON")
    command.set_defaults(func=_serve_command)
