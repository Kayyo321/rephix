"""Pre-registered text-only recovery experiment with an isolated evaluator.

Roles are separated by process, directory and file permissions:

* builder/evaluator (this process) creates private targets from a per-run secret and,
  after recovery has exited, reads private ground truth to score the export;
* oracle (started here) loads the private target into memory and then answers only
  with sampled strings. Recovery reaches it through a line relay on a loopback socket;
* recovery (separate interpreter) runs ``rephix text-recover`` while the private
  directory has mode 000, so any read of it fails at the operating-system level,
  including from native code. A Python audit hook additionally logs every file the
  recovery process opens, every process it spawns and every chmod it attempts.

Corpus text and public prompt pools are read from docs/ at the protocol's pinned
``corpus_commit`` (not the working tree), so results are reproducible from git.

Usage (POSIX):
  python examples/text_recovery_experiment.py --protocol examples/text_recovery_protocol.json \
      --split final --work /abs/work --output /abs/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

SANDBOX = r"""
import json, os, sys
forbidden = [os.path.realpath(p) for p in json.loads(os.environ['REPHIX_FORBID'])]
log_path = os.environ['REPHIX_AUDIT_LOG']
opened, denied, spawned, chmods = set(), [], [], []
def hook(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            path = os.path.realpath(os.fsdecode(args[0]))
        except (TypeError, ValueError):
            return
        if any(path == f or path.startswith(f + os.sep) for f in forbidden):
            denied.append(path)
            raise PermissionError('text-only boundary: ' + path)
        opened.add(path)
    elif event == 'subprocess.Popen':
        spawned.append([os.fsdecode(a) for a in args[1]] if args[1] else [])
    elif event in ('os.chmod', 'os.chown'):
        chmods.append(repr(args[0]))
sys.addaudithook(hook)
from rephix.cli import main
try:
    code = main(sys.argv[1:])
finally:
    prefixes = (os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix))
    with open(log_path, 'w', encoding='utf-8') as handle:
        json.dump({'opened_outside_python': sorted(p for p in opened if not p.startswith(prefixes)),
                   'denied': denied, 'spawned': spawned, 'chmod': chmods}, handle, indent=1)
sys.exit(code)
"""

CORPUS_ALPHABET = list("abcdefghijklmnopqrstuvwxyz ")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class Docs:
    """docs/*.md as stored at a pinned commit."""

    def __init__(self, commit):
        self.commit = commit
        listing = subprocess.run(
            ["git", "ls-tree", "--name-only", commit, "docs/"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        self.names = sorted(n for n in listing if n.endswith(".md"))
        self.texts = {
            n: subprocess.run(
                ["git", "show", f"{commit}:{n}"],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout
            for n in self.names
        }

    def lines(self):
        result = []
        for name in self.names:
            for line in self.texts[name].splitlines():
                words = "".join(c if c.isalnum() else " " for c in line).split()
                if len(words) >= 4:
                    result.append(words)
        return result


def _corpus(seed, docs):
    rng = np.random.default_rng(seed)
    chosen = [
        docs.names[i] for i in rng.choice(len(docs.names), size=len(docs.names) // 2, replace=False)
    ]
    text = " ".join(docs.texts[n].lower() for n in chosen)
    cleaned = "".join(ch if ch in CORPUS_ALPHABET else " " for ch in text)
    return " ".join(cleaned.split()), chosen


def _fit_corpus_target(seed, order, docs):
    from rephix.text_recovery import NgramFeatures, fit_multinomial, span_basis

    text, names = _corpus(seed, docs)
    features = NgramFeatures(CORPUS_ALPHABET, order)
    counts = {}
    for position in range(order, len(text)):
        context = text[position - order : position]
        counts.setdefault(context, np.zeros(len(CORPUS_ALPHABET)))[
            features.index[text[position]]
        ] += 1
    contexts = sorted(counts)
    design = features.features(contexts)
    basis, _ = span_basis(design, features.span_dimension())
    gamma, _ = fit_multinomial(design @ basis, np.stack([counts[c] for c in contexts]), ridge=1.0)
    return gamma @ basis.T, {"corpus_files": names, "corpus_characters": len(text)}


def _tiny_backbone(path, seed, architecture, docs):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast, Qwen2Config

    words = sorted({w for line in docs.lines() for w in line})
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    torch.manual_seed(seed)
    if architecture == "tiny-gpt2":  # LayerNorm with bias: features lie on a hyperplane
        config = GPT2Config(vocab_size=len(vocab), n_embd=48, n_layer=2, n_head=4, n_positions=128)
    else:
        config = Qwen2Config(
            vocab_size=len(vocab),
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
    AutoModelForCausalLM.from_config(config).save_pretrained(path)
    tokenizer.save_pretrained(path)


def _prompt_pool(seed, count, docs):
    """Distinct public prompts: windows of documentation text at the pinned commit."""
    rng = np.random.default_rng(seed)
    lines = docs.lines()
    pool, seen = [], set()
    while len(pool) < count:
        words = lines[rng.integers(len(lines))]
        start = rng.integers(0, max(1, len(words) - 3))
        length = rng.integers(3, 12)
        prompt = " ".join(words[start : start + length])
        if prompt not in seen:
            seen.add(prompt)
            pool.append(prompt)
    return pool


def _hf_features(model_path, prompts):
    from rephix.text_recovery import BackboneFeatures

    return BackboneFeatures(model_path, ["a", "b"]).features(prompts)


def build_case(case, seed, root, docs):
    private, public, recovery = root / "private", root / "public", root / "recovery"
    for path in (private, public, recovery):
        path.mkdir(parents=True)
    # Private randomness comes from a per-run secret, never from the public seeds.
    secret = int.from_bytes(os.urandom(8), "big")
    rng = np.random.default_rng([secret, seed])
    family = case["family"]
    truth = {"secret": secret}
    if family == "ngram":
        vocab = (
            CORPUS_ALPHABET
            if case.get("corpus")
            else [chr(ord("a") + i) for i in range(case["vocab_size"])]
        )
        order = case["order"]
        target_order = case.get("target_order", order)
        prompt_length = case.get("prompt_length", max(order, target_order))
        if case.get("corpus"):
            theta, truth["corpus"] = _fit_corpus_target(seed, target_order, docs)
        else:
            theta = case.get("scale", 1.0) * rng.normal(
                size=(len(vocab), target_order * len(vocab) + 1)
            )
        arrays = {"theta": theta}
        if case.get("mlp_hidden"):
            arrays["w_in"] = rng.normal(size=(theta.shape[1], case["mlp_hidden"]))
            arrays["w_out"] = case["mlp_scale"] * rng.normal(size=(case["mlp_hidden"], len(vocab)))
        np.savez(private / "target.npz", **arrays)
        server = {
            "kind": "ngram",
            "vocabulary": vocab,
            "order": target_order,
            "parameters": str(private / "target.npz"),
            "seed": int(rng.integers(2**62)),
            "greedy": case.get("greedy", False),
        }
        if case.get("trigger"):
            server["trigger"] = case["trigger"]
        features = {
            "kind": "ngram",
            "vocabulary": vocab,
            "order": order,
            "prompt_length": prompt_length,
        }
        truth.update(
            {
                "theta": theta,
                "vocabulary": vocab,
                "order": target_order,
                "prompt_length": prompt_length,
                "arrays": arrays,
                "trigger": case.get("trigger"),
            }
        )
    else:
        backbone = public / "backbone"
        if case.get("backbone") in ("tiny-qwen2", "tiny-gpt2"):
            _tiny_backbone(backbone, seed, case["backbone"], docs)
        else:
            shutil.copytree(Path(os.path.expandvars(case["backbone_path"])).expanduser(), backbone)
        labels = case["labels"]
        target_backbone = backbone
        if case.get("perturb_backbone"):
            import torch
            from transformers import AutoModelForCausalLM

            target_backbone = private / "backbone"
            model = AutoModelForCausalLM.from_pretrained(backbone, dtype=torch.float32)
            generator = torch.Generator().manual_seed(int(rng.integers(2**62)))
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if ".mlp." in name and parameter.ndim == 2:
                        noise = torch.randn(parameter.shape, generator=generator)
                        parameter.add_(case["perturb_backbone"] * parameter.std() * noise)
            model.save_pretrained(target_backbone)
            for name in os.listdir(backbone):
                if name.startswith("tokenizer") or name in (
                    "vocab.json",
                    "merges.txt",
                    "special_tokens_map.json",
                ):
                    shutil.copy(backbone / name, target_backbone / name)
        pool = _prompt_pool(
            seed, case["contexts"] + case["validation_contexts"] + case["test_contexts"], docs
        )
        train_pool = pool[: case["contexts"] + case["validation_contexts"]]
        test_pool = pool[len(train_pool) :]
        (public / "prompts.txt").write_text("\n".join(train_pool) + "\n", encoding="utf-8")
        hidden = _hf_features(target_backbone, train_pool[:64])[:, :-1]
        direction = rng.normal(size=(len(labels), hidden.shape[1]))
        weight = case.get("logit_scale", 1.5) * direction / (hidden @ direction.T).std()
        bias = rng.normal(size=len(labels)) * 0.5
        bias -= (hidden @ weight.T + bias).mean(0)
        np.savez(private / "head.npz", weight=weight, bias=bias)
        server = {
            "kind": "hf",
            "model": str(target_backbone),
            "head": str(private / "head.npz"),
            "labels": labels,
            "seed": int(rng.integers(2**62)),
            "greedy": case.get("greedy", False),
        }
        features = {"kind": "hf-backbone", "model": str(backbone), "labels": labels}
        truth.update(
            {
                "weight": weight,
                "bias": bias,
                "test_prompts": test_pool,
                "target_backbone": str(target_backbone),
                "public_backbone": str(backbone),
            }
        )
    (private / "server.json").write_text(json.dumps(server), encoding="utf-8")
    (recovery / "features.json").write_text(json.dumps(features), encoding="utf-8")
    return truth


class Oracle:
    """Hosts the target process and relays string-only lines over loopback TCP."""

    def __init__(self, root, python, relabel=False):
        self.process = subprocess.Popen(
            [
                python,
                "-m",
                "rephix",
                "text-target-serve",
                "--config",
                str(root / "private" / "server.json"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.hello = self.process.stdout.readline()  # target fully loaded
        if not self.hello:
            raise RuntimeError("oracle failed to start")
        self.relabel = relabel
        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        self.lock = threading.Lock()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._bridge, args=(connection,), daemon=True).start()

    def _bridge(self, connection):
        with connection, connection.makefile("rw", encoding="utf-8", newline="\n") as stream:
            stream.write(self.hello)
            stream.flush()
            for line in stream:
                with self.lock:
                    self.process.stdin.write(line)
                    self.process.stdin.flush()
                    reply = self.process.stdout.readline()
                if self.relabel:  # boundary case: a string outside the declared classes
                    data = json.loads(reply)
                    data["outputs"] = [o + "!" if o == "a" else o for o in data["outputs"]]
                    reply = json.dumps(data) + "\n"
                stream.write(reply)
                stream.flush()

    def close(self):
        self.server.close()
        self.process.stdin.close()
        self.process.wait(timeout=60)


def relay(port):
    """Target command given to recovery: copy lines between stdio and the oracle."""
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    with (
        socket.create_connection(("127.0.0.1", int(port))) as connection,
        connection.makefile("rw", encoding="utf-8", newline="\n") as stream,
    ):
        sys.stdout.write(stream.readline())
        sys.stdout.flush()
        for line in sys.stdin:
            stream.write(line)
            stream.flush()
            sys.stdout.write(stream.readline())
            sys.stdout.flush()


def relabel_proxy(argv):
    """Protocol v1 boundary case (kept so v1 runs remain reproducible)."""
    child = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1
    )
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(child.stdout.readline())
    sys.stdout.flush()
    for line in sys.stdin:
        child.stdin.write(line)
        child.stdin.flush()
        reply = json.loads(child.stdout.readline())
        reply["outputs"] = [o + "!" if o == "a" else o for o in reply["outputs"]]
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


def run_recovery(case, root, python, oracle):
    recovery = root / "recovery"
    target_argv = [python, "-I", str(Path(__file__).resolve()), "--relay", str(oracle.port)]
    (recovery / "target.json").write_text(json.dumps({"argv": target_argv}), encoding="utf-8")
    argv = [
        "text-recover",
        "--features",
        str(recovery / "features.json"),
        "--target",
        str(recovery / "target.json"),
        "--output",
        str(recovery / "export"),
        "--samples-per-context",
        str(int(case["samples_per_context"])),
        "--validation-contexts",
        str(case["validation_contexts"]),
        "--bootstrap",
        str(case["bootstrap"]),
        "--seed",
        str(case["query_seed"]),
        "--contexts",
        str(case["contexts"]),
    ]
    if case["family"] != "ngram":
        argv += ["--contexts-file", str(root / "public" / "prompts.txt")]
    env = {
        **os.environ,
        "REPHIX_FORBID": json.dumps([str(root / "private")]),
        "REPHIX_AUDIT_LOG": str(recovery / "audit.json"),
        "PYTHONIOENCODING": "utf-8",
    }
    os.chmod(root / "private", 0)  # OS-level boundary for the whole recovery run
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [python, "-c", SANDBOX, *argv],
            cwd=recovery,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        os.chmod(root / "private", 0o700)
    seconds = time.perf_counter() - started
    audit = json.loads((recovery / "audit.json").read_text(encoding="utf-8"))
    return result, seconds, audit, target_argv


def evaluate(case, root, truth, python):
    from rephix.text_recovery import NgramFeatures

    export = root / "recovery" / "export"
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    with np.load(export / "parameters.npz") as data:
        gamma, basis = data["gamma"], data["basis"]
        se = data["gamma_standard_error"] if "gamma_standard_error" in data.files else None
        theta_hat = data["canonical_theta"]
    with np.load(export / "baseline.npz") as data:
        baseline = data["canonical_theta"]
    classes = manifest["classes"]
    centre = np.eye(len(classes)) - 1.0 / len(classes)
    metrics = {"status": manifest["status"], "reasons": manifest["reasons"]}
    queried = set()
    with (export / "transcript.jsonl").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            queried.add(json.loads(line)["prompt"])
    trigger_prompts = []
    if case["family"] == "ngram":
        vocab, order = truth["vocabulary"], truth["order"]
        rng = np.random.default_rng(case["query_seed"] + 99991)
        target_features = NgramFeatures(vocab, order, prompt_length=truth["prompt_length"])
        test = []
        while len(test) < case["test_contexts"]:
            (prompt,) = target_features.design_contexts(1, rng)
            if prompt not in queried and prompt not in test:
                test.append(prompt)
        if truth.get("trigger"):
            # Prompts ending in the trigger suffix after random prefixes of 1-4 tokens.
            suffix = truth["trigger"]["suffix"]
            while len(trigger_prompts) < 20:
                (prefix,) = target_features.design_contexts(1, rng)
                prompt = prefix[: int(rng.integers(1, len(prefix) + 1))] + suffix
                if prompt not in queried and prompt not in trigger_prompts:
                    trigger_prompts.append(prompt)

        def true_law(prompts):
            tf = target_features.features(prompts)
            logits = tf @ truth["theta"].T
            if "w_in" in truth["arrays"]:
                logits = logits + np.tanh(tf @ truth["arrays"]["w_in"]) @ truth["arrays"]["w_out"]
            if truth.get("trigger"):
                trig = truth["trigger"]
                for row, prompt in enumerate(prompts):
                    if prompt.endswith(trig["suffix"]):
                        logits[row, vocab.index(trig["class"])] += trig["delta"]
            return _softmax(logits)

        recovered = NgramFeatures(
            manifest["prior_knowledge"]["vocabulary"],
            manifest["prior_knowledge"]["order"],
            prompt_length=truth["prompt_length"],
        )

        def recovered_law(prompts, parameters=theta_hat):
            return _softmax(recovered.features(prompts) @ parameters.T)

        true_theta = truth["theta"] if case["canonical_truth"] else None
    else:
        test = truth["test_prompts"]

        def true_law(prompts):
            hidden = _hf_features(truth["target_backbone"], prompts)
            return _softmax(hidden[:, :-1] @ truth["weight"].T + truth["bias"])

        public = {}

        def recovered_law(prompts, parameters=theta_hat):
            key = tuple(prompts)
            if key not in public:
                public[key] = _hf_features(truth["public_backbone"], prompts)
            return _softmax(public[key] @ parameters.T)

        true_theta = (
            np.concatenate([truth["weight"], truth["bias"][:, None]], 1)
            if case["canonical_truth"]
            else None
        )
    reference = true_law(test)
    metrics["test_contexts"] = len(test)
    metrics["test_contexts_disjoint_from_queries"] = not bool(set(test) & queried)
    metrics["test_mean_total_variation"] = float(
        0.5 * np.abs(recovered_law(test) - reference).sum(1).mean()
    )
    metrics["baseline_test_mean_total_variation"] = float(
        0.5 * np.abs(recovered_law(test, baseline) - reference).sum(1).mean()
    )
    metrics["uniform_test_mean_total_variation"] = float(
        0.5 * np.abs(1.0 / len(classes) - reference).sum(1).mean()
    )
    if trigger_prompts:
        metrics["trigger_mean_total_variation"] = float(
            0.5 * np.abs(recovered_law(trigger_prompts) - true_law(trigger_prompts)).sum(1).mean()
        )
    if true_theta is not None:
        truth_canonical = centre @ true_theta @ basis
        error = gamma - truth_canonical
        norm = np.linalg.norm(truth_canonical)
        metrics["canonical_relative_l2_error"] = float(np.linalg.norm(error) / norm)
        metrics["canonical_max_abs_error"] = float(np.abs(error).max())
        metrics["baseline_canonical_relative_l2_error"] = float(
            np.linalg.norm(centre @ baseline @ basis - truth_canonical) / norm
        )
        metrics["reported_relative_standard_error"] = manifest["estimated_relative_standard_error"]
        metrics["truth_outside_identified_span"] = float(
            np.linalg.norm(centre @ true_theta - centre @ true_theta @ basis @ basis.T)
            / np.linalg.norm(centre @ true_theta)
        )
        if se is not None:
            z = (error / np.where(se > 0, se, np.inf))[se > 0]
            metrics["ci95_coverage"] = float((np.abs(z) < 1.959964).mean())
            metrics["ci95_coordinates"] = int(z.size)
            metrics["z_rms"] = float(np.sqrt((z**2).mean()))
    # Independent reload in a fresh sandboxed interpreter, private directory mode 000.
    probes = test[: min(5, len(test))]
    reload, audits = [], []
    os.chmod(root / "private", 0)
    try:
        for index, prompt in enumerate(probes):
            log = root / "recovery" / f"reload-audit-{index}.json"
            env = {
                **os.environ,
                "REPHIX_FORBID": json.dumps([str(root / "private")]),
                "REPHIX_AUDIT_LOG": str(log),
                "PYTHONIOENCODING": "utf-8",
            }
            out = subprocess.run(
                [
                    python,
                    "-c",
                    SANDBOX,
                    "text-model",
                    "logprobs",
                    "--model",
                    str(export),
                    "--prompt",
                    prompt,
                ],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            values = json.loads(out.stdout)["log_probabilities"]
            reload.append([values[c] for c in classes])
            audits.append(json.loads(log.read_text(encoding="utf-8")))
    finally:
        os.chmod(root / "private", 0o700)
    expected = np.log(recovered_law(probes))
    metrics["reload_max_abs_logprob_difference"] = float(np.abs(np.array(reload) - expected).max())
    metrics["reload_denied_opens"] = sum(len(a["denied"]) for a in audits)
    metrics["reload_spawned_processes"] = sum(len(a["spawned"]) for a in audits)
    metrics["reload_probes_audited"] = len(audits)
    metrics["budget"] = manifest["budget"]
    metrics["design"] = manifest["design"]
    metrics["misspecification_test"] = manifest["misspecification_test"]
    metrics["transcript_sha256"] = manifest["transcript"]["sha256"]
    metrics["parameters_sha256"] = manifest["parameters_sha256"]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    parser.add_argument("--split", choices=("dev", "final"))
    parser.add_argument("--work")
    parser.add_argument("--output")
    parser.add_argument("--cases", help="comma-separated case names")
    parser.add_argument("--keep", action="store_true", help="keep work directories")
    parser.add_argument("--relay")
    parser.add_argument("--relabel-proxy")
    args = parser.parse_args()
    if args.relay:
        relay(args.relay)
        return 0
    if args.relabel_proxy:
        relabel_proxy(json.loads(args.relabel_proxy))
        return 0
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    docs = Docs(protocol["corpus_commit"])
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    wanted = set(args.cases.split(",")) if args.cases else None
    results = {
        "protocol": (
            str(protocol_path.relative_to(REPO))
            if protocol_path.is_relative_to(REPO)
            else str(protocol_path)
        ),
        "protocol_sha256": _sha(protocol_path),
        "protocol_version": protocol.get("version"),
        "split": args.split,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip(),
        "git_worktree_clean_for_src": not subprocess.run(
            ["git", "status", "--porcelain", "src", "examples"],
            cwd=REPO,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "corpus_commit": docs.commit,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "runs": [],
    }
    try:
        import torch
        import transformers

        results["torch"], results["transformers"] = torch.__version__, transformers.__version__
    except ImportError:
        pass
    for case in protocol["cases"]:
        if wanted and case["name"] not in wanted:
            continue
        for seed in case["seeds"][args.split]:
            root = work / f"{case['name']}-{seed}"
            if root.exists():
                shutil.rmtree(root)
            run_case = {**case, "query_seed": seed + 1000}
            entry = {
                "case": case["name"],
                "seed": seed,
                "expected": case["allowed_statuses"],
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            truth = build_case(run_case, seed, root, docs)
            oracle = Oracle(root, python, relabel=case.get("emit_unknown", False))
            try:
                completed, seconds, audit, target_argv = run_recovery(
                    run_case, root, python, oracle
                )
            finally:
                oracle.close()
            entry["recovery_seconds"] = seconds
            entry["recovery_exit"] = completed.returncode
            entry["audit"] = {
                "denied_private_opens": len(audit["denied"]),
                "opened_files_outside_python": len(audit["opened_outside_python"]),
                "opened_under_private": [
                    p for p in audit["opened_outside_python"] if p.startswith(str(root / "private"))
                ],
                "spawned": audit["spawned"],
                "spawned_only_the_target_relay": all(s == target_argv for s in audit["spawned"]),
                "chmod_attempts": audit["chmod"],
                "private_directory_mode_during_recovery": "000",
            }
            if completed.returncode:
                entry["status"] = "error"
                entry["error"] = completed.stderr.strip()[-600:]
            else:
                entry.update(evaluate(run_case, root, truth, python))
            if truth.get("corpus"):
                entry["corpus"] = truth["corpus"]
            entry["target_secret"] = truth["secret"]  # published only after recovery
            entry["status_allowed"] = entry["status"] in case["allowed_statuses"]
            entry["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            results["runs"].append(entry)
            print(
                json.dumps(
                    {
                        k: entry.get(k)
                        for k in (
                            "case",
                            "seed",
                            "status",
                            "canonical_relative_l2_error",
                            "reported_relative_standard_error",
                            "ci95_coverage",
                            "test_mean_total_variation",
                            "trigger_mean_total_variation",
                        )
                    }
                ),
                flush=True,
            )
            if not args.keep:
                shutil.rmtree(root)
            Path(args.output).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(args.output).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
