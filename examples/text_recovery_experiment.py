"""Pre-registered text-only recovery experiment with an isolated evaluator.

Roles are separated by process and directory:

* builder/evaluator (this process) creates private targets and, after recovery has
  finished, reads private ground truth to score the export;
* oracle (child of recovery) reads private files and writes sampled strings only;
* recovery (separate interpreter) runs ``rephix text-recover`` under a Python audit
  hook that rejects every open of the private directory and logs all file access.

Usage:
  python examples/text_recovery_experiment.py --protocol examples/text_recovery_protocol.json \
      --split dev --work /abs/work --output /abs/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SANDBOX = r"""
import json, os, sys
forbidden = [os.path.realpath(p) for p in json.loads(os.environ['REPHIX_FORBID'])]
log_path = os.environ['REPHIX_AUDIT_LOG']
opened, denied, spawned = set(), [], []
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
sys.addaudithook(hook)
from rephix.cli import main
try:
    code = main(sys.argv[1:])
finally:
    prefixes = (os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix))
    with open(log_path, 'w', encoding='utf-8') as handle:
        json.dump({'opened_outside_python': sorted(p for p in opened if not p.startswith(prefixes)),
                   'denied': denied, 'spawned': spawned}, handle, indent=1)
sys.exit(code)
"""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


CORPUS_ALPHABET = list("abcdefghijklmnopqrstuvwxyz ")


def _corpus(seed, repo):
    rng = np.random.default_rng(seed)
    files = sorted((repo / "docs").glob("*.md"))
    chosen = [files[i] for i in rng.choice(len(files), size=len(files) // 2, replace=False)]
    text = " ".join(path.read_text(encoding="utf-8").lower() for path in chosen)
    cleaned = "".join(ch if ch in CORPUS_ALPHABET else " " for ch in text)
    return " ".join(cleaned.split()), [path.name for path in chosen]


def _fit_corpus_target(seed, order, repo):
    from rephix.text_recovery import NgramFeatures, fit_multinomial, span_basis

    text, names = _corpus(seed, repo)
    features = NgramFeatures(CORPUS_ALPHABET, order)
    size = len(CORPUS_ALPHABET)
    counts = {}
    for position in range(order, len(text)):
        context = text[position - order : position]
        counts.setdefault(context, np.zeros(size))[features.index[text[position]]] += 1
    contexts = sorted(counts)
    design = features.features(contexts)
    basis, _ = span_basis(design, features.span_dimension())
    gamma, _ = fit_multinomial(design @ basis, np.stack([counts[c] for c in contexts]), ridge=1.0)
    return gamma @ basis.T, {"corpus_files": names, "corpus_characters": len(text)}


def _tiny_backbone(path, seed):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config

    words = sorted({w for line in _prompt_pool_words() for w in line})
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    torch.manual_seed(seed)
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


def _prompt_pool_words():
    lines = []
    for path in sorted((Path(__file__).resolve().parents[1] / "docs").glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            words = [w for w in "".join(c if c.isalnum() else " " for c in line).split()]
            if len(words) >= 4:
                lines.append(words)
    return lines


def _prompt_pool(seed, count):
    """Distinct public prompts: windows of documentation text."""
    rng = np.random.default_rng(seed)
    lines = _prompt_pool_words()
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


def build_case(case, seed, root, repo, python):
    private, public, recovery = root / "private", root / "public", root / "recovery"
    for path in (private, public, recovery):
        path.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    family = case["family"]
    truth = {}
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
            vocab = CORPUS_ALPHABET
            theta, truth["corpus"] = _fit_corpus_target(seed, target_order, repo)
        else:
            width = target_order * len(vocab) + 1
            theta = case.get("scale", 1.0) * rng.normal(size=(len(vocab), width))
        arrays = {"theta": theta}
        if case.get("mlp_hidden"):
            width = theta.shape[1]
            arrays["w_in"] = rng.normal(size=(width, case["mlp_hidden"]))
            arrays["w_out"] = case["mlp_scale"] * rng.normal(size=(case["mlp_hidden"], len(vocab)))
        np.savez(private / "target.npz", **arrays)
        server = {
            "kind": "ngram",
            "vocabulary": vocab,
            "order": target_order,
            "parameters": str(private / "target.npz"),
            "seed": seed + 7,
            "greedy": case.get("greedy", False),
        }
        # The undeclared-output boundary case relabels outputs in a proxy (see below).
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
            }
        )
    else:
        if case.get("backbone") == "tiny-qwen2":
            backbone = public / "backbone"
            _tiny_backbone(backbone, seed)
        else:
            source = Path(os.path.expandvars(case["backbone_path"])).expanduser()
            backbone = public / "backbone"
            shutil.copytree(source, backbone)
        labels = case["labels"]
        target_backbone = backbone
        if case.get("perturb_backbone"):
            import torch
            from transformers import AutoModelForCausalLM

            target_backbone = private / "backbone"
            model = AutoModelForCausalLM.from_pretrained(backbone, dtype=torch.float32)
            generator = torch.Generator().manual_seed(seed)
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
            seed, case["contexts"] + case["validation_contexts"] + case["test_contexts"]
        )
        train_pool = pool[: case["contexts"] + case["validation_contexts"]]
        test_pool = pool[len(train_pool) :]
        (public / "prompts.txt").write_text("\n".join(train_pool) + "\n", encoding="utf-8")
        # Calibrate the private head so logits on public prompts have the requested spread.
        hidden = _hf_features(target_backbone, train_pool[:64])[:, :-1]
        dim = hidden.shape[1]
        direction = rng.normal(size=(len(labels), dim))
        spread = (hidden @ direction.T).std()
        weight = case.get("logit_scale", 1.5) * direction / spread
        bias = rng.normal(size=len(labels)) * 0.5
        bias -= (hidden @ weight.T + bias).mean(0)
        np.savez(private / "head.npz", weight=weight, bias=bias)
        server = {
            "kind": "hf",
            "model": str(target_backbone),
            "head": str(private / "head.npz"),
            "labels": labels,
            "seed": seed + 7,
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
    argv = [python, "-m", "rephix", "text-target-serve", "--config", str(private / "server.json")]
    if case.get("emit_unknown"):
        argv = [python, str(Path(__file__).resolve()), "--relabel-proxy", json.dumps(argv)]
    (recovery / "target.json").write_text(json.dumps({"argv": argv}), encoding="utf-8")
    return truth


def run_recovery(case, root, python, budget_scale=1.0):
    recovery = root / "recovery"
    argv = [
        "text-recover",
        "--features",
        str(recovery / "features.json"),
        "--target",
        str(recovery / "target.json"),
        "--output",
        str(recovery / "export"),
        "--samples-per-context",
        str(int(case["samples_per_context"] * budget_scale)),
        "--validation-contexts",
        str(case["validation_contexts"]),
        "--bootstrap",
        str(case.get("bootstrap", 99)),
        "--seed",
        str(case["query_seed"]),
    ]
    if case["family"] == "ngram":
        argv += ["--contexts", str(case["contexts"])]
    else:
        argv += [
            "--contexts-file",
            str(root / "public" / "prompts.txt"),
            "--contexts",
            str(case["contexts"]),
        ]
    env = {
        **os.environ,
        "REPHIX_FORBID": json.dumps([str(root / "private")]),
        "REPHIX_AUDIT_LOG": str(recovery / "audit.json"),
        "PYTHONIOENCODING": "utf-8",
    }
    started = time.perf_counter()
    result = subprocess.run(
        [python, "-c", SANDBOX, *argv],
        cwd=recovery,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    seconds = time.perf_counter() - started
    audit = json.loads((recovery / "audit.json").read_text(encoding="utf-8"))
    return result, seconds, audit


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
    transcript = root / "recovery" / "export" / "transcript.jsonl"
    queried = set()
    with transcript.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            queried.add(json.loads(line)["prompt"])
    if case["family"] == "ngram":
        vocab, order = truth["vocabulary"], truth["order"]
        rng = np.random.default_rng(case["query_seed"] + 99991)
        target_features = NgramFeatures(vocab, order, prompt_length=truth["prompt_length"])
        test = []
        while len(test) < case["test_contexts"]:
            (prompt,) = target_features.design_contexts(1, rng)
            if prompt not in queried and prompt not in test:
                test.append(prompt)
        tf = target_features.features(test)
        logits = tf @ truth["theta"].T
        if "w_in" in truth["arrays"]:
            logits = logits + np.tanh(tf @ truth["arrays"]["w_in"]) @ truth["arrays"]["w_out"]
        true_law = _softmax(logits)
        recovered_features = NgramFeatures(
            manifest["prior_knowledge"]["vocabulary"],
            manifest["prior_knowledge"]["order"],
            prompt_length=truth["prompt_length"],
        ).features(test)
        in_class = case["in_class"]
        true_theta = truth["theta"] if in_class else None
    else:
        test = truth["test_prompts"]
        assert not set(test) & queried
        hidden = _hf_features(truth["target_backbone"], test)
        true_law = _softmax(hidden[:, :-1] @ truth["weight"].T + truth["bias"])
        recovered_features = _hf_features(truth["public_backbone"], test)
        in_class = case["in_class"]
        true_theta = (
            np.concatenate([truth["weight"], truth["bias"][:, None]], 1) if in_class else None
        )
    law = _softmax(recovered_features @ theta_hat.T)
    base_law = _softmax(recovered_features @ baseline.T)
    marginal = np.full(len(classes), 1.0 / len(classes))
    metrics["test_contexts"] = len(test)
    metrics["test_contexts_disjoint_from_queries"] = not bool(set(test) & queried)
    metrics["test_mean_total_variation"] = float(0.5 * np.abs(law - true_law).sum(1).mean())
    metrics["baseline_test_mean_total_variation"] = float(
        0.5 * np.abs(base_law - true_law).sum(1).mean()
    )
    metrics["uniform_test_mean_total_variation"] = float(
        0.5 * np.abs(marginal - true_law).sum(1).mean()
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
        # Truth outside the identified span would make the canonical comparison vacuous.
        metrics["truth_outside_identified_span"] = (
            float(
                np.linalg.norm(centre @ true_theta - centre @ true_theta @ basis @ basis.T)
                / np.linalg.norm(centre @ true_theta)
            )
            if case["family"] == "hf-backbone"
            else 0.0
        )
        if se is not None:
            z = (error / np.where(se > 0, se, np.inf))[se > 0]
            metrics["ci95_coverage"] = float((np.abs(z) < 1.959964).mean())
            metrics["z_rms"] = float(np.sqrt((z**2).mean()))
    # Independent reload in a fresh sandboxed interpreter with the private directory denied.
    probe = test[: min(5, len(test))]
    reload = []
    env = {
        **os.environ,
        "REPHIX_FORBID": json.dumps([str(root / "private")]),
        "REPHIX_AUDIT_LOG": str(root / "recovery" / "reload-audit.json"),
        "PYTHONIOENCODING": "utf-8",
    }
    for prompt in probe:
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
    expected = np.log(law[: len(probe)])
    metrics["reload_max_abs_logprob_difference"] = float(np.abs(np.array(reload) - expected).max())
    reload_audit = json.loads((root / "recovery" / "reload-audit.json").read_text(encoding="utf-8"))
    metrics["reload_denied_opens"] = len(reload_audit["denied"])
    metrics["reload_spawned_processes"] = len(reload_audit["spawned"])
    metrics["budget"] = manifest["budget"]
    metrics["design"] = manifest["design"]
    metrics["misspecification_test"] = manifest["misspecification_test"]
    metrics["transcript_sha256"] = manifest["transcript"]["sha256"]
    metrics["parameters_sha256"] = manifest["parameters_sha256"]
    return metrics


def relabel_proxy(argv):
    """Boundary case: a target that emits a string outside the declared classes."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    parser.add_argument("--split", choices=("dev", "final"))
    parser.add_argument("--work")
    parser.add_argument("--output")
    parser.add_argument("--cases", help="comma-separated case names")
    parser.add_argument("--keep", action="store_true", help="keep work directories")
    parser.add_argument("--relabel-proxy")
    args = parser.parse_args()
    if args.relabel_proxy:
        relabel_proxy(json.loads(args.relabel_proxy))
        return 0
    repo = Path(__file__).resolve().parents[1]
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    wanted = set(args.cases.split(",")) if args.cases else None
    results = {
        "protocol": str(protocol_path.relative_to(repo))
        if protocol_path.is_relative_to(repo)
        else str(protocol_path),
        "protocol_sha256": _sha(protocol_path),
        "split": args.split,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
        ).stdout.strip(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "runs": [],
    }
    try:
        import torch  # noqa: E401
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
                "in_class": case["in_class"],
                "expected": case["expected_status"],
            }
            truth = build_case(run_case, seed, root, repo, python)
            completed, seconds, audit = run_recovery(run_case, root, python)
            entry["recovery_seconds"] = seconds
            entry["recovery_exit"] = completed.returncode
            entry["audit"] = {
                "denied_private_opens": len(audit["denied"]),
                "opened_files_outside_python": len(audit["opened_outside_python"]),
                "opened_under_private": [
                    p for p in audit["opened_outside_python"] if p.startswith(str(root / "private"))
                ],
                "spawned": audit["spawned"],
            }
            if completed.returncode:
                entry["status"] = "error"
                entry["error"] = completed.stderr.strip()[-600:]
            else:
                entry.update(evaluate(run_case, root, truth, python))
            if truth.get("corpus"):
                entry["corpus"] = truth["corpus"]
            entry["status_matches_expectation"] = entry["status"] == case["expected_status"]
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
                            "baseline_canonical_relative_l2_error",
                        )
                    }
                ),
                flush=True,
            )
            if not args.keep:
                shutil.rmtree(root)
    Path(args.output).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
