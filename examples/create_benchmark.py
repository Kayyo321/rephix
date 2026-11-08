"""Create Mode benchmarks, pre-registered in examples/create_protocol.json.

Reconstruction always runs as a separate ``rephix create`` process that receives only
the declared evidence. Source checkpoints are evaluator-only: during reconstruction
their directories have mode 000 and a Python audit hook logs every opened path; a run
that opens any source path is a failed run. Evaluator metrics are computed afterwards.

Usage (work directory outside the checkout; models and texts are downloaded pinned):
  python examples/create_benchmark.py --work ~/.local/share/rephix-create-041 prepare
  python examples/create_benchmark.py --work W e1        # planted same-architecture
  python examples/create_benchmark.py --work W e2        # different architecture
  python examples/create_benchmark.py --work W e3-head   # declared-class head
  python examples/create_benchmark.py --work W e3-ec     # Expand-and-Cluster
  python examples/create_benchmark.py --work W e4        # underdetermined/misspecified
  python examples/create_benchmark.py --work W e5        # corpus-only
  python examples/create_benchmark.py --work W e6        # tools and visible reasoning
Results are written as JSON under --results (default examples/results/create-0.4.1).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((REPO / "examples" / "create_protocol.json").read_text())
RESULTS = REPO / "examples" / "results" / "create-0.4.1"

OPENERS = [
    "The morning after the storm,",
    "It was generally agreed that",
    "In the small town of",
    "When the letter finally arrived,",
    "There is no doubt that",
    "My dear friend,",
    "The question of whether",
    "At the end of the long road",
    "Nobody in the family knew",
    "The first thing to consider is",
    "Years later, she would remember",
    "According to the report,",
    "The house stood on a hill",
    "He had always believed that",
    "The most remarkable feature of",
    "On the evening of the ball,",
    "It must be admitted that",
    "The children gathered around",
    "After a long silence,",
    "The committee concluded that",
    "In my opinion,",
    "The village was quiet until",
    "She turned to her sister and",
    "What surprised everyone was",
    "The road to the city",
    "If one considers the history of",
    "The old captain said that",
    "During the winter months,",
    "The secret of success is",
    "Her mother had often said that",
    "The experiment began with",
    "Among the many visitors,",
    "It is a truth that",
    "The garden behind the house",
    "Without any warning,",
    "The lawyer explained that",
    "Everyone expected the news",
    "The king's advisers",
    "From the window she could see",
    "The new teacher arrived",
]
TOPICS = [
    "",
    " the weather",
    " the harvest",
    " the journey",
    " the marriage",
    " the war",
    " the new railway",
    " the inheritance",
    " the neighbours",
    " the music",
    " money",
    " the sea",
]


def prompts():
    rows = []
    for i, opener in enumerate(OPENERS):
        for j, topic in enumerate(TOPICS):
            text = opener if not topic else f"{opener} concerning{topic},"
            rows.append((f"p{i:02d}-{j:02d}", text))
    return rows  # 480 prompts


# ----------------------------------------------------------------------------- setup


def venv_python():
    return sys.executable


def ensure_models(work):
    from huggingface_hub import snapshot_download

    root = work / "models"
    for name, spec in PROTOCOL["models"].items():
        target = root / name
        if not (target / "config.json").is_file():
            snapshot_download(
                spec["repo"],
                revision=spec["revision"],
                local_dir=target,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
                ignore_patterns=["onnx/*", "*/*"],
            )
    return root


def ensure_texts(work):
    import urllib.request

    root = work / "data"
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, spec in PROTOCOL["texts"].items():
        raw = root / f"pg{spec['gutenberg']}.txt"
        if not raw.is_file():
            url = f"https://www.gutenberg.org/cache/epub/{spec['gutenberg']}/pg{spec['gutenberg']}.txt"
            urllib.request.urlretrieve(url, raw)
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        if digest != spec["sha256"]:
            raise SystemExit(f"{raw}: sha256 {digest} != registered {spec['sha256']}")
        text = raw.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        start = text.index("\n", text.index("*** START OF THE PROJECT")) + 1
        end = text.index("*** END OF THE PROJECT")
        body = text[start:end].strip()
        # Undo typesetting line wraps inside paragraphs; keep paragraph breaks.
        paragraphs = [
            re.sub(r"\s*\n\s*", " ", p.strip()) for p in re.split(r"\n\s*\n", body) if p.strip()
        ]
        clean = root / f"{name}.txt"
        clean.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
        out[name] = clean
    return out


# ----------------------------------------------------------------------------- models


def load(path, device="cuda", dtype="float32"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype)).to(device)
    model.eval()
    return model, tokenizer


def _windows(tokenizer, text, length=256, limit=4000):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [ids[i : i + length] for i in range(0, min(len(ids), limit * length) - length, length)]


def plant_source(base_dir, out_dir, regions, style_text, seed, *, rank=4, steps=300, lr=3e-3):
    """Evaluator: base + rank-4 edits confined to ``regions``, trained on style text."""
    import torch

    from rephix.create.modeling import RegionEdits, collate, lora_frames, region_shapes, token_nll
    from rephix.create.serialize import Example

    model, tokenizer = load(base_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(seed + 1000)
    frames = lora_frames(rng, regions, region_shapes(model, regions), rank)
    edits = RegionEdits(torch, model, frames, train_frames=True, init="lora")
    optimizer = torch.optim.AdamW(list(edits.parameters()), lr=lr)
    windows = _windows(tokenizer, style_text)
    order = np.random.default_rng(seed).permutation(len(windows))
    pad = tokenizer.eos_token_id
    for step in range(steps):
        batch = [
            Example(f"w{i}", windows[i], [1] * len(windows[i]), "passage")
            for i in order[(step * 8) % len(order) : (step * 8) % len(order) + 8]
        ]
        if not batch:
            continue
        ids, attention, labels = collate(torch, batch, pad, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=attention).logits
        nll, count = token_nll(torch, logits, labels)
        loss = nll.sum() / count.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    deltas = {r: edits.delta(r).detach().double().cpu().numpy() for r in regions}
    edits.merge()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    return deltas, float(loss.item())


def permute_layer(source_dir, out_dir, layer, seed):
    """Same function, different coordinates: permute one layer's MLP neurons."""
    import torch

    model, tokenizer = load(source_dir)
    mlp = model.model.layers[layer].mlp
    perm = torch.as_tensor(
        np.random.default_rng(seed).permutation(mlp.gate_proj.out_features),
        device=mlp.gate_proj.weight.device,
    )
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(mlp.gate_proj.weight[perm])
        mlp.up_proj.weight.copy_(mlp.up_proj.weight[perm])
        mlp.down_proj.weight.copy_(mlp.down_proj.weight[:, perm])
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)


def sample_continuations(
    model_dir, rows, *, new_tokens=128, temperature=1.0, seed=0, greedy=False, batch=32
):
    import torch

    model, tokenizer = load(model_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(seed)
    outputs = []
    for start in range(0, len(rows), batch):
        chunk = [text for _, text in rows[start : start + batch]]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=new_tokens,
                do_sample=not greedy,
                temperature=temperature if not greedy else None,
                top_k=0 if not greedy else None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        for row in range(len(chunk)):
            outputs.append(
                tokenizer.decode(out[row, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            )
    del model
    torch.cuda.empty_cache()
    return outputs


def write_passages(path, rows, outputs, *, drop_context=False, decoding=None):
    lines = []
    for (pid, prompt), text in zip(rows, outputs):
        if not text.strip():
            continue
        # Prompts sharing an opener are related paraphrases: one leakage group.
        record = {
            "id": pid,
            "kind": "passage",
            "text": text,
            "group_id": pid.split("-")[0],
            "decoding": decoding or {},
        }
        if drop_context:
            record["context_status"] = "missing"
        else:
            record["context"] = prompt
        lines.append(json.dumps(record))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


# ----------------------------------------------------------------------------- isolation

_LAUNCHER = r"""
import json, sys
log_path, forbidden = sys.argv[1], json.loads(sys.argv[2])
argv = json.loads(sys.argv[3])
hits = []
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        path = args[0].decode() if isinstance(args[0], bytes) else args[0]
        for root in forbidden:
            if path.startswith(root):
                hits.append(path)
sys.addaudithook(hook)
from rephix.cli import main
code = main(argv)
json.dump({"forbidden_opens": hits, "exit": code}, open(log_path, "w"))
sys.exit(code)
"""


def run_create(argv, *, forbidden=(), log_dir):
    """Run ``rephix create`` in a separate process with source directories locked."""
    log_dir.mkdir(parents=True, exist_ok=True)
    modes = {}
    for path in forbidden:
        modes[path] = os.stat(path).st_mode
        os.chmod(path, 0)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [
                venv_python(),
                "-c",
                _LAUNCHER,
                str(log_dir / "audit.json"),
                json.dumps([str(Path(p).resolve()) for p in forbidden]),
                json.dumps(argv),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
    finally:
        for path, mode in modes.items():
            os.chmod(path, stat.S_IMODE(mode))
    (log_dir / "stderr.log").write_text(completed.stderr)
    audit = (
        json.loads((log_dir / "audit.json").read_text())
        if (log_dir / "audit.json").is_file()
        else {}
    )
    if completed.returncode != 0:
        raise RuntimeError(f"create failed ({completed.returncode}): {completed.stderr[-2500:]}")
    if audit.get("forbidden_opens"):
        raise RuntimeError(f"reconstruction opened source paths: {audit['forbidden_opens'][:5]}")
    return {
        "wall_seconds": round(time.monotonic() - started, 1),
        "isolation": {
            "source_dirs_mode_000": [str(p) for p in forbidden],
            "forbidden_opens": audit.get("forbidden_opens", []),
        },
    }


def create_args(base, out, *, evidence=None, corpus=None, method="structured", seed=0, extra=()):
    budgets = PROTOCOL["budgets"]
    argv = [
        "create",
        "--base",
        str(base),
        "--output",
        str(out),
        "--method",
        method,
        "--seed",
        str(seed),
        "--profile",
        budgets["profile"],
        "--steps",
        str(budgets["steps"]),
        "--eval-every",
        str(budgets["eval_every"]),
        "--regions",
        str(budgets["regions"]),
        "--rank",
        str(budgets["rank"]),
    ]
    if evidence:
        argv += ["--evidence", str(evidence)]
    if corpus:
        argv += ["--corpus", str(corpus)]
    return argv + list(extra)


def summary(run_dir):
    results = json.loads((Path(run_dir) / "results.json").read_text())
    ev = results["evaluation"]
    gen = ev.get("generation") or {}

    def style(label, key="style_distance_held_out"):
        return ((gen.get(label) or {}).get(key) or {}).get("distance")

    structure = results.get("structure") or {}
    ident = results.get("identification") or {}
    fit = ev.get("fit") or {}
    return {
        "method": results["plan"]["method"],
        "trainable_parameters": fit.get("trainable_parameters"),
        "validation_nll": ev["validation_nll"]["created_nll"],
        "base_validation_nll": ev["validation_nll"]["base_nll"],
        "test_nll": ev["test_nll"]["created_nll"],
        "base_test_nll": ev["test_nll"]["base_nll"],
        "test_improvement": ev["test_nll"]["improvement_nats_per_token"],
        "style_held_out": {"base": style("base"), "created": style("created")},
        "style_new_topics": {
            "base": style("base", "style_distance_new_topics"),
            "created": style("created", "style_distance_new_topics"),
        },
        "fluency_base_nll": {
            k: (gen.get(k) or {}).get("base_model_nll_of_generation") for k in ("base", "created")
        },
        "overlap8": (gen.get("created") or {}).get("ngram8_overlap_with_training_mean"),
        "longest_copied": (gen.get("created") or {}).get("longest_copied_words_max"),
        "fact_cloze": ev.get("fact_cloze"),
        "retain_kl": ev.get("retain_kl_evaluation"),
        "agent": {k: v for k, v in (ev.get("agent") or {}).items() if k in ("base", "created")},
        "trial_summary": structure.get("trial_summary"),
        "selected_regions": structure.get("selected_regions"),
        "ranked_top": (structure.get("ranked") or [])[:24],
        "region_class_test": {
            k: v
            for k, v in (ident.get("region_class_test") or {}).items()
            if k
            in (
                "status",
                "best_region",
                "single_region_fits",
                "unrestricted_validation_nll",
                "seed_variation_of_delta_x",
            )
        },
        "claims": {
            k: v["status"] for k, v in (results.get("claims") or {}).items() if isinstance(v, dict)
        },
        "export_passed": ((results.get("export") or {}).get("verification") or {}).get("passed"),
        "resources": ev.get("resources_measured"),
        "timings": results.get("timings_seconds"),
    }


# ----------------------------------------------------------------------------- evaluator


def _test_records(run_dir):
    from rephix.create.records import CreateRecord

    split = json.loads((Path(run_dir) / "evidence" / "splits.json").read_text())
    rows = [
        json.loads(line)
        for line in (Path(run_dir) / "evidence" / "records.jsonl").read_text().split("\n")
        if line
    ]
    return [r for r in rows if split["assignment"][r["id"]] == "test"], CreateRecord


def source_kl(source_dir, model_dir, run_dir, limit=120):
    """Evaluator: mean KL(source || model) per target token on test passages."""
    import torch

    rows, _ = _test_records(run_dir)
    rows = [r for r in rows if r.get("context")][:limit]
    source, tokenizer = load(source_dir)
    model, _ = load(model_dir)
    total, tokens = 0.0, 0
    with torch.inference_mode():
        for row in rows:
            context = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
            target = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
            ids = torch.tensor([context + target], device="cuda")
            p = torch.log_softmax(
                source(input_ids=ids).logits.float()[0, len(context) - 1 : -1], -1
            )
            q = torch.log_softmax(model(input_ids=ids).logits.float()[0, len(context) - 1 : -1], -1)
            total += float((p.exp() * (p - q)).sum().item())
            tokens += len(target)
    del source, model
    torch.cuda.empty_cache()
    return total / max(tokens, 1)


def source_judge(source_dir, texts):
    """Evaluator: NLL per token of texts under the source (any tokenizer)."""
    import torch

    model, tokenizer = load(source_dir)
    total, tokens = 0.0, 0
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt")["input_ids"][:, :256].to("cuda")
            if ids.shape[1] < 2:
                continue
            out = model(input_ids=ids, labels=ids)
            total += float(out.loss.item()) * (ids.shape[1] - 1)
            tokens += ids.shape[1] - 1
    del model
    torch.cuda.empty_cache()
    return total / max(tokens, 1)


def held_out_generations(model_dir, run_dir, *, count=48, new_tokens=96, seed=5):
    rows, _ = _test_records(run_dir)
    rows = [(r["id"], r["context"]) for r in rows if r.get("context")][:count]
    return sample_continuations(model_dir, rows, new_tokens=new_tokens, temperature=0.8, seed=seed)


def region_weight(model_dir, region):
    from safetensors import safe_open

    layer, block, module = region.split(".")
    name = f"model.layers.{layer}.{block}.{module}.weight"
    for file in Path(model_dir).glob("*.safetensors"):
        with safe_open(str(file), framework="pt") as handle:
            if name in handle.keys():
                return handle.get_tensor(name).double().numpy()
    raise KeyError(name)


def region_inputs(base_dir, region, run_dir, limit=4096):
    import torch

    from rephix.create.modeling import region_module

    model, tokenizer = load(base_dir)
    rows, _ = _test_records(run_dir)
    captured = []
    handle = region_module(model, region).register_forward_hook(
        lambda m, i, o: captured.append(i[0].detach().float().reshape(-1, i[0].shape[-1]).cpu())
    )
    with torch.inference_mode():
        for row in rows[:40]:
            ids = tokenizer((row.get("context") or "") + row["text"], return_tensors="pt")[
                "input_ids"
            ].to("cuda")
            model(input_ids=ids)
    handle.remove()
    del model
    return torch.cat(captured)[:limit].double().numpy()


def block_delta(base_dir, created_dir, source_dir, region, run_dir, limit=24):
    """Evaluator: change of the planted region's whole block (self_attn or mlp) output.

    The block receives identical inputs (captured from the base) in base, created and
    source models, so edits that are equivalent within the block (for example a v_proj
    edit standing in for an o_proj edit) compare as equal. Added after registration as
    a symmetry-aware evaluator metric; it does not change any reconstruction.
    """
    import torch

    layer, block, _ = region.split(".")
    rows, _ = _test_records(run_dir)
    model, tokenizer = load(base_dir)
    captured = []

    def pre(module, args, kwargs):
        captured.append(
            (
                tuple(a.detach() if torch.is_tensor(a) else a for a in args),
                {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("past_key_values", "past_key_value", "use_cache", "cache_position")
                },
            )
        )

    target = getattr(model.model.layers[int(layer)], block)
    handle = target.register_forward_pre_hook(pre, with_kwargs=True)
    with torch.inference_mode():
        for row in rows[:limit]:
            ids = tokenizer((row.get("context") or "") + row["text"], return_tensors="pt")[
                "input_ids"
            ].to("cuda")
            model(input_ids=ids, use_cache=False)
    handle.remove()
    del model
    outputs = {}
    for name, directory in (("base", base_dir), ("created", created_dir), ("source", source_dir)):
        m, _ = load(directory)
        module = getattr(m.model.layers[int(layer)], block)
        chunks = []
        with torch.inference_mode():
            for args, kwargs in captured:
                out = module(*args, **kwargs)
                out = out[0] if isinstance(out, tuple) else out
                chunks.append(out.double().reshape(-1, out.shape[-1]).cpu())
        outputs[name] = torch.cat(chunks)
        del m
        torch.cuda.empty_cache()
    created = outputs["created"] - outputs["base"]
    planted = outputs["source"] - outputs["base"]
    return {
        "block": f"{layer}.{block}",
        "relative_error": float((created - planted).norm() / planted.norm().clamp_min(1e-30)),
        "cosine": float(
            (created * planted).sum() / (created.norm() * planted.norm()).clamp_min(1e-30)
        ),
        "created_change_norm_ratio": float(created.norm() / planted.norm().clamp_min(1e-30)),
    }


def delta_error(true_delta, estimate, inputs):
    a = inputs @ true_delta.T
    b = inputs @ estimate.T
    rel = float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30))
    cos = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    return {"relative_error_delta_x": rel, "cosine_delta_x": cos}


# ----------------------------------------------------------------------------- experiments


def _save(name, data):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=1, default=float) + "\n")


def _parallel(jobs, workers=2):
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fn): key for key, fn in jobs}
        out = {}
        for future in concurrent.futures.as_completed(futures):
            out[futures[future]] = future.result()
    return out


def e1(work, seeds=None, methods=None):
    from rephix.create.base import resolve_contract

    models = work / "models"
    texts = ensure_texts(work)
    base = models / "smollm2-135m"
    regions = resolve_contract(base)["regions"]
    style = texts["austen-pride"].read_text()
    spec = PROTOCOL["experiments"]["E1_planted_same_architecture"]
    seeds = seeds if seeds is not None else spec["seeds"]
    methods = methods or spec["methods"]
    results = {}
    for seed in seeds:
        secret = random.Random(f"rephix-e1-secret-{seed}-{os.urandom(8).hex()}")
        region = secret.choice(regions)
        root = work / "e1" / f"seed{seed}"
        source = root / "evaluator-only" / "source"
        if not (source / "config.json").is_file():
            deltas, loss = plant_source(base, source, [region], style, seed)
            np.savez(root / "evaluator-only" / "truth.npz", delta=deltas[region])
            (root / "evaluator-only" / "truth.json").write_text(
                json.dumps({"region": region, "plant_final_loss": loss})
            )
        truth = json.loads((root / "evaluator-only" / "truth.json").read_text())
        region = truth["region"]
        evidence = root / "evidence.jsonl"
        if not evidence.is_file():
            rows = prompts()
            outputs = sample_continuations(source, rows, seed=seed)
            write_passages(
                evidence, rows, outputs, decoding={"temperature": 1.0, "declared_by": "benchmark"}
            )
        jobs = []
        for method in methods:
            out = root / "runs" / method
            if (out / "results.json").is_file() and json.loads((out / "state.json").read_text())[
                "status"
            ] == "complete":
                continue
            if out.exists():
                shutil.rmtree(out)
            argv = create_args(
                base,
                out,
                evidence=evidence,
                method=method,
                seed=seed,
                extra=["--no-identify"] if method != "structured" else [],
            )
            jobs.append(
                (
                    method,
                    lambda argv=argv, method=method: run_create(
                        argv, forbidden=[root / "evaluator-only"], log_dir=root / "logs" / method
                    ),
                )
            )
        runs = _parallel(jobs)
        seed_result = {
            "secret_region_revealed_after_runs": region,
            "plant_final_loss": truth["plant_final_loss"],
            "methods": {},
        }
        true_delta = np.load(root / "evaluator-only" / "truth.npz")["delta"]
        inputs = None
        for method in methods:
            out = root / "runs" / method
            entry = summary(out)
            entry["run"] = runs.get(method)
            entry["kl_source_to_created"] = source_kl(source, out / "export", out)
            if method == "structured":
                entry["kl_source_to_base"] = source_kl(source, base, out)
                ranked = json.loads((out / "phases" / "structure.json").read_text())["ranked"]
                entry["planted_region_rank"] = ranked.index(region) + 1
                entry["planted_layer_rank"] = min(
                    i + 1 for i, r in enumerate(ranked) if r.split(".")[0] == region.split(".")[0]
                )
                if inputs is None:
                    inputs = region_inputs(base, region, out)
                estimate = region_weight(out / "export", region) - region_weight(base, region)
                entry["planted_region_selected"] = region in entry["selected_regions"]
                entry["delta_x_export"] = delta_error(true_delta, estimate, inputs)
                entry["block_delta_vs_planted"] = block_delta(
                    base, out / "export", source, region, out
                )
                ident = json.loads((out / "phases" / "identify.json").read_text())[
                    "region_class_test"
                ]
                entry["identify_best_region_is_planted"] = ident.get("best_region") == region
            seed_result["methods"][method] = entry
        results[f"seed{seed}"] = seed_result
        _save("e1_planted", results)
    return results


def e2(work):
    models = work / "models"
    spec = PROTOCOL["experiments"]["E2_different_architecture"]
    root = work / "e2"
    source = models / "gpt2"
    evidence = root / "evidence.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    if not evidence.is_file():
        rows = prompts()
        outputs = sample_continuations(source, rows, seed=11)
        write_passages(
            evidence, rows, outputs, decoding={"temperature": 1.0, "declared_by": "benchmark"}
        )
    results = {}
    for base_name in spec["bases"]:
        base = models / base_name
        for seed in spec["seeds"][base_name]:
            jobs = []
            for method in spec["methods"][base_name]:
                out = root / base_name / f"seed{seed}" / method
                if (out / "state.json").is_file() and json.loads((out / "state.json").read_text())[
                    "status"
                ] == "complete":
                    continue
                if out.exists():
                    shutil.rmtree(out)
                argv = create_args(
                    base,
                    out,
                    evidence=evidence,
                    method=method,
                    seed=seed,
                    extra=["--no-identify"] if method != "structured" else [],
                )
                jobs.append(
                    (
                        method,
                        lambda argv=argv, method=method, out=out: run_create(
                            argv, forbidden=[], log_dir=out.parent / "logs" / method
                        ),
                    )
                )
            runs = _parallel(jobs, workers=2 if base_name.startswith("smollm") else 1)
            for method in spec["methods"][base_name]:
                out = root / base_name / f"seed{seed}" / method
                entry = summary(out)
                entry["run"] = runs.get(method)
                generations = held_out_generations(out / "export", out)
                entry["source_judge_nll_created"] = source_judge(source, generations)
                if method == spec["methods"][base_name][0]:
                    entry["source_judge_nll_base"] = source_judge(
                        source, held_out_generations(base, out)
                    )
                    entry["source_judge_nll_source_itself"] = source_judge(
                        source, held_out_generations(source, out)
                    )
                results.setdefault(base_name, {}).setdefault(f"seed{seed}", {})[method] = entry
            _save("e2_cross_architecture", results)
    return results


def e4(work):
    from rephix.create.base import resolve_contract

    models = work / "models"
    base = models / "smollm2-135m"
    texts = ensure_texts(work)
    results = {}
    e1root = work / "e1" / "seed0"
    truth = json.loads((e1root / "evaluator-only" / "truth.json").read_text())
    region = truth["region"]
    layer = int(region.split(".")[0])
    root = work / "e4"
    # Permuted-equivalent source.
    permuted = root / "permuted" / "evaluator-only" / "source"
    if not (permuted / "config.json").is_file():
        permute_layer(e1root / "evaluator-only" / "source", permuted, layer, seed=99)
    evidence = root / "permuted" / "evidence.jsonl"
    if not evidence.is_file():
        rows = prompts()
        write_passages(
            evidence,
            rows,
            sample_continuations(permuted, rows, seed=0),
            decoding={"temperature": 1.0, "declared_by": "benchmark"},
        )
    out = root / "permuted" / "run"
    if not (out / "results.json").is_file():
        run_create(
            create_args(base, out, evidence=evidence),
            forbidden=[permuted.parent],
            log_dir=root / "permuted" / "logs",
        )
    entry = summary(out)
    true_delta = np.load(e1root / "evaluator-only" / "truth.npz")["delta"]
    estimate = region_weight(out / "export", region) - region_weight(base, region)
    inputs = region_inputs(base, region, out)
    # In the permuted source's own coordinates the edit is spread over the whole MLP.
    permuted_coordinates = {}
    for module in ("gate_proj", "up_proj", "down_proj"):
        name = f"{layer}.mlp.{module}"
        permuted_coordinates[module] = float(
            np.linalg.norm(region_weight(permuted, name) - region_weight(base, name))
        )
    entry["evaluator"] = {
        "kl_source_vs_permuted_source": source_kl(
            e1root / "evaluator-only" / "source", permuted, out
        ),
        "delta_x_vs_unpermuted_truth": delta_error(true_delta, estimate, inputs)
        if region in (entry["selected_regions"] or [])
        else None,
        "permuted_source_minus_base_frobenius_by_module": permuted_coordinates,
        "planted_region": region,
    }
    results["permuted_equivalent_source"] = entry
    _save("e4_underdetermined", results)
    # Out-of-class: three regions in different layers.
    ooc = root / "out-of-class" / "evaluator-only" / "source"
    rng = random.Random(f"e4-ooc-{os.urandom(4).hex()}")
    if not (ooc / "config.json").is_file():
        regions_all = resolve_contract(base)["regions"]
        layers = rng.sample(range(30), 3)
        chosen = [rng.choice([r for r in regions_all if r.startswith(f"{lay}.")]) for lay in layers]
        deltas, _ = plant_source(base, ooc, chosen, texts["austen-pride"].read_text(), seed=7)
        (ooc.parent / "truth.json").write_text(json.dumps({"regions": chosen}))
    evidence = root / "out-of-class" / "evidence.jsonl"
    if not evidence.is_file():
        rows = prompts()
        write_passages(
            evidence,
            rows,
            sample_continuations(ooc, rows, seed=1),
            decoding={"temperature": 1.0, "declared_by": "benchmark"},
        )
    out = root / "out-of-class" / "run"
    if not (out / "results.json").is_file():
        run_create(
            create_args(base, out, evidence=evidence),
            forbidden=[ooc.parent],
            log_dir=root / "out-of-class" / "logs",
        )
    entry = summary(out)
    entry["evaluator"] = {
        "planted_regions": json.loads((ooc.parent / "truth.json").read_text())["regions"]
    }
    results["out_of_class_source"] = entry
    _save("e4_underdetermined", results)
    # Missing context.
    rows = prompts()
    e1_evidence = [
        json.loads(line) for line in (e1root / "evidence.jsonl").read_text().splitlines()
    ]
    missing = root / "missing-context" / "evidence.jsonl"
    missing.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in e1_evidence:
        record = dict(record)
        record.pop("context", None)
        record["context_status"] = "missing"
        lines.append(json.dumps(record))
    missing.write_text("\n".join(lines) + "\n")
    out = root / "missing-context" / "run"
    if not (out / "results.json").is_file():
        run_create(
            create_args(base, out, evidence=missing),
            forbidden=[e1root / "evaluator-only"],
            log_dir=root / "missing-context" / "logs",
        )
    entry = summary(out)
    results["missing_context"] = entry
    _save("e4_underdetermined", results)
    # Sparse corpus.
    words = texts["darwin-origin"].read_text().split()
    sparse = root / "sparse" / "corpus.txt"
    sparse.parent.mkdir(parents=True, exist_ok=True)
    # Keep paragraph structure for the first 300 words.
    text, count = [], 0
    for paragraph in texts["darwin-origin"].read_text().split("\n\n"):
        text.append(paragraph)
        count += len(paragraph.split())
        if count >= 300:
            break
    sparse.write_text("\n\n".join(text) + "\n")
    out = root / "sparse" / "run"
    try:
        if not (out / "results.json").is_file():
            run_create(
                create_args(base, out, corpus=sparse, extra=["--section-chars", "200"]),
                log_dir=root / "sparse" / "logs",
            )
        entry = summary(out)
        entry["hypothesis_limitation"] = json.loads(
            (out / "phases" / "hypotheses.json").read_text()
        )["style_and_behavior"]["limitation"]
    except RuntimeError as exc:
        entry = {"refused_or_failed": str(exc)[-600:]}
    del words
    results["sparse_corpus"] = entry
    _save("e4_underdetermined", results)
    return results


def e5(work):
    models = work / "models"
    texts = ensure_texts(work)
    corpus = texts["darwin-origin"]
    spec = PROTOCOL["experiments"]["E5_corpus_only"]
    root = work / "e5"
    results = {}

    def runs_for(base_name, seed, method, extra=(), label=None):
        label = label or method
        out = root / base_name / f"seed{seed}" / label
        if not (
            (out / "state.json").is_file()
            and json.loads((out / "state.json").read_text())["status"] == "complete"
        ):
            if out.exists():
                shutil.rmtree(out)
            argv = create_args(
                models / base_name,
                out,
                corpus=corpus,
                method=method,
                seed=seed,
                extra=list(extra) + (["--no-identify"] if method != "structured" else []),
            )
            return label, lambda: run_create(argv, log_dir=out.parent / "logs" / label)
        return label, lambda: None

    for base_name in spec["bases"]:
        for seed in spec["seeds"][base_name]:
            jobs = [runs_for(base_name, seed, m) for m in spec["methods"][base_name]]
            if base_name == spec["instructed"]["base"] and seed in spec["instructed"]["seeds"]:
                jobs.append(
                    runs_for(
                        base_name,
                        seed,
                        "structured",
                        ["--prompt", spec["instructed"]["prompt"]],
                        "structured-instructed",
                    )
                )
            runs = _parallel(jobs, workers=2 if base_name.startswith("smollm") else 1)
            for label in runs:
                out = root / base_name / f"seed{seed}" / label
                entry = summary(out)
                entry["run"] = runs[label]
                results.setdefault(base_name, {}).setdefault(f"seed{seed}", {})[label] = entry
            _save("e5_corpus", results)
    return results


def e3_head(work):
    """Declared-class head: base backbone + private head over 4 labels."""
    import torch

    models = work / "models"
    base = models / "smollm2-135m"
    model, tokenizer = load(base)
    labels = ["alpha", "beta", "gamma", "delta"]
    rows = prompts()[:60]
    root = work / "e3-head"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    features = {}
    with torch.inference_mode():
        for layer_choice in ("final", "layer20"):
            feats = []
            for _, text in rows:
                ids = tokenizer(text, return_tensors="pt")["input_ids"].to("cuda")
                out = model(input_ids=ids, output_hidden_states=True)
                hidden = (
                    model.model(input_ids=ids).last_hidden_state[0, -1]
                    if layer_choice == "final"
                    else out.hidden_states[20][0, -1]
                )
                feats.append(np.concatenate([hidden.double().cpu().numpy(), [1.0]]))
            features[layer_choice] = np.array(feats)
    theta = rng.standard_normal((len(labels), features["final"].shape[1]))
    theta *= 1.5 / (features["final"] @ theta.T).std()
    results = {}
    for condition in ("sampled", "greedy", "wrong-backbone-layer"):
        feats = features["layer20" if condition == "wrong-backbone-layer" else "final"]
        logits = feats @ theta.T
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        lines = []
        for i, (pid, text) in enumerate(rows):
            if condition == "greedy":
                draws = [int(np.argmax(p[i]))] * 300
            else:
                draws = rng.choice(len(labels), size=300, p=p[i])
            for j, label in enumerate(draws):
                lines.append(
                    json.dumps(
                        {
                            "id": f"{pid}-{j}",
                            "kind": "passage",
                            "context": text,
                            "text": labels[label],
                            "group_id": pid,
                        }
                    )
                )
        evidence = root / condition / "evidence.jsonl"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("\n".join(lines) + "\n")
        out = root / condition / "run"
        if out.exists():
            shutil.rmtree(out)
        run_create(
            [
                "create",
                "--base",
                str(base),
                "--evidence",
                str(evidence),
                "--output",
                str(out),
                "--method",
                "lora",
                "--no-identify",
                "--stop-after",
                "identify",
                "--profile",
                "rtx5080-16gb",
            ],
            log_dir=root / condition / "logs",
        )
        head = json.loads((out / "phases" / "identify.json").read_text())["declared_class_head"]
        entry = {k: v for k, v in head.items() if k != "canonical_parameters"}
        if head.get("canonical_parameters") is not None:
            split = json.loads((out / "evidence" / "splits.json").read_text())
            train_prompts = sorted(
                {
                    json.loads(line)["context"]
                    for line in lines
                    if split["assignment"][json.loads(line)["id"]] == "train"
                }
            )
            index = {text: i for i, (_, text) in enumerate(rows)}
            order = [index[t] for t in train_prompts]
            design = features["final"][order]
            _, s, vt = np.linalg.svd(design, full_matrices=False)
            basis = vt[: int((s > 1e-9 * s[0]).sum())].T
            projector = basis @ basis.T
            truth = (theta - theta.mean(0)) @ projector
            estimate = np.array(head["canonical_parameters"])
            entry["evaluator_relative_canonical_error"] = float(
                np.linalg.norm(estimate - truth) / np.linalg.norm(truth)
            )
            entry["evaluator_note"] = (
                "error of C·Θ·P_S against the truth's canonical form on the queried span"
            )
        results[condition] = entry
        _save("e3_declared_head", results)
    return results


def e3_ec(work):
    from rephix.create import expand_cluster as ec

    results = {"main": [], "expansion_1": [], "relu_teacher": [], "noisy": [], "string": []}
    rng = np.random.default_rng(2024)

    def inputs(d, n=3000):
        return rng.uniform(-math.sqrt(3), math.sqrt(3), size=(n, d))

    for d in (4, 8):
        for r in (2, 4):
            for t in range(3):
                teacher = ec.make_teacher(rng, d, r, 1, "tanh", scale=1.0)
                x = inputs(d)
                y = ec.teacher_forward(teacher, x, "tanh")
                x_test = inputs(d, 1000)
                # Main condition: width search, then expansion 4 (recovery sees only x, y).
                report, params = ec.recover(
                    x, y, activation="tanh", students=10, steps=3000, seed=t, expansion=4.0
                )
                width = report["student_width"]
                evaluation = ec.evaluate_recovery(params, teacher, "tanh", x_test)
                results["main"].append(
                    {"d": d, "r": r, "teacher": t, **_ec_row(report, evaluation, r)}
                )
                # Controls reuse the main condition's student width (no second search).
                noisy = y + rng.normal(0, 0.05 * y.std(), y.shape)
                for condition, yy, student_width in (
                    ("expansion_1", y, r),
                    ("noisy", noisy, width),
                ):
                    report, params = ec.recover(
                        x,
                        yy,
                        activation="tanh",
                        students=10,
                        steps=3000,
                        seed=t,
                        student_width=student_width,
                    )
                    evaluation = ec.evaluate_recovery(params, teacher, "tanh", x_test)
                    results[condition].append(
                        {"d": d, "r": r, "teacher": t, **_ec_row(report, evaluation, r)}
                    )
                relu = ec.make_teacher(rng, d, r, 1, "relu")
                yr = ec.teacher_forward(relu, x, "relu")
                report, params = ec.recover(
                    x, yr, activation="tanh", students=10, steps=3000, seed=t, student_width=width
                )
                evaluation = ec.evaluate_recovery(params, relu, "tanh", x_test)
                results["relu_teacher"].append(
                    {"d": d, "r": r, "teacher": t, **_ec_row(report, evaluation, r)}
                )
                _save("e3_expand_cluster", results)
    # String-only extension: sampled class labels instead of real outputs. The student
    # width is fixed at 4x the teacher width (an evaluator-known quantity, so a favourable
    # assumption), and every student is kept: there is no zero-loss convergence for
    # sampled labels. Recovery still sees only (x, label counts).
    for samples in (20, 1000, "argmax"):
        for t in range(3):
            teacher = ec.make_teacher(rng, 4, 3, 5, "tanh", scale=1.5)
            x = inputs(4, 1500)
            p = ec._softmax(ec.teacher_forward(teacher, x, "tanh"))
            counts = np.zeros_like(p)
            for i in range(len(x)):
                if samples == "argmax":
                    counts[i, int(np.argmax(p[i]))] = 50
                else:
                    counts[i] = rng.multinomial(samples, p[i])
            report, params = ec.recover(
                x,
                counts,
                activation="tanh",
                students=10,
                steps=3000,
                seed=t,
                student_width=12,
                mode="categorical",
                convergence=1e9,
                gamma=0.5,
                beta=math.pi / 12,
            )
            evaluation = ec.evaluate_recovery(
                params, teacher, "tanh", inputs(4, 1000), mode="categorical"
            )
            results["string"].append(
                {"samples_per_input": samples, "teacher": t, **_ec_row(report, evaluation, 3)}
            )
            _save("e3_expand_cluster", results)
    return results


def _ec_row(report, evaluation, r):
    exact = bool(
        evaluation.get("recovered")
        and evaluation["recovered_width"] == r
        and evaluation.get("max_cosine_distance") is not None
        and evaluation["max_cosine_distance"] < 0.01
    )
    return {
        "status": report["status"],
        "student_width": report["student_width"],
        "converged_students": report["converged_students"],
        "recovered_width": report.get("recovered_width"),
        "exact_recovery": exact,
        "evaluation": evaluation,
    }


# ----------------------------------------------------------------------------- tools

CITIES = {"Paris": 18, "Oslo": 4, "Cairo": 31, "Lima": 22, "Tokyo": 16, "Denver": 9}
CONVERSIONS = {
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.609344,
    ("kg", "lb"): 2.204623,
    ("lb", "kg"): 0.453592,
}


def mock_execute(name, arguments):
    """The benchmark's own mock environment. Parses structured arguments only."""
    try:
        if name in ("add", "subtract", "multiply"):
            a, b = int(arguments["a"]), int(arguments["b"])
            return {"add": a + b, "subtract": a - b, "multiply": a * b}[name]
        if name == "weather":
            return CITIES.get(str(arguments["city"]))
        if name == "convert":
            factor = CONVERSIONS[(arguments["from_unit"], arguments["to_unit"])]
            return round(float(arguments["value"]) * factor, 2)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def agent_dataset(count, seed):
    rng = random.Random(seed)
    phrasings = {
        "add": ["What is {a} plus {b}?", "Add {a} and {b}.", "Compute the sum of {a} and {b}."],
        "subtract": [
            "What is {a} minus {b}?",
            "Subtract {b} from {a}.",
            "How much is {a} take away {b}?",
        ],
        "multiply": [
            "What is {a} times {b}?",
            "Multiply {a} by {b}.",
            "Compute {a} multiplied by {b}.",
        ],
        "weather": [
            "What is the temperature in {city}?",
            "How warm is it in {city} today?",
            "Tell me the current temperature in {city}.",
        ],
        "convert": [
            "Convert {value} {from_unit} to {to_unit}.",
            "How many {to_unit} is {value} {from_unit}?",
        ],
    }
    records = []
    for index in range(count):
        tool = rng.choice(sorted(phrasings))
        if tool in ("add", "subtract", "multiply"):
            args = {"a": rng.randint(2, 99), "b": rng.randint(2, 99)}
        elif tool == "weather":
            args = {"city": rng.choice(sorted(CITIES))}
        else:
            pair = rng.choice(sorted(CONVERSIONS))
            args = {"value": rng.randint(1, 200), "from_unit": pair[0], "to_unit": pair[1]}
        request = rng.choice(phrasings[tool]).format(**args)
        result = mock_execute(tool, args)
        reasoning = rng.choice(
            [
                f"The user is asking for {tool}. I should call the {tool} tool.",
                f"This needs the {tool} tool with the given values.",
                f"I will use {tool} to answer this.",
            ]
        )
        answer = rng.choice([f"The answer is {result}.", f"It is {result}.", f"Result: {result}."])
        call_id = f"c{index}"
        records.append(
            {
                "id": f"t{index:04d}",
                "kind": "exchange",
                "conversation_id": f"t{index:04d}",
                "messages": [
                    {"role": "user", "content": request},
                    {
                        "role": "assistant",
                        "reasoning": reasoning,
                        "tool_calls": [{"id": call_id, "name": tool, "arguments": args}],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({"result": result}),
                    },
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {"expected": result},
            }
        )
    return records


def closed_loop(model_dir, records, *, limit=80):
    """Mock-environment task outcome: model's tool call -> mock result -> model's answer."""
    import torch

    from rephix.create.serialize import parse_tool_calls

    model, tokenizer = load(model_dir)
    outcomes = []
    for record in records[:limit]:
        messages = [{"role": "user", "content": record["messages"][0]["content"]}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            out = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        calls, malformed = parse_tool_calls(text)
        expected_call = record["messages"][1]["tool_calls"][0]
        row = {
            "tool_call_parsed": bool(calls),
            "malformed": malformed,
            "selection": bool(calls) and calls[0]["name"] == expected_call["name"],
            "arguments_exact": bool(calls) and calls[0]["arguments"] == expected_call["arguments"],
        }
        success = False
        if calls:
            result = mock_execute(calls[0]["name"], calls[0]["arguments"])
            messages += [
                {"role": "assistant", "content": text},
                {"role": "tool", "content": json.dumps({"result": result})},
            ]
            ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")
            with torch.inference_mode():
                out = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            final = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
            success = str(record["metadata"]["expected"]) in final
        row["task_success"] = success
        outcomes.append(row)
    del model
    torch.cuda.empty_cache()
    keys = ["tool_call_parsed", "selection", "arguments_exact", "task_success"]
    return {k: float(np.mean([o[k] for o in outcomes])) for k in keys} | {"items": len(outcomes)}


def e6(work):
    models = work / "models"
    base = models / "smollm2-135m-instruct"
    spec = PROTOCOL["experiments"]["E6_tools_and_visible_reasoning"]
    root = work / "e6"
    root.mkdir(parents=True, exist_ok=True)
    records = agent_dataset(600, seed=6)
    evidence = root / "traces.jsonl"
    evidence.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    results = {}
    for seed in spec["seeds"]:
        jobs = []
        for method in spec["methods"]:
            out = root / f"seed{seed}" / method
            if (out / "state.json").is_file() and json.loads((out / "state.json").read_text())[
                "status"
            ] == "complete":
                continue
            if out.exists():
                shutil.rmtree(out)
            argv = create_args(
                base,
                out,
                evidence=evidence,
                method=method,
                seed=seed,
                extra=["--no-identify"] if method != "structured" else [],
            )
            jobs.append(
                (
                    method,
                    lambda argv=argv, out=out, method=method: run_create(
                        argv, log_dir=out.parent / "logs" / method
                    ),
                )
            )
        runs = _parallel(jobs)
        for method in spec["methods"]:
            out = root / f"seed{seed}" / method
            entry = summary(out)
            entry["run"] = runs.get(method)
            test, _ = _test_records(out)
            entry["closed_loop_created"] = closed_loop(out / "export", test)
            if method == spec["methods"][0]:
                entry["closed_loop_base"] = closed_loop(base, test)
            results.setdefault(f"seed{seed}", {})[method] = entry
        _save("e6_tools", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path.home() / ".local/share/rephix-create-041")
    parser.add_argument(
        "experiment", choices=["prepare", "e1", "e2", "e3-head", "e3-ec", "e4", "e5", "e6"]
    )
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--methods", nargs="*")
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    if work.is_relative_to(REPO):
        raise SystemExit("keep the work directory outside the checkout")
    work.mkdir(parents=True, exist_ok=True)
    if args.experiment == "prepare":
        ensure_models(work)
        print(json.dumps({k: str(v) for k, v in ensure_texts(work).items()}))
        return
    function = {
        "e1": lambda: e1(work, args.seeds, args.methods),
        "e2": lambda: e2(work),
        "e3-head": lambda: e3_head(work),
        "e3-ec": lambda: e3_ec(work),
        "e4": lambda: e4(work),
        "e5": lambda: e5(work),
        "e6": lambda: e6(work),
    }
    function[args.experiment]()


if __name__ == "__main__":
    main()
