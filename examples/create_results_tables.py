"""Render Create Mode benchmark JSON (examples/results/create-0.4.1) as Markdown tables
and evaluate the pre-registered acceptance criteria. Prints to stdout."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "examples" / "results" / "create-0.4.1"


def load(name):
    path = ROOT / f"{name}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def f(x, digits=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def interval(entry):
    imp = entry["test_improvement"]
    return f"{imp['mean']:.3f} [{imp['low']:.3f}, {imp['high']:.3f}]"


def method_table(runs, methods, extra=()):
    lines = [
        "| Seed | Method | Trainable params | Test NLL (base → created) | Improvement [95% CI] | "
        "Style distance (base → created) | Fact cloze (base → created) | Retain KL | "
        + " | ".join(name for name, _ in extra)
        + (" |" if extra else "|"),
        "|" + " --- |" * (8 + len(extra)),
    ]
    for seed, by_method in runs.items():
        for method in methods:
            e = by_method.get(method)
            if not e:
                continue
            style = e["style_held_out"]
            cloze = e["fact_cloze"] or {}
            cells = [
                seed,
                method,
                f(e["trainable_parameters"]),
                f"{e['base_test_nll']:.3f} → {e['test_nll']:.3f}",
                interval(e),
                f"{f(style['base'])} → {f(style['created'])}",
                f"{f(cloze.get('base_accuracy'), 2)} → {f(cloze.get('created_accuracy'), 2)}",
                f(e["retain_kl"], 4),
            ] + [fn(e) for _, fn in extra]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mean_gap(runs, worse, better="structured"):
    gaps = [
        r[worse]["test_nll"] - r[better]["test_nll"]
        for r in runs.values()
        if worse in r and better in r
    ]
    return gaps


def main():
    out = []
    criteria = {}
    e1 = load("e1_planted")
    if e1:
        runs = {seed: data["methods"] for seed, data in e1.items()}
        out.append("### E1 planted same-architecture source\n")
        out.append(
            method_table(
                runs,
                ["structured", "random-subspace", "random-regions", "lora", "lora-matched", "sft"],
                extra=[("KL(source‖model)", lambda e: f(e.get("kl_source_to_created"), 4))],
            )
        )
        out.append("")
        rows = [
            "| Seed | Planted region (revealed after runs) | Rank of planted region (of 210) | "
            "Rank of its layer | Selected | Region-class test | Best region | ΔX relative error | ΔX cosine | KL(source‖base) |",
            "|" + " --- |" * 10,
        ]
        hits = 0
        for seed, data in e1.items():
            s = data["methods"].get("structured", {})
            rc = s.get("region_class_test") or {}
            dx = s.get("delta_x_export") or {}
            rows.append(
                "| "
                + " | ".join(
                    [
                        seed,
                        data["secret_region_revealed_after_runs"],
                        f(s.get("planted_region_rank")),
                        f(s.get("planted_layer_rank")),
                        f(s.get("planted_region_selected")),
                        f(rc.get("status")),
                        f(rc.get("best_region")),
                        f(dx.get("relative_error_delta_x")),
                        f(dx.get("cosine_delta_x")),
                        f(s.get("kl_source_to_base"), 4),
                    ]
                )
                + " |"
            )
            hits += (s.get("planted_region_rank") or 999) <= 12
        out.append("\n".join(rows))
        criteria["A4_planted_localization"] = f"{hits}/{len(e1)} seeds within top 12: " + (
            "passed" if hits >= 2 else "failed"
        )
        for name, worse in (
            ("A1_selection_beats_random_regions (E1)", "random-regions"),
            ("A2_subspaces_beat_random_subspaces (E1)", "random-subspace"),
        ):
            gaps = mean_gap(runs, worse)
            ok = gaps and statistics.mean(gaps) > 0.005 and all(g > 0 for g in gaps)
            criteria[name] = f"gaps {[round(g, 4) for g in gaps]}: " + (
                "passed" if ok else "failed"
            )
    e2 = load("e2_cross_architecture")
    if e2:
        out.append("\n### E2 different-architecture source (GPT-2)\n")
        for base, runs in e2.items():
            out.append(f"Base `{base}`:\n")
            out.append(
                method_table(
                    runs,
                    ["structured", "random-regions", "lora", "sft"],
                    extra=[
                        (
                            "Source-judged NLL of generations (created)",
                            lambda e: f(e.get("source_judge_nll_created")),
                        ),
                        (
                            "(base / source itself)",
                            lambda e: (
                                f"{f(e.get('source_judge_nll_base'))} / {f(e.get('source_judge_nll_source_itself'))}"
                            ),
                        ),
                    ],
                )
            )
            out.append("")
            if base.startswith("smollm"):
                gaps = mean_gap(runs, "random-regions")
                ok = gaps and statistics.mean(gaps) > 0.005 and all(g > 0 for g in gaps)
                criteria["A1_selection_beats_random_regions (E2)"] = (
                    f"gaps {[round(g, 4) for g in gaps]}: " + ("passed" if ok else "failed")
                )
    e5 = load("e5_corpus")
    if e5:
        out.append("\n### E5 corpus only (Darwin, *On the Origin of Species*, 1859)\n")
        for base, runs in e5.items():
            out.append(f"Base `{base}`:\n")
            out.append(
                method_table(
                    runs,
                    [
                        "structured",
                        "structured-instructed",
                        "random-subspace",
                        "random-regions",
                        "lora",
                        "sft",
                    ],
                    extra=[
                        (
                            "New-topic style (base → created)",
                            lambda e: (
                                f"{f(e['style_new_topics']['base'])} → {f(e['style_new_topics']['created'])}"
                            ),
                        ),
                        (
                            "8-gram overlap / longest copy",
                            lambda e: f"{f(e.get('overlap8'), 4)} / {f(e.get('longest_copied'))}",
                        ),
                        (
                            "Base-judged NLL of generations (base → created)",
                            lambda e: (
                                f"{f(e['fluency_base_nll']['base'])} → {f(e['fluency_base_nll']['created'])}"
                            ),
                        ),
                    ],
                )
            )
            out.append("")
            if base.startswith("smollm"):
                for name, worse in (
                    ("A1_selection_beats_random_regions (E5)", "random-regions"),
                    ("A2_subspaces_beat_random_subspaces (E5)", "random-subspace"),
                ):
                    gaps = mean_gap(runs, worse)
                    ok = gaps and statistics.mean(gaps) > 0.005 and all(g > 0 for g in gaps)
                    criteria[name] = f"gaps {[round(g, 4) for g in gaps]}: " + (
                        "passed" if ok else "failed"
                    )
    e6 = load("e6_tools")
    if e6:
        out.append("\n### E6 tools and visible reasoning\n")
        rows = [
            "| Seed | Method | Test NLL (base → created) | Schema valid | Selection | Args exact | Reasoning F1 | Answer F1 | Closed-loop task success (base → created) |",
            "|" + " --- |" * 9,
        ]
        for seed, by_method in e6.items():
            base_loop = next(
                (
                    e.get("closed_loop_base")
                    for e in by_method.values()
                    if e.get("closed_loop_base")
                ),
                {},
            )
            for method, e in by_method.items():
                a = (e.get("agent") or {}).get("created") or {}
                rows.append(
                    "| "
                    + " | ".join(
                        [
                            seed,
                            method,
                            f"{e['base_test_nll']:.3f} → {e['test_nll']:.3f}",
                            f(a.get("schema_valid_fraction")),
                            f(a.get("selection_accuracy")),
                            f(a.get("arguments_exact")),
                            f(a.get("reasoning_token_f1")),
                            f(a.get("visible_answer_token_f1")),
                            f"{f(base_loop.get('task_success'))} → {f((e.get('closed_loop_created') or {}).get('task_success'))}",
                        ]
                    )
                    + " |"
                )
        out.append("\n".join(rows))
    head = load("e3_declared_head")
    if head:
        out.append("\n### E3 declared-class head through Create Mode\n")
        rows = [
            "| Condition | Status | Relative SE | Class test p (upper / lower) | Evaluator canonical error |",
            "|" + " --- |" * 5,
        ]
        for condition, e in head.items():
            ct = e.get("class_test") or {}
            rows.append(
                "| "
                + " | ".join(
                    [
                        condition,
                        f(e.get("status")),
                        f(e.get("relative_standard_error")),
                        f"{f(ct.get('p_value'))} / {f(ct.get('p_value_lower_tail'))}",
                        f(e.get("evaluator_relative_canonical_error")),
                    ]
                )
                + " |"
            )
        out.append("\n".join(rows))
    ec = load("e3_expand_cluster")
    if ec:
        out.append("\n### E3 Expand-and-Cluster reproduction\n")
        rows = [
            "| Condition | Teachers | Students converged (mean) | Exact recovery | Correct width | Median max cosine distance |",
            "|" + " --- |" * 6,
        ]
        for condition, items in ec.items():
            if not items:
                continue
            groups = {}
            for item in items:
                key = (
                    condition
                    if condition != "string"
                    else f"string ({item['samples_per_input']} samples/input)"
                )
                groups.setdefault(key, []).append(item)
            for key, group in groups.items():
                width_r = [it for it in group if it.get("recovered_width") == (it.get("r") or 3)]
                cos = [
                    it["evaluation"].get("max_cosine_distance")
                    for it in group
                    if it["evaluation"].get("recovered")
                    and it["evaluation"].get("max_cosine_distance") is not None
                ]
                rows.append(
                    "| "
                    + " | ".join(
                        [
                            key,
                            str(len(group)),
                            f(statistics.mean(it["converged_students"] for it in group), 1),
                            f"{sum(it['exact_recovery'] for it in group)}/{len(group)}",
                            f"{len(width_r)}/{len(group)}",
                            f(statistics.median(cos), 4) if cos else "—",
                        ]
                    )
                    + " |"
                )
        out.append("\n".join(rows))
    e4 = load("e4_underdetermined")
    if e4:
        out.append("\n### E4 underdetermined and misspecified cases\n")
        for name, e in e4.items():
            out.append(
                f"* **{name}**: "
                + json.dumps(
                    {
                        k: e.get(k)
                        for k in (
                            "claims",
                            "region_class_test",
                            "evaluator",
                            "hypothesis_limitation",
                            "test_improvement",
                            "refused_or_failed",
                        )
                        if e.get(k) is not None
                    },
                    default=str,
                )[:1500]
            )
    out.append("\n### Acceptance criteria (computed)\n")
    for name, value in criteria.items():
        out.append(f"* {name}: {value}")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
