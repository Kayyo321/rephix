"""Score a protocol-v2 results file against its pre-registered success criteria.

python examples/text_recovery_analysis.py --protocol examples/text_recovery_protocol_v2.json \
    --results results.json --output summary.json

Expectations are read from the protocol, never from the results file, and the
results must carry the protocol's hash, the final split and every registered seed.
"""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

OUT_OF_CLASS_OR_BOUNDARY = (
    "ngram-order3-target",
    "ngram-mlp-target",
    "head-perturbed-backbone",
    "ngram-greedy-target",
    "ngram-undeclared-output",
    "ngram-underdetermined",
)


def score(protocol, protocol_sha256, results):
    cases = {case["name"]: case for case in protocol["cases"]}
    runs = results["runs"]
    by_case = defaultdict(list)
    for run in runs:
        by_case[run["case"]].append(run)
    criteria = {}

    provenance_problems = []
    if results.get("protocol_sha256") != protocol_sha256:
        provenance_problems.append("results were produced with a different protocol file")
    if results.get("split") != "final":
        provenance_problems.append("results are not from the final split")
    for name, case in cases.items():
        ran = sorted(r["seed"] for r in by_case.get(name, []))
        if ran != sorted(case["seeds"]["final"]):
            provenance_problems.append(f"{name}: seeds run {ran} != registered")

    mismatches = []
    for run in runs:
        case = cases.get(run["case"])
        if case is None:
            mismatches.append({"case": run["case"], "problem": "case not registered"})
            continue
        ok = run["status"] in case["allowed_statuses"]
        if ok and run["status"] == "error":
            ok = case.get("error_contains", "") in run.get("error", "")
        if not ok:
            mismatches.append(
                {
                    "case": run["case"],
                    "seed": run["seed"],
                    "status": run["status"],
                    "allowed": case["allowed_statuses"],
                    "error": run.get("error", "")[-200:],
                }
            )
    criteria["S1_status"] = {
        "passed": not mismatches and not provenance_problems,
        "mismatches": mismatches,
        "provenance": provenance_problems,
    }

    rates = {}
    for name, case in cases.items():
        if "min_recovered" in case:
            got = sum(r["status"] == "recovered_in_class" for r in by_case.get(name, []))
            rates[name] = {
                "recovered": got,
                "required": case["min_recovered"],
                "passed": got >= case["min_recovered"],
            }
    criteria["S2_recovery_rate"] = {
        "passed": all(v["passed"] for v in rates.values()),
        "cases": rates,
    }

    measured = [
        r
        for r in runs
        if r["status"] == "recovered_in_class" and "canonical_relative_l2_error" in r
    ]
    worst = max((r["canonical_relative_l2_error"] for r in measured), default=None)
    criteria["S3_accuracy"] = {
        "passed": bool(measured) and worst <= 0.10,
        "runs": len(measured),
        "max_canonical_error": worst,
    }
    ratios = [
        r["canonical_relative_l2_error"] / r["reported_relative_standard_error"]
        for r in measured
        if r.get("reported_relative_standard_error")
    ]
    criteria["S4_self_report"] = {
        "passed": bool(ratios) and len(ratios) == len(measured) and max(ratios) <= 2.5,
        "max_error_to_reported_se": max(ratios, default=None),
    }
    covered = sum(
        r["ci95_coverage"] * r["ci95_coordinates"] for r in measured if "ci95_coordinates" in r
    )
    coordinates = sum(r["ci95_coordinates"] for r in measured if "ci95_coordinates" in r)
    pooled = covered / coordinates if coordinates else None
    criteria["S5_coverage"] = {
        "passed": pooled is not None and 0.90 <= pooled <= 0.99,
        "pooled_coverage": pooled,
        "coordinates": coordinates,
    }
    behaviour = [
        r for r in runs if r["status"] == "recovered_in_class" and r["case"] != "ngram-blind-spot"
    ]
    tv = max((r["test_mean_total_variation"] for r in behaviour), default=None)
    criteria["S6_behaviour"] = {
        "passed": bool(behaviour) and tv <= 0.02,
        "max_heldout_total_variation": tv,
    }
    wins = [
        r["canonical_relative_l2_error"] < r["baseline_canonical_relative_l2_error"]
        for r in measured
    ]
    criteria["S7_baseline"] = {
        "passed": bool(wins) and sum(wins) >= 0.8 * len(wins),
        "mle_better_fraction": sum(wins) / len(wins) if wins else None,
    }
    full = {
        r["seed"]: r for r in by_case.get("ngram-random", []) if "canonical_relative_l2_error" in r
    }
    quad = {
        r["seed"]: r
        for r in by_case.get("ngram-random-4x-budget", [])
        if "canonical_relative_l2_error" in r
    }
    pairs = [
        full[s]["canonical_relative_l2_error"] / quad[s]["canonical_relative_l2_error"]
        for s in sorted(full)
        if s in quad
    ]
    median = statistics.median(pairs) if pairs else None
    criteria["S8_consistency"] = {
        "passed": median is not None and 1.5 <= median <= 2.7,
        "median_ratio": median,
        "ratios": pairs,
    }
    false = [
        (r["case"], r["seed"])
        for r in runs
        if r["case"] in OUT_OF_CLASS_OR_BOUNDARY and r["status"] == "recovered_in_class"
    ]
    criteria["S9_no_false_recovery"] = {"passed": not false, "false_recoveries": false}
    leaks = []
    for r in runs:
        audit = r["audit"]
        if (
            audit["denied_private_opens"]
            or audit["opened_under_private"]
            or not audit["spawned_only_the_target_relay"]
            or audit["chmod_attempts"]
        ):
            leaks.append((r["case"], r["seed"], "recovery"))
        if r["status"] != "error" and (
            r.get("reload_denied_opens")
            or r.get("reload_spawned_processes")
            or r.get("reload_probes_audited", 0) < min(5, r.get("test_contexts", 5))
        ):
            leaks.append((r["case"], r["seed"], "reload"))
    reload = max(
        (
            r["reload_max_abs_logprob_difference"]
            for r in runs
            if "reload_max_abs_logprob_difference" in r
        ),
        default=None,
    )
    criteria["S10_isolation"] = {
        "passed": not leaks and reload is not None and reload <= 1e-9,
        "violations": leaks,
        "max_reload_difference": reload,
    }
    blind = by_case.get("ngram-blind-spot", [])
    blind_ok = [
        r
        for r in blind
        if r["status"] == "recovered_in_class" and r.get("trigger_mean_total_variation", 0) >= 0.3
    ]
    criteria["S11_blind_spot"] = {
        "passed": len(blind_ok) >= 4,
        "demonstrated_runs": len(blind_ok),
        "trigger_tv": [r.get("trigger_mean_total_variation") for r in blind],
    }

    summary_cases = {}
    for name, items in by_case.items():
        errors = [
            r["canonical_relative_l2_error"]
            for r in items
            if r["status"] == "recovered_in_class" and "canonical_relative_l2_error" in r
        ]
        tvs = [r["test_mean_total_variation"] for r in items if "test_mean_total_variation" in r]
        summary_cases[name] = {
            "allowed": cases[name]["allowed_statuses"] if name in cases else None,
            "statuses": [r["status"] for r in items],
            "recovered": sum(r["status"] == "recovered_in_class" for r in items),
            "median_canonical_error_when_recovered": statistics.median(errors) if errors else None,
            "median_heldout_tv": statistics.median(tvs) if tvs else None,
            "sampled_outputs_per_run": items[0].get("budget", {}).get("sampled_outputs"),
            "median_recovery_seconds": statistics.median(r["recovery_seconds"] for r in items),
        }
    return {
        "protocol_sha256": protocol_sha256,
        "results_commit": results.get("git_commit"),
        "split": results.get("split"),
        "runs": len(runs),
        "all_criteria_passed": all(c["passed"] for c in criteria.values()),
        "criteria": criteria,
        "cases": summary_cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    summary = score(
        json.loads(protocol_bytes),
        hashlib.sha256(protocol_bytes).hexdigest(),
        json.loads(args.results.read_text()),
    )
    args.output.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v["passed"] for k, v in summary["criteria"].items()}, indent=1))


if __name__ == "__main__":
    main()
