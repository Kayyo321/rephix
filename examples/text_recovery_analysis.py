"""Score a text-recovery results file against the pre-registered success criteria.

python examples/text_recovery_analysis.py --protocol examples/text_recovery_protocol.json \
    --results results.json --output summary.json
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    registered = {case["name"]: case["expected_status"] for case in protocol["cases"]}
    results = json.loads(args.results.read_text())
    runs = results["runs"]
    recovered = [r for r in runs if r["status"] == "recovered_in_class"]
    by_case = defaultdict(list)
    for run in runs:
        by_case[run["case"]].append(run)

    criteria = {}
    mismatched = [
        {"case": r["case"], "seed": r["seed"], "expected": r["expected"], "status": r["status"]}
        for r in runs
        if not r["status_matches_expectation"]
    ]
    missing = sorted(set(registered) - {r["case"] for r in runs})
    criteria["S1_status"] = {
        "passed": not mismatched and not missing,
        "mismatches": mismatched,
        "cases_not_run": missing,
    }
    worst = max((r["canonical_relative_l2_error"] for r in recovered), default=None)
    criteria["S2_accuracy"] = {
        "passed": worst is not None and worst <= 0.10,
        "max_error_among_recovered": worst,
    }
    ratios = [
        r["canonical_relative_l2_error"] / r["reported_relative_standard_error"] for r in recovered
    ]
    criteria["S3_self_report"] = {
        "passed": bool(ratios) and max(ratios) <= 2.5,
        "max_error_to_reported_se": max(ratios, default=None),
    }
    coverage = statistics.fmean(r["ci95_coverage"] for r in recovered if "ci95_coverage" in r)
    criteria["S4_coverage"] = {"passed": 0.90 <= coverage <= 0.99, "pooled_coverage": coverage}
    tv = max((r["test_mean_total_variation"] for r in recovered), default=None)
    criteria["S5_behaviour"] = {
        "passed": tv is not None and tv <= 0.02,
        "max_heldout_total_variation": tv,
    }
    wins = [
        r["canonical_relative_l2_error"] < r["baseline_canonical_relative_l2_error"]
        for r in recovered
    ]
    criteria["S6_baseline"] = {
        "passed": sum(wins) >= 0.8 * len(wins),
        "mle_better_fraction": sum(wins) / len(wins) if wins else None,
    }
    full = {r["seed"]: r for r in by_case["ngram-random"]}
    quad = {r["seed"]: r for r in by_case["ngram-random-4x-budget"]}
    pairs = [
        full[s]["canonical_relative_l2_error"] / quad[s]["canonical_relative_l2_error"]
        for s in full
        if s in quad and "canonical_relative_l2_error" in quad[s]
    ]
    median_ratio = statistics.median(pairs) if pairs else None
    criteria["S7_consistency"] = {
        "passed": median_ratio is not None and 1.5 <= median_ratio <= 2.7,
        "median_error_ratio_full_to_4x": median_ratio,
        "ratios": pairs,
    }
    boundary = [
        r
        for r in runs
        if r["expected"] != "recovered_in_class"
        and r["case"] not in ("ngram-random-low-budget", "ngram-peaked", "ngram-corpus")
    ]
    false_recovery = [
        (r["case"], r["seed"]) for r in boundary if r["status"] == "recovered_in_class"
    ]
    criteria["S8_boundary"] = {"passed": not false_recovery, "false_recoveries": false_recovery}
    leaks = [
        (r["case"], r["seed"])
        for r in runs
        if r["audit"]["denied_private_opens"]
        or r["audit"]["opened_under_private"]
        or r.get("reload_denied_opens")
    ]
    reload = max(
        (
            r.get("reload_max_abs_logprob_difference", 0.0)
            for r in runs
            if "reload_max_abs_logprob_difference" in r
        ),
        default=None,
    )
    criteria["S9_isolation"] = {
        "passed": not leaks and reload is not None and reload <= 1e-9,
        "boundary_violations": leaks,
        "max_reload_difference": reload,
    }

    cases = {}
    for name, items in by_case.items():
        errors = [
            r["canonical_relative_l2_error"]
            for r in items
            if "canonical_relative_l2_error" in r and r["status"] == "recovered_in_class"
        ]
        cases[name] = {
            "expected": items[0]["expected"],
            "statuses": [r["status"] for r in items],
            "recovered": sum(r["status"] == "recovered_in_class" for r in items),
            "runs": len(items),
            "median_canonical_error_when_recovered": statistics.median(errors) if errors else None,
            "median_heldout_tv": statistics.median(
                r["test_mean_total_variation"] for r in items if "test_mean_total_variation" in r
            )
            if any("test_mean_total_variation" in r for r in items)
            else None,
            "sampled_outputs_per_run": items[0].get("budget", {}).get("sampled_outputs"),
            "median_recovery_seconds": statistics.median(r["recovery_seconds"] for r in items),
        }
    summary = {
        "protocol_sha256": results["protocol_sha256"],
        "results_commit": results["git_commit"],
        "split": results["split"],
        "runs": len(runs),
        "all_criteria_passed": all(c["passed"] for c in criteria.values()),
        "criteria": criteria,
        "cases": cases,
    }
    args.output.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v["passed"] for k, v in criteria.items()}, indent=1))


if __name__ == "__main__":
    main()
