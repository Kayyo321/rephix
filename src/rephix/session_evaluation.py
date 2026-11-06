"""Evaluate observable text and controlled closed-loop tool behavior separately."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from .session_targets import model_context, sandbox_rollout, tool_action, training_segments


def examples_for_observation(observation):
    """Create aligned UTF-8 examples without teaching an artificial EOS at a cap."""
    if observation["status"] != "ok":
        return []
    segments = training_segments(observation)
    return [
        {
            "input": context,
            "output": output,
            "truncated": (
                index == len(segments) - 1 and observation["usage"].get("finish_reason") == "length"
            ),
        }
        for index, (context, output) in enumerate(segments)
    ]


def _trace(events):
    return [
        {"name": event["name"], "arguments": event["arguments"]}
        for event in events
        if event["kind"] == "tool_call" and event["source"] == "model"
    ]


def _tool_results(events):
    return [event["result"] for event in events if event["kind"] == "tool_result"]


def _result_text(value):
    return unicodedata.normalize("NFC", str(value)).strip().casefold()


def _expected_actions(actual, expected):
    if not isinstance(expected, list):
        raise ValueError("expect.tools must be an array of tool names and optional arguments")
    for target in expected:
        if (
            not isinstance(target, dict)
            or not isinstance(target.get("name"), str)
            or not target["name"]
            or ("arguments" in target and not isinstance(target["arguments"], (dict, str)))
        ):
            raise ValueError("expected tool entries need a name and optional arguments")
    return len(actual) == len(expected) and all(
        action["name"] == target["name"]
        and ("arguments" not in target or tool_action(action) == tool_action(target))
        for action, target in zip(actual, expected)
    )


def _task_outcome(expectation, completion, actions, results):
    if not expectation:
        return None
    conditions = []
    report = {}
    if "completion" in expectation:
        exact = completion == expectation["completion"]
        report["expected_completion_exact"] = exact
        conditions.append(exact)
    if "result" in expectation:
        expected = _result_text(expectation["result"])
        matches = _result_text(completion) == expected
        report["expected_result_in_final"] = matches
        conditions.append(matches)
        if results:
            report["expected_last_tool_result"] = _result_text(results[-1]) == expected
    if "tools" in expectation:
        matches = _expected_actions(actions, expectation["tools"])
        report["expected_tool_trace_exact"] = matches
        conditions.append(matches)
    if conditions:
        report["success"] = all(conditions)
    return report


def _rate(values):
    return {
        "matches": sum(values),
        "compared": len(values),
        "rate": sum(values) / len(values) if values else None,
    }


def _prefix_bytes(reference, predicted):
    correct = 0
    for expected, actual in zip(reference.encode(), predicted.encode()):
        if expected != actual:
            break
        correct += 1
    return correct


def _rollout(candidate, request, byte_cap):
    def choose(events, options):
        return candidate.choose(model_context(events), [tool_action(option) for option in options])

    def complete(events):
        return candidate.generate([model_context(events)], byte_cap)[0]

    return sandbox_rollout(request, choose, complete)


def _controlled_request(row):
    return {
        "session_id": "local-candidate-evaluation",
        "request_id": row["id"],
        "messages": [{"role": "user", "content": row["prompt"]}],
        "task": row["task"],
    }


def _plain_context(observation):
    events = observation["events"]
    for index, event in enumerate(events):
        if event["kind"] == "message" and event["source"] == "model":
            return model_context(events[:index])
    raise ValueError("plain text evaluation requires a model message")


def evaluate(candidate, values, byte_cap):
    """Compare independent generation and, where available, full sandbox rollouts."""
    values = list(values)
    examples = [
        example for _row, observation in values for example in examples_for_observation(observation)
    ]
    teacher_forced = candidate.teacher_forced(examples, byte_cap)
    rows = []
    plain = []
    for row, observation in values:
        report = {
            "prompt_id": row["id"],
            "target_status": observation["status"],
            "target_completion": observation["completion"],
            "target_finish_reason": observation["usage"].get("finish_reason"),
        }
        rows.append(report)
        if observation["status"] != "ok":
            report["evaluation_mode"] = "target_error"
            continue
        target_actions = _trace(observation["events"])
        if (
            observation["capabilities"].get("sandbox") == "bounded_integer_tools_v1"
            and "task" in row
        ):
            candidate_events, output = _rollout(candidate, _controlled_request(row), byte_cap)
            actions = _trace(candidate_events)
            results = _tool_results(candidate_events)
            target_results = _tool_results(observation["events"])
            names = [
                index < len(actions) and actions[index]["name"] == target["name"]
                for index, target in enumerate(target_actions)
            ]
            args = [
                index < len(actions)
                and json.dumps(actions[index]["arguments"], sort_keys=True)
                == json.dumps(target["arguments"], sort_keys=True)
                for index, target in enumerate(target_actions)
            ]
            trace_exact = _expected_actions(actions, target_actions)
            results_exact = results == target_results
            report.update(
                {
                    "evaluation_mode": "closed_loop_sandbox",
                    "candidate_completion": output,
                    "target_actions": target_actions,
                    "candidate_actions": actions,
                    "target_tool_results": target_results,
                    "candidate_tool_results": results,
                    "tool_name_matches": names,
                    "tool_argument_matches": args,
                    "action_trace_exact": trace_exact,
                    "tool_results_exact": results_exact,
                    "multi_step_tool_exact": trace_exact and results_exact,
                    "candidate_task": _task_outcome(row.get("expect"), output, actions, results),
                    "target_task": _task_outcome(
                        row.get("expect"), observation["completion"], target_actions, target_results
                    ),
                }
            )
        elif target_actions or _tool_results(observation["events"]):
            report["evaluation_mode"] = "recorded_context_only"
        else:
            report["evaluation_mode"] = "free_running_text"
            plain.append((row, observation, report))
    if plain:
        predictions = candidate.generate([_plain_context(obs) for _, obs, _ in plain], byte_cap)
        for (row, observation, report), output in zip(plain, predictions, strict=True):
            report["candidate_completion"] = output
            if row.get("expect") and "tools" not in row["expect"]:
                report["candidate_task"] = _task_outcome(row["expect"], output, [], [])
                report["target_task"] = _task_outcome(
                    row["expect"], observation["completion"], [], []
                )
    for report in rows:
        if "candidate_completion" in report:
            expected, actual = report["target_completion"], report["candidate_completion"]
            expected_bytes, actual_bytes = expected.encode(), actual.encode()
            report["completion_exact"] = expected == actual
            report["matched_prefix_bytes"] = _prefix_bytes(expected, actual)
            report["target_completion_bytes"] = len(expected_bytes)
            if report["target_finish_reason"] == "length" and len(expected_bytes) < byte_cap:
                if expected_bytes:
                    report["censored_target_visible_prefix"] = actual_bytes.startswith(
                        expected_bytes
                    )
            else:
                report["completion_exact_at_common_byte_cap"] = (
                    expected_bytes[:byte_cap] == actual_bytes[:byte_cap]
                )
    closed_loop = [row for row in rows if row["evaluation_mode"] == "closed_loop_sandbox"]
    return {
        "teacher_forced": teacher_forced,
        "free_running": {
            "generation_byte_cap": byte_cap,
            "completion_exact": _rate(
                [row["completion_exact"] for row in rows if "completion_exact" in row]
            ),
            "completion_exact_at_common_byte_cap": _rate(
                [
                    row["completion_exact_at_common_byte_cap"]
                    for row in rows
                    if "completion_exact_at_common_byte_cap" in row
                ]
            ),
            "censored_target_visible_prefix": _rate(
                [
                    row["censored_target_visible_prefix"]
                    for row in rows
                    if "censored_target_visible_prefix" in row
                ]
            ),
            "matched_prefix_bytes": sum(row.get("matched_prefix_bytes", 0) for row in rows),
            "target_completion_bytes": sum(row.get("target_completion_bytes", 0) for row in rows),
            "recorded_context_only": sum(
                row["evaluation_mode"] == "recorded_context_only" for row in rows
            ),
            "target_errors": sum(row["target_status"] != "ok" for row in rows),
        },
        "tools": {
            "selection": _rate(
                [match for row in closed_loop for match in row["tool_name_matches"]]
            ),
            "arguments": _rate(
                [match for row in closed_loop for match in row["tool_argument_matches"]]
            ),
            "action_trace_exact": _rate([row["action_trace_exact"] for row in closed_loop]),
            "tool_results_exact": _rate([row["tool_results_exact"] for row in closed_loop]),
            "multi_step_tool_exact": _rate([row["multi_step_tool_exact"] for row in closed_loop]),
        },
        "task_success": {
            actor: _rate(
                [
                    row[f"{actor}_task"]["success"]
                    for row in rows
                    if row.get(f"{actor}_task") and "success" in row[f"{actor}_task"]
                ]
            )
            for actor in ("candidate", "target")
        },
        "failed_prompt_ids": [
            row["prompt_id"]
            for row in rows
            if row.get("completion_exact") is False
            or row.get("multi_step_tool_exact") is False
            or (row.get("candidate_task") or {}).get("success") is False
        ],
        "per_prompt": rows,
    }


def run_export_agent(directory: str | Path, request: dict, byte_cap: int = 64, device="cpu"):
    """Run a trained standalone candidate with just the shared pure sandbox."""
    from .session_candidate import load_export

    candidate = load_export(Path(directory), device=device)
    events, completion = _rollout(candidate, request, byte_cap)
    return {
        "completion": completion,
        "events": events,
        "target_access": False,
        "environment": "bounded_integer_tools_v1",
        "action_constraint": "candidate scores each supplied allowed tool and argument choice",
    }
