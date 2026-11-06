"""Freeze prompt families and budget-matched settings for reconstruction sessions."""

from __future__ import annotations

import argparse
import json
import random
from itertools import product
from pathlib import Path

from rephix.io import write_json
from rephix.session import _config

PLAN_SEED = 20261106
PROBE_FILE = Path(__file__).with_name("identification-probes.json")
TEXT = {
    "pool": {
        "repair": "In two full sentences, explain how to troubleshoot a loose {object}.",
        "school": "Write two sentences of a school announcement about a visitor named {name}.",
        "fable": "Write the first two sentences of a fable about a {animal} on a hill.",
        "history": "Explain in two sentences why people in the past needed a {object}.",
        "travel": "Write a two-sentence invitation to visit {place} in autumn.",
        "diet": "Write two sentences about a lunch a person could make with {food}.",
        "boundary": "Write a polite two-sentence refusal after {name} asks to borrow your {object}.",
        "comparison": "Give two sentences comparing the habits of a {animal} and a house cat.",
        "translation": "Translate to Spanish: I lent {name} a lantern. We will collect it after sunset.",
        "descriptive": "Describe the sounds of {place} in two sentences as daylight breaks.",
        "code": "In two sentences, explain what this Python line returns: [x*x for x in range({n})].",
        "sequence": "Write two sentences explaining the pattern {n}, {m}, {p}, {q}.",
    },
    "validation": {
        "museum": "Write a two-sentence museum label introducing an old {object}.",
        "radio": "Write two sentences of a radio news update about snowfall in {place}.",
        "argument": "Write two sentences supporting {name}'s decision to adopt a {animal}.",
        "factcheck": "Explain in two sentences why the statement 'all {food} tastes the same' can be misleading.",
        "conversion": "Explain in two sentences how to convert {n} hours and 17 minutes into minutes.",
    },
    "test": {
        "apology": "Write a brief apology note to {name} for arriving late, using two complete sentences.",
        "invent": "Describe a future invention made out of a {object}, in two sentences.",
        "ecology": "Give two sentences about how a {animal} might adapt to winters in {place}.",
        "review": "Write a two-sentence review of a restaurant where the main ingredient is {food}.",
        "reading": "Write two sentences explaining why the title 'The Last {object}' might suit a mystery novel.",
    },
}
TEXT_FIELDS = (
    ("Mira", "flashlight", "fox", "Oslo", "lentils", 9),
    ("Jules", "doorbell", "otter", "Dakar", "rice", 14),
    ("Anika", "toaster", "tortoise", "Kyoto", "squash", 21),
    ("Mateo", "bicycle", "heron", "Lima", "oats", 27),
    ("Nora", "teapot", "squirrel", "Toulouse", "mushrooms", 35),
)
TOOLS = {
    "pool": (
        "Use sandbox tool {first} with a={a} and b={b}. Then apply {second} to that result. Return the final value alone.",
        "With operands {a} and {b}, run {first} first and {second} next. Give only what the second tool returns.",
        "Please execute {first} on ({a}, {b}), pass the output into {second}, and reply with the resulting value.",
        "Run this two-tool pipeline: ({a}, {b}) -> {first} -> {second}. Output the final result only.",
        "Start with {first} for {a} and {b}. The result must go into {second}. Respond with just the result.",
        "Apply the named tools in this order: {first}(a={a}, b={b}), then {second}(previous value). Answer with the final value.",
    ),
    "validation": (
        "I need a sandbox calculation using a={a}, b={b}. Invoke {first} and then {second} on its return value. Reply with that value.",
        "The second tool is {second}; its input comes from {first} with operands {a} and {b}. Use both tools, then report just the answer.",
        "Use the output of {first} on a={a} and b={b} as the argument to {second}. What value comes out?",
    ),
    "test": (
        "Report only the output from {second} after you feed it the result of {first} on ({a}, {b}). Use both sandbox tools.",
        "Perform {first} with a={a} and b={b}. Treat its return as the next input and invoke {second}. What is the final value?",
        "Process this instruction with the available tools. Inputs are {a} and {b}; first operation is {first}; next is {second}. Give the final result alone.",
    ),
}
OPERANDS = {
    "pool": (range(1, 10), range(1, 10)),
    "validation": (range(11, 20), range(1, 10)),
    "test": (range(1, 10), range(11, 20)),
}
ACTION_PAIRS = tuple(product(("add", "multiply", "subtract"), ("double", "negate", "format")))


def text_rows(seed):
    rng = random.Random(seed)
    for partition, families in TEXT.items():
        count = 5 if partition == "pool" else 3
        for family, template in families.items():
            fields = rng.sample(TEXT_FIELDS, count)
            for index, (name, item, animal, place, food, n) in enumerate(fields, 1):
                yield {
                    "id": f"text-{partition}-{family}-{index:02}",
                    "group": f"text-{family}",
                    "partition": partition,
                    "prompt": template.format(
                        name=name,
                        object=item,
                        animal=animal,
                        place=place,
                        food=food,
                        n=n,
                        m=n + 3,
                        p=n + 6,
                        q=n + 9,
                    ),
                }


def tool_rows(seed):
    from rephix.session_targets import sandbox_execute

    rng = random.Random(seed)
    for partition, templates in TOOLS.items():
        count = 10 if partition == "pool" else 4
        size = count * len(templates)
        operands = rng.sample(list(product(*OPERANDS[partition])), size)
        actions = list(ACTION_PAIRS) * ((size + len(ACTION_PAIRS) - 1) // len(ACTION_PAIRS))
        actions = actions[:size]
        rng.shuffle(actions)
        for index, ((a, b), (first, second)) in enumerate(zip(operands, actions, strict=True)):
            family, variant = divmod(index, count)
            arguments = {"a": a, "b": b}
            previous = sandbox_execute(first, arguments)
            next_arguments = {"value": previous}
            yield {
                "id": f"tool-{partition}-{family + 1:02}-{variant + 1:02}",
                "group": f"tool-{partition}-style-{family + 1:02}",
                "partition": partition,
                "prompt": templates[family].format(a=a, b=b, first=first, second=second),
                "task": arguments,
                "expect": {
                    "tools": [
                        {"name": first, "arguments": arguments},
                        {"name": second, "arguments": next_arguments},
                    ],
                    "result": sandbox_execute(second, next_arguments),
                },
            }


def session_spec(args):
    rows = list(text_rows(args.plan_seed) if args.mode == "text" else tool_rows(args.plan_seed))
    profile = None
    if args.identity_profile:
        profile = json.loads(args.identity_profile.read_text(encoding="utf-8"))
        probes = json.loads(PROBE_FILE.read_text(encoding="utf-8"))
        if profile.get("probes") != probes:
            raise ValueError("identity profile probes differ from the frozen probe panel")
        rows.extend(
            {
                "id": f"probe-{key}",
                "group": f"identity-probe-{key}",
                "partition": "pool",
                "prompt": prompt,
                "probe_id": key,
            }
            for key, prompt in probes.items()
        )
    tool_mode = args.mode == "tool"
    options = {
        "width": args.width if args.width is not None else (32 if tool_mode else 64),
        "context": args.context if args.context is not None else (1024 if tool_mode else 128),
        "batch": args.batch if args.batch is not None else (16 if tool_mode else 64),
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": args.device,
    }
    prior = {
        "plan": "synthetic held-out-family text and sandbox tasks v1",
        "plan_seed": args.plan_seed,
        "model_identity": "withheld by the target integration",
    }
    if tool_mode:
        prior["sandbox"] = "bounded_integer_tools_v1 tool names, candidate options and execution"
        prior["expected_tool_tasks"] = "available to the evaluator for reserved task scoring"
    if profile:
        prior["identity_fingerprints"] = "empirical references for the optional probe panel"
    spec = {
        "schema_version": 1,
        "prompts": rows,
        "candidates": {
            f"byte-{architecture}": options | {"architecture": architecture}
            for architecture in args.architecture or ["gru"]
        },
        "round": {
            "acquire": args.acquire,
            "updates": args.updates,
            "byte_cap": args.byte_cap if args.byte_cap is not None else (64 if tool_mode else 24),
            "policy": args.policy,
            "replay": args.replay,
            "seed": args.seed,
        },
        "identity_profile": profile,
        "priors": prior,
        "scope": "agent_combined" if tool_mode else "model_observable",
    }
    _config(spec)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("text", "tool"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-profile", type=Path)
    parser.add_argument("--policy", choices=("active", "random"), default="active")
    parser.add_argument("--replay", choices=("cumulative", "recent"), default="cumulative")
    parser.add_argument("--plan-seed", type=int, default=PLAN_SEED)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--acquire", type=int, default=12)
    parser.add_argument("--updates", type=int, default=96)
    parser.add_argument("--byte-cap", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--architecture", choices=("gru", "conv"), action="append")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    write_json(args.output, session_spec(args))


if __name__ == "__main__":
    main()
