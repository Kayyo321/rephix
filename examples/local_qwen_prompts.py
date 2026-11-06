"""Freeze synthetic prompt-only observations for the local Qwen experiment.

A family is a reusable task template. Its variants share one group_id and
are assigned to a single split before any model output exists.
"""

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

DEFAULT_SEED = 20261106
WORDS = (
    "amber",
    "beacon",
    "cabin",
    "daisy",
    "ember",
    "falcon",
    "garden",
    "harbor",
    "island",
    "jacket",
    "kettle",
    "lantern",
    "magnet",
    "needle",
    "orange",
    "pencil",
    "quartz",
    "ribbon",
    "saddle",
    "tunnel",
    "velvet",
    "walnut",
    "yarrow",
    "zipper",
)
NAMES = (
    "Ana",
    "Bo",
    "Cleo",
    "Dan",
    "Eli",
    "Fay",
    "Gus",
    "Hana",
    "Ira",
    "Jin",
    "Kai",
    "Luz",
    "Maya",
    "Ned",
    "Ona",
    "Paz",
    "Quinn",
    "Raj",
    "Sol",
    "Tess",
    "Uma",
    "Vic",
    "Wren",
    "Zoe",
)
ANIMALS = (
    "ant",
    "bear",
    "cat",
    "dog",
    "eagle",
    "fox",
    "goat",
    "hare",
    "ibis",
    "jaguar",
    "koala",
    "lemur",
    "mouse",
    "newt",
    "otter",
    "pig",
    "quail",
    "raven",
    "seal",
    "toad",
    "urchin",
    "vole",
    "whale",
    "yak",
)
OBJECTS = (
    "alarm clock",
    "backpack",
    "candle",
    "doorbell",
    "envelope",
    "flashlight",
    "glove",
    "hammer",
    "ice tray",
    "jar",
    "key",
    "lunchbox",
    "mirror",
    "notebook",
    "oven mitt",
    "paintbrush",
    "quilt",
    "ruler",
    "scissors",
    "teapot",
    "umbrella",
    "vase",
    "whistle",
    "yoga mat",
)

# Split membership is explicit. Related variants are never independently split.
SPLITS = {
    "train": (
        24,
        {
            "integer_subtraction": "{a} - {b} =",
            "alphabetical_order": "Sort alphabetically: {third}, {word}, {other}.",
            "imperial_length": "Convert {a} feet to inches.",
            "binary_to_decimal": "What decimal number does binary {bits} represent?",
            "list_index": "What is item {index} in this list, counting from 1? [{word}, {third}, {other}]",
            "title_case": "Rewrite in title case: the {word} at the {other}",
            "kv_to_json": "Convert this record to a JSON object. Keep count numeric. label={word}; count={a}",
            "pronoun_reference": "{name} told {friend}, “I left my keys in the car.” Whose keys are they?",
            "inference_from_rule": "Each {animal} in the shelter has a red tag. One animal has a blue tag. Can it be a {animal}?",
            "explain_tool": "In one sentence, explain what a {object} is used for.",
            "soften_request": "Rewrite this as a polite request: Bring me the {object} right now.",
            "product_tagline": "Invent a short tagline for a store that sells {object_plural}.",
            "warning_sign": "Write a four-to-seven-word warning sign for a room with {a} loose {animal_plural}.",
            "scene_opening": "Start a mystery story with one sentence about a {object} that vanished overnight.",
            "literal_translation": "Translate into Spanish: {name} has a book.",
            "python_literal_type": "What type does Python assign to the literal '{a}'?",
        },
    ),
    "validation": (
        12,
        {
            "fraction_reduce": "Reduce {numerator}/{denominator} to lowest terms.",
            "word_middle": "What is the third letter of {word}?",
            "temporal_compare": "{name} arrives at {hour}:05 and {friend} arrives at {later}:05 on the same morning. Who arrives first?",
            "csv_header": "Write one comma-separated header line for columns named {word}, {other}, and {third}.",
            "visual_simile": "Write a one-sentence simile describing a {animal} without using its name.",
            "message_confirmation": "Write a brief text confirming you can meet {name} at {hour}:30 tomorrow morning.",
        },
    ),
    "test": (
        12,
        {
            "rectangular_area": "A rectangle is {a} cm long and {b} cm wide. Find its area.",
            "acronym": "Give the initials of these three words in order: {word} {other} {third}.",
            "explicit_fact_extract": "{name} owns a {animal} called {friend} and a bicycle. What is the pet's name?",
            "markdown_link": "Write a Markdown link with the text {word} and URL https://example.com/{other}.",
            "interview_question": "Write one open-ended interview question for someone who designs {object_plural}.",
            "name_fictional_place": "Invent a name for a seaside village famous for its {animal_plural}.",
        },
    ),
    "pool": (
        6,
        {
            "parity": "Is {a} odd or even?",
            "sum_of_digits": "Add the digits of {a}{b}.",
            "numeric_rounding": "Round {a}.{decimal} to the nearest whole number.",
            "equal_distribution": "Share {total} tokens equally among {b} people. How many does each person get?",
            "average_speed": "A bicycle covers {total} km in {b} hours. What is its average speed in km/h?",
            "weekday_shift": "Today is {weekday}. What day of the week will it be in {b} days?",
            "string_reverse": "Reverse these characters: {word}",
            "word_length_compare": "Which word has more characters, '{word}' or '{shorter}'?",
            "repeat_with_delimiter": "Write {word} three times, separated by /.",
            "lowercase_normalization": "Convert to lowercase: {upper} {friend}",
            "conditional_program_trace": "In Python, what is printed by print('yes' if {a} > {threshold} else 'no')?",
            "spreadsheet_sum": "Write a spreadsheet formula that sums the range {column}2:{column}{a}.",
            "semantic_category": "Does a {animal} belong to the plant or animal kingdom?",
            "set_intersection": "Which word is in both sets? {{{word}, {other}}} and {{{third}, {other}}}",
            "possessive_rewrite": "Rewrite using an apostrophe for possession: the {object} belonging to {name}.",
            "define_job": "In plain language, what does someone who repairs {object_plural} do?",
            "constraint_question": "Write a yes-or-no question about a {animal}.",
            "email_subject": "Give a short email subject line for asking {name} about a missing {object}.",
            "micro_poem": "Write a two-line poem featuring a {animal} and a {object}.",
            "reason_from_absence": "{name} only waters the {word} plant on weekends. Today is {weekday}. Does the rule allow watering today?",
        },
    ),
}


def plural(noun):
    """Pluralize the small noun inventory used by the templates."""
    if noun == "mouse":
        return "mice"
    if noun == "scissors":
        return noun
    if noun.endswith(("ch", "sh", "s", "x")):
        return noun + "es"
    return noun + "s"


def fields(n):
    word = WORDS[n]
    object_name = OBJECTS[n]
    animal = ANIMALS[n]
    a = n + 14
    b = 2 + n % 9
    return {
        "a": a,
        "b": b,
        "bits": format(a, "b"),
        "decimal": n % 10,
        "total": a * b,
        "threshold": 26,
        "numerator": 2 * a,
        "denominator": 4 * a + 2 * b,
        "index": 1 + n % 3,
        "hour": 1 + n % 10,
        "later": 2 + n % 10,
        "weekday": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")[
            n % 7
        ],
        "column": chr(ord("A") + n),
        "name": NAMES[n],
        "friend": NAMES[(n + 5) % len(NAMES)],
        "word": word,
        "upper": word.upper(),
        "other": WORDS[(n + 7) % len(WORDS)],
        "third": WORDS[(n + 13) % len(WORDS)],
        "shorter": ("a", "an", "any")[n % 3],
        "object": object_name,
        "object_plural": plural(object_name),
        "animal": animal,
        "animal_plural": plural(animal),
    }


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_line(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n"


def build(seed):
    seen_ids = set()
    seen_groups = set()
    seen_prompts = set()
    manifest = {"version": 1, "seed": seed, "splits": {}}
    files = {}
    for split, (size, families) in SPLITS.items():
        records = []
        for family, template in families.items():
            if family in seen_groups:
                raise ValueError(f"Family occurs in multiple splits: {family}")
            seen_groups.add(family)
            variants = sorted(
                range(len(WORDS)),
                key=lambda n: digest(f"{seed}/{family}/{n}".encode()),
            )
            if size > len(variants):
                raise ValueError(f"Too few candidate variants for {family}")
            for n in variants[:size]:
                prompt = template.format(**fields(n))
                normalized = " ".join(unicodedata.normalize("NFC", prompt).split())
                record_id = f"{family}-{n:02d}"
                if record_id in seen_ids or normalized in seen_prompts:
                    raise ValueError(f"Duplicate id or normalized prompt: {record_id}")
                seen_ids.add(record_id)
                seen_prompts.add(normalized)
                records.append(
                    {
                        "id": record_id,
                        "prompt": prompt,
                        "output": "",
                        "metadata": {"group_id": family},
                    }
                )
        filename = f"{split}.jsonl"
        files[filename] = "".join(map(json_line, records)).encode("utf-8")
        manifest["splits"][split] = {
            "file": filename,
            "groups": sorted(families),
            "rows": len(records),
            "sha256": digest(files[filename]),
        }
    files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory for the frozen corpus"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    files = build(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, data in files.items():
        (args.output / filename).write_bytes(data)


if __name__ == "__main__":
    main()
