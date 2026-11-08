"""Stage 6: held-out measurements and separate claim statuses.

Teacher-forced likelihood, free-running generation, memorization overlap, retention,
tool behavior and visible reasoning are measured separately. Claim statuses come only
from the rules in docs/CREATE_MODE_METHOD.md; process completion is a different field.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from .constraints import FAMILIES, features, record_features, style_distance
from .data import FACT_CLOZE, NEW_TOPICS
from .modeling import generate, group_nll, paired_bootstrap, per_example_nll
from .serialize import REASONING_CLOSE, REASONING_OPEN, parse_tool_calls

_WORD = re.compile(r"\w+")


def variant(ctx, edits, label):
    """(model, context manager) computing the base or the created model."""
    if label == "created":
        return ctx.model, _null()
    if edits is not None:
        return ctx.model, edits.off()
    if ctx.base_model is None:
        raise RuntimeError("a separate base model is required for full fine-tuning runs")
    return ctx.base_model, _null()


def nll_comparison(ctx, examples, group_of, *, edits=None, seed=0):
    """Created-vs-base NLL per token on held-out examples, bootstrapped over groups."""
    torch = ctx.torch
    created, counts = per_example_nll(
        torch, ctx.model, examples, ctx.tokenizer, ctx.device, autocast=ctx.autocast
    )
    model, context = variant(ctx, edits, "base")
    with context:
        base, _ = per_example_nll(
            torch, model, examples, ctx.tokenizer, ctx.device, autocast=ctx.autocast
        )
    keys, created_g, count_g = group_nll(examples, created, counts, group_of)
    _, base_g, _ = group_nll(examples, base, counts, group_of)
    improvement = paired_bootstrap(base_g - created_g, count_g, seed=seed)
    tokens = int(counts.sum())
    return {
        "tokens": tokens,
        "groups": len(keys),
        "base_nll": float(base.sum() / max(tokens, 1)),
        "created_nll": float(created.sum() / max(tokens, 1)),
        "improvement_nats_per_token": improvement,
        "base_perplexity": float(np.exp(base.sum() / max(tokens, 1))),
        "created_perplexity": float(np.exp(created.sum() / max(tokens, 1))),
    }


def restoration(ctx, edits, examples, group_of):
    """Restore selected regions to the base weights and measure how much gain disappears."""
    torch = ctx.torch
    created, counts = per_example_nll(
        torch, ctx.model, examples, ctx.tokenizer, ctx.device, autocast=ctx.autocast
    )
    with edits.off():
        base, _ = per_example_nll(
            torch, ctx.model, examples, ctx.tokenizer, ctx.device, autocast=ctx.autocast
        )
    tokens = max(int(counts.sum()), 1)
    gain = (base.sum() - created.sum()) / tokens
    rows = []
    for region in edits.regions:
        with edits.off([region]):
            partial, _ = per_example_nll(
                torch, ctx.model, examples, ctx.tokenizer, ctx.device, autocast=ctx.autocast
            )
        rows.append(
            {
                "region": region,
                "gain_removed_nats_per_token": float((partial.sum() - created.sum()) / tokens),
            }
        )
    return {
        "total_gain_nats_per_token": float(gain),
        "restore_all_removes_fraction": 1.0 if gain != 0 else None,
        "per_region": rows,
        "note": "Restoring every edited region returns exactly the base weights, so the "
        "whole gain is removed by construction; per-region rows show how the gain is "
        "distributed (single-region removal can exceed or fall short of a share because "
        "regions interact).",
    }


# ----------------------------------------------------------------------------- overlap


def _ngrams(words, n):
    return {
        hashlib.blake2b(" ".join(words[i : i + n]).encode(), digest_size=8).digest()
        for i in range(len(words) - n + 1)
    }


class OverlapIndex:
    """Word 8-gram index of training text to measure verbatim reuse in generations."""

    def __init__(self, texts, n=8):
        self.n = n
        self.grams = set()
        for text in texts:
            self.grams |= _ngrams([w.lower() for w in _WORD.findall(text)], n)

    def measure(self, text):
        words = [w.lower() for w in _WORD.findall(text)]
        if len(words) < self.n:
            return {"ngram_overlap": 0.0, "longest_copied_words": 0}
        hits = []
        for i in range(len(words) - self.n + 1):
            key = hashlib.blake2b(" ".join(words[i : i + self.n]).encode(), digest_size=8).digest()
            hits.append(key in self.grams)
        longest = current = 0
        for hit in hits:
            current = current + 1 if hit else 0
            longest = max(longest, current)
        return {
            "ngram_overlap": float(np.mean(hits)),
            "longest_copied_words": int(longest + self.n - 1) if longest else 0,
        }


def repetition(text):
    words = [w.lower() for w in _WORD.findall(text)]
    grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def _split_reasoning(text):
    start = text.find(REASONING_OPEN.strip())
    if start < 0:
        return None, text
    end = text.find(REASONING_CLOSE.strip(), start)
    if end < 0:
        return text[start + len(REASONING_OPEN.strip()) :].strip(), ""
    return (
        text[start + len(REASONING_OPEN.strip()) : end].strip(),
        (text[:start] + text[end + len(REASONING_CLOSE.strip()) :]).strip(),
    )


def token_f1(a, b):
    left, right = _WORD.findall((a or "").lower()), _WORD.findall((b or "").lower())
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    common = 0
    pool = {}
    for word in right:
        pool[word] = pool.get(word, 0) + 1
    for word in left:
        if pool.get(word, 0) > 0:
            common += 1
            pool[word] -= 1
    if common == 0:
        return 0.0
    precision, recall = common / len(left), common / len(right)
    return 2 * precision * recall / (precision + recall)


def tool_schema(records):
    """Tool names and argument keys observed in training evidence (the inferred schema)."""
    schema = {}
    for record in records:
        for message in record.assistant_messages():
            for call in message.tool_calls or []:
                schema.setdefault(call["name"], set()).update(call["arguments"])
    return {name: sorted(keys) for name, keys in schema.items()}


def score_tools(generated_text, observed_calls, schema):
    calls, malformed = parse_tool_calls(generated_text)
    valid = [
        call
        for call in calls
        if call["name"] in schema and set(call["arguments"]) <= set(schema[call["name"]])
    ]
    expected_names = [c["name"] for c in observed_calls]
    produced_names = [c["name"] for c in calls]
    return {
        "expected_calls": len(observed_calls),
        "produced_calls": len(calls),
        "malformed_blocks": malformed,
        "schema_valid_fraction": (len(valid) / len(calls)) if calls else None,
        "selection_correct": produced_names[:1] == expected_names[:1]
        if expected_names
        else not calls,
        "sequence_exact": produced_names == expected_names,
        "arguments_exact": bool(
            calls
            and observed_calls
            and calls[0]["name"] == observed_calls[0]["name"]
            and calls[0]["arguments"] == observed_calls[0]["arguments"]
        ),
    }


def free_running(
    ctx,
    edits,
    records,
    *,
    train_records,
    supported_features,
    count,
    max_new_tokens,
    temperature,
    seed,
    new_topics=True,
):
    """Generate from held-out contexts (and new topics) with base and created weights."""
    serializer = ctx.serializer
    usable = []
    for record in records:
        prefix = serializer.generation_prefix(record)
        if record.kind == "passage":
            # Start from the first words of the passage when its own context is unknown.
            head = ctx.tokenizer(record.text, add_special_tokens=False)["input_ids"][:16]
            prefix = (prefix or []) + head
        if prefix:
            usable.append((record, prefix))
    usable = usable[:count]
    prefixes = [prefix for _, prefix in usable]
    topic_prefixes = []
    if new_topics and all(r.kind == "passage" for r in records):
        topic_prefixes = [serializer.prompt_text_ids(t + "\n\n") for t in NEW_TOPICS]
    outputs = {}
    for label in ("base", "created"):
        model, context = variant(ctx, edits, label)
        with context:
            held = generate(
                ctx.torch,
                model,
                ctx.tokenizer,
                prefixes,
                max_new_tokens=max_new_tokens,
                device=ctx.device,
                seed=seed,
                temperature=temperature,
            )
            topics = (
                generate(
                    ctx.torch,
                    model,
                    ctx.tokenizer,
                    topic_prefixes,
                    max_new_tokens=max_new_tokens,
                    device=ctx.device,
                    seed=seed + 1,
                    temperature=temperature,
                )
                if topic_prefixes
                else []
            )
        outputs[label] = {"held_out": held, "new_topics": topics}
    reference_rows = [record_features(r) for r, _ in usable]
    scale_rows = [record_features(r) for r in train_records] or reference_rows
    train_texts = [r.target_text() for r in train_records]
    index = OverlapIndex(train_texts)
    result = {
        "decoding": {"temperature": temperature, "max_new_tokens": max_new_tokens, "seed": seed},
        "contexts": len(prefixes),
        "new_topic_prompts": len(topic_prefixes),
    }
    for label in ("base", "created"):
        held_rows = [features(t) for t in outputs[label]["held_out"]]
        topic_rows = [features(t) for t in outputs[label]["new_topics"]]
        overlap = [
            index.measure(t) for t in outputs[label]["held_out"] + outputs[label]["new_topics"]
        ]
        fluency = _base_fluency(
            ctx, edits, outputs[label]["held_out"] + outputs[label]["new_topics"]
        )
        result[label] = {
            "style_distance_held_out": style_distance(
                held_rows, reference_rows, scale_rows, supported_features
            ),
            "style_distance_new_topics": style_distance(
                topic_rows, reference_rows, scale_rows, supported_features
            )
            if topic_rows
            else None,
            "repetition_3gram": float(
                np.mean(
                    [
                        repetition(t)
                        for t in outputs[label]["held_out"] + outputs[label]["new_topics"]
                    ]
                )
            )
            if prefixes or topic_prefixes
            else None,
            "base_model_nll_of_generation": fluency,
            "ngram8_overlap_with_training_mean": float(
                np.mean([o["ngram_overlap"] for o in overlap])
            )
            if overlap
            else None,
            "longest_copied_words_max": int(
                max([o["longest_copied_words"] for o in overlap] or [0])
            ),
            "samples": {
                "held_out": outputs[label]["held_out"][:3],
                "new_topics": outputs[label]["new_topics"][:3],
            },
        }
    return result, outputs, usable


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _base_fluency(ctx, edits, texts):
    """Mean NLL per token of generated text under the *base* model (fluency proxy)."""
    from .serialize import Example

    examples = []
    for i, text in enumerate(texts):
        ids = ctx.tokenizer(text, add_special_tokens=False)["input_ids"][
            : ctx.serializer.max_tokens
        ]
        if len(ids) >= 2:
            examples.append(Example(f"g{i}", ids, [1] * len(ids), "generation"))
    if not examples:
        return None
    model, context = variant(ctx, edits, "base")
    with context:
        totals, counts = per_example_nll(
            ctx.torch, model, examples, ctx.tokenizer, ctx.device, autocast=ctx.autocast
        )
    return float(totals.sum() / max(counts.sum(), 1))


def fact_cloze(ctx, edits):
    """Greedy first word for public cloze facts, base and created."""
    prefixes = [ctx.serializer.prompt_text_ids(prompt) for prompt, _ in FACT_CLOZE]
    scores = {}
    for label in ("base", "created"):
        model, context = variant(ctx, edits, label)
        with context:
            outputs = generate(
                ctx.torch,
                model,
                ctx.tokenizer,
                prefixes,
                max_new_tokens=4,
                device=ctx.device,
                seed=0,
                temperature=0.0,
            )
        correct = 0
        for (_, answer), text in zip(FACT_CLOZE, outputs):
            words = _WORD.findall(text.lower())
            correct += bool(words) and words[0] == answer
        scores[label] = correct / len(FACT_CLOZE)
    return {
        "items": len(FACT_CLOZE),
        "base_accuracy": scores["base"],
        "created_accuracy": scores["created"],
    }


def agent_metrics(ctx, edits, test_records, train_records, *, max_new_tokens):
    """Tool calls and visible reasoning on held-out exchanges (greedy, both models)."""
    schema = tool_schema(train_records)
    items = []
    for record in test_records:
        if record.kind != "exchange":
            continue
        # Every observed assistant turn with tool calls or visible reasoning, generated
        # from exactly the context that preceded it (earlier tool results included).
        for index, message in enumerate(record.messages):
            if message.role != "assistant" or not (message.tool_calls or message.reasoning):
                continue
            if index == 0:
                continue
            prefix = ctx.serializer.prompt_ids(record.messages[:index])
            items.append((record, message, prefix))
    if not items:
        return None
    result = {"items": len(items), "inferred_tool_schema": schema}
    for label in ("base", "created"):
        model, context = variant(ctx, edits, label)
        with context:
            texts = generate(
                ctx.torch,
                model,
                ctx.tokenizer,
                [p for _, _, p in items],
                max_new_tokens=max_new_tokens,
                device=ctx.device,
                seed=0,
                temperature=0.0,
            )
        tool_rows, reasoning_f1, answer_f1, reasoning_present = [], [], [], []
        for (record, final, _), text in zip(items, texts):
            reasoning, rest = _split_reasoning(text)
            if final.tool_calls:
                tool_rows.append(score_tools(rest, final.tool_calls, schema))
            if final.reasoning:
                reasoning_present.append(reasoning is not None)
                reasoning_f1.append(token_f1(reasoning, final.reasoning))
            if final.content:
                visible = rest.split("<tool_call>")[0]
                answer_f1.append(token_f1(visible, final.content))

        def mean(values):
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if values else None

        result[label] = {
            "tool_items": len(tool_rows),
            "schema_valid_fraction": mean([r["schema_valid_fraction"] for r in tool_rows]),
            "selection_accuracy": mean([float(r["selection_correct"]) for r in tool_rows]),
            "sequence_exact": mean([float(r["sequence_exact"]) for r in tool_rows]),
            "arguments_exact": mean([float(r["arguments_exact"]) for r in tool_rows]),
            "malformed_blocks": int(sum(r["malformed_blocks"] for r in tool_rows)),
            "reasoning_items": len(reasoning_f1),
            "reasoning_present_rate": mean([float(v) for v in reasoning_present]),
            "reasoning_token_f1": mean(reasoning_f1),
            "visible_answer_token_f1": mean(answer_f1),
            "sample": texts[:2],
        }
    result["note"] = (
        "Visible reasoning is scored as observable text (token F1 against the observed "
        "reasoning), separately from the visible answer. Neither measures hidden computation."
    )
    return result


def supported_style_features(hypotheses):
    return [
        row["feature"]
        for row in hypotheses["hypotheses"]
        if row["status"] == "supported" and row["family"] not in ("tools",)
    ]


def instruction_following(instruction_map, generation):
    """Did mapped instruction targets move base -> created in the requested direction?"""
    if not instruction_map or not instruction_map.get("targets"):
        return None
    rows = []
    for target in instruction_map["targets"]:
        names = FAMILIES.get(target["target"], [target["target"]])
        if target["direction"] == 0:
            continue
        for name in names:
            rows.append({"feature": name, "requested_direction": target["direction"]})
    return {
        "directional_targets": rows,
        "note": "Family-level targets ('match the "
        "evidence') are measured by style distance; directional targets are listed here "
        "and evaluated in the benchmark reports.",
    }


def select_claims(*, method, evaluation, structure, identification):
    """Separate claim statuses from measured results. Never from process completion."""
    test = evaluation.get("test_nll") or {}
    improvement = test.get("improvement_nats_per_token") or {}
    gen = evaluation.get("generation") or {}
    base_style = ((gen.get("base") or {}).get("style_distance_held_out") or {}).get("distance")
    created_style = ((gen.get("created") or {}).get("style_distance_held_out") or {}).get(
        "distance"
    )
    behavioral = "inconclusive"
    reasons = []
    if improvement.get("low") is not None:
        if improvement["low"] > 0 and (
            base_style is None or created_style is None or created_style <= base_style
        ):
            behavioral = "supported"
        elif improvement["high"] is not None and improvement["high"] < 0:
            behavioral = "not_supported"
        reasons.append(
            f"test NLL improvement {improvement['mean']:.4f} nats/token, 95% interval "
            f"[{improvement['low']:.4f}, {improvement['high']:.4f}] over {improvement['groups']} groups"
        )
    if base_style is not None and created_style is not None:
        reasons.append(f"style distance base {base_style:.3f} -> created {created_style:.3f}")
    mechanistic = "not_attempted"
    mech_reasons = []
    if method in ("structured", "random-subspace") and structure:
        trial = structure.get("trial_summary") or {}
        selected_beats_random = trial.get("selected_beats_random")
        restore = evaluation.get("restoration_validation") or {}
        top_share = None
        if restore.get("total_gain_nats_per_token"):
            removed = sum(max(0.0, r["gain_removed_nats_per_token"]) for r in restore["per_region"])
            top_share = removed / restore["total_gain_nats_per_token"]
        if (
            selected_beats_random
            and restore.get("total_gain_nats_per_token", 0) > 0
            and (top_share is not None and top_share >= 0.5)
        ):
            mechanistic = "supported"
        else:
            mechanistic = "not_supported"
        mech_reasons.append(
            f"selected-region trial edits beat matched random regions: {selected_beats_random}; "
            f"summed single-region restoration removes {top_share if top_share is None else round(top_share, 3)} "
            "of the validation gain"
        )
    partial = "not_established"
    partial_reason = "no identification route with declared assumptions applied"
    if identification:
        region_test = identification.get("region_class_test") or {}
        if region_test.get("status") == "explained_by_single_region":
            partial = "estimate_conditional_on_declared_region_class"
            partial_reason = (
                f"region {region_test.get('best_region')} explains the held-out change within "
                "tolerance; the estimate is Δ·P_X for that region, conditional on the source "
                "being the base plus an edit confined to it (not testable against "
                "function-preserving reparameterizations)"
            )
        elif region_test.get("status"):
            partial_reason = f"region-class test: {region_test['status']}"
        head = identification.get("declared_class_head") or {}
        if head.get("status") == "recovered_in_class":
            partial = "estimate_conditional_on_declared_head_class"
            partial_reason = "declared-class head route: " + head.get("summary", "")
    if method in ("lora", "lora-matched", "sft"):
        mechanistic = "not_attempted"
        mech_reasons = ["behavioral fitting backend: structural inference was not used"]
    return {
        "run_method": method,
        "behavioral_reconstruction": {"status": behavioral, "evidence": reasons},
        "mechanistic_synthesis_in_base": {"status": mechanistic, "evidence": mech_reasons},
        "partial_source_parameter_recovery": {"status": partial, "evidence": [partial_reason]},
        "full_source_parameter_recovery": {
            "status": "not_established",
            "evidence": [
                "No Create Mode observation contract identifies a whole unknown model; output "
                "similarity never establishes source parameters."
            ],
        },
    }
