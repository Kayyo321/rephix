"""Stage 2: measurable behavioral constraints and their hypothesis tests.

Each feature maps a text to a number. A hypothesis "the evidence differs from the base
in feature f" is *supported* only when the group-bootstrap interval of the standardized
difference (evidence minus base continuations of the same contexts) excludes zero on the
training groups and the difference keeps its sign on validation groups. Content (topic
words) is reported separately and never counted as style.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*|\d+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
FUNCTION_WORDS = (
    "the of and to a in that is it for as with was on be by this which are not or from at "
    "but an have has had were their its they he she we you i his her them these those there "
    "would could should may might must can will shall also however therefore thus"
).split()
CONTRACTION = re.compile(r"\b\w+(?:'|’)(?:s|t|re|ve|ll|d|m)\b", re.I)
HEDGES = re.compile(
    r"\b(perhaps|possibly|probably|likely|seems?|appears?|suggests?|may|might)\b", re.I
)
CONNECTIVES = re.compile(
    r"\b(however|moreover|furthermore|therefore|thus|hence|consequently|nevertheless|"
    r"indeed|accordingly|whereas|although)\b",
    re.I,
)
FIRST_PERSON = re.compile(r"\b(i|me|my|mine|we|us|our|ours)\b", re.I)
SECOND_PERSON = re.compile(r"\b(you|your|yours)\b", re.I)

FAMILIES = {
    "length": ["characters", "words"],
    "sentence_structure": ["sentence_words_mean", "sentence_words_sd", "word_length_mean"],
    "paragraph_structure": ["paragraph_sentences_mean", "paragraphs_per_1k_words"],
    "lexical": ["type_token_ratio_100", "long_word_rate"],
    "punctuation": [
        "comma_rate",
        "semicolon_rate",
        "colon_rate",
        "dash_rate",
        "parenthesis_rate",
        "question_rate",
        "exclamation_rate",
        "quote_rate",
    ],
    "register": [
        "contraction_rate",
        "first_person_rate",
        "second_person_rate",
        "hedge_rate",
        "function_word_rate",
    ],
    "discourse": ["connective_rate"],
    "formatting": ["bullet_line_rate", "heading_line_rate", "code_fence_rate", "emphasis_rate"],
    "tools": ["tool_call_count", "reasoning_present"],
}
FEATURES = [name for names in FAMILIES.values() for name in names]

# Instruction terms mapped to feature families and the direction they request
# (+1 more, -1 less, 0 "match the evidence"). Unmapped terms are reported as unmapped.
INSTRUCTION_TERMS = {
    "formal": [("register", 0), ("contraction_rate", -1)],
    "informal": [("register", 0), ("contraction_rate", +1)],
    "casual": [("register", 0), ("contraction_rate", +1)],
    "voice": [("register", 0)],
    "tone": [("register", 0)],
    "register": [("register", 0)],
    "paragraph": [("paragraph_structure", 0)],
    "structure": [("paragraph_structure", 0), ("sentence_structure", 0)],
    "sentence": [("sentence_structure", 0)],
    "concise": [("length", 0), ("words", -1)],
    "brief": [("length", 0), ("words", -1)],
    "short": [("length", 0), ("words", -1)],
    "detailed": [("length", 0), ("words", +1)],
    "list": [("formatting", 0)],
    "bullet": [("formatting", 0), ("bullet_line_rate", +1)],
    "format": [("formatting", 0)],
    "punctuation": [("punctuation", 0)],
    "vocabulary": [("lexical", 0)],
    "style": [
        (family, 0) for family in ("sentence_structure", "lexical", "punctuation", "register")
    ],
    "tool": [("tools", 0)],
    "reasoning": [("tools", 0)],
    "hedge": [("register", 0), ("hedge_rate", +1)],
    "argument": [("discourse", 0)],
    "connective": [("discourse", 0)],
}


def features(text: str, *, tool_calls: int = 0, reasoning: bool = False) -> dict:
    words = _WORD.findall(text)
    n_words = max(len(words), 1)
    chars = max(len(text), 1)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences = [s for s in _SENTENCE.split(text.replace("\n", " ")) if _WORD.search(s)]
    sentence_lengths = [len(_WORD.findall(s)) for s in sentences] or [0]
    lines = [line for line in text.split("\n") if line.strip()]
    n_lines = max(len(lines), 1)
    window = [w.lower() for w in words[:100]]
    per_k = 1000.0 / chars
    return {
        "characters": float(len(text)),
        "words": float(len(words)),
        "sentence_words_mean": float(np.mean(sentence_lengths)),
        "sentence_words_sd": float(np.std(sentence_lengths)),
        "word_length_mean": float(np.mean([len(w) for w in words])) if words else 0.0,
        "paragraph_sentences_mean": float(len(sentences) / max(len(paragraphs), 1)),
        "paragraphs_per_1k_words": 1000.0 * len(paragraphs) / n_words,
        "type_token_ratio_100": len(set(window)) / max(len(window), 1),
        "long_word_rate": sum(len(w) >= 9 for w in words) / n_words,
        "comma_rate": text.count(",") * per_k,
        "semicolon_rate": text.count(";") * per_k,
        "colon_rate": text.count(":") * per_k,
        "dash_rate": (text.count("—") + text.count(" - ") + text.count("--")) * per_k,
        "parenthesis_rate": text.count("(") * per_k,
        "question_rate": text.count("?") * per_k,
        "exclamation_rate": text.count("!") * per_k,
        "quote_rate": (text.count('"') + text.count("“") + text.count("”")) * per_k,
        "contraction_rate": len(CONTRACTION.findall(text)) / n_words,
        "first_person_rate": len(FIRST_PERSON.findall(text)) / n_words,
        "second_person_rate": len(SECOND_PERSON.findall(text)) / n_words,
        "hedge_rate": len(HEDGES.findall(text)) / n_words,
        "function_word_rate": sum(w.lower() in FUNCTION_WORDS for w in words) / n_words,
        "connective_rate": len(CONNECTIVES.findall(text)) / n_words,
        "bullet_line_rate": sum(bool(re.match(r"\s*([-*•]|\d+[.)])\s", ln)) for ln in lines)
        / n_lines,
        "heading_line_rate": sum(bool(re.match(r"\s*#{1,6}\s", ln)) for ln in lines) / n_lines,
        "code_fence_rate": text.count("```") / n_lines,
        "emphasis_rate": (text.count("**") + text.count("__")) * per_k,
        "tool_call_count": float(tool_calls),
        "reasoning_present": 1.0 if reasoning else 0.0,
    }


def record_features(record) -> dict:
    tool_calls = sum(len(m.tool_calls or []) for m in record.assistant_messages())
    reasoning = any(m.reasoning for m in record.assistant_messages())
    if record.kind == "passage":
        text = record.text
    else:
        text = "\n\n".join(m.content or "" for m in record.assistant_messages())
    return features(text, tool_calls=tool_calls, reasoning=reasoning)


def _group_means(rows, groups):
    by_group = {}
    for row, group in zip(rows, groups):
        by_group.setdefault(group, []).append(row)
    keys = sorted(by_group)
    return keys, {
        name: np.array([np.mean([r[name] for r in by_group[k]]) for k in keys]) for name in FEATURES
    }


def standardized_differences(evidence_rows, base_rows, groups, *, replicates=1000, seed=0):
    """Per-feature (evidence − base) / pooled sd, with a bootstrap over paired groups."""
    keys, evidence = _group_means(evidence_rows, groups)
    _, base = _group_means(base_rows, groups)
    rng = np.random.default_rng(seed)
    n = len(keys)
    draws = rng.integers(0, n, size=(replicates, n)) if n else None
    result = {}
    for name in FEATURES:
        e, b = evidence[name], base[name]
        sd = math.sqrt((np.var(e) + np.var(b)) / 2) if n > 1 else 0.0
        if n > 1 and sd < 1e-12 and abs(float(np.mean(e - b))) > 1e-12:
            # Both sides constant but different: a deterministic, maximal separation.
            effect = 1e3 * float(np.sign(np.mean(e - b)))
            result[name] = {
                "evidence_mean": float(np.mean(e)),
                "base_mean": float(np.mean(b)),
                "effect": effect,
                "low": effect,
                "high": effect,
                "degenerate": False,
                "constant_difference": True,
            }
            continue
        if n == 0 or sd < 1e-12:
            diff = float(np.mean(e - b)) if n else 0.0
            result[name] = {
                "evidence_mean": float(np.mean(e)) if n else None,
                "base_mean": float(np.mean(b)) if n else None,
                "effect": None,
                "low": None,
                "high": None,
                "degenerate": True,
                "raw_difference": diff,
            }
            continue
        diff = e - b
        boot = diff[draws].mean(axis=1) / sd
        result[name] = {
            "evidence_mean": float(np.mean(e)),
            "base_mean": float(np.mean(b)),
            "effect": float(np.mean(diff) / sd),
            "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975)),
            "degenerate": False,
        }
    return result


def hypotheses(train_diff, validation_diff, *, groups_train, groups_validation):
    """Supported only with a train interval excluding zero and a same-sign validation effect."""
    rows = []
    for family, names in FAMILIES.items():
        for name in names:
            t = train_diff[name]
            v = validation_diff.get(name, {}) if validation_diff else {}
            if t["degenerate"]:
                status = "not_measurable"
            elif t["low"] > 0 or t["high"] < 0:
                if (
                    v
                    and not v.get("degenerate")
                    and v.get("effect") is not None
                    and (np.sign(v["effect"]) == np.sign(t["effect"]))
                ):
                    status = "supported"
                else:
                    status = "not_replicated_on_validation"
            else:
                status = "not_supported"
            rows.append(
                {
                    "feature": name,
                    "family": family,
                    "kind": "style" if family not in ("tools", "length") else family,
                    "status": status,
                    "train_effect": t["effect"],
                    "train_interval": [t["low"], t["high"]],
                    "validation_effect": v.get("effect") if v else None,
                    "evidence_mean": t["evidence_mean"],
                    "base_mean": t["base_mean"],
                }
            )
    limitation = None
    if groups_train < 8 or groups_validation < 3:
        limitation = (
            f"only {groups_train} training and {groups_validation} validation groups: style "
            "hypotheses are exploratory and no confident style claim is made"
        )
    return {
        "rule": "supported = train 95% group-bootstrap interval excludes 0 and the validation "
        "effect has the same sign",
        "groups": {"train": groups_train, "validation": groups_validation},
        "limitation": limitation,
        "hypotheses": rows,
    }


def content_terms(evidence_texts, base_texts, top=25):
    """Distinctive content words (topic), kept apart from style. Log-odds with a prior."""
    stop = set(FUNCTION_WORDS)
    ev = Counter(w.lower() for t in evidence_texts for w in _WORD.findall(t) if len(w) > 3)
    bs = Counter(w.lower() for t in base_texts for w in _WORD.findall(t) if len(w) > 3)
    for word in list(ev):
        if word in stop:
            del ev[word]
    total_e, total_b = sum(ev.values()) or 1, sum(bs.values()) or 1
    scores = {
        w: math.log((ev[w] + 0.5) / total_e) - math.log((bs.get(w, 0) + 0.5) / total_b)
        for w in ev
        if ev[w] >= 3
    }
    return [w for w, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top]]


def map_instruction(instruction: str | None) -> dict:
    if not instruction:
        return {"instruction": None, "targets": [], "unmapped_terms": []}
    words = [w.lower() for w in _WORD.findall(instruction)]
    targets, mapped = [], set()
    for word in words:
        stem = word.rstrip("s") if word not in INSTRUCTION_TERMS else word
        key = word if word in INSTRUCTION_TERMS else stem if stem in INSTRUCTION_TERMS else None
        if key is None:
            continue
        mapped.add(word)
        for target, direction in INSTRUCTION_TERMS[key]:
            entry = {"target": target, "direction": direction, "term": word}
            if entry not in targets:
                targets.append(entry)
    content_words = [w for w in words if len(w) > 3 and w not in FUNCTION_WORDS]
    return {
        "instruction": instruction,
        "targets": targets,
        "unmapped_terms": sorted(set(content_words) - mapped),
        "note": "Mapped terms select measured feature families; unmapped terms still shape "
        "the context-distillation objective but have no dedicated measurement.",
    }


def style_distance(generated_rows, reference_rows, scale_rows, feature_names):
    """Mean |mean_gen − mean_ref| / sd_scale over the given features."""
    if not generated_rows or not reference_rows or not feature_names:
        return None
    total, used = 0.0, 0
    for name in feature_names:
        sd = float(np.std([r[name] for r in scale_rows]))
        if sd < 1e-9:
            continue
        gen = float(np.mean([r[name] for r in generated_rows]))
        ref = float(np.mean([r[name] for r in reference_rows]))
        total += abs(gen - ref) / sd
        used += 1
    return {"distance": total / used if used else None, "features": used}
