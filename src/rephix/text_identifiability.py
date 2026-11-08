"""Executable counterexamples bounding what text-only observations can identify.

Each check constructs two targets and verifies, exhaustively or on every queried
input, that the observation channel cannot distinguish them while their parameters
(or unqueried behavior) differ. Together they show that no procedure, at any query
budget, recovers original parameters for every target from text alone.
"""

from __future__ import annotations

import itertools

import numpy as np


def _softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _contexts(vocabulary, order):
    return np.array(list(itertools.product(range(vocabulary), repeat=order)))


def _one_hot(contexts, vocabulary):
    count, order = contexts.shape
    features = np.zeros((count, order * vocabulary + 1))
    for position in range(order):
        features[np.arange(count), position * vocabulary + contexts[:, position]] = 1.0
    features[:, -1] = 1.0
    return features


def hidden_unit_permutation(seed=0, vocabulary=6, order=2, width=8):
    """Two MLP language models related by a hidden permutation emit identical text laws."""
    rng = np.random.default_rng(seed)
    features = _one_hot(_contexts(vocabulary, order), vocabulary)
    incoming = rng.normal(size=(features.shape[1], width))
    outgoing = rng.normal(size=(width, vocabulary))
    permutation = rng.permutation(width)
    while (permutation == np.arange(width)).all():
        permutation = rng.permutation(width)

    def law(w_in, w_out):
        return _softmax(np.tanh(features @ w_in) @ w_out)

    first = law(incoming, outgoing)
    second = law(incoming[:, permutation], outgoing[permutation])
    return {
        "claim": "a function-preserving reparameterization leaves every output law unchanged",
        "inputs_checked": "all contexts (exhaustive)",
        "contexts": int(len(features)),
        "max_probability_difference": float(np.abs(first - second).max()),
        "relative_parameter_distance": float(
            np.sqrt(
                np.linalg.norm(incoming - incoming[:, permutation]) ** 2
                + np.linalg.norm(outgoing - outgoing[permutation]) ** 2
            )
            / np.sqrt(np.linalg.norm(incoming) ** 2 + np.linalg.norm(outgoing) ** 2)
        ),
        "indistinguishable": bool(np.abs(first - second).max() < 1e-12),
    }


def greedy_temperature(seed=0, vocabulary=6, order=2, scale=2.5):
    """Greedy text cannot distinguish a model from its logit-scaled copy."""
    rng = np.random.default_rng(seed)
    features = _one_hot(_contexts(vocabulary, order), vocabulary)
    theta = rng.normal(size=(vocabulary, features.shape[1]))
    first, second = features @ theta.T, features @ (scale * theta).T
    p, q = _softmax(first), _softmax(second)
    centre = np.eye(vocabulary) - 1.0 / vocabulary
    return {
        "claim": "argmax outputs are invariant to positive logit scaling",
        "inputs_checked": "all contexts (exhaustive)",
        "contexts": int(len(features)),
        "greedy_outputs_identical": bool((first.argmax(1) == second.argmax(1)).all()),
        "mean_total_variation_between_laws": float(0.5 * np.abs(p - q).sum(1).mean()),
        "relative_canonical_parameter_distance": float(
            np.linalg.norm(centre @ theta * (scale - 1)) / np.linalg.norm(centre @ theta)
        ),
        "indistinguishable": bool((first.argmax(1) == second.argmax(1)).all()),
    }


def finite_query_budget(seed=0, vocabulary=6, order=2, queried=24, delta=8.0):
    """Any finite query set misses a context where a richer target behaves differently."""
    rng = np.random.default_rng(seed)
    contexts = _contexts(vocabulary, order + 1)
    selected = rng.choice(len(contexts), size=queried, replace=False)
    unqueried = np.setdiff1d(np.arange(len(contexts)), selected)
    trigger = contexts[unqueried[0]]
    features = _one_hot(contexts[:, 1:], vocabulary)  # order-k features of each context
    theta = rng.normal(size=(vocabulary, features.shape[1]))
    base = features @ theta.T
    altered = base.copy()
    hit = (contexts == trigger).all(axis=1)
    altered[hit, int(base[hit][0].argmin())] += delta  # raise the least likely output
    p, q = _softmax(base), _softmax(altered)
    return {
        "claim": "an order-(k+1) target can agree on every queried prompt and differ elsewhere",
        "queried_prompts": int(queried),
        "max_difference_on_queried": float(np.abs(p[selected] - q[selected]).max()),
        "total_variation_on_trigger": float(0.5 * np.abs(p[hit] - q[hit]).sum()),
        "indistinguishable_on_queries": bool(np.abs(p[selected] - q[selected]).max() == 0.0),
    }


def in_class_identification(seed=0, vocabulary=6, order=2):
    """Positive control: inside the declared class, equal laws force equal canonical parameters."""
    rng = np.random.default_rng(seed)
    contexts = _contexts(vocabulary, order)
    features = _one_hot(contexts, vocabulary)
    theta = rng.normal(size=(vocabulary, features.shape[1]))
    # Any other parameterization with the same laws on all contexts: add gauge terms.
    gauge_class = np.outer(np.ones(vocabulary), rng.normal(size=features.shape[1]))
    null = np.linalg.svd(features)[2][np.linalg.matrix_rank(features) :]
    gauge_null = rng.normal(size=(vocabulary, len(null))) @ null
    other = theta + gauge_class + gauge_null
    same_law = np.abs(_softmax(features @ theta.T) - _softmax(features @ other.T)).max()
    centre = np.eye(vocabulary) - 1.0 / vocabulary
    basis = np.linalg.svd(features, full_matrices=False)[2][: np.linalg.matrix_rank(features)].T
    canonical_gap = np.abs(centre @ theta @ basis - centre @ other @ basis).max()
    return {
        "claim": "class-centred parameters on the feature span are identified by the output law",
        "max_law_difference": float(same_law),
        "raw_parameter_difference": float(np.abs(theta - other).max()),
        "canonical_parameter_difference": float(canonical_gap),
        "identified": bool(same_law < 1e-12 and canonical_gap < 1e-9),
    }


def transformer_neuron_permutation(seed=0):
    """A small Qwen2 transformer and its MLP-permuted copy emit identical text."""
    from .hf import _deps

    torch, AutoConfig, AutoModelForCausalLM, _ = _deps()
    from transformers import Qwen2Config

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    first = AutoModelForCausalLM.from_config(config).double().eval()
    second = AutoModelForCausalLM.from_config(config).double().eval()
    second.load_state_dict(first.state_dict())
    generator = torch.Generator().manual_seed(seed)
    changed = 0.0
    total = 0.0
    with torch.no_grad():
        for layer in second.model.layers:
            order = torch.randperm(config.intermediate_size, generator=generator)
            mlp = layer.mlp
            for name in ("gate_proj", "up_proj"):
                weight = getattr(mlp, name).weight
                changed += float((weight - weight[order]).pow(2).sum())
                weight.copy_(weight[order])
            down = mlp.down_proj.weight
            changed += float((down - down[:, order]).pow(2).sum())
            down.copy_(down[:, order])
        total = sum(float(p.pow(2).sum()) for p in first.parameters())
        prompts = torch.randint(0, 64, (32, 12), generator=generator)
        logits_a = first(prompts).logits
        logits_b = second(prompts).logits
        samples_a = torch.multinomial(
            torch.softmax(logits_a[:, -1], -1),
            16,
            replacement=True,
            generator=torch.Generator().manual_seed(1),
        )
        samples_b = torch.multinomial(
            torch.softmax(logits_b[:, -1], -1),
            16,
            replacement=True,
            generator=torch.Generator().manual_seed(1),
        )
    difference = float((logits_a - logits_b).abs().max())
    return {
        "claim": "permuting transformer MLP neurons preserves every logit and sampled token",
        "prompts": 32,
        "max_logit_difference": difference,
        "sampled_tokens_identical": bool(torch.equal(samples_a, samples_b)),
        "relative_parameter_distance": float(np.sqrt(changed / total)),
        "indistinguishable": bool(difference < 1e-10 and torch.equal(samples_a, samples_b)),
    }


def run_all(seed=0, include_transformer=True):
    report = {
        "conclusion": (
            "Original parameters are not identifiable from text for every target: symmetric "
            "reparameterizations are indistinguishable at any budget, greedy text hides the "
            "output law, and finite queries cannot pin down unrestricted targets. Recovery is "
            "possible only up to a declared equivalence inside a declared class."
        ),
        "hidden_unit_permutation": hidden_unit_permutation(seed),
        "greedy_temperature": greedy_temperature(seed),
        "finite_query_budget": finite_query_budget(seed),
        "in_class_identification": in_class_identification(seed),
    }
    if include_transformer:
        try:
            report["transformer_neuron_permutation"] = transformer_neuron_permutation(seed)
        except (ImportError, RuntimeError) as exc:
            report["transformer_neuron_permutation"] = {"skipped": str(exc)}
    return report
