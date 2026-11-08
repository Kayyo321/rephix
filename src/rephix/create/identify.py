"""Stage 4: capability-gated parameter identification.

Region-class test
    Hypothesis H_r,k: the source is the base plus a rank-<=k edit confined to one linear
    map r. Each top-ranked region is fitted alone, and the unrestricted structured edit
    is fitted with equal steps; held-out validation NLL is compared per group with a
    paired bootstrap. The test can reject H or fail to separate regions; it cannot prove
    H, and it cannot see function-preserving reparameterizations of the source.

Declared-class head
    The Rephix 0.4 estimator (softmax-affine head on a public backbone, here the base's
    final hidden state), used only when the evidence has repeated samples of identical
    contexts over a small closed set of output strings.
"""

from __future__ import annotations

import numpy as np

from .modeling import group_nll, paired_bootstrap, per_example_nll, region_module

TOLERANCE_NATS = 0.01  # Allowed excess validation NLL per token of a single region.


def _validation_groups(ctx, edits, group_of):
    totals, counts = per_example_nll(
        ctx.torch,
        ctx.model,
        ctx.validation_examples,
        ctx.tokenizer,
        ctx.device,
        autocast=ctx.autocast,
    )
    keys, sums, tokens = group_nll(ctx.validation_examples, totals, counts, group_of)
    return keys, sums, tokens


def region_inputs(ctx, region, examples, limit=2048):
    """Inputs reaching ``region`` on held-out examples (for Δ·X comparisons)."""
    torch = ctx.torch
    captured = []
    module = region_module(ctx.model, region)

    def hook(mod, inputs, output):
        captured.append(inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]).cpu())

    handle = module.register_forward_hook(hook)
    try:
        per_example_nll(torch, ctx.model, examples[:16], ctx.tokenizer, ctx.device)
    finally:
        handle.remove()
    rows = torch.cat(captured)[:limit]
    return rows


def region_class_test(ctx, fit_one, top_regions, all_regions, group_of, *, seeds=(1, 2)):
    """Fit single regions and the unrestricted edit; compare held-out NLL; test seeds."""
    fits = {}
    for region in top_regions:
        edits, report = fit_one(f"single-{region}", [region], seed=0)
        fits[region] = (_validation_groups(ctx, edits, group_of), report, edits.state())
        edits.remove()
    edits, report = fit_one("unrestricted", all_regions, seed=0)
    unrestricted = _validation_groups(ctx, edits, group_of)
    edits.remove()
    keys = unrestricted[0]
    u_sums, tokens = unrestricted[1], unrestricted[2]
    rows = []
    for region in top_regions:
        (r_keys, r_sums, _), rep, _ = fits[region]
        if r_keys != keys:
            raise RuntimeError("validation groups changed between fits")
        excess = paired_bootstrap(r_sums - u_sums, tokens, seed=3)
        rows.append(
            {
                "region": region,
                "validation_nll": float(r_sums.sum() / tokens.sum()),
                "excess_over_unrestricted": excess,
                "steps": rep["steps"],
            }
        )
    rows.sort(key=lambda r: r["validation_nll"])
    best = rows[0]
    runner = rows[1] if len(rows) > 1 else None
    separation = None
    if runner is not None:
        b_sums = fits[best["region"]][0][1]
        r_sums = fits[runner["region"]][0][1]
        separation = paired_bootstrap(r_sums - b_sums, tokens, seed=4)
    excess = best["excess_over_unrestricted"]
    if excess["low"] is not None and excess["low"] > TOLERANCE_NATS:
        status = "rejected_single_region"
    elif (
        excess["high"] is not None
        and excess["high"] <= TOLERANCE_NATS
        and (separation is None or (separation["low"] is not None and separation["low"] > 0))
    ):
        status = "explained_by_single_region"
    else:
        status = "not_identified"
    result = {
        "hypothesis": "source = base + rank-<=k edit confined to one linear map",
        "tolerance_nats_per_token": TOLERANCE_NATS,
        "unrestricted_validation_nll": float(u_sums.sum() / tokens.sum()),
        "unrestricted_regions": list(all_regions),
        "single_region_fits": rows,
        "best_region": best["region"],
        "best_minus_runner_up": separation,
        "status": status,
        "limits": [
            "Failure to reject is not proof that the source has this structure.",
            "A function-preserving reparameterization of the source produces the same "
            "evidence and cannot be distinguished.",
            "The identifiable object is Δ·P_X: the edit on the span of inputs reached on "
            "the evidence distribution.",
        ],
    }
    if status == "explained_by_single_region":
        region = best["region"]
        inputs = region_inputs(ctx, region, ctx.validation_examples).to(ctx.device)
        outputs = []
        states = [fits[region][2]]
        for seed in seeds:
            edits, _ = fit_one(f"single-{region}-seed{seed}", [region], seed=seed)
            states.append(edits.state())
            edits.remove()
        torch = ctx.torch
        for state in states:
            tensors = state[region]
            delta = (tensors["P"] @ tensors["M"] @ tensors["Q"].T).to(ctx.device)
            outputs.append(inputs @ delta.T)
        variation = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                variation.append(
                    float((outputs[i] - outputs[j]).norm() / outputs[i].norm().clamp_min(1e-12))
                )
        result["seed_variation_of_delta_x"] = {
            "pairs": variation,
            "max": max(variation) if variation else None,
            "interpretation": "relative difference of Δx on held-out region inputs between "
            "seed refits; large values mean the finite-data estimate is not unique",
        }
        result["estimate"] = {
            "region": region,
            "delta_frobenius": float(
                (states[0][region]["P"] @ states[0][region]["M"] @ states[0][region]["Q"].T)
                .norm()
                .item()
            ),
        }
        if variation and max(variation) > 0.5:
            result["status"] = "explained_but_nonunique"
        del torch
    return result, fits


def declared_class_head(ctx, records, *, min_contexts=20, max_labels=32):
    """Rephix 0.4 declared-class estimator on the base's final hidden state, if applicable."""
    contexts = {}
    for record in records:
        key = record.context_key()
        if key is None:
            continue
        contexts.setdefault(key, []).append(record)
    repeated = {k: v for k, v in contexts.items() if len(v) > 1}
    labels = sorted({r.target_text() for v in repeated.values() for r in v})
    reasons = []
    if len(repeated) < min_contexts:
        reasons.append(f"{len(repeated)} repeated contexts (need >= {min_contexts})")
    if not 2 <= len(labels) <= max_labels:
        reasons.append(f"{len(labels)} distinct outputs (need a closed set of 2..{max_labels})")
    if reasons:
        return {"status": "not_applicable", "reasons": reasons}
    if all(len({r.target_text() for r in v}) == 1 for v in repeated.values()):
        return {
            "status": "deterministic_outputs_not_identifiable",
            "reasons": [
                "every repeated context returned one output: sampled-law parameters "
                "are not identifiable from deterministic (for example greedy) output"
            ],
        }
    from ..text_recovery import (
        bootstrap_misspecification,
        fisher_covariance,
        fit_multinomial,
        span_basis,
    )

    torch = ctx.torch
    keys = sorted(repeated)
    rows = []
    with torch.inference_mode():
        for key in keys:
            record = repeated[key][0]
            if record.kind == "passage":
                # Declared feature map: the base's final hidden state at the last token of
                # the context, tokenized with the base's default special tokens.
                prefix = ctx.tokenizer(record.context or "", add_special_tokens=True)["input_ids"]
            else:
                prefix = ctx.serializer.generation_prefix(record)
            ids = torch.tensor([prefix], device=ctx.device)
            hidden = ctx.model.model(input_ids=ids).last_hidden_state[0, -1]
            rows.append(hidden.double().cpu().numpy())
    features = np.concatenate([np.asarray(rows), np.ones((len(rows), 1))], axis=1)
    counts = np.zeros((len(keys), len(labels)))
    for i, key in enumerate(keys):
        for record in repeated[key]:
            counts[i, labels.index(record.target_text())] += 1
    basis, span = span_basis(features)
    z = features @ basis
    gamma, fit = fit_multinomial(z, counts)
    covariance = fisher_covariance(z, counts, gamma)
    theta = gamma @ basis.T
    canonical = theta - theta.mean(axis=0, keepdims=True)
    standard_error = None
    if covariance is not None:
        standard_error = float(np.sqrt(max(np.trace(covariance), 0.0)))
    empty = [labels[j] for j in range(len(labels)) if counts[:, j].sum() == 0]
    # Separation: a label never or always observed at some context drives the MLE to the
    # boundary; the ridge then determines the estimate.
    separated = int(
        ((counts == 0) | (counts == counts.sum(axis=1, keepdims=True))).all(axis=1).sum()
    )
    class_test = bootstrap_misspecification(z, counts, gamma, replicates=199, seed=7, ridge=1e-6)
    relative_se = None
    if standard_error is not None and np.linalg.norm(canonical) > 0:
        relative_se = standard_error / float(np.linalg.norm(canonical))
    if not fit.get("converged") or empty:
        status = "insufficient_evidence"
    elif min(class_test["p_value"], class_test["p_value_lower_tail"]) < 0.025:
        status = "misspecified_approximation"
    elif relative_se is None or relative_se > 0.1:
        status = "insufficient_evidence"
    else:
        status = "recovered_in_class"
    # Representability in the base: label logits from tied output rows have no bias term.
    representable = _head_representability(ctx, labels, features, canonical)
    return {
        "status": status,
        "class": "softmax-affine over the base's final hidden state (declared, not proven)",
        "labels": labels,
        "contexts": len(keys),
        "samples": int(counts.sum()),
        "span": span,
        "fit": {k: v for k, v in fit.items() if k != "objective"},
        "canonical_frobenius": float(np.linalg.norm(canonical)),
        "standard_error_frobenius": standard_error,
        "relative_standard_error": relative_se,
        "class_test": class_test,
        "canonical_parameters": canonical.tolist(),
        "span_basis_rank": int(basis.shape[1]),
        "contexts_with_separation": separated,
        "empty_labels": empty,
        "representability_in_base": representable,
        "summary": f"{len(labels)} labels, {len(keys)} contexts, {int(counts.sum())} samples",
        "limits": [
            "Assumes the source computes the base's final hidden state on these contexts.",
            "Identifies the sampled law; parameters additionally need a known sampling law.",
            "A nonrejected class test is not a recovery certificate.",
        ],
    }


def _head_representability(ctx, labels, features, canonical):
    tokenizer = ctx.tokenizer
    single = []
    for label in labels:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        single.append(ids[0] if len(ids) == 1 else None)
    if any(token is None for token in single):
        return {"exact": False, "reason": "some labels are not single base tokens"}
    tied = bool(getattr(ctx.model.config, "tie_word_embeddings", False))
    # Best bias substitute: a vector u with u·h ≈ 1 on observed contexts.
    hidden = features[:, :-1]
    u, *_ = np.linalg.lstsq(hidden, np.ones(len(hidden)), rcond=None)
    residual = float(np.linalg.norm(hidden @ u - 1) / np.sqrt(len(hidden)))
    return {
        "exact": False,
        "label_token_ids": single,
        "bias_column_representable_rms_error": residual,
        "tied_output_rows": tied,
        "reason": "the base output head has no bias; a bias needs a direction u with "
        "u·h(x) = 1, approximated with the reported error"
        + (
            "; tied rows would also change the input embeddings of the label tokens" if tied else ""
        ),
        "written_to_export": False,
    }
