"""Stage 3 structural inference: where in the base, and in which subspaces, to edit.

For every candidate linear map ``r`` (q/k/v/o/gate/up/down of each layer) we estimate

* ``A_r = E[x xᵀ]``: input second moments on retain contexts,
* ``B_r = E[g gᵀ]``: output-gradient second moments on retain contexts, with labels
  sampled from the base itself (true Fisher, not empirical Fisher),
* ``G_r``: the gradient of the evidence NLL at the base weights.

With Kronecker-factored retain curvature ``F_r ≈ A_r ⊗ B_r``, the whitened gradient
``Ĝ_r = B̃_r^{-1/2} G_r Ã_r^{-1/2}`` (tilde: damped) gives the predicted first-order
evidence-NLL decrease of the best rank-k edit under a fixed second-order retain-KL budget
ε: ``sqrt(2ε) · sqrt(Σ_{i≤k} σ_i(Ĝ_r)²)``. That is the region score. The top singular
vectors, mapped back through the inverse whitening, are the inferred edit subspaces.

These are *base-model* findings. They say nothing about which circuit an inaccessible
source used.
"""

from __future__ import annotations

import math

import numpy as np

from .modeling import collate, pad_id, per_example_nll, region_module


def _input_site(region: str) -> str:
    layer, block, module = region.split(".")
    if module in ("q_proj", "k_proj", "v_proj"):
        return f"{layer}.attn_in"
    if module in ("gate_proj", "up_proj"):
        return f"{layer}.mlp_in"
    return region


class _Moments:
    """Running second moments: dense up to ``max_dense`` features, diagonal beyond."""

    def __init__(self, torch, dim, device, max_dense):
        self.dense = dim <= max_dense
        self.dim = dim
        self.count = 0
        shape = (dim, dim) if self.dense else (dim,)
        self.value = torch.zeros(shape, dtype=torch.float32, device=device)

    def add(self, rows):
        rows = rows.float()
        if self.dense:
            self.value.addmm_(rows.T, rows)
        else:
            self.value.add_((rows * rows).sum(0))
        self.count += rows.shape[0]

    def inverse_sqrt(self, torch, damping):
        mean = self.value / max(self.count, 1)
        if self.dense:
            trace = torch.diagonal(mean).mean().clamp_min(1e-12)
            matrix = mean + damping * trace * torch.eye(self.dim, device=mean.device)
            values, vectors = torch.linalg.eigh(matrix.double())
            values = values.clamp_min(1e-12)
            return (vectors * values.rsqrt()) @ vectors.T, (vectors * values.sqrt()) @ vectors.T
        trace = mean.mean().clamp_min(1e-12)
        diag = (mean + damping * trace).double()
        return torch.diag(diag.rsqrt()), torch.diag(diag.sqrt())


def collect_curvature(
    torch,
    model,
    regions,
    curvature_examples,
    tokenizer,
    device,
    *,
    max_dense=2048,
    batch_size=4,
    seed=0,
    progress=None,
):
    """Kronecker factors of the model Fisher (labels sampled from the base) per region.

    ``curvature_examples`` mixes retain contexts with training-evidence contexts, so the
    natural-gradient metric bounds the change of the model on both distributions.
    """
    sites = {}
    outputs = {}
    handles = []
    mask_holder = {}
    for region in regions:
        module = region_module(model, region)
        site = _input_site(region)
        if site not in sites:
            sites[site] = _Moments(torch, module.in_features, device, max_dense)

            def forward_hook(mod, inputs, output, site=site):
                x = inputs[0].detach()
                mask = mask_holder["mask"]
                sites[site].add(x[mask])

            handles.append(module.register_forward_hook(forward_hook))
        outputs[region] = _Moments(torch, module.out_features, device, max_dense)

        def backward_hook(mod, grad_input, grad_output, region=region):
            g = grad_output[0].detach()
            mask = mask_holder["mask"]
            outputs[region].add(g[mask])

        handles.append(module.register_full_backward_hook(backward_hook))
    embed = model.get_input_embeddings()
    handles.append(embed.register_forward_hook(lambda m, i, o: o.requires_grad_(True)))
    generator = torch.Generator(device=device).manual_seed(seed)
    padding = pad_id(tokenizer)
    try:
        for start in range(0, len(curvature_examples), batch_size):
            batch = curvature_examples[start : start + batch_size]
            ids, attention, _ = collate(torch, batch, padding, device)
            mask_holder["mask"] = attention.bool()
            logits = model(input_ids=ids, attention_mask=attention).logits.float()
            with torch.no_grad():
                probabilities = torch.softmax(logits, -1)
                sampled = torch.multinomial(
                    probabilities.view(-1, probabilities.shape[-1]), 1, generator=generator
                ).view(probabilities.shape[:-1])
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]), sampled.view(-1), reduction="none"
            ).view(sampled.shape)
            (loss * attention).sum().backward()
            model.zero_grad(set_to_none=True)
            if progress:
                progress(min(start + batch_size, len(curvature_examples)), len(curvature_examples))
    finally:
        for handle in handles:
            handle.remove()
    return sites, outputs


def evidence_gradients(
    torch, model, regions, examples, tokenizer, device, *, batch_size=4, progress=None
):
    """Gradient of the mean evidence token NLL with respect to each region's weight."""
    weights = {region: region_module(model, region).weight for region in regions}
    previous = {name: p.requires_grad for name, p in model.named_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for weight in weights.values():
        weight.requires_grad_(True)
        weight.grad = None
    total_tokens = sum(example.target_tokens for example in examples)
    padding = pad_id(tokenizer)
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            ids, attention, labels = collate(torch, batch, padding, device)
            logits = model(input_ids=ids, attention_mask=attention).logits
            from .modeling import token_nll

            nll, _ = token_nll(torch, logits, labels)
            (nll.sum() / total_tokens).backward()
            if progress:
                progress(min(start + batch_size, len(examples)), len(examples))
        gradients = {region: weights[region].grad.detach().float().clone() for region in regions}
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(previous[name])
            parameter.grad = None
    return gradients


def score_regions(torch, regions, gradients, sites, outputs, *, rank, damping):
    """Whitened rank-k gradient energy, inferred frames and natural-step factors."""
    results = {}
    cache = {}
    for region in regions:
        site = _input_site(region)
        if site not in cache:
            cache[site] = sites[site].inverse_sqrt(torch, damping)
        a_inv_sqrt, _ = cache[site]
        b_inv_sqrt, _ = outputs[region].inverse_sqrt(torch, damping)
        gradient = gradients[region].double()
        whitened = b_inv_sqrt @ gradient @ a_inv_sqrt
        u, s, vh = torch.linalg.svd(whitened, full_matrices=False)
        k = min(rank, s.shape[0])
        energy = float((s[:k] ** 2).sum().item())
        total = float((s**2).sum().item())
        left = b_inv_sqrt @ u[:, :k]  # output-side natural directions
        right = a_inv_sqrt @ vh[:k].T  # input-side natural directions
        P, _ = torch.linalg.qr(left)
        Q, _ = torch.linalg.qr(right)
        results[region] = {
            "score": math.sqrt(energy),
            "whitened_gradient_norm": math.sqrt(total),
            "rank_k_energy_fraction": energy / total if total > 0 else 0.0,
            "singular_values": [float(v) for v in s[:k].tolist()],
            "gradient_frobenius": float(gradient.norm().item()),
            "input_dense": sites[site].dense,
            "output_dense": outputs[region].dense,
            # Factors of the natural rank-k step D = (B^-1/2 U Σ)(A^-1/2 V)ᵀ, unscaled.
            "_step_left": (left * s[:k]).float(),
            "_step_right": right.float(),
            "_frame": (P.float(), Q.float()),
        }
    return results


def select_regions(scores: dict, count: int, *, max_per_layer: int = 2) -> list[str]:
    ranked = sorted(scores, key=lambda r: scores[r]["score"], reverse=True)
    chosen, per_layer = [], {}
    for region in ranked:
        layer = region.split(".")[0]
        if per_layer.get(layer, 0) >= max_per_layer:
            continue
        chosen.append(region)
        per_layer[layer] = per_layer.get(layer, 0) + 1
        if len(chosen) == count:
            break
    return chosen


class _TrialEdit:
    def __init__(self, torch, model, region, left, right, scale):
        self.torch = torch
        self.left, self.right, self.scale = left, right, scale
        self.handle = region_module(model, region).register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        x = inputs[0].float()
        return output + (self.scale * ((x @ self.right) @ self.left.T)).to(output.dtype)

    def remove(self):
        self.handle.remove()


def trial_edits(
    torch,
    model,
    tokenizer,
    device,
    scores,
    candidates,
    validation,
    kl_examples,
    *,
    epsilon,
    seed,
    edits_kl,
    progress=None,
):
    """Causal test of the localization: one natural step per region, measured held out.

    Each region receives its natural-gradient rank-k step, first scaled to a predicted KL
    of ``epsilon`` and then halved until the *measured* KL from the base on
    ``kl_examples`` (retain and training-evidence contexts, never validation) is at most
    ``2 epsilon``. Validation NLL is measured once, at the accepted scale, and compared
    with a random rank-k direction of equal whitened norm at the same scale.
    """
    base_totals, base_counts = per_example_nll(torch, model, validation, tokenizer, device)
    base_nll = base_totals.sum() / max(base_counts.sum(), 1)
    rng = np.random.default_rng(seed)
    rows = []
    for index, region in enumerate(candidates):
        entry = scores[region]
        energy = entry["score"]
        if energy <= 0:
            continue
        initial = -math.sqrt(2 * epsilon) / energy
        scale = initial
        for attempt in range(10):
            trial = _TrialEdit(
                torch, model, region, entry["_step_left"], entry["_step_right"], scale
            )
            try:
                kl = edits_kl(kl_examples)
                if kl <= 2 * epsilon or attempt == 9:
                    totals, counts = per_example_nll(torch, model, validation, tokenizer, device)
                    break
            finally:
                trial.remove()
            scale /= 2
        # Random direction with the same whitened Frobenius norm in the same region.
        left = entry["_step_left"]
        right = entry["_step_right"]
        k = left.shape[1]
        mix = torch.tensor(
            np.linalg.qr(rng.standard_normal((k, k)))[0], dtype=torch.float32, device=left.device
        )
        signs = torch.tensor(
            rng.choice([-1.0, 1.0], size=k), dtype=torch.float32, device=left.device
        )
        random_trial = _TrialEdit(torch, model, region, left @ mix * signs, right, scale)
        try:
            r_totals, r_counts = per_example_nll(torch, model, validation, tokenizer, device)
        finally:
            random_trial.remove()
        rows.append(
            {
                "region": region,
                "score": energy,
                "predicted_train_gain_at_applied_scale": abs(scale) * energy**2,
                "applied_scale_fraction": scale / initial,
                "measured_kl": float(kl),
                "validation_gain": float(base_nll - totals.sum() / max(counts.sum(), 1)),
                "random_direction_validation_gain": float(
                    base_nll - r_totals.sum() / max(r_counts.sum(), 1)
                ),
            }
        )
        if progress:
            progress(index + 1, len(candidates))
    return {
        "base_validation_nll": float(base_nll),
        "epsilon": epsilon,
        "kl_budget": "measured KL(base || trial) <= 2 epsilon on retain and training-evidence "
        "contexts",
        "rows": rows,
    }


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def describe_directions(torch, model, tokenizer, scores, regions, top_tokens=8):
    """Logit-lens reading of write-side edit directions: a hypothesis label, not a neuron."""
    head = model.get_output_embeddings().weight
    norm = getattr(model.model, "norm", None)
    scale = norm.weight.float() if norm is not None and hasattr(norm, "weight") else None
    out = {}
    for region in regions:
        if not region.endswith(("o_proj", "down_proj")):
            continue
        P, _ = scores[region]["_frame"]
        readings = []
        for column in range(min(3, P.shape[1])):
            direction = P[:, column].to(head.device)
            if scale is not None:
                direction = direction * scale
            logits = head.float() @ direction
            top = torch.topk(logits, top_tokens).indices.tolist()
            bottom = torch.topk(-logits, top_tokens).indices.tolist()
            readings.append(
                {
                    "axis": column,
                    "one_end": [tokenizer.decode([t]) for t in top],
                    "other_end": [tokenizer.decode([t]) for t in bottom],
                }
            )
        out[region] = readings
    return out


def public_scores(scores):
    return {
        region: {key: value for key, value in entry.items() if not key.startswith("_")}
        for region, entry in scores.items()
    }
