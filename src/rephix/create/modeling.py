"""Model-side utilities: region edits, batching, held-out likelihood, KL and generation.

A region is one linear map of one decoder layer, named ``"{layer}.{block}.{module}"``
such as ``"7.mlp.down_proj"``. An edit is ``Δ = P M Qᵀ`` added through a forward hook,
so the base weights stay untouched until an explicit merge.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np


def region_module(model, region: str):
    layer, block, module = region.split(".")
    return getattr(getattr(model.model.layers[int(layer)], block), module)


def _key(region: str) -> str:
    return region.replace(".", "__")


class RegionEdits:
    """Low-rank additive edits on selected linear maps, kept outside the base weights."""

    def __init__(self, torch, model, frames: dict, *, train_frames: bool, init: str = "zero_core"):
        self.torch = torch
        self.model = model
        self.regions = list(frames)
        self.scale = 1.0
        self.disabled = set()
        self.P, self.Q, self.M = {}, {}, {}
        device = next(model.parameters()).device
        for region, (P, Q) in frames.items():
            P = torch.as_tensor(P, dtype=torch.float32, device=device).clone()
            Q = torch.as_tensor(Q, dtype=torch.float32, device=device).clone()
            k = P.shape[1]
            if init == "zero_core":
                M = torch.zeros(k, k, device=device)
            elif init == "lora":
                # Standard LoRA: Δ = B A with B = 0. Here P plays B (zero), M = I.
                M = torch.eye(k, device=device)
                P = torch.zeros_like(P)
            else:
                raise ValueError(f"unknown edit init {init}")
            self.P[region] = torch.nn.Parameter(P, requires_grad=train_frames or init == "lora")
            self.Q[region] = torch.nn.Parameter(Q, requires_grad=train_frames or init == "lora")
            self.M[region] = torch.nn.Parameter(M, requires_grad=init != "lora")
        self.handles = []
        for region in self.regions:
            self.handles.append(
                region_module(model, region).register_forward_hook(self._hook(region))
            )

    def _hook(self, region):
        def hook(module, inputs, output):
            if self.scale == 0.0 or region in self.disabled:
                return output
            x = inputs[0].to(self.torch.float32)
            delta = ((x @ self.Q[region]) @ self.M[region].T) @ self.P[region].T
            return output + (self.scale * delta).to(output.dtype)

        return hook

    def parameters(self):
        for region in self.regions:
            for table in (self.P, self.Q, self.M):
                if table[region].requires_grad:
                    yield table[region]

    def trainable_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def delta(self, region):
        return self.P[region] @ self.M[region] @ self.Q[region].T

    def l2(self):
        torch = self.torch
        total = torch.zeros((), device=next(iter(self.M.values())).device)
        for region in self.regions:
            # ||P M Qᵀ||² computed without materializing Δ.
            pm = self.P[region] @ self.M[region]
            total = total + ((pm.T @ pm) * (self.Q[region].T @ self.Q[region])).sum()
        return total

    @contextlib.contextmanager
    def off(self, regions=None):
        """Temporarily restore base behavior for all or some regions."""
        previous_scale, previous = self.scale, set(self.disabled)
        if regions is None:
            self.scale = 0.0
        else:
            self.disabled |= set(regions)
        try:
            yield
        finally:
            self.scale, self.disabled = previous_scale, previous

    def state(self):
        return {
            region: {
                "P": self.P[region].detach().cpu(),
                "Q": self.Q[region].detach().cpu(),
                "M": self.M[region].detach().cpu(),
            }
            for region in self.regions
        }

    def load_state(self, state):
        if set(state) != set(self.regions):
            raise ValueError("checkpoint edit regions do not match this run")
        with self.torch.no_grad():
            for region, tensors in state.items():
                self.P[region].copy_(tensors["P"])
                self.Q[region].copy_(tensors["Q"])
                self.M[region].copy_(tensors["M"])

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def merge(self, model=None):
        """Write Δ into the weights of ``model`` (default: the edited model). Hooks removed."""
        torch = self.torch
        model = model or self.model
        self.remove()
        report = {}
        with torch.no_grad():
            for region in self.regions:
                weight = region_module(model, region).weight
                delta = self.delta(region).to(weight.device)
                exact = weight.detach().to(torch.float64) + delta.to(torch.float64)
                merged = exact.to(weight.dtype)
                lost = (merged.to(torch.float64) - exact).norm().item()
                report[region] = {
                    "delta_frobenius": float(delta.norm().item()),
                    "weight_frobenius": float(weight.detach().float().norm().item()),
                    "rounding_error_frobenius": float(lost),
                    "rounding_error_relative_to_delta": float(
                        lost / max(delta.norm().item(), 1e-30)
                    ),
                }
                weight.copy_(merged)
        return report


def random_frames(rng, regions, shapes, rank):
    frames = {}
    for region in regions:
        out_dim, in_dim = shapes[region]
        P, _ = np.linalg.qr(rng.standard_normal((out_dim, rank)))
        Q, _ = np.linalg.qr(rng.standard_normal((in_dim, rank)))
        frames[region] = (P.astype(np.float32), Q.astype(np.float32))
    return frames


def lora_frames(rng, regions, shapes, rank):
    frames = {}
    for region in regions:
        out_dim, in_dim = shapes[region]
        Q = rng.standard_normal((in_dim, rank)) / math.sqrt(in_dim)
        frames[region] = (np.zeros((out_dim, rank), np.float32), Q.astype(np.float32))
    return frames


def region_shapes(model, regions):
    return {region: tuple(region_module(model, region).weight.shape) for region in regions}


# ----------------------------------------------------------------------------- batching


def pad_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0


def collate(torch, examples, padding, device):
    width = max(len(example.input_ids) for example in examples)
    ids = torch.full((len(examples), width), padding, dtype=torch.long)
    attention = torch.zeros((len(examples), width), dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for row, example in enumerate(examples):
        n = len(example.input_ids)
        ids[row, :n] = torch.tensor(example.input_ids)
        attention[row, :n] = 1
        target = torch.tensor(example.input_ids)
        mask = torch.tensor(example.target_mask, dtype=torch.bool)
        labels[row, :n] = torch.where(mask, target, torch.full_like(target, -100))
    return ids.to(device), attention.to(device), labels.to(device)


def token_nll(torch, logits, labels):
    """Summed NLL and token count with the usual one-position shift."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid = shift_labels != -100
    return (loss * valid).sum(dim=1), valid.sum(dim=1)


def per_example_nll(torch, model, examples, tokenizer, device, batch_size=8, autocast=None):
    """Summed NLL and target-token count for every example (no gradient)."""
    totals = np.zeros(len(examples))
    counts = np.zeros(len(examples), dtype=np.int64)
    padding = pad_id(tokenizer)
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].input_ids))
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            ids, attention, labels = collate(torch, [examples[i] for i in index], padding, device)
            with autocast() if autocast else contextlib.nullcontext():
                logits = model(input_ids=ids, attention_mask=attention).logits
            nll, count = token_nll(torch, logits, labels)
            totals[index] = nll.double().cpu().numpy()
            counts[index] = count.cpu().numpy()
    return totals, counts


def group_nll(examples, totals, counts, group_of):
    """Aggregate per-example NLL into leakage groups (the bootstrap resampling unit)."""
    groups = {}
    for example, total, count in zip(examples, totals, counts):
        record = example.record_id.split("#w")[0]
        key = group_of.get(record, record)
        entry = groups.setdefault(key, [0.0, 0])
        entry[0] += float(total)
        entry[1] += int(count)
    keys = sorted(groups)
    return keys, np.array([groups[k][0] for k in keys]), np.array([groups[k][1] for k in keys])


def paired_bootstrap(delta_sums, counts, *, replicates=2000, seed=0):
    """Bootstrap over groups of a per-token mean difference. Returns mean and 95% interval."""
    delta_sums = np.asarray(delta_sums, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0:
        return {"mean": None, "low": None, "high": None, "groups": 0}
    rng = np.random.default_rng(seed)
    n = len(counts)
    draws = rng.integers(0, n, size=(replicates, n))
    values = delta_sums[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1)
    return {
        "mean": float(delta_sums.sum() / counts.sum()),
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "groups": int(n),
    }


def kl_to_reference(torch, model, edits, examples, tokenizer, device, batch_size=8):
    """Mean per-token KL(p_base || p_edited) over all non-padding positions."""
    padding = pad_id(tokenizer)
    total, tokens = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            ids, attention, _ = collate(torch, batch, padding, device)
            with edits.off():
                reference = torch.log_softmax(
                    model(input_ids=ids, attention_mask=attention).logits.float(), -1
                )
            current = torch.log_softmax(
                model(input_ids=ids, attention_mask=attention).logits.float(), -1
            )
            kl = (reference.exp() * (reference - current)).sum(-1)
            mask = attention.bool()
            total += float(kl[mask].sum().item())
            tokens += int(mask.sum().item())
    return total / max(tokens, 1)


def generate(
    torch, model, tokenizer, prefixes, *, max_new_tokens, device, seed, temperature, batch_size=16
):
    """Sampled (temperature > 0) or greedy continuations; returns decoded new text.

    Prefixes are left-padded into batches; position ids follow the attention mask, so
    padding does not shift positions. Results are deterministic for a fixed seed,
    device and batch size.
    """
    from ..hf import _generation_config

    config = _generation_config(model, tokenizer, max_new_tokens)
    if temperature > 0:
        config.do_sample = True
        config.temperature = temperature
        config.top_k = 0
        config.top_p = 1.0
    padding = pad_id(tokenizer)
    outputs = [None] * len(prefixes)
    generator_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        with torch.inference_mode():
            for start in range(0, len(prefixes), batch_size):
                chunk = prefixes[start : start + batch_size]
                width = max(len(p) for p in chunk)
                ids = torch.full((len(chunk), width), padding, dtype=torch.long)
                mask = torch.zeros((len(chunk), width), dtype=torch.long)
                for row, prefix in enumerate(chunk):
                    ids[row, width - len(prefix) :] = torch.tensor(prefix)
                    mask[row, width - len(prefix) :] = 1
                out = model.generate(
                    input_ids=ids.to(device),
                    attention_mask=mask.to(device),
                    generation_config=config,
                )
                for row in range(len(chunk)):
                    outputs[start + row] = tokenizer.decode(
                        out[row, width:], skip_special_tokens=True
                    )
    finally:
        torch.random.set_rng_state(generator_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return outputs
