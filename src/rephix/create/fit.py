"""Stage 5 fitting: constrained edits with durable checkpoints and exact resume.

Objective per step (all terms reported separately):

    L_obs (evidence NLL on target tokens)
  + λ_retain · KL(p_base || p_edit) on retain contexts
  + λ_prior  · ||Δ||²
  + λ_instr  · KL(p_base(· | C, x) || p_edit(· | x)) on evidence contexts (only with C)

Checkpoints hold edit tensors, optimizer state, the sampler position, Python/NumPy/Torch
RNG states, the step counter and the best validation state. Model selection uses the
frozen validation split only.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np

from .modeling import RegionEdits, collate, pad_id, per_example_nll, token_nll


class Budget(Exception):
    """Raised at a durable boundary when the per-invocation time budget is exhausted."""


def _rng_state(torch):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(torch, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class Sampler:
    """Deterministic epoch permutations; position is part of the checkpoint."""

    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.position = 0, 0
        self._order = self._permutation()

    def _permutation(self):
        return np.random.default_rng([self.seed, self.epoch]).permutation(self.size).tolist()

    def next(self):
        if self.position + self.batch_size > self.size:
            self.epoch += 1
            self.position = 0
            self._order = self._permutation()
        batch = self._order[self.position : self.position + self.batch_size]
        self.position += self.batch_size
        return batch

    def state(self):
        return {"epoch": self.epoch, "position": self.position}

    def load(self, state):
        self.epoch, self.position = state["epoch"], state["position"]
        self._order = self._permutation()


def _instruction_batch(torch, examples, instruction_ids, padding, device):
    """Teacher inputs: the instruction tokens prepended to each evidence sequence."""
    shifted = []
    for example in examples:
        shifted.append(
            type(example)(
                example.record_id,
                list(instruction_ids) + list(example.input_ids),
                [0] * len(instruction_ids) + list(example.target_mask),
                example.kind,
            )
        )
    return collate(torch, shifted, padding, device)


def fit_edits(
    ctx, spec, frames, *, checkpoint_dir: Path, progress=None, max_seconds=None, interrupt=None
):
    """Fit (or resume) one edit configuration. Returns (edits with best state, report)."""
    from ..hf import _atomic_torch_save

    torch = ctx.torch
    model = ctx.model
    tokenizer = ctx.tokenizer
    device = ctx.device
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "checkpoint.pt"
    torch.manual_seed(spec["seed"])
    random.seed(spec["seed"])
    np.random.seed(spec["seed"] % (2**32))
    edits = RegionEdits(torch, model, frames, train_frames=spec["train_frames"], init=spec["init"])
    parameters = list(edits.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=spec["lr"], weight_decay=0.0)
    train = ctx.train_examples
    retain = ctx.retain_examples
    sampler = Sampler(len(train), min(spec["batch_size"], len(train)), spec["seed"])
    retain_sampler = Sampler(
        len(retain), min(spec["retain_batch_size"], len(retain)), spec["seed"] + 1
    )
    padding = pad_id(tokenizer)
    step = 0
    history = []
    best = {"validation_nll": None, "step": 0, "state": edits.state()}
    elapsed_before = 0.0
    if checkpoint_file.exists():
        try:
            saved = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(
                f"{checkpoint_file}: checkpoint is unreadable or corrupted ({exc}); remove it "
                "to restart this fit from the beginning"
            ) from None
        if saved.get("spec") != spec:
            raise ValueError(
                f"{checkpoint_file}: checkpoint belongs to different fit settings; refusing "
                "to continue with changed settings"
            )
        edits.load_state(saved["edits"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.load(saved["sampler"])
        retain_sampler.load(saved["retain_sampler"])
        _set_rng_state(torch, saved["rng"])
        step = saved["step"]
        history = saved["history"]
        best = saved["best"]
        elapsed_before = saved.get("elapsed_seconds", 0.0)

    def save():
        _atomic_torch_save(
            torch,
            {
                "spec": spec,
                "edits": edits.state(),
                "optimizer": optimizer.state_dict(),
                "sampler": sampler.state(),
                "retain_sampler": retain_sampler.state(),
                "rng": _rng_state(torch),
                "step": step,
                "history": history,
                "best": best,
                "elapsed_seconds": elapsed_before + (time.monotonic() - started),
            },
            checkpoint_file,
        )

    def validate():
        totals, counts = per_example_nll(
            torch, model, ctx.validation_examples, tokenizer, device, autocast=ctx.autocast
        )
        return float(totals.sum() / max(counts.sum(), 1))

    started = time.monotonic()
    total_steps = spec["steps"]
    warmup = max(1, total_steps // 20)
    if best["validation_nll"] is None:
        best["validation_nll"] = validate()
        history.append({"step": 0, "validation_nll": best["validation_nll"]})
    model.train(False)  # dropout off; base weights are frozen
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        while step < total_steps:
            if interrupt is not None:
                interrupt()
            lr_scale = (
                min(1.0, (step + 1) / warmup)
                * 0.5
                * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))
            )
            for group in optimizer.param_groups:
                group["lr"] = spec["lr"] * max(lr_scale, 0.05)
            batch = [train[i] for i in sampler.next()]
            ids, attention, labels = collate(torch, batch, padding, device)
            with ctx.autocast():
                logits = model(input_ids=ids, attention_mask=attention).logits
            nll, count = token_nll(torch, logits, labels)
            obs = nll.sum() / count.sum().clamp_min(1)
            loss = obs
            terms = {"obs": float(obs.item())}
            if spec["retain_weight"] > 0:
                rbatch = [retain[i] for i in retain_sampler.next()]
                rids, rattention, _ = collate(torch, rbatch, padding, device)
                with torch.no_grad(), edits.off(), ctx.autocast():
                    reference = torch.log_softmax(
                        model(input_ids=rids, attention_mask=rattention).logits.float(), -1
                    )
                with ctx.autocast():
                    current = torch.log_softmax(
                        model(input_ids=rids, attention_mask=rattention).logits.float(), -1
                    )
                kl = (reference.exp() * (reference - current)).sum(-1)
                retain_kl = (kl * rattention).sum() / rattention.sum()
                loss = loss + spec["retain_weight"] * retain_kl
                terms["retain_kl"] = float(retain_kl.item())
            if spec["prior_weight"] > 0:
                prior = edits.l2()
                loss = loss + spec["prior_weight"] * prior
                terms["prior_l2"] = float(prior.item())
            if spec.get("instruction_weight", 0) > 0 and ctx.instruction_ids:
                tids, tattention, tlabels = _instruction_batch(
                    torch, batch, ctx.instruction_ids, padding, device
                )
                offset = len(ctx.instruction_ids)
                with torch.no_grad(), edits.off(), ctx.autocast():
                    teacher = model(input_ids=tids, attention_mask=tattention).logits.float()
                teacher = torch.log_softmax(teacher[:, offset:], -1)
                student = torch.log_softmax(logits.float(), -1)
                width = min(teacher.shape[1], student.shape[1])
                mask = (labels[:, :width] != -100).float()
                kl = (teacher[:, :width].exp() * (teacher[:, :width] - student[:, :width])).sum(-1)
                instr = (kl * mask).sum() / mask.sum().clamp_min(1)
                loss = loss + spec["instruction_weight"] * instr
                terms["instruction_kl"] = float(instr.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            step += 1
            if step % spec["eval_every"] == 0 or step == total_steps:
                value = validate()
                entry = {"step": step, "validation_nll": value, **terms}
                history.append(entry)
                if value < best["validation_nll"]:
                    best = {"validation_nll": value, "step": step, "state": edits.state()}
                save()
                if progress:
                    progress(step, total_steps, entry)
                if max_seconds is not None and time.monotonic() - started > max_seconds:
                    raise Budget()
            elif step % spec["save_every"] == 0:
                save()
    except (Budget, KeyboardInterrupt):
        save()
        raise
    except Exception as exc:
        oom = getattr(torch, "OutOfMemoryError", None)
        if oom is not None and isinstance(exc, oom):
            raise RuntimeError(
                f"out of accelerator memory at step {step}; the last checkpoint in "
                f"{checkpoint_dir} is intact. Resume with a smaller --batch-size, "
                "--max-tokens or --profile"
            ) from None
        raise
    save()
    edits.load_state(best["state"])
    report = {
        "name": spec["name"],
        "method": spec["method"],
        "regions": edits.regions,
        "rank": spec["rank"],
        "trainable_parameters": edits.trainable_count(),
        "steps": step,
        "best_step": best["step"],
        "best_validation_nll": best["validation_nll"],
        "initial_validation_nll": history[0]["validation_nll"],
        "history": history,
        "elapsed_seconds": elapsed_before + (time.monotonic() - started),
        "spec": spec,
    }
    return edits, report


def fit_full(ctx, spec, *, checkpoint_dir: Path, progress=None, max_seconds=None):
    """Ordinary supervised fine-tuning of every parameter: a behavioral baseline only."""
    from ..hf import _atomic_torch_save

    torch = ctx.torch
    model = ctx.model
    tokenizer = ctx.tokenizer
    device = ctx.device
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "checkpoint.pt"
    torch.manual_seed(spec["seed"])
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=0.0)
    train = ctx.train_examples
    sampler = Sampler(len(train), min(spec["batch_size"], len(train)), spec["seed"])
    padding = pad_id(tokenizer)
    step, history = 0, []
    best_value, best_step, best_state = None, 0, None
    if checkpoint_file.exists():
        saved = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if saved.get("spec") != spec:
            raise ValueError(f"{checkpoint_file}: checkpoint belongs to different settings")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.load(saved["sampler"])
        _set_rng_state(torch, saved["rng"])
        step, history = saved["step"], saved["history"]
        best_value, best_step, best_state = (
            saved["best_value"],
            saved["best_step"],
            saved["best_state"],
        )

    def validate():
        model.eval()
        totals, counts = per_example_nll(
            torch, model, ctx.validation_examples, tokenizer, device, autocast=ctx.autocast
        )
        return float(totals.sum() / max(counts.sum(), 1))

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_value is None:
        best_value, best_state = validate(), snapshot()
        history.append({"step": 0, "validation_nll": best_value})
    started = time.monotonic()

    def save():
        _atomic_torch_save(
            torch,
            {
                "spec": spec,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampler": sampler.state(),
                "rng": _rng_state(torch),
                "step": step,
                "history": history,
                "best_value": best_value,
                "best_step": best_step,
                "best_state": best_state,
            },
            checkpoint_file,
        )

    total_steps = spec["steps"]
    try:
        while step < total_steps:
            lr = spec["lr"] * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            for group in optimizer.param_groups:
                group["lr"] = max(lr, spec["lr"] * 0.05)
            batch = [train[i] for i in sampler.next()]
            ids, attention, labels = collate(torch, batch, padding, device)
            with ctx.autocast():
                logits = model(input_ids=ids, attention_mask=attention).logits
            nll, count = token_nll(torch, logits, labels)
            loss = nll.sum() / count.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % spec["eval_every"] == 0 or step == total_steps:
                value = validate()
                history.append({"step": step, "validation_nll": value, "obs": float(loss)})
                if value < best_value:
                    best_value, best_step, best_state = value, step, snapshot()
                save()
                if progress:
                    progress(step, total_steps, history[-1])
                if max_seconds is not None and time.monotonic() - started > max_seconds:
                    raise Budget()
    except (Budget, KeyboardInterrupt):
        save()
        raise
    save()
    model.load_state_dict(best_state)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return {
        "name": spec["name"],
        "method": "sft",
        "regions": "all parameters",
        "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
        "steps": step,
        "best_step": best_step,
        "best_validation_nll": best_value,
        "initial_validation_nll": history[0]["validation_nll"],
        "history": history,
        "spec": spec,
    }
