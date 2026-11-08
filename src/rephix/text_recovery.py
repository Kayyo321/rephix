"""Text-only recovery of softmax-affine models with a declared, public feature map.

Observation contract
--------------------
The recovery process may send prompt strings to a target and read the returned
strings. Nothing else crosses the boundary: no logits, probabilities, token IDs,
hidden states, weights or logit-bias controls. Each query asks for ``n`` independent
single-step continuations sampled at the target's documented temperature.

Permitted prior knowledge is explicit in a feature specification:

* ``ngram``: a public token list and context order ``k``. The target class is the
  additive log-linear k-gram model ``logits = b + sum_i E_i[:, x_-i]``.
* ``hf-backbone``: a public Transformers backbone and the public label strings of a
  private softmax-affine head ``logits = W h(x) + b`` over those labels.

Identification (proved in docs/TEXT_RECOVERY.md): with features ``a(x) = [phi(x); 1]``
and a queried design whose span equals the span of all valid inputs ``S``, two
parameter matrices give identical output distributions on every valid input iff
their class-centred restrictions to ``S`` agree. The canonical parameterization is
``Theta_c = C Theta P_S`` with ``C = I - 11^T/V``. Sampled strings estimate it with
the multinomial-logit maximum-likelihood estimator, which is consistent and
asymptotically normal for in-class targets. A parametric-bootstrap deviance test
separates in-class recovery from out-of-class approximation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .io import atomic_save_npz, atomic_write_text, file_sha256, write_json

PROTOCOL = "rephix-text-oracle/1"
EXPORT_FORMAT = "rephix-text-model/1"
STATUS_RECOVERED = "recovered_in_class"
STATUS_MISSPECIFIED = "misspecified_approximation"
STATUS_INSUFFICIENT = "insufficient_evidence"
STATUS_DETERMINISTIC = "deterministic_outputs_not_identifiable"


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Public feature maps (permitted prior knowledge)


class NgramFeatures:
    """One-hot features of the last ``order`` tokens of a known, injective vocabulary."""

    kind = "ngram"

    def __init__(self, vocabulary, order, separator="", prompt_length=None):
        vocabulary = [str(token) for token in vocabulary]
        if len(vocabulary) < 2 or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("ngram vocabulary needs at least two distinct token strings")
        if type(order) is not int or order < 1:
            raise ValueError("ngram order must be a positive integer")
        if separator == "" and any(len(token) != 1 for token in vocabulary):
            raise ValueError("without a separator every ngram token must be one character")
        if separator and any(separator in token or not token for token in vocabulary):
            raise ValueError("ngram tokens must be nonempty and must not contain the separator")
        self.vocabulary = vocabulary
        self.order = order
        self.separator = separator
        self.index = {token: position for position, token in enumerate(vocabulary)}
        self.width = order * len(vocabulary) + 1
        self.prompt_length = order if prompt_length is None else int(prompt_length)
        if self.prompt_length < order:
            raise ValueError("ngram prompt_length must be at least the order")

    def spec(self):
        return {
            "kind": self.kind,
            "vocabulary": self.vocabulary,
            "order": self.order,
            "separator": self.separator,
            "prompt_length": self.prompt_length,
        }

    @property
    def classes(self):
        return list(self.vocabulary)

    def tokenize(self, prompt):
        tokens = list(prompt) if self.separator == "" else prompt.split(self.separator)
        try:
            return [self.index[token] for token in tokens]
        except KeyError as exc:
            raise ValueError(f"prompt contains a token outside the vocabulary: {exc}") from None

    def render(self, token_ids):
        return self.separator.join(self.vocabulary[int(token)] for token in token_ids)

    def features(self, prompts):
        size = len(self.vocabulary)
        matrix = np.zeros((len(prompts), self.width))
        for row, prompt in enumerate(prompts):
            tokens = self.tokenize(prompt)
            if len(tokens) < self.order:
                raise ValueError(f"prompt needs at least {self.order} tokens: {prompt!r}")
            for position in range(self.order):
                matrix[row, position * size + tokens[-1 - position]] = 1.0
            matrix[row, -1] = 1.0
        return matrix

    def span_dimension(self):
        # Each one-hot block sums to the bias coordinate: k independent constraints.
        return self.order * (len(self.vocabulary) - 1) + 1

    def design_contexts(self, count, rng):
        size = len(self.vocabulary)
        return [self.render(rng.integers(0, size, self.prompt_length)) for _ in range(count)]


class BackboneFeatures:
    """Final post-norm hidden state of a public Transformers backbone at the last token."""

    kind = "hf-backbone"

    def __init__(self, model, labels, device="cpu", dtype="float32"):
        labels = [str(label) for label in labels]
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("hf-backbone needs at least two distinct public label strings")
        self.model_path = str(Path(model).resolve())
        self.labels = labels
        self.device_name = device
        self.dtype_name = dtype
        self._model = None

    def spec(self):
        return {
            "kind": self.kind,
            "model": self.model_path,
            "labels": self.labels,
            "device": self.device_name,
            "dtype": self.dtype_name,
        }

    @property
    def classes(self):
        return list(self.labels)

    def _load(self):
        if self._model is None:
            from .hf import _deps, _device, _load_tokenizer

            torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer = _deps()
            config = AutoConfig.from_pretrained(self.model_path)
            self._torch = torch
            self._device = _device(torch, self.device_name)
            self._tokenizer = _load_tokenizer(AutoTokenizer, self.model_path, model_config=config)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=getattr(torch, self.dtype_name)
            )
            self._model = model.to(self._device).eval()
        return self._model

    def features(self, prompts):
        model = self._load()
        torch = self._torch
        rows = []
        with torch.inference_mode():
            for prompt in prompts:
                ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"].to(self._device)
                hidden = model.base_model(input_ids=ids).last_hidden_state[0, -1]
                rows.append(hidden.double().cpu().numpy())
        matrix = np.asarray(rows)
        return np.concatenate([matrix, np.ones((len(prompts), 1))], axis=1)

    def span_dimension(self):
        return None  # Determined numerically from the queried design.

    @property
    def width(self):
        return int(self._load().config.hidden_size) + 1


def feature_map(spec):
    kind = spec.get("kind")
    if kind == "ngram":
        return NgramFeatures(
            spec["vocabulary"], spec["order"], spec.get("separator", ""), spec.get("prompt_length")
        )
    if kind == "hf-backbone":
        return BackboneFeatures(
            spec["model"], spec["labels"], spec.get("device", "cpu"), spec.get("dtype", "float32")
        )
    raise ValueError("feature kind must be ngram or hf-backbone")


# ----------------------------------------------------------------------------
# String-only oracle


class CommandOracle:
    """A target process speaking JSON lines: {"prompt", "n"} -> {"outputs": [str]}.

    Only strings are accepted from the target. Any other field is rejected so a
    misconfigured target cannot leak privileged signals into the estimator.
    """

    def __init__(self, argv, timeout=600):
        if not argv or not all(isinstance(part, str) for part in argv):
            raise ValueError("target command must be a nonempty list of strings")
        self.argv = list(argv)
        self.timeout = timeout
        self._process = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        import queue
        import threading

        self._lines = queue.Queue()

        def pump(stream=self._process.stdout, lines=self._lines):
            for line in stream:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        hello = self._receive()
        if hello.get("protocol") != PROTOCOL or set(hello) != {"protocol"}:
            self.close()
            raise RuntimeError(f"target did not announce {PROTOCOL}")

    def _receive(self):
        import queue

        try:
            line = self._lines.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError(f"target did not reply within {self.timeout} s") from None
        if line is None:
            raise RuntimeError(f"target exited (status {self._process.poll()})")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError:
            raise RuntimeError("target reply is not a JSON line") from None
        if not isinstance(reply, dict):
            raise RuntimeError("target reply must be a JSON object")
        return reply

    def sample(self, prompt, n):
        self._process.stdin.write(_canonical_json({"prompt": prompt, "n": int(n)}) + "\n")
        self._process.stdin.flush()
        reply = self._receive()
        if set(reply) != {"outputs"} or not isinstance(reply["outputs"], list):
            raise RuntimeError("target reply must contain only an outputs list")
        outputs = reply["outputs"]
        if len(outputs) != n or not all(isinstance(item, str) for item in outputs):
            raise RuntimeError("target must return exactly n output strings")
        return outputs

    def close(self):
        if self._process.poll() is None:
            try:
                self._process.stdin.close()
                self._process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.wait()


# ----------------------------------------------------------------------------
# Durable transcript with resume


class Transcript:
    """Append-only JSONL record of every prompt and returned string."""

    def __init__(self, path, plan):
        self.path = Path(path)
        self.plan_sha256 = _sha256_text(_canonical_json(plan))
        self.records = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                lines = [line for line in handle.read().split("\n") if line]
            if not lines:
                raise ValueError(f"{self.path}: empty transcript")
            header = json.loads(lines[0])
            if header.get("plan_sha256") != self.plan_sha256:
                raise ValueError(
                    f"{self.path}: transcript belongs to a different query plan; "
                    "use a new transcript path"
                )
            for number, line in enumerate(lines[1:], start=2):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if number != len(lines):
                        raise ValueError(
                            f"{self.path}: unreadable record on line {number} before the end"
                        ) from None
                    break  # Torn final write from an interrupted run; requery it.
                digest = record.pop("sha256", None)
                if digest != _sha256_text(_canonical_json(record)):
                    raise ValueError(f"{self.path}: transcript record checksum mismatch")
                self.records[record["context"]] = record
            # Rewrite without any torn tail so appends stay line-aligned.
            self._rewrite(header)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rewrite({"format": "rephix-text-transcript/1", "plan_sha256": self.plan_sha256})

    def _rewrite(self, header):
        lines = [_canonical_json(header)]
        for record in sorted(self.records.values(), key=lambda item: item["context"]):
            lines.append(_canonical_json({**record, "sha256": _record_digest(record)}))
        atomic_write_text(self.path, "\n".join(lines) + "\n")

    def append(self, record):
        line = _canonical_json({**record, "sha256": _record_digest(record)})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[record["context"]] = record


def _record_digest(record):
    return _sha256_text(_canonical_json(record))


# ----------------------------------------------------------------------------
# Estimation


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _log_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def span_basis(features, expected_dimension=None, tolerance=1e-9):
    """Orthonormal basis of the queried feature span and its conditioning."""
    _, singular, vt = np.linalg.svd(features, full_matrices=False)
    rank = int((singular > tolerance * singular[0]).sum())
    basis = vt[:rank].T
    report = {
        "rank": rank,
        "expected_rank": expected_dimension,
        "feature_width": int(features.shape[1]),
        "largest_singular_value": float(singular[0]),
        "smallest_retained_singular_value": float(singular[rank - 1]),
        "condition_number": float(singular[0] / singular[rank - 1]),
    }
    return basis, report


def fit_multinomial(z, counts, *, ridge=1e-6, start=None, max_newton=60, tolerance=1e-9):
    """Penalized multinomial-logit MLE by matrix-free Newton-CG.

    ``z`` is (contexts, r) reduced features, ``counts`` is (contexts, V). The ridge
    makes the objective strictly convex; because the likelihood is invariant to
    adding ``1 a^T``, the penalized optimum is automatically class-centred.
    """
    contexts, width = z.shape
    classes = counts.shape[1]
    totals = counts.sum(axis=1)
    gamma = np.zeros((classes, width)) if start is None else start.copy()

    def objective(parameters):
        log_p = _log_softmax(z @ parameters.T)
        return -(counts * log_p).sum() + 0.5 * ridge * (parameters**2).sum()

    value = objective(gamma)
    for iteration in range(max_newton):
        probabilities = _softmax(z @ gamma.T)
        gradient = (totals[:, None] * probabilities - counts).T @ z + ridge * gamma

        def hessian(direction, p=probabilities):
            u = z @ direction.T
            s = p * u - p * (p * u).sum(axis=1, keepdims=True)
            return (totals[:, None] * s).T @ z + ridge * direction

        # Conjugate gradient for the Newton direction.
        step = np.zeros_like(gamma)
        residual = -gradient
        direction = residual.copy()
        rho = (residual * residual).sum()
        gradient_norm = np.sqrt((gradient * gradient).sum())
        for _ in range(250):
            hd = hessian(direction)
            curvature = (direction * hd).sum()
            if curvature <= 0:
                break
            alpha = rho / curvature
            step += alpha * direction
            residual -= alpha * hd
            new_rho = (residual * residual).sum()
            if np.sqrt(new_rho) <= min(0.1, np.sqrt(gradient_norm)) * gradient_norm * 1e-3:
                break
            direction = residual + (new_rho / rho) * direction
            rho = new_rho
        # The Newton decrement is in log-likelihood units: an absolute tolerance keeps
        # precision independent of the (possibly huge) total sample count.
        decrement = -(gradient * step).sum()
        if decrement / 2 <= tolerance:
            return gamma, {"newton_iterations": iteration, "converged": True, "objective": value}
        scale = 1.0
        accepted = False
        while scale > 1e-8:
            candidate = gamma + scale * step
            candidate_value = objective(candidate)
            if candidate_value <= value - 0.25 * scale * decrement:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            # Floating-point resolution of the objective reached; accept the full step
            # only if it does not increase the objective.
            candidate = gamma + step
            if objective(candidate) <= value:
                gamma = candidate
            return gamma, {
                "newton_iterations": iteration,
                "converged": bool(decrement / 2 <= 1e-6 * max(1.0, abs(value)) * 1e-6),
                "objective": value,
                "stopped_at_precision_limit": True,
            }
        gamma, value = candidate, candidate_value
    return gamma, {"newton_iterations": max_newton, "converged": False, "objective": value}


def fisher_information(z, counts, gamma):
    """Fisher information of the multinomial logit, shape (V*r, V*r).

    Assembled per class pair as a weighted Gram matrix so memory stays O(V^2 r^2)
    regardless of the number of contexts.
    """
    classes, width = gamma.shape
    probabilities = _softmax(z @ gamma.T)
    totals = counts.sum(axis=1)
    information = np.zeros((classes, width, classes, width))
    for a in range(classes):
        for c in range(a, classes):
            weight = totals * probabilities[:, a] * ((a == c) - probabilities[:, c])
            block = (z * weight[:, None]).T @ z
            information[a, :, c, :] = block
            information[c, :, a, :] = block.T
    return information.reshape(classes * width, classes * width)


def fisher_covariance(z, counts, gamma, limit=6000):
    """Inverse Fisher information on the class-centred subspace, when small enough."""
    classes, width = gamma.shape
    if classes * width > limit:
        return None
    information = fisher_information(z, counts, gamma)
    centre = np.eye(classes) - 1.0 / classes
    projector = np.kron(centre, np.eye(width))
    return projector @ np.linalg.pinv(information, hermitian=True) @ projector


def deviance(counts, probabilities):
    with np.errstate(divide="ignore", invalid="ignore"):
        totals = counts.sum(axis=1, keepdims=True)
        ratio = np.where(counts > 0, counts / (totals * probabilities), 1.0)
        return float(2.0 * (counts * np.log(ratio)).sum())


def plugin_estimate(z, counts, smoothing=0.5):
    """Baseline: smoothed empirical log-frequencies projected by least squares."""
    totals = counts.sum(axis=1, keepdims=True)
    log_frequency = np.log((counts + smoothing) / (totals + smoothing * counts.shape[1]))
    centred = log_frequency - log_frequency.mean(axis=1, keepdims=True)
    solution, *_ = np.linalg.lstsq(z, centred, rcond=None)
    return solution.T


def bootstrap_misspecification(z, counts, gamma, replicates, seed, ridge, held_out=None):
    """Parametric-bootstrap test of the declared class, two-sided.

    The statistic is the deviance against the per-context saturated multinomial on the
    fitted contexts plus, when given, held-out validation contexts that the fit never
    saw. Each replicate redraws both from the fitted law and refits on the redrawn
    training counts, so estimation error is part of the null distribution. A large
    statistic rejects the class; an improbably small one means the outputs are not
    independent samples (for example rounded expected counts).
    """
    rng = np.random.default_rng(seed)
    probabilities = _softmax(z @ gamma.T)
    totals = counts.sum(axis=1).astype(int)
    if held_out is not None:
        z_val, counts_val = held_out
        p_val = _softmax(z_val @ gamma.T)
        totals_val = counts_val.sum(axis=1).astype(int)
    observed = deviance(counts, probabilities)
    if held_out is not None:
        observed += deviance(counts_val, p_val)
    upper = lower = 0
    simulated = []
    for _ in range(replicates):
        draw = np.stack([rng.multinomial(n, p) for n, p in zip(totals, probabilities)]).astype(
            float
        )
        refit, _ = fit_multinomial(z, draw, ridge=ridge, start=gamma, max_newton=30)
        value = deviance(draw, _softmax(z @ refit.T))
        if held_out is not None:
            draw_val = np.stack([rng.multinomial(n, p) for n, p in zip(totals_val, p_val)]).astype(
                float
            )
            value += deviance(draw_val, _softmax(z_val @ refit.T))
        simulated.append(value)
        upper += value >= observed
        lower += value <= observed
    return {
        "statistic": "deviance against per-context saturated multinomials"
        + (" on fitted and held-out validation contexts" if held_out is not None else ""),
        "includes_validation_contexts": held_out is not None,
        "observed_deviance": observed,
        "bootstrap_replicates": replicates,
        "bootstrap_median_deviance": float(np.median(simulated)) if simulated else None,
        "p_value": (1 + upper) / (replicates + 1),
        "p_value_lower_tail": (1 + lower) / (replicates + 1),
    }


# ----------------------------------------------------------------------------
# Recovery workflow


def _parse_outputs(outputs, classes):
    index = {label: position for position, label in enumerate(classes)}
    counts = np.zeros(len(classes))
    unparsed = 0
    for output in outputs:
        position = index.get(output)
        if position is None:
            unparsed += 1
        else:
            counts[position] += 1
    return counts, unparsed


def recover(
    *,
    features_spec,
    target_argv,
    output,
    transcript_path,
    contexts=None,
    contexts_file=None,
    samples_per_context=200,
    validation_contexts=0,
    max_queries=None,
    seed=0,
    ridge=1e-6,
    bootstrap=199,
    alpha=0.01,
    max_relative_standard_error=0.1,
    span_tolerance=None,
):
    """Query a string-only target, fit the canonical parameters and export a model."""
    started = time.perf_counter()
    if bootstrap and (bootstrap + 1) * alpha / 2 < 1:
        # Each tail is tested at alpha/2 and a Monte Carlo p-value is at least 1/(B+1);
        # fewer replicates could never reject.
        raise ValueError("--bootstrap must satisfy (bootstrap + 1) * alpha / 2 >= 1")
    mapping = feature_map(features_spec)
    rng = np.random.default_rng(seed)
    if contexts_file:
        # Split on newlines only: prompts may contain other Unicode line separators.
        pool = [
            line.rstrip("\r")
            for line in Path(contexts_file).read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        if len(set(pool)) != len(pool):
            raise ValueError("context file contains duplicate prompts")
        order = rng.permutation(len(pool))
        if contexts is None:
            contexts = len(pool) - validation_contexts
        if contexts < 1:
            raise ValueError("need at least one training context")
        wanted = contexts + validation_contexts
        if wanted > len(pool):
            raise ValueError("context file has fewer prompts than requested")
        selected = [pool[i] for i in order[:wanted]]
    elif mapping.kind == "ngram":
        if contexts is None:
            contexts = 4 * mapping.span_dimension()
        selected, seen = [], set()
        attempts = 0
        while len(selected) < contexts + validation_contexts:
            attempts += 1
            if attempts > 100 * (contexts + validation_contexts):
                raise ValueError("cannot draw enough distinct contexts for this vocabulary/order")
            (prompt,) = mapping.design_contexts(1, rng)
            if prompt not in seen:
                seen.add(prompt)
                selected.append(prompt)
    else:
        raise ValueError("hf-backbone recovery needs --contexts-file with public prompts")
    validation = selected[len(selected) - validation_contexts :] if validation_contexts else []
    training = selected[: len(selected) - validation_contexts]
    planned_queries = len(selected) * samples_per_context
    if max_queries is not None and planned_queries > max_queries:
        raise ValueError(
            f"plan needs {planned_queries} sampled outputs, above --max-queries {max_queries}"
        )
    plan = {
        "target_argv": list(target_argv),
        "features": mapping.spec(),
        "training": training,
        "validation": validation,
        "samples_per_context": samples_per_context,
        "protocol": PROTOCOL,
    }
    transcript = Transcript(transcript_path, plan)
    resumed = len(transcript.records)
    oracle = None
    try:
        for position, prompt in enumerate(selected):
            if position in transcript.records:
                continue
            if oracle is None:
                oracle = CommandOracle(target_argv)
            outputs = oracle.sample(prompt, samples_per_context)
            transcript.append({"context": position, "prompt": prompt, "outputs": outputs})
    finally:
        if oracle is not None:
            oracle.close()

    classes = mapping.classes
    counts = np.zeros((len(selected), len(classes)))
    unparsed = 0
    for position in range(len(selected)):
        record = transcript.records[position]
        if record["prompt"] != selected[position]:
            raise ValueError("transcript prompt does not match the query plan")
        counts[position], bad = _parse_outputs(record["outputs"], classes)
        unparsed += bad
    if unparsed:
        raise ValueError(
            f"{unparsed} target outputs are not exactly one public class string; the target "
            "violates the declared output contract, so parameters are not identifiable here"
        )

    design = mapping.features(selected)
    train_design, train_counts = design[: len(training)], counts[: len(training)]
    expected = mapping.span_dimension()
    if span_tolerance is None:
        # One-hot ngram features are exact. Backbone features carry float32 rounding
        # (~1e-7 relative), so directions below 1e-5 of the largest singular value
        # (e.g. the hyperplane normal of a LayerNorm with bias) are unidentified.
        span_tolerance = 1e-9 if mapping.kind == "ngram" else 1e-5
    basis, span = span_basis(train_design, expected, tolerance=span_tolerance)
    span["tolerance"] = span_tolerance
    z = train_design @ basis
    deterministic = bool(np.all(train_counts.max(axis=1) == train_counts.sum(axis=1)))

    gamma, optimisation = fit_multinomial(z, train_counts, ridge=ridge)
    baseline = plugin_estimate(z, train_counts)
    covariance = fisher_covariance(z, train_counts, gamma)
    standard_error = None if covariance is None else np.sqrt(np.clip(np.diag(covariance), 0, None))
    held_out = None
    if validation:
        held_out = (design[len(training) :] @ basis, counts[len(training) :])
    misspecification = (
        bootstrap_misspecification(
            z, train_counts, gamma, bootstrap, seed + 1, ridge, held_out=held_out
        )
        if bootstrap
        else None
    )
    theta = gamma @ basis.T
    relative_se = (
        None
        if standard_error is None
        else float(np.sqrt((standard_error**2).sum()) / max(np.linalg.norm(gamma), 1e-300))
    )

    validation_report = None
    if validation:
        vz = design[len(training) :]
        vcounts = counts[len(training) :]
        log_p = _log_softmax(vz @ theta.T)
        empirical = vcounts / vcounts.sum(axis=1, keepdims=True)
        validation_report = {
            "contexts": len(validation),
            "samples": int(vcounts.sum()),
            "mean_negative_log_likelihood": float(-(vcounts * log_p).sum() / vcounts.sum()),
            "mean_total_variation_to_empirical": float(
                0.5 * np.abs(empirical - np.exp(log_p)).sum(axis=1).mean()
            ),
            "outside_training_span": float(
                np.linalg.norm(vz - (vz @ basis) @ basis.T) / max(np.linalg.norm(vz), 1e-300)
            ),
        }

    reasons = []
    if deterministic:
        status = STATUS_DETERMINISTIC
        reasons.append(
            "every context returned one string for all samples; greedy or deterministic "
            "outputs identify at most a decision rule, not softmax parameters"
        )
    elif expected is not None and span["rank"] < expected:
        status = STATUS_INSUFFICIENT
        reasons.append("queried contexts do not span the valid input feature space")
    elif not optimisation["converged"]:
        status = STATUS_INSUFFICIENT
        reasons.append("the maximum-likelihood optimisation did not converge")
    elif misspecification is not None and misspecification["p_value"] <= alpha / 2:
        status = STATUS_MISSPECIFIED
        reasons.append(
            "deviance test rejects the declared softmax-affine class; the export is an "
            "approximation, not recovery of the target's parameters"
        )
    elif misspecification is None:
        status = STATUS_INSUFFICIENT
        reasons.append("misspecification test disabled; class membership is untested")
    elif misspecification["p_value_lower_tail"] <= alpha / 2:
        status = STATUS_INSUFFICIENT
        reasons.append(
            "outputs are under-dispersed relative to independent sampling; the target "
            "does not satisfy the sampling contract"
        )
    elif relative_se is None or relative_se > max_relative_standard_error:
        status = STATUS_INSUFFICIENT
        reasons.append("estimated canonical-parameter standard error exceeds the threshold")
    else:
        status = STATUS_RECOVERED
        if not validation:
            reasons.append("no validation contexts: the class test covers fitted contexts only")
    if (
        status == STATUS_RECOVERED
        and validation_report
        and validation_report["outside_training_span"] > 10 * span_tolerance
    ):
        status = STATUS_INSUFFICIENT
        reasons.append("validation contexts leave the identified feature span")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"canonical_theta": theta, "gamma": gamma, "basis": basis}
    if standard_error is not None:
        arrays["gamma_standard_error"] = standard_error.reshape(gamma.shape)
    atomic_save_npz(output / "parameters.npz", **arrays)
    atomic_save_npz(output / "baseline.npz", canonical_theta=baseline @ basis.T)
    report = {
        "format": EXPORT_FORMAT,
        "status": status,
        "reasons": reasons,
        "observation_contract": {
            "received": "prompt strings sent and returned strings only",
            "forbidden": [
                "logits",
                "probabilities",
                "token ids from the target",
                "hidden states",
                "weights",
                "logit bias",
            ],
            "query": "independent single-step samples at the target's documented temperature",
            "target_command": target_argv,
        },
        "prior_knowledge": mapping.spec(),
        "guarantee": (
            "Conditional on the target belonging to the declared class, the exported "
            "canonical parameters estimate the target's and determine its output law on "
            "every input in the identified span. The class test can reject membership but "
            "cannot prove it: a target that departs from the class only on inputs that "
            "were never queried is indistinguishable here."
        ),
        "canonical_parameterization": (
            "Theta_c = C Theta P_S: class-centred, restricted to the span S of the queried "
            "feature vectors [phi(x); 1]; logits(x) = Theta_c [phi(x); 1] up to a per-input "
            "constant for every x whose features lie in S"
        ),
        "classes": classes,
        "budget": {
            "training_contexts": len(training),
            "validation_contexts": len(validation),
            "samples_per_context": samples_per_context,
            "sampled_outputs": int(counts.sum()),
            "resumed_contexts": resumed,
        },
        "design": span,
        "optimisation": optimisation,
        "estimated_relative_standard_error": relative_se,
        "misspecification_test": misspecification,
        "alpha": alpha,
        "validation": validation_report,
        "transcript": {"path": str(transcript_path), "sha256": file_sha256(transcript_path)},
        "parameters_sha256": file_sha256(output / "parameters.npz"),
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "manifest.json", report)
    return report


# ----------------------------------------------------------------------------
# Standalone reconstructed model


class RecoveredTextModel:
    """Runs an exported recovery without the target or the recovery transcript."""

    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != EXPORT_FORMAT:
            raise ValueError("not a Rephix text-model export")
        if file_sha256(directory / "parameters.npz") != self.manifest["parameters_sha256"]:
            raise ValueError("exported parameters do not match the manifest hash")
        with np.load(directory / "parameters.npz", allow_pickle=False) as data:
            self.theta = data["canonical_theta"]
        self.features = feature_map(self.manifest["prior_knowledge"])
        self.classes = self.manifest["classes"]

    def log_probabilities(self, prompts):
        return _log_softmax(self.features.features(list(prompts)) @ self.theta.T)

    def generate(self, prompt, steps=1, seed=0, greedy=False):
        if steps > 1 and self.features.kind != "ngram":
            raise ValueError("multi-step generation is defined for ngram exports only")
        rng = np.random.default_rng(seed)
        text = prompt
        produced = []
        for _ in range(steps):
            p = np.exp(self.log_probabilities([text])[0])
            choice = int(p.argmax()) if greedy else int(rng.choice(len(p), p=p / p.sum()))
            token = self.classes[choice]
            produced.append(token)
            separator = getattr(self.features, "separator", "")
            text = text + separator + token
        return produced


# ----------------------------------------------------------------------------
# CLI


def _recover_command(args):
    spec = json.loads(Path(args.features).read_text(encoding="utf-8"))
    target = json.loads(Path(args.target).read_text(encoding="utf-8"))
    if set(target) != {"argv"} or not isinstance(target["argv"], list):
        raise ValueError('target JSON must be {"argv": [...]} for a string-only oracle process')
    transcript = args.transcript or str(Path(args.output) / "transcript.jsonl")
    return recover(
        features_spec=spec,
        target_argv=target["argv"],
        output=args.output,
        transcript_path=transcript,
        contexts=args.contexts,
        contexts_file=args.contexts_file,
        samples_per_context=args.samples_per_context,
        validation_contexts=args.validation_contexts,
        max_queries=args.max_queries,
        seed=args.seed,
        bootstrap=args.bootstrap,
        alpha=args.alpha,
        max_relative_standard_error=args.max_relative_standard_error,
    )


def _model_command(args):
    model = RecoveredTextModel(args.model)
    if args.action == "logprobs":
        values = model.log_probabilities([args.prompt])[0]
        return {
            "prompt": args.prompt,
            "log_probabilities": dict(zip(model.classes, values.tolist())),
        }
    return {
        "prompt": args.prompt,
        "generated": model.generate(args.prompt, args.steps, args.seed, args.greedy),
        "status": model.manifest["status"],
    }


def _identifiability_command(args):
    from .text_identifiability import run_all

    result = run_all(seed=args.seed, include_transformer=not args.numpy_only)
    if args.output:
        write_json(args.output, result)
    return result


def add_subcommands(commands):
    recover_parser = commands.add_parser(
        "text-recover",
        help="recover canonical softmax-affine parameters from sampled strings only",
    )
    recover_parser.add_argument("--features", required=True, help="public feature spec JSON")
    recover_parser.add_argument("--target", required=True, help='{"argv": [...]} oracle JSON')
    recover_parser.add_argument("--output", required=True, help="export directory")
    recover_parser.add_argument("--transcript", help="resumable query transcript JSONL")
    recover_parser.add_argument("--contexts", type=int, help="training contexts to query")
    recover_parser.add_argument("--contexts-file", help="public prompt pool, one per line")
    recover_parser.add_argument("--samples-per-context", type=int, default=200)
    recover_parser.add_argument(
        "--validation-contexts",
        type=int,
        default=20,
        help="held-out contexts added to the class test (default 20)",
    )
    recover_parser.add_argument("--max-queries", type=int, help="ceiling on sampled outputs")
    recover_parser.add_argument("--seed", type=int, default=0)
    recover_parser.add_argument("--bootstrap", type=int, default=199)
    recover_parser.add_argument("--alpha", type=float, default=0.01)
    recover_parser.add_argument("--max-relative-standard-error", type=float, default=0.1)
    recover_parser.set_defaults(func=_recover_command)

    model_parser = commands.add_parser(
        "text-model", help="run an exported text-only recovery without the target"
    )
    model_parser.add_argument("action", choices=("logprobs", "generate"))
    model_parser.add_argument("--model", required=True)
    model_parser.add_argument("--prompt", required=True)
    model_parser.add_argument("--steps", type=int, default=1)
    model_parser.add_argument("--seed", type=int, default=0)
    model_parser.add_argument("--greedy", action="store_true")
    model_parser.set_defaults(func=_model_command)

    ident = commands.add_parser(
        "text-identifiability",
        help="run executable counterexamples to universal text-only parameter recovery",
    )
    ident.add_argument("--seed", type=int, default=0)
    ident.add_argument("--numpy-only", action="store_true")
    ident.add_argument("--output")
    ident.set_defaults(func=_identifiability_command)

    from .text_targets import add_subcommands as add_target_commands

    add_target_commands(commands)


if __name__ == "__main__":  # pragma: no cover
    from .cli import main

    sys.exit(main())
