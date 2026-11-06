"""Export a private output head from dense scores and an unchanged public backbone."""

from __future__ import annotations

import inspect
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hf import _deps, _device, _load_tokenizer
from .io import write_json


@dataclass(frozen=True)
class _Evidence:
    token_ids: np.ndarray
    lengths: np.ndarray
    scores: np.ndarray

    def contexts(self):
        for row, length in zip(self.token_ids, self.lengths):
            yield row[: int(length)]


def _load_evidence(directory, *, inputs, classes, context_limit):
    path = Path(directory)
    arrays = [
        np.load(path / name, mmap_mode="r", allow_pickle=False)
        for name in ("input_ids.npy", "lengths.npy", "scores.npy")
    ]
    if not all(isinstance(array, np.ndarray) for array in arrays):
        raise ValueError(f"{path}: evidence files must be NumPy arrays")
    token_ids, lengths, scores = arrays
    if (
        token_ids.ndim != 2
        or not all(token_ids.shape)
        or lengths.shape != (len(token_ids),)
        or not np.issubdtype(token_ids.dtype, np.integer)
        or not np.issubdtype(lengths.dtype, np.integer)
    ):
        raise ValueError(f"{path}: expected a nonempty integer token matrix and a length per row")
    if (
        (lengths < 1).any()
        or (lengths > token_ids.shape[1]).any()
        or (context_limit is not None and (lengths > context_limit).any())
    ):
        raise ValueError(f"{path}: context lengths exceed the token matrix or model limits")
    if scores.shape != (len(token_ids), classes) or scores.dtype.kind != "f":
        raise ValueError(f"{path}: expected one floating-point score per output ID in every row")
    evidence = _Evidence(token_ids, lengths, scores)
    if any(row.min() < 0 or row.max() >= inputs for row in evidence.contexts()):
        raise ValueError(f"{path}: context contains an out-of-range token ID")
    return evidence


def _reject_overlap(train, validation):
    seen = {np.asarray(row, dtype=np.int64).tobytes() for row in train.contexts()}
    if any(np.asarray(row, dtype=np.int64).tobytes() in seen for row in validation.contexts()):
        raise ValueError("training and validation share a token context")


def _forward_kwargs(model):
    parameters = inspect.signature(model.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return {name: 1}
    return {}


def _context_batches(torch, evidence, device, batch_size):
    # Equal lengths put the final position at a real token and let supported
    # architectures skip computing vocabulary logits at earlier positions.
    for length in np.unique(evidence.lengths):
        members = np.flatnonzero(evidence.lengths == length)
        for offset in range(0, len(members), batch_size):
            rows = members[offset : offset + batch_size]
            inputs = torch.as_tensor(
                evidence.token_ids[rows, : int(length)].astype(np.int64), device=device
            )
            yield rows, inputs


def _features(torch, model, head, evidence, device, batch_size):
    observed = []

    def capture(module, args, output):
        observed.append((args[0], output))

    hook = head.register_forward_hook(capture)
    precision = np.float64 if head.weight.dtype == torch.float64 else np.float32
    features = np.empty((len(evidence.lengths), head.in_features), dtype=precision)
    forward_kwargs = _forward_kwargs(model)
    try:
        with torch.inference_mode():
            for rows, inputs in _context_batches(torch, evidence, device, batch_size):
                observed.clear()
                output = model(input_ids=inputs, use_cache=False, **forward_kwargs)
                if len(observed) != 1:
                    raise ValueError("base forward did not call the output head exactly once")
                hidden, raw = observed[0]
                if (
                    hidden.ndim != 3
                    or raw.ndim != 3
                    or output.logits.shape != raw.shape
                    or hidden.shape[:2] != raw.shape[:2]
                    or raw.shape[0] != len(rows)
                    or not torch.equal(output.logits[:, -1], raw[:, -1])
                ):
                    raise ValueError("base model logits do not match the affine output head")
                features[rows] = hidden[:, -1].cpu().numpy()
    finally:
        hook.remove()
    return features


def _runtime_error(torch, model, evidence, device, batch_size, reference_id):
    squared_error = 0.0
    forward_kwargs = _forward_kwargs(model)
    with torch.inference_mode():
        for rows, inputs in _context_batches(torch, evidence, device, batch_size):
            output = model(input_ids=inputs, use_cache=False, **forward_kwargs)
            logits = output.logits[:, -1].cpu().numpy().astype(np.float64)
            scores = evidence.scores[rows].astype(np.float64)
            if reference_id is not None:
                logits -= logits[:, reference_id : reference_id + 1]
                scores -= scores[:, reference_id : reference_id + 1]
            delta = logits - scores
            squared_error += float(np.vdot(delta, delta))
    columns = evidence.scores.shape[1] - (reference_id is not None)
    return float(np.sqrt(squared_error / (len(evidence.lengths) * columns)))


def recover_endpoint_head(
    base,
    train,
    validation,
    output,
    *,
    anchor_id=None,
    mode="anchored",
    dtype="float32",
    device="cpu",
    batch_size=16,
    chunk_size=512,
    max_validation_rmse=1e-3,
    include_tokenizer=True,
):
    """Recover the output head using known base features and endpoint scores.

    The target must share the base backbone and token coordinates. Raw logits
    determine absolute head scores. For log probabilities, supply an unchanged
    anchor row or declare that a strict majority of base rows are unchanged.
    """
    from .head_solver import fit_anchored_head, fit_unanchored_head

    base, output = Path(base), Path(output)
    if not (base / "config.json").is_file():
        raise ValueError("endpoint head recovery needs a local base directory with config.json")
    if output.exists() or output.is_symlink() or base.resolve() in output.resolve().parents:
        raise ValueError("output must be a new directory outside the base checkpoint")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    if mode not in {"anchored", "raw_logits", "base_median"}:
        raise ValueError("mode must be anchored, raw_logits, or base_median")
    if mode != "anchored" and anchor_id is not None:
        raise ValueError("anchor ID applies only to anchored mode")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")

    torch, AutoConfig, AutoModel, AutoTokenizer = _deps()
    device = _device(torch, device)
    opts = {"trust_remote_code": False, "local_files_only": True}
    config = AutoConfig.from_pretrained(base, **opts)
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("endpoint head recovery requires a dense, nonquantized base")
    model, loading = AutoModel.from_pretrained(
        base,
        config=config,
        use_safetensors=True,
        torch_dtype=getattr(torch, dtype),
        output_loading_info=True,
        **opts,
    )
    if any(loading[key] for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"base checkpoint did not fully load: {loading}")
    model.eval().to(device)
    head = model.get_output_embeddings()
    embeddings = model.get_input_embeddings()
    if type(head) is not torch.nn.Linear or type(embeddings) is not torch.nn.Embedding:
        raise ValueError("base must have a standard Linear output head and token embedding")
    if mode == "anchored" and (
        type(anchor_id) is not int or not 0 <= anchor_id < head.out_features
    ):
        raise ValueError("anchor ID must identify an existing output row")
    evidence_options = {
        "inputs": embeddings.num_embeddings,
        "classes": head.out_features,
        "context_limit": getattr(config, "max_position_embeddings", None),
    }
    training = _load_evidence(train, **evidence_options)
    heldout = _load_evidence(validation, **evidence_options)
    _reject_overlap(training, heldout)

    features = _features(torch, model, head, training, device, batch_size)
    heldout_features = _features(torch, model, head, heldout, device, batch_size)
    samples = (features, training.scores, heldout_features, heldout.scores)
    solver_options = {"chunk_size": chunk_size, "max_validation_rmse": max_validation_rmse}
    if mode == "anchored":
        weights, bias, fit = fit_anchored_head(
            *samples,
            anchor_id=anchor_id,
            anchor_weight=head.weight[anchor_id].detach().cpu().numpy(),
            anchor_bias=None if head.bias is None else float(head.bias[anchor_id].detach()),
            **solver_options,
        )
    else:
        prior = {}
        if mode == "base_median":
            prior = {
                "base_weight": head.weight.detach().cpu().numpy(),
                "base_bias": None if head.bias is None else head.bias.detach().cpu().numpy(),
            }
        weights, bias, fit = fit_unanchored_head(
            *samples, mode=mode, has_bias=head.bias is not None, **prior, **solver_options
        )
        del prior
    was_tied = head.weight is embeddings.weight
    if was_tied:
        head.weight = torch.nn.Parameter(torch.empty_like(head.weight))
        model.config.tie_word_embeddings = False
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(weights))
        if bias is not None:
            head.bias.copy_(torch.from_numpy(bias))
    reference_id = 0 if mode == "base_median" else anchor_id
    metric = "logit" if reference_id is None else "contrast"
    runtime_error = _runtime_error(torch, model, heldout, device, batch_size, reference_id)
    if not np.isfinite(runtime_error) or runtime_error > max_validation_rmse:
        raise ValueError(
            f"{dtype} candidate held-out {metric} RMSE {runtime_error:.6g} exceeds "
            f"max_validation_rmse {max_validation_rmse:.6g} or is nonfinite"
        )
    model.to("cpu")

    report = {
        "observation_mode": (
            "dense_endpoint_scores_with_known_backbone_and_anchor"
            if mode == "anchored"
            else "dense_endpoint_scores_with_known_backbone"
        ),
        "model_type": model.config.model_type,
        "model_dtype": dtype,
        "base_output_was_tied": was_tied,
        "base_ignored_keys": loading["unexpected_keys"],
        "tokenizer_exported": include_tokenizer,
        "checkpoint": str(output),
        f"validation_runtime_{metric}_rmse": runtime_error,
        **fit,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rephix-endpoint-head-", dir=output.parent) as temp:
        staged = Path(temp) / "checkpoint"
        model.save_pretrained(staged, safe_serialization=True)
        if include_tokenizer:
            tokenizer = _load_tokenizer(AutoTokenizer, base, model_config=config, **opts)
            tokenizer.save_pretrained(staged)
        write_json(staged / "endpoint-head.json", report)
        if output.exists() or output.is_symlink():
            raise ValueError("output already exists")
        os.rename(staged, output)
    return report


def _command(args):
    return recover_endpoint_head(
        args.base,
        args.train,
        args.validation,
        args.output,
        anchor_id=args.anchor_id,
        mode=args.mode,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        max_validation_rmse=args.max_validation_rmse,
        include_tokenizer=not args.without_tokenizer,
    )


def add_subcommands(commands):
    command = commands.add_parser(
        "recover-endpoint-head",
        help="recover a private output head from dense scores and a known backbone",
    )
    command.add_argument("--base", required=True, help="local Transformers base checkpoint")
    command.add_argument("--train", required=True, help="directory of fitting evidence arrays")
    command.add_argument(
        "--validation", required=True, help="directory of held-out evidence arrays"
    )
    command.add_argument("--output", required=True, help="new Transformers checkpoint directory")
    command.add_argument("--anchor-id", type=int, help="unchanged output row for anchored mode")
    command.add_argument(
        "--mode",
        choices=("anchored", "raw_logits", "base_median"),
        default="anchored",
        help="use a trusted row, absolute logits, or a majority of unchanged base rows",
    )
    command.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="known backbone and export precision, matching the endpoint backbone",
    )
    command.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    command.add_argument("--batch-size", type=int, default=16)
    command.add_argument("--chunk-size", type=int, default=512)
    command.add_argument("--max-validation-rmse", type=float, default=1e-3)
    command.add_argument("--without-tokenizer", action="store_true")
    command.set_defaults(func=_command)
