"""Inspect local experiment capabilities without fetching model weights."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
import sys
from pathlib import Path


def inspect(args):
    packages = {}
    for name in ("numpy", "torch", "transformers", "safetensors", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hf_ready = all(packages[key] for key in ("torch", "transformers", "safetensors"))
    if args.require_hf and not hf_ready:
        raise RuntimeError("Model dependencies missing; install rephix[hf]")
    disk = shutil.disk_usage(Path(args.workspace).resolve())
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "capabilities": {
            "core": packages["numpy"] is not None,
            "local_models": bool(hf_ready),
            "plots": packages["matplotlib"] is not None,
        },
        "disk_free_bytes": disk.free,
        "devices": {"cpu": True, "cuda": False, "mps": False},
        "downloads_performed": False,
    }
    if importlib.util.find_spec("torch") is not None:
        import torch

        report["devices"].update(
            cuda=torch.cuda.is_available(),
            mps=bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        )
        report["torch_threads"] = torch.get_num_threads()
    if args.model:
        if not hf_ready:
            raise RuntimeError("Inspecting a model config requires rephix[hf]")
        from .hf import _deps, _resolve_model_path

        _, AutoConfig, _, _ = _deps()
        config = AutoConfig.from_pretrained(
            _resolve_model_path(args.model), local_files_only=True, trust_remote_code=False
        )
        h = getattr(config, "hidden_size", None)
        vocab = getattr(config, "vocab_size", None)
        layers = getattr(config, "num_hidden_layers", None)
        info = {
            "model_type": config.model_type,
            "hidden_size": h,
            "vocab_size": vocab,
            "layers": layers,
            "context_limit": getattr(config, "max_position_embeddings", None),
            "source": str(args.model),
        }
        if config.model_type == "qwen2" and h and vocab and layers:
            head_width = h // config.num_attention_heads
            kv_width = config.num_key_value_heads * head_width
            # Qwen2 dense decoder architecture; excludes negligible buffers.
            per_layer = 2 * h * h + 2 * h * kv_width + 3 * h * config.intermediate_size
            per_layer += 3 * h + 2 * kv_width
            parameters = vocab * h + layers * per_layer + h
            if not config.tie_word_embeddings:
                parameters += vocab * h
            info["approximate_parameters"] = parameters
            info["float32_weights_bytes"] = parameters * 4
            info["full_adam_training_state_bytes_excluding_activations"] = parameters * 16
            info["memory_estimate_scope"] = (
                "Dense float32 weights, gradients and two Adam moments; excludes activations, "
                "logits, checkpoint copies and framework overhead."
            )
        report["model"] = info
    return report


def add_subcommands(subparsers):
    command = subparsers.add_parser(
        "doctor", help="inspect installed libraries and local model resources"
    )
    command.add_argument("--workspace", default=".", help="existing path for free-disk inspection")
    command.add_argument("--model", help="local or already cached config; never downloads")
    command.add_argument("--require-hf", action="store_true")
    command.set_defaults(func=inspect)
