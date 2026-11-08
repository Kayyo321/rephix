"""Base-model contract: the executable architecture the exported model must keep.

The base defines tensor names, shapes, dtypes, ties, tokenizer and special tokens.
Create Mode edits only the per-layer linear maps listed in ``REGION_MODULES``; it never
widens layers, adds vocabulary, changes attention structure or unties parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from ..io import file_sha256

# Small, explicit support matrix. ``verified`` states the evidence behind each entry.
SUPPORTED = {
    "llama": {
        "class": "LlamaForCausalLM",
        "verified": "pretrained: HuggingFaceTB/SmolLM2-135M and -Instruct (CUDA and CPU)",
    },
    "qwen2": {
        "class": "Qwen2ForCausalLM",
        "verified": "pretrained: Qwen/Qwen2.5-0.5B-Instruct (CUDA)",
    },
    "mistral": {
        "class": "MistralForCausalLM",
        "verified": "random tiny fixture only (CPU); no pretrained Mistral run",
    },
    "qwen3": {
        "class": "Qwen3ForCausalLM",
        "verified": "random tiny fixture only (CPU); no pretrained Qwen3 run",
    },
}
REGION_MODULES = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def resolve_base_path(base: str, revision: str | None) -> tuple[Path, dict]:
    """A local directory, or a hub id pinned to an exact 40-hex commit."""
    path = Path(base).expanduser()
    if path.is_dir():
        if revision is not None:
            raise ValueError("--base-revision applies to hub identifiers, not local directories")
        return path.resolve(), {"kind": "local", "path": str(path.resolve())}
    if path.exists():
        raise ValueError(f"base must be a model directory: {base}")
    if not re.match(r"^[\w.-]+/[\w.-]+$", base):
        raise ValueError(f"base is neither a local directory nor a hub model id: {base}")
    if revision is None or not _HEX40.match(revision):
        raise ValueError(
            "a hub base needs --base-revision with the exact 40-hex commit so the "
            "contract is reproducible"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install huggingface_hub to resolve hub bases") from exc
    local = snapshot_download(
        base,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )
    return Path(local).resolve(), {"kind": "hub", "repo_id": base, "revision": revision}


def _safetensor_schema(root: Path) -> list[dict]:
    from safetensors import safe_open

    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError(
            f"{root}: no safetensors weights; Create Mode needs safetensors (no pickle loading)"
        )
    schema = []
    for file in files:
        with safe_open(str(file), framework="pt") as handle:
            for name in handle.keys():
                tensor_slice = handle.get_slice(name)
                schema.append(
                    {
                        "name": name,
                        "shape": list(tensor_slice.get_shape()),
                        "dtype": str(tensor_slice.get_dtype()),
                    }
                )
    return sorted(schema, key=lambda item: item["name"])


def content_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in root.iterdir() if p.is_file()):
        if file.name.startswith("."):
            continue
        digest.update(file.name.encode())
        digest.update(file_sha256(file).encode())
    return digest.hexdigest()


def resolve_contract(base: str, revision: str | None = None) -> dict:
    root, origin = resolve_base_path(base, revision)
    config_file = root / "config.json"
    if not config_file.is_file():
        raise ValueError(f"{root}: missing config.json")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if config.get("auto_map"):
        raise ValueError("bases that need custom remote code are not supported")
    if model_type not in SUPPORTED:
        raise ValueError(
            f"unsupported base architecture {model_type!r}. Create Mode supports "
            f"{sorted(SUPPORTED)} decoder-only models with per-layer q/k/v/o and "
            "gate/up/down projections. Other families are rejected rather than guessed."
        )
    architectures = config.get("architectures") or []
    if architectures and SUPPORTED[model_type]["class"] not in architectures:
        raise ValueError(f"unexpected architecture {architectures} for model_type {model_type}")
    schema = _safetensor_schema(root)
    tokenizer = {
        name: file_sha256(root / name) for name in TOKENIZER_FILES if (root / name).is_file()
    }
    if "tokenizer.json" not in tokenizer and "tokenizer.model" not in tokenizer:
        raise ValueError(f"{root}: no tokenizer files")
    layers = int(config["num_hidden_layers"])
    names = {item["name"] for item in schema}
    regions = []
    for layer in range(layers):
        for block, module in REGION_MODULES:
            name = f"model.layers.{layer}.{block}.{module}.weight"
            if name not in names:
                raise ValueError(f"base tensor schema lacks expected {name}")
            regions.append(f"{layer}.{block}.{module}")
    tied = bool(config.get("tie_word_embeddings", False))
    if tied and "lm_head.weight" in names:
        # A stored lm_head with tying declared is still tied at load time.
        pass
    return {
        "schema": "rephix-create-base/1",
        "origin": origin,
        "path": str(root),
        "model_type": model_type,
        "architecture": SUPPORTED[model_type]["class"],
        "support_evidence": SUPPORTED[model_type]["verified"],
        "config_sha256": file_sha256(config_file),
        "config": {
            key: config.get(key)
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "vocab_size",
                "max_position_embeddings",
                "rope_theta",
                "rope_scaling",
                "tie_word_embeddings",
                "torch_dtype",
                "bos_token_id",
                "eos_token_id",
            )
        },
        "tied_embeddings": tied,
        "tensor_schema": schema,
        "tensor_schema_sha256": hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode()
        ).hexdigest(),
        "tokenizer_files": tokenizer,
        "content_sha256": content_fingerprint(root),
        "regions": regions,
        "excluded_from_edits": [
            "embeddings",
            "lm_head",
            "norms",
            "rotary buffers",
            "biases",
        ],
    }


def describe_tokenizer(tokenizer) -> dict:
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "adds_bos": bool(
            tokenizer.bos_token_id is not None
            and tokenizer("x", add_special_tokens=True)["input_ids"][:1] == [tokenizer.bos_token_id]
        ),
    }


def load_base(contract: dict, *, device: str, dtype: str):
    from ..hf import _deps, _load_tokenizer

    torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer = _deps()
    root = contract["path"]
    current = content_fingerprint(Path(root))
    if current != contract["content_sha256"]:
        raise ValueError(
            "base files changed since the run started (content fingerprint mismatch); "
            "start a new run for a different base revision"
        )
    config = AutoConfig.from_pretrained(root)
    tokenizer = _load_tokenizer(AutoTokenizer, root, model_config=config)
    model = AutoModelForCausalLM.from_pretrained(root, dtype=getattr(torch, dtype))
    model.to(device)
    model.eval()
    if contract["tied_embeddings"]:
        head = model.get_output_embeddings().weight
        embed = model.get_input_embeddings().weight
        if head.data_ptr() != embed.data_ptr():
            raise ValueError("base declares tied embeddings but loads untied parameters")
    return model, tokenizer


def verify_export_schema(contract: dict, export_dir: Path) -> dict:
    """The export must have exactly the base's tensor schema and tokenizer bytes."""
    exported = _safetensor_schema(export_dir)
    base = contract["tensor_schema"]
    base_names = {item["name"]: item for item in base}
    exported_names = {item["name"]: item for item in exported}
    missing = sorted(set(base_names) - set(exported_names))
    extra = sorted(set(exported_names) - set(base_names))
    mismatched = sorted(
        name
        for name in set(base_names) & set(exported_names)
        if base_names[name] != exported_names[name]
    )
    tokenizer_mismatch = sorted(
        name
        for name, digest in contract["tokenizer_files"].items()
        if not (export_dir / name).is_file() or file_sha256(export_dir / name) != digest
    )
    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    config_mismatch = sorted(
        key
        for key, value in contract["config"].items()
        if key != "torch_dtype" and config.get(key) != value
    )
    ok = not (missing or extra or mismatched or tokenizer_mismatch or config_mismatch)
    return {
        "passed": ok,
        "tensors": len(exported),
        "missing_tensors": missing,
        "extra_tensors": extra,
        "shape_or_dtype_mismatch": mismatched,
        "tokenizer_files_not_byte_identical": tokenizer_mismatch,
        "config_mismatch": config_mismatch,
        "tied_embeddings": contract["tied_embeddings"],
    }


def available_memory() -> dict:
    result = {"system_ram_bytes": None, "system_ram_available_bytes": None}
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        result["system_ram_bytes"] = pages * size
        result["system_ram_available_bytes"] = os.sysconf("SC_AVPHYS_PAGES") * size
    except (ValueError, OSError, AttributeError):
        pass
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    result["system_ram_available_bytes"] = int(line.split()[1]) * 1024
    except OSError:
        pass
    return result
