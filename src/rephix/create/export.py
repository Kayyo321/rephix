"""Standalone export: merged weights in the base's exact tensor schema, verified.

Unedited tensors are copied byte-for-byte from the base. Edited regions are merged in
float64 and cast to the base's stored dtype; the rounding loss is reported. Config,
generation config and tokenizer files are byte-identical copies of the base's. A fresh
``python -I`` process then loads the export through the ordinary Transformers runtime
(no Rephix import, no hooks, no adapters, no evidence) and must reproduce the in-process
logits within a stated tolerance.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from ..io import atomic_write_text, file_sha256, write_json
from .base import TOKENIZER_FILES, verify_export_schema
from .modeling import region_module

MARKER = "EXPORT_COMPLETE"
TOLERANCE = {"float32": 2e-3, "bfloat16": 6e-2, "float16": 3e-2}

_FRESH = r"""
import json, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path, device, probe_file, out_file = sys.argv[1:5]
probe = json.load(open(probe_file))
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path).to(device).eval()
leaked = sorted(m for m in sys.modules if m == "rephix" or m.startswith("rephix."))
with torch.no_grad():
    ids = torch.tensor([probe["ids"]], device=device)
    logits = model(input_ids=ids).logits[0].float().cpu().numpy()
    prefix = ids[:, : probe["prefix"]]
    out = model.generate(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                         max_new_tokens=16, do_sample=False, num_beams=1,
                         pad_token_id=tokenizer.eos_token_id)
np.save(out_file, logits)
print(json.dumps({"rephix_modules_loaded": leaked,
                  "generated_ids": out[0, prefix.shape[1]:].tolist(),
                  "generated_text": tokenizer.decode(out[0, prefix.shape[1]:], skip_special_tokens=True),
                  "torch": torch.__version__,
                  "dtype": str(next(model.parameters()).dtype)}))
"""


def _stored_dtype(contract, region_weight_name):
    for item in contract["tensor_schema"]:
        if item["name"] == region_weight_name:
            return item["dtype"]
    raise ValueError(f"{region_weight_name} not in base schema")


def _torch_dtype(torch, name):
    return {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}[name]


def merge_rounded(torch, model, edits, contract):
    """Merge Δ into the in-process weights, rounded through the base's stored dtype."""
    report = {}
    with torch.no_grad():
        for region in edits.regions:
            weight = region_module(model, region).weight
            layer, block, module = region.split(".")
            name = f"model.layers.{layer}.{block}.{module}.weight"
            stored = _torch_dtype(torch, _stored_dtype(contract, name))
            delta = edits.delta(region).detach().to(weight.device, torch.float64)
            exact = weight.detach().to(torch.float64) + delta
            rounded = exact.to(stored).to(torch.float64)
            lost = float((rounded - exact).norm().item())
            report[region] = {
                "tensor": name,
                "stored_dtype": str(stored).replace("torch.", ""),
                "delta_frobenius": float(delta.norm().item()),
                "rounding_error_frobenius": lost,
                "rounding_error_relative_to_delta": lost / max(float(delta.norm().item()), 1e-30),
            }
            weight.copy_(rounded.to(weight.dtype))
    edits.remove()
    return report


def write_weights(torch, model, edits_regions, contract, destination: Path, *, all_tensors=False):
    """Write safetensors with the base's names, shapes and dtypes; copy unedited bytes."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    base = Path(contract["path"])
    edited = {
        f"model.layers.{r.split('.')[0]}.{r.split('.')[1]}.{r.split('.')[2]}.weight": r
        for r in edits_regions
    }
    if all_tensors:
        edited = {item["name"]: item["name"] for item in contract["tensor_schema"]}
    state = model.state_dict()
    unchanged, changed = 0, []
    for file in sorted(base.glob("*.safetensors")):
        tensors = {}
        with safe_open(str(file), framework="pt") as handle:
            metadata = handle.metadata() or {}
            for name in handle.keys():
                original = handle.get_tensor(name)
                if name in edited:
                    tensors[name] = state[name].detach().to(original.dtype).cpu().contiguous()
                    changed.append(name)
                else:
                    tensors[name] = original
                    unchanged += 1
        save_file(tensors, str(destination / file.name), metadata={**metadata, "format": "pt"})
    for extra in ("model.safetensors.index.json",):
        if (base / extra).is_file():
            shutil.copyfile(base / extra, destination / extra)
    return {"edited_tensors": sorted(changed), "copied_unchanged_tensors": unchanged}


def confinement_check(contract, destination: Path, edited_names):
    """Every tensor outside the edited regions must equal the base bit for bit."""
    from safetensors import safe_open

    base = Path(contract["path"])
    differing_outside, identical_inside = [], []
    for file in sorted(base.glob("*.safetensors")):
        with (
            safe_open(str(file), framework="pt") as a,
            safe_open(str(destination / file.name), framework="pt") as b,
        ):
            for name in a.keys():
                left, right = a.get_tensor(name), b.get_tensor(name)
                same = (
                    left.dtype == right.dtype
                    and left.shape == right.shape
                    and bool(
                        (
                            left.view(-1).view(dtype=_int_view(left))
                            == right.view(-1).view(dtype=_int_view(right))
                        ).all()
                    )
                )
                if name in edited_names and same:
                    identical_inside.append(name)
                elif name not in edited_names and not same:
                    differing_outside.append(name)
    return {
        "passed": not differing_outside,
        "tensors_outside_edits_not_bit_identical": differing_outside,
        "edited_tensors_unchanged_after_rounding": identical_inside,
    }


def _int_view(tensor):
    import torch

    return {2: torch.int16, 4: torch.int32, 8: torch.int64, 1: torch.uint8}[tensor.element_size()]


def fresh_process_check(
    destination: Path, probe_ids, prefix, device, reference_logits, reference_generation, tolerance
):
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="rephix-reload-") as scratch:
        probe_file = Path(scratch) / "probe.json"
        out_file = Path(scratch) / "logits.npy"
        probe_file.write_text(json.dumps({"ids": probe_ids, "prefix": prefix}))
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP")}
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                _FRESH,
                str(destination),
                device,
                str(probe_file),
                str(out_file),
            ],
            capture_output=True,
            text=True,
            cwd=scratch,
            env=env,
            timeout=900,
        )
        if completed.returncode != 0:
            return {"passed": False, "error": completed.stderr[-2000:]}
        info = json.loads(completed.stdout.strip().splitlines()[-1])
        logits = np.load(out_file)
    difference = float(np.abs(logits - reference_logits).max())
    same_generation = info["generated_ids"] == reference_generation
    return {
        "passed": difference <= tolerance and not info["rephix_modules_loaded"],
        "python": "-I (isolated: no PYTHONPATH, no user site, no current directory)",
        "rephix_modules_loaded": info["rephix_modules_loaded"],
        "runtime_dtype": info["dtype"],
        "max_abs_logit_difference": difference,
        "tolerance": tolerance,
        "greedy_16_tokens_identical": same_generation,
        "generated_text": info["generated_text"],
    }


def write_checksums(directory: Path):
    lines = []
    for file in sorted(p for p in directory.rglob("*") if p.is_file()):
        if file.name in (MARKER, "CHECKSUMS.sha256"):
            continue
        lines.append(f"{file_sha256(file)}  {file.relative_to(directory).as_posix()}")
    text = "\n".join(lines) + "\n"
    atomic_write_text(directory / "CHECKSUMS.sha256", text)
    return hashlib.sha256(text.encode()).hexdigest()


def verify_export(directory: str | Path) -> dict:
    """Check the completion marker and every checksum of an export directory."""
    directory = Path(directory)
    marker = directory / MARKER
    if not marker.is_file():
        return {"valid": False, "reason": f"no {MARKER} marker: incomplete or not an export"}
    info = json.loads(marker.read_text(encoding="utf-8"))
    sums = (directory / "CHECKSUMS.sha256").read_text(encoding="utf-8")
    if hashlib.sha256(sums.encode()).hexdigest() != info["checksums_sha256"]:
        return {"valid": False, "reason": "CHECKSUMS.sha256 does not match the marker"}
    bad = []
    lines = [line for line in sums.split("\n") if line]
    for line in lines:
        digest, name = line.split("  ", 1)
        path = directory / name
        if not path.is_file() or file_sha256(path) != digest:
            bad.append(name)
    return {
        "valid": not bad,
        "corrupted_or_missing": bad,
        "files": len(lines),
        "created_by": info.get("created_by"),
    }


def export(
    ctx,
    edits,
    contract,
    destination: Path,
    *,
    provenance: dict,
    report_md: str,
    results: dict,
    device: str,
    full_model: bool = False,
):
    """Build the export in a staging directory and publish it atomically when verified."""
    torch = ctx.torch
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.incomplete-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        merge_report = {}
        regions = []
        if edits is not None:
            regions = list(edits.regions)
            merge_report = merge_rounded(torch, ctx.model, edits, contract)
        elif full_model:
            # Every tensor changed (ordinary fine-tuning baseline): round all of them
            # through their stored dtype so the in-process reference equals the export.
            stored = {item["name"]: item["dtype"] for item in contract["tensor_schema"]}
            with torch.no_grad():
                for name, tensor in ctx.model.state_dict().items():
                    if name in stored:
                        tensor.copy_(tensor.to(_torch_dtype(torch, stored[name])).to(tensor.dtype))
            merge_report = {"all_parameters": "rounded through stored dtypes"}
        base = Path(contract["path"])
        for name in ("config.json", *TOKENIZER_FILES):
            if (base / name).is_file():
                shutil.copyfile(base / name, staging / name)
        weights = write_weights(
            torch, ctx.model, regions, contract, staging, all_tensors=full_model
        )
        schema = verify_export_schema(contract, staging)
        confinement = confinement_check(contract, staging, set(weights["edited_tensors"]))
        # Reference: the in-process model with the same rounded weights, in the dtype the
        # ordinary runtime loads (float32 upcast of the stored values).
        probe_text = "The quick brown fox jumps over the lazy dog. In the beginning,"
        ids = ctx.tokenizer(probe_text, add_special_tokens=True)["input_ids"][:32]
        prefix = max(2, len(ids) // 2)
        with torch.inference_mode():
            tensor = torch.tensor([ids], device=ctx.device)
            reference = ctx.model(input_ids=tensor).logits[0].float().cpu().numpy()
            from ..hf import _generation_config

            config = _generation_config(ctx.model, ctx.tokenizer, 16)
            generated = ctx.model.generate(
                input_ids=tensor[:, :prefix],
                attention_mask=torch.ones_like(tensor[:, :prefix]),
                generation_config=config,
            )[0, prefix:].tolist()
        stored = {item["dtype"] for item in contract["tensor_schema"]}
        working = str(next(ctx.model.parameters()).dtype).replace("torch.", "")
        tolerance = TOLERANCE.get(working, 6e-2)
        if device == "cuda":
            torch.cuda.empty_cache()  # leave room for the separate reload process
        reload = fresh_process_check(staging, ids, prefix, device, reference, generated, tolerance)
        if not reload["passed"] and "OutOfMemoryError" in reload.get("error", ""):
            # Accelerator full (for example, other jobs): reload on the CPU instead, where
            # float32 kernels differ slightly from the accelerator's, so the tolerance widens.
            reload = fresh_process_check(
                staging, ids, prefix, "cpu", reference, generated, max(tolerance, 2e-2)
            )
            reload["device_fallback"] = "cpu (accelerator out of memory)"
        verification = {
            "schema": schema,
            "confinement": confinement,
            "merge_rounding": merge_report,
            "weights": weights,
            "fresh_process_reload": reload,
            "stored_dtypes": sorted(stored),
            "quantization": "none performed",
            "tokenizer_conversion": "none performed (byte-identical base tokenizer files)",
            "passed": bool(schema["passed"] and confinement["passed"] and reload["passed"]),
        }
        write_json(staging / "export-verification.json", verification)
        write_json(staging / "rephix-create.json", provenance)
        write_json(staging / "results.json", {**results, "export_verification": verification})
        atomic_write_text(staging / "REPORT.md", report_md)
        if not verification["passed"]:
            failed = staging.parent / f".{destination.name}.failed-{uuid.uuid4().hex[:8]}"
            os.replace(staging, failed)
            raise RuntimeError(
                f"export verification failed; the unverified files are kept at {failed} "
                "and no complete export was published"
            )
        digest = write_checksums(staging)
        write_json(
            staging / MARKER,
            {"checksums_sha256": digest, "created_by": provenance.get("rephix_version")},
        )
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return verification
