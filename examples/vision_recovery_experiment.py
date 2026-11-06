"""Audit a recovered vision checkpoint and its special-token gauge."""

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

from rephix.io import file_sha256, write_json
from rephix.weight_audit import compare_checkpoints


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def _scores(torch, model, pixels, *, interpolated):
    with torch.inference_mode():
        return torch.cat(
            [
                model(
                    pixel_values=batch.to(device=model.device, dtype=model.dtype),
                    interpolate_pos_encoding=interpolated,
                )
                .logits.cpu()
                .double()
                for batch in pixels.split(4)
            ]
        )


def _compare(a, b):
    delta = b - a
    return {
        "images": len(a),
        "logit_rmse": float(delta.square().mean().sqrt()),
        "logit_max_abs_error": float(delta.abs().max()),
        "top1_matches": int((a.argmax(dim=1) == b.argmax(dim=1)).sum()),
    }


def _audit(source, candidate):
    report = compare_checkpoints(source, candidate)
    report.pop("reference")
    report.pop("candidate")
    return report


def evaluate(args):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    fit = json.loads((args.candidate / "layer-recovery.json").read_text(encoding="utf-8"))
    fit.pop("source")
    fit.pop("checkpoint")
    dtype = getattr(torch, fit["oracle_dtype"].removeprefix("torch."))
    opts = {"local_files_only": True, "trust_remote_code": False}

    def load(path):
        return (
            AutoModelForImageClassification.from_pretrained(
                path, use_safetensors=True, torch_dtype=dtype, **opts
            )
            .eval()
            .to(args.device)
        )

    reference, candidate = load(args.reference), load(args.candidate)
    raw_audit = _audit(args.reference, args.candidate)
    height, width = _pair(reference.config.image_size)
    patch_h, patch_w = _pair(reference.config.patch_size)
    rng = torch.Generator(device="cpu").manual_seed(7115)
    inputs = {
        "native_synthetic": (
            torch.randn((8, reference.config.num_channels, height, width), generator=rng),
            False,
        ),
        "interpolated_synthetic": (
            torch.randn(
                (8, reference.config.num_channels, height + 2 * patch_h, width + patch_w),
                generator=rng,
            ),
            True,
        ),
    }
    image_hashes = {}
    if args.image:
        processors = [
            AutoImageProcessor.from_pretrained(path, use_fast=False, **opts)
            for path in (args.reference, args.candidate)
        ]
        images = []
        for path in args.image:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
            image_hashes[path.name] = file_sha256(path)
        a, b = [
            processor(images=images, return_tensors="pt").pixel_values for processor in processors
        ]
        if not torch.equal(a, b):
            raise ValueError("exported image processor changed the input pixels")
        inputs["processed_real_images"] = (a, False)

    baseline = {
        label: _scores(torch, reference, pixels, interpolated=interpolation)
        for label, (pixels, interpolation) in inputs.items()
    }
    behavior = {
        label: _compare(
            baseline[label], _scores(torch, candidate, pixels, interpolated=interpolation)
        )
        for label, (pixels, interpolation) in inputs.items()
    }

    gauges = []
    with torch.no_grad():
        for gauge in fit["embedding_gauges"]:
            a = reference.get_submodule(gauge["module"])
            b = candidate.get_submodule(gauge["module"])
            for index, token in enumerate(gauge["special_tokens"]):
                av = getattr(a, token)
                bv = getattr(b, token)
                ap = a.position_embeddings[:, index : index + 1]
                bp = b.position_embeddings[:, index : index + 1]
                error = float(((av + ap) - (bv + bp)).abs().max())
                # Change only the candidate gauge for the separate parameter audit.
                displacement = ap - bp
                bp.add_(displacement)
                bv.sub_(displacement)
                # Independently perturb the source gauge and rerun the image panel.
                shift = torch.linspace(-0.25, 0.25, av.shape[-1], device=av.device, dtype=av.dtype)
                av.add_(shift)
                ap.sub_(shift)
                gauges.append(
                    {"module": gauge["module"], "token": token, "sum_max_abs_error": error}
                )
    control = {
        label: _compare(
            baseline[label], _scores(torch, reference, pixels, interpolated=interpolation)
        )
        for label, (pixels, interpolation) in inputs.items()
    }
    del reference
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        candidate.to("cpu").save_pretrained(temporary, safe_serialization=True)
        aligned_audit = _audit(args.reference, temporary)

    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "reference_sha256": {
            path.name: file_sha256(path)
            for path in sorted(args.reference.iterdir())
            if path.suffix in {".safetensors", ".json"}
        },
        "candidate_sha256": {
            path.name: file_sha256(path)
            for path in sorted(args.candidate.iterdir())
            if path.suffix in {".safetensors", ".json"}
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "real_images_sha256": image_hashes,
        "recovery": fit,
        "raw_checkpoint_audit": raw_audit,
        "gauge_aligned_checkpoint_audit": aligned_audit,
        "special_token_sum_errors": gauges,
        "held_out_behavior": behavior,
        "function_preserving_reference_gauge_control": control,
        "evaluation_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image", type=Path, action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
