"""Audit dense-image checkpoint reconstruction across complete output maps."""

import argparse
import json
import platform
from hashlib import sha256
from pathlib import Path

from rephix.io import file_sha256, write_json
from rephix.weight_audit import compare_checkpoints


def _hash_tensor(torch, value):
    return sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _checkpoint_hashes(path):
    return {
        file.name: file_sha256(file)
        for file in sorted(path.iterdir())
        if file.is_file()
        and file.name != "layer-recovery.json"
        and file.suffix in {".json", ".safetensors", ".txt", ".model"}
    }


def _capture(torch, path, panels, image, *, device, dtype):
    from PIL import Image
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForDepthEstimation,
        AutoModelForSemanticSegmentation,
    )

    local = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(path, **local)
    architectures = set(config.architectures or ())
    if architectures == {"SegformerForSemanticSegmentation"}:
        wrapper, field = AutoModelForSemanticSegmentation, "logits"
    elif architectures == {"DPTForDepthEstimation"}:
        wrapper, field = AutoModelForDepthEstimation, "predicted_depth"
    else:
        raise ValueError("expected native SegFormer segmentation or DPT depth architecture")
    model = (
        wrapper.from_pretrained(path, torch_dtype=dtype, use_safetensors=True, **local)
        .eval()
        .to(device)
    )
    processor = AutoImageProcessor.from_pretrained(path, **local)
    with Image.open(image) as picture:
        inputs = processor(images=picture.convert("RGB"), return_tensors="pt")
    processed = inputs["pixel_values"]
    inputs_to_run = [*panels, ("processed_image", processed)]
    outputs, observations = [], []
    with torch.inference_mode():
        for label, pixels in inputs_to_run:
            result = getattr(model(pixel_values=pixels.to(device=device, dtype=dtype)), field).cpu()
            if not bool(torch.isfinite(result).all()):
                raise ValueError(f"nonfinite dense output for {label}")
            outputs.append(result)
            observations.append(
                {
                    "label": label,
                    "pixel_shape": list(pixels.shape),
                    "pixel_sha256": _hash_tensor(torch, pixels),
                    "output_shape": list(result.shape),
                    "output_sha256": _hash_tensor(torch, result),
                }
            )
    return {
        "field": field,
        "class": type(model).__name__,
        "outputs": outputs,
        "observations": observations,
        "processed_sha256": _hash_tensor(torch, processed),
    }


def _compare(torch, reference, candidate):
    if reference["field"] != candidate["field"]:
        raise ValueError("dense output task changed")
    pairs = list(zip(reference["outputs"], candidate["outputs"], strict=True))
    if any(a.shape != b.shape for a, b in pairs):
        raise ValueError("dense output shape changed")
    total = equal = pixel_matches = segmentation_pixels = 0
    squared_error = max_error = 0.0
    for a, b in pairs:
        difference = a.double() - b.double()
        total += a.numel()
        equal += int((a == b).sum())
        squared_error += float(difference.square().sum())
        max_error = max(max_error, float(difference.abs().max()))
        if reference["field"] == "logits":
            segmentation_pixels += a.shape[0] * a.shape[-2] * a.shape[-1]
            pixel_matches += int((a.argmax(1) == b.argmax(1)).sum())
    result = {
        "images": sum(a.shape[0] for a, _ in pairs),
        "output_coordinates": total,
        "exact_output_coordinates": equal,
        "output_max_abs_error": max_error,
        "output_rmse": (squared_error / total) ** 0.5,
        "processed_image_pixels_equal": reference["processed_sha256"]
        == candidate["processed_sha256"],
    }
    if segmentation_pixels:
        result["segmentation_pixels"] = segmentation_pixels
        result["segmentation_argmax_matches"] = pixel_matches
    return result


def evaluate(args):
    import torch
    import transformers
    from transformers import AutoImageProcessor

    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    processor = AutoImageProcessor.from_pretrained(
        args.reference, local_files_only=True, trust_remote_code=False
    )
    size = processor.size
    height = size.get("height", size.get("shortest_edge", 256))
    width = size.get("width", size.get("shortest_edge", 256))
    generator = torch.Generator(device="cpu").manual_seed(7119)
    panels = [
        (f"native_{index}", torch.randn(1, 3, height, width, generator=generator))
        for index in range(4)
    ]
    panels += [
        (f"rectangular_{index}", torch.randn(1, 3, height + 32, width + 64, generator=generator))
        for index in range(4)
    ]
    reference, candidate = (
        _capture(torch, path, panels, args.image, device=args.device, dtype=dtype)
        for path in (args.reference, args.candidate)
    )
    audit = compare_checkpoints(args.reference, args.candidate)
    for name in ("reference", "candidate"):
        audit.pop(name)
    recovery = json.loads((args.candidate / "layer-recovery.json").read_text(encoding="utf-8"))
    for name in ("source", "checkpoint", "evidence_dir"):
        recovery.pop(name, None)
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "reference_sha256": _checkpoint_hashes(args.reference),
        "candidate_sha256": _checkpoint_hashes(args.candidate),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "image_sha256": file_sha256(args.image),
        "reference_class": reference["class"],
        "candidate_class": candidate["class"],
        "output_field": reference["field"],
        "raw_checkpoint_audit": audit,
        "recovery": recovery,
        "held_out_behavior": _compare(torch, reference, candidate),
        "reference_observations": reference["observations"],
        "candidate_observations": candidate["observations"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if any(
        args.output.resolve().is_relative_to(p.resolve()) for p in (args.reference, args.candidate)
    ):
        parser.error("output must be outside both checkpoint directories")
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
