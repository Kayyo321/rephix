"""Audit reconstructed native Diffusers autoencoders on images and latent inputs."""

import argparse
import json
import platform
from hashlib import sha256
from pathlib import Path

from rephix.io import file_sha256, write_json
from rephix.weight_audit import compare_checkpoints


def _tensor_signature(value):
    values = value.detach().cpu().contiguous()
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": sha256(values.numpy().tobytes()).hexdigest(),
    }


def _panels(torch, image):
    import numpy as np
    from PIL import Image

    generator = torch.Generator(device="cpu").manual_seed(7116)
    panels = [
        ("square_random", 2 * torch.rand((4, 3, 64, 64), generator=generator) - 1),
        ("rectangular_random", 2 * torch.rand((4, 3, 80, 96), generator=generator) - 1),
    ]
    with Image.open(image) as picture:
        pixels = np.array(
            picture.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
    panels.append(
        ("public_image", torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1)
    )
    return panels


def _weight_file(path):
    weights = path / "diffusion_pytorch_model.safetensors"
    if not weights.is_file():
        raise ValueError("this experiment requires the native single-file Diffusers checkpoint")
    return weights


def _checkpoint_hashes(path):
    return {
        file.name: file_sha256(file)
        for file in sorted(path.iterdir())
        if file.is_file()
        and file.name != "layer-recovery.json"
        and file.suffix in {".json", ".safetensors"}
    }


def _capture(torch, path, panels, *, dtype, device):
    from diffusers import AutoencoderKL, AutoencoderTiny

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    classes = {"AutoencoderKL": AutoencoderKL, "AutoencoderTiny": AutoencoderTiny}
    class_name = config.get("_class_name")
    if class_name not in classes:
        raise ValueError("expected a native AutoencoderKL or AutoencoderTiny checkpoint")
    model = (
        classes[class_name]
        .from_pretrained(path, local_files_only=True, use_safetensors=True, torch_dtype=dtype)
        .eval()
        .to(device)
    )
    outputs, observations = [], []
    with torch.inference_mode():
        for index, (name, pixels) in enumerate(panels):
            inputs = pixels.to(device=device, dtype=dtype)
            encoded = model.encode(inputs)
            values = {}
            if class_name == "AutoencoderKL":
                posterior = encoded.latent_dist
                latents = posterior.mode()
                generator = torch.Generator(device=device).manual_seed(7116 + index)
                sampled = posterior.sample(generator=generator)
                values.update(
                    latent_mean=posterior.mean,
                    latent_logvar=posterior.logvar,
                    posterior_sample=sampled,
                    decoded_posterior_sample=model.decode(sampled).sample,
                )
                generator = torch.Generator(device=device).manual_seed(7116 + index)
                native = model(inputs, sample_posterior=True, generator=generator).sample
                torch.testing.assert_close(
                    native, values["decoded_posterior_sample"], atol=0, rtol=0
                )
            else:
                latents = encoded.latents
                values["encoded_latents"] = latents
                native = model(inputs).sample
            generator = torch.Generator(device=device).manual_seed(8116 + index)
            chosen = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype) / 2
            values.update(
                decoded_encoder_latents=model.decode(latents).sample,
                decoded_chosen_latents=model.decode(chosen).sample,
                native_forward=native,
            )
            if not all(bool(torch.isfinite(value).all()) for value in values.values()):
                raise ValueError(f"nonfinite autoencoder output for {name}")
            outputs.append({key: value.cpu() for key, value in values.items()})
            observations.append(
                {
                    "panel": name,
                    "input": _tensor_signature(pixels),
                    "chosen_latent_input": _tensor_signature(chosen),
                    "outputs": {key: _tensor_signature(value) for key, value in values.items()},
                }
            )
    return {"class": class_name, "outputs": outputs, "observations": observations}


def _compare(torch, reference, candidate):
    if reference["class"] != candidate["class"]:
        raise ValueError("candidate autoencoder class disagrees with reference")
    rows = list(zip(reference["observations"], candidate["observations"], strict=True))
    if any(
        any(left[key] != right[key] for key in ("panel", "input", "chosen_latent_input"))
        for left, right in rows
    ):
        raise ValueError("autoencoder evaluation inputs changed")
    comparisons = {}
    pairs = list(zip(reference["outputs"], candidate["outputs"], strict=True))
    for left, right in pairs:
        if left.keys() != right.keys():
            raise ValueError("autoencoder output fields changed")
        for field, values in left.items():
            other = right[field]
            if values.shape != other.shape:
                raise ValueError("autoencoder output shape changed")
            delta = values.double() - other.double()
            result = comparisons.setdefault(
                field,
                {
                    "coordinates": 0,
                    "exact_coordinates": 0,
                    "max_abs_error": 0.0,
                    "squared_error": 0.0,
                },
            )
            result["coordinates"] += values.numel()
            result["exact_coordinates"] += int((values == other).sum())
            result["max_abs_error"] = max(result["max_abs_error"], float(delta.abs().max()))
            result["squared_error"] += float(delta.square().sum())
    for result in comparisons.values():
        result["rmse"] = (result.pop("squared_error") / result["coordinates"]) ** 0.5
    return {
        "images": sum(left["input"]["shape"][0] for left, _ in rows),
        "input_panels": len(rows),
        "exact_input_panels": len(rows),
        "fields": comparisons,
    }


def evaluate(args):
    import diffusers
    import numpy as np
    import torch
    from PIL import __version__ as pillow_version

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    panels = _panels(torch, args.image)
    dtype = getattr(torch, args.dtype)
    reference, candidate = (
        _capture(torch, path, panels, dtype=dtype, device=args.device)
        for path in (args.reference, args.candidate)
    )
    audit = compare_checkpoints(_weight_file(args.reference), _weight_file(args.candidate))
    for field in ("reference", "candidate"):
        audit.pop(field)
    recovery = json.loads((args.candidate / "layer-recovery.json").read_text(encoding="utf-8"))
    for field in ("source", "checkpoint", "evidence_dir"):
        recovery.pop(field, None)
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "diffusers": diffusers.__version__,
            "numpy": np.__version__,
            "pillow": pillow_version,
            "device": args.device,
            "dtype": args.dtype,
        },
        "class": reference["class"],
        "reference_sha256": _checkpoint_hashes(args.reference),
        "candidate_sha256": _checkpoint_hashes(args.candidate),
        "public_image_sha256": file_sha256(args.image),
        "recovery": recovery,
        "raw_checkpoint_audit": audit,
        "held_out_behavior": _compare(torch, reference, candidate),
        "reference_observations": reference["observations"],
        "candidate_observations": candidate["observations"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if any(
        args.output.resolve().is_relative_to(path.resolve())
        for path in (args.reference, args.candidate)
    ):
        parser.error("output must be outside both checkpoint directories")
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
