"""Audit a reconstructed vision-language checkpoint on held-out image prompts."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from hashlib import sha256
from pathlib import Path

from rephix.io import file_sha256, write_json
from rephix.weight_audit import _inventory, compare_checkpoints

_LOCAL = {"local_files_only": True, "trust_remote_code": False}


def _held_out_panel():
    from PIL import Image, ImageDraw

    def picture():
        image = Image.new("RGB", (224, 192), "#f4f3ed")
        return image, ImageDraw.Draw(image)

    color_question = "What color is the shape? Answer with one word."
    red, draw = picture()
    draw.rectangle((48, 36, 176, 156), fill="#d32f2f")
    blue, draw = picture()
    draw.rectangle((48, 36, 176, 156), fill="#1964c0")
    balls, draw = picture()
    for left in (18, 83, 148):
        draw.ellipse((left, 67, left + 56, 123), fill="#0b7e51")
    split, draw = picture()
    draw.rectangle((12, 20, 106, 170), fill="#e8b900")
    draw.rectangle((116, 20, 212, 170), fill="#7e2ba8")
    nesting, draw = picture()
    draw.rectangle((44, 29, 180, 163), fill="#136f63")
    draw.ellipse((76, 60, 148, 132), fill="#f9e784")
    arrows, draw = picture()
    draw.polygon(
        [(18, 75), (142, 75), (142, 44), (206, 96), (142, 148), (142, 117), (18, 117)],
        fill="#232323",
    )
    return (
        ("red_rectangle", red, color_question),
        ("blue_rectangle", blue, color_question),
        ("green_circles", balls, "How many green circles do you see?"),
        ("colored_panels", split, "Which color is on the right?"),
        ("nested_shapes", nesting, "Which shape is inside the square?"),
        ("arrow", arrows, "Which direction does the arrow point?"),
    )


def _prompt(processor, question):
    if getattr(processor, "chat_template", None):
        message = {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
        return processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    token = getattr(processor, "image_token", None)
    return f"{token}\n{question}" if isinstance(token, str) and token else question


def _tensor_digest(torch, value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return sha256(raw.numpy().tobytes()).hexdigest()


def _signature(torch, inputs):
    tensors = {}
    for name, value in sorted(inputs.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Processor returned a non-tensor input: {name}")
        tensors[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_digest(torch, value),
        }
    return tensors


def _capture(torch, path, panel, *, dtype, device, max_new_tokens):
    from inspect import signature

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = (
        AutoModelForImageTextToText.from_pretrained(
            path, use_safetensors=True, dtype=dtype, **_LOCAL
        )
        .eval()
        .to(device)
    )
    if model.config.is_encoder_decoder:
        raise ValueError("This experiment requires an autoregressive vision-language model")
    processor = AutoProcessor.from_pretrained(path, **_LOCAL)
    forward_options = (
        {"logits_to_keep": 1} if "logits_to_keep" in signature(model.forward).parameters else {}
    )
    scores = []
    observations = []
    with torch.inference_mode():
        for label, image, question in panel:
            text = _prompt(processor, question)
            inputs = processor(text=text, images=[image], return_tensors="pt")
            inputs_signature = _signature(torch, inputs)
            prompt_ids = inputs["input_ids"][0].tolist()
            if "pixel_values" not in inputs or not inputs["pixel_values"].numel():
                raise ValueError(f"Image prompt did not produce model pixels: {label}")
            inputs = inputs.to(device=device, dtype=model.dtype)
            logits = model(**inputs, use_cache=True, **forward_options).logits[0, -1].cpu()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError(f"Nonfinite next-token logits: {label}")
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )[0, len(prompt_ids) :].tolist()
            scores.append(logits.clone())
            observations.append(
                {
                    "label": label,
                    "prompt": question,
                    "image_size": list(image.size),
                    "image_rgb_sha256": sha256(image.tobytes()).hexdigest(),
                    "processor_tensors": inputs_signature,
                    "prompt_token_ids": prompt_ids,
                    "next_token_top_1_id": int(logits.argmax()),
                    "next_token_logits_dtype": str(logits.dtype),
                    "next_token_logits_sha256": _tensor_digest(torch, logits),
                    "cached_greedy_ids": generated,
                    "cached_greedy_text": processor.decode(generated, skip_special_tokens=True),
                }
            )
    return torch.stack(scores), observations, str(model.dtype), type(model).__name__


def _compare(torch, reference, candidate):
    left, left_rows, _, _ = reference
    right, right_rows, _, _ = candidate
    if left.shape != right.shape:
        raise ValueError("Incompatible next-token logit shapes")
    delta = right.double() - left.double()
    pairs = list(zip(left_rows, right_rows, strict=True))
    # The first two cases share text and differ only in image color.
    image_control = {
        label: float((scores[0].double() - scores[1].double()).abs().max())
        for label, scores in (("reference", left), ("candidate", right))
    }
    return {
        "prompt_count": len(pairs),
        "exact_processor_outputs": sum(
            a["processor_tensors"] == b["processor_tensors"] for a, b in pairs
        ),
        "exact_prompt_token_ids": sum(
            a["prompt_token_ids"] == b["prompt_token_ids"] for a, b in pairs
        ),
        "next_token_top_1_matches": int((left.argmax(dim=1) == right.argmax(dim=1)).sum()),
        "next_token_logit_coordinates": left.numel(),
        "next_token_exact_logit_coordinates": int((left == right).sum()),
        "next_token_logit_max_abs_error": float(delta.abs().max()),
        "next_token_logit_rmse": float(delta.square().mean().sqrt()),
        "exact_cached_greedy_matches": sum(
            a["cached_greedy_ids"] == b["cached_greedy_ids"] for a, b in pairs
        ),
        "same_text_different_image_logit_max_abs_change": image_control,
    }


def _checkpoint_hashes(path):
    files = list(path.iterdir()) + list(path.glob("chat_templates/*.jinja"))
    return {
        file.relative_to(path).as_posix(): file_sha256(file)
        for file in sorted(files)
        if file.is_file()
        and file.name != "layer-recovery.json"
        and file.suffix in {".safetensors", ".json", ".jinja", ".txt", ".model", ".tiktoken"}
    }


def _audit(reference, candidate):
    report = compare_checkpoints(reference, candidate)
    for key in ("reference", "candidate"):
        report.pop(key)
    source_tensors, candidate_tensors = _inventory(Path(reference)), _inventory(Path(candidate))
    components = {}
    for name, tensor in source_tensors.items():
        parts = set(name.split("."))
        if parts & {"connector", "multi_modal_projector", "merger", "deepstack_merger_list"}:
            component = "projector"
        elif parts & {"vision_model", "vision_tower", "visual"}:
            component = "vision_encoder"
        else:
            component = "language_and_other"
        counts = components.setdefault(component, {"reference_elements": 0, "compared_elements": 0})
        counts["reference_elements"] += tensor.size
        other = candidate_tensors.get(name)
        if other is not None and other.shape == tensor.shape:
            counts["compared_elements"] += tensor.size
    report["components"] = components
    return report


def _native_reference_repack(reference, candidate, output_parent):
    from transformers import AutoModelForImageTextToText

    model, loading_info = AutoModelForImageTextToText.from_pretrained(
        reference, use_safetensors=True, dtype="auto", output_loading_info=True, **_LOCAL
    )
    for field in ("missing_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(field):
            raise ValueError(
                f"Reference loader could not normalize checkpoint names: {loading_info}"
            )
    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_parent, prefix="vlm-reference-repack-") as path:
        model.save_pretrained(path, safe_serialization=True)
        del model
        report = _audit(path, candidate)
    report["reference_preparation"] = "independent Transformers load in source dtype and save"
    report["reference_unexpected_keys"] = loading_info.get("unexpected_keys", [])
    return report


def evaluate(args):
    import torch
    import transformers
    from PIL import __version__ as pillow_version

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    panel = _held_out_panel()
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    reference, candidate = (
        _capture(
            torch,
            path,
            panel,
            dtype=dtype,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        for path in (args.reference, args.candidate)
    )
    audit = _audit(args.reference, args.candidate)
    repacked = None
    if audit["coverage"] != 1.0:
        repacked = _native_reference_repack(args.reference, args.candidate, args.output.parent)
    recovery = None
    sidecar = args.candidate / "layer-recovery.json"
    if sidecar.is_file():
        recovery = json.loads(sidecar.read_text(encoding="utf-8"))
        for key in ("source", "checkpoint", "evidence_dir"):
            recovery.pop(key, None)
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "pillow": pillow_version,
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "reference_dtype": reference[2],
            "candidate_dtype": candidate[2],
        },
        "reference_class": reference[3],
        "candidate_class": candidate[3],
        "reference_sha256": _checkpoint_hashes(args.reference),
        "candidate_sha256": _checkpoint_hashes(args.candidate),
        "recovery": recovery,
        "raw_checkpoint_audit": audit,
        "native_reference_repack_audit": repacked,
        "max_new_tokens": args.max_new_tokens,
        "held_out_behavior": _compare(torch, reference, candidate),
        "reference_observations": reference[1],
        "candidate_observations": candidate[1],
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float64", "bfloat16"), default="auto"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.max_new_tokens < 1:
        parser.error("threads and max-new-tokens must be positive")
    if any(
        args.output.resolve().is_relative_to(path.resolve())
        for path in (args.reference, args.candidate)
    ):
        parser.error("output must be outside both checkpoint directories")
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
