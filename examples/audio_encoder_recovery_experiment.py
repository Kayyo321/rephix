"""Audit native waveform encoder reconstruction on held-out LibriSpeech clips."""

import argparse
import json
import platform
from pathlib import Path

from whisper_recovery_experiment import _hash_tensor, _read_clips

from rephix.io import file_sha256, write_json
from rephix.weight_audit import _inventory, compare_checkpoints


def _checkpoint_hashes(path):
    return {
        file.name: file_sha256(file)
        for file in sorted(path.iterdir())
        if file.is_file()
        and file.name != "layer-recovery.json"
        and file.suffix in {".json", ".bin", ".safetensors", ".txt"}
    }


def _weight_norm_audit(reference, candidate):
    from safetensors import safe_open

    left, right = _inventory(reference), _inventory(candidate)
    values = equal = 0
    names = [name for name in left if name.endswith((".weight_g", ".weight_v"))]
    for name in names:
        if name not in right:
            raise ValueError("candidate omitted an original weight-normalization tensor")
        with (
            safe_open(left[name].shard, framework="pt") as source,
            safe_open(right[name].shard, framework="pt") as recovered,
        ):
            a, b = source.get_tensor(name), recovered.get_tensor(name)
            if a.shape != b.shape:
                raise ValueError("candidate changed a weight-normalization tensor shape")
            values += a.numel()
            equal += int((a == b).sum())
    return {"tensor_names": names, "stored_values": values, "exactly_equal_values": equal}


def _capture(torch, path, clips, *, dtype, device):
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoModel,
        AutoModelForCTC,
        AutoProcessor,
    )

    local = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(path, **local)
    prefix = {"wav2vec2": "Wav2Vec2", "hubert": "Hubert", "wavlm": "WavLM"}.get(config.model_type)
    architectures = set(config.architectures or ())
    if prefix is None or architectures not in ({f"{prefix}Model"}, {f"{prefix}ForCTC"}):
        raise ValueError("expected a native base audio encoder or CTC architecture")
    ctc = architectures == {f"{prefix}ForCTC"}
    wrapper = AutoModelForCTC if ctc else AutoModel
    processor = (AutoProcessor if ctc else AutoFeatureExtractor).from_pretrained(path, **local)
    model = wrapper.from_pretrained(path, torch_dtype=dtype, use_safetensors=True, **local).eval()
    model.to(device)
    outputs, observations = [], []
    with torch.inference_mode():
        for metadata, waveform in clips:
            inputs = processor(
                waveform, sampling_rate=16_000, return_attention_mask=True, return_tensors="pt"
            )
            features = {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": _hash_tensor(value),
                }
                for name, value in inputs.items()
            }
            inputs = inputs.to(device=device, dtype=model.dtype)
            result = model(**inputs)
            values = (result.logits if ctc else result.last_hidden_state)[0].cpu()
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"nonfinite output for {metadata['id']}")
            observed = {
                "clip": metadata,
                "processor_outputs": features,
                "output_shape": list(values.shape),
                "output_sha256": _hash_tensor(values),
            }
            if ctc:
                ids = values.argmax(-1).tolist()
                observed["frame_argmax_ids"] = ids
                observed["transcription"] = processor.batch_decode([ids])[0]
            outputs.append(values)
            observations.append(observed)
        silence = dict(inputs)
        silence["input_values"] = torch.zeros_like(inputs["input_values"])
        result = model(**silence)
        control = (result.logits if ctc else result.last_hidden_state)[0].cpu()
        audio_change = float((outputs[-1].double() - control.double()).abs().max())
    return {
        "task": "ctc" if ctc else "features",
        "observations": observations,
        "outputs": outputs,
        "audio_change": audio_change,
    }


def _compare(torch, reference, candidate):
    if reference["task"] != candidate["task"]:
        raise ValueError("candidate audio task disagrees with reference")
    pairs = list(zip(reference["observations"], candidate["observations"], strict=True))
    if any(a["output_shape"] != b["output_shape"] for a, b in pairs):
        raise ValueError("candidate audio output shape disagrees with reference")
    left, right = torch.cat(reference["outputs"]), torch.cat(candidate["outputs"])
    delta = left.double() - right.double()
    report = {
        "clips": len(pairs),
        "exact_processor_outputs": sum(
            a["processor_outputs"] == b["processor_outputs"] for a, b in pairs
        ),
        "output_coordinates": left.numel(),
        "exact_output_coordinates": int((left == right).sum()),
        "output_max_abs_error": float(delta.abs().max()),
        "output_rmse": float(delta.square().mean().sqrt()),
        "waveform_to_silence_max_abs_change": {
            "reference": reference["audio_change"],
            "candidate": candidate["audio_change"],
        },
    }
    if reference["task"] == "ctc":
        report.update(
            ctc_frames=left.shape[0],
            ctc_frame_argmax_matches=int((left.argmax(-1) == right.argmax(-1)).sum()),
            exact_transcriptions=sum(a["transcription"] == b["transcription"] for a, b in pairs),
        )
    return report


def evaluate(args):
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    clips = _read_clips(args.dataset)
    reference, candidate = (
        _capture(torch, path, clips, dtype=getattr(torch, args.dtype), device=args.device)
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
        "audio_task": reference["task"],
        "dataset_id": "hf-internal-testing/librispeech_asr_dummy",
        "dataset_revision": args.dataset_revision,
        "dataset_sha256": file_sha256(args.dataset),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "dtype": args.dtype,
        },
        "reference_sha256": _checkpoint_hashes(args.reference),
        "candidate_sha256": _checkpoint_hashes(args.candidate),
        "recovery": recovery,
        "raw_checkpoint_audit": audit,
        "weight_norm_audit": _weight_norm_audit(args.reference, args.candidate),
        "held_out_behavior": _compare(torch, reference, candidate),
        "reference_observations": reference["observations"],
        "candidate_observations": candidate["observations"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if any(
        args.output.resolve().is_relative_to(path.resolve())
        for path in (args.reference, args.candidate)
    ):
        parser.error("output must be outside both checkpoints")
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
