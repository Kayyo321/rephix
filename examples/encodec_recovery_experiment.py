"""Audit reconstructed Encodec codes and audio at native sample rates."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

from rephix.io import file_sha256, write_json
from rephix.weight_audit import compare_checkpoints


def _panel(rate, channels):
    import numpy as np

    cases = []
    for name, duration in (
        ("silence", 0.37),
        ("impulse", 0.51),
        ("tone", 0.67),
        ("chirp", 0.89),
        ("mixture", 1.11),
        ("noise", 1.31),
    ):
        t = np.arange(round(rate * duration), dtype=np.float64) / rate
        waves = []
        for channel in range(channels):
            if name == "silence":
                wave = np.zeros_like(t)
            elif name == "impulse":
                wave = np.zeros_like(t)
                wave[rate // 10 + 7 * channel] = 0.5
            elif name == "tone":
                wave = 0.2 * np.sin(2 * np.pi * (440 + 220 * channel) * t)
            elif name == "chirp":
                wave = 0.2 * np.sin(2 * np.pi * (100 * t + 1200 * np.square(t) / duration))
            elif name == "mixture":
                wave = 0.1 * np.sin(2 * np.pi * 220 * t) + 0.07 * np.sin(
                    2 * np.pi * 880 * t + channel
                )
            else:
                wave = np.random.default_rng(7116 + channel).normal(0, 0.08, len(t))
            waves.append(wave.astype(np.float32))
        values = waves[0] if channels == 1 else np.stack(waves)
        cases.append((name, values))
    return cases


def _digest(tensor):
    values = tensor.detach().cpu().contiguous()
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
    }


def _checkpoint_hashes(path):
    return {
        file.name: file_sha256(file)
        for file in sorted(path.iterdir())
        if file.is_file()
        and file.name != "layer-recovery.json"
        and file.suffix in {".safetensors", ".json"}
    }


def _run(model, inputs, bandwidth):
    encoded = model.encode(**inputs, bandwidth=bandwidth, return_dict=True)
    audio = model.decode(
        encoded.audio_codes,
        encoded.audio_scales,
        padding_mask=inputs.get("padding_mask"),
        last_frame_pad_length=encoded.last_frame_pad_length,
        return_dict=True,
    ).audio_values
    return encoded, audio


def _capture(torch, path, device):
    from transformers import AutoFeatureExtractor, AutoModel

    local = {"local_files_only": True, "trust_remote_code": False}
    model = (
        AutoModel.from_pretrained(path, use_safetensors=True, torch_dtype=torch.float32, **local)
        .eval()
        .to(device)
    )
    if model.config.model_type != "encodec":
        raise ValueError("this evaluator requires a native Encodec model")
    processor = AutoFeatureExtractor.from_pretrained(path, **local)
    rate, channels = model.config.sampling_rate, model.config.audio_channels
    bandwidths = [model.config.target_bandwidths[0], model.config.target_bandwidths[-1]]
    observations, codes, audio_values = [], [], []
    with torch.inference_mode():
        for name, waveform in _panel(rate, channels):
            prepared = processor(waveform, sampling_rate=rate, padding=True, return_tensors="pt")
            signatures = {key: _digest(value) for key, value in prepared.items()}
            inputs = prepared.to(device=device, dtype=torch.float32)
            for bandwidth in bandwidths:
                encoded, audio = _run(model, inputs, bandwidth)
                if not bool(torch.isfinite(audio).all()):
                    raise ValueError("codec returned nonfinite audio")
                observations.append(
                    {
                        "signal": name,
                        "bandwidth_kbps": bandwidth,
                        "input_samples": waveform.shape[-1],
                        "waveform_sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
                        "processor_outputs": signatures,
                        "codes": _digest(encoded.audio_codes),
                        "decoded_audio": _digest(audio),
                        "scales": [
                            None if value is None else _digest(value)
                            for value in encoded.audio_scales
                        ],
                        "last_frame_pad_length": encoded.last_frame_pad_length,
                    }
                )
                codes.append(encoded.audio_codes.cpu())
                audio_values.append(audio.cpu())
        changed = 0
        for module in model.modules():
            if type(module).__name__ == "EncodecEuclideanCodebook":
                for field in ("inited", "cluster_size", "embed_avg"):
                    values = getattr(module, field)
                    previous = values.clone()
                    values.add_(1)
                    changed += int((values != previous).sum())
        altered, altered_audio = _run(model, inputs, bandwidths[-1])
        inactive_control = {
            "changed_registered_values": changed,
            "codes_unchanged": torch.equal(altered.audio_codes.cpu(), codes[-1]),
            "decoded_audio_unchanged": torch.equal(altered_audio.cpu(), audio_values[-1]),
        }
    return {
        "sampling_rate": rate,
        "channels": channels,
        "bandwidths": bandwidths,
        "observations": observations,
        "codes": codes,
        "audio": audio_values,
        "inactive_control": inactive_control,
    }


def _compare(torch, left, right):
    if any(left[name] != right[name] for name in ("sampling_rate", "channels", "bandwidths")):
        raise ValueError("codec sample rates, channels, or bandwidths disagree")
    pairs = list(zip(left["observations"], right["observations"], strict=True))
    if any(
        any(
            a[name] != b[name]
            for name in ("signal", "bandwidth_kbps", "input_samples", "waveform_sha256")
        )
        for a, b in pairs
    ):
        raise ValueError("codec evaluation input cases disagree")
    if any(
        a["codes"]["shape"] != b["codes"]["shape"]
        or a["decoded_audio"]["shape"] != b["decoded_audio"]["shape"]
        for a, b in pairs
    ):
        raise ValueError("codec output shapes disagree")
    code_pairs = list(zip(left["codes"], right["codes"], strict=True))
    audio_pairs = list(zip(left["audio"], right["audio"], strict=True))
    delta = torch.cat([(a.double() - b.double()).flatten() for a, b in audio_pairs])
    return {
        "cases": len(pairs),
        "exact_processor_outputs": sum(
            a["processor_outputs"] == b["processor_outputs"] for a, b in pairs
        ),
        "exact_audio_scale_sequences": sum(a["scales"] == b["scales"] for a, b in pairs),
        "exact_padding_values": sum(
            a["last_frame_pad_length"] == b["last_frame_pad_length"] for a, b in pairs
        ),
        "code_coordinates": sum(a.numel() for a, b in code_pairs),
        "exact_code_coordinates": sum(int((a == b).sum()) for a, b in code_pairs),
        "exact_code_sequences": sum(torch.equal(a, b) for a, b in code_pairs),
        "decoded_coordinates": delta.numel(),
        "exact_decoded_coordinates": sum(int((a == b).sum()) for a, b in audio_pairs),
        "exact_decoded_waveforms": sum(torch.equal(a, b) for a, b in audio_pairs),
        "decoded_max_abs_error": float(delta.abs().max()),
        "decoded_rmse": float(delta.square().mean().sqrt()),
    }


def evaluate(args):
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference, candidate = (
        _capture(torch, path, args.device) for path in (args.reference, args.candidate)
    )
    audit = compare_checkpoints(args.reference, args.candidate)
    for key in ("reference", "candidate"):
        audit.pop(key)
    recovery = json.loads((args.candidate / "layer-recovery.json").read_text())
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
            "device": args.device,
            "dtype": "torch.float32",
        },
        "sampling_rate": reference["sampling_rate"],
        "audio_channels": reference["channels"],
        "bandwidths_kbps": reference["bandwidths"],
        "reference_sha256": _checkpoint_hashes(args.reference),
        "candidate_sha256": _checkpoint_hashes(args.candidate),
        "recovery": recovery,
        "raw_checkpoint_audit": audit,
        "held_out_behavior": _compare(torch, reference, candidate),
        "reference_observations": reference["observations"],
        "candidate_observations": candidate["observations"],
        "inactive_state_control": {
            "reference": reference["inactive_control"],
            "candidate": candidate["inactive_control"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if any(
        args.output.resolve().is_relative_to(path.resolve())
        for path in (args.reference, args.candidate)
    ):
        parser.error("output must be outside the checkpoint directories")
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
