"""Audit speech-to-text reconstruction on a pinned local LibriSpeech panel."""

import argparse
import json
import platform
from pathlib import Path

from whisper_recovery_experiment import _hash_tensor, _read_clips

from rephix.io import file_sha256, write_json
from rephix.weight_audit import compare_checkpoints


def _checkpoint_hashes(path):
    return {
        file.name: file_sha256(file)
        for file in sorted(path.iterdir())
        if file.is_file()
        and file.name != "layer-recovery.json"
        and file.suffix in {".json", ".model", ".safetensors", ".txt"}
    }


def _capture(torch, path, clips, *, dtype, device, max_tokens):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    local = {"local_files_only": True, "trust_remote_code": False}
    processor = AutoProcessor.from_pretrained(path, **local)
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(
            path, use_safetensors=True, torch_dtype=dtype, **local
        )
        .eval()
        .to(device)
    )
    config = model.generation_config
    if config.decoder_start_token_id is None:
        raise ValueError("the native generation config must specify the decoder start token")
    observations = []
    logits = []
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
            targets = processor.tokenizer(metadata["text"], add_special_tokens=False).input_ids
            prefix = [config.decoder_start_token_id, *targets]
            forced = (
                model(
                    **inputs,
                    decoder_input_ids=torch.tensor([prefix], device=device),
                    use_cache=False,
                )
                .logits[0]
                .cpu()
            )
            if not bool(torch.isfinite(forced).all()):
                raise ValueError(f"nonfinite teacher-forced logits for {metadata['id']}")
            generated = model.generate(
                **inputs, do_sample=False, num_beams=1, use_cache=True, max_new_tokens=max_tokens
            )[0].tolist()
            logits.append(forced)
            observations.append(
                {
                    "clip": metadata,
                    "processor_outputs": features,
                    "decoder_input_ids": prefix,
                    "teacher_forced_logits_sha256": _hash_tensor(forced),
                    "greedy_ids": generated,
                    "greedy_text": processor.batch_decode([generated], skip_special_tokens=True)[0],
                }
            )
    return observations, logits


def _compare(torch, reference, candidate):
    left, left_logits = reference
    right, right_logits = candidate
    pairs = list(zip(left, right, strict=True))
    if any(a["decoder_input_ids"] != b["decoder_input_ids"] for a, b in pairs):
        raise ValueError("the recovered tokenizer changed the teacher-forced decoder inputs")
    a, b = torch.cat(left_logits), torch.cat(right_logits)
    if a.shape != b.shape:
        raise ValueError("the recovered checkpoint changed the logit shape")
    delta = (a.double() - b.double()).abs()
    return {
        "clips": len(pairs),
        "exact_processor_outputs": sum(
            x["processor_outputs"] == y["processor_outputs"] for x, y in pairs
        ),
        "exact_greedy_generations": sum(x["greedy_ids"] == y["greedy_ids"] for x, y in pairs),
        "teacher_forced_positions": a.shape[0],
        "teacher_forced_top_1_matches": int((a.argmax(-1) == b.argmax(-1)).sum()),
        "teacher_forced_logit_coordinates": a.numel(),
        "teacher_forced_exact_logit_coordinates": int((a == b).sum()),
        "teacher_forced_max_abs_error": float(delta.max()),
        "teacher_forced_rmse": float(delta.square().mean().sqrt()),
    }


def evaluate(args):
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    clips = _read_clips(args.dataset)
    reference, candidate = (
        _capture(
            torch,
            path,
            clips,
            dtype=getattr(torch, args.dtype),
            device=args.device,
            max_tokens=args.max_new_tokens,
        )
        for path in (args.reference, args.candidate)
    )
    audit = compare_checkpoints(args.reference, args.candidate)
    for name in ("reference", "candidate"):
        audit.pop(name)
    report = args.candidate / "layer-recovery.json"
    recovery = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else None
    if recovery is not None:
        for name in ("source", "checkpoint", "evidence_dir"):
            recovery.pop(name, None)
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
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
        "max_new_tokens": args.max_new_tokens,
        "reference_sha256": _checkpoint_hashes(args.reference),
        "candidate_sha256": _checkpoint_hashes(args.candidate),
        "recovery": recovery,
        "raw_checkpoint_audit": audit,
        "held_out_behavior": _compare(torch, reference, candidate),
        "reference_observations": reference[0],
        "candidate_observations": candidate[0],
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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.threads < 1 or args.max_new_tokens < 1:
        parser.error("threads and max-new-tokens must be positive")
    if any(
        args.output.resolve().is_relative_to(path.resolve())
        for path in (args.reference, args.candidate)
    ):
        parser.error("output must be outside both checkpoints")
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
