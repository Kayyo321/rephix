"""Audit Whisper reconstruction on a local LibriSpeech dummy Parquet file."""

import argparse
import hashlib
import io
import json
import platform
import time
from pathlib import Path

from rephix.io import file_sha256, write_json
from rephix.weight_audit import compare_checkpoints

_SAMPLE_RATE = 16_000
_DATASET = "hf-internal-testing/librispeech_asr_dummy"


def _read_clips(path):
    import pyarrow.parquet as pq
    import soundfile as sf

    rows = pq.read_table(path, columns=["audio", "text", "id", "chapter_id", "speaker_id"])
    usable = []
    for row in rows.to_pylist():
        payload = row.pop("audio")["bytes"]
        info = sf.info(io.BytesIO(payload))
        if info.samplerate != _SAMPLE_RATE or info.channels != 1:
            raise ValueError(f"{row['id']}: expected mono 16 kHz audio")
        if 0 < info.frames <= 30 * _SAMPLE_RATE:
            usable.append((info.frames, row["id"], row, payload))
    usable.sort(key=lambda item: item[:2])
    if len({item[1] for item in usable}) < 5:
        raise ValueError("the audio panel requires at least five unique short clips")

    # Use the duration extremes and one median-length example per chapter.
    selected = {usable[0][1]: usable[0], usable[-1][1]: usable[-1]}
    chapters = sorted({item[2]["chapter_id"] for item in usable})
    for chapter in chapters[:3]:
        available = [
            item for item in usable if item[2]["chapter_id"] == chapter and item[1] not in selected
        ]
        if available:
            clip = available[len(available) // 2]
            selected[clip[1]] = clip
    for clip in usable:
        if len(selected) == 5:
            break
        selected.setdefault(clip[1], clip)

    clips = []
    for frames, _, metadata, payload in sorted(selected.values(), key=lambda item: item[:2]):
        waveform, _ = sf.read(io.BytesIO(payload), dtype="float32")
        clips.append(
            (
                {
                    **metadata,
                    "samples": frames,
                    "duration_seconds": frames / _SAMPLE_RATE,
                    "audio_sha256": hashlib.sha256(payload).hexdigest(),
                },
                waveform,
            )
        )
    return clips


def _long_waveform(clips):
    import numpy as np

    pieces = []
    clip_ids = []
    frames = 0
    for meta, waveform in reversed(clips):
        if pieces:
            pieces.append(np.zeros(_SAMPLE_RATE // 4, dtype=np.float32))
            frames += len(pieces[-1])
        pieces.append(waveform)
        clip_ids.append(meta["id"])
        frames += len(waveform)
        if frames > 32 * _SAMPLE_RATE:
            return np.concatenate(pieces), clip_ids
    raise ValueError("the selected clips do not total enough audio for long-form generation")


def _audio_inputs(processor, waveform, *, longform=False):
    return processor(
        audio=waveform,
        sampling_rate=_SAMPLE_RATE,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=not longform,
        padding="longest" if longform else "max_length",
    )


def _hash_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _check_audio(torch, reference, candidate):
    for field in ("input_features", "attention_mask"):
        if reference[field].dtype != candidate[field].dtype or not torch.equal(
            reference[field], candidate[field]
        ):
            raise ValueError(f"exported processor changed {field}")
    return {f"{field}_sha256": _hash_tensor(reference[field]) for field in reference}


def _speech_options(config):
    return {"language": "en", "task": "transcribe"} if config.is_multilingual else {}


def _decoder_inputs(processor, config, clips):
    options = _speech_options(config)
    prompt_ids = processor.get_decoder_prompt_ids(**options)
    if [pos for pos, _ in prompt_ids] != list(range(1, len(prompt_ids) + 1)):
        raise ValueError("Whisper prompt IDs do not form a contiguous decoder prefix")
    prefix = [config.decoder_start_token_id, *(token for _, token in prompt_ids)]
    targets = []
    for clip, _ in clips:
        text = " " + clip["text"].lower().capitalize()
        target = processor.tokenizer(text, add_special_tokens=False).input_ids
        targets.append([*target, config.eos_token_id])
    return prefix, targets


def _generate(model, features, mask, max_tokens, *, longform=False):
    options = _speech_options(model.generation_config)
    generated = model.generate(
        input_features=features,
        attention_mask=mask,
        use_cache=True,
        num_beams=1,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=max_tokens,
        condition_on_prev_tokens=longform,
        return_timestamps=longform,
        return_segments=longform,
        **options,
    )
    if longform:
        segments = [
            {
                "start_seconds": float(segment["start"]),
                "end_seconds": float(segment["end"]),
                "token_ids": segment["tokens"].cpu().tolist(),
            }
            for segment in generated["segments"][0]
        ]
        return {"token_ids": generated["sequences"][0].cpu().tolist(), "segments": segments}
    return {"token_ids": generated[0].cpu().tolist()}


def _model_outputs(torch, args, path, dtype, processor, features, prefix, targets, long_features):
    from transformers import AutoModelForSpeechSeq2Seq

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        path,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        trust_remote_code=False,
    ).eval()
    model.to(args.device)

    def input_tensors(batch):
        return (
            batch.input_features.to(device=args.device, dtype=model.dtype),
            batch.attention_mask.to(args.device),
        )

    scores, generations = [], []
    with torch.inference_mode():
        for batch, target in zip(features, targets):
            inputs, mask = input_tensors(batch)
            decoder_ids = torch.tensor([prefix + target[:-1]], device=args.device)
            logits = model(
                input_features=inputs,
                attention_mask=mask,
                decoder_input_ids=decoder_ids,
                use_cache=False,
                return_dict=True,
            ).logits[0, len(prefix) - 1 :]
            scores.append(logits.cpu().double())
            generations.append(_generate(model, inputs, mask, args.max_new_tokens))
        if long_features is not None:
            inputs, mask = input_tensors(long_features)
            generations.append(_generate(model, inputs, mask, args.max_new_tokens, longform=True))
    for result in generations:
        result["text"] = processor.decode(result["token_ids"], skip_special_tokens=True)
    return scores, generations


def _compare_logits(torch, reference, candidate):
    elements = tokens = argmax = 0
    squared_error = maximum = 0.0
    for a, b in zip(reference, candidate):
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("held-out logits have inconsistent shapes or non-finite values")
        diff = b - a
        elements += diff.numel()
        tokens += diff.shape[0]
        argmax += int((a.argmax(-1) == b.argmax(-1)).sum())
        maximum = max(maximum, float(diff.abs().max()))
        squared_error += float(diff.square().sum())
    return {
        "target_tokens": tokens,
        "logit_elements": elements,
        "argmax_matches": argmax,
        "logit_max_abs_error": maximum,
        "logit_rmse": (squared_error / elements) ** 0.5,
    }


def _generation_comparison(reference, candidate):
    return {
        "reference": reference,
        "candidate": candidate,
        "exact_token_match": reference["token_ids"] == candidate["token_ids"],
        "exact_text_match": reference["text"] == candidate["text"],
    }


def _file_hashes(directory):
    return {
        path.name: file_sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt"}
    }


def evaluate(args):
    import torch
    import transformers
    from transformers import GenerationConfig, WhisperProcessor

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    fit = json.loads((args.candidate / "layer-recovery.json").read_text(encoding="utf-8"))
    fit.pop("source", None)
    fit.pop("checkpoint", None)
    dtype = getattr(torch, args.inference_dtype or fit["oracle_dtype"].removeprefix("torch."))
    options = {"local_files_only": True, "trust_remote_code": False}
    paths = (args.reference, args.candidate)
    processors = [WhisperProcessor.from_pretrained(path, **options) for path in paths]
    configs = [GenerationConfig.from_pretrained(path, local_files_only=True) for path in paths]
    clips = _read_clips(args.dataset)
    a, b = [_decoder_inputs(p, c, clips) for p, c in zip(processors, configs)]
    if a != b:
        raise ValueError("exported tokenizer changed held-out decoder IDs")
    prefix, targets = a
    preprocessed = [[_audio_inputs(p, audio) for _, audio in clips] for p in processors]
    fixture_details = [
        {**clip, **_check_audio(torch, ref, cand), "teacher_forced_ids": target}
        for (clip, _), ref, cand, target in zip(clips, *preprocessed, targets)
    ]
    long_features = [None, None]
    long_metadata = None
    if args.longform:
        waveform, clip_ids = _long_waveform(clips)
        long_features = [_audio_inputs(p, waveform, longform=True) for p in processors]
        long_metadata = {
            "constituent_clip_ids": clip_ids,
            "silence_between_seconds": 0.25,
            "duration_seconds": len(waveform) / _SAMPLE_RATE,
            "mel_frames": long_features[0].input_features.shape[-1],
            **_check_audio(torch, *long_features),
        }

    ref_scores, ref_generated = _model_outputs(
        torch,
        args,
        paths[0],
        dtype,
        processors[0],
        preprocessed[0],
        prefix,
        targets,
        long_features[0],
    )
    if args.device == "cuda":
        torch.cuda.empty_cache()
    cand_scores, cand_generated = _model_outputs(
        torch,
        args,
        paths[1],
        dtype,
        processors[1],
        preprocessed[1],
        prefix,
        targets,
        long_features[1],
    )
    short_results = [
        {"id": clip["id"], **_generation_comparison(ref, cand)}
        for (clip, _), ref, cand in zip(clips, ref_generated, cand_generated)
    ]
    if long_metadata is not None:
        ref, cand = ref_generated[-1], cand_generated[-1]
        long_metadata.update(_generation_comparison(ref, cand))
        long_metadata["exact_timestamp_segments"] = ref["segments"] == cand["segments"]
    audit = compare_checkpoints(*paths)
    audit.pop("reference")
    audit.pop("candidate")
    reference_hashes, candidate_hashes = [_file_hashes(path) for path in paths]
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": args.device,
            "inference_dtype": str(dtype),
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "recovery": fit,
        "reference_sha256": reference_hashes,
        "candidate_sha256": candidate_hashes,
        "raw_checkpoint_audit": audit,
        "dataset": {
            "id": _DATASET,
            "config": "clean",
            "split": "validation",
            "revision": args.dataset_revision,
            "parquet_sha256": file_sha256(args.dataset),
            "sample_rate_hz": _SAMPLE_RATE,
            "clips": fixture_details,
        },
        "preprocessing": {
            "exact_input_features_and_masks": True,
            "exact_teacher_forced_ids": True,
            "preprocessor_config_byte_identical": (
                reference_hashes["preprocessor_config.json"]
                == candidate_hashes["preprocessor_config.json"]
            ),
            "input_shape_per_clip": list(preprocessed[0][0].input_features.shape),
        },
        "teacher_forced": {"prompt_ids": prefix, **_compare_logits(torch, ref_scores, cand_scores)},
        "free_running": {
            "examples": len(short_results),
            "cached": True,
            "max_new_tokens": args.max_new_tokens,
            "exact_token_matches": sum(r["exact_token_match"] for r in short_results),
            "exact_text_matches": sum(r["exact_text_match"] for r in short_results),
            "clips": short_results,
        },
        "longform": long_metadata,
        "evaluation_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--inference-dtype", choices=("float32", "float64"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--longform", action="store_true")
    args = parser.parse_args()
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
