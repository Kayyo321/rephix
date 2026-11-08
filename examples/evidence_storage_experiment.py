"""Measure lossless journal compression and compact convolution observations.

Use a disposable, pinned local ConvNeXT V2 checkpoint. The optional source-free
check temporarily moves its directory, always restoring it before returning.
No downloaded or recovered model files belong in the committed result.
"""

import argparse
import copy
import hashlib
import json
import math
import platform
import sqlite3
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from rephix.io import file_sha256, write_json
from rephix.weight_audit import compare_checkpoints

MODES = (
    ("full_raw", False, False),
    ("full_zlib", False, True),
    ("compact_raw", True, False),
    ("compact_zlib", True, True),
)


def _sanitize(report):
    return {
        key: value
        for key, value in report.items()
        if key not in {"source", "checkpoint", "evidence_dir", "reference", "candidate"}
    }


def _checkpoint_hashes(path):
    return {
        file.name: file_sha256(file)
        for file in sorted(path.iterdir())
        if file.suffix in {".json", ".safetensors"} and file.name != "layer-recovery.json"
    }


def _journal_metrics(torch, path):
    with sqlite3.connect(path) as db:
        schema, total = db.execute(
            "SELECT schema_version, total_calls FROM journal_metadata"
        ).fetchone()
        rows = db.execute(
            "SELECT dtype, shape, length(payload), encoding FROM responses"
        ).fetchall()
    decoded = sum(
        math.prod(json.loads(shape))
        * torch.empty((), dtype=getattr(torch, dtype.removeprefix("torch."))).element_size()
        for dtype, shape, _, _ in rows
    )
    return {
        "schema_version": schema,
        "response_count": len(rows),
        "complete_total_calls": total,
        "database_bytes": path.stat().st_size,
        "decoded_payload_bytes": decoded,
        "stored_payload_bytes": sum(length for _, _, length, _ in rows),
        "encoding_counts": dict(sorted(Counter(encoding for _, _, _, encoding in rows).items())),
    }


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _panel(torch, reference):
    from PIL import Image, ImageDraw
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    options = {"local_files_only": True, "trust_remote_code": False}
    model = AutoModelForImageClassification.from_pretrained(reference, **options).eval()
    generator = torch.Generator().manual_seed(71551)
    image = Image.new("RGB", (263, 217), (13, 81, 160))
    draw = ImageDraw.Draw(image)
    draw.rectangle((17, 29, 151, 179), fill=(235, 169, 24))
    draw.ellipse((111, 31, 239, 173), fill=(31, 194, 126))
    processor = AutoImageProcessor.from_pretrained(reference, use_fast=False, **options)
    inputs = {
        "native_seeded": torch.randn((2, 3, 224, 224), generator=generator),
        "rectangular_seeded": torch.randn((2, 3, 232, 240), generator=generator),
        "processed_raster": processor(images=image, return_tensors="pt").pixel_values,
    }
    with torch.inference_mode():
        logits = {name: model(pixel_values=pixels).logits for name, pixels in inputs.items()}
    return inputs, logits, image


def _behavior(torch, checkpoint, inputs, baseline, image):
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    options = {"local_files_only": True, "trust_remote_code": False}
    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=False, **options)
    pixels = processor(images=image, return_tensors="pt").pixel_values
    if not torch.equal(pixels, inputs["processed_raster"]):
        raise AssertionError("candidate processor changed held-out pixels")
    model = AutoModelForImageClassification.from_pretrained(checkpoint, **options).eval()
    result = {}
    with torch.inference_mode():
        for name, values in inputs.items():
            scores = model(pixel_values=values).logits
            delta = scores - baseline[name]
            result[name] = {
                "images": len(values),
                "logit_count": scores.numel(),
                "exact_logits": int((scores == baseline[name]).sum()),
                "max_abs_error": float(delta.abs().max()),
                "top1_matches": int((scores.argmax(1) == baseline[name].argmax(1)).sum()),
            }
            if not torch.equal(scores, baseline[name]):
                raise AssertionError("candidate changed held-out logits")
    return result


def _ratios(runs):
    baseline = runs[0]
    for run in runs:
        run["relative_to_full_raw"] = {
            key: run["journal"][key] / baseline["journal"][key]
            for key in ("database_bytes", "decoded_payload_bytes", "stored_payload_bytes")
        }


def _synthetic(torch, root, batch_size):
    from rephix.convolution import observe_compact_convolution
    from rephix.layer_journal import LayerJournal
    from rephix.layer_recovery import recover_from_layer_calls

    results = []
    for label, channels, groups, kernel in (
        ("grouped_3x3", 8, 2, 3),
        ("depthwise_21x21", 4, 4, 21),
    ):
        source = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel, groups=groups, padding=kernel // 2).double()
        )
        generator = torch.Generator().manual_seed(9156)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.copy_(
                    torch.randint(-128, 129, parameter.shape, generator=generator).double() / 128
                )
        source_hashes = {name: _tensor_hash(value) for name, value in source.state_dict().items()}
        inputs = torch.randn((2, channels, kernel + 5, kernel + 7), generator=generator).double()
        with torch.inference_mode():
            expected = source(inputs)

        def observe(path, values):
            module_path, _, operation = path.partition(":")
            module = source.get_submodule(module_path)
            if operation:
                return observe_compact_convolution(torch, module, values, operation)
            return module(values)

        runs = []
        for mode, compact, compressed in MODES:
            path = root / f"{label}-{mode}.sqlite3"
            candidate = copy.deepcopy(source)
            with torch.no_grad():
                for parameter in candidate.parameters():
                    parameter.zero_()
            with LayerJournal(path, torch, binding=label, compression=compressed) as journal:
                recovery = recover_from_layer_calls(
                    candidate,
                    observe,
                    batch_size=batch_size,
                    compact_convolutions=compact,
                    _journal=journal,
                )
                journal.finish()
            hashes = {name: _tensor_hash(value) for name, value in candidate.state_dict().items()}
            with torch.inference_mode():
                output = candidate(inputs)
            if hashes != source_hashes or not torch.equal(output, expected):
                raise AssertionError("synthetic recovery changed weights or outputs")
            runs.append(
                {
                    "mode": mode,
                    "recovery": recovery,
                    "journal": _journal_metrics(torch, path),
                    "weights_bitwise_equal": True,
                    "held_out_outputs_bitwise_equal": True,
                }
            )
        _ratios(runs)
        results.append(
            {
                "case": label,
                "channels": channels,
                "groups": groups,
                "kernel_size": [kernel, kernel],
                "runs": runs,
            }
        )
    return results


def evaluate(args):
    import torch
    import transformers

    import rephix.layer_journal as journal_module
    import rephix.layer_recovery as recovery_module

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    reference = args.reference.resolve()
    root = args.work_dir.resolve()
    if root.exists() or reference == root or root in reference.parents or reference in root.parents:
        raise ValueError("work directory must be new and disjoint from the reference")
    root.mkdir(parents=True)
    reference_hashes = _checkpoint_hashes(reference)
    inputs, logits, image = _panel(torch, reference)
    journal_class = journal_module.LayerJournal
    recover = recovery_module.recover_from_layer_calls
    runs = []
    for mode, compact, compressed in MODES:
        candidate, evidence = root / f"{mode}-candidate", root / f"{mode}-evidence"
        options = dict(
            dtype="float64",
            device="cpu",
            batch_size=args.batch_size,
            restore_source_dtypes=True,
            compact_convolutions=compact,
            evidence_dir=evidence,
        )

        def make_journal(*values, **keywords):
            return journal_class(*values, **keywords, compression=compressed)

        interruption = None
        with patch.object(journal_module, "LayerJournal", side_effect=make_journal):
            if mode == "compact_zlib":

                def interrupt_after_commits(model, call_layer, **keywords):
                    calls = 0

                    def oracle(path, values):
                        nonlocal calls
                        if calls == args.interrupt_after:
                            raise InterruptedError("controlled evidence capture interruption")
                        result = call_layer(path, values)
                        calls += 1
                        return result

                    return recover(model, oracle, **keywords)

                with patch.object(
                    recovery_module, "recover_from_layer_calls", interrupt_after_commits
                ):
                    try:
                        recovery_module.export_recovered_model(reference, candidate, **options)
                    except InterruptedError:
                        pass
                    else:
                        raise AssertionError("controlled interruption did not occur")
                interruption = _journal_metrics(torch, evidence / "observations.sqlite3")
                if interruption["response_count"] != args.interrupt_after or candidate.exists():
                    raise AssertionError(
                        "interrupted capture did not retain exactly the committed prefix"
                    )
            report = recovery_module.export_recovered_model(reference, candidate, **options)
        audit = _sanitize(compare_checkpoints(reference, candidate))
        if audit["coverage"] != 1 or audit["fraction_exactly_equal"] != 1:
            raise AssertionError("real checkpoint recovery did not reproduce every stored value")
        metrics = _journal_metrics(torch, evidence / "observations.sqlite3")
        if metrics["decoded_payload_bytes"] != report["evidence_decoded_payload_bytes"]:
            raise AssertionError("journal decoded-byte accounting mismatch")
        if metrics["stored_payload_bytes"] != report["evidence_stored_payload_bytes"]:
            raise AssertionError("journal stored-byte accounting mismatch")
        run = {
            "mode": mode,
            "compact_convolutions": compact,
            "compression": compressed,
            "recovery": _sanitize(report),
            "journal": metrics,
            "raw_checkpoint_audit": audit,
            "checkpoint_sha256": _checkpoint_hashes(candidate),
            "held_out_behavior": _behavior(torch, candidate, inputs, logits, image),
        }
        if interruption:
            if report["evidence_observations_cached"] != args.interrupt_after:
                raise AssertionError("resumed capture did not reuse the complete committed prefix")
            run["interrupted_capture"] = interruption
        if runs and run["checkpoint_sha256"] != runs[0]["checkpoint_sha256"]:
            raise AssertionError("capture modes produced different checkpoint bytes")
        runs.append(run)
        print(json.dumps({"completed": mode, "journal": metrics}), flush=True)

    if args.verify_source_free:
        hidden_source = root / "temporarily-unavailable-source"
        reference.rename(hidden_source)
        try:
            if reference.exists():
                raise AssertionError("source remained available during replay")
            for run in runs:
                mode = run["mode"]
                replay = root / f"{mode}-replay"
                report = recovery_module.replay_layer_evidence(root / f"{mode}-evidence", replay)
                hashes = _checkpoint_hashes(replay)
                if hashes != run["checkpoint_sha256"]:
                    raise AssertionError("source-free replay changed checkpoint bytes")
                if report["evidence_observations_collected"] != 0:
                    raise AssertionError("source-free replay made new oracle queries")
                run["source_free_replay"] = {
                    "source_directory_unavailable": True,
                    "checkpoint_bytes_identical": True,
                    "recovery": _sanitize(report),
                    "held_out_behavior": _behavior(torch, replay, inputs, logits, image),
                }
        finally:
            hidden_source.rename(reference)
    _ratios(runs)
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": "cpu",
            "threads": args.threads,
        },
        "settings": {
            "batch_size": args.batch_size,
            "probe_dtype": "float64",
            "restore_source_dtypes": True,
            "interrupt_after_committed_responses": args.interrupt_after,
        },
        "reference_sha256": reference_hashes,
        "held_out_input_sha256": {name: _tensor_hash(value) for name, value in inputs.items()},
        "runs": runs,
        "synthetic_convolutions": _synthetic(torch, root, args.batch_size),
        "experiment_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--interrupt-after", type=int, default=17)
    parser.add_argument("--verify-source-free", action="store_true")
    args = parser.parse_args()
    if args.interrupt_after < 1:
        parser.error("--interrupt-after must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
