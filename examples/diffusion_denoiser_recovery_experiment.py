"""Audit a reconstructed native DDPM denoiser and its scheduler trajectory."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

from rephix.io import file_sha256, write_json
from rephix.weight_audit import compare_checkpoints


def _digest(value):
    data = value.detach().cpu().contiguous()
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
    }


def _hashes(path):
    return {
        file.name: file_sha256(file)
        for file in sorted(path.iterdir())
        if file.is_file()
        and file.name != "layer-recovery.json"
        and file.suffix in {".safetensors", ".json"}
    }


def _capture(torch, path, *, dtype, device):
    from diffusers import DDPMScheduler, UNet2DModel

    model = (
        UNet2DModel.from_pretrained(
            path, local_files_only=True, use_safetensors=True, torch_dtype=dtype
        )
        .eval()
        .to(device)
    )
    scheduler = DDPMScheduler.from_pretrained(path, local_files_only=True)
    if model.config.in_channels != model.config.out_channels:
        raise ValueError("DDPM trajectory requires matching input and output channels")
    size = model.config.sample_size
    if not isinstance(size, int):
        raise ValueError("this panel requires a native square sample size")
    channels = model.config.in_channels
    shapes = ((1, channels, size, size), (1, channels, size, size + 8), (2, channels, size, size))
    train_steps = scheduler.config.num_train_timesteps
    times = (0, train_steps // 8, train_steps // 2, train_steps - 1)
    generator = torch.Generator(device="cpu").manual_seed(7117)
    observations, outputs = [], []
    with torch.inference_mode():
        for shape in shapes:
            sample = torch.randn(shape, generator=generator, dtype=torch.float32).to(
                device=device, dtype=dtype
            )
            for timestep in times:
                values = model(sample, timestep=timestep).sample
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("denoiser returned nonfinite values")
                outputs.append(values.cpu())
                observations.append(
                    {
                        "timestep": timestep,
                        "sample": _digest(sample),
                        "prediction": _digest(values),
                    }
                )

        scheduler.set_timesteps(12, device=device)
        trajectory_generator = torch.Generator(device=device).manual_seed(6128)
        state = torch.randn(
            (1, channels, size, size), generator=trajectory_generator, device=device, dtype=dtype
        )
        initial = _digest(state)
        trajectory, states = [], []
        for timestep in scheduler.timesteps:
            prediction = model(
                scheduler.scale_model_input(state, timestep), timestep=timestep
            ).sample
            state = scheduler.step(
                prediction, timestep, state, generator=trajectory_generator
            ).prev_sample
            if not bool(torch.isfinite(state).all()):
                raise ValueError("scheduler trajectory became nonfinite")
            states.append(state.cpu())
            trajectory.append(
                {
                    "timestep": int(timestep),
                    "prediction": _digest(prediction),
                    "state": _digest(state),
                }
            )
    return {
        "model_settings": {
            "sample_size": size,
            "in_channels": channels,
            "out_channels": model.config.out_channels,
        },
        "scheduler_config": dict(scheduler.config),
        "observations": observations,
        "outputs": outputs,
        "initial_state": initial,
        "trajectory": trajectory,
        "states": states,
    }


def _comparison(torch, left, right):
    for name in ("model_settings", "scheduler_config", "initial_state"):
        if left[name] != right[name]:
            raise ValueError(f"independent denoiser loads disagree on {name}")
    pairs = list(zip(left["observations"], right["observations"], strict=True))
    if any(a["timestep"] != b["timestep"] or a["sample"] != b["sample"] for a, b in pairs):
        raise ValueError("denoisers received different input coordinates")
    outputs = list(zip(left["outputs"], right["outputs"], strict=True))
    if any(a.shape != b.shape for a, b in outputs):
        raise ValueError("denoiser output shapes disagree")
    delta = torch.cat([(a.double() - b.double()).flatten() for a, b in outputs])
    trajectories = list(zip(left["trajectory"], right["trajectory"], strict=True))
    if any(a["timestep"] != b["timestep"] for a, b in trajectories):
        raise ValueError("scheduler timesteps disagree")
    state_pairs = list(zip(left["states"], right["states"], strict=True))
    state_delta = torch.cat([(a.double() - b.double()).flatten() for a, b in state_pairs])
    return {
        "denoiser_cases": len(pairs),
        "output_coordinates": delta.numel(),
        "exact_output_coordinates": sum(int((a == b).sum()) for a, b in outputs),
        "exact_denoiser_outputs": sum(torch.equal(a, b) for a, b in outputs),
        "output_max_abs_error": float(delta.abs().max()),
        "output_rmse": float(delta.square().mean().sqrt()),
        "trajectory_steps": len(state_pairs),
        "trajectory_state_coordinates": state_delta.numel(),
        "exact_trajectory_state_coordinates": sum(int((a == b).sum()) for a, b in state_pairs),
        "exact_trajectory_states": sum(torch.equal(a, b) for a, b in state_pairs),
        "exact_trajectory_predictions": sum(
            a["prediction"] == b["prediction"] for a, b in trajectories
        ),
        "trajectory_state_max_abs_error": float(state_delta.abs().max()),
    }


def evaluate(args):
    import diffusers
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference, candidate = (
        _capture(torch, path, dtype=getattr(torch, args.dtype), device=args.device)
        for path in (args.reference, args.candidate)
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
            "diffusers": diffusers.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "reference_sha256": _hashes(args.reference),
        "candidate_sha256": _hashes(args.candidate),
        "recovery": recovery,
        "raw_checkpoint_audit": audit,
        "model_settings": reference["model_settings"],
        "scheduler_config": reference["scheduler_config"],
        "held_out_behavior": _comparison(torch, reference, candidate),
        "reference_observations": reference["observations"],
        "candidate_observations": candidate["observations"],
        "initial_trajectory_state": reference["initial_state"],
        "reference_trajectory": reference["trajectory"],
        "candidate_trajectory": candidate["trajectory"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
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
