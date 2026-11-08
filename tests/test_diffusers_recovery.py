"""Verify native Diffusers checkpoint recovery and sealed source-free replay."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rephix.io import file_sha256
from rephix.weight_audit import compare_checkpoints

HAS_DIFFUSERS = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "safetensors", "diffusers")
)


@unittest.skipUnless(HAS_DIFFUSERS, "needs optional Diffusers dependencies")
class DiffusersRecoveryTests(unittest.TestCase):
    def _model(self, kind):
        import diffusers
        import torch

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(612)
            if kind == "AutoencoderKL":
                return diffusers.AutoencoderKL(
                    in_channels=3,
                    out_channels=3,
                    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
                    block_out_channels=(8, 16),
                    layers_per_block=1,
                    latent_channels=2,
                    norm_num_groups=2,
                    sample_size=8,
                ).eval()
            if kind == "AutoencoderTiny":
                return diffusers.AutoencoderTiny(
                    in_channels=3,
                    out_channels=3,
                    encoder_block_out_channels=(8, 8),
                    decoder_block_out_channels=(8, 8),
                    num_encoder_blocks=(1, 1),
                    num_decoder_blocks=(1, 1),
                    latent_channels=2,
                ).eval()
            if kind == "UNet2DModel":
                return diffusers.UNet2DModel(
                    sample_size=8,
                    in_channels=3,
                    out_channels=3,
                    down_block_types=("DownBlock2D", "AttnDownBlock2D"),
                    up_block_types=("AttnUpBlock2D", "UpBlock2D"),
                    block_out_channels=(8, 16),
                    layers_per_block=1,
                    norm_num_groups=2,
                    attention_head_dim=4,
                ).eval()
            return diffusers.UNet2DConditionModel(
                sample_size=8,
                in_channels=4,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                block_out_channels=(8, 16),
                layers_per_block=1,
                norm_num_groups=2,
                attention_head_dim=2,
                cross_attention_dim=6,
            ).eval()

    def _outputs(self, model, kind):
        import torch

        outputs = []
        channels = 4 if kind == "UNet2DConditionModel" else 3
        dtype = next(model.parameters()).dtype
        for width in (8, 12):
            image = torch.linspace(-1, 1, channels * 8 * width, dtype=dtype).reshape(
                1, channels, 8, width
            )
            if kind.startswith("Autoencoder"):
                encoded = model.encode(image)
                latent = encoded.latent_dist.mean if kind == "AutoencoderKL" else encoded.latents
                outputs.extend((latent, model.decode(latent).sample))
                changed = model.encode(-image)
                changed_latent = (
                    changed.latent_dist.mean if kind == "AutoencoderKL" else changed.latents
                )
                self.assertFalse(torch.equal(latent, changed_latent))
            else:
                options = {}
                if kind == "UNet2DConditionModel":
                    options["encoder_hidden_states"] = torch.linspace(
                        -1, 1, 18, dtype=dtype
                    ).reshape(1, 3, 6)
                values = model(image, 7, **options).sample
                outputs.append(values)
                self.assertFalse(torch.equal(values, model(-image, 7, **options).sample))
        return outputs

    def test_native_models_restore_exact_storage_and_replay_without_sources(self):
        import torch
        from diffusers import DDPMScheduler
        from safetensors.torch import load_file, save_file

        from rephix.diffusers_recovery import recover_diffusion_model, replay_diffusion_evidence

        kinds = ("AutoencoderKL", "AutoencoderTiny", "UNet2DModel", "UNet2DConditionModel")
        for kind in kinds:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, output, evidence, replay = (
                    root / name for name in ("source", "output", "evidence", "replay")
                )
                original = self._model(kind)
                original.save_pretrained(source, safe_serialization=True)
                if kind == "UNet2DModel":
                    weights = source / "diffusion_pytorch_model.safetensors"
                    state = {}
                    for name, value in load_file(weights).items():
                        for current, legacy in (
                            (".to_q.", ".query."),
                            (".to_k.", ".key."),
                            (".to_v.", ".value."),
                            (".to_out.0.", ".proj_attn."),
                        ):
                            name = name.replace(current, legacy)
                        state[name] = value
                    save_file(state, weights, metadata={"format": "pt"})
                    DDPMScheduler(num_train_timesteps=20).save_pretrained(source)
                with torch.inference_mode():
                    expected = self._outputs(original.double(), kind)
                report = recover_diffusion_model(
                    source,
                    output,
                    dtype="float64",
                    batch_size=7,
                    evidence_dir=evidence,
                    restore_source_dtypes=True,
                )
                self.assertEqual(report["architecture"], kind)
                self.assertEqual(compare_checkpoints(source, output)["fraction_exactly_equal"], 1)
                self.assertTrue((output / "diffusion_pytorch_model.safetensors").is_file())
                shutil.rmtree(source)
                with patch.object(
                    type(original),
                    "from_pretrained",
                    side_effect=AssertionError("replay loaded source weights"),
                ):
                    replay_report = replay_diffusion_evidence(evidence, replay)
                self.assertEqual(replay_report["evidence_observations_collected"], 0)
                self.assertEqual(
                    replay_report["evidence_observations_cached"], report["layer_calls"]
                )
                for name in ("diffusion_pytorch_model.safetensors", "config.json"):
                    self.assertEqual(file_sha256(output / name), file_sha256(replay / name))
                if kind == "UNet2DModel":
                    self.assertEqual(
                        file_sha256(output / "scheduler_config.json"),
                        file_sha256(replay / "scheduler_config.json"),
                    )
                    DDPMScheduler.from_pretrained(replay, local_files_only=True)
                candidate = (
                    type(original)
                    .from_pretrained(replay, torch_dtype=torch.float64, local_files_only=True)
                    .eval()
                )
                with torch.inference_mode():
                    actual = self._outputs(candidate, kind)
                for left, right in zip(expected, actual, strict=True):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_interrupted_capture_reuses_committed_observations(self):
        from rephix import diffusers_recovery as runtime

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output, evidence = (root / name for name in ("source", "output", "evidence"))
            self._model("UNet2DModel").save_pretrained(source, safe_serialization=True)
            recover = runtime.recover_from_layer_calls
            observed = []

            def interrupted(model, oracle, **options):
                def call(path, inputs):
                    observed.append(path)
                    if len(observed) == 5:
                        raise RuntimeError("injected diffusion capture interruption")
                    return oracle(path, inputs)

                return recover(model, call, **options)

            options = dict(
                dtype="float64", batch_size=7, evidence_dir=evidence, restore_source_dtypes=True
            )
            with patch.object(runtime, "recover_from_layer_calls", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "injected diffusion"):
                    runtime.recover_diffusion_model(source, output, **options)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "incomplete"):
                runtime.replay_diffusion_evidence(evidence, root / "incomplete")
            report = runtime.recover_diffusion_model(source, output, **options)
            self.assertEqual(report["evidence_observations_cached"], 4)
            self.assertEqual(report["evidence_observations_collected"], report["layer_calls"] - 4)
            self.assertEqual(compare_checkpoints(source, output)["fraction_exactly_equal"], 1)


if __name__ == "__main__":
    unittest.main()
