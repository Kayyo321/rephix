"""Check native Diffusers archive names, mixed precision, and runtime boundaries."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from functools import wraps
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rephix.diffusers_recovery import recover_diffusion_model, replay_diffusion_evidence
from rephix.io import file_sha256
from rephix.layer_bundle import prepare_bundle
from rephix.weight_audit import compare_checkpoints

HAS_DIFFUSERS = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "safetensors", "diffusers")
)


@unittest.skipUnless(HAS_DIFFUSERS, "needs optional Diffusers dependencies")
class DiffusersRuntimeTests(unittest.TestCase):
    def _model(self):
        import torch
        from diffusers import AutoencoderKL

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(991)
            return AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=("DownEncoderBlock2D",),
                up_block_types=("UpDecoderBlock2D",),
                block_out_channels=(8,),
                layers_per_block=1,
                latent_channels=2,
                norm_num_groups=2,
                sample_size=8,
            ).eval()

    def test_cli_capture_and_source_free_replay(self):
        from rephix.cli import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, evidence, replay = (
                root / name for name in ("source", "output", "evidence", "replay")
            )
            self._model().save_pretrained(source)
            captured, errors = StringIO(), StringIO()
            with redirect_stdout(captured), redirect_stderr(errors):
                status = main(
                    [
                        "--threads",
                        "1",
                        "recover-diffusion",
                        "--model",
                        str(source),
                        "--output",
                        str(output),
                        "--evidence-dir",
                        str(evidence),
                        "--dtype",
                        "float64",
                        "--restore-source-dtypes",
                        "--stage-layers",
                    ]
                )
            self.assertEqual(status, 0, errors.getvalue())
            capture = json.loads(captured.getvalue())
            self.assertEqual(capture["architecture"], "AutoencoderKL")
            self.assertEqual(compare_checkpoints(source, output)["fraction_exactly_equal"], 1)
            shutil.rmtree(source)
            captured, errors = StringIO(), StringIO()
            with redirect_stdout(captured), redirect_stderr(errors):
                status = main(
                    ["replay-diffusion", "--evidence-dir", str(evidence), "--output", str(replay)]
                )
            self.assertEqual(status, 0, errors.getvalue())
            report = json.loads(captured.getvalue())
            self.assertEqual(report["evidence_observations_collected"], 0)
            self.assertEqual(report["evidence_observations_cached"], capture["layer_calls"])
            self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)

    def test_legacy_projection_names_survive_native_reload_and_replay(self):
        import torch
        from diffusers import AutoencoderKL
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, evidence, replay = (
                root / name for name in ("source", "output", "evidence", "replay")
            )
            model = self._model()
            model.save_pretrained(source)
            state = load_file(source / "diffusion_pytorch_model.safetensors")
            replacements = (
                (".to_q.", ".query."),
                (".to_k.", ".key."),
                (".to_v.", ".value."),
                (".to_out.0.", ".proj_attn."),
            )
            legacy = {}
            for name, value in state.items():
                for current, old in replacements:
                    name = name.replace(current, old)
                legacy[name] = value
            save_file(
                legacy, source / "diffusion_pytorch_model.safetensors", metadata={"format": "pt"}
            )
            report = recover_diffusion_model(
                source, output, dtype="float64", restore_source_dtypes=True, evidence_dir=evidence
            )
            self.assertEqual(len(report["serialized_parameter_aliases"]), 16)
            self.assertEqual(compare_checkpoints(source, output)["fraction_exactly_equal"], 1)
            candidate = AutoencoderKL.from_pretrained(output, local_files_only=True).eval()
            pixels = torch.linspace(-1, 1, 192).reshape(1, 3, 8, 8)
            with torch.inference_mode():
                torch.testing.assert_close(
                    candidate(pixels).sample, model(pixels).sample, rtol=0, atol=0
                )
            shutil.rmtree(source)
            from rephix.layer_recovery import replay_layer_evidence

            with self.assertRaisesRegex(ValueError, "replay-diffusion"):
                replay_layer_evidence(evidence, root / "wrong-runtime")
            self.assertFalse((root / "wrong-runtime").exists())
            replay_diffusion_evidence(evidence, replay)
            for name in ("diffusion_pytorch_model.safetensors", "config.json"):
                self.assertEqual(file_sha256(output / name), file_sha256(replay / name))

    def test_auto_dtype_keeps_regenerated_float_buffers_at_load_precision(self):
        import torch
        from diffusers import AutoencoderKL

        from rephix import diffusers_recovery as runtime

        original_init = AutoencoderKL.__init__

        @wraps(original_init)
        def buffered_init(model, *args, **kwargs):
            original_init(model, *args, **kwargs)
            model.register_buffer(
                "regenerated_precision",
                torch.tensor([1.0001], dtype=torch.float32),
                persistent=False,
            )

        with (
            patch.object(AutoencoderKL, "__init__", buffered_init),
            tempfile.TemporaryDirectory() as directory,
        ):
            root = Path(directory)
            source, output, evidence, replay = (
                root / name for name in ("source", "output", "evidence", "replay")
            )
            model = self._model().half()
            for module in model.modules():
                if type(module) is torch.nn.GroupNorm:
                    module.float()
            model.save_pretrained(source)
            recover = runtime.recover_from_layer_calls
            inspected = []

            def check(scaffold, *args, **kwargs):
                self.assertEqual(scaffold.regenerated_precision.dtype, torch.float32)
                torch.testing.assert_close(
                    scaffold.regenerated_precision,
                    torch.tensor([1.0001], dtype=torch.float32),
                    rtol=0,
                    atol=0,
                )
                inspected.append(scaffold.dtype)
                return recover(scaffold, *args, **kwargs)

            with patch.object(runtime, "recover_from_layer_calls", side_effect=check):
                report = recover_diffusion_model(
                    source, output, dtype="auto", evidence_dir=evidence
                )
                shutil.rmtree(source)
                replay_diffusion_evidence(evidence, replay)
            self.assertEqual(report["oracle_dtype"], "torch.float16")
            self.assertEqual(inspected, [torch.float16, torch.float16])
            metadata = json.loads((evidence / "manifest.json").read_text())["metadata"]
            self.assertEqual(metadata["scaffold_dtype"], "torch.float32")
            self.assertEqual(
                file_sha256(output / "diffusion_pytorch_model.safetensors"),
                file_sha256(replay / "diffusion_pytorch_model.safetensors"),
            )

    def test_indexed_native_source_recovers_and_replays_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, evidence, replay = (
                root / name for name in ("source", "output", "evidence", "replay")
            )
            self._model().save_pretrained(source, max_shard_size="2KB")
            self.assertTrue((source / "diffusion_pytorch_model.safetensors.index.json").is_file())
            recover_diffusion_model(
                source, output, dtype="float64", restore_source_dtypes=True, evidence_dir=evidence
            )
            self.assertEqual(compare_checkpoints(source, output)["fraction_exactly_equal"], 1)
            shutil.rmtree(source)
            replay_diffusion_evidence(evidence, replay)
            for name in ("diffusion_pytorch_model.safetensors", "config.json"):
                self.assertEqual(file_sha256(output / name), file_sha256(replay / name))

    def test_foreign_runtime_and_ambiguous_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            self._model().save_pretrained(source)
            shutil.copyfile(
                source / "diffusion_pytorch_model.safetensors", source / "model.safetensors"
            )
            with self.assertRaisesRegex(ValueError, "unambiguous"):
                recover_diffusion_model(source, root / "output")
            self.assertFalse((root / "output").exists())

            def assets(destination):
                shutil.copyfile(source / "config.json", destination / "config.json")

            prepare_bundle(root / "foreign", {"runtime": "transformers"}, assets)
            with self.assertRaisesRegex(ValueError, "runtime"):
                replay_diffusion_evidence(root / "foreign", root / "replay")
            self.assertFalse((root / "replay").exists())


if __name__ == "__main__":
    unittest.main()
