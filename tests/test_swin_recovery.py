"""Verify native Swin position observations and checkpoint replay."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from rephix.swin_recovery import (
    observe_swin_bias,
    observe_swin_index,
    observe_swin_logit_scale,
    recover_swin_bias,
    recover_swin_index,
    recover_swin_logit_scale,
    validate_swin_bias,
    validate_swin_index,
    validate_swin_logit_scale,
)

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class SwinRecoveryTests(unittest.TestCase):
    def _model(self, family):
        import torch
        from transformers import AutoConfig, AutoModelForImageClassification

        config = AutoConfig.for_model(
            family,
            image_size=32,
            patch_size=2,
            embed_dim=8,
            depths=[1, 1],
            num_heads=[2, 4],
            window_size=4,
            mlp_ratio=2,
            num_labels=5,
            drop_path_rate=0,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(122)
            model = AutoModelForImageClassification.from_config(config).double().eval()
        return model

    def test_native_swin_lookup_recovers_modified_index_and_all_table_rows(self):
        import torch

        source = self._model("swin").swin.encoder.layers[0].blocks[0].attention.self
        target = self._model("swin").swin.encoder.layers[0].blocks[0].attention.self
        with torch.no_grad():
            source.relative_position_bias_table.copy_(
                torch.arange(source.relative_position_bias_table.numel()).reshape_as(
                    source.relative_position_bias_table
                )
                / 32
            )
            source.relative_position_index.copy_(source.relative_position_index.flip(0))
            pending = {"relative_position_bias_table": target.relative_position_bias_table}
            validate_swin_bias(target, pending)
            validate_swin_index(target)

            def read(inputs, shape, operation):
                observe = (
                    observe_swin_bias
                    if operation == "relative_position_bias"
                    else observe_swin_index
                )
                values = observe(torch, source, inputs)
                self.assertEqual(values.shape, shape)
                return values

            recover_swin_bias(torch, target, pending, read, batch_size=7)
            recover_swin_index(torch, target, {"bias": target.relative_position_index}, read)
            torch.testing.assert_close(
                target.relative_position_bias_table,
                source.relative_position_bias_table,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                target.relative_position_index, source.relative_position_index, rtol=0, atol=0
            )
            source.relative_position_index[0, 0] = -1
            with self.assertRaisesRegex(ValueError, "unsupported Swin relative position index"):
                observe_swin_index(torch, source, torch.zeros(1, dtype=torch.long))

    def test_native_swinv2_clamp_probe_exposes_saturated_scales(self):
        import torch

        source = self._model("swinv2").swinv2.encoder.layers[0].blocks[0].attention.self
        target = self._model("swinv2").swinv2.encoder.layers[0].blocks[0].attention.self
        pixels = torch.linspace(-1, 1, 16 * 8, dtype=torch.float64).reshape(1, 16, 8)
        with torch.no_grad():
            source.logit_scale.copy_(torch.tensor([-7.0, 12.0]).reshape(2, 1, 1))
            original = source(pixels)[0]
            pending = {"logit_scale": target.logit_scale}
            validate_swin_logit_scale(target, pending)

            def read(inputs, shape, operation):
                self.assertEqual(operation, "unclamped_logit_scale")
                values = observe_swin_logit_scale(torch, source, inputs)
                self.assertEqual(values.shape, shape)
                return values

            recover_swin_logit_scale(torch, target, pending, read)
            torch.testing.assert_close(target.logit_scale, source.logit_scale, rtol=0, atol=0)
            torch.testing.assert_close(source(pixels)[0], original, rtol=0, atol=0)

    def test_native_checkpoints_export_and_replay_without_sources(self):
        import torch
        from transformers import AutoModelForImageClassification

        from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
        from rephix.weight_audit import compare_checkpoints

        for family in ("swin", "swinv2"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output, evidence, replay = (
                    root / name for name in ("source", "output", "evidence", "replay")
                )
                model = self._model(family)
                if family == "swin":
                    for module in model.modules():
                        if hasattr(module, "relative_position_bias_table"):
                            module.relative_position_index.copy_(
                                module.relative_position_index.flip(0)
                            )
                else:
                    with torch.no_grad():
                        for module in model.modules():
                            if hasattr(module, "logit_scale"):
                                module.logit_scale.fill_(12)
                model.save_pretrained(source)
                report = export_recovered_model(
                    source, output, dtype="float64", include_tokenizer=False, evidence_dir=evidence
                )
                self.assertEqual(
                    compare_checkpoints(source, output)["fraction_within_tolerance"], 1
                )
                shutil.rmtree(source)
                replay_report = replay_layer_evidence(evidence, replay)
                self.assertEqual(
                    replay_report["evidence_observations_cached"], report["layer_calls"]
                )
                self.assertEqual(replay_report["evidence_observations_collected"], 0)
                self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)
                candidate = AutoModelForImageClassification.from_pretrained(
                    replay, local_files_only=True, torch_dtype=torch.float64
                ).eval()
                pixels = torch.linspace(-1, 1, 3 * 32 * 32, dtype=torch.float64).reshape(
                    1, 3, 32, 32
                )
                with torch.inference_mode():
                    torch.testing.assert_close(
                        candidate(pixels).logits, model(pixels).logits, rtol=1e-13, atol=1e-13
                    )


if __name__ == "__main__":
    unittest.main()
