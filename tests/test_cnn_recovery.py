"""Verify native CNN recovery with stored normalization statistics and counters."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
from rephix.weight_audit import compare_checkpoints


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors")),
    "needs optional Transformers dependencies",
)
class CnnRecoveryTests(unittest.TestCase):
    def test_native_normalization_buffers_survive_source_free_replay(self):
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            ConvNextImageProcessor,
            EfficientNetConfig,
            EfficientNetImageProcessor,
            ResNetConfig,
        )

        configurations = (
            (
                "resnet",
                ResNetConfig(
                    embedding_size=8,
                    hidden_sizes=[8, 16],
                    depths=[1, 1],
                    layer_type="basic",
                    num_labels=5,
                ),
                ConvNextImageProcessor(size={"shortest_edge": 32}),
            ),
            (
                "efficientnet",
                EfficientNetConfig(
                    width_coefficient=0.25,
                    depth_coefficient=0.1,
                    depth_divisor=4,
                    hidden_dim=320,
                    kernel_sizes=[3, 5],
                    in_channels=[32, 16],
                    out_channels=[16, 24],
                    strides=[1, 2],
                    num_block_repeats=[1, 1],
                    expand_ratios=[1, 2],
                    num_labels=5,
                ),
                EfficientNetImageProcessor(size={"height": 32, "width": 32}),
            ),
        )
        for family, config, processor in configurations:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                with torch.random.fork_rng(devices=[]), torch.no_grad():
                    torch.manual_seed(102)
                    model = AutoModelForImageClassification.from_config(config).eval()
                    for module in model.modules():
                        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                            module.weight.copy_(torch.randint(-8, 9, module.weight.shape) / 32)
                        if type(module) is torch.nn.BatchNorm2d:
                            channels = module.num_features
                            module.weight.copy_(torch.linspace(-0.5, 0.75, channels))
                            module.bias.copy_(torch.linspace(-0.125, 0.25, channels))
                            module.running_mean.copy_(torch.linspace(-1.5, 0.75, channels))
                            module.running_var.copy_(torch.linspace(0.25, 2.5, channels))
                            module.num_batches_tracked.fill_((1 << 60) + 13)
                root = Path(directory)
                source, output, evidence, replay = (
                    root / name for name in ("source", "output", "evidence", "replay")
                )
                model.save_pretrained(source)
                processor.save_pretrained(source)
                report = export_recovered_model(
                    source,
                    output,
                    dtype="float64",
                    batch_size=17,
                    restore_source_dtypes=True,
                    evidence_dir=evidence,
                )
                audit = compare_checkpoints(source, output)
                self.assertEqual(audit["coverage"], 1)
                self.assertEqual(audit["fraction_within_tolerance"], 1)
                self.assertGreater(report["persistent_buffer_elements_recovered"], 0)
                shutil.rmtree(source)
                replayed = replay_layer_evidence(evidence, replay)
                self.assertEqual(replayed["evidence_observations_collected"], 0)
                self.assertEqual(replayed["evidence_observations_cached"], report["layer_calls"])
                self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)
                self.assertEqual(
                    (output / "model.safetensors").read_bytes(),
                    (replay / "model.safetensors").read_bytes(),
                )
                restored_processor = AutoImageProcessor.from_pretrained(
                    replay, local_files_only=True, use_fast=False
                )
                self.assertEqual(type(restored_processor), type(processor))
                candidate = AutoModelForImageClassification.from_pretrained(
                    replay, local_files_only=True
                ).eval()
                for module in candidate.modules():
                    if type(module) is torch.nn.BatchNorm2d:
                        self.assertEqual(module.num_batches_tracked.item(), (1 << 60) + 13)
                pixels = torch.linspace(-1, 1, 3 * 64 * 56).reshape(1, 3, 64, 56)
                with torch.inference_mode():
                    expected = model(pixel_values=pixels).logits
                    self.assertFalse(torch.equal(expected, model(pixel_values=-pixels).logits))
                    torch.testing.assert_close(
                        candidate(pixel_values=pixels).logits, expected, atol=1e-6, rtol=1e-5
                    )


if __name__ == "__main__":
    unittest.main()
