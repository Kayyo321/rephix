"""Test native ConvNeXT probes and source-free replay."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
from rephix.weight_audit import compare_checkpoints


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors")),
    "needs optional Transformers dependencies",
)
class ConvNextRecoveryTests(unittest.TestCase):
    def test_channel_scales_recover_with_source_free_replay(self):
        import torch
        from transformers import AutoConfig, AutoModelForImageClassification

        for family in ("convnext", "convnextv2"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temporary:
                config = AutoConfig.for_model(
                    family,
                    patch_size=2,
                    hidden_sizes=[8, 16],
                    depths=[1, 1],
                    num_stages=2,
                    num_labels=5,
                    image_size=16,
                )
                with torch.random.fork_rng(devices=[]), torch.no_grad():
                    torch.manual_seed(734)
                    model = AutoModelForImageClassification.from_config(config).eval().double()
                    for path, parameter in model.named_parameters():
                        if path.endswith("layer_scale_parameter"):
                            parameter.copy_(torch.linspace(-0.2, 0.2, parameter.numel()).double())
                        elif ".grn." in path:
                            parameter.copy_(torch.randn_like(parameter) / 16)
                root = Path(temporary)
                source, output, evidence, replay = (
                    root / name for name in ("source", "output", "evidence", "replay")
                )
                model.save_pretrained(source)
                report = export_recovered_model(
                    source,
                    output,
                    dtype="float64",
                    batch_size=11,
                    include_tokenizer=False,
                    evidence_dir=evidence,
                )
                audit = compare_checkpoints(source, output)
                self.assertEqual(audit["coverage"], 1)
                self.assertEqual(audit["fraction_within_tolerance"], 1)
                self.assertGreater(audit["fraction_exactly_equal"], 0.99)
                source.rename(root / "source-offline")
                replay_report = replay_layer_evidence(evidence, replay)
                self.assertEqual(
                    replay_report["evidence_observations_cached"], report["layer_calls"]
                )
                self.assertEqual(replay_report["evidence_observations_collected"], 0)
                self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)

                candidate = AutoModelForImageClassification.from_pretrained(
                    replay, local_files_only=True, torch_dtype=torch.float64
                ).eval()
                pixels = torch.linspace(-1, 1, 3 * 12 * 16, dtype=torch.float64).reshape(
                    1, 3, 12, 16
                )
                with torch.inference_mode():
                    expected = model(pixel_values=pixels).logits
                    self.assertFalse(torch.equal(expected, model(pixel_values=-pixels).logits))
                    torch.testing.assert_close(
                        candidate(pixel_values=pixels).logits, expected, atol=1e-12, rtol=1e-12
                    )


if __name__ == "__main__":
    unittest.main()
