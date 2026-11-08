"""Verify native segmentation and depth checkpoints and exact DPT positions."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from rephix.dense_vision_recovery import (
    observe_dpt_embeddings,
    recover_dpt_embeddings,
    validate_dpt_embeddings,
)
from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
from rephix.weight_audit import compare_checkpoints

HAS_HF = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors"))


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class DenseVisionRecoveryTests(unittest.TestCase):
    def _dpt_config(self):
        from transformers import DPTConfig

        return DPTConfig(
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=24,
            image_size=32,
            patch_size=8,
            backbone_out_indices=[0, 1, 2, 3],
            neck_hidden_sizes=[8, 8, 8, 8],
            fusion_hidden_size=8,
            reassemble_factors=[4, 2, 1, 0.5],
        )

    def test_dpt_class_and_positions_are_separately_identified(self):
        import torch
        from transformers.models.dpt.modeling_dpt import DPTViTEmbeddings

        source = DPTViTEmbeddings(self._dpt_config()).double().eval()
        target = DPTViTEmbeddings(self._dpt_config()).double().eval()
        with torch.no_grad():
            source.cls_token.copy_(torch.linspace(-1e-6, 1e-6, 16).reshape(1, 1, 16))
            source.position_embeddings.copy_(torch.arange(272).reshape(1, 17, 16) * 1e10)
            saved = {name: value.clone() for name, value in source.state_dict().items()}
            token = source.cls_token
            pending = dict(target.named_parameters(recurse=False))

            def read(inputs, shape, operation):
                values = observe_dpt_embeddings(torch, source, inputs, operation)
                self.assertEqual(tuple(values.shape), shape)
                return values

            validate_dpt_embeddings(target, pending)
            recover_dpt_embeddings(torch, target, pending, read)
            self.assertIs(source.cls_token, token)
            torch.testing.assert_close(target.cls_token, source.cls_token, atol=0, rtol=0)
            torch.testing.assert_close(
                target.position_embeddings, source.position_embeddings, atol=0, rtol=0
            )
            for name, value in source.state_dict().items():
                torch.testing.assert_close(value, saved[name], atol=0, rtol=0)

    def test_segmentation_and_depth_replay_preserve_full_pixel_outputs(self):
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForDepthEstimation,
            AutoModelForSemanticSegmentation,
            DPTForDepthEstimation,
            DPTImageProcessor,
            SegformerConfig,
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        segment_config = SegformerConfig(
            num_encoder_blocks=2,
            depths=[1, 1],
            sr_ratios=[2, 1],
            hidden_sizes=[8, 16],
            patch_sizes=[3, 3],
            strides=[2, 2],
            num_attention_heads=[1, 2],
            mlp_ratios=[2, 2],
            decoder_hidden_size=8,
            num_labels=3,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(105)
            cases = (
                (
                    SegformerForSemanticSegmentation(segment_config),
                    AutoModelForSemanticSegmentation,
                    SegformerImageProcessor(size={"height": 32, "width": 32}),
                    "logits",
                ),
                (
                    DPTForDepthEstimation(self._dpt_config()),
                    AutoModelForDepthEstimation,
                    DPTImageProcessor(size={"height": 32, "width": 32}),
                    "predicted_depth",
                ),
            )
        for model, wrapper, processor, field in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                model.eval()
                if field == "predicted_depth":
                    with torch.no_grad():
                        model.head.head[-2].weight.fill_(0.125)
                        model.head.head[-2].bias.zero_()
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
                self.assertEqual(audit["fraction_exactly_equal"], 1)
                shutil.rmtree(source)
                replayed = replay_layer_evidence(evidence, replay)
                self.assertEqual(replayed["evidence_observations_collected"], 0)
                self.assertEqual(replayed["evidence_observations_cached"], report["layer_calls"])
                self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)
                self.assertEqual(
                    (output / "model.safetensors").read_bytes(),
                    (replay / "model.safetensors").read_bytes(),
                )
                candidate = wrapper.from_pretrained(replay, local_files_only=True).eval()
                with torch.inference_mode():
                    pixels = torch.linspace(-1, 1, 3072).reshape(1, 3, 32, 32)
                    expected = getattr(model(pixels), field)
                    changed = getattr(model(-pixels), field)
                    self.assertFalse(torch.equal(expected, changed))
                    torch.testing.assert_close(
                        getattr(candidate(pixels), field), expected, atol=0, rtol=0
                    )
                reloaded = AutoImageProcessor.from_pretrained(replay, local_files_only=True)
                self.assertEqual(reloaded.size, processor.size)


if __name__ == "__main__":
    unittest.main()
