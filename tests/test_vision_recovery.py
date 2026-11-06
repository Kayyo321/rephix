"""Check standalone classifier export and explicit special-token gauges."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from rephix.layer_recovery import export_recovered_model

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class VisionRecoveryTests(unittest.TestCase):
    def test_vit_and_distilled_deit_export_at_native_and_interpolated_sizes(self):
        import torch
        from transformers import (
            AutoModelForImageClassification,
            DeiTConfig,
            DeiTForImageClassificationWithTeacher,
            ViTConfig,
            ViTForImageClassification,
        )

        common = dict(
            image_size=16,
            patch_size=4,
            hidden_size=12,
            num_attention_heads=3,
            num_hidden_layers=1,
            intermediate_size=24,
            num_labels=5,
        )
        cases = (
            (ViTForImageClassification, ViTConfig(**common), "vit.embeddings", ["cls_token"]),
            (
                DeiTForImageClassificationWithTeacher,
                DeiTConfig(**common),
                "deit.embeddings",
                ["cls_token", "distillation_token"],
            ),
        )
        for model_type, config, path, tokens in cases:
            with self.subTest(model=config.model_type), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(47)
                source, output = Path(tmp) / "source", Path(tmp) / "candidate"
                target = model_type(config).double().eval()
                embeddings = target.get_submodule(path)
                with torch.no_grad():
                    embeddings.position_embeddings.normal_(std=0.125)
                    for token in tokens:
                        getattr(embeddings, token).normal_(std=0.125)
                target.save_pretrained(source, safe_serialization=True)
                with self.assertRaisesRegex(ValueError, "no preprocessor_config.json"):
                    export_recovered_model(source, output, dtype="float64")
                processor = b'{"image_processor_type":"ViTImageProcessor","size":16}\n'
                (source / "preprocessor_config.json").write_bytes(processor)
                report = export_recovered_model(source, output, dtype="float64", batch_size=13)
                self.assertTrue(report["image_processor_exported"])
                self.assertEqual((output / "preprocessor_config.json").read_bytes(), processor)
                self.assertEqual(report["embedding_gauges"][0]["special_tokens"], tokens)
                self.assertEqual(report["embedding_gauges"][0]["free_dimensions"], len(tokens) * 12)
                self.assertEqual(
                    report["individually_identified_parameter_elements"],
                    sum(p.numel() for p in target.parameters()) - 2 * len(tokens) * 12,
                )

                source.rename(Path(tmp) / "source-offline")
                candidate = AutoModelForImageClassification.from_pretrained(
                    output, local_files_only=True, torch_dtype=torch.float64
                ).eval()
                fitted = candidate.get_submodule(path)
                prefix = len(tokens)
                self.assertEqual(
                    int(torch.count_nonzero(fitted.position_embeddings[:, :prefix])), 0
                )
                torch.testing.assert_close(
                    fitted.position_embeddings[:, prefix:],
                    embeddings.position_embeddings[:, prefix:],
                    atol=1e-10,
                    rtol=1e-8,
                )
                for index, token in enumerate(tokens):
                    torch.testing.assert_close(
                        getattr(fitted, token),
                        getattr(embeddings, token)
                        + embeddings.position_embeddings[:, index : index + 1],
                    )
                for name, value in candidate.named_parameters():
                    if name not in {f"{path}.{key}" for key in (*tokens, "position_embeddings")}:
                        torch.testing.assert_close(
                            value, target.get_parameter(name), atol=1e-10, rtol=1e-8
                        )
                with torch.inference_mode():
                    for shape, interpolation in (((16, 16), False), ((24, 20), True)):
                        pixels = torch.randn((2, 3, *shape), dtype=torch.float64)
                        kwargs = {"pixel_values": pixels, "interpolate_pos_encoding": interpolation}
                        torch.testing.assert_close(
                            candidate(**kwargs).logits,
                            target(**kwargs).logits,
                            atol=1e-10,
                            rtol=1e-8,
                        )


if __name__ == "__main__":
    unittest.main()
