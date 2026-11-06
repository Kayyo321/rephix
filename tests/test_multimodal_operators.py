"""Exercise native video patch projections and learned vision tokens."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from rephix.convolution import recover_conv3d, validate_conv3d
from rephix.multimodal_operators import (
    LLAVA_IMAGE_NEWLINE_MODELS,
    observe_clip_class_embedding,
    observe_llava_image_newline,
    recover_clip_class_embedding,
    recover_llava_image_newline,
    validate_clip_class_embedding,
    validate_llava_image_newline,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs PyTorch")
class Conv3dRecoveryTests(unittest.TestCase):
    def test_recovers_grouped_patch_kernels_with_native_convolution(self):
        import torch

        for padding, bias in (("valid", False), ((1, 2, 3), True)):
            with self.subTest(padding=padding, bias=bias), torch.no_grad():
                options = {
                    "in_channels": 4,
                    "out_channels": 6,
                    "kernel_size": (2, 2, 3),
                    "stride": (2, 1, 3),
                    "padding": padding,
                    "dilation": (1, 2, 1),
                    "groups": 2,
                    "bias": bias,
                    "dtype": torch.float64,
                }
                source = torch.nn.Conv3d(**options)
                target = torch.nn.Conv3d(**options).to(memory_format=torch.channels_last_3d)

                def read(inputs, shape):
                    output = source(inputs)
                    self.assertEqual(output.shape, shape)
                    return output

                recover_conv3d(torch, target, dict(target.named_parameters()), read, batch_size=7)
                atol = 1e-15 if bias else 0
                torch.testing.assert_close(target.weight, source.weight, rtol=0, atol=atol)
                if bias:
                    torch.testing.assert_close(target.bias, source.bias, rtol=0, atol=0)
                video = torch.randn(2, 4, 7, 8, 9, dtype=torch.float64)
                torch.testing.assert_close(target(video), source(video), rtol=1e-13, atol=1e-13)

    def test_same_padding_is_rejected(self):
        import torch

        with self.assertRaisesRegex(ValueError, "Conv3d probes require numeric nonnegative"):
            validate_conv3d(torch.nn.Conv3d(1, 1, 3, padding="same"))


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "transformers")),
    "needs Transformers and PyTorch",
)
class ClipClassEmbeddingRecoveryTests(unittest.TestCase):
    def test_class_embedding_probe_is_exact_and_restores_position_lookup(self):
        import torch
        from transformers import CLIPVisionConfig
        from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings

        config = CLIPVisionConfig(hidden_size=16, num_channels=3, image_size=8, patch_size=4)
        source = CLIPVisionEmbeddings(config)
        target = CLIPVisionEmbeddings(config)
        pending = {"class_embedding": target.class_embedding}
        pixels = torch.zeros(1, 3, 8, 8)
        with torch.no_grad():
            source.patch_embedding.double()
            source.class_embedding.copy_(torch.linspace(-1e-5, 1e-5, 16))
            source.position_embedding.weight.fill_(1e16)
            original = source(pixels)

            def read(inputs, shape, operation):
                self.assertEqual(operation, "clip_class")
                values = observe_clip_class_embedding(torch, source, inputs)
                self.assertEqual(values.shape, shape)
                return values

            validate_clip_class_embedding(target, pending)
            recover_clip_class_embedding(target, pending, read)
            torch.testing.assert_close(
                target.class_embedding, source.class_embedding, rtol=0, atol=0
            )
            torch.testing.assert_close(source(pixels), original, rtol=0, atol=0)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors")),
    "needs optional Transformers dependencies",
)
class MultimodalRecoveryTests(unittest.TestCase):
    def test_native_llava_variants_expose_the_appended_newline(self):
        import torch
        import transformers
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        for family, wrapper_name in (
            ("llava_next", "LlavaNextForConditionalGeneration"),
            ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),
            ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ):
            with self.subTest(family=family), torch.no_grad():
                if family not in CONFIG_MAPPING:
                    self.skipTest(f"installed Transformers has no {family}")
                config = AutoConfig.for_model(
                    family,
                    text_config={
                        "model_type": "llama",
                        "vocab_size": 64,
                        "hidden_size": 24,
                        "intermediate_size": 48,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 2,
                    },
                    vision_config={
                        "model_type": (
                            "siglip_vision_model"
                            if family == "llava_onevision"
                            else "clip_vision_model"
                        ),
                        "hidden_size": 24,
                        "intermediate_size": 32,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 2,
                        "image_size": 8,
                        "patch_size": 4,
                    },
                    image_grid_pinpoints=[[8, 8]],
                )
                wrapper = getattr(transformers, wrapper_name)
                model = (
                    next(
                        module
                        for module in wrapper(config).modules()
                        if f"{type(module).__module__}.{type(module).__qualname__}"
                        in LLAVA_IMAGE_NEWLINE_MODELS
                        and "image_newline" in dict(module.named_parameters(recurse=False))
                    )
                    .eval()
                    .double()
                )
                expected = torch.linspace(-0.125, 0.25, 24, dtype=torch.float64)
                model.image_newline.copy_(expected)
                target = {"image_newline": torch.empty_like(expected)}
                tower_forward = model.vision_tower.forward
                projector_forward = model.multi_modal_projector.forward
                get_features = model.get_image_features
                pack_features = model.pack_image_features

                def read(inputs, shape, operation):
                    self.assertEqual(operation, "image_newline")
                    observed = observe_llava_image_newline(torch, model, inputs)
                    self.assertEqual(observed.shape, shape)
                    return observed

                validate_llava_image_newline(model, target)
                recover_llava_image_newline(model, target, read)
                torch.testing.assert_close(target["image_newline"], expected, atol=0, rtol=0)
                self.assertEqual(model.vision_tower.forward, tower_forward)
                self.assertEqual(model.multi_modal_projector.forward, projector_forward)
                self.assertEqual(model.get_image_features, get_features)
                self.assertEqual(model.pack_image_features, pack_features)

    def test_video_and_clip_image_models_export_and_replay(self):
        import torch
        import transformers
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
        from rephix.weight_audit import compare_checkpoints

        wrapper = getattr(transformers, "AutoModelForImageTextToText", None)
        if wrapper is None:
            self.skipTest("needs the native image-text model wrapper")
        text = {
            "vocab_size": 64,
            "hidden_size": 24,
            "intermediate_size": 48,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
        }
        for family in ("qwen3_vl", "llava", "llava_next"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temporary:
                if family not in CONFIG_MAPPING:
                    self.skipTest(f"installed Transformers has no {family}")
                if family == "qwen3_vl":
                    config = AutoConfig.for_model(
                        family,
                        text_config={
                            **text,
                            "head_dim": 12,
                            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 2, 2]},
                        },
                        vision_config={
                            "depth": 1,
                            "hidden_size": 24,
                            "intermediate_size": 32,
                            "out_hidden_size": 24,
                            "num_heads": 2,
                            "patch_size": 2,
                            "temporal_patch_size": 2,
                            "spatial_merge_size": 2,
                            "num_position_embeddings": 16,
                            "deepstack_visual_indexes": [0],
                        },
                        image_token_id=44,
                        video_token_id=43,
                        vision_start_token_id=45,
                        vision_end_token_id=46,
                    )
                    inputs = {
                        "input_ids": torch.tensor([[1, 45, 44, 44, 44, 44, 46, 5]]),
                        "pixel_values": torch.linspace(-1, 1, 384, dtype=torch.float64).reshape(
                            16, 24
                        ),
                        "image_grid_thw": torch.tensor([[1, 4, 4]]),
                    }
                else:
                    any_resolution = family == "llava_next"
                    config = AutoConfig.for_model(
                        family,
                        text_config={**text, "model_type": "llama"},
                        vision_config={
                            "model_type": "clip_vision_model",
                            "hidden_size": 24,
                            "intermediate_size": 32,
                            "num_hidden_layers": 1,
                            "num_attention_heads": 2,
                            "image_size": 8,
                            "patch_size": 4,
                        },
                        image_token_index=44,
                        image_seq_length=10 if any_resolution else 4,
                        vision_feature_layer=-1,
                        **({"image_grid_pinpoints": [[8, 8]]} if any_resolution else {}),
                    )
                    inputs = {
                        "input_ids": torch.tensor([[1, 44, 44, 44, 44, 5]]),
                        "pixel_values": torch.linspace(-1, 1, 192, dtype=torch.float64).reshape(
                            1, 3, 8, 8
                        ),
                    }
                    if any_resolution:
                        inputs = {
                            "input_ids": torch.tensor([[1, *([44] * 10), 5]]),
                            "pixel_values": inputs["pixel_values"]
                            .unsqueeze(1)
                            .repeat(1, 2, 1, 1, 1),
                            "image_sizes": torch.tensor([[8, 8]]),
                        }
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(7)
                    model = wrapper.from_config(config).eval().double()
                root = Path(temporary)
                source, output, evidence, replay = (
                    root / name for name in ("source", "output", "evidence", "replay")
                )
                model.save_pretrained(source, safe_serialization=True)
                report = export_recovered_model(
                    source, output, dtype="float64", include_tokenizer=False, evidence_dir=evidence
                )
                self.assertEqual(
                    report["unique_parameter_elements_recovered"],
                    sum(p.numel() for p in model.parameters()),
                )
                self.assertEqual(compare_checkpoints(source, output)["fraction_exactly_equal"], 1)

                source.rename(root / "source-offline")
                replay_report = replay_layer_evidence(evidence, replay)
                self.assertEqual(
                    replay_report["evidence_observations_cached"], report["layer_calls"]
                )
                self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)
                candidate = wrapper.from_pretrained(
                    replay, local_files_only=True, torch_dtype=torch.float64
                ).eval()
                with torch.inference_mode():
                    expected = model(**inputs, use_cache=False).logits
                    changed_image = {**inputs, "pixel_values": -inputs["pixel_values"]}
                    self.assertFalse(
                        torch.equal(expected, model(**changed_image, use_cache=False).logits)
                    )
                    torch.testing.assert_close(
                        candidate(**inputs, use_cache=False).logits,
                        expected,
                        atol=1e-12,
                        rtol=1e-12,
                    )


if __name__ == "__main__":
    unittest.main()
