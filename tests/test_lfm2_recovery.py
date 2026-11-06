"""Recover native LFM2 hybrid checkpoints and reload them without the source."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
from rephix.weight_audit import compare_checkpoints

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class Lfm2RecoveryTests(unittest.TestCase):
    def test_hybrid_language_model_and_headless_vision_tower_replay(self):
        import torch
        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        for family in ("lfm2", "lfm2_vl"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temporary:
                if family not in CONFIG_MAPPING:
                    self.skipTest(f"installed Transformers has no {family}")
                text_config = transformers.Lfm2Config(
                    hidden_size=16,
                    intermediate_size=24,
                    vocab_size=47,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    layer_types=["conv", "full_attention"],
                    block_auto_adjust_ff_dim=False,
                    conv_L_cache=3,
                    conv_bias=True,
                )
                inputs = {"input_ids": torch.tensor([[1, 5, 6, 7]])}
                if family == "lfm2_vl":
                    config = transformers.Lfm2VlConfig(
                        text_config=text_config,
                        vision_config={
                            "model_type": "siglip2_vision_model",
                            "hidden_size": 16,
                            "intermediate_size": 24,
                            "num_hidden_layers": 1,
                            "num_attention_heads": 4,
                            "num_patches": 4,
                            "patch_size": 2,
                            "vision_use_head": False,
                        },
                        image_token_id=5,
                        projector_hidden_size=24,
                    )
                    inputs.update(
                        pixel_values=torch.linspace(-1, 1, 48, dtype=torch.float64).reshape(
                            1, 4, 12
                        ),
                        spatial_shapes=torch.tensor([[2, 2]]),
                        pixel_attention_mask=torch.ones(1, 4, dtype=torch.long),
                    )
                    wrapper = transformers.AutoModelForImageTextToText
                else:
                    config = text_config
                    wrapper = transformers.AutoModelForCausalLM

                with torch.random.fork_rng(devices=[]), torch.no_grad():
                    torch.manual_seed(708)
                    model = wrapper.from_config(config).double().eval()
                    for parameter in model.parameters():
                        parameter.copy_(torch.randint(-32, 33, parameter.shape).double() / 128)
                root = Path(temporary)
                source, output, evidence, replay = (
                    root / name for name in ("source", "output", "evidence", "replay")
                )
                model.save_pretrained(source)
                if family == "lfm2_vl":
                    # The publisher uses the spelling registered in AutoConfig.
                    path = source / "config.json"
                    native = json.loads(path.read_text(encoding="utf-8"))
                    native["model_type"] = family
                    path.write_text(json.dumps(native), encoding="utf-8")

                report = export_recovered_model(
                    source,
                    output,
                    dtype="float64",
                    batch_size=13,
                    include_tokenizer=False,
                    evidence_dir=evidence,
                )
                audit = compare_checkpoints(source, output)
                self.assertEqual(audit["coverage"], 1)
                self.assertEqual(audit["fraction_within_tolerance"], 1)
                shutil.rmtree(source)
                replayed = replay_layer_evidence(evidence, replay)
                self.assertEqual(replayed["evidence_observations_cached"], report["layer_calls"])
                self.assertEqual(replayed["evidence_observations_collected"], 0)
                self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)

                candidate = wrapper.from_pretrained(
                    replay, local_files_only=True, torch_dtype=torch.float64
                ).eval()
                with torch.inference_mode():
                    expected = model(**inputs, use_cache=False).logits
                    torch.testing.assert_close(
                        candidate(**inputs, use_cache=False).logits,
                        expected,
                        atol=1e-12,
                        rtol=1e-12,
                    )
                    if family == "lfm2_vl":
                        changed = {**inputs, "pixel_values": -inputs["pixel_values"]}
                        self.assertFalse(
                            torch.equal(model(**changed, use_cache=False).logits, expected)
                        )
                    options = {"max_new_tokens": 4, "do_sample": False, "use_cache": True}
                    self.assertTrue(
                        candidate.generate(**inputs, **options).equal(
                            model.generate(**inputs, **options)
                        )
                    )
