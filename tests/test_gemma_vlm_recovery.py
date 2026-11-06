"""Exercise native Gemma-family image projections and offline VLM replay."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from rephix.gemma_vlm_recovery import observe_gemma3_projector, recover_gemma3_projector
from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
from rephix.weight_audit import compare_checkpoints

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class GemmaVlmRecoveryTests(unittest.TestCase):
    def _model(self, family):
        import torch
        import transformers

        vision = transformers.SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=8,
            patch_size=2,
            vision_use_head=False,
        )
        text = dict(
            vocab_size=47,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=64,
            tie_word_embeddings=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        if family == "gemma3":
            if not hasattr(transformers, "Gemma3ForConditionalGeneration"):
                self.skipTest("installed Transformers has no Gemma 3 vision-language model")
            config = transformers.Gemma3Config(
                vision_config=vision,
                text_config=transformers.Gemma3TextConfig(
                    **text, query_pre_attn_scalar=4, sliding_window=32
                ),
                image_token_index=46,
                boi_token_index=44,
                eoi_token_index=45,
                mm_tokens_per_image=4,
            )
            cls, image_tokens = transformers.Gemma3ForConditionalGeneration, 4
        else:
            if not hasattr(transformers, "PaliGemmaForConditionalGeneration"):
                self.skipTest("installed Transformers has no PaliGemma vision-language model")
            config = transformers.PaliGemmaConfig(
                vision_config=vision,
                text_config=transformers.GemmaConfig(**text),
                image_token_index=46,
                projection_dim=16,
                hidden_size=16,
            )
            cls, image_tokens = transformers.PaliGemmaForConditionalGeneration, 16
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(501)
            model = cls(config).double().eval()
            # Dyadic values preserve exact affine and RMS gain arithmetic.
            for parameter in model.parameters():
                values = torch.randint(-32, 33, parameter.shape).double() / 128
                parameter.copy_(values)
        return model, image_tokens

    def test_native_projector_uses_the_matrix_after_normalization(self):
        import torch

        oracle = self._model("gemma3")[0].model.multi_modal_projector
        candidate = self._model("gemma3")[0].model.multi_modal_projector
        features = torch.arange(16 * 16, dtype=torch.float64).reshape(1, 16, 16) / 64
        with torch.no_grad():
            candidate.mm_input_projection_weight.zero_()
            baseline = oracle(features)
            calls = []

            def read(inputs, shape):
                observed = observe_gemma3_projector(oracle, inputs)
                self.assertEqual(observed.shape, shape)
                calls.append(len(inputs))
                return observed

            recover_gemma3_projector(torch, candidate, read, batch_size=7)
            self.assertEqual(calls, [7, 7, 2])
            torch.testing.assert_close(
                candidate.mm_input_projection_weight,
                oracle.mm_input_projection_weight,
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(oracle(features), baseline, atol=0, rtol=0)

    def test_full_checkpoint_replay_preserves_held_out_images_and_text(self):
        import torch

        for family in ("gemma3", "paligemma"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                model, image_tokens = self._model(family)
                root = Path(directory)
                source, candidate, evidence, replay = (
                    root / name for name in ("source", "candidate", "evidence", "replay")
                )
                model.save_pretrained(source, safe_serialization=True)
                report = export_recovered_model(
                    source,
                    candidate,
                    dtype="float64",
                    batch_size=13,
                    include_tokenizer=False,
                    evidence_dir=evidence,
                )
                audit = compare_checkpoints(source, candidate)
                self.assertGreater(audit["fraction_exactly_equal"], 0.99)
                self.assertEqual(audit["fraction_within_tolerance"], 1)
                oracle = (
                    type(model)
                    .from_pretrained(source, local_files_only=True, torch_dtype=torch.float64)
                    .eval()
                )
                shutil.rmtree(source)
                offline = replay_layer_evidence(evidence, replay)
                self.assertEqual(offline["evidence_observations_cached"], report["layer_calls"])
                self.assertEqual(offline["evidence_observations_collected"], 0)
                self.assertEqual(
                    compare_checkpoints(candidate, replay)["fraction_exactly_equal"], 1
                )
                restored = (
                    type(model)
                    .from_pretrained(replay, local_files_only=True, torch_dtype=torch.float64)
                    .eval()
                )
                input_ids = torch.tensor([[1] + [46] * image_tokens + [8, 4, 7]])
                pixels = torch.arange(3 * 8 * 8, dtype=torch.float64).reshape(1, 3, 8, 8) / 96
                kwargs = {
                    "input_ids": input_ids,
                    "pixel_values": pixels,
                    "attention_mask": torch.ones_like(input_ids),
                    "token_type_ids": (input_ids == 46).long(),
                }
                with torch.inference_mode():
                    original = oracle(**kwargs).logits
                    self.assertTrue(torch.isfinite(original).all())
                    torch.testing.assert_close(
                        restored(**kwargs).logits, original, atol=1e-14, rtol=1e-14
                    )
                    changed = oracle(**{**kwargs, "pixel_values": pixels.flip(-1)}).logits
                    self.assertFalse(torch.equal(changed[:, -1], original[:, -1]))
                    generation = dict(max_new_tokens=2, do_sample=False, pad_token_id=0)
                    torch.testing.assert_close(
                        restored.generate(**kwargs, **generation),
                        oracle.generate(**kwargs, **generation),
                        atol=0,
                        rtol=0,
                    )


if __name__ == "__main__":
    unittest.main()
