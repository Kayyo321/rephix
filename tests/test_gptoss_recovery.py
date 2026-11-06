"""Verify full dense GPT-OSS recovery through virtual native operations."""

import importlib.util
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
class GptOssRecoveryTests(unittest.TestCase):
    def test_exact_dense_checkpoint_and_offline_replay(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "gpt_oss" not in CONFIG_MAPPING:
            self.skipTest("installed Transformers has no native GPT-OSS model")

        config = AutoConfig.for_model(
            "gpt_oss",
            vocab_size=31,
            hidden_size=16,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=3,
            num_experts_per_tok=2,
            head_dim=4,
            max_position_embeddings=32,
            rope_scaling=None,
            sliding_window=8,
            eos_token_id=2,
            pad_token_id=0,
        )
        tokens = torch.tensor([[1, 3, 4, 8, 10], [1, 13, 7, 5, 22]])
        with tempfile.TemporaryDirectory() as tmp, torch.random.fork_rng(devices=[]):
            torch.manual_seed(246)
            root = Path(tmp)
            source, capture, replay = (root / name for name in ("source", "capture", "replay"))
            evidence = root / "evidence"
            model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).eval()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.endswith(".bias") and "self_attn" in name:
                        parameter.zero_()
                    else:
                        parameter.copy_(torch.randint(-23, 24, parameter.shape) / 128)
            model.save_pretrained(source, safe_serialization=True)

            report = export_recovered_model(
                source,
                capture,
                dtype="auto",
                batch_size=7,
                include_tokenizer=False,
                evidence_dir=evidence,
            )
            self.assertEqual(
                report["unique_parameter_elements_recovered"],
                sum(parameter.numel() for parameter in model.parameters()),
            )
            audit = compare_checkpoints(source, capture)
            self.assertEqual(audit["fraction_exactly_equal"], 1)
            self.assertFalse(audit["dtype_mismatches"])
            shutil.rmtree(source)
            replay_report = replay_layer_evidence(evidence, replay)
            self.assertEqual(replay_report["evidence_observations_collected"], 0)
            self.assertEqual(replay_report["evidence_observations_cached"], report["layer_calls"])
            self.assertEqual(compare_checkpoints(capture, replay)["fraction_exactly_equal"], 1)

            restored = AutoModelForCausalLM.from_pretrained(
                replay,
                local_files_only=True,
                torch_dtype="auto",
            ).eval()
            with torch.inference_mode():
                options = {"output_router_logits": True, "output_attentions": True}
                original = model(tokens, **options)
                candidate = restored(tokens, **options)
                torch.testing.assert_close(candidate.logits, original.logits, rtol=0, atol=0)
                torch.testing.assert_close(candidate.router_logits, original.router_logits)
                torch.testing.assert_close(candidate.attentions, original.attentions)
                options = {"max_new_tokens": 3, "do_sample": False, "use_cache": True}
                torch.testing.assert_close(
                    restored.generate(tokens, **options), model.generate(tokens, **options)
                )


if __name__ == "__main__":
    unittest.main()
