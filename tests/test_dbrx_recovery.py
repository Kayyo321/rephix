"""Verify native flattened DBRX experts survive exact offline reconstruction."""

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
class DbrxRecoveryTests(unittest.TestCase):
    def test_flattened_experts_export_and_replay_exactly(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "dbrx" not in CONFIG_MAPPING:
            self.skipTest("installed Transformers has no native DBRX model")
        config = AutoConfig.for_model(
            "dbrx",
            d_model=16,
            n_heads=4,
            n_layers=2,
            max_seq_len=32,
            vocab_size=31,
            attn_config={"kv_n_heads": 2, "clip_qkv": 8.0},
            ffn_config={"ffn_hidden_size": 10, "moe_num_experts": 3, "moe_top_k": 2},
            eos_token_id=2,
            pad_token_id=0,
        )
        tokens = torch.tensor([[1, 4, 8, 13, 19], [1, 11, 8, 6, 23]])
        with tempfile.TemporaryDirectory() as tmp, torch.random.fork_rng(devices=[]):
            torch.manual_seed(89)
            root = Path(tmp)
            source, capture, replay = (root / name for name in ("source", "capture", "replay"))
            evidence = root / "evidence"
            model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).eval()
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
                replay, local_files_only=True, torch_dtype="auto"
            ).eval()
            with torch.inference_mode():
                original = model(tokens, output_router_logits=True)
                candidate = restored(tokens, output_router_logits=True)
                torch.testing.assert_close(candidate.logits, original.logits, rtol=0, atol=0)
                torch.testing.assert_close(candidate.router_logits, original.router_logits)
                options = {"max_new_tokens": 3, "do_sample": False, "use_cache": True}
                torch.testing.assert_close(
                    restored.generate(tokens, **options), model.generate(tokens, **options)
                )


if __name__ == "__main__":
    unittest.main()
