"""Verify chosen-layer reconstruction of dense and sparse Phi checkpoints."""

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
class PhiRecoveryTests(unittest.TestCase):
    def test_phi_dense_and_sparse_checkpoints_replay_without_source(self):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        common = {
            "vocab_size": 23,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32,
            "eos_token_id": 2,
            "pad_token_id": 0,
        }
        model_options = {
            # Older Phi implementations normalize projected Q/K before splitting heads.
            "phi": {
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "qk_layernorm": True,
                "layer_norm_eps": 1e-9,
            },
            "phi3": {"tie_word_embeddings": True},
            "phimoe": {
                "num_local_experts": 3,
                "num_experts_per_tok": 2,
                "attention_bias": True,
                "lm_head_bias": True,
                "rms_norm_eps": 1e-9,
            },
        }
        tokens = torch.tensor([[1, 3, 4, 6, 5], [1, 10, 5, 3, 6]])
        for model_type, options in model_options.items():
            with self.subTest(architecture=model_type), tempfile.TemporaryDirectory() as tmp:
                if model_type not in CONFIG_MAPPING:
                    self.skipTest(f"installed Transformers has no {model_type} architecture")
                config = CONFIG_MAPPING[model_type](**(common | options))
                root = Path(tmp)
                source, capture, replay = (root / name for name in ("source", "capture", "replay"))
                evidence = root / "evidence"
                with torch.random.fork_rng(devices=[]), torch.no_grad():
                    torch.manual_seed(178)
                    model = AutoModelForCausalLM.from_config(config).eval()
                    # Dyadic weights and biases isolate probe coverage from cancellation.
                    for parameter in model.parameters():
                        parameter.copy_(torch.randint(-16, 17, parameter.shape) / 256)
                    model.save_pretrained(source, safe_serialization=True)
                capture_report = export_recovered_model(
                    source, capture, batch_size=7, include_tokenizer=False, evidence_dir=evidence
                )
                self.assertEqual(
                    capture_report["unique_parameter_elements_recovered"],
                    sum(parameter.numel() for parameter in model.parameters()),
                )
                self.assertEqual(compare_checkpoints(source, capture)["fraction_exactly_equal"], 1)
                shutil.rmtree(source)
                replay_report = replay_layer_evidence(evidence, replay)
                self.assertEqual(replay_report["evidence_observations_collected"], 0)
                self.assertEqual(
                    replay_report["evidence_observations_cached"], capture_report["layer_calls"]
                )
                self.assertEqual(compare_checkpoints(capture, replay)["fraction_exactly_equal"], 1)

                recovered = AutoModelForCausalLM.from_pretrained(replay, local_files_only=True)
                if config.tie_word_embeddings:
                    self.assertIs(
                        recovered.get_input_embeddings().weight,
                        recovered.get_output_embeddings().weight,
                    )
                forward_options = {"output_router_logits": True} if model_type == "phimoe" else {}
                with torch.inference_mode():
                    actual = recovered(tokens, **forward_options)
                    expected = model(tokens, **forward_options)
                    torch.testing.assert_close(actual.logits, expected.logits)
                    if model_type == "phimoe":
                        torch.testing.assert_close(actual.router_logits, expected.router_logits)
                    generation = {"max_new_tokens": 3, "do_sample": False, "use_cache": True}
                    torch.testing.assert_close(
                        recovered.generate(tokens, **generation),
                        model.generate(tokens, **generation),
                    )


if __name__ == "__main__":
    unittest.main()
