"""Verify native sparse router probes through export and source-free replay."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from rephix.deepseek_experts import observe_router
from rephix.layer_recovery import (
    export_recovered_model,
    recover_from_layer_calls,
    replay_layer_evidence,
)
from rephix.weight_audit import compare_checkpoints

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class DeepseekRecoveryTests(unittest.TestCase):
    def test_native_routing_and_persistent_correction_survive_offline_replay(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        common = {
            "vocab_size": 23,
            "hidden_size": 16,
            "intermediate_size": 24,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_group": 2,
            "topk_group": 1,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 0,
            "max_position_embeddings": 32,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
        mla = {
            "kv_lora_rank": 4,
            "q_lora_rank": 4,
            "qk_rope_head_dim": 4,
            "v_head_dim": 4,
            "qk_nope_head_dim": 4,
        }
        tokens = torch.tensor([[1, 3, 4, 6, 5], [1, 10, 5, 3, 6]])
        for model_type, dtype, options in (
            ("deepseek_v2", torch.float32, {**mla, "topk_method": "group_limited_greedy"}),
            ("deepseek_v3", torch.bfloat16, mla),
            ("glm4_moe", torch.bfloat16, {"use_qk_norm": True}),
        ):
            with self.subTest(architecture=model_type), tempfile.TemporaryDirectory() as tmp:
                if model_type not in CONFIG_MAPPING:
                    self.skipTest(f"installed Transformers has no {model_type} architecture")
                config = AutoConfig.for_model(model_type, **common, **options)
                root = Path(tmp)
                source, capture, replay = (root / name for name in ("source", "capture", "replay"))
                evidence = root / "evidence"
                with torch.random.fork_rng(devices=[]), torch.no_grad():
                    torch.manual_seed(267)
                    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype).eval()
                    for layer in model.model.layers:
                        gate = layer.mlp.gate
                        gate.weight.uniform_(-0.25, 0.25)
                        if hasattr(gate, "e_score_correction_bias"):
                            # Exercise mixed BF16/FP32 storage for GLM's router correction.
                            gate.e_score_correction_bias.copy_(
                                torch.tensor([0.123456, -0.373456, -0.81234, 0.587654])
                            )
                    model.save_pretrained(source, safe_serialization=True)

                report = export_recovered_model(
                    source,
                    capture,
                    dtype="auto",
                    batch_size=5,
                    include_tokenizer=False,
                    evidence_dir=evidence,
                )
                self.assertEqual(
                    report["unique_parameter_elements_recovered"],
                    sum(parameter.numel() for parameter in model.parameters()),
                )
                self.assertEqual(
                    report["persistent_buffer_elements_recovered"],
                    0 if model_type == "deepseek_v2" else 8,
                )
                audit = compare_checkpoints(source, capture)
                self.assertEqual(audit["fraction_exactly_equal"], 1)
                self.assertEqual(audit["dtype_mismatches"], {})
                shutil.rmtree(source)
                replay_report = replay_layer_evidence(evidence, replay)
                self.assertEqual(replay_report["evidence_observations_collected"], 0)
                self.assertEqual(
                    replay_report["evidence_observations_cached"], report["layer_calls"]
                )
                self.assertEqual(compare_checkpoints(capture, replay)["fraction_exactly_equal"], 1)

                recovered = AutoModelForCausalLM.from_pretrained(
                    replay, local_files_only=True, torch_dtype="auto"
                ).eval()
                with torch.inference_mode():
                    torch.testing.assert_close(recovered(tokens).logits, model(tokens).logits)
                    generation = {"max_new_tokens": 3, "do_sample": False, "use_cache": True}
                    torch.testing.assert_close(
                        recovered.generate(tokens, **generation),
                        model.generate(tokens, **generation),
                    )

    def test_fp64_router_observations_identify_fp32_representative_only(self):
        import torch
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "deepseek_v3" not in CONFIG_MAPPING:
            self.skipTest("installed Transformers has no DeepSeek V3 architecture")
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

        config = CONFIG_MAPPING["deepseek_v3"](
            hidden_size=3, n_routed_experts=4, n_group=2, topk_group=1, num_experts_per_tok=2
        )
        oracle, recovered = (DeepseekV3TopkRouter(config).double() for _ in range(2))
        with torch.no_grad():
            values = torch.linspace(0.25, 0.75, 12, dtype=torch.float64).reshape(4, 3) + 2**-44
            oracle.weight.copy_(values)

            def observe(path, inputs):
                return observe_router(torch, oracle, inputs, path.removeprefix(":"))

            report = recover_from_layer_calls(recovered, observe, batch_size=2)
            self.assertEqual(report["individually_identified_parameter_elements"], 0)
            self.assertEqual(
                report["router_rounding"],
                [{"module": "", "ambiguous_parameter_elements": oracle.weight.numel()}],
            )
            self.assertFalse(torch.equal(oracle.weight, recovered.weight))
            torch.testing.assert_close(
                oracle.weight.float(), recovered.weight.float(), atol=0, rtol=0
            )
            held_out = torch.tensor([[1.0, -0.5, 0.25], [0.25, 0.3, 0.4]], dtype=torch.float64)
            torch.testing.assert_close(oracle(held_out), recovered(held_out), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
