"""Verify chosen native expert banks and their offline replay."""

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
class FusedExpertRecoveryTests(unittest.TestCase):
    def test_grouped_and_individual_experts_export_and_replay(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        common = dict(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_key_value_heads=2,
            num_experts_per_tok=2,
            max_position_embeddings=32,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        cases = {
            "granitemoe": (torch.bfloat16, dict(num_attention_heads=4, num_local_experts=3)),
            "granitemoeshared": (
                torch.float32,
                dict(num_attention_heads=4, num_local_experts=3, shared_intermediate_size=8),
            ),
            "jetmoe": (torch.float32, dict(kv_channels=4, num_local_experts=3)),
            "qwen3_moe": (
                torch.float32,
                dict(num_attention_heads=4, num_experts=3, moe_intermediate_size=8),
            ),
        }
        tokens = torch.tensor([[1, 7, 3, 12, 8], [1, 4, 8, 9, 13]])
        for family, (dtype, options) in cases.items():
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                if family not in CONFIG_MAPPING:
                    self.skipTest(f"installed Transformers has no {family} architecture")
                config = AutoConfig.for_model(family, **common, **options)
                root = Path(directory)
                source, candidate = root / "source", root / "candidate"
                evidence, replay = root / "evidence", root / "replay"
                with torch.random.fork_rng(devices=[]), torch.no_grad():
                    torch.manual_seed(38)
                    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype).eval()
                    for name, parameter in model.named_parameters():
                        if name.endswith(".bias"):
                            parameter.fill_(0.125)
                        elif parameter.ndim == 3:
                            for expert, value in enumerate(parameter):
                                value.add_((expert + 1) / 16)
                    model.save_pretrained(source, safe_serialization=True)

                capture = export_recovered_model(
                    source,
                    candidate,
                    dtype="auto",
                    batch_size=7,
                    include_tokenizer=False,
                    evidence_dir=evidence,
                )
                count = sum(parameter.numel() for parameter in model.parameters())
                self.assertEqual(capture["unique_parameter_elements_recovered"], count)
                self.assertEqual(capture["observed_output_scalars"], count)
                self.assertEqual(
                    compare_checkpoints(source, candidate)["fraction_exactly_equal"], 1
                )
                shutil.rmtree(source)
                offline = replay_layer_evidence(evidence, replay)
                self.assertEqual(offline["evidence_observations_collected"], 0)
                self.assertEqual(offline["evidence_observations_cached"], capture["layer_calls"])
                self.assertEqual(
                    compare_checkpoints(candidate, replay)["fraction_exactly_equal"], 1
                )

                loaded = AutoModelForCausalLM.from_pretrained(
                    replay, local_files_only=True, torch_dtype=dtype
                ).eval()
                with torch.inference_mode():
                    torch.testing.assert_close(loaded(tokens).logits, model(tokens).logits)
                    # Read the backbone routers without invoking the wrapper's auxiliary loss.
                    actual = loaded.model(tokens, output_router_logits=True)
                    expected = model.model(tokens, output_router_logits=True)
                    torch.testing.assert_close(actual.router_logits, expected.router_logits)
                    generation = {"max_new_tokens": 3, "do_sample": False, "use_cache": True}
                    torch.testing.assert_close(
                        loaded.generate(tokens, **generation), model.generate(tokens, **generation)
                    )

    def test_jamba_hybrid_exports_and_replays_recurrent_state(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "jamba" not in CONFIG_MAPPING:
            self.skipTest("installed Transformers has no Jamba architecture")
        from transformers.models.jamba.modeling_jamba import HybridMambaAttentionDynamicCache

        config = AutoConfig.for_model(
            "jamba",
            vocab_size=31,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=3,
            num_experts_per_tok=2,
            max_position_embeddings=32,
            attn_layer_period=2,
            attn_layer_offset=1,
            expert_layer_period=1,
            expert_layer_offset=0,
            mamba_dt_rank=2,
            mamba_d_state=4,
            mamba_proj_bias=True,
            use_mamba_kernels=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, candidate = root / "source", root / "candidate"
            evidence, replay = root / "evidence", root / "replay"
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(39)
                model = AutoModelForCausalLM.from_config(config).eval()
                target_mixer = model.model.layers[0].mamba
                target_mixer.A_log.uniform_(0.3, 1.8)
                target_mixer.D.uniform_(-0.7, 1.0)
                for name, parameter in model.named_parameters():
                    if name.endswith(".bias"):
                        parameter.fill_(0.125)
                model.save_pretrained(source, safe_serialization=True)

            capture = export_recovered_model(
                source, candidate, batch_size=7, include_tokenizer=False, evidence_dir=evidence
            )
            audit = compare_checkpoints(source, candidate)
            self.assertGreaterEqual(audit["fraction_exactly_equal"], 0.95)
            self.assertEqual(audit["fraction_within_tolerance"], 1)
            self.assertEqual(len(capture["state_space_decays"]), 1)
            report = capture["state_space_decays"][0]
            self.assertEqual(
                report["effective_rate_elements_recovered"], target_mixer.A_log.numel()
            )
            shutil.rmtree(source)
            offline = replay_layer_evidence(evidence, replay)
            self.assertEqual(offline["evidence_observations_collected"], 0)
            self.assertEqual(compare_checkpoints(candidate, replay)["fraction_exactly_equal"], 1)

            loaded = AutoModelForCausalLM.from_pretrained(replay, local_files_only=True).eval()
            mixer = loaded.model.layers[0].mamba
            tokens = torch.tensor([[1, 7, 3, 12, 8], [1, 4, 8, 9, 13]])
            with torch.inference_mode():
                self.assertTrue(torch.equal(mixer.A_log.float().exp(), target_mixer.A_log.exp()))
                self.assertTrue(torch.equal(mixer.D, target_mixer.D))
                actual = loaded(tokens, output_router_logits=True, use_cache=False)
                expected = model(tokens, output_router_logits=True, use_cache=False)
                torch.testing.assert_close(actual.logits, expected.logits)
                torch.testing.assert_close(actual.router_logits, expected.router_logits)
                for decoder in (loaded, model):
                    cache = HybridMambaAttentionDynamicCache(
                        decoder.config, tokens.shape[0], dtype=torch.float32
                    )
                    prefill = decoder(tokens[:, :-1], past_key_values=cache, use_cache=True)
                    self.assertTrue(prefill.past_key_values.has_previous_state)
                    step = decoder(
                        tokens[:, -1:],
                        past_key_values=prefill.past_key_values,
                        cache_position=torch.tensor([tokens.shape[1] - 1]),
                        use_cache=True,
                    )
                    torch.testing.assert_close(
                        step.logits, expected.logits[:, -1:], atol=1e-6, rtol=1e-5
                    )
                generation = {"max_new_tokens": 3, "do_sample": False, "use_cache": True}
                torch.testing.assert_close(
                    loaded.generate(tokens, **generation), model.generate(tokens, **generation)
                )
