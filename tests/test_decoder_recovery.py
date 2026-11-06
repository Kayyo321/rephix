"""Verify native decoder probes and independent checkpoint reloads."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

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
class DecoderRecoveryTests(unittest.TestCase):
    def test_opt_recovers_reserved_position_rows_and_replays(self):
        import torch
        from transformers import AutoModelForCausalLM, OPTConfig

        config = OPTConfig(
            vocab_size=23,
            hidden_size=16,
            word_embed_proj_dim=12,
            ffn_dim=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=19,
            tie_word_embeddings=True,
        )
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(8)
            root = Path(directory)
            source, candidate, evidence = root / "source", root / "candidate", root / "evidence"
            model = AutoModelForCausalLM.from_config(config).eval()
            positions = model.model.decoder.embed_positions.weight
            with torch.no_grad():
                positions[0].fill_(1.25)
                positions[1].fill_(-0.625)
            model.save_pretrained(source)
            report = export_recovered_model(
                source, candidate, batch_size=7, include_tokenizer=False, evidence_dir=evidence
            )
            audit = compare_checkpoints(source, candidate)
            self.assertEqual(audit["coverage"], 1.0)
            self.assertEqual(audit["fraction_within_tolerance"], 1.0)
            source.rename(root / "removed")
            replay = root / "replay"
            replay_report = replay_layer_evidence(evidence, replay)
            self.assertEqual(replay_report["evidence_observations_collected"], 0)
            self.assertEqual(replay_report["evidence_observations_cached"], report["layer_calls"])
            self.assertEqual(compare_checkpoints(candidate, replay)["fraction_exactly_equal"], 1.0)

            recovered = AutoModelForCausalLM.from_pretrained(replay, local_files_only=True).eval()
            torch.testing.assert_close(
                recovered.model.decoder.embed_positions.weight, positions, rtol=0, atol=0
            )
            self.assertIs(
                recovered.get_input_embeddings().weight, recovered.get_output_embeddings().weight
            )
            tokens = torch.tensor([[1, 1, 2, 6, 7], [2, 3, 4, 6, 7]])
            mask = tokens.ne(1).long()
            with torch.no_grad():
                torch.testing.assert_close(
                    recovered(tokens, attention_mask=mask).logits[:, -1],
                    model(tokens, attention_mask=mask).logits[:, -1],
                )

    def test_falcon_bloom_olmo_and_gpt_j_recover_native_decoders(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        cases = {
            "falcon": dict(num_attention_heads=4, bias=True, alibi=True),
            "bloom": dict(n_head=4),
            "olmo": dict(num_attention_heads=4, num_key_value_heads=2, intermediate_size=24),
            "olmo2": dict(num_attention_heads=4, num_key_value_heads=2, intermediate_size=24),
            "olmo3": dict(num_attention_heads=4, num_key_value_heads=2, intermediate_size=24),
            "olmoe": dict(
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=24,
                num_experts=4,
                num_experts_per_tok=2,
            ),
            "gptj": dict(n_head=4, n_positions=23, rotary_dim=4),
        }
        for model_type, options in cases.items():
            with self.subTest(model_type=model_type):
                if model_type not in CONFIG_MAPPING:
                    self.skipTest(f"installed Transformers has no {model_type} architecture")
                config = AutoConfig.for_model(
                    model_type, vocab_size=23, hidden_size=16, num_hidden_layers=1, **options
                )
                with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as directory:
                    torch.manual_seed(15)
                    root = Path(directory)
                    source, candidate = root / "source", root / "candidate"
                    model = AutoModelForCausalLM.from_config(config).eval()
                    with torch.no_grad():
                        for name, parameter in model.named_parameters():
                            if name.endswith(".bias"):
                                parameter.fill_(0.03125)
                            elif parameter.ndim == 1:
                                parameter.copy_(torch.linspace(0.5, 1.25, parameter.numel()))
                    model.save_pretrained(source)
                    report = export_recovered_model(
                        source, candidate, batch_size=7, include_tokenizer=False
                    )
                    self.assertEqual(
                        report["unique_parameter_elements_recovered"],
                        sum(parameter.numel() for parameter in model.parameters()),
                    )
                    audit = compare_checkpoints(source, candidate)
                    self.assertEqual(audit["coverage"], 1.0)
                    self.assertEqual(audit["fraction_within_tolerance"], 1.0)
                    recovered = AutoModelForCausalLM.from_pretrained(
                        candidate, local_files_only=True
                    )
                    tokens = torch.tensor([[1, 3, 4, 8, 9]])
                    with torch.no_grad():
                        torch.testing.assert_close(recovered(tokens).logits, model(tokens).logits)

    def test_olmo2_normalization_recovers_large_epsilon_and_native_dtypes(self):
        import torch
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "olmo2" not in CONFIG_MAPPING:
            self.skipTest("installed Transformers has no OLMo 2 architecture")
        from transformers.models.olmo2.modeling_olmo2 import Olmo2RMSNorm

        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                source = Olmo2RMSNorm(9, eps=8).to(dtype=dtype)
                candidate = Olmo2RMSNorm(9, eps=8).to(dtype=dtype)
                values = torch.linspace(0.37, 1.51, 9, dtype=dtype)
                with torch.no_grad():
                    source.weight.copy_(values)
                report = recover_from_layer_calls(candidate, lambda _path, inputs: source(inputs))
                self.assertEqual(report["layer_calls"], 1)
                torch.testing.assert_close(candidate.weight, values, rtol=0, atol=0)
