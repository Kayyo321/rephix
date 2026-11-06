"""Exercise native Gemma offset norms and unscaled text embedding recovery."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from rephix.gemma_recovery import GEMMA_OFFSET_RMS_NORMS, recover_gemma_rms
from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
from rephix.weight_audit import compare_checkpoints

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class GemmaRecoveryTests(unittest.TestCase):
    def test_native_text_exports_and_offline_replay(self):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM

        for config_name, dtype, tied in (
            ("GemmaConfig", torch.float32, True),
            ("Gemma2Config", torch.float64, True),
            ("Gemma3TextConfig", torch.bfloat16, False),
        ):
            with self.subTest(config=config_name), tempfile.TemporaryDirectory() as directory:
                config_class = getattr(transformers, config_name, None)
                if config_class is None:
                    self.skipTest(f"{config_name} is unavailable in this Transformers version")
                root = Path(directory)
                source, output = root / "source", root / "output"
                evidence, replayed = root / "evidence", root / "replayed"
                config = config_class(
                    vocab_size=37,
                    hidden_size=24,
                    intermediate_size=40,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=6,
                    max_position_embeddings=32,
                    tie_word_embeddings=tied,
                )
                with torch.random.fork_rng(devices=[]), torch.no_grad():
                    torch.manual_seed(456)
                    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype).eval()
                    for module in model.modules():
                        name = f"{type(module).__module__}.{type(module).__name__}"
                        if name in GEMMA_OFFSET_RMS_NORMS:
                            values = (torch.arange(module.weight.numel()) % 13 - 6) / 128
                            module.weight.copy_(values)
                model.save_pretrained(source, safe_serialization=True)
                report = export_recovered_model(
                    source,
                    output,
                    dtype="auto",
                    batch_size=13,
                    include_tokenizer=False,
                    evidence_dir=evidence,
                )
                self.assertEqual(compare_checkpoints(source, output)["fraction_exactly_equal"], 1)
                ambiguous = sum(
                    entry["ambiguous_parameter_elements"] for entry in report["gemma_rms_rounding"]
                )
                self.assertGreater(ambiguous, 0)
                self.assertEqual(
                    report["unique_parameter_elements_recovered"]
                    - report["individually_identified_parameter_elements"],
                    ambiguous,
                )
                shutil.rmtree(source)
                replay_report = replay_layer_evidence(evidence, replayed)
                self.assertEqual(
                    replay_report["evidence_observations_cached"], report["layer_calls"]
                )
                self.assertEqual(compare_checkpoints(output, replayed)["fraction_exactly_equal"], 1)
                self.assertEqual(replay_report["gemma_rms_rounding"], report["gemma_rms_rounding"])
                loaded = AutoModelForCausalLM.from_pretrained(
                    replayed, local_files_only=True, torch_dtype="auto"
                ).eval()
                tokens = torch.tensor([[2, 5, 11, 7, 19]])
                with torch.inference_mode():
                    torch.testing.assert_close(loaded(tokens).logits, model(tokens).logits)

    def test_offset_rounding_can_hide_native_parameters(self):
        import torch
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

        oracle, candidate = GemmaRMSNorm(4), GemmaRMSNorm(4)
        with torch.no_grad():
            oracle.weight.copy_(torch.tensor([1e-8, -1e-8, -0.25, -1.0]))
            ambiguous = recover_gemma_rms(torch, candidate, lambda inputs, _shape: oracle(inputs))
            self.assertEqual(ambiguous, 3)
            self.assertEqual(candidate.weight.tolist(), [0.0, 0.0, -0.25, -1.0])
            held_out = torch.tensor([[3.0, 1.0, -2.0, 0.5], [0.2, -0.7, 1.3, 2.5]])
            torch.testing.assert_close(oracle(held_out), candidate(held_out), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
