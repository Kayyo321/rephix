"""Exercise coordinate probes and independent Transformers export."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from rephix.cli import main
from rephix.layer_recovery import recover_from_layer_calls

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class LayerRecoveryTests(unittest.TestCase):
    def test_solver_reads_only_layer_outputs(self):
        import torch

        skeleton = torch.nn.Sequential(
            torch.nn.Embedding(5, 3),
            torch.nn.Linear(3, 3),
            torch.nn.LayerNorm(3),
        )
        table = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        matrix = torch.tensor([[0.25, 0.5, -0.5], [0.5, 0.0, -0.5], [0.0, 1.0, 2.0]])
        bias = torch.tensor([0.5, -0.125, 0.25])
        gamma = torch.tensor([1.25, -0.25, 0.5])

        def oracle(path, x):
            if path == "0":
                return table[x]
            if path == "1":
                return x @ matrix.T + bias
            return torch.nn.functional.layer_norm(x, (3,), gamma, bias)

        report = recover_from_layer_calls(skeleton, oracle, batch_size=2)
        self.assertEqual(report["unique_parameter_elements_recovered"], 33)
        self.assertEqual(report["layer_calls"], 8)
        torch.testing.assert_close(skeleton[0].weight, table)
        torch.testing.assert_close(skeleton[1].weight, matrix)
        torch.testing.assert_close(skeleton[1].bias, bias)
        torch.testing.assert_close(skeleton[2].weight, gamma)
        torch.testing.assert_close(skeleton[2].bias, bias)

    def test_unsupported_state_is_rejected_before_oracle_calls(self):
        import torch

        def fail(*_):
            self.fail("invalid recovery plan should be rejected before probing")

        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Conv1d(2, 2, 1))
        with self.assertRaisesRegex(ValueError, "unsupported parameterized operator"):
            recover_from_layer_calls(model, fail)
        model = torch.nn.Linear(2, 2)
        model.register_buffer("offset", torch.ones(2))
        with self.assertRaisesRegex(ValueError, "persistent state"):
            recover_from_layer_calls(model, fail)

    def test_oracle_shape_and_finite_values_are_validated(self):
        import torch

        model = torch.nn.Linear(2, 3, bias=False)
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            recover_from_layer_calls(model, lambda path, x: x)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            recover_from_layer_calls(model, lambda path, x: torch.full((len(x), 3), float("nan")))

    def test_round_trip_exports_runnable_transformers(self):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            GPT2Config,
            GPTNeoXConfig,
            LlamaConfig,
            MistralConfig,
        )

        common = dict(
            vocab_size=19,
            hidden_size=12,
            intermediate_size=18,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            max_position_embeddings=16,
        )
        configs = (
            GPT2Config(vocab_size=19, n_positions=16, n_embd=12, n_layer=1, n_head=3),
            GPTNeoXConfig(
                **{key: value for key, value in common.items() if key != "num_key_value_heads"},
                rotary_pct=0.5,
            ),
            LlamaConfig(**common, tie_word_embeddings=True),
            MistralConfig(**common),
        )
        for config in configs:
            with self.subTest(config=config.model_type), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(12)
                source, output = Path(tmp) / "source", Path(tmp) / "recovered"
                target = AutoModelForCausalLM.from_config(config).eval()
                target.save_pretrained(source, safe_serialization=True)
                tokens = torch.tensor([[1, 3, 4, 6, 2, 8]])
                with torch.no_grad():
                    logits = target(tokens).logits
                stdout = StringIO()
                with redirect_stdout(stdout):
                    self.assertEqual(
                        main(
                            [
                                "recover-layers",
                                "--model",
                                str(source),
                                "--output",
                                str(output),
                                "--without-tokenizer",
                                "--batch-size",
                                "7",
                            ]
                        ),
                        0,
                    )
                report = json.loads(stdout.getvalue())
                self.assertEqual(
                    report["unique_parameter_elements_recovered"],
                    sum(p.numel() for p in target.parameters()),
                )
                stdout = StringIO()
                with redirect_stdout(stdout):
                    self.assertEqual(main(["audit-weights", str(source), str(output)]), 0)
                audit = json.loads(stdout.getvalue())
                self.assertEqual(audit["coverage"], 1.0)
                self.assertEqual(audit["fraction_within_tolerance"], 1.0)
                source.rename(Path(tmp) / "offline-source")
                recovered = AutoModelForCausalLM.from_pretrained(output, local_files_only=True)
                if config.tie_word_embeddings:
                    self.assertIs(
                        recovered.get_input_embeddings().weight,
                        recovered.get_output_embeddings().weight,
                    )
                with torch.no_grad():
                    torch.testing.assert_close(recovered(tokens).logits, logits)


if __name__ == "__main__":
    unittest.main()
