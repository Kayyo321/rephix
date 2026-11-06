"""Exercise independent passive trace collection and weights-free export."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rephix.trace_recovery import capture_layer_traces, recover_layer_traces

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class PassiveTraceTests(unittest.TestCase):
    def test_recovers_checkpoint_from_passive_traces_and_config_only(self):
        import torch
        from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

        for tied in (False, True):
            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as temporary:
                torch.manual_seed(37)
                rng = np.random.default_rng(29)
                root = Path(temporary)
                source, config, traces, output = (
                    root / name for name in ("source", "config", "traces", "recovered")
                )
                target = (
                    LlamaForCausalLM(
                        LlamaConfig(
                            vocab_size=24,
                            hidden_size=8,
                            intermediate_size=16,
                            num_hidden_layers=1,
                            num_attention_heads=2,
                            tie_word_embeddings=tied,
                            max_position_embeddings=32,
                        )
                    )
                    .double()
                    .eval()
                )
                target.save_pretrained(source, safe_serialization=True)
                config.mkdir()
                shutil.copyfile(source / "config.json", config / "config.json")
                train, validation = root / "train.npy", root / "val.npy"
                rows = np.vstack(
                    (rng.permutation(24).reshape(-1, 4), rng.integers(24, size=(32, 4)))
                )
                np.save(train, rows, allow_pickle=False)
                np.save(validation, rng.integers(24, size=(12, 4)), allow_pickle=False)
                capture = capture_layer_traces(source, train, validation, traces)
                self.assertEqual(capture["train_counts"][0], 24)
                source.rename(root / "offline-source")
                report = recover_layer_traces(config, traces, output, include_tokenizer=False)
                self.assertEqual(
                    report["unique_parameter_elements_recovered"],
                    sum(p.numel() for p in target.parameters()),
                )
                candidate = AutoModelForCausalLM.from_pretrained(
                    output, local_files_only=True, torch_dtype=torch.float64
                ).eval()
                for (name, reference), (other, fitted) in zip(
                    target.named_parameters(), candidate.named_parameters(), strict=True
                ):
                    self.assertEqual(name, other)
                    torch.testing.assert_close(fitted, reference, atol=1e-7, rtol=1e-6)
                test = torch.tensor([[2, 0, 8, 16, 9]])
                with torch.no_grad():
                    torch.testing.assert_close(candidate(test).logits, target(test).logits)

    def test_missing_embedding_rows_prevent_export(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, traces, output = (root / name for name in ("source", "traces", "export"))
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=24,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    max_position_embeddings=16,
                )
            ).save_pretrained(source, safe_serialization=True)
            rng = np.random.default_rng(117)
            train, val = root / "train.npy", root / "val.npy"
            np.save(train, rng.integers(20, size=(20, 4)), allow_pickle=False)
            np.save(val, rng.integers(20, size=(8, 4)), allow_pickle=False)
            capture_layer_traces(source, train, val, traces)
            with self.assertRaisesRegex(ValueError, "embedding token coverage 20/24"):
                recover_layer_traces(source, traces, output, include_tokenizer=False)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
