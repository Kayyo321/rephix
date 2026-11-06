"""Exercise the endpoint evidence path, gate, and independent export."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from rephix.cli import main
from rephix.endpoint_head import recover_endpoint_head

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


def _save_scores(torch, model, path, ids, *, raw=False):
    path.mkdir()
    lengths = np.arange(len(ids)) % 3 + 3
    with torch.no_grad():
        scores = []
        for row, length in zip(ids, lengths):
            logits = model(torch.from_numpy(row[None, :length])).logits[0, -1]
            scores.append((logits if raw else logits.log_softmax(-1)).numpy())
    np.save(path / "input_ids.npy", ids, allow_pickle=False)
    np.save(path / "lengths.npy", lengths, allow_pickle=False)
    np.save(path / "scores.npy", np.stack(scores), allow_pickle=False)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class EndpointHeadTests(unittest.TestCase):
    def test_unanchored_cli_exports_dense_raw_and_sparse_logprob_heads(self):
        import torch
        from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

        for mode in ("raw_logits", "base_median"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(117)
                rng = np.random.default_rng(47)
                base, train, validation, output = (
                    Path(tmp) / name for name in ("base", "train", "validation", "export")
                )
                config = LlamaConfig(
                    vocab_size=29,
                    max_position_embeddings=16,
                    hidden_size=8,
                    intermediate_size=12,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                )
                target = LlamaForCausalLM(config).eval().double()
                target.save_pretrained(base, safe_serialization=True)
                head = target.get_output_embeddings()
                raw = mode == "raw_logits"
                with torch.no_grad():
                    selected = head.weight if raw else head.weight[:4]
                    selected.add_(torch.randn_like(selected) * 0.12)
                _save_scores(torch, target, train, rng.integers(0, 29, size=(40, 5)), raw=raw)
                _save_scores(torch, target, validation, rng.integers(0, 29, size=(12, 5)), raw=raw)
                stream = StringIO()
                with redirect_stdout(stream):
                    status = main(
                        [
                            "--threads",
                            "2",
                            "recover-endpoint-head",
                            "--base",
                            str(base),
                            "--train",
                            str(train),
                            "--validation",
                            str(validation),
                            "--output",
                            str(output),
                            "--mode",
                            mode,
                            "--dtype",
                            "float64",
                            "--without-tokenizer",
                        ]
                    )
                self.assertEqual(status, 0)
                report = json.loads(stream.getvalue())
                self.assertEqual(report["fit_mode"], mode)
                self.assertEqual(report["model_dtype"], "float64")
                metric = "logit" if raw else "contrast"
                self.assertLess(report[f"validation_runtime_{metric}_rmse"], 1e-10)
                candidate = AutoModelForCausalLM.from_pretrained(
                    output, local_files_only=True, torch_dtype=torch.float64
                )
                torch.testing.assert_close(
                    candidate.get_output_embeddings().weight, head.weight, atol=1e-11, rtol=1e-10
                )

    def test_exports_private_head_without_changing_tied_base_embeddings(self):
        import torch
        from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

        torch.manual_seed(110)
        rng = np.random.default_rng(35)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, train, validation, output = (
                root / name for name in ("base", "train", "validation", "export")
            )
            config = LlamaConfig(
                vocab_size=29,
                max_position_embeddings=16,
                hidden_size=8,
                intermediate_size=12,
                num_hidden_layers=1,
                num_attention_heads=2,
                tie_word_embeddings=True,
            )
            target = LlamaForCausalLM(config).eval()
            target.save_pretrained(base, safe_serialization=True)
            head = target.get_output_embeddings()
            input_table = head.weight.detach().clone()
            delta = torch.randn_like(head.weight) * 0.1
            delta[2].zero_()
            head.weight = torch.nn.Parameter(input_table + delta)
            target.config.tie_word_embeddings = False
            _save_scores(torch, target, train, rng.integers(0, 29, size=(40, 5)))
            _save_scores(torch, target, validation, rng.integers(0, 29, size=(12, 5)))
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = main(
                    [
                        "--threads",
                        "2",
                        "recover-endpoint-head",
                        "--base",
                        str(base),
                        "--train",
                        str(train),
                        "--validation",
                        str(validation),
                        "--output",
                        str(output),
                        "--anchor-id",
                        "2",
                        "--without-tokenizer",
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(stdout.getvalue())
            self.assertTrue(report["base_output_was_tied"])
            self.assertLess(report["validation_runtime_contrast_rmse"], 1e-5)
            base.rename(root / "base-offline")
            candidate = AutoModelForCausalLM.from_pretrained(output, local_files_only=True)
            self.assertFalse(candidate.config.tie_word_embeddings)
            torch.testing.assert_close(candidate.get_input_embeddings().weight, input_table)
            torch.testing.assert_close(
                candidate.get_output_embeddings().weight, head.weight, atol=2e-6, rtol=1e-5
            )

            test_ids = torch.tensor([[1, 6, 5, 9, 3]])
            with torch.no_grad():
                torch.testing.assert_close(
                    candidate(test_ids).logits, target(test_ids).logits, atol=5e-6, rtol=1e-5
                )

    def test_rejects_overlapping_or_inconsistent_evidence_without_export(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(113)
        rng = np.random.default_rng(39)
        with tempfile.TemporaryDirectory() as tmp:
            base, train, validation, output = (
                Path(tmp) / name for name in ("base", "train", "validation", "export")
            )
            target = LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=21,
                    max_position_embeddings=16,
                    hidden_size=8,
                    intermediate_size=12,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                )
            ).eval()
            target.save_pretrained(base, safe_serialization=True)
            _save_scores(torch, target, train, rng.integers(0, 21, size=(40, 5)))
            _save_scores(torch, target, validation, rng.integers(0, 21, size=(12, 5)))
            with self.assertRaisesRegex(ValueError, "share a token context"):
                recover_endpoint_head(base, train, train, output, anchor_id=0)
            scores = np.load(validation / "scores.npy")
            scores[:, 3] += 1
            np.save(validation / "scores.npy", scores, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "held-out contrast RMSE"):
                recover_endpoint_head(base, train, validation, output, anchor_id=0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
