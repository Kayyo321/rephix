"""Verify sparse capture and recovery of parameters absent from passive traces."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rephix.io import file_sha256, write_json
from rephix.trace_calibration import collect_trace_calibration
from rephix.trace_recovery import capture_layer_traces, recover_layer_traces

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class SparseTraceRecoveryTests(unittest.TestCase):
    def test_rebuilds_an_unrouted_expert_from_fit_and_separate_validation_probes(self):
        import torch
        from transformers import AutoModelForCausalLM, MixtralConfig, MixtralForCausalLM

        torch.manual_seed(118)
        rng = np.random.default_rng(119)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, config, traces, chosen, output = (
                root / name for name in ("source", "config", "traces", "chosen", "recovered")
            )
            target = (
                MixtralForCausalLM(
                    MixtralConfig(
                        vocab_size=32,
                        hidden_size=12,
                        intermediate_size=16,
                        num_hidden_layers=1,
                        num_attention_heads=3,
                        num_key_value_heads=1,
                        num_local_experts=5,
                        num_experts_per_tok=2,
                        max_position_embeddings=32,
                    )
                )
                .double()
                .eval()
            )
            sparse = target.model.layers[0].block_sparse_moe
            with torch.no_grad():
                direction = torch.randn(12, dtype=torch.float64)
                gates = torch.tensor([2, 1, -2, -1, 0], dtype=torch.float64)
                sparse.gate.weight.copy_(gates[:, None] * direction)
            target.save_pretrained(source, safe_serialization=True)
            config.mkdir()
            shutil.copyfile(source / "config.json", config / "config.json")
            train, validation = root / "train.npy", root / "validation.npy"
            train_ids = np.vstack(
                (rng.permutation(32).reshape(-1, 4), rng.integers(32, size=(64, 4)))
            )
            val_ids = rng.integers(32, size=(16, 4))
            np.save(train, train_ids, allow_pickle=False)
            np.save(validation, val_ids, allow_pickle=False)
            passive = capture_layer_traces(source, train, validation, traces)
            dormant = [
                index
                for index, operator in enumerate(passive["layout"])
                if ".experts.4." in operator["path"]
            ]
            self.assertEqual(len(dormant), 3)
            for index in dormant:
                self.assertEqual(passive["train_counts"][index], 0)
                self.assertEqual(passive["validation_counts"][index], 0)

            probe = torch.from_numpy(val_ids)
            with torch.inference_mode():
                initial = target(probe).logits
                sparse.experts[4].w1.weight.add_(0.25)
                torch.testing.assert_close(target(probe).logits, initial, atol=0, rtol=0)

            def observe(path, inputs):
                with torch.inference_mode():
                    return target.get_submodule(path)(torch.from_numpy(inputs)).numpy()

            def no_oracle(*_):
                self.fail("the complete budget must be checked before calling any layer")

            with self.assertRaisesRegex(ValueError, "validation probe budget"):
                collect_trace_calibration(traces, chosen, no_oracle)
            report = collect_trace_calibration(traces, chosen, observe, max_validation_probes=3)
            self.assertEqual(report["affine_rows"], 40)
            self.assertEqual(report["validation_probe_rows"], 3)
            source.rename(root / "offline-source")
            recover_layer_traces(
                config, traces, output, calibration=chosen, include_tokenizer=False
            )
            candidate = AutoModelForCausalLM.from_pretrained(
                output, local_files_only=True, torch_dtype=torch.float64
            ).eval()
            for fitted, reference in zip(candidate.parameters(), target.parameters(), strict=True):
                torch.testing.assert_close(fitted, reference, atol=1e-7, rtol=1e-6)
            with torch.inference_mode():
                torch.testing.assert_close(candidate(probe).logits, initial)

            # Change only an independent held-out response and update its digest.
            # A fitter using those rows for training would no longer have this check.
            path = chosen / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            entry = manifest["operators"][0]
            array = chosen / f"{entry['operator_index']:04d}" / "validation" / "y.npy"
            np.save(array, np.load(array, allow_pickle=False) + 1, allow_pickle=False)
            entry["validation_sha256"]["y.npy"] = file_sha256(array)
            write_json(path, manifest)
            invalid = root / "invalid"
            with self.assertRaisesRegex(ValueError, "held-out RMSE"):
                recover_layer_traces(
                    config, traces, invalid, calibration=chosen, include_tokenizer=False
                )
            self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()
