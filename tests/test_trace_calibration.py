"""Test the calibrated trace contract and a full decoder export."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rephix.trace_calibration import collect_trace_calibration
from rephix.trace_recovery import capture_layer_traces, recover_layer_traces

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class TraceCalibrationTests(unittest.TestCase):
    def test_gpt2_affine_and_position_calibration_with_no_source_at_export(self):
        import torch
        from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

        torch.manual_seed(78)
        rng = np.random.default_rng(82)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, config, traces, chosen, output = (
                root / name for name in ("source", "config", "traces", "chosen", "recovered")
            )
            target = (
                GPT2LMHeadModel(
                    GPT2Config(vocab_size=20, n_positions=16, n_embd=8, n_layer=1, n_head=2)
                )
                .double()
                .eval()
            )
            target.save_pretrained(source, safe_serialization=True)
            config.mkdir()
            shutil.copyfile(source / "config.json", config / "config.json")
            train, validation = root / "train.npy", root / "validation.npy"
            rows = np.vstack((rng.permutation(20).reshape(-1, 4), rng.integers(20, size=(42, 4))))
            np.save(train, rows, allow_pickle=False)
            np.save(validation, rng.integers(20, size=(8, 4)), allow_pickle=False)
            capture_layer_traces(source, train, validation, traces)
            with self.assertRaisesRegex(ValueError, "embedding token coverage 4/16"):
                recover_layer_traces(config, traces, output, include_tokenizer=False)
            self.assertFalse(output.exists())

            def observe(path, inputs):
                with torch.inference_mode():
                    return target.get_submodule(path)(torch.from_numpy(inputs)).numpy()

            def no_oracle(*_):
                self.fail("budget must be checked before invoking the oracle")

            with self.assertRaisesRegex(ValueError, "remaining embedding budget"):
                collect_trace_calibration(traces, chosen, no_oracle, max_embedding_rows=11)
            self.assertFalse(chosen.exists())
            with self.assertRaisesRegex(ValueError, "budget"):
                collect_trace_calibration(
                    traces, chosen, no_oracle, max_probes=0, max_embedding_rows=12
                )
            self.assertFalse(chosen.exists())
            plan = collect_trace_calibration(traces, chosen, observe, max_embedding_rows=12)
            self.assertGreater(plan["affine_rows"], 0)
            self.assertEqual(plan["embedding_rows"], 12)

            source.rename(root / "offline-source")
            report = recover_layer_traces(
                config, traces, output, calibration=chosen, include_tokenizer=False
            )
            self.assertEqual(report["calibration"]["chosen_rows"], plan["chosen_rows"])
            candidate = AutoModelForCausalLM.from_pretrained(
                output, local_files_only=True, torch_dtype=torch.float64
            ).eval()
            for (name, reference), (other, fitted) in zip(
                target.named_parameters(), candidate.named_parameters(), strict=True
            ):
                self.assertEqual(name, other)
                torch.testing.assert_close(fitted, reference, atol=1e-8, rtol=1e-6)
            with torch.inference_mode():
                ids = torch.tensor([[6, 7, 1, 11, 9, 10]])
                torch.testing.assert_close(candidate(ids).logits, target(ids).logits)

            first = plan["operators"][0]["operator_index"]
            evidence = chosen / f"{first:04d}" / "y.npy"
            values = np.load(evidence, allow_pickle=False)
            np.save(evidence, values + 1, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "calibration evidence hash mismatch"):
                recover_layer_traces(
                    config,
                    traces,
                    root / "tampered-export",
                    calibration=chosen,
                    include_tokenizer=False,
                )
            self.assertFalse((root / "tampered-export").exists())


if __name__ == "__main__":
    unittest.main()
