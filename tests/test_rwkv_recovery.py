"""Verify native RWKV chosen observations and exported recurrent inference."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from rephix.rwkv_recovery import (
    observe_rwkv_operator,
    recover_rwkv_operator,
    validate_rwkv_operator,
)

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class RwkvRecoveryTests(unittest.TestCase):
    def _model(self):
        import torch
        from transformers import RwkvConfig, RwkvForCausalLM

        config = RwkvConfig(
            vocab_size=37,
            hidden_size=16,
            attention_hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            context_length=32,
            rescale_every=1,
        )
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(91)
            model = RwkvForCausalLM(config).double().eval()
            for name, parameter in model.named_parameters():
                if name.endswith("time_decay"):
                    parameter.copy_(
                        torch.linspace(-4, 1, parameter.numel()).reshape(parameter.shape)
                    )
                else:
                    parameter.copy_(torch.randint(-32, 33, parameter.shape).double() / 128)
        return model

    def test_native_time_observations_preserve_source(self):
        import torch

        source, target = self._model(), self._model()
        with torch.no_grad():
            for child in ("attention", "feed_forward"):
                original = getattr(source.rwkv.blocks[0], child)
                candidate = getattr(target.rwkv.blocks[0], child)
                pending = dict(candidate.named_parameters(recurse=False))
                expected = {name: value.clone() for name, value in original.named_parameters()}
                for parameter in pending.values():
                    parameter.zero_()

                def read(inputs, shape, operation):
                    response = observe_rwkv_operator(torch, original, inputs, operation)
                    self.assertEqual(response.dtype, torch.float64)
                    self.assertEqual(response.shape, shape)
                    return response

                validate_rwkv_operator(candidate, pending)
                recover_rwkv_operator(torch, candidate, pending, read)
                for name, parameter in pending.items():
                    torch.testing.assert_close(parameter, expected[name], rtol=0, atol=5e-16)
                for name, parameter in original.named_parameters():
                    torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)

    def test_checkpoint_replay_preserves_recurrent_generation(self):
        import torch
        from transformers import AutoModelForCausalLM

        from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
        from rephix.weight_audit import compare_checkpoints

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, evidence, replay = (
                root / name for name in ("source", "output", "evidence", "replay")
            )
            model = self._model()
            model.save_pretrained(source)
            report = export_recovered_model(
                source,
                output,
                dtype="float64",
                batch_size=11,
                include_tokenizer=False,
                evidence_dir=evidence,
            )
            self.assertEqual(compare_checkpoints(source, output)["fraction_within_tolerance"], 1)
            shutil.rmtree(source)
            offline = replay_layer_evidence(evidence, replay)
            self.assertEqual(offline["evidence_observations_collected"], 0)
            self.assertEqual(offline["evidence_observations_cached"], report["layer_calls"])
            self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)
            candidate = AutoModelForCausalLM.from_pretrained(
                replay, torch_dtype=torch.float64, local_files_only=True
            ).eval()
            tokens = torch.tensor([[1, 4, 8, 3, 7]])
            with torch.inference_mode():
                expected = model(tokens, use_cache=True)
                actual = candidate(tokens, use_cache=True)
                torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-12, atol=1e-12)
                step = torch.tensor([[5]])
                continued = candidate(step, state=actual.state, use_cache=True)
                reference = model(step, state=expected.state, use_cache=True)
                torch.testing.assert_close(
                    continued.logits, reference.logits, rtol=1e-12, atol=1e-12
                )
                a = candidate.generate(tokens, max_new_tokens=5, do_sample=False, use_cache=True)
                b = model.generate(tokens, max_new_tokens=5, do_sample=False, use_cache=True)
                self.assertTrue(torch.equal(a, b))

    def test_restored_source_precision_recovers_all_original_coordinates(self):
        from safetensors.torch import load_file

        from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
        from rephix.weight_audit import compare_checkpoints

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, evidence, replay = (
                root / name for name in ("source", "output", "evidence", "replay")
            )
            self._model().float().save_pretrained(source)
            export_recovered_model(
                source,
                output,
                dtype="float64",
                include_tokenizer=False,
                evidence_dir=evidence,
                restore_source_dtypes=True,
            )
            self.assertEqual(compare_checkpoints(source, output)["fraction_exactly_equal"], 1)
            shutil.rmtree(source)
            replay_layer_evidence(evidence, replay)
            original = load_file(output / "model.safetensors")
            restored = load_file(replay / "model.safetensors")
            self.assertEqual(
                {name: value.dtype for name, value in original.items()},
                {name: value.dtype for name, value in restored.items()},
            )
            self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)

    def test_unresolvable_decay_exponent_is_rejected(self):
        import torch

        for exponent in (-1000, 1000):
            with self.subTest(exponent=exponent), torch.no_grad():
                source = self._model().rwkv.blocks[0].attention
                target = self._model().rwkv.blocks[0].attention
                source.time_decay.fill_(exponent)

                def read(inputs, shape, operation):
                    return observe_rwkv_operator(torch, source, inputs, operation)

                with self.assertRaisesRegex(ValueError, "FP64 exponential range"):
                    recover_rwkv_operator(
                        torch, target, dict(target.named_parameters(recurse=False)), read
                    )


if __name__ == "__main__":
    unittest.main()
