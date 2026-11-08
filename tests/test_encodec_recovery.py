"""Verify explicit Encodec inactive-state access and complete codec replay."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from rephix.encodec_recovery import (
    ENCODEC_BUFFER_OPERATIONS,
    ENCODEC_INACTIVE_BUFFERS,
    observe_encodec_buffer,
    recover_encodec_buffer,
)

HAS_HF = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors"))


@unittest.skipUnless(HAS_HF, "needs native Transformers dependencies")
class EncodecRecoveryTests(unittest.TestCase):
    def _model(self):
        import torch
        from transformers import EncodecConfig, EncodecModel

        config = EncodecConfig(
            audio_channels=1,
            sampling_rate=240,
            hidden_size=8,
            num_filters=4,
            num_residual_layers=1,
            upsampling_ratios=[2, 2],
            num_lstm_layers=2,
            codebook_size=1024,
            codebook_dim=8,
            target_bandwidths=[0.6, 1.2],
            use_causal_conv=True,
        )
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(57)
            model = EncodecModel(config).double().eval()
            for parameter in model.parameters():
                parameter.copy_(torch.randint(-8, 9, parameter.shape).double() / 64)
                if parameter.ndim == 3:
                    parameter.flatten()[0] = 0.125
            for module in model.modules():
                if type(module).__name__ == "EncodecEuclideanCodebook":
                    module.embed.copy_(torch.randn_like(module.embed) / 8)
                    module.embed_avg.copy_(torch.randn_like(module.embed_avg) / 16)
                    module.cluster_size.copy_(torch.arange(module.codebook_size).double() / 4)
                    module.inited.fill_(2.25)
        return model

    def test_inactive_state_does_not_change_native_outputs_and_requires_separate_lookup(self):
        import torch

        with torch.no_grad():
            model = self._model()
            codebook = model.quantizer.layers[0].codebook
            indices = torch.arange(codebook.codebook_size)
            hidden = torch.randn(5, codebook.embed.shape[1], dtype=torch.float64)
            before_decode, before_encode = codebook.decode(indices), codebook.quantize(hidden)
            for name in ENCODEC_INACTIVE_BUFFERS:
                getattr(codebook, name).add_(3.5)
            torch.testing.assert_close(codebook.decode(indices), before_decode, rtol=0, atol=0)
            torch.testing.assert_close(codebook.quantize(hidden), before_encode, rtol=0, atol=0)
            for name, operation in ENCODEC_BUFFER_OPERATIONS.items():
                expected = getattr(codebook, name)
                target = torch.zeros_like(expected)

                def read(inputs, shape, operation):
                    result = observe_encodec_buffer(torch, codebook, inputs, operation)
                    self.assertEqual(result.shape, shape)
                    return result

                recover_encodec_buffer(
                    torch, codebook, {"bias": target}, read, batch_size=3, operation=operation
                )
                torch.testing.assert_close(target, expected, rtol=0, atol=0)

    def test_full_codec_requires_opt_in_and_replays_inactive_state(self):
        import torch
        from transformers import EncodecModel

        from rephix.layer_recovery import (
            export_recovered_model,
            recover_from_layer_calls,
            replay_layer_evidence,
        )
        from rephix.weight_audit import compare_checkpoints

        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            root = Path(directory)
            source, output, evidence, replay = (
                root / name for name in ("source", "output", "evidence", "replay")
            )
            original = self._model()
            original.save_pretrained(source)
            calls = []

            def unavailable(path, inputs):
                calls.append(path)
                raise AssertionError("inactive-state rejection reached the source oracle")

            with self.assertRaisesRegex(ValueError, "inactive"):
                recover_from_layer_calls(original, unavailable)
            self.assertEqual(calls, [])
            with self.assertRaisesRegex(ValueError, "inactive"):
                export_recovered_model(
                    source, root / "rejected", dtype="float64", include_tokenizer=False
                )
            self.assertFalse((root / "rejected").exists())
            report = export_recovered_model(
                source,
                output,
                dtype="float64",
                batch_size=257,
                include_tokenizer=False,
                evidence_dir=evidence,
                probe_inactive_state=True,
            )
            self.assertEqual(compare_checkpoints(source, output)["fraction_within_tolerance"], 1)
            shutil.rmtree(source)
            replayed = replay_layer_evidence(evidence, replay)
            self.assertEqual(replayed["evidence_observations_collected"], 0)
            self.assertEqual(replayed["evidence_observations_cached"], report["layer_calls"])
            self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)
            candidate = EncodecModel.from_pretrained(replay, torch_dtype=torch.float64).eval()
            waveform = torch.linspace(-0.5, 0.5, 128, dtype=torch.float64).reshape(1, 1, -1)
            reference = original(waveform, bandwidth=0.6)
            actual = candidate(waveform, bandwidth=0.6)
            self.assertTrue(torch.equal(reference.audio_codes, actual.audio_codes))
            torch.testing.assert_close(
                reference.audio_values, actual.audio_values, rtol=0, atol=1e-13
            )
            self.assertFalse(torch.equal(original.encoder(waveform), original.encoder(-waveform)))


if __name__ == "__main__":
    unittest.main()
