"""Check compact observation binding, restart, and source-free native replay."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rephix.weight_audit import compare_checkpoints

HAS_HF = all(importlib.util.find_spec(n) for n in ("torch", "transformers", "safetensors"))
HAS_DIFFUSERS = HAS_HF and importlib.util.find_spec("diffusers") is not None


@unittest.skipUnless(HAS_HF, "needs model dependencies")
class CompactRuntimeTests(unittest.TestCase):
    def _exercise(self, family):
        from rephix import diffusers_recovery, layer_recovery

        if family == "transformers":
            from transformers import ConvNextConfig, ConvNextForImageClassification

            model = ConvNextForImageClassification(
                ConvNextConfig(
                    num_channels=3,
                    num_stages=2,
                    hidden_sizes=[4, 8],
                    depths=[1, 1],
                    patch_size=2,
                    num_labels=3,
                )
            )
            runtime = layer_recovery
            recover, replay = runtime.export_recovered_model, runtime.replay_layer_evidence
            extra = {"include_tokenizer": False}
        else:
            from diffusers import AutoencoderTiny

            model = AutoencoderTiny(
                encoder_block_out_channels=(8,),
                decoder_block_out_channels=(8,),
                num_encoder_blocks=(1,),
                num_decoder_blocks=(1,),
                latent_channels=2,
            )
            runtime = diffusers_recovery
            recover, replay = runtime.recover_diffusion_model, runtime.replay_diffusion_evidence
            extra = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, evidence, output = (root / n for n in ("source", "evidence", "output"))
            model.save_pretrained(source, safe_serialization=True)
            original_recover = runtime.recover_from_layer_calls

            def interrupted(candidate, oracle, **options):
                calls = 0

                def call(path, inputs):
                    nonlocal calls
                    if calls == 5:
                        raise InterruptedError("controlled compact interruption")
                    calls += 1
                    return oracle(path, inputs)

                return original_recover(candidate, call, **options)

            options = dict(
                dtype="float64",
                batch_size=7,
                compact_convolutions=True,
                restore_source_dtypes=True,
                evidence_dir=evidence,
                **extra,
            )
            with patch.object(runtime, "recover_from_layer_calls", interrupted):
                with self.assertRaisesRegex(InterruptedError, "controlled"):
                    recover(source, output, **options)
            self.assertFalse(output.exists())
            # An incompatible resume must fail before consuming or changing evidence.
            database = evidence / "observations.sqlite3"
            original_bytes = database.read_bytes()
            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                recover(source, root / "wrong", **{**options, "compact_convolutions": False})
            self.assertEqual(original_bytes, database.read_bytes())
            report = recover(source, output, **options)
            self.assertEqual(report["evidence_observations_cached"], 5)
            self.assertEqual(compare_checkpoints(source, output)["fraction_exactly_equal"], 1)
            shutil.rmtree(source)
            replayed = root / "replay"
            replay_report = replay(evidence, replayed)
            self.assertEqual(replay_report["evidence_observations_collected"], 0)
            self.assertEqual(compare_checkpoints(output, replayed)["fraction_exactly_equal"], 1)
            self.assertGreater(report["evidence_stored_payload_bytes"], 0)
            self.assertLessEqual(
                report["evidence_stored_payload_bytes"], report["evidence_decoded_payload_bytes"]
            )

    def test_invalid_compact_flags_fail_before_reading_models_or_writing_output(self):
        from rephix.diffusers_recovery import recover_diffusion_model
        from rephix.layer_recovery import export_recovered_model

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for recover in (export_recovered_model, recover_diffusion_model):
                for value in ("false", 1, None):
                    with self.subTest(recover=recover.__name__, value=value):
                        with self.assertRaisesRegex(ValueError, "must be a boolean"):
                            recover(root / "missing", root / "output", compact_convolutions=value)
                        self.assertEqual(list(root.iterdir()), [])

    def test_transformers_compact_capture_resume_and_replay(self):
        self._exercise("transformers")

    @unittest.skipUnless(HAS_DIFFUSERS, "needs Diffusers")
    def test_diffusers_compact_capture_resume_and_replay(self):
        self._exercise("diffusers")


if __name__ == "__main__":
    unittest.main()
