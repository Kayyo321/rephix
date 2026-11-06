"""Exercise speech checkpoint export and source-free replay with native audio inputs."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
from rephix.weight_audit import compare_checkpoints

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class MoonshineRecoveryTests(unittest.TestCase):
    def test_native_audio_processor_and_logits_survive_source_free_replay(self):
        import torch
        import transformers
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel

        if not hasattr(transformers, "MoonshineForConditionalGeneration"):
            self.skipTest("installed Transformers has no native Moonshine model")
        config = transformers.MoonshineConfig(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=24,
            encoder_num_hidden_layers=1,
            decoder_num_hidden_layers=1,
            encoder_num_attention_heads=2,
            decoder_num_attention_heads=2,
            partial_rotary_factor=1.0,
            pad_token_id=0,
        )
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(54)
            model = transformers.MoonshineForConditionalGeneration(config).double().eval()
            for parameter in model.parameters():
                parameter.copy_(torch.randint(-16, 17, parameter.shape) / 64)
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<s>": 1, "</s>": 2}, "<pad>")),
            unk_token="<pad>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
        )
        processor = transformers.Wav2Vec2Processor(
            feature_extractor=transformers.Wav2Vec2FeatureExtractor(sampling_rate=16000),
            tokenizer=tokenizer,
        )
        audio = torch.arange(2048).float().sin().numpy()
        prepared = processor(audio=audio, sampling_rate=16000, return_tensors="pt")
        with tempfile.TemporaryDirectory() as directory:
            source, candidate, evidence, replay = (
                Path(directory) / name for name in ("source", "candidate", "evidence", "replay")
            )
            model.save_pretrained(source)
            processor.save_pretrained(source)
            capture = export_recovered_model(
                source, candidate, dtype="float64", batch_size=23, evidence_dir=evidence
            )
            audit = compare_checkpoints(source, candidate)
            self.assertEqual(audit["fraction_within_tolerance"], 1)
            self.assertTrue(capture["audio_feature_extractor_exported"])
            shutil.rmtree(source)
            offline = replay_layer_evidence(evidence, replay)
            self.assertEqual(offline["evidence_observations_cached"], capture["layer_calls"])
            self.assertEqual(offline["evidence_observations_collected"], 0)
            self.assertEqual(compare_checkpoints(candidate, replay)["fraction_exactly_equal"], 1)

            restored_processor = transformers.AutoProcessor.from_pretrained(
                replay, local_files_only=True
            )
            restored_audio = restored_processor(
                audio=audio, sampling_rate=16000, return_tensors="pt"
            )
            self.assertEqual(set(prepared), set(restored_audio))
            for name, values in prepared.items():
                torch.testing.assert_close(restored_audio[name], values, atol=0, rtol=0)
            recovered = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
                replay, local_files_only=True, torch_dtype=torch.float64
            ).eval()
            inputs = {
                "input_values": restored_audio.input_values.double(),
                "decoder_input_ids": torch.tensor([[1, 3, 5]]),
                "use_cache": False,
            }
            with torch.no_grad():
                reference = model(**inputs).logits
                changed = model(
                    **{**inputs, "input_values": inputs["input_values"].flip(-1)}
                ).logits
                self.assertFalse(torch.equal(reference, changed))
                torch.testing.assert_close(
                    recovered(**inputs).logits, reference, atol=1e-12, rtol=0
                )
