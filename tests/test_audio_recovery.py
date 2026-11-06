"""Exercise a standalone Whisper export through weights, audio, and decoding."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from rephix.cli import main

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class AudioRecoveryTests(unittest.TestCase):
    def _cli_json(self, args):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(args), 0)
        return json.loads(stdout.getvalue())

    def _recover(self, source, candidate, *flags):
        return self._cli_json(
            [
                "recover-layers",
                "--model",
                str(source),
                "--output",
                str(candidate),
                "--dtype",
                "float64",
                "--batch-size",
                "5",
                *flags,
            ]
        )

    def _whisper_fixture(self, source, *, with_processor):
        import torch
        from transformers import (
            GenerationConfig,
            WhisperConfig,
            WhisperFeatureExtractor,
            WhisperForConditionalGeneration,
            WhisperProcessor,
            WhisperTokenizerFast,
        )
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

        vocab = {character: index for index, character in enumerate(bytes_to_unicode().values())}
        for token in ("<|endoftext|>", "<|startoftranscript|>", "<|notimestamps|>"):
            vocab[token] = len(vocab)
        pad = vocab["<|endoftext|>"]
        start = vocab["<|startoftranscript|>"]
        torch.manual_seed(42)
        target = (
            WhisperForConditionalGeneration(
                WhisperConfig(
                    vocab_size=len(vocab),
                    d_model=12,
                    encoder_layers=1,
                    decoder_layers=1,
                    encoder_attention_heads=3,
                    decoder_attention_heads=3,
                    encoder_ffn_dim=18,
                    decoder_ffn_dim=18,
                    num_mel_bins=8,
                    max_source_positions=20,
                    max_target_positions=12,
                    pad_token_id=pad,
                    bos_token_id=pad,
                    eos_token_id=pad,
                    decoder_start_token_id=start,
                    suppress_tokens=None,
                    begin_suppress_tokens=None,
                )
            )
            .double()
            .eval()
        )
        with torch.no_grad():
            encoder, decoder = target.get_encoder(), target.get_decoder()
            for layer in (encoder.conv1, encoder.conv2):
                layer.bias.uniform_(-0.2, 0.2)
            # Even the frozen encoder table must be recovered from its observed
            # values. Its usual sinusoidal initialization would hide a missed probe.
            encoder.embed_positions.weight.uniform_(-0.4, 0.4)
            decoder.embed_positions.weight.uniform_(-0.4, 0.4)
        target.generation_config = GenerationConfig(
            bos_token_id=pad,
            pad_token_id=pad,
            eos_token_id=pad,
            decoder_start_token_id=start,
            no_timestamps_token_id=vocab["<|notimestamps|>"],
            is_multilingual=False,
            min_new_tokens=3,
            max_new_tokens=3,
        )
        target.save_pretrained(source, safe_serialization=True)
        if with_processor:
            vocab_file, merges_file = source / "vocab.json", source / "merges.txt"
            vocab_file.write_text(json.dumps(vocab), encoding="utf-8")
            merges_file.write_text("#version: 0.2\n", encoding="utf-8")
            tokenizer = WhisperTokenizerFast(
                vocab_file=str(vocab_file), merges_file=str(merges_file)
            )
            extractor = WhisperFeatureExtractor(
                feature_size=8, chunk_length=1, hop_length=400, return_attention_mask=True
            )
            WhisperProcessor(extractor, tokenizer).save_pretrained(source)
            (source / "processor_config.json").write_text(
                '{\n  "processor_class": "WhisperProcessor"\n}\n', encoding="utf-8"
            )
        return target

    def test_whisper_recovery_preserves_weights_processor_and_cached_generation(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "source", Path(temporary) / "candidate"
            target = self._whisper_fixture(source, with_processor=True)
            target_processor = AutoProcessor.from_pretrained(source, local_files_only=True)
            report = self._recover(source, output)
            elements = sum(parameter.numel() for parameter in target.parameters())
            self.assertEqual(report["model_type"], "whisper")
            self.assertEqual(report["unique_parameter_elements_recovered"], elements)
            self.assertEqual(report["individually_identified_parameter_elements"], elements)
            self.assertTrue(report["audio_feature_extractor_exported"])
            self.assertTrue(report["tokenizer_exported"])
            for filename in ("preprocessor_config.json", "processor_config.json"):
                self.assertEqual((output / filename).read_bytes(), (source / filename).read_bytes())

            audit = self._cli_json(["audit-weights", str(source), str(output)])
            self.assertEqual(audit["coverage"], 1.0)
            self.assertEqual(audit["fraction_within_tolerance"], 1.0)
            self.assertFalse(audit["extra_tensors"])
            self.assertFalse(audit["dtype_mismatches"])

            source.rename(Path(temporary) / "offline-source")
            recovered = AutoModelForSpeechSeq2Seq.from_pretrained(
                output, local_files_only=True, torch_dtype=torch.float64
            ).eval()
            recovered_processor = AutoProcessor.from_pretrained(output, local_files_only=True)
            self.assertIs(
                recovered.get_decoder().embed_tokens.weight,
                recovered.get_output_embeddings().weight,
            )
            self.assertFalse(recovered.get_encoder().embed_positions.weight.requires_grad)
            self.assertEqual(recovered.get_encoder().conv2.stride, (2,))
            self.assertEqual(
                recovered.generation_config.no_timestamps_token_id,
                target.generation_config.no_timestamps_token_id,
            )

            rate = target_processor.feature_extractor.sampling_rate
            time = torch.arange(5500, dtype=torch.float32) / rate
            audio = [
                (torch.sin(2 * torch.pi * 273 * time) / 2).numpy(),
                (torch.cos(2 * torch.pi * 604 * time[:3200]) / 4).numpy(),
            ]
            a, b = [
                processor(audio=audio, sampling_rate=rate, return_tensors="pt")
                for processor in (target_processor, recovered_processor)
            ]
            self.assertTrue(torch.equal(a.input_features, b.input_features))
            self.assertTrue(torch.equal(a.attention_mask, b.attention_mask))
            text = "Six réglages."
            self.assertEqual(
                target_processor(text=text).input_ids, recovered_processor(text=text).input_ids
            )

            start = target.config.decoder_start_token_id
            decoder_ids = torch.tensor([[start, 4, 8, 13], [start, 9, 11, 8]])
            inputs = dict(input_features=a.input_features.double(), attention_mask=a.attention_mask)
            with torch.inference_mode():
                torch.testing.assert_close(
                    target(**inputs, decoder_input_ids=decoder_ids).logits,
                    recovered(**inputs, decoder_input_ids=decoder_ids).logits,
                    atol=1e-10,
                    rtol=1e-8,
                )
                generation = dict(use_cache=True, do_sample=False, num_beams=1)
                tokens = target.generate(**inputs, **generation)
                self.assertEqual(len(tokens), 2)
                self.assertGreaterEqual(tokens.shape[1], 3)
                self.assertTrue(torch.equal(tokens, recovered.generate(**inputs, **generation)))

    def test_whisper_weights_only_does_not_need_audio_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "source", Path(temporary) / "candidate"
            self._whisper_fixture(source, with_processor=False)
            report = self._recover(source, output, "--without-preprocessor")
            self.assertFalse(report["audio_feature_extractor_exported"])
            self.assertFalse(report["tokenizer_exported"])
            self.assertTrue((output / "model.safetensors").is_file())
            self.assertFalse((output / "preprocessor_config.json").exists())
            self.assertFalse((output / "tokenizer.json").exists())


if __name__ == "__main__":
    unittest.main()
