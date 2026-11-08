"""Verify native waveform encoder observations and complete CTC checkpoint replay."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from itertools import product
from pathlib import Path

from rephix.audio_encoder_recovery import (
    initialize_missing_audio_masks,
    observe_audio_mask_embedding,
    observe_wavlm_position_gate,
    recover_audio_mask_embedding,
    recover_wavlm_position_gate,
    validate_audio_mask_embedding,
    validate_wavlm_position_gate,
)
from rephix.layer_recovery import export_recovered_model, replay_layer_evidence
from rephix.weight_audit import compare_checkpoints

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class AudioEncoderRecoveryTests(unittest.TestCase):
    def _model(self, family):
        import torch
        import transformers

        names = {
            "wav2vec2": ("Wav2Vec2Config", "Wav2Vec2ForCTC"),
            "hubert": ("HubertConfig", "HubertForCTC"),
            "wavlm": ("WavLMConfig", "WavLMForCTC"),
        }
        config_name, model_name = names[family]
        config = getattr(transformers, config_name)(
            vocab_size=7,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
            conv_dim=(4, 8),
            conv_stride=(2, 2),
            conv_kernel=(4, 2),
            conv_bias=True,
            num_conv_pos_embeddings=4,
            num_conv_pos_embedding_groups=2,
            num_buckets=8,
            max_bucket_distance=16,
            mask_time_prob=0.1,
            apply_spec_augment=False,
            pad_token_id=0,
        )
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(47)
            model = getattr(transformers, model_name)(config).double().eval()
            for name, parameter in model.named_parameters():
                parameter.copy_(torch.randint(-16, 17, parameter.shape) / 64)
                if name.endswith("original0") or name.endswith("weight_g"):
                    values = torch.arange(parameter.numel()).reshape(parameter.shape) / 8 - 0.25
                    parameter.copy_(values)
                    parameter.flatten()[0] = 0
        return model

    def test_chosen_mask_and_position_gate_restore_native_runtime(self):
        import torch

        for family in ("wav2vec2", "hubert", "wavlm"):
            with self.subTest(family=family), torch.no_grad():
                model = self._model(family)
                encoder = getattr(model, family)
                expected = encoder.masked_spec_embed.detach().clone()
                inputs = expected.new_zeros((1, 1, len(expected)))
                self.assertFalse(encoder.config.apply_spec_augment)
                torch.testing.assert_close(
                    encoder._mask_hidden_states(inputs), inputs, rtol=0, atol=0
                )
                pending = {"masked_spec_embed": torch.empty_like(expected)}

                def read(values, shape, operation):
                    self.assertEqual(operation, "masked_spec")
                    result = observe_audio_mask_embedding(torch, encoder, values)
                    self.assertEqual(result.shape, shape)
                    return result

                validate_audio_mask_embedding(encoder, pending)
                recover_audio_mask_embedding(encoder, pending, read)
                torch.testing.assert_close(pending["masked_spec_embed"], expected, rtol=0, atol=0)
                self.assertFalse(encoder.config.apply_spec_augment)
                if family == "wavlm":
                    layer = encoder.encoder.layers[0].attention
                    gate = layer.gru_rel_pos_const.detach().clone()
                    target = {"gru_rel_pos_const": torch.empty_like(gate)}
                    hidden = torch.randn(1, 3, model.config.hidden_size, dtype=torch.float64)
                    original = layer(hidden)[0]

                    def read_gate(values, shape, operation):
                        self.assertEqual(operation, "position_gate")
                        result = observe_wavlm_position_gate(torch, layer, values)
                        self.assertEqual(result.shape, shape)
                        return result

                    validate_wavlm_position_gate(layer, target)
                    recover_wavlm_position_gate(layer, target, read_gate)
                    torch.testing.assert_close(target["gru_rel_pos_const"], gate, rtol=0, atol=0)
                    torch.testing.assert_close(layer(hidden)[0], original, rtol=0, atol=0)

    def test_missing_mask_initialization_preserves_unrelated_loading_errors(self):
        import torch

        model = self._model("wav2vec2")
        mask_name = "wav2vec2.masked_spec_embed"
        missing = [mask_name, "lm_head.weight", "unknown.masked_spec_embed"]
        initialized, remaining = initialize_missing_audio_masks(torch, model, missing)
        self.assertEqual(initialized, [mask_name])
        self.assertEqual(remaining, missing[1:])
        self.assertEqual(int(torch.count_nonzero(model.wav2vec2.masked_spec_embed)), 0)

    def test_base_and_ctc_checkpoints_survive_source_free_replay(self):
        import torch
        import transformers
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        for family, ctc in product(("wav2vec2", "hubert", "wavlm"), (False, True)):
            with self.subTest(family=family, ctc=ctc), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output, evidence, replay = (
                    root / name for name in ("source", "output", "evidence", "replay")
                )
                model = self._model(family)
                if not ctc:
                    model = getattr(model, family)
                unstored_mask = family == "wav2vec2" and ctc
                if unstored_mask:
                    model.config.apply_spec_augment = True
                model.save_pretrained(source, safe_serialization=True)
                if ctc:
                    archive = source / "model.safetensors"
                    with safe_open(archive, framework="pt") as saved:
                        metadata = saved.metadata()
                    tensors = {
                        name.replace(".parametrizations.weight.original0", ".weight_g").replace(
                            ".parametrizations.weight.original1", ".weight_v"
                        ): value
                        for name, value in load_file(archive).items()
                    }
                    self.assertTrue(any(name.endswith(".weight_g") for name in tensors))
                    if unstored_mask:
                        tensors.pop("wav2vec2.masked_spec_embed")
                    save_file(tensors, archive, metadata=metadata)
                processor = transformers.Wav2Vec2FeatureExtractor(
                    sampling_rate=16000, return_attention_mask=True
                )
                if ctc:
                    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "|": 3, "A": 4, "B": 5, "C": 6}
                    (source / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
                    processor = transformers.Wav2Vec2Processor(
                        processor,
                        transformers.Wav2Vec2CTCTokenizer(str(source / "vocab.json")),
                    )
                processor.save_pretrained(source)
                capture = export_recovered_model(
                    source, output, dtype="float64", batch_size=13, evidence_dir=evidence
                )
                audit = compare_checkpoints(source, output)
                self.assertEqual(audit["coverage"], 1)
                self.assertEqual(
                    audit["candidate_tensors"], audit["reference_tensors"] + int(unstored_mask)
                )
                self.assertEqual(audit["fraction_within_tolerance"], 1)
                self.assertTrue(capture["audio_feature_extractor_exported"])
                if unstored_mask:
                    self.assertEqual(
                        capture["source_initialized_audio_masks"], ["wav2vec2.masked_spec_embed"]
                    )
                shutil.rmtree(source)
                offline = replay_layer_evidence(evidence, replay)
                self.assertEqual(offline["evidence_observations_collected"], 0)
                self.assertEqual(offline["evidence_observations_cached"], capture["layer_calls"])
                self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)
                wrapper = transformers.AutoModelForCTC if ctc else transformers.AutoModel
                recovered = wrapper.from_pretrained(
                    replay, local_files_only=True, torch_dtype=torch.float64
                ).eval()
                if unstored_mask:
                    self.assertEqual(
                        int(torch.count_nonzero(recovered.wav2vec2.masked_spec_embed)), 0
                    )
                processor_wrapper = (
                    transformers.AutoProcessor if ctc else transformers.AutoFeatureExtractor
                )
                restored_processor = processor_wrapper.from_pretrained(
                    replay, local_files_only=True
                )
                waveforms = [torch.sin(torch.arange(128).float() / 7).numpy()]
                prepared = processor(waveforms, sampling_rate=16000, return_tensors="pt")
                restored = restored_processor(waveforms, sampling_rate=16000, return_tensors="pt")
                for name in prepared:
                    torch.testing.assert_close(restored[name], prepared[name], rtol=0, atol=0)
                inputs = {
                    name: value.double() if value.is_floating_point() else value
                    for name, value in prepared.items()
                }
                with torch.no_grad():
                    field = "logits" if ctc else "last_hidden_state"
                    expected = getattr(model(**inputs), field)
                    changed = model(**{**inputs, "input_values": inputs["input_values"].flip(-1)})
                    changed = getattr(changed, field)
                    self.assertFalse(torch.equal(changed, expected))
                    actual = getattr(recovered(**inputs), field)
                    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
                    if ctc:
                        self.assertEqual(
                            processor.batch_decode(expected.argmax(-1)),
                            restored_processor.batch_decode(actual.argmax(-1)),
                        )
