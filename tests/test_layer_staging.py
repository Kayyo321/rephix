"""Verify accelerator staging through native layer checkpoint recovery."""

import importlib.util
import json
import tempfile
import unittest
import weakref
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rephix.cli import main
from rephix.layer_recovery import export_recovered_model
from rephix.weight_audit import compare_checkpoints

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class LayerStagingTests(unittest.TestCase):
    def test_dependency_group_reuses_one_copy_per_registered_alias(self):
        import torch

        from rephix.layer_staging import LayerStager

        embedding = torch.nn.Embedding(7, 3)
        decoder = torch.nn.Linear(3, 7, bias=False)
        decoder.weight = embedding.weight
        decoder.register_buffer("positions", torch.arange(4), persistent=False)
        staging = LayerStager(torch, "cpu")
        staged_embedding, staged_decoder = staging.get(embedding, decoder)
        self.assertIs(staged_embedding.weight, staged_decoder.weight)
        self.assertNotEqual(staged_embedding.weight.data_ptr(), embedding.weight.data_ptr())
        torch.testing.assert_close(staged_embedding.weight, embedding.weight)
        torch.testing.assert_close(staged_decoder.positions, decoder.positions)
        self.assertEqual(
            staging.peak_state_bytes, embedding.weight.nbytes + decoder.positions.nbytes
        )
        self.assertIs(staging.get(embedding, decoder)[0], staged_embedding)
        self.assertEqual(staging.moves, 1)
        old_group = weakref.ref(staged_embedding)
        del staged_embedding, staged_decoder
        staging.clear()
        self.assertIsNone(old_group())

    def _cuda_fixture(self, source):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        torch.manual_seed(114)
        config = LlamaConfig(
            vocab_size=19,
            hidden_size=12,
            intermediate_size=18,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            max_position_embeddings=16,
            tie_word_embeddings=True,
            attention_bias=True,
            mlp_bias=True,
        )
        model = LlamaForCausalLM(config).double().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-0.25, 0.25)
        model.save_pretrained(source, safe_serialization=True)
        return model

    def _track_oracle(self, stack, *, fail=False):
        import torch
        from transformers import AutoModelForCausalLM

        targets, original_state, observed = [], {}, []
        load = AutoModelForCausalLM.from_pretrained

        def track(*args, **kwargs):
            target, loading = load(*args, **kwargs)
            targets.append(target)
            for name, parameter in target.named_parameters(remove_duplicate=False):
                self.assertEqual(parameter.device.type, "cpu")
                original_state[name] = (parameter, parameter.data_ptr(), parameter.detach().clone())
            original_to = target.to

            def guard_to(*args, **kwargs):
                device = kwargs.get("device", args[0] if args else None)
                if isinstance(device, (str, torch.device)) and torch.device(device).type != "cpu":
                    self.fail("staged recovery moved the entire source model to an accelerator")
                return original_to(*args, **kwargs)

            stack.enter_context(patch.object(target, "to", side_effect=guard_to))
            attention = target.model.layers[0].self_attn

            def observe(module, inputs):
                self.assertEqual(inputs[0].device.type, "cuda")
                self.assertEqual(module.weight.device.type, "cuda")
                self.assertEqual(attention.k_proj.weight.device.type, "cpu")
                observed.append("q_proj")
                if fail:
                    raise RuntimeError("injected staged oracle failure")

            stack.callback(attention.q_proj.register_forward_pre_hook(observe).remove)
            return target, loading

        stack.enter_context(
            patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=track)
        )
        return targets, original_state, observed

    def _assert_source_restored(self, targets, original_state, observed):
        import torch

        self.assertEqual(len(targets), 1)
        self.assertTrue(observed)
        target = targets[0]
        self.assertIs(target.get_input_embeddings().weight, target.get_output_embeddings().weight)
        for name, (parameter, pointer, values) in original_state.items():
            restored = target.get_parameter(name)
            self.assertIs(restored, parameter, name)
            self.assertEqual(restored.device.type, "cpu", name)
            self.assertEqual(restored.data_ptr(), pointer, name)
            torch.testing.assert_close(restored, values, atol=0, rtol=0)

    def test_staged_cuda_matches_resident_export_and_keeps_source_on_cpu(self):
        import torch
        from transformers import AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, resident, staged = root / "source", root / "resident", root / "staged"
            original = self._cuda_fixture(source)
            options = dict(device="cuda", dtype="float64", batch_size=3, include_tokenizer=False)
            resident_report = export_recovered_model(source, resident, **options)
            with ExitStack() as stack:
                state = self._track_oracle(stack)
                stdout = StringIO()
                with redirect_stdout(stdout):
                    self.assertEqual(
                        main(
                            [
                                "recover-layers",
                                "--model",
                                str(source),
                                "--output",
                                str(staged),
                                "--device",
                                "cuda",
                                "--dtype",
                                "float64",
                                "--batch-size",
                                "3",
                                "--without-tokenizer",
                                "--stage-layers",
                            ]
                        ),
                        0,
                    )
                staged_report = json.loads(stdout.getvalue())
                self._assert_source_restored(*state)

            self.assertEqual(resident_report["oracle_placement"], "resident")
            self.assertEqual(staged_report["oracle_placement"], "staged")
            self.assertGreater(staged_report["oracle_staged_groups"], 1)
            self.assertEqual(staged_report["oracle_host_layer_calls"], 0)
            self.assertGreater(staged_report["oracle_peak_staged_state_bytes"], 0)
            self.assertLess(
                staged_report["oracle_peak_staged_state_bytes"],
                sum(p.numel() * p.element_size() for p in original.parameters()),
            )
            for key in ("layer_calls", "probe_input_rows", "observed_output_scalars"):
                self.assertEqual(staged_report[key], resident_report[key])
            for reference in (source, resident):
                audit = compare_checkpoints(reference, staged)
                self.assertEqual(audit["coverage"], 1.0)
                self.assertEqual(audit["fraction_within_tolerance"], 1.0)
            source.rename(root / "offline-source")
            candidate = AutoModelForCausalLM.from_pretrained(
                staged, local_files_only=True, torch_dtype=torch.float64
            ).eval()
            self.assertIs(
                candidate.get_input_embeddings().weight, candidate.get_output_embeddings().weight
            )
            tokens = torch.tensor([[1, 3, 7, 2, 8]])
            with torch.inference_mode():
                torch.testing.assert_close(candidate(tokens).logits, original(tokens).logits)

    def test_staged_cuda_restores_source_after_probe_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "candidate"
            self._cuda_fixture(source)
            with ExitStack() as stack:
                state = self._track_oracle(stack, fail=True)
                with self.assertRaisesRegex(RuntimeError, "injected staged oracle failure"):
                    export_recovered_model(
                        source,
                        output,
                        device="cuda",
                        dtype="float64",
                        batch_size=3,
                        include_tokenizer=False,
                        stage_layers=True,
                    )
                self._assert_source_restored(*state)
            self.assertFalse(output.exists())

    def test_staged_cuda_supports_parent_oracles_with_extra_dependencies(self):
        import torch
        from transformers import (
            BartConfig,
            BartForConditionalGeneration,
            DebertaV2Config,
            DebertaV2ForMaskedLM,
        )

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        bart = (
            BartForConditionalGeneration(
                BartConfig(
                    vocab_size=31,
                    d_model=12,
                    encoder_layers=1,
                    decoder_layers=1,
                    encoder_attention_heads=3,
                    decoder_attention_heads=3,
                    encoder_ffn_dim=17,
                    decoder_ffn_dim=17,
                    max_position_embeddings=12,
                )
            )
            .double()
            .eval()
        )
        deberta = (
            DebertaV2ForMaskedLM(
                DebertaV2Config(
                    vocab_size=31,
                    hidden_size=12,
                    num_hidden_layers=1,
                    num_attention_heads=3,
                    intermediate_size=17,
                    max_position_embeddings=12,
                    legacy=False,
                    tie_word_embeddings=False,
                )
            )
            .double()
            .eval()
        )
        with torch.no_grad():
            bart.final_logits_bias.uniform_(-0.25, 0.25)
            deberta.lm_predictions.lm_head.bias.uniform_(-0.25, 0.25)
        tokens = torch.tensor([[1, 3, 7, 2, 8]])
        for model, host_calls, extra_inputs in (
            (bart, 1, {"decoder_input_ids": torch.tensor([[2, 4, 5]]), "use_cache": False}),
            (deberta, 0, {}),
        ):
            with (
                self.subTest(model=model.config.model_type),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source, output = root / "source", root / "candidate"
                model.save_pretrained(source, safe_serialization=True)
                report = export_recovered_model(
                    source,
                    output,
                    device="cuda",
                    dtype="float64",
                    batch_size=5,
                    include_tokenizer=False,
                    stage_layers=True,
                )
                self.assertEqual(report["oracle_host_layer_calls"], host_calls)
                audit = compare_checkpoints(source, output)
                self.assertEqual(audit["coverage"], 1.0)
                self.assertEqual(audit["fraction_within_tolerance"], 1.0)
                source.rename(root / "offline-source")
                candidate = (
                    type(model)
                    .from_pretrained(output, local_files_only=True, torch_dtype=torch.float64)
                    .eval()
                )
                with torch.inference_mode():
                    torch.testing.assert_close(
                        candidate(tokens, **extra_inputs).logits,
                        model(tokens, **extra_inputs).logits,
                    )


if __name__ == "__main__":
    unittest.main()
