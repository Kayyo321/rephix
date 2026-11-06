"""Exercise independent encoder-decoder exports from layer probes."""

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
class Seq2SeqRecoveryTests(unittest.TestCase):
    def _cli_json(self, args):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(args), 0)
        return json.loads(stdout.getvalue())

    def _recover(self, source, output, *, dtype="float64", attention_only=False):
        args = [
            "recover-layers",
            "--model",
            str(source),
            "--output",
            str(output),
            "--without-tokenizer",
            "--dtype",
            dtype,
            "--batch-size",
            "7",
        ]
        if attention_only:
            args.extend(("--relative-bias-mode", "attention"))
        return self._cli_json(args)

    def _t5_configs(self):
        from transformers import MT5Config, T5Config, UMT5Config

        common = dict(
            vocab_size=23,
            d_model=12,
            d_kv=4,
            d_ff=18,
            num_layers=2,
            num_decoder_layers=1,
            num_heads=3,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
            decoder_start_token_id=0,
            feed_forward_proj="gated-gelu",
        )
        return (
            T5Config(**common, tie_word_embeddings=True),
            MT5Config(**common, tie_word_embeddings=False),
            UMT5Config(**common, tie_word_embeddings=True),
        )

    def _check_export(self, config, *, attention_only=False):
        import torch
        from transformers import AutoModelForSeq2SeqLM

        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(31)
            source, output = Path(tmp) / "source", Path(tmp) / "candidate"
            target = AutoModelForSeq2SeqLM.from_config(config).double().eval()
            target.generation_config.min_new_tokens = 3
            target.generation_config.max_new_tokens = 3
            omitted = set()
            with torch.no_grad():
                if config.model_type == "bart":
                    target.final_logits_bias.copy_(torch.linspace(0.125, 0.5, config.vocab_size))
                    target.get_encoder().embed_positions.weight[:2].fill_(0.375)
                    target.get_decoder().embed_positions.weight[:2].fill_(-0.625)
                else:
                    for name, module in target.named_modules():
                        table = getattr(module, "relative_attention_bias", None)
                        if table is None:
                            continue
                        table.weight.uniform_(0.25, 0.75)
                        if attention_only and name.startswith("encoder."):
                            omitted.add(name)
            target.save_pretrained(source, safe_serialization=True)
            report = self._recover(source, output, attention_only=attention_only)
            self.assertEqual(report["model_type"], config.model_type)
            expected_disagreements = len(omitted) * getattr(config, "num_heads", 0)
            if config.model_type != "bart":
                parameter_elements = sum(parameter.numel() for parameter in target.parameters())
                self.assertEqual(report["unique_parameter_elements_recovered"], parameter_elements)
                self.assertEqual(
                    report["individually_identified_parameter_elements"],
                    parameter_elements - expected_disagreements,
                )
            if omitted:
                self.assertEqual(
                    len(omitted), config.num_layers if config.model_type == "umt5" else 1
                )
                gauges = report["relative_bias_gauges"]
                self.assertEqual({entry["module"] for entry in gauges}, omitted)
                for entry in gauges:
                    self.assertEqual(entry["unreachable_buckets"], [4])
                    self.assertEqual(entry["affected_stored_elements"], config.num_heads)

            audit = self._cli_json(["audit-weights", str(source), str(output)])
            self.assertEqual(audit["coverage"], 1.0)
            self.assertFalse(audit["extra_tensors"])
            self.assertFalse(audit["dtype_mismatches"])
            self.assertEqual(
                audit["reference_elements"] - audit["within_tolerance_elements"],
                expected_disagreements,
            )

            source.rename(Path(tmp) / "offline-source")
            candidate = AutoModelForSeq2SeqLM.from_pretrained(
                output, local_files_only=True, torch_dtype=torch.float64
            ).eval()
            embeddings = candidate.get_input_embeddings().weight
            self.assertIs(candidate.get_encoder().get_input_embeddings().weight, embeddings)
            self.assertIs(candidate.get_decoder().get_input_embeddings().weight, embeddings)
            self.assertEqual(
                candidate.get_output_embeddings().weight is embeddings, config.tie_word_embeddings
            )
            for path in omitted:
                weights = candidate.get_submodule(path).relative_attention_bias.weight
                self.assertEqual(torch.count_nonzero(weights[4]).item(), 0)

            # Different source and target lengths, with a padded source. Generation
            # additionally exercises the decoder cache and saved generation config.
            input_ids = torch.tensor(
                [[4, 6, 5, 8, 9, 7, 2], [7, 4, 2, 6, 5, 2, config.pad_token_id]]
            )
            mask = input_ids.ne(config.pad_token_id)
            decoder_input_ids = torch.tensor([[config.decoder_start_token_id, 5, 9, 12]] * 2)
            with torch.inference_mode():
                source_kwargs = dict(input_ids=input_ids, attention_mask=mask)
                forward_kwargs = dict(**source_kwargs, decoder_input_ids=decoder_input_ids)
                torch.testing.assert_close(
                    candidate(**forward_kwargs).logits,
                    target(**forward_kwargs).logits,
                    atol=1e-10,
                    rtol=1e-8,
                )
                expected_tokens = target.generate(**source_kwargs)
                self.assertEqual(expected_tokens.shape, (2, 4))
                torch.testing.assert_close(candidate.generate(**source_kwargs), expected_tokens)

    def test_native_t5_variants_and_bart_export_stored_weights(self):
        from transformers import BartConfig

        bart = BartConfig(
            vocab_size=23,
            d_model=12,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=3,
            decoder_attention_heads=3,
            encoder_ffn_dim=18,
            decoder_ffn_dim=18,
            max_position_embeddings=16,
            scale_embedding=True,
        )
        for config in (*self._t5_configs(), bart):
            with self.subTest(model=config.model_type):
                self._check_export(config)

    def test_attention_bias_probes_report_unreachable_encoder_rows(self):
        for config in self._t5_configs():
            with self.subTest(model=config.model_type):
                self._check_export(config, attention_only=True)

    def test_mt5_auto_dtype_preserves_mixed_precision_projection_weights(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM

        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(73)
            source, output = Path(tmp) / "source", Path(tmp) / "candidate"
            target = AutoModelForSeq2SeqLM.from_config(self._t5_configs()[1]).eval()
            # The native mT5 loader keeps the feed-forward output projections in
            # float32 even when the model's declared checkpoint dtype is float16.
            with torch.no_grad():
                for name, weight in target.named_parameters():
                    if name.endswith(".wo.weight"):
                        weight[0, 0] = 1.0001
                    else:
                        weight.data = weight.detach().half()
            target.save_pretrained(source, safe_serialization=True)

            report = self._recover(source, output, dtype="auto")
            self.assertEqual(report["oracle_dtype"], "torch.float16")
            audit = self._cli_json(["audit-weights", str(source), str(output)])
            self.assertEqual(audit["coverage"], 1.0)
            self.assertFalse(audit["dtype_mismatches"])
            self.assertEqual(audit["fraction_exactly_equal"], 1.0)

            source.rename(Path(tmp) / "offline-source")
            candidate = AutoModelForSeq2SeqLM.from_pretrained(
                output, local_files_only=True, torch_dtype="auto"
            ).eval()
            self.assertEqual(candidate.get_input_embeddings().weight.dtype, torch.float16)
            for name, expected in target.named_parameters():
                recovered = candidate.get_parameter(name)
                if name.endswith(".wo.weight"):
                    self.assertEqual(recovered.dtype, torch.float32)
                    self.assertNotEqual(recovered[0, 0].item(), recovered[0, 0].half().item())
                torch.testing.assert_close(recovered, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
