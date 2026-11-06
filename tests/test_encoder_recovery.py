"""Exercise native encoder and task-head exports through the recovery CLI."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from rephix.cli import main

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class EncoderRecoveryTests(unittest.TestCase):
    def _cli_json(self, args):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(args), 0)
        return json.loads(stdout.getvalue())

    def _fixture_inputs(self, torch, config):
        inputs = {
            "input_ids": torch.tensor([[2, 5, 6, 7, 3], [2, 11, 8, 3, config.pad_token_id]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        }
        if config.model_type in {"bert", "albert"}:
            inputs["token_type_ids"] = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]])
        return inputs

    def _save_bert_tokenizer(self, source, vocab_size):
        from transformers import BertTokenizer

        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "model", "recovery", "."]
        vocab.extend(f"word{i}" for i in range(vocab_size - len(vocab)))
        vocab_file = source / "vocab.txt"
        vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        BertTokenizer(vocab_file=str(vocab_file), do_lower_case=True).save_pretrained(source)

    def _roundtrip(
        self, model_class, config, fields, *, tokenizer=False, multiple_choice=False, override=False
    ):
        import torch
        from transformers import AutoTokenizer

        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "candidate"
            torch.manual_seed(124)
            target = model_class(config).double().eval()
            with torch.no_grad():
                # Nondefault normalization, embedding, and head parameters ensure
                # the raw audit catches any coordinates left at skeleton defaults.
                for parameter in target.parameters():
                    parameter.uniform_(-0.2, 0.2)
            target.save_pretrained(source, safe_serialization=True)
            if tokenizer:
                self._save_bert_tokenizer(source, config.vocab_size)
                original_tokens = AutoTokenizer.from_pretrained(source, local_files_only=True)(
                    ["Model recovery.", "Model [MASK]"], padding=True, return_tensors="pt"
                )

            args = [
                "recover-layers",
                "--model",
                str(source),
                "--output",
                str(output),
                "--dtype",
                "float64",
                "--batch-size",
                "5",
            ]
            if not tokenizer:
                args.append("--without-tokenizer")
            if override:
                config_file = source / "config.json"
                saved_config = json.loads(config_file.read_text(encoding="utf-8"))
                saved_config["architectures"] = ["BertModel"]
                config_file.write_text(json.dumps(saved_config), encoding="utf-8")
                args.extend(("--architecture", model_class.__name__))
            report = self._cli_json(args)
            self.assertEqual(report["model_type"], config.model_type)
            self.assertEqual(report["architecture"], model_class.__name__)
            elements = sum(parameter.numel() for parameter in target.parameters())
            self.assertEqual(report["unique_parameter_elements_recovered"], elements)
            self.assertEqual(report["individually_identified_parameter_elements"], elements)
            self.assertEqual(report["tokenizer_exported"], tokenizer)

            audit = self._cli_json(["audit-weights", str(source), str(output)])
            self.assertEqual(audit["coverage"], 1.0)
            self.assertEqual(audit["fraction_within_tolerance"], 1.0)
            self.assertFalse(audit["extra_tensors"])
            self.assertFalse(audit["dtype_mismatches"])

            source.rename(Path(directory) / "offline-source")
            candidate = model_class.from_pretrained(
                output, local_files_only=True, torch_dtype=torch.float64
            ).eval()
            self.assertEqual(candidate.config.architectures, [model_class.__name__])
            self.assertEqual(candidate.config.id2label, config.id2label)
            if tokenizer:
                candidate_tokens = AutoTokenizer.from_pretrained(output, local_files_only=True)(
                    ["Model recovery.", "Model [MASK]"], padding=True, return_tensors="pt"
                )
                self.assertEqual(original_tokens.keys(), candidate_tokens.keys())
                for name in original_tokens:
                    self.assertTrue(torch.equal(original_tokens[name], candidate_tokens[name]))

            head = candidate.get_output_embeddings()
            if config.model_type == "deberta-v2" and not config.legacy:
                self.assertEqual(head.weight.shape, (config.hidden_size, config.hidden_size))
                self.assertIsNot(head.weight, candidate.get_input_embeddings().weight)
                torch.testing.assert_close(
                    candidate.lm_predictions.lm_head.bias, target.lm_predictions.lm_head.bias
                )
            elif head is not None:
                self.assertEqual(
                    head.weight is candidate.get_input_embeddings().weight,
                    config.tie_word_embeddings,
                )
                aliases = [
                    name
                    for name, parameter in candidate.named_parameters(remove_duplicate=False)
                    if parameter is head.bias
                ]
                self.assertEqual(len(aliases), 2)

            inputs = self._fixture_inputs(torch, config)
            if multiple_choice:
                inputs = {
                    name: torch.stack((value, value.flip(0)), dim=1)
                    for name, value in inputs.items()
                }
            with torch.inference_mode():
                expected, observed = target(**inputs), candidate(**inputs)
            for field in fields:
                torch.testing.assert_close(
                    getattr(observed, field), getattr(expected, field), atol=1e-12, rtol=1e-9
                )

    def test_encoder_masked_lm_preserves_heads_and_all_weights(self):
        from transformers import (
            AlbertConfig,
            AlbertForMaskedLM,
            BertConfig,
            BertForMaskedLM,
            DebertaV2Config,
            DebertaV2ForMaskedLM,
        )

        common = dict(
            vocab_size=31,
            hidden_size=12,
            num_hidden_layers=2,
            num_attention_heads=3,
            intermediate_size=17,
            max_position_embeddings=16,
        )
        cases = (
            (BertForMaskedLM, BertConfig(**common), True),
            (
                BertForMaskedLM,
                BertConfig(
                    **common,
                    position_embedding_type="relative_key_query",
                    tie_word_embeddings=False,
                ),
                False,
            ),
            (
                AlbertForMaskedLM,
                AlbertConfig(**common, embedding_size=7, num_hidden_groups=1),
                False,
            ),
            (
                DebertaV2ForMaskedLM,
                DebertaV2Config(**common, legacy=False, tie_word_embeddings=False),
                False,
            ),
        )
        for model_class, config, tokenizer in cases:
            with self.subTest(model=model_class.__name__, tokenizer=tokenizer):
                self._roundtrip(model_class, config, ("logits",), tokenizer=tokenizer)

    def test_encoder_task_heads_and_base_outputs(self):
        from transformers import (
            BertConfig,
            BertForMultipleChoice,
            BertForPreTraining,
            BertModel,
            DebertaV2Config,
            DebertaV2ForTokenClassification,
            DistilBertConfig,
            DistilBertForQuestionAnswering,
            RobertaConfig,
            RobertaForSequenceClassification,
        )

        common = dict(
            vocab_size=31,
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            intermediate_size=17,
            max_position_embeddings=16,
        )
        distilbert = DistilBertConfig(
            vocab_size=31, dim=12, n_layers=1, n_heads=3, hidden_dim=17, max_position_embeddings=16
        )
        cases = (
            (BertModel, BertConfig(**common), ("last_hidden_state", "pooler_output")),
            (
                BertForPreTraining,
                BertConfig(**common),
                ("prediction_logits", "seq_relationship_logits"),
            ),
            (BertForMultipleChoice, BertConfig(**common), ("logits",)),
            (
                RobertaForSequenceClassification,
                RobertaConfig(**common, id2label={0: "negative", 1: "neutral", 2: "positive"}),
                ("logits",),
            ),
            (DistilBertForQuestionAnswering, distilbert, ("start_logits", "end_logits")),
            (
                DebertaV2ForTokenClassification,
                DebertaV2Config(
                    **common,
                    relative_attention=True,
                    position_buckets=8,
                    pos_att_type=["c2p", "p2c"],
                    norm_rel_ebd="layer_norm",
                    conv_kernel_size=3,
                    conv_groups=3,
                ),
                ("logits",),
            ),
        )
        for model_class, config, fields in cases:
            with self.subTest(model=model_class.__name__):
                self._roundtrip(
                    model_class,
                    config,
                    fields,
                    multiple_choice=model_class is BertForMultipleChoice,
                )

    def test_encoder_architecture_override_repairs_stale_config(self):
        from transformers import BertConfig, BertForMaskedLM

        config = BertConfig(
            vocab_size=31,
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            intermediate_size=17,
            max_position_embeddings=16,
        )
        self._roundtrip(BertForMaskedLM, config, ("logits",), override=True)

    def test_encoder_rejects_missing_ambiguous_and_non_native_architectures(self):
        from transformers import BertConfig

        cases = (
            None,
            [],
            ["BertModel", "BertForMaskedLM"],
            ["BertSelfOutput"],
            ["RobertaForMaskedLM"],
        )
        for declaration in cases:
            with (
                self.subTest(architectures=declaration),
                tempfile.TemporaryDirectory() as directory,
            ):
                source, output = Path(directory) / "source", Path(directory) / "candidate"
                BertConfig(architectures=declaration).save_pretrained(source)
                stderr = StringIO()
                with redirect_stderr(stderr):
                    self.assertEqual(
                        main(
                            [
                                "recover-layers",
                                "--model",
                                str(source),
                                "--output",
                                str(output),
                                "--without-tokenizer",
                            ]
                        ),
                        1,
                    )
                self.assertIn("architecture", stderr.getvalue().lower())
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
