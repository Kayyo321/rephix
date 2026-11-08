"""Verify chosen-layer capture resumes and exports without its source checkpoint."""

import importlib.util
import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack, closing, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rephix.cli import main
from rephix.weight_audit import compare_checkpoints

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class LayerEvidenceTests(unittest.TestCase):
    def test_journal_captures_singleton_views_with_nonunit_strides(self):
        import torch

        from rephix.layer_journal import LayerJournal

        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                values = torch.arange(100, dtype=dtype)
                inputs = values.as_strided((1, 1, 1), (64, 32, 16), storage_offset=7)
                response = values.as_strided((1, 1, 1), (28, 28, 28), storage_offset=11)
                path = Path(directory) / "observations.sqlite3"
                with LayerJournal(path, torch, binding="singleton") as journal:
                    actual = journal.observe(
                        "view", inputs, (1, 1, 1), dtype, oracle=lambda *_: response
                    )
                    torch.testing.assert_close(actual, response, atol=0, rtol=0)
                    journal.finish()
                with LayerJournal(path, torch, binding="singleton", readonly=True) as journal:
                    replay = journal.observe("view", inputs, (1, 1, 1), dtype, oracle=None)
                    torch.testing.assert_close(replay, response, atol=0, rtol=0)
                    self.assertEqual(journal.fresh_calls, 0)
                    self.assertEqual(journal.reused_calls, 1)
                    journal.finish()

    def _source(self, path):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        config = LlamaConfig(
            vocab_size=19,
            hidden_size=12,
            intermediate_size=18,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            max_position_embeddings=16,
            rms_norm_eps=0.0,
            tie_word_embeddings=True,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(114)
            model = LlamaForCausalLM(config).double().eval()
        model.generation_config.max_new_tokens = 7
        model.save_pretrained(path, safe_serialization=True)
        tokenizer = Tokenizer(
            WordLevel({f"tok{index}": index for index in range(19)}, unk_token="tok0")
        )
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="tok0", bos_token="tok1", eos_token="tok2"
        ).save_pretrained(path)

    def _track_source_calls(self, stack, *, interrupt_at=None):
        from transformers import AutoModelForCausalLM

        seen = []
        load = AutoModelForCausalLM.from_pretrained

        def track(*args, **kwargs):
            model, loading = load(*args, **kwargs)

            def observe(path, inputs):
                seen.append((path, inputs[0].tolist()))
                if len(seen) == interrupt_at:
                    raise RuntimeError("injected source interruption")

            for path, module in model.named_modules():
                if next(module.parameters(recurse=False), None) is not None:
                    handle = module.register_forward_pre_hook(
                        lambda _layer, inputs, path=path: observe(path, inputs)
                    )
                    stack.callback(handle.remove)
            return model, loading

        stack.enter_context(
            patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=track)
        )
        return seen

    def _command(self, args, *, error=None):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(args)
        if error is not None:
            self.assertEqual(status, 1, stdout.getvalue())
            self.assertRegex(stderr.getvalue(), error)
            return None
        self.assertEqual(status, 0, stderr.getvalue())
        return json.loads(stdout.getvalue())

    def _capture(self, source, evidence, output, *, batch_size=3, error=None):
        return self._command(
            [
                "recover-layers",
                "--model",
                str(source),
                "--output",
                str(output),
                "--evidence-dir",
                str(evidence),
                "--dtype",
                "float64",
                "--batch-size",
                str(batch_size),
            ],
            error=error,
        )

    def _replay(self, evidence, output, *, error=None):
        return self._command(
            ["replay-layers", "--evidence-dir", str(evidence), "--output", str(output)],
            error=error,
        )

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "needs Pillow for a native processor")
    def test_vision_language_replay_preserves_processor_assets(self):
        from PIL import Image
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            AutoProcessor,
            CLIPImageProcessor,
            LlavaConfig,
            LlavaForConditionalGeneration,
            LlavaProcessor,
            PreTrainedTokenizerFast,
        )

        config = LlavaConfig(
            vision_config={
                "model_type": "clip_vision_model",
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 8,
                "patch_size": 4,
            },
            text_config={
                "model_type": "llama",
                "vocab_size": 16,
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            },
            image_token_index=3,
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                WordLevel({"[UNK]": 0, "hello": 1, "other": 2, "<image>": 3}, "[UNK]")
            ),
            unk_token="[UNK]",
            additional_special_tokens=["<image>"],
        )
        processor = LlavaProcessor(
            image_processor=CLIPImageProcessor(
                size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}
            ),
            tokenizer=tokenizer,
            patch_size=4,
            vision_feature_select_strategy="default",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, evidence, output, replay = (
                root / name for name in ("source", "evidence", "output", "replay")
            )
            LlavaForConditionalGeneration(config).double().save_pretrained(source)
            processor.save_pretrained(source)
            template = source / "chat_templates" / "alternate.jinja"
            template.parent.mkdir()
            template.write_text("{{ messages[0]['content'] }}\n", encoding="utf-8")
            inputs = {"text": "<image>", "images": Image.new("RGB", (8, 8), (16, 64, 128))}
            reference = AutoProcessor.from_pretrained(source, local_files_only=True)(
                **inputs, return_tensors="pt"
            )
            report = self._capture(source, evidence, output, batch_size=11)
            self.assertTrue(report["multimodal_processor_exported"])
            shutil.rmtree(source)
            self.assertEqual(self._replay(evidence, replay)["evidence_observations_collected"], 0)
            for checkpoint in (output, replay):
                self.assertEqual(
                    (checkpoint / "chat_templates" / template.name).read_bytes(),
                    b"{{ messages[0]['content'] }}\n",
                )
                recovered = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)(
                    **inputs, return_tensors="pt"
                )
                self.assertEqual(recovered.keys(), reference.keys())
                self.assertTrue(
                    all(recovered[name].equal(value) for name, value in reference.items())
                )

    def test_resume_skips_committed_observations_and_replays_offline(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, evidence = root / "source", root / "evidence"
            output, replay = root / "captured", root / "replayed"
            self._source(source)
            original = AutoModelForCausalLM.from_pretrained(
                source, local_files_only=True, torch_dtype=torch.float64
            ).eval()
            with ExitStack() as stack:
                interrupted = self._track_source_calls(stack, interrupt_at=5)
                self._capture(source, evidence, output, error="injected source interruption")
            self.assertEqual(len(interrupted), 5)
            self.assertFalse(output.exists())
            self._replay(evidence, replay, error="incomplete|unsealed|not complete")
            self.assertFalse(replay.exists())

            with ExitStack() as stack:
                resumed = self._track_source_calls(stack)
                captured_report = self._capture(source, evidence, output)
            self.assertEqual(len(resumed), captured_report["layer_calls"] - 4)
            self.assertEqual(captured_report["evidence_observations_cached"], 4)
            self.assertEqual(captured_report["evidence_observations_collected"], len(resumed))
            self.assertGreater(len(resumed), 0)
            self.assertEqual(resumed[0], interrupted[-1])
            for response in interrupted[:4]:
                self.assertNotIn(response, resumed)
            self.assertEqual(compare_checkpoints(source, output)["fraction_exactly_equal"], 1.0)
            self.assertFalse(any((evidence / "assets").rglob("*.safetensors")))

            shutil.rmtree(source)
            with patch.object(
                AutoModelForCausalLM,
                "from_pretrained",
                side_effect=AssertionError("offline replay loaded a source model"),
            ):
                replay_report = self._replay(evidence, replay)
            self.assertEqual(replay_report["layer_calls"], captured_report["layer_calls"])
            self.assertEqual(
                replay_report["evidence_observations_cached"], replay_report["layer_calls"]
            )
            self.assertEqual(replay_report["evidence_observations_collected"], 0)
            self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1.0)
            tokenizer = AutoTokenizer.from_pretrained(replay, local_files_only=True)
            self.assertEqual(tokenizer.encode("tok4 tok7"), [4, 7])
            recovered = AutoModelForCausalLM.from_pretrained(
                replay, local_files_only=True, torch_dtype=torch.float64
            ).eval()
            self.assertIs(
                recovered.get_input_embeddings().weight, recovered.get_output_embeddings().weight
            )
            self.assertEqual(recovered.generation_config.max_new_tokens, 7)
            tokens = torch.tensor([[1, 3, 4, 6, 2, 8]])
            with torch.inference_mode():
                torch.testing.assert_close(
                    recovered(tokens).logits, original(tokens).logits, atol=1e-14, rtol=0
                )

    def test_restored_storage_dtypes_survive_source_free_replay(self):
        import torch
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, evidence, output, replay = (
                root / name for name in ("source", "evidence", "output", "replay")
            )
            self._source(source)
            weights = source / "model.safetensors"
            state = {name: value.float() for name, value in load_file(weights).items()}
            save_file(state, weights, metadata={"format": "pt"})
            report = self._command(
                [
                    "recover-layers",
                    "--model",
                    str(source),
                    "--output",
                    str(output),
                    "--evidence-dir",
                    str(evidence),
                    "--dtype",
                    "float64",
                    "--batch-size",
                    "3",
                    "--restore-source-dtypes",
                ]
            )
            self.assertEqual(report["oracle_dtype"], "torch.float64")
            self.assertEqual(set(report["export_parameter_dtypes"].values()), {"torch.float32"})
            self.assertTrue(
                all(
                    value.dtype == torch.float32
                    for value in load_file(output / weights.name).values()
                )
            )
            self.assertEqual(compare_checkpoints(source, output)["fraction_exactly_equal"], 1)
            self._capture(source, evidence, root / "changed-option", error="metadata")
            shutil.rmtree(source)
            replayed = self._replay(evidence, replay)
            self.assertEqual(replayed["export_parameter_dtypes"], report["export_parameter_dtypes"])
            self.assertEqual(compare_checkpoints(output, replay)["fraction_exactly_equal"], 1)
            self.assertEqual(
                (output / weights.name).read_bytes(), (replay / weights.name).read_bytes()
            )

    def test_resuming_older_dtype_restore_bundle_preserves_buffer_precision(self):
        import torch
        from transformers import BartConfig, BartForConditionalGeneration

        from rephix.layer_recovery import export_recovered_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, evidence = root / "source", root / "evidence"
            config = BartConfig(
                vocab_size=19,
                d_model=12,
                encoder_layers=1,
                decoder_layers=1,
                encoder_attention_heads=3,
                decoder_attention_heads=3,
                encoder_ffn_dim=18,
                decoder_ffn_dim=18,
                max_position_embeddings=16,
            )
            BartForConditionalGeneration(config).save_pretrained(source)
            options = dict(dtype="float64", include_tokenizer=False, restore_source_dtypes=True)
            with patch("rephix.layer_recovery.source_buffer_dtypes", return_value={}):
                original = export_recovered_model(
                    source, root / "original", evidence_dir=evidence, **options
                )
            resumed = export_recovered_model(
                source, root / "resumed", evidence_dir=evidence, **options
            )
            self.assertNotIn("export_buffer_dtypes", resumed)
            self.assertEqual(resumed["evidence_observations_cached"], original["layer_calls"])
            self.assertEqual(resumed["evidence_observations_collected"], 0)
            self.assertEqual(
                (root / "original" / "model.safetensors").read_bytes(),
                (root / "resumed" / "model.safetensors").read_bytes(),
            )
            fresh = export_recovered_model(source, root / "fresh", **options)
            self.assertEqual(fresh["export_buffer_dtypes"], {"final_logits_bias": "torch.float32"})
            self.assertEqual(
                BartForConditionalGeneration.from_pretrained(
                    root / "fresh"
                ).final_logits_bias.dtype,
                torch.float32,
            )

    def test_resume_rejects_changed_probe_options_or_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, evidence, output = root / "source", root / "evidence", root / "captured"
            self._source(source)
            with ExitStack() as stack:
                self._track_source_calls(stack, interrupt_at=2)
                self._capture(source, evidence, output, error="injected source interruption")

            with ExitStack() as stack:
                seen = self._track_source_calls(stack)
                self._capture(source, evidence, output, batch_size=4, error="evidence|journal")
                config_path = source / "config.json"
                config = json.loads(config_path.read_text())
                config["max_position_embeddings"] = 32
                config_path.write_text(json.dumps(config))
                self._capture(source, evidence, output, error="evidence|journal")
            self.assertEqual(seen, [])
            self.assertFalse(output.exists())

    def test_replay_rejects_corrupted_assets_and_responses_without_exporting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, evidence = root / "source", root / "evidence"
            output, replay = root / "captured", root / "replayed"
            self._source(source)
            self._capture(source, evidence, output)

            config = evidence / "assets" / "config.json"
            original = config.read_bytes()
            config.write_bytes(original + b"\n")
            self._replay(evidence, replay, error="scaffold assets changed")
            self.assertFalse(replay.exists())
            config.write_bytes(original)

            with closing(sqlite3.connect(evidence / "observations.sqlite3")) as database, database:
                database.execute(
                    "UPDATE responses SET payload = zeroblob(length(payload)) "
                    "WHERE position = (SELECT min(position) FROM responses)"
                )
            self._replay(evidence, replay, error="checksum|corrupt")
            self.assertFalse(replay.exists())


if __name__ == "__main__":
    unittest.main()
