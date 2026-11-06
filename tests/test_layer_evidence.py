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
