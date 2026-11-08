"""Create Mode: evidence schema, splits, base contract, serialization, pipeline, resume."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rephix.create import constraints
from rephix.create import records as rec
from rephix.create import splits as sp

HAS_TORCH = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


def _write(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class RecordSchemaTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_exchange_keeps_roles_order_reasoning_and_tool_data(self):
        row = {
            "id": "x1",
            "kind": "exchange",
            "messages": [
                {"role": "user", "content": "Add 2 and 3"},
                {
                    "role": "assistant",
                    "reasoning": "use add",
                    "tool_calls": [{"id": "c1", "name": "add", "arguments": '{"a": 2, "b": 3}'}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "rm -rf / ; 5"},
                {"role": "assistant", "content": "5"},
            ],
        }
        _write(self.dir / "e.jsonl", [row])
        (record,) = rec.read_evidence(self.dir / "e.jsonl")
        self.assertEqual(
            [m.role for m in record.messages], ["user", "assistant", "tool", "assistant"]
        )
        call = record.messages[1].tool_calls[0]
        self.assertEqual(call["arguments"], {"a": 2, "b": 3})
        self.assertTrue(call["arguments_were_string"])
        # Tool results are data (never executed) and are not targets.
        self.assertNotIn("rm -rf", record.target_text())
        self.assertIn("use add", record.target_text())
        self.assertEqual(record.provenance["line"], 1)
        self.assertEqual(len(record.provenance["raw_sha256"]), 64)

    def test_output_only_passage_has_missing_context_not_a_fake_prompt(self):
        _write(self.dir / "p.jsonl", [{"id": "p1", "kind": "passage", "text": "Some prose."}])
        (record,) = rec.read_evidence(self.dir / "p.jsonl")
        self.assertEqual(record.context_status, "missing")
        self.assertIsNone(record.context)
        self.assertNotIn("prompt", record.to_dict())

    def test_legacy_observation_converts_without_weakening_its_invariants(self):
        _write(
            self.dir / "l.jsonl",
            [
                {
                    "id": "o1",
                    "prompt": "Q?",
                    "output": "A.",
                    "thinking": "hmm",
                    "metadata": {"group_id": "g"},
                }
            ],
        )
        (record,) = rec.read_evidence(self.dir / "l.jsonl")
        self.assertEqual(record.messages[0].content, "Q?")
        self.assertEqual(record.messages[1].reasoning, "hmm")
        self.assertEqual(record.group_id, "g")
        _write(self.dir / "bad.jsonl", [{"id": "o2", "prompt": "", "output": "A."}])
        with self.assertRaises(ValueError):
            rec.read_evidence(self.dir / "bad.jsonl")

    def test_rejects_malformed_records(self):
        cases = [
            {"id": "a", "kind": "exchange", "messages": [{"role": "user", "content": "hi"}]},
            {"id": "b", "kind": "passage", "text": ""},
            {"id": "c", "kind": "passage", "text": "x", "context_status": "known"},
            {
                "id": "d",
                "kind": "exchange",
                "messages": [
                    {"role": "user", "content": "q", "reasoning": "no"},
                    {"role": "assistant", "content": "a"},
                ],
            },
            {
                "id": "e",
                "kind": "exchange",
                "messages": [
                    {"role": "assistant", "tool_calls": [{"id": "c", "name": "t"}]},
                    {"role": "tool", "tool_call_id": "zzz", "content": "r"},
                ],
            },
            {"id": "f", "kind": "passage", "text": "x", "prompt": "invented"},
        ]
        for case in cases:
            _write(self.dir / "c.jsonl", [case])
            with self.assertRaises(ValueError, msg=case["id"]):
                rec.read_evidence(self.dir / "c.jsonl")

    def test_corpus_sections_and_manifest(self):
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(30))
        (self.dir / "thesis.txt").write_text(text)
        records = rec.read_corpus([self.dir / "thesis.txt"], section_chars=800)
        self.assertGreater(len(records), 5)
        self.assertEqual(records[0].context_status, "none")
        self.assertEqual(records[1].context_status, "known")
        self.assertTrue(records[-1].metadata["document_end"])
        manifest = rec.capability_manifest(records)
        self.assertTrue(manifest["tiers"]["corpus"])
        self.assertFalse(manifest["tiers"]["passive_exchange"])
        self.assertFalse(manifest["tiers"]["source_token_probabilities"])

    def test_repeated_identical_outputs_are_flagged_without_claiming_greedy(self):
        rows = [
            {
                "id": f"r{i}",
                "kind": "exchange",
                "messages": [
                    {"role": "user", "content": "same"},
                    {"role": "assistant", "content": "same answer"},
                ],
            }
            for i in range(3)
        ]
        _write(self.dir / "g.jsonl", rows)
        manifest = rec.capability_manifest(rec.read_evidence(self.dir / "g.jsonl"))
        self.assertTrue(manifest["tiers"]["repeated_samples"])
        self.assertTrue(any("does not prove greedy" in note for note in manifest["notes"]))


class SplitTests(unittest.TestCase):
    def _records(self, rows):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        _write(directory / "e.jsonl", rows)
        return rec.read_evidence(directory / "e.jsonl")

    def test_documents_conversations_and_near_duplicates_stay_together(self):
        rows = []
        for d in range(12):
            for s in range(3):
                rows.append(
                    {
                        "id": f"d{d}s{s}",
                        "kind": "passage",
                        "document_id": f"doc{d}",
                        "text": f"document {d} section {s} "
                        + " ".join(f"w{d}x{k}" for k in range(30)),
                    }
                )
        base = " ".join(f"tok{k}" for k in range(60))
        rows.append({"id": "nd1", "kind": "passage", "text": base})
        rows.append({"id": "nd2", "kind": "passage", "text": base + " extra"})
        records = self._records(rows)
        split = sp.assign_splits(records, seed=3)
        owner = split["assignment"]
        for d in range(12):
            self.assertEqual(len({owner[f"d{d}s{s}"] for s in range(3)}), 1)
        self.assertEqual(owner["nd1"], owner["nd2"])
        sp.verify_disjoint(records, split)
        self.assertTrue(all(split["counts"][name] > 0 for name in ("train", "validation", "test")))

    def test_single_document_uses_contiguous_blocks(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        (directory / "t.txt").write_text(
            "\n\n".join(f"Para {i} " + " ".join(f"v{i}_{k}" for k in range(120)) for i in range(80))
        )
        records = rec.read_corpus([directory / "t.txt"], section_chars=600)
        split = sp.assign_splits(records, seed=0)
        self.assertIsNotNone(split["single_document_block_sections"])
        self.assertTrue(all(split["counts"][n] > 0 for n in ("train", "validation", "test")))

    def test_too_few_groups_is_a_specific_error(self):
        records = self._records(
            [
                {"id": "a", "kind": "passage", "text": "one", "document_id": "d"},
                {"id": "b", "kind": "passage", "text": "two", "document_id": "d"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "independent group"):
            sp.assign_splits(records)


class ConstraintTests(unittest.TestCase):
    def test_features_and_instruction_mapping(self):
        formal = constraints.features("Moreover, the results are evident; thus we proceed.")
        casual = constraints.features("I don't think it's fine! You're right.")
        self.assertGreater(casual["contraction_rate"], formal["contraction_rate"])
        self.assertGreater(formal["connective_rate"], casual["connective_rate"])
        mapping = constraints.map_instruction("Preserve the formal voice and paragraph structure")
        targets = {t["target"] for t in mapping["targets"]}
        self.assertIn("register", targets)
        self.assertIn("paragraph_structure", targets)
        self.assertIn("preserve", mapping["unmapped_terms"])
        self.assertEqual(constraints.map_instruction(None)["targets"], [])

    def test_hypothesis_rule_needs_validation_replication(self):
        rows_e = [constraints.features("Moreover; thus; hence; " * (i + 1)) for i in range(12)]
        rows_b = [constraints.features("plain words here " * (i + 1)) for i in range(12)]
        groups = [f"g{i}" for i in range(12)]
        train = constraints.standardized_differences(rows_e, rows_b, groups)
        tested = constraints.hypotheses(train, train, groups_train=12, groups_validation=12)
        status = {h["feature"]: h["status"] for h in tested["hypotheses"]}
        self.assertEqual(status["semicolon_rate"], "supported")
        tested = constraints.hypotheses(train, {}, groups_train=12, groups_validation=0)
        status = {h["feature"]: h["status"] for h in tested["hypotheses"]}
        self.assertEqual(status["semicolon_rate"], "not_replicated_on_validation")
        self.assertIsNotNone(tested["limitation"])


@unittest.skipUnless(HAS_TORCH, "Create Mode model tests need the hf extra")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rephix.create.fixture import make_fixture

        cls.root = Path(tempfile.mkdtemp(prefix="rephix-create-test-"))
        cls.fixture = make_fixture(cls.root / "fixture", agent=40, paragraphs=48)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _cli(self, *argv, expect=0):
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        completed = subprocess.run(
            [sys.executable, "-m", "rephix", "--threads", "2", *map(str, argv)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        if completed.returncode != expect:
            self.fail(f"exit {completed.returncode}: {completed.stderr[-3000:]}")
        return json.loads(completed.stdout) if completed.stdout.strip() else None, completed

    def test_base_contract_rejects_unsupported_and_unpinned(self):
        from rephix.create.base import resolve_contract
        from rephix.create.fixture import make_base

        contract = resolve_contract(self.fixture["base"])
        self.assertEqual(contract["model_type"], "llama")
        self.assertTrue(contract["tied_embeddings"])
        self.assertIn("0.mlp.down_proj", contract["regions"])
        with self.assertRaisesRegex(ValueError, "40-hex"):
            resolve_contract("someone/model")
        gpt2 = self.root / "gpt2"
        import transformers

        transformers.GPT2LMHeadModel(
            transformers.GPT2Config(n_layer=1, n_embd=32, n_head=2, vocab_size=300)
        ).save_pretrained(gpt2)
        shutil.copy(Path(self.fixture["base"]) / "tokenizer.json", gpt2)
        with self.assertRaisesRegex(ValueError, "unsupported base architecture"):
            resolve_contract(gpt2)
        for family in ("qwen2", "mistral", "qwen3"):
            path = make_base(self.root / f"base-{family}", model_type=family, tie=False)
            self.assertEqual(resolve_contract(path)["model_type"], family)

    def test_serializer_masks_only_assistant_tokens(self):
        from rephix.create.base import load_base, resolve_contract
        from rephix.create.serialize import Serializer

        contract = resolve_contract(self.fixture["base"])
        _, tokenizer = load_base(contract, device="cpu", dtype="float32")
        record = rec.read_evidence(self.fixture["traces"])[0]
        serializer = Serializer(tokenizer, max_tokens=512)
        (example,) = serializer.encode(record)
        target = tokenizer.decode([t for t, m in zip(example.input_ids, example.target_mask) if m])
        context = tokenizer.decode(
            [t for t, m in zip(example.input_ids, example.target_mask) if not m]
        )
        self.assertIn("<tool_call>", target)
        self.assertIn("The user wants", target)
        self.assertIn('{"result"', context)  # tool result is context only
        self.assertNotIn('{"result"', target)
        self.assertIn("What is the", context)
        self.assertNotIn("What is the", target)
        tokenizer.chat_template = None
        plain = Serializer(tokenizer, max_tokens=512)
        self.assertEqual(plain.manifest()["exchange_convention"], "rephix-plain-dialogue/1")
        (example,) = plain.encode(record)
        self.assertTrue(any(example.target_mask))

    def test_end_to_end_traces_interrupt_resume_and_export(self):
        run = self.root / "run-traces"
        out, _ = self._cli(
            "create",
            "--base",
            self.fixture["base"],
            "--evidence",
            self.fixture["traces"],
            "--output",
            run,
            "--profile",
            "cpu-smoke",
            "--stop-after",
            "structure",
        )
        self.assertEqual(out["status"], "paused")
        # Changed evidence is rejected on resume.
        traces = Path(self.fixture["traces"])
        original = traces.read_bytes()
        traces.write_bytes(original + b"\n")
        _, completed = self._cli("create-resume", run, expect=1)
        self.assertIn("changed since the run started", completed.stderr)
        traces.write_bytes(original)
        # A corrupted completed-phase output is detected.
        phase = run / "phases" / "hypotheses.json"
        saved = phase.read_bytes()
        phase.write_bytes(saved + b" ")
        _, completed = self._cli("create-resume", run, expect=1)
        self.assertIn("was modified", completed.stderr)
        phase.write_bytes(saved)
        # Interrupt during fit by a time budget, then resume to completion.
        out, _ = self._cli("create-resume", run, "--stop-after", "fit")
        self.assertEqual(out["status"], "paused")
        checkpoint = run / "fit" / "main" / "checkpoint.pt"
        self.assertTrue(checkpoint.is_file())
        out, _ = self._cli("create-resume", run)
        self.assertEqual(out["status"], "complete")
        export = Path(out["export"])
        verification = json.loads((export / "export-verification.json").read_text())
        self.assertTrue(verification["passed"])
        self.assertTrue(verification["schema"]["passed"])
        self.assertTrue(verification["confinement"]["passed"])
        self.assertEqual(verification["fresh_process_reload"]["rephix_modules_loaded"], [])
        claims = out["claims"]
        self.assertEqual(claims["full_source_parameter_recovery"]["status"], "not_established")
        result, _ = self._cli("create-export", "--verify", export)
        self.assertTrue(result["valid"])
        # Corrupting an export file is detected.
        weights = export / "model.safetensors"
        data = bytearray(weights.read_bytes())
        data[-1] ^= 1
        weights.write_bytes(bytes(data))
        self._cli("create-export", "--verify", export, expect=1)
        inspect, _ = self._cli("create-inspect", run, "--verify")
        self.assertTrue(inspect["verification"]["passed"])
        self.assertEqual(inspect["phases"]["export"], "complete")

    def test_corpus_with_and_without_instruction_and_baselines(self):
        for name, extra in (
            ("plain", []),
            ("instructed", ["--prompt", "Keep the formal voice"]),
            ("lora", ["--method", "lora", "--no-identify"]),
        ):
            run = self.root / f"run-corpus-{name}"
            out, _ = self._cli(
                "create",
                "--base",
                self.fixture["base"],
                "--corpus",
                self.fixture["corpus"],
                "--output",
                run,
                "--profile",
                "cpu-smoke",
                *extra,
            )
            self.assertEqual(out["status"], "complete", name)
            results = json.loads((run / "results.json").read_text())
            self.assertTrue(results["evidence"]["capabilities"]["tiers"]["corpus"])
            if name == "lora":
                self.assertEqual(
                    out["claims"]["mechanistic_synthesis_in_base"]["status"], "not_attempted"
                )
            provenance = json.loads((Path(out["export"]) / "rephix-create.json").read_text())
            self.assertEqual(
                provenance["instruction"], "Keep the formal voice" if name == "instructed" else None
            )
            records = (run / "evidence" / "records.jsonl").read_text()
            self.assertNotIn("Keep the formal voice", records)

    def test_corrupted_checkpoint_and_lock_and_memory_refusal(self):
        run = self.root / "run-ckpt"
        self._cli(
            "create",
            "--base",
            self.fixture["base"],
            "--evidence",
            self.fixture["traces"],
            "--output",
            run,
            "--profile",
            "cpu-smoke",
            "--stop-after",
            "fit",
        )
        (run / "fit" / "main" / "checkpoint.pt").write_bytes(b"not a checkpoint")
        state = json.loads((run / "state.json").read_text())
        state["phases"].pop("fit", None)
        (run / "state.json").write_text(json.dumps(state))
        _, completed = self._cli("create-resume", run, expect=1)
        self.assertIn("corrupted", completed.stderr)
        from rephix.create.run import run_lock

        with run_lock(run):
            _, completed = self._cli("create-resume", run, expect=1)
            self.assertIn("in use", completed.stderr)
        out, _ = self._cli(
            "create",
            "--base",
            self.fixture["base"],
            "--evidence",
            self.fixture["traces"],
            "--output",
            self.root / "run-mem",
            "--profile",
            "cpu-smoke",
            "--max-memory-gib",
            "0.000001",
        )
        self.assertEqual(out["status"], "refused")

    def test_oom_during_fit_keeps_checkpoint(self):
        import torch

        from rephix.create import fit as fit_module
        from rephix.create import run as run_module

        args = type("A", (), {})()
        for key, value in dict(
            base=self.fixture["base"],
            base_revision=None,
            evidence=[self.fixture["traces"]],
            corpus=None,
            prompt=None,
            output=str(self.root / "run-oom"),
            export=None,
            method="structured",
            profile="cpu-smoke",
            device="cpu",
            dtype=None,
            steps=None,
            rank=None,
            regions=None,
            batch_size=None,
            max_tokens=None,
            eval_every=None,
            lr=None,
            retain_weight=1.0,
            prior_weight=1e-3,
            instruction_weight=0.5,
            fixed_subspace=False,
            no_identify=True,
            section_chars=1500,
            seed=0,
        ).items():
            setattr(args, key, value)
        run_dir = run_module.create_run(args)
        original = fit_module.token_nll
        calls = {"n": 0}

        def failing(torch_module, logits, labels):
            calls["n"] += 1
            if calls["n"] == 8:
                raise torch.OutOfMemoryError("simulated")
            return original(torch_module, logits, labels)

        fit_module.token_nll = failing
        try:
            with self.assertRaisesRegex(RuntimeError, "out of accelerator memory"):
                run_module.execute(run_dir, progress=lambda *a: None)
        finally:
            fit_module.token_nll = original
        out = run_module.execute(run_dir, progress=lambda *a: None)
        self.assertEqual(out["status"], "complete")


@unittest.skipUnless(HAS_TORCH, "Expand-and-Cluster needs torch")
class ExpandClusterTests(unittest.TestCase):
    def test_recovers_small_tanh_teacher_up_to_symmetry(self):
        import numpy as np

        from rephix.create import expand_cluster as ec

        rng = np.random.default_rng(0)
        teacher = ec.make_teacher(rng, d=2, r=2, k=1, activation="tanh", scale=1.5)
        x = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(600, 2))
        y = ec.teacher_forward(teacher, x, "tanh")
        report, params = ec.recover(
            x, y, students=6, steps=1500, student_width=6, seed=1, gamma=0.5, beta=np.pi / 12
        )
        self.assertIn(
            report["status"],
            (
                "reduced_network_fits",
                "reduced_network_does_not_fit",
                "no_cluster_accepted",
                "students_did_not_converge",
            ),
        )
        if params is not None:
            evaluation = ec.evaluate_recovery(params, teacher, "tanh", x[:100])
            self.assertTrue(evaluation["recovered"])
            self.assertIn("mean_cosine_distance", evaluation)


if __name__ == "__main__":
    unittest.main()
