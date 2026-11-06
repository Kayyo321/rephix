"""Exercise packed GGUF recovery using disposable local fixtures."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

HAS_GGUF = importlib.util.find_spec("gguf") is not None


def _q8_block(scale, codes):
    return np.concatenate(
        (
            np.array([scale], dtype="<f2").view(np.uint8),
            np.asarray(codes, dtype=np.int8).view(np.uint8),
        )
    )


def _q4_block(scale, codes):
    nibbles = (np.asarray(codes, dtype=np.int8) + 8).astype(np.uint8)
    return np.concatenate(
        (np.array([scale], dtype="<f2").view(np.uint8), nibbles[:16] | (nibbles[16:] << 4))
    )


class GgufBlockRecoveryTests(unittest.TestCase):
    def test_q8_nonstandard_signed_extreme_still_round_trips(self):
        from rephix.gguf_quant import recover_blocks

        codes = np.resize(np.array([-128, 127, -13, 1, 0], dtype=np.int8), 32)
        scale = np.float16(0.375)
        observed = (codes * np.float32(scale))[None, :]
        recovered, stats = recover_blocks(observed, "Q8_0")

        np.testing.assert_array_equal(recovered[0], _q8_block(scale, codes))
        self.assertEqual(stats["blocks"], 1)
        self.assertEqual(stats["reference_convention_blocks"], 0)

    def test_q8_report_distinguishes_zero_and_sign_ambiguity(self):
        from rephix.gguf_quant import recover_blocks

        codes = np.resize(np.array([127, -61, 3, 0], dtype=np.int8), 32)
        scale = np.float16(0.125)
        observed = np.stack([codes * np.float32(scale), np.zeros(32, dtype=np.float32)])
        packed, stats = recover_blocks(observed, "Q8_0")

        # Opposite signs on the scale and codes give the same observable values.
        # A zero scale similarly leaves no evidence of the stored codes.
        alternate = np.stack([_q8_block(-scale, -codes), _q8_block(0, codes)])
        self.assertFalse(np.array_equal(packed, alternate))
        np.testing.assert_array_equal(packed[0], _q8_block(scale, codes))
        np.testing.assert_array_equal(packed[1], np.zeros(34, dtype=np.uint8))
        self.assertEqual(stats["blocks"], 2)
        self.assertEqual(stats["zero_blocks"], 1)
        self.assertEqual(stats["ambiguous_blocks"], 2)

    def test_nonfinite_or_nonlattice_outputs_are_rejected(self):
        from rephix.gguf_quant import recover_blocks

        bad = np.ones((1, 32), dtype=np.float32)
        bad[0, 1] = np.nextafter(np.float32(1), np.float32(2))
        with self.assertRaises(ValueError):
            recover_blocks(bad, "Q8_0")
        bad[0, 1] = np.inf
        with self.assertRaises(ValueError):
            recover_blocks(bad, "Q8_0")

    def test_q4_prefers_signed_extreme_and_zero_reference_encodings(self):
        from rephix.gguf_quant import recover_blocks

        signed = np.resize(np.array([-8, 7, -3, 1, 0], dtype=np.int8), 32)
        noncanonical = np.resize(np.array([-4, 3, 1, 0], dtype=np.int8), 32)
        observed = np.stack(
            [
                signed * np.float32(-0.5),
                noncanonical * np.float32(0.5),
                np.zeros(32, dtype=np.float32),
            ]
        )
        packed, stats = recover_blocks(observed, "Q4_0")

        np.testing.assert_array_equal(packed[0], _q4_block(-0.5, signed))
        np.testing.assert_array_equal(packed[1], _q4_block(0.25, noncanonical * 2))
        np.testing.assert_array_equal(packed[2], _q4_block(-0.0, np.zeros(32, dtype=np.int8)))
        self.assertEqual(stats["reference_convention_blocks"], 3)
        self.assertEqual(stats["ambiguous_blocks"], 2)


@unittest.skipUnless(HAS_GGUF, "needs the optional GGUF package")
class GgufRecoveryTests(unittest.TestCase):
    @staticmethod
    def _fixture(path, *, seed=24, ambiguous=False, unsupported=False):
        import gguf

        kinds = gguf.GGMLQuantizationType
        rng = np.random.default_rng(seed)
        source_values = rng.normal(size=(5, 64)).astype(np.float32)
        # Explicitly use a different leading rank for the floating tensors.
        tensors = {
            "output_norm.weight": (np.array([1.25, -0.5, 0.75, 2.0], dtype=np.float32), None),
            "blk.0.ffn_norm.weight": (rng.normal(size=(2, 3, 8)).astype(np.float16), None),
            "blk.0.attn_output.weight": (gguf.quantize(source_values, kinds.BF16), kinds.BF16),
            "blk.0.attn_q.weight": (gguf.quantize(source_values, kinds.Q8_0), kinds.Q8_0),
            "blk.0.attn_k.weight": (gguf.quantize(source_values, kinds.Q4_0), kinds.Q4_0),
        }
        if ambiguous:
            codes = np.resize(np.array([127, -61, 3, 0], dtype=np.int8), 32)
            tensors["token_embd.weight"] = (
                np.stack([_q8_block(-0.125, -codes), _q8_block(0, codes)]),
                kinds.Q8_0,
            )
        if unsupported:
            tensors["blk.0.attn_v.weight"] = (
                gguf.quantize(source_values, kinds.Q4_1),
                kinds.Q4_1,
            )

        writer = gguf.GGUFWriter(path, "llama")
        try:
            writer.add_name("Tiny reconstruction fixture")
            writer.add_custom_alignment(64)
            writer.add_array("tokenizer.ggml.tokens", ["α", " hello", "</s>"])
            writer.add_key_value(
                "fixture.token_types", [1, 1, 3], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32
            )
            writer.add_key_value(
                "fixture.empty", ["x" * 56], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING
            )
            for name, (data, kind) in tensors.items():
                writer.add_tensor(name, data, raw_dtype=kind)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        # The writer rejects empty arrays even though the reader accepts them.
        # Remove one alignment unit and set the array's element count to zero.
        reader = gguf.GGUFReader(path)
        field = reader.fields["fixture.empty"]
        count_offset = field.offset + sum(part.nbytes for part in field.parts[:4])
        end = count_offset + sum(part.nbytes for part in field.parts[4:])
        del field, reader
        archive = path.read_bytes()
        path.write_bytes(archive[:count_offset] + bytes(8) + archive[end:])
        return {
            name: data.astype(np.float32) if kind is None else gguf.dequantize(data, kind)
            for name, (data, kind) in tensors.items()
        }

    def _compare(self, source, candidate, *, changed=()):
        import gguf

        a, b = gguf.GGUFReader(source), gguf.GGUFReader(candidate)
        self.assertEqual(a.alignment, 64)
        self.assertEqual(a.alignment, b.alignment)
        self.assertEqual(list(a.fields), list(b.fields))
        for name, field in a.fields.items():
            self.assertEqual(field.types, b.fields[name].types, name)
            self.assertEqual(
                [part.tobytes() for part in field.parts],
                [part.tobytes() for part in b.fields[name].parts],
                name,
            )

        self.assertEqual([t.name for t in a.tensors], [t.name for t in b.tensors])
        for original, recovered in zip(a.tensors, b.tensors, strict=True):
            self.assertEqual(original.tensor_type, recovered.tensor_type)
            np.testing.assert_array_equal(original.shape, recovered.shape)
            if original.name in changed:
                self.assertNotEqual(original.data.tobytes(), recovered.data.tobytes())
            else:
                self.assertEqual(original.data.tobytes(), recovered.data.tobytes())
            np.testing.assert_array_equal(
                gguf.dequantize(original.data, original.tensor_type),
                gguf.dequantize(recovered.data, recovered.tensor_type),
            )

    def test_cli_preserves_metadata_and_standard_packed_values(self):
        from rephix.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            source, candidate = Path(tmp) / "reference.gguf", Path(tmp) / "candidate.gguf"
            self._fixture(source)
            original = source.read_bytes()
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = main(["recover-gguf", "--model", str(source), "--output", str(candidate)])
            self.assertEqual(status, 0)
            self.assertIsInstance(json.loads(stdout.getvalue()), dict)
            self.assertEqual(source.read_bytes(), original)
            self._compare(source, candidate)
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = main(["audit-gguf", str(source), str(candidate)])
            self.assertEqual(status, 0)
            audit = json.loads(stdout.getvalue())
            self.assertTrue(audit["metadata_prefix_exact"])
            totals = audit["totals"]
            self.assertEqual(totals["exact_payload_bytes"], totals["payload_bytes"])
            self.assertEqual(totals["exact_dequantized_values"], totals["dequantized_values"])

    def test_external_oracle_reconstructs_weights_unrelated_to_source_payload(self):
        from rephix.gguf_recovery import recover_gguf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, decoy = root / "reference.gguf", root / "decoy.gguf"
            candidate = root / "candidate.gguf"
            observed_weights = self._fixture(reference, seed=24)
            self._fixture(decoy, seed=35)
            calls = []

            def oracle(name, start, stop, inputs):
                self.assertEqual(inputs.dtype, np.float32)
                self.assertEqual(inputs.ndim, 2)
                self.assertLessEqual(len(inputs), 3)
                self.assertTrue(np.all((inputs == 0) | (inputs == 1)))
                np.testing.assert_array_equal(inputs.sum(axis=1), np.ones(len(inputs)))
                weights = observed_weights[name]
                self.assertEqual(inputs.shape[1], weights.shape[-1])
                rows = weights.reshape(-1, weights.shape[-1])
                self.assertLessEqual(stop, len(rows))
                calls.append((name, start, stop))
                return inputs @ rows[start:stop].T

            recover_gguf(decoy, candidate, batch_size=3, call_tensor=oracle)
            self.assertEqual({name for name, *_ in calls}, set(observed_weights))
            self.assertGreater(len(calls), len(observed_weights))
            self.assertNotEqual(decoy.read_bytes(), candidate.read_bytes())
            self._compare(reference, candidate)

    def test_noncanonical_blocks_preserve_dequantization(self):
        from rephix.gguf_audit import audit_gguf
        from rephix.gguf_recovery import recover_gguf

        with tempfile.TemporaryDirectory() as tmp:
            source, candidate = Path(tmp) / "reference.gguf", Path(tmp) / "candidate.gguf"
            self._fixture(source, ambiguous=True)
            report = recover_gguf(source, candidate, batch_size=7)
            self._compare(source, candidate, changed={"token_embd.weight"})
            stats = report["by_encoding"]["Q8_0"]
            self.assertEqual(stats["zero_blocks"], 1)
            self.assertEqual(stats["ambiguous_blocks"], stats["blocks"])
            totals = audit_gguf(source, candidate)["totals"]
            self.assertLess(totals["exact_payload_bytes"], totals["payload_bytes"])
            self.assertEqual(totals["exact_quantized_blocks"], totals["quantized_blocks"] - 2)
            self.assertEqual(totals["exact_dequantized_values"], totals["dequantized_values"])

    def test_cuda_probes_keep_full_precision_and_restore_caller_setting(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("needs the optional PyTorch dependency")
        import torch

        from rephix.gguf_recovery import recover_gguf

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        original_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            with tempfile.TemporaryDirectory() as tmp:
                source, candidate = Path(tmp) / "reference.gguf", Path(tmp) / "candidate.gguf"
                self._fixture(source)
                recover_gguf(source, candidate, batch_size=11, device="cuda")
                self._compare(source, candidate)
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
        finally:
            torch.set_float32_matmul_precision(original_precision)

    def test_failed_oracle_leaves_source_intact_and_no_partial_export(self):
        from rephix.gguf_recovery import recover_gguf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, candidate = root / "reference.gguf", root / "candidate.gguf"
            observed_weights = self._fixture(source)
            original = source.read_bytes()
            calls = []

            def oracle(name, start, stop, inputs):
                calls.append(name)
                if name == "blk.0.attn_q.weight":
                    raise RuntimeError("injected GGUF probe failure")
                weights = observed_weights[name]
                return inputs @ weights.reshape(-1, weights.shape[-1])[start:stop].T

            with self.assertRaisesRegex(RuntimeError, "injected GGUF probe failure"):
                recover_gguf(source, candidate, batch_size=7, call_tensor=oracle)
            self.assertIn("blk.0.attn_q.weight", calls)
            self.assertFalse(candidate.exists())
            self.assertEqual(list(root.iterdir()), [source])
            self.assertEqual(source.read_bytes(), original)

    def test_unsupported_encoding_fails_before_observations(self):
        from rephix.gguf_recovery import recover_gguf

        with tempfile.TemporaryDirectory() as tmp:
            source, candidate = Path(tmp) / "reference.gguf", Path(tmp) / "candidate.gguf"
            self._fixture(source, unsupported=True)

            def unexpected_oracle(*args):
                self.fail("an unsupported archive should fail before any probe")

            with self.assertRaisesRegex(ValueError, "Q4_1|unsupported"):
                recover_gguf(source, candidate, call_tensor=unexpected_oracle)
            self.assertFalse(candidate.exists())


if __name__ == "__main__":
    unittest.main()
