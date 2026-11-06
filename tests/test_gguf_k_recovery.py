"""Exercise mixed GGUF files and the K-quant storage conventions."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rephix.gguf_audit import audit_gguf
from rephix.gguf_quant import recover_blocks
from rephix.gguf_recovery import read_gguf, recover_gguf

HAS_GGUF = importlib.util.find_spec("gguf") is not None


def _q4k():
    scales = np.array([63, 29, 31, 24, 47, 17, 51, 35], dtype=np.uint8)
    minima = np.array([31, 48, 35, 63, 59, 25, 33, 19], dtype=np.uint8)
    low = scales[:4] | (scales[4:] >> 4) << 6
    high = minima[:4] | (minima[4:] >> 4) << 6
    tail = (scales[4:] & 15) | ((minima[4:] & 15) << 4)
    codes = np.random.default_rng(84).integers(0, 16, (8, 32), dtype=np.uint8)
    codes[:, :2] = [0, 15]
    packed = (codes[::2] | (codes[1::2] << 4)).ravel()
    return np.concatenate((np.array([0.001, 0.012], "<f2").view(np.uint8), low, high, tail, packed))


def _q6k(*, anchor=True):
    scales = np.array(
        [-128 if anchor else -125, 29, 31, -24, 47, -17, 51, 35, 89, -37, 123, 61, 42, -9, 3, -77],
        dtype=np.int8,
    )
    rng = np.random.default_rng(190)
    codes = rng.integers(-32 if anchor else -31, 32, (16, 16), dtype=np.int8)
    codes[:, :2] = [-32 if anchor else -31, 31]
    codes = (codes.astype(np.int16) + 32).astype(np.uint8).reshape(2, 4, 32)
    low = (codes[:, :2] & 15) | ((codes[:, 2:] & 15) << 4)
    high = (codes >> 4) << np.array([0, 2, 4, 6], dtype=np.uint8)[None, :, None]
    return np.concatenate(
        (
            low.ravel(),
            np.bitwise_or.reduce(high, axis=1).ravel(),
            scales.view(np.uint8),
            np.array([0.00013], "<f2").view(np.uint8),
        )
    )


@unittest.skipUnless(HAS_GGUF, "requires optional gguf")
class KQuantRecoveryTests(unittest.TestCase):
    def test_standard_q5_0_and_k_quant_blocks_round_trip(self):
        import gguf

        kinds = gguf.GGMLQuantizationType
        q5 = gguf.quantize(np.linspace(-0.4, 0.9, 32, dtype=np.float32)[None], kinds.Q5_0)[0]
        for encoding, original in ((kinds.Q5_0, q5), (kinds.Q4_K, _q4k()), (kinds.Q6_K, _q6k())):
            with self.subTest(encoding=encoding.name):
                observed = gguf.dequantize(original[None], encoding)
                recovered, counts = recover_blocks(observed, encoding)
                np.testing.assert_array_equal(gguf.dequantize(recovered, encoding), observed)
                np.testing.assert_array_equal(recovered[0], original)
                self.assertEqual(counts["blocks"], 1)

    def test_q6k_alternate_signs_preserve_effective_values(self):
        import gguf

        encoding = gguf.GGMLQuantizationType.Q6_K
        original = _q6k(anchor=False)
        observed = gguf.dequantize(original[None], encoding)
        recovered, counts = recover_blocks(observed, encoding)
        np.testing.assert_array_equal(gguf.dequantize(recovered, encoding), observed)
        self.assertEqual(counts["ambiguous_blocks"], 1)

    def test_q4k_signed_shared_scales_round_trip(self):
        import gguf

        encoding = gguf.GGMLQuantizationType.Q4_K
        block = _q4k()
        block[[1, 3]] ^= 128
        observed = gguf.dequantize(block[None], encoding)
        recovered, _ = recover_blocks(observed, encoding)
        np.testing.assert_array_equal(gguf.dequantize(recovered, encoding), observed)

    def test_mixed_file_recovery_and_independent_stored_field_audit(self):
        import gguf

        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.gguf", Path(directory) / "candidate.gguf"
            writer = gguf.GGUFWriter(source, "fixture")
            kinds = gguf.GGMLQuantizationType
            for name, encoding, block in (
                ("a.weight", kinds.Q4_K, _q4k()),
                ("b.weight", kinds.Q6_K, _q6k()),
            ):
                writer.add_tensor(name, np.tile(block, (2, 2)), raw_dtype=encoding)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            report = recover_gguf(source, target, batch_size=128)
            self.assertEqual(report["observed_output_scalars"], 2 * 4 * 256)
            audit = audit_gguf(source, target)
            self.assertTrue(audit["metadata_prefix_exact"])
            for name, fields in (("Q4_K", 274), ("Q6_K", 273)):
                result = audit["by_encoding"][name]
                self.assertEqual(result["stored_fields"], 4 * fields)
                self.assertEqual(result["exact_stored_fields"], 4 * fields)
                self.assertEqual(result["exact_dequantized_values"], 4 * 256)

            _, _, layouts = read_gguf(target)
            offset = layouts[1].offset
            with target.open("r+b") as file:
                # Change both packed pieces of the same Q6 code.
                for index in (offset, offset + 128):
                    file.seek(index)
                    value = file.read(1)[0] ^ 1
                    file.seek(index)
                    file.write(bytes([value]))
            changed = audit_gguf(source, target)["by_encoding"]["Q6_K"]
            self.assertEqual(changed["exact_stored_fields"], 4 * 273 - 1)
            self.assertEqual(changed["exact_payload_bytes"], 4 * 210 - 2)
            self.assertEqual(changed["exact_dequantized_values"], 4 * 256 - 1)


if __name__ == "__main__":
    unittest.main()
