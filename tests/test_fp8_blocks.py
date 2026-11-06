"""Check numeric recovery and indistinguishable E4M3FN storage choices."""

import unittest

import numpy as np

from rephix.fp8_blocks import decode_fp8_blocks, infer_fp8_blocks


class Fp8BlocksTests(unittest.TestCase):
    def test_decode_finite_boundaries_and_reserved_codes(self):
        codes = np.array([0, 1, 7, 8, 56, 126, 128, 129, 254], dtype=np.uint8)
        values = decode_fp8_blocks(codes, np.float32(1))
        np.testing.assert_array_equal(
            values, [0, 2**-9, 7 * 2**-9, 2**-6, 1, 448, 0, -(2**-9), -448]
        )
        self.assertTrue(np.signbit(values[6]))
        for reserved in (127, 255):
            with self.subTest(reserved=reserved), self.assertRaisesRegex(ValueError, "non-finite"):
                decode_fp8_blocks(np.array([reserved], dtype=np.uint8), np.float32(1))

    def test_full_codebook_recovers_arbitrary_scales_at_both_precisions(self):
        fields = np.concatenate((np.arange(127), np.arange(128, 255))).astype(np.uint8)
        codes = np.broadcast_to(fields, (2, 2, len(fields)))
        for scale_dtype, scales in (
            ("float32", np.array([[0.12417, 0.037], [0.987, 0.931]], dtype=np.float32)),
            (
                "bfloat16",
                np.array([[0.1240234375, 0.037109375], [0.9921875, 0.9375]], dtype=np.float32),
            ),
        ):
            for dtype in (np.float32, np.float64):
                with self.subTest(scale_dtype=scale_dtype, dtype=dtype):
                    observed = decode_fp8_blocks(codes, scales, dtype=dtype)
                    recovered, recovered_codes, counts = infer_fp8_blocks(
                        observed, scale_dtype=scale_dtype
                    )
                    np.testing.assert_array_equal(recovered, scales)
                    np.testing.assert_array_equal(recovered_codes, np.where(codes == 128, 0, codes))
                    self.assertEqual(counts["saturated_blocks"], 4)
                    self.assertEqual(counts["ambiguous_scale_blocks"], 0)
                    self.assertEqual(counts["ambiguous_packed_fields"], 8)

    def test_fp32_rounding_can_hide_adjacent_scales(self):
        a, b = np.array([0x3FA00004, 0x3FA00005], dtype=np.uint32).view(np.float32)
        codes = np.array([126, 7], dtype=np.uint8)
        observed = decode_fp8_blocks(codes, a)
        np.testing.assert_array_equal(observed, decode_fp8_blocks(codes, b))
        scale, selected, counts = infer_fp8_blocks(observed)
        np.testing.assert_array_equal(decode_fp8_blocks(selected, scale), observed)
        self.assertEqual(counts["ambiguous_scale_blocks"], 1)

        scale, selected, counts = infer_fp8_blocks(decode_fp8_blocks(codes, a, dtype=np.float64))
        self.assertEqual(scale, a)
        np.testing.assert_array_equal(selected, codes)
        self.assertEqual(counts["ambiguous_packed_fields"], 0)

    def test_alternate_scales_codes_and_zero_blocks(self):
        original = np.array([[126, 254, 0, 128], [0, 128, 128, 0]], dtype=np.uint8)
        observed = decode_fp8_blocks(original, np.array([1, 0.25], np.float32), dtype=np.float64)
        scales, codes, counts = infer_fp8_blocks(observed, scale_dtype="bfloat16")
        np.testing.assert_array_equal(decode_fp8_blocks(codes, scales, dtype=np.float64), observed)
        self.assertEqual(counts["zero_blocks"], 1)
        self.assertEqual(counts["ambiguous_scale_blocks"], 2)
        self.assertEqual(counts["ambiguous_code_scale_blocks"], 1)
        self.assertEqual(counts["ambiguous_zero_codes"], 6)
        self.assertEqual(counts["ambiguous_packed_fields"], 10)

    def test_fp64_preserves_subnormal_scales_and_rejects_outside_lattice(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        codes = np.arange(127, dtype=np.uint8)
        observed = decode_fp8_blocks(codes, tiny, dtype=np.float64)
        recovered, result, _ = infer_fp8_blocks(observed)
        self.assertEqual(recovered, tiny)
        np.testing.assert_array_equal(result, codes)
        with self.assertRaisesRegex(ValueError, "normal nonzero"):
            infer_fp8_blocks(observed.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "no exact"):
            infer_fp8_blocks(np.array([1, np.sqrt(2)], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
