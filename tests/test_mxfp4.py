"""Check exact MXFP4 inversion and which fields remain ambiguous."""

import importlib.util
import unittest

import numpy as np

from rephix.mxfp4 import infer_mxfp4_fields, recover_mxfp4

HAS_GGUF = importlib.util.find_spec("gguf") is not None


class Mxfp4Tests(unittest.TestCase):
    def test_representative_and_ambiguity(self):
        values = np.tile([0, -0.5, 1.5, 6], (2, 8)).astype(np.float32)
        exponents, codes, counts = infer_mxfp4_fields(values, max_exponent=254)
        np.testing.assert_array_equal(exponents, [127, 127])
        np.testing.assert_array_equal(codes, np.tile([0, 9, 3, 7], (2, 8)))
        self.assertEqual(counts["ambiguous_scale_blocks"], 0)
        self.assertEqual(counts["ambiguous_zero_code_blocks"], 2)
        self.assertEqual(counts["ambiguous_packed_fields"], 16)

        alternate = np.stack((np.tile([3.0, 6.0], 16), np.zeros(32))).astype(np.float32)
        exponents, codes, counts = infer_mxfp4_fields(alternate)
        np.testing.assert_array_equal(exponents, [127, 0])
        np.testing.assert_array_equal(codes[0], np.tile([5, 7], 16))
        self.assertEqual(counts["zero_blocks"], 1)
        self.assertEqual(counts["ambiguous_scale_blocks"], 2)
        self.assertEqual(counts["ambiguous_packed_fields"], 66)

    def test_rejects_inexact_nonfinite_and_off_lattice_inputs(self):
        for value in (5.0, np.inf, np.nan, 2.0**-149, 1.0 + 2.0**-40):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    recover_mxfp4(np.full(32, value, dtype=np.float64))

    @unittest.skipUnless(HAS_GGUF, "needs optional GGUF reference decoder")
    def test_matches_reference_decoder_at_every_finite_constant_block(self):
        import gguf

        encoding = gguf.GGMLQuantizationType.MXFP4
        raw = np.empty((256 * 16, 17), dtype=np.uint8)
        raw[:, 0] = np.repeat(np.arange(256, dtype=np.uint8), 16)
        codes = np.tile(np.arange(16, dtype=np.uint8), 256)
        raw[:, 1:] = (codes | (codes << np.uint8(4)))[:, None]
        with np.errstate(over="ignore"):
            values = gguf.dequantize(raw, encoding)
        finite = values[np.isfinite(values).all(axis=1)]
        recovered, counts = recover_mxfp4(finite)
        np.testing.assert_array_equal(gguf.dequantize(recovered, encoding), finite)
        self.assertEqual(counts["blocks"], 4072)

    @unittest.skipUnless(HAS_GGUF, "needs optional GGUF reference encoder")
    def test_matches_reference_packed_fields_for_ordinary_rounded_values(self):
        import gguf

        encoding = gguf.GGMLQuantizationType.MXFP4
        rng = np.random.default_rng(155)
        values = rng.normal(size=(2048, 32)).astype(np.float32)
        packed = gguf.quantize(values, encoding)
        recovered, counts = recover_mxfp4(gguf.dequantize(packed, encoding))
        np.testing.assert_array_equal(recovered, packed)
        self.assertEqual(counts["reference_convention_blocks"], len(values))
