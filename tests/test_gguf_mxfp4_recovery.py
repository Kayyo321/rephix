"""Exercise packed field and numeric agreement for a three-dimensional GGUF bank."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rephix.gguf_audit import audit_gguf
from rephix.gguf_recovery import read_gguf, recover_gguf

HAS_GGUF = importlib.util.find_spec("gguf") is not None


@unittest.skipUnless(HAS_GGUF, "requires optional gguf")
class GGufMxfp4RecoveryTests(unittest.TestCase):
    def test_mixed_checkpoint_keeps_native_axes_and_separates_storage_agreement(self):
        import gguf

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.gguf"
            candidate = Path(directory) / "candidate.gguf"
            writer = gguf.GGUFWriter(source, "gpt-oss")
            # The second block has a larger-than-necessary scale and negative zero.
            unique = np.array([127, *([0x71] * 16)], dtype=np.uint8)
            alias = np.array([128, 0x48, *([0x42] * 15)], dtype=np.uint8)
            original = np.tile(np.concatenate((unique, alias)), (2, 3, 1))
            writer.add_tensor(
                "blk.0.ffn_gate_exps.weight", original, raw_dtype=gguf.GGMLQuantizationType.MXFP4
            )
            writer.add_tensor("blk.0.ffn_gate_exps.bias", np.arange(6, dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            report = recover_gguf(source, candidate, batch_size=9)
            self.assertEqual(report["observed_output_scalars"], 390)
            counts = report["by_encoding"]["MXFP4"]
            self.assertEqual(counts["blocks"], 12)
            self.assertEqual(counts["ambiguous_scale_blocks"], 6)
            self.assertEqual(counts["ambiguous_zero_code_blocks"], 6)

            audit = audit_gguf(source, candidate)
            self.assertTrue(audit["metadata_prefix_exact"])
            self.assertEqual(audit["totals"]["exact_dequantized_values"], 390)
            packed = audit["by_encoding"]["MXFP4"]
            self.assertEqual(packed["exact_quantized_blocks"], 6)
            self.assertEqual(packed["stored_fields"], 12 * 33)
            self.assertEqual(packed["exact_stored_fields"], 6 * 33)

            _, _, layouts = read_gguf(candidate)
            self.assertEqual(layouts[0].shape, (64, 3, 2))
            with candidate.open("r+b") as file:
                offset = layouts[0].offset + 1
                file.seek(offset)
                old = file.read(1)[0]
                file.seek(offset)
                file.write(bytes([old ^ 0x10]))
            changed = audit_gguf(source, candidate)["by_encoding"]["MXFP4"]
            self.assertEqual(changed["exact_stored_fields"], 6 * 33 - 1)
            self.assertEqual(changed["exact_dequantized_values"], 383)


if __name__ == "__main__":
    unittest.main()
